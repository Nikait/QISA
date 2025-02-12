import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cmath
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical
import random



def rx_matrix(theta: float) -> torch.Tensor:
    """
    Computes the RX rotation matrix (2x2) for a given angle.

    Args:
        theta (float): The rotation angle in radians.

    Returns:
        torch.Tensor: A 2x2 complex tensor representing the RX gate.
    """
    return torch.tensor([[math.cos(theta/2), -1j*math.sin(theta/2)],
                         [-1j*math.sin(theta/2), math.cos(theta/2)]],
                        dtype=torch.cdouble)

def ry_matrix(theta: float) -> torch.Tensor:
    """
    Computes the RY rotation matrix (2x2) for a given angle.

    Args:
        theta (float): The rotation angle in radians.

    Returns:
        torch.Tensor: A 2x2 complex tensor representing the RY gate.
    """
    return torch.tensor([[math.cos(theta/2), -math.sin(theta/2)],
                         [math.sin(theta/2),  math.cos(theta/2)]],
                        dtype=torch.cdouble)

def cnot_matrix() -> torch.Tensor:
    """
    Returns the 4x4 matrix representation of the CNOT gate.

    Returns:
        torch.Tensor: A 4x4 complex tensor representing the CNOT gate.
    """
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=torch.cdouble)


def expand_operator(gate: torch.Tensor, target_qubits: list, n: int) -> torch.Tensor:
    """
    Expands a gate (of size 2^(len(target_qubits)) x 2^(len(target_qubits))) 
    to act on an n-qubit system, resulting in a 2^n x 2^n matrix.
    
    The gate acts nontrivially only on the qubits specified in target_qubits.
    (Assumes little-endian ordering: qubit 0 is the least-significant bit.)

    Args:
        gate (torch.Tensor): A matrix representing a quantum gate on len(target_qubits) qubits.
        target_qubits (list): A list of qubit indices where the gate acts.
        n (int): Total number of qubits in the system.

    Returns:
        torch.Tensor: The expanded gate as a 2^n x 2^n complex tensor.
    """
    dim = 1 << n  # 2^n
    U_full = torch.zeros((dim, dim), dtype=torch.cdouble)
    # Iterate over all basis states of the full Hilbert space.
    for i in range(dim):
        # Represent i as a list of bits.
        bits = [(i >> b) & 1 for b in range(n)]
        # Build the sub-index for the target qubits.
        sub_index = 0
        for idx, qubit in enumerate(target_qubits):
            sub_index |= bits[qubit] << idx
        # For each possible outcome of the gate acting on the target qubits.
        for j in range(gate.shape[0]):
            new_bits = bits.copy()
            # Replace bits at the target qubit positions.
            for idx, qubit in enumerate(target_qubits):
                new_bits[qubit] = (j >> idx) & 1
            new_i = 0
            for b in range(n):
                new_i |= new_bits[b] << b
            U_full[new_i, i] += gate[j, sub_index]
    return U_full






# ----------------- Quantum Self-Attention Layer -----------------

class QSA(nn.Module):
    def __init__(self, n_embed: int, n_context: int, head_size: int):
        """
        Quantum Self-Attention layer that uses two implementations:
          - Slow branch: for training (step-by-step simulation)
          - Fast branch: for inference (precomputation of the ansatz into a single matrix)

        Args:
            n_embed (int): Dimension of the input embedding.
            n_context (int): Maximum context length (used for masking).
            head_size (int): Number of quantum measurable values (typically corresponds to the number of heads).

        Returns:
            None
        """
        super().__init__()
        self.__precomputed = False
        self.head_size = head_size
        self.n_context = n_context
        self.n_wires = int(np.ceil(np.log2(n_embed)))
        self.register_buffer('tril', torch.tril(torch.ones(n_context, n_context)))
        self.ops = [self.choose_op() for _ in range(self.head_size)]
        
        # Fast branch for inference: uses precomputed composite matrices.
        # Note: The unitary matrices U are computed only when switching to eval mode.
        self.q_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size)
        self.k_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size)
        self.v_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size)
        
        # Slow branch for training: uses step-by-step simulation.
        self.q_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size)
        self.k_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size)
        self.v_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size)
        
    def choose_op(self):
        """
        Randomly generates a Pauli operator string of length equal to the number of qubits,
        ensuring that the generated operator is nontrivial (i.e., not all 'I').

        Returns:
            str: A nontrivial Pauli operator string.
        """
        import random
        op_s = 'IXYZ'
        while True:
            op = ''.join(random.choice(op_s) for _ in range(self.n_wires))
            if op != 'I' * self.n_wires:
                return op

    def _forward_slow(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using the slow branch (step-by-step simulation).
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size).
        """
        B, T, C = x.shape
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B*T, device=x.device)
        q = self.q_slow(x, qdev)  # (B, T, head_size)
        k = self.k_slow(x, qdev)  # (B, T, head_size)
        v = self.v_slow(x, qdev)  # (B, T, head_size)
        q = q.unsqueeze(2)  # Expand dimension for broadcasting: (B, T, 1, head_size)
        k = k.unsqueeze(1)  # (B, 1, T, head_size)
        alpha = torch.exp(-((q - k) ** 2).sum(dim=-1))  # (B, T, T)
        alpha = alpha.masked_fill(self.tril == 0, float('-inf'))
        normalized_alpha = F.softmax(alpha, dim=-1)
        out = normalized_alpha.permute(0, 2, 1) @ v  # (B, T, head_size)
        return out

    def _forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass using the fast branch (with precomputed composite unitary).
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size).
        """
        q = self.q_fast(x)  # (B, T, head_size)
        k = self.k_fast(x)  # (B, T, head_size)
        v = self.v_fast(x)  # (B, T, head_size)
        q = q.unsqueeze(2)
        k = k.unsqueeze(1)
        alpha = torch.exp(-((q - k) ** 2).sum(dim=-1))
        alpha = alpha.masked_fill(self.tril == 0, float('-inf'))
        normalized_alpha = F.softmax(alpha, dim=-1)
        out = normalized_alpha.permute(0, 2, 1) @ v
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Chooses the branch to use based on the module's mode:
         - Training mode: uses the slow branch (step-by-step simulation).
         - Evaluation mode: uses the fast branch with precomputed composite matrices.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size).
        """
        if self.training:
            return self._forward_slow(x)
        else:
            if not self.__precomputed:
                self.precompute()
                self.__precomputed = True
            return self._forward_fast(x)

    def precompute(self):
        """
        Precomputes the composite matrices for the fast branch when switching to inference mode.
        This replaces a chain of quantum operations with a single matrix multiplication on the state vector.
        Also logs the precomputation process.

        Returns:
            None
        """
        super().eval()
        print("[INFO] Switching to inference mode. Starting ansatz precomputation...")
        
        # Copy parameters from the slow branch into the fast branch.
        self.q_fast.rx0_params = self.q_slow.rx0
        self.q_fast.ry0_params = self.q_slow.ry0
        self.q_fast.ry1_params = self.q_slow.ry1
        
        self.k_fast.rx0_params = self.k_slow.rx0
        self.k_fast.ry0_params = self.k_slow.ry0
        self.k_fast.ry1_params = self.k_slow.ry1
        
        self.v_fast.rx0_params = self.v_slow.rx0
        self.v_fast.ry0_params = self.v_slow.ry0
        self.v_fast.ry1_params = self.v_slow.ry1
        
        # Build the composite unitaries for each branch.
        self.q_fast._build_unitary()
        self.k_fast._build_unitary()
        self.v_fast._build_unitary()
        
        print("[INFO] Ansätze precomputation completed. Fast branch matrices updated.")


# ----------------- Inference Branch: ValueLayerFast -----------------

class ValueLayerFast(nn.Module):
    def __init__(self, n_wires: int, ops: list[str], hidden_dim: int):
        """
        Fast value layer for inference.
        Precomputes a composite ansatz unitary representing the quantum circuit in one shot.

        Args:
            n_wires (int): Number of qubits.
            ops (list[str]): List of Pauli operator strings to be measured.
            hidden_dim (int): Number of output features (heads).

        Returns:
            None
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.ops = ops
        self.encoding = tq.AmplitudeEncoder()
        
        # Trainable gate modules (lists of quantum modules).
        self.rx0_params = tq.QuantumModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry0_params = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry1_params = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        
        dim = 1 << self.n_wires
        self.register_buffer('U', torch.eye(dim, dtype=torch.cdouble))

    def _build_unitary(self):
        """
        Precomputes the overall composite unitary (ansatz) for the inference branch.
        The composite unitary is built by starting with the identity matrix and then
        sequentially multiplying (from the left) each gate (expanded to the full Hilbert space)
        in the following order:
          1. For each qubit j, apply the local rotation given by (RY0 followed by RX0).
          2. For a number of layers equal to the number of qubits:
             a. For each qubit j, apply an expanded two-qubit CNOT gate from qubit j to qubit (j+1)%n.
             b. For each qubit j, apply an expanded local RY gate (using ry1).
        
        Returns:
            None. The computed composite unitary is stored in the buffer 'U'.
        """
        print(f"[DEBUG] Precomputing composite ansatz matrix for {self.n_wires} qubits...")
        n = self.n_wires
        dim = 1 << n  # 2^n
        U_total = torch.eye(dim, dtype=torch.cdouble)
        
        # 1. Apply local rotations on each qubit: (RY0 then RX0)
        for j in range(n):
            angle_rx = self.rx0_params[j].params.item()
            angle_ry = self.ry0_params[j].params.item()
            local_gate = torch.matmul(ry_matrix(angle_ry), rx_matrix(angle_rx))
            expanded_gate = expand_operator(local_gate, [j], n)
            U_total = expanded_gate @ U_total
        
        # 2. Apply entangling layers (repeat n times):
        for _ in range(n):
            for j in range(n):
                # Expand the CNOT gate to act on qubits j and (j+1)%n.
                expanded_cnot = expand_operator(cnot_matrix(), [j, (j+1) % n], n)
                U_total = expanded_cnot @ U_total
            for j in range(n):
                angle = self.ry1_params[j].params.item()
                local_ry1 = ry_matrix(angle)
                expanded_ry1 = expand_operator(local_ry1, [j], n)
                U_total = expanded_ry1 @ U_total
        
        print(f"[DEBUG] Composite unitary shape: {U_total.shape}, expected ({dim}, {dim})")
        assert U_total.shape == (dim, dim), "Error: Composite unitary U has incorrect dimensions!"
        self.register_buffer('U', U_total)

    def _expand_two_qubit_gate(self, two_qubit_gate: torch.Tensor, control: int, target: int) -> torch.Tensor:
        """
        Expands a two-qubit gate (4x4) into the full Hilbert space for the n-qubit system.
        Uses the helper function 'expand_operator' to embed the gate on the specified qubits.

        Args:
            two_qubit_gate (torch.Tensor): A 4x4 complex tensor representing the two-qubit gate.
            control (int): The index of the control qubit.
            target (int): The index of the target qubit.

        Returns:
            torch.Tensor: The expanded 2^n x 2^n complex matrix.
        """
        return expand_operator(two_qubit_gate, [control, target], self.n_wires)

    def copy_params_from(self, slow_layer):
        """
        Copies parameters from the slow (training) layer to the fast (inference) layer.
        This ensures that the fast branch uses the optimized parameters after training.

        Args:
            slow_layer (ValueLayerSlow): The slow branch instance from which to copy parameters.

        Returns:
            None
        """
        self.rx0_params.data = slow_layer.rx0.data.clone()
        self.ry0_params.data = slow_layer.ry0.data.clone()
        self.ry1_params.data = slow_layer.ry1.data.clone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass for the inference branch.
        It encodes the input, applies the precomputed composite unitary,
        and measures the expectation values of the specified operators.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, hidden_dim) as a float tensor.
        """
        if self.U is None:
            self._build_unitary()

        B, T, C = x.shape
        x_flat = x.view(B * T, C)

        # Create the quantum device.
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B*T, device=x.device)
        dim = 1 << self.n_wires
        base_states = torch.zeros((B * T, dim), device=x.device, dtype=torch.cfloat)  # float-complex type
        base_states[:, 0] = 1.0
        qdev.set_states(base_states)

        # Encode the input into the quantum state.
        self.encoding(qdev, x_flat)
        
        # Reshape the state vector.
        psi = qdev.states.reshape(B * T, -1)  # (B*T, 2^n_wires)

        # Dimension check.
        assert psi.shape[1] == self.U.shape[0], f"Dimension mismatch: psi {psi.shape}, U {self.U.shape}"

        # Ensure U is on the correct device and type.
        U = self.U.to(x.device).to(torch.cfloat)
        psi_out = torch.matmul(psi, U.T)

        # Update the quantum device state.
        qdev.set_states(psi_out)

        # Measure expectation values of the specified operators.
        measurements = [expval_joint_analytical(qdev, op) for op in self.ops]
        out = torch.stack(measurements, dim=-1)  # (B*T, hidden_dim)
        return out.view(B, T, self.hidden_dim).float()

# ----------------- Training Branch: ValueLayerSlow -----------------

class ValueLayerSlow(nn.Module):
    def __init__(self, n_wires: int, ops: list[str], hidden_dim: int):
        """
        Slow value layer for training.
        Simulates the quantum circuit step-by-step using TorchQuantum.

        Args:
            n_wires (int): Number of qubits.
            ops (list[str]): List of Pauli operator strings to measure.
            hidden_dim (int): Number of output features (heads).

        Returns:
            None
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.ops = ops
        self.encoding = tq.AmplitudeEncoder()
        self.rx0 = tq.QuantumModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry0 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry1 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        
    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice) -> torch.Tensor:
        """
        Performs a forward pass for the training branch.
        It encodes the input data and applies the quantum gates sequentially.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).
            qdev (tq.QuantumDevice): Quantum device used for simulation.

        Returns:
            torch.Tensor: Output tensor of shape (B, T, hidden_dim) as a float tensor.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, C = x.shape
        base_states = torch.zeros((B, 1 << self.n_wires), device=x.device)
        base_states[:, 0] = 1.0
        qdev.set_states(base_states)
        x_flat = x.view(B*T, C)
        self.encoding(qdev, x_flat)
        # Apply local rotations for each qubit (RY0 then RX0).
        for j, (rx, ry) in enumerate(zip(self.rx0, self.ry0)):
            rx(qdev, wires=j)
            ry(qdev, wires=j)
        # Apply entangling layers sequentially.
        for _ in range(self.n_wires):
            for j in range(self.n_wires):
                tqf.cnot(qdev, wires=[j, (j+1)%self.n_wires])
            for j, ry in enumerate(self.ry1):
                ry(qdev, wires=j)
        measurements = [expval_joint_analytical(qdev, op) for op in self.ops]
        out = torch.stack(measurements, dim=-1)  # (B*T, hidden_dim)
        return out.view(B, T, self.hidden_dim).float()

    def __call__(self, x: torch.Tensor, qdev: tq.QuantumDevice = None) -> torch.Tensor:
        """
        Calls the forward pass. If qdev is not provided, a new quantum device is created.

        Args:
            x (torch.Tensor): Input tensor.
            qdev (tq.QuantumDevice, optional): Quantum device; if None, one is created.

        Returns:
            torch.Tensor: Output tensor from the forward pass.
        """
        B, T, _ = x.shape
        if qdev is None:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B*T, device=x.device)
        return self.forward(x, qdev)
