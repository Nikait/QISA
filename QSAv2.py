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
    Returns the RX rotation matrix for a given angle.

    Args:
        theta (float): The rotation angle in radians.

    Returns:
        torch.Tensor: A 2x2 complex tensor representing the RX gate.
    """
    return torch.tensor([[math.cos(theta/2), -1j*math.sin(theta/2)],
                         [-1j*math.sin(theta/2), math.cos(theta/2)]],
                        dtype=torch.complex64)

def ry_matrix(theta: float) -> torch.Tensor:
    """
    Returns the RY rotation matrix for a given angle.

    Args:
        theta (float): The rotation angle in radians.

    Returns:
        torch.Tensor: A 2x2 complex tensor representing the RY gate.
    """
    theta = nn.Parameter(theta, requires_grad=True)
    theta = theta.type(torch.complex64)
    co = torch.cos(theta / 2)
    si = torch.sin(theta / 2)
    mat = torch.stack(
        [torch.cat([co, -si], dim=-1), torch.cat([si, co], dim=-1)], dim=-2
    ).squeeze(0)
    return mat

def cnot_matrix() -> torch.Tensor:
    """
    Returns the 4x4 matrix representation of the CNOT gate.

    Returns:
        torch.Tensor: A 4x4 complex tensor representing the CNOT gate.
    """
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=torch.complex64)

def expand_operator(gate: torch.Tensor, target_qubits: list, n: int) -> torch.Tensor:
    """
    Expands a given gate operator to a full operator acting on n qubits.
    
    The gate is assumed to be a 2^m x 2^m matrix acting on m target qubits.
    
    Args:
        gate (torch.Tensor): The operator matrix of size (2**m, 2**m).
        target_qubits (list): List of qubit indices on which the gate acts.
        n (int): Total number of qubits.
    
    Returns:
        torch.Tensor: The full operator matrix of size (2**n, 2**n).
    """
    dim = 1 << n  # 2^n
    U_full = torch.zeros((dim, dim), dtype=gate.dtype, device=gate.device)
    m = len(target_qubits)
    for i in range(dim):
        # Represent the integer i as a binary vector of length n (MSB first)
        bits = [(i >> (n - 1 - b)) & 1 for b in range(n)]
        sub_index = 0
        for qubit in target_qubits:
            sub_index = (sub_index << 1) | bits[qubit]
        for j in range(2**m):
            # Represent j as a binary vector of length m (MSB first)
            replacement_bits = [(j >> (m - 1 - b)) & 1 for b in range(m)]
            new_bits = bits.copy()
            for idx, qubit in enumerate(target_qubits):
                new_bits[qubit] = replacement_bits[idx]
            new_i = 0
            for b in range(n):
                new_i = (new_i << 1) | new_bits[b]
            U_full[new_i, i] = gate[j, sub_index]
    return U_full

# ----------------- Quantum Self-Attention Layer -----------------

class QSA(nn.Module):
    """
    Quantum Self-Attention layer.

    The constructor now takes two additional parameters:
      - layer_id: the unique transformer layer number
      - head_id: the head index within that layer

    These parameters are used to generate unique names for the unitary (U) matrices.

    Args:
        n_embed (int): Embedding dimension.
        n_context (int): Context length (number of tokens).
        head_size (int): Number of measurement outcomes per head.
        layer_id (int): Unique transformer layer number.
        head_id (int): Head index within the transformer layer.
    
    Returns:
        QSA: An instance of the quantum self-attention layer.
    """
    def __init__(self, n_embed: int, n_context: int, head_size: int, layer_id: int, head_id: int):
        super().__init__()
        self.__precomputed = False
        self.head_size = head_size
        self.n_context = n_context
        self.transformer_layer_id = layer_id  # transformer layer number
        self.head_id = head_id                # head index
        self.register_buffer('tril', torch.tril(torch.ones(n_context, n_context)))
        self.n_wires = int(np.ceil(np.log2(n_embed)))
        self.ops = [self.choose_op() for _ in range(self.head_size)]
        print(f"Generated Pauli strings for layer {layer_id} head {head_id}: ", self.ops)
        
        # Fast branch for inference - pass both identifiers
        self.q_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id, debug=True)
        self.k_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id, debug=True)
        self.v_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id, debug=True)
        
        # Slow branch for training
        self.q_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size, debug=True)
        self.k_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size, debug=True)
        self.v_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size, debug=True)
        self.unitary_count = 0

    def choose_op(self) -> str:
        """
        Randomly chooses a Pauli string operator for measurement.

        Returns:
            str: A Pauli string (e.g., 'IXZ') where at least one character is non-identity.
        """
        prob_identity = 0.9
        non_identity_ops = ['X', 'Y', 'Z']
        while True:
            op_chars = []
            for _ in range(self.n_wires):
                if random.random() < prob_identity:
                    op_chars.append('I')
                else:
                    op_chars.append(random.choice(non_identity_ops))
            op = ''.join(op_chars)
            if op != 'I' * self.n_wires:
                return op

    def _forward_slow(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slow forward pass for training that simulates the full quantum circuit step-by-step.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size) after applying quantum self-attention.
        """
        B, T, C = x.shape
        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B * T, device=x.device)
    
        q = self.q_slow(x, qdev)
        k = self.k_slow(x, qdev)
        v = self.v_slow(x, qdev)
        q = q.unsqueeze(2)
        k = k.unsqueeze(1)
        alpha = torch.exp(-((q - k) ** 2).sum(dim=-1))
        alpha = alpha.masked_fill(self.tril == 0, float('-inf'))
        normalized_alpha = F.softmax(alpha, dim=-1)
        out = normalized_alpha.permute(0, 2, 1) @ v
        return out

    def _forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast forward pass for inference using precomputed unitary matrices.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, head_size) after applying quantum self-attention.
        """
        q = self.q_fast(x)
        k = self.k_fast(x)
        v = self.v_fast(x)
        q = q.unsqueeze(2)
        k = k.unsqueeze(1)
        alpha = torch.exp(-((q - k) ** 2).sum(dim=-1))
        alpha = alpha.masked_fill(self.tril == 0, float('-inf'))
        normalized_alpha = F.softmax(alpha, dim=-1)
        out = normalized_alpha.permute(0, 2, 1) @ v
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the QSA layer. Uses the slow branch during training and the fast branch during inference.

        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor after applying quantum self-attention.
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
        Precomputes and registers the unitary operators for the fast branch by copying parameters
        from the slow branch. The unitary matrices are given unique names based on the transformer
        layer and head identifiers.

        Returns:
            None
        """
        super().eval()
        print("[INFO] Switching to inference mode. Starting precomputation of unitaries...")
        # Copy parameters from the slow branch to the fast branch
        self.q_fast.rx0_params = self.q_slow.rx0
        self.q_fast.ry0_params = self.q_slow.ry0
        self.q_fast.ry1_params = self.q_slow.ry1

        self.k_fast.rx0_params = self.k_slow.rx0
        self.k_fast.ry0_params = self.k_slow.ry0
        self.k_fast.ry1_params = self.k_slow.ry1

        self.v_fast.rx0_params = self.v_slow.rx0
        self.v_fast.ry0_params = self.v_slow.ry0
        self.v_fast.ry1_params = self.v_slow.ry1

        # Build unitary operators with unique names that include the transformer layer and head numbers
        self.q_fast._build_unitary(self.unitary_count, layer_id="q_fast", 
                                   transformer_layer_id=self.transformer_layer_id, head_id=self.head_id)
        self.unitary_count += 1
        self.k_fast._build_unitary(self.unitary_count, layer_id="k_fast", 
                                   transformer_layer_id=self.transformer_layer_id, head_id=self.head_id)
        self.unitary_count += 1
        self.v_fast._build_unitary(self.unitary_count, layer_id="v_fast", 
                                   transformer_layer_id=self.transformer_layer_id, head_id=self.head_id)
        self.unitary_count += 1

        print("[INFO] Precomputation completed. Fast branch matrices updated.")

# ----------------- Fast Branch: ValueLayerFast -----------------

class ValueLayerFast(nn.Module):
    """
    Fast value layer for inference.

    This layer encodes the input state, applies a precomputed unitary evolution, and performs measurements.
    The constructor accepts transformer_layer_id and head_id to generate unique names for the unitary matrices.

    Args:
        n_wires (int): Number of qubits.
        ops (list[str]): List of Pauli string operators for measurements.
        hidden_dim (int): Number of measurement outcomes.
        transformer_layer_id (int): Unique transformer layer number.
        head_id (int): Head index within the transformer layer.
        debug (bool): If True, enable debug logging.
    
    Returns:
        ValueLayerFast: An instance of the fast value layer.
    """
    def __init__(self, n_wires: int, ops: list[str], hidden_dim: int, transformer_layer_id: int, head_id: int, debug: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.ops = ops
        self.encoding = tq.AmplitudeEncoder()
        self.debug = debug
        self.transformer_layer_id = transformer_layer_id
        self.head_id = head_id

        self.rx0_params = tq.QuantumModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry0_params = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry1_params = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])

        dim = 1 << self.n_wires
        self.register_buffer('U', torch.eye(dim, dtype=torch.complex64))
        self.unitary_counter = 0
        self.current_unitary_id = None

    def _build_unitary(self, unitary_count: int, layer_id: str, transformer_layer_id: int, head_id: int):
        """
        Builds the full unitary operator for the quantum circuit by composing gate operations,
        and registers it with a unique name that includes the transformer layer and head identifiers.

        Args:
            unitary_count (int): A counter for unique unitary matrices.
            layer_id (str): A string identifier for the current branch (e.g., "q_fast").
            transformer_layer_id (int): The unique transformer layer number.
            head_id (int): The head index within the transformer layer.

        Returns:
            None
        """
        n = self.n_wires
        dim = 1 << n
        device = self.rx0_params[0].params.device  
        U_total = torch.eye(dim, dtype=torch.complex64, device=device)

        for j in range(n):
            rx_val = self.rx0_params[j].params.clone().item()
            ry_val = self.ry0_params[j].params.clone()
            
            rx_param_tensor = torch.tensor(rx_val, device=device)
            expanded_rx = expand_operator(rx_matrix(rx_param_tensor), [j], n).to(device=device, dtype=torch.complex64)
            U_total = expanded_rx @ U_total

            ry_param_tensor = torch.tensor(ry_val, device=device)
            expanded_ry = expand_operator(ry_matrix(ry_param_tensor), [j], n).to(device=device, dtype=torch.complex64)
            U_total = expanded_ry @ U_total

        for j in range(n):
            expanded_cnot = expand_operator(cnot_matrix(), [j, (j + 1) % n], n).to(device=device, dtype=torch.complex64)
            U_total = expanded_cnot @ U_total

        for j in range(n):
            ry1_val = self.ry1_params[j].params.clone()
            mat = ry_matrix(ry1_val).to('cpu')
            expanded_ry1 = expand_operator(mat, [j], n).to(device=device, dtype=torch.complex64)
            U_total = expanded_ry1 @ U_total

        unique_name = f"layer{transformer_layer_id}_head{head_id}_{layer_id}_U{unitary_count}"
        if hasattr(self, unique_name):
            getattr(self, unique_name).data.copy_(U_total)
        else:
            self.register_buffer(unique_name, U_total)
        self.current_unitary_id = unique_name
        # Uncomment the following line for additional logging if needed:
        # print(f"[INFO] Built operator {unique_name} for {layer_id}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the fast value layer.
        
        This method encodes the input, applies the precomputed unitary evolution, and performs measurements.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, hidden_dim) containing measurement results.
        """
        B, T, C = x.shape
        x_flat = x.view(B * T, C)
        device = x.device

        qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B * T, device=device)
        dim = 1 << self.n_wires
        base_states = torch.zeros((B * T, dim), device=device, dtype=torch.complex64)
        base_states[:, 0] = 1.0
        qdev.set_states(base_states)
        self.encoding(qdev, x_flat)
        psi = qdev.states.reshape(B * T, -1).to(device=device, dtype=torch.complex64)
        psi_reshaped = psi.view(B * T, *([2] * self.n_wires))

        if self.current_unitary_id is None:
            raise ValueError("Unitary operator has not been built. Please call _build_unitary first.")
        U = getattr(self, self.current_unitary_id).to(device=device, dtype=torch.complex64)
        # Uncomment the following line for additional logging if needed:
        # print(f"[INFO] Using operator {self.current_unitary_id} in forward pass.")

        psi_out = tqf.apply_unitary_bmm(psi_reshaped, U, list(range(self.n_wires)))
        psi_out = psi_out.view(B * T, -1)
        qdev.set_states(psi_out)
        qdev.states = qdev.states.to(device=device, dtype=torch.complex64)

        measurements = [expval_joint_analytical(qdev, op) for op in self.ops]
        out = torch.stack(measurements, dim=-1)
        return out.view(B, T, self.hidden_dim).float()

# ------------------ Slow Branch: ValueLayerSlow ------------------

class ValueLayerSlow(nn.Module):
    """
    Slow value layer for training.

    This layer encodes the input state and applies quantum operations step-by-step,
    providing a detailed simulation of the quantum circuit.

    Args:
        n_wires (int): Number of qubits.
        ops (list[str]): List of Pauli string operators for measurements.
        hidden_dim (int): Number of measurement outcomes.
        debug (bool): If True, enable debug logging.
    
    Returns:
        ValueLayerSlow: An instance of the slow value layer.
    """
    def __init__(self, n_wires: int, ops: list[str], hidden_dim: int, debug: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.ops = ops
        self.encoding = tq.AmplitudeEncoder()
        self.debug = debug

        self.rx0 = tq.QuantumModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry0 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry1 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice = None) -> torch.Tensor:
        """
        Forward pass of the slow value layer.
        
        This method simulates the quantum circuit step-by-step by applying individual gates
        and performing measurements.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) or (B, C).
            qdev (tq.QuantumDevice, optional): An optional pre-initialized quantum device.
        
        Returns:
            torch.Tensor: Output tensor of shape (B, T, hidden_dim) containing measurement results.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, C = x.shape
        device = x.device

        if qdev is None:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B * T, device=device)
        dim = 1 << self.n_wires
        base_states = torch.zeros((B, dim), device=device, dtype=torch.float32)
        base_states[:, 0] = 1.0
        qdev.set_states(base_states)
        x_flat = x.view(B * T, C)
        self.encoding(qdev, x_flat)
        
        for j, (rx, ry) in enumerate(zip(self.rx0, self.ry0)):
            rx(qdev, wires=j)
            ry(qdev, wires=j)

        for j in range(self.n_wires):
            tqf.cnot(qdev, wires=[j, (j+1) % self.n_wires])

        for j, ry in enumerate(self.ry1):
            ry(qdev, wires=j)
        
        measurements = [expval_joint_analytical(qdev, op) for op in self.ops]
        out = torch.stack(measurements, dim=-1)
        return out.view(B, T, self.hidden_dim).float()
