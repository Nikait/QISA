import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from abc import abstractmethod
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
    def __init__(self, n_embed: int, n_context: int, head_size: int, layer_id: int, head_id: int, version=1):
        super().__init__()
        assert version in [1, 2], "the version of the QSA has to be equal 1 or 2"

        self.__use_kv_one_measurement = True if version == 1 else False
        self.__precomputed = False
        self.head_size = head_size
        self.n_context = n_context
        self.transformer_layer_id = layer_id  # transformer layer number
        self.head_id = head_id                # head index
        self.n_wires = int(np.ceil(np.log2(n_embed)))
        self.ops = [self.choose_op() for _ in range(self.head_size)]
        self.register_buffer('tril', torch.tril(torch.ones(n_context, n_context)))
        

        print(f"Generated Pauli strings for layer {layer_id} head {head_id}: ", self.ops)
        
        # Fast branch for inference - pass both identifiers
        self.q_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id, debug=True, qk=self.__use_kv_one_measurement)
        self.k_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id, debug=True, qk=self.__use_kv_one_measurement)
        self.v_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id, debug=True)
        
        # Slow branch for training
        self.q_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size, debug=True, qk=self.__use_kv_one_measurement)
        self.k_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size, debug=True, qk=self.__use_kv_one_measurement)
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
            self.__precomputed = False
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




class ValueLayerBase(nn.Module):
    """
    Base class for quantum value layers.

    Provides shared initialization and forward pass logic, delegating quantum evolution to subclasses.

    Args:
        n_wires (int): Number of qubits in the quantum circuit.
        ops (list[str]): List of Pauli operators for measurement.
        hidden_dim (int): Dimension of the output (number of measurements).
        debug (bool): If True, enables debug mode (default: False).
    """
    def __init__(self, n_wires: int, ops: list[str], hidden_dim: int, debug: bool = False, qk=True):
        super().__init__()
        self.__qk = qk
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.ops = ops
        self.encoding = tq.AmplitudeEncoder()
        self.debug = debug

        # Quantum rotation gates for all wires
        self.rx0 = tq.QuantumModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry0 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry1 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice = None) -> torch.Tensor:
        """
        Forward pass for the value layer.

        Handles input processing, state initialization, encoding, quantum evolution, and measurements.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C) or (B, T, C).
            qdev (tq.QuantumDevice, optional): Pre-initialized quantum device (default: None).

        Returns:
            torch.Tensor: Output tensor of shape (B, T, hidden_dim) with measurement results.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add time dimension if missing
        B, T, C = x.shape
        device = x.device

        # Initialize quantum device if not provided
        if qdev is None:
            qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B * T, device=device)

        # Set initial state to |0...0>
        dim = 1 << self.n_wires
        base_states = torch.zeros((B * T, dim), device=device, dtype=torch.complex64)
        base_states[:, 0] = 1.0
        qdev.set_states(base_states)

        # Encode input data
        x_flat = x.view(B * T, C)
        self.encoding(qdev, x_flat)

        # Delegate quantum operations to subclass
        self.apply_quantum_evolution(qdev)

        # Compute measurements
        if self.__qk:
            out = tq.expval(qdev, 0, tq.PauliZ())
            measurements = [out] * len(self.ops)
        else:
            measurements = [expval_joint_analytical(qdev, op) for op in self.ops]

        out = torch.stack(measurements, dim=-1)
        return out.view(B, T, self.hidden_dim).float()

    @abstractmethod
    def apply_quantum_evolution(self, qdev: tq.QuantumDevice):
        """
        Abstract method to apply quantum operations.

        Subclasses implement this to define their specific quantum evolution logic.

        Args:
            qdev (tq.QuantumDevice): Quantum device to operate on.
        """
        pass

class ValueLayerSlow(ValueLayerBase):
    """
    Slow value layer for training with step-by-step quantum operations.

    Applies gates individually, suitable for gradient computation.

    Args:
        n_wires (int): Number of qubits.
        ops (list[str]): List of Pauli operators for measurement.
        hidden_dim (int): Output dimension.
        debug (bool): If True, enables debug mode (default: False).
    """
    def __init__(self, n_wires, ops, hidden_dim, debug = False, qk=True):
        super().__init__(n_wires, ops, hidden_dim, debug, qk)

    def apply_quantum_evolution(self, qdev: tq.QuantumDevice):
        """
        Applies quantum gates sequentially on the device.

        Args:
            qdev (tq.QuantumDevice): Quantum device to apply operations on.
        """
        # Apply RX and RY gates per wire
        for j, (rx, ry) in enumerate(zip(self.rx0, self.ry0)):
            rx(qdev, wires=j)
            ry(qdev, wires=j)

        # Apply CNOT gates in a ring topology
        for j in range(self.n_wires):
            tqf.cnot(qdev, wires=[j, (j + 1) % self.n_wires])

        # Apply final RY gates
        for j, ry in enumerate(self.ry1):
            ry(qdev, wires=j)

class ValueLayerFast(ValueLayerBase):
    """
    Fast value layer for inference with precomputed unitary evolution.

    Uses a single unitary matrix application for efficiency.

    Args:
        n_wires (int): Number of qubits.
        ops (list[str]): List of Pauli operators for measurement.
        hidden_dim (int): Output dimension.
        transformer_layer_id (int): Transformer layer identifier.
        head_id (int): Attention head identifier.
        debug (bool): If True, enables debug mode (default: False).
    """
    def __init__(self, n_wires: int, ops: list[str], hidden_dim: int, transformer_layer_id: int, head_id: int, debug: bool = False, qk=True):
        super().__init__(n_wires, ops, hidden_dim, debug)
        self.transformer_layer_id = transformer_layer_id
        self.head_id = head_id
        self.unitary_counter = 0
        self.current_unitary_id = None

    def _build_unitary(self, unitary_count: int, layer_id: str):
        """
        Builds and registers the unitary matrix for the quantum circuit.

        Args:
            unitary_count (int): Counter for unique unitary buffer names.
            layer_id (str): Context identifier (e.g., 'q_fast').
        """
        n = self.n_wires
        dim = 1 << n
        device = self.rx0[0].params.device
        U_total = torch.eye(dim, dtype=torch.complex64, device=device)

        # Build unitary from RX and RY gates
        for j in range(n):
            rx_val = self.rx0[j].params.item()
            ry_val = self.ry0[j].params
            expanded_rx = expand_operator(rx_matrix(torch.tensor(rx_val, device=device)), [j], n).to(dtype=torch.complex64)
            U_total = expanded_rx @ U_total
            expanded_ry = expand_operator(ry_matrix(ry_val), [j], n).to(device=device, dtype=torch.complex64)
            U_total = expanded_ry @ U_total

        # Add CNOT gates
        for j in range(n):
            expanded_cnot = expand_operator(cnot_matrix(), [j, (j + 1) % n], n).to(device=device, dtype=torch.complex64)
            U_total = expanded_cnot @ U_total

        # Add final RY gates
        for j in range(n):
            ry1_val = self.ry1[j].params
            expanded_ry1 = expand_operator(ry_matrix(ry1_val), [j], n).to(device=device, dtype=torch.complex64)
            U_total = expanded_ry1 @ U_total

        # Register unitary buffer with unique name
        unique_name = f"layer{self.transformer_layer_id}_head{self.head_id}_{layer_id}_U{unitary_count}"
        self.register_buffer(unique_name, U_total)
        self.current_unitary_id = unique_name
        self.unitary_counter += 1

    def apply_quantum_evolution(self, qdev: tq.QuantumDevice):
        """
        Applies the precomputed unitary matrix to the quantum state.

        Args:
            qdev (tq.QuantumDevice): Quantum device to operate on.
        """
        if self.current_unitary_id is None:
            raise ValueError("Unitary operator not built. Call _build_unitary first.")
        U = getattr(self, self.current_unitary_id).to(device=qdev.states.device)
        psi = qdev.states.view(qdev.bsz, *([2] * self.n_wires))
        psi_out = tqf.apply_unitary_bmm(psi, U, list(range(self.n_wires)))
        qdev.set_states(psi_out.view(qdev.bsz, -1))
