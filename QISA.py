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

def pauli_matrix(op: str) -> torch.Tensor:
    mapping = {
        'I': torch.eye(2, dtype=torch.complex64),
        'X': torch.tensor([[0, 1], [1, 0]], dtype=torch.complex64),
        'Y': torch.tensor([[0, -1j], [1j, 0]], dtype=torch.complex64),
        'Z': torch.tensor([[1, 0], [0, -1]], dtype=torch.complex64)
    }
    return mapping[op].to('cuda')

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

class QSA(nn.Module):
    def __init__(
        self, 
        n_embed: int, 
        n_context: int, 
        head_size: int, 
        layer_id: int, 
        head_id: int,
        version=3
    ):
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

        super().__init__()
        assert version in [1, 2, 3], "version must be 1, 2 or 3"
        
        print(f"Instantiate QSA v{version} for the head {head_id}")

        self.version = version
        self.__use_kv_one_measurement = True if version == 1 else False
        self.__precomputed = False
        self.head_size = head_size
        self.n_context = n_context
        self.transformer_layer_id = layer_id  # transformer layer number
        self.head_id = head_id                # head index
        self.register_buffer('tril', torch.tril(torch.ones(n_context, n_context)))
        self.n_wires = int(np.ceil(np.log2(n_embed)))
        self.ops = [self.choose_op() for _ in range(self.head_size)]
        print(f"Generated Pauli strings for layer {layer_id} head {head_id}: ", self.ops)
        
        if version != 3:
            self.q_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id, qk=self.__use_kv_one_measurement)
            self.k_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id, qk=self.__use_kv_one_measurement)
            self.q_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size, qk=self.__use_kv_one_measurement)
            self.k_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size, qk=self.__use_kv_one_measurement)
        
        else:
            self.q_linear = nn.Linear(n_embed, head_size)
            self.k_linear = nn.Linear(n_embed, head_size)
            self.v_linear = nn.Linear(n_embed, n_embed, bias=False)
        
        self.v_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size, n_embed,
                                     self.transformer_layer_id, self.head_id)
        self.v_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size)

        # Slow branch for training
        

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

        if self.version == 3:
            q = self.q_linear(x)
            k = self.k_linear(x)
            v = self.v_linear(x)
        else:
            q = self.q_slow(x, qdev)
            k = self.k_slow(x, qdev)
        v = self.v_slow(v, qdev)

        q = q.unsqueeze(2)
        k = k.unsqueeze(1)
        alpha = torch.exp(-((q - k) ** 2).sum(dim=-1))
        alpha = alpha.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
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
        B, T, C = x.shape
        if self.version == 3:
            q = self.q_linear(x)
            k = self.k_linear(x)
        else:
            q = self.q_fast(x)
            k = self.k_fast(x)
        v = self.v_fast(x)

        q = q.unsqueeze(2)
        k = k.unsqueeze(1)
        alpha = torch.exp(-((q - k) ** 2).sum(dim=-1))
        alpha = alpha.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
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

        Then precomputes observables as O' = <psi| U^dagger O U |psi>
        
        Returns:
            None
        """
        super().eval()
        # Copy parameters from slow to fast for quantum parts
        if self.version != 3:
            self.q_fast.rx0_params = self.q_slow.rx0
            self.q_fast.ry0_params = self.q_slow.ry0
            self.q_fast.ry1_params = self.q_slow.ry1

            self.k_fast.rx0_params = self.k_slow.rx0
            self.k_fast.ry0_params = self.k_slow.ry0
            self.k_fast.ry1_params = self.k_slow.ry1
        
        # Note: No copy needed for v_fast, as unitaries are removed

        # Build unitaries (skip for v_fast)
        if self.version != 3:
            self.q_fast._build_unitary(self.unitary_count, "q_fast", self.transformer_layer_id, self.head_id)
            self.unitary_count += 1
            self.k_fast._build_unitary(self.unitary_count, "k_fast", self.transformer_layer_id, self.head_id)
            self.unitary_count += 1

        print("[INFO] Precomputation of Unitaries completed.")

        # Precompute observables (only for v_fast, with absorption)
        self.v_fast.precompute_observables(self.v_linear)

        print("[INFO] Precomputation of observables completed.")

        self.unitary_count += 1

        


# ----------------- Fast Branch: ValueLayerFast -----------------

class ValueLayerFast(nn.Module):
    """
    Modified fast value layer with NO unitaries.
    
    - Uses raw amplitude loading (no normalization) for exact absorption of the classical linear layer.
    - Precomputes transformed observables O' = L^\dagger O L after training.
    - At inference: direct measurement on raw encoded x, no classical linear needed.
    """
    def __init__(self, n_wires: int, ops: list[str], head_size: int, n_embed: int,
                                     transformer_layer_id: int, head_id: int):
        super().__init__()
        self.n_wires = n_wires
        self.ops = ops
        self.head_size = head_size
        self.n_embed = n_embed
        self.transformer_layer_id = transformer_layer_id
        self.head_id = head_id

        self.dim = 1 << n_wires

        # Precomputed transformed observables O' (one per measurement outcome)
        self.obs_primes = nn.ParameterList([
            nn.Parameter(torch.zeros((self.dim, self.dim), dtype=torch.complex64), requires_grad=False)
            for _ in range(self.head_size)
        ])

    def precompute_observables(self, v_linear: nn.Linear):
        """
        Absorb the classical linear layer into the observables.
        
        Computes O' = L^\dagger O L for each fixed Pauli string O,
        where L = v_linear.weight  (the linear transformation matrix such that v_col = L @ x_col).
        
        Pads L to the full 2**n_wires dimension (assuming zero-padding in encoding).
        """
        device = v_linear.weight.device
        # Linear transformation: v = x @ weight.t() , but for columns, v_col = weight @ x_col => L = weight
        L = v_linear.weight.to(dtype=torch.complex64)  # (n_embed, n_embed)

        # Pad to full Hilbert space (data subspace only)
        L_full = torch.zeros((self.dim, self.dim), dtype=torch.complex64, device=device)
        L_full[:self.n_embed, :self.n_embed] = L

        L_dag = L_full.conj().t()

        for idx, op in enumerate(self.ops):
            # Build observable O from Pauli string
            mats = [pauli_matrix(c) for c in op]
            O = mats[0]
            for m in mats[1:]:
                O = torch.kron(O, m).to(device)

            # Transformed observable O' = L^\dagger O L
            O_prime = L_dag @ O @ L_full

            self.obs_primes[idx].data.copy_(O_prime)

        print(f"[INFO] Precomputed absorbed observables for value in layer {self.transformer_layer_id} head {self.head_id}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast inference forward: encode raw x (no linear, no normalization), measure precomputed O'.
        
        Output shape: (B, T, head_size)
        """
        B, T, C = x.shape
        assert C == self.n_embed, f"Expected channel dim {self.n_embed}, got {C}"
        x_flat = x.view(B * T, C)  # (B*T, n_embed)
        device = x.device

        dim = self.dim
        # Raw amplitude loading: place data in first n_embed basis states, pad zeros, NO normalization
        psi = torch.zeros((B * T, dim), dtype=torch.complex64, device=device)
        psi[:, :self.n_embed] = x_flat.to(dtype=torch.complex64)

        # Compute raw expectations psi^\dagger O' psi for each transformed observable
        exps = []
        for O_prime in self.obs_primes:
            O_psi = torch.matmul(O_prime, psi.t()).t()  # (B*T, dim)
            exp = (psi.conj() * O_psi).sum(dim=-1).real  # scalar per batch element
            exps.append(exp)

        out = torch.stack(exps, dim=-1)  # (B*T, head_size)
        return out.view(B, T, self.head_size)


# ------------------ Slow Branch: ValueLayerSlow ------------------

class ValueLayerSlow(nn.Module):
    """
    Simplified slow value layer for training (no gates, raw classical computation without normalization).
    """
    def __init__(self, n_wires: int, ops: list[str], hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.ops = ops
        self.dim = 1 << n_wires

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice = None) -> torch.Tensor:
        """
        Forward pass of the slow value layer (classical simulation without normalization).
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, T, C) or (B, C).
            qdev (tq.QuantumDevice, optional): Ignored (for compatibility).
        
        Returns:
            torch.Tensor: Output tensor of shape (B, T, hidden_dim) containing measurement results.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, C = x.shape
        device = x.device

        x_flat = x.view(B * T, C)  # (B*T, n_embed)

        states = torch.zeros((B * T, self.dim), dtype=torch.complex64, device=device)
        states[:, :C] = x_flat.to(dtype=torch.complex64)

        exps = []
        for op in self.ops:
            mats = [pauli_matrix(c) for c in op]
            O = mats[0]
            for m in mats[1:]:
                O = torch.kron(O, m).to(device)
            O = O.to(dtype=torch.complex64)
            O_psi = torch.matmul(O, states.t()).t()
            exp = (states.conj() * O_psi).sum(dim=-1).real
            exps.append(exp)

        out = torch.stack(exps, dim=-1)
        return out.view(B, T, self.hidden_dim)
