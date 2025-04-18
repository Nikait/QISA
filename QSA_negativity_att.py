from os import stat
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import cmath
import random
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical, expval

torch.manual_seed(0)
random.seed(0)

# ----------------- Helper Function -------------------

def batch_kron(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Computes the Kronecker product for batches.
    
    Parameters:
      a: Tensor of shape (B, n)
      b: Tensor of shape (B, m)
      
    Returns:
      Tensor of shape (B, n*m) where each batch element is the kronecker product of a and b.
    """
    return torch.einsum('bi,bj->bij', a, b).reshape(a.shape[0], -1)

# ------------- Gate Matrices and Operator Expansion ---------------

def rx_matrix(theta: float) -> torch.Tensor:
    """
    Returns the RX rotation matrix for a given angle.
    """
    return torch.tensor([[math.cos(theta/2), -1j*math.sin(theta/2)],
                         [-1j*math.sin(theta/2), math.cos(theta/2)]],
                        dtype=torch.complex64)

def ry_matrix(theta: float) -> torch.Tensor:
    """
    Returns the RY rotation matrix for a given angle.
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
    """
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0]], dtype=torch.complex64)

def expand_operator(gate: torch.Tensor, target_qubits: list, n: int) -> torch.Tensor:
    """
    Expands a given gate operator to a full operator acting on n qubits.
    
    Parameters:
      gate: The gate matrix to expand.
      target_qubits: A list of target qubit indices where the gate acts.
      n: Total number of qubits.
    """
    dim = 1 << n  # 2^n
    U_full = torch.zeros((dim, dim), dtype=gate.dtype, device=gate.device)
    m = len(target_qubits)
    for i in range(dim):
        # Represent the number i as a binary vector of length n (MSB first)
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

# ------------------ Negativity Circuit -----------------------

class NegativityCircuit(tq.QuantumModule):
    def __init__(self, n):
        super().__init__()
        self.n = n
        self.total_qubits = 1 + 2 * n  # with ancilla

    def forward(self, qdev: tq.QuantumDevice, state: torch.Tensor):
        """
        Computes the negativity of the state using a quantum circuit.
        
        This version supports batched input.
        
        Parameters:
          qdev: A quantum device with batch size equal to state.shape[0].
          state: A batched state tensor of shape (B, D)
          
        Returns:
          A tensor of negativity values with shape (B,).
        """
        B = state.shape[0]
        # Normalize each state in the batch
        state = state / torch.norm(state, dim=1, keepdim=True)
        # Construct state_with_ancilla: |0> ⊗ |psi> for each element in the batch
        zero_ancilla = torch.tensor([1.0 + 0j, 0.0 + 0j], device=state.device).unsqueeze(0).repeat(B, 1)
        state_with_ancilla = batch_kron(zero_ancilla, state)
        # Full state: |state_with_ancilla> ⊗ |psi>
        full_state = batch_kron(state_with_ancilla, state)
        qdev.set_states(full_state)
        qdev.h(wires=0)
        qdev.cswap(wires=[0, self.n, 2 * self.n])
        qdev.h(wires=0)
        exp_z = expval(qdev, 0, tq.PauliZ())
        return 1 - exp_z

# ------------------ Quantum Self-Attention Layer -----------------------

class QSA(nn.Module):
    """
    Quantum Self-Attention layer.
    
    Parameters:
      n_embed (int): embedding size.
      n_context (int): number of tokens (context size).
      head_size (int): head size (number of measurements).
      layer_id (int): identifier for the transformer layer.
      head_id (int): identifier for the head within the layer.
      version (int): determines the measurement mode for query/key.
    """
    def __init__(
        self, 
        n_embed: int, 
        n_context: int, 
        head_size: int, 
        layer_id: int, 
        head_id: int,
        version=2
    ):
        super().__init__()
        assert version in [1, 2], "the version of the QSA has to be equal to 1 or 2"
        self.__use_kv_one_measurement = True if version == 1 else False
        self.__precomputed = False
        self.head_size = head_size
        self.n_context = n_context
        self.transformer_layer_id = layer_id
        self.head_id = head_id
        # Lower triangular matrix for attention masking (causal mask)
        self.register_buffer('tril', torch.tril(torch.ones(n_context, n_context)))
        # The number of quantum wires is determined as ceil(log2(n_embed))
        self.n_wires = int(np.ceil(np.log2(n_embed)))
        self.ops = [self.choose_op() for _ in range(self.head_size)]
        print(f"Generated Pauli strings for layer {layer_id} head {head_id}: ", self.ops)
        
        # Fast branch for inference:
        # For query use a special flag layer_type="query"
        self.q_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id,
                                     layer_type="query", qk=self.__use_kv_one_measurement)
        self.v_fast = ValueLayerFast(self.n_wires, self.ops, self.head_size,
                                     self.transformer_layer_id, self.head_id,
                                     layer_type="value")
        
        # Slow branch for training:
        self.q_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size,
                                     layer_type="query", qk=self.__use_kv_one_measurement)
        self.v_slow = ValueLayerSlow(self.n_wires, self.ops, self.head_size,
                                     layer_type="value")
        self.unitary_count = 0

    def choose_op(self) -> str:
        """
        Randomly selects a Pauli string for measurement.
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

    def _forward_fast(self, x: torch.Tensor) -> torch.Tensor:
        """
        Fast forward pass for inference.
        For the query branch (q_fast), pairs of states are formed,
        then the negativity of each pair is computed and the attention matrix is calculated.
        The negativity computation is performed in a batched way.
        """
        # Get quantum state pairs from the query and value fast branches.
        q = self.q_fast(x)  # q has shape (B, T^2, D), where D = 2^(2*n_wires)
        v = self.v_fast(x)  # v has standard shape (B, T, head_size)
        B = x.shape[0]
        T = self.n_context
        
        # Reshape q to (B, T, T, D)
        q_reshaped = q.view(B, T, T, -1)
        # Flatten the batch and token dimensions: shape (B*T*T, D)
        psi_flat = q_reshaped.view(B * T * T, -1)
        negativity_circuit = NegativityCircuit(n=self.n_wires * 2)
        # Create a quantum device with batch size equal to B*T*T
        qdev_neg = tq.QuantumDevice(n_wires=1 + 2 * 2 * self.n_wires, bsz=B * T * T, device=x.device)
        negativity_values = negativity_circuit(qdev_neg, psi_flat)
        # Reshape the negativity values back to (B, T, T)
        attn_scores = negativity_values.view(B, T, T)
        # Apply causal masking using the lower-triangular mask
        attn_scores = attn_scores.masked_fill(self.tril == 0, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('bij,bjk->bik', attn, v)
        return out

    def _forward_slow(self, x: torch.Tensor) -> torch.Tensor:
        """
        Slow forward pass for training.
        Behavior is similar to the fast branch: the query branch is used to form pairs.
        The negativity is computed in a batched manner.
        """
        q = self.q_slow(x)  # q has shape (B, T^2, D)
        v = self.v_slow(x)  # value branch
        B = x.shape[0]
        T = self.n_context

        q_reshaped = q.view(B, T, T, -1)
        psi_flat = q_reshaped.view(B * T * T, -1)
        # In the slow branch, the negativity circuit is defined for 2*n_wires,
        # so the quantum device is created with the corresponding number of wires.
        negativity_circuit = NegativityCircuit(n=self.n_wires * 2)
        qdev_neg = tq.QuantumDevice(n_wires=1 + 2 * 2 * self.n_wires, bsz=B * T * T, device=x.device)
        negativity_values = negativity_circuit(qdev_neg, psi_flat)
        attn_scores = negativity_values.view(B, T, T)
        # Apply causal masking using the lower-triangular mask
        attn_scores = attn_scores.masked_fill(self.tril == 0, float('-inf'))
        attn = F.softmax(attn_scores, dim=-1)
        out = torch.einsum('bij,bjk->bik', attn, v)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quantum self-attention forward pass:
          - During training, the slow branch (_forward_slow) is used.
          - During inference, the fast branch (_forward_fast) is applied.
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
        Precomputes the unitary operators for the fast branch.
        Parameters are copied from the slow branch and the matrices are assigned unique names.
        """
        super().eval()
        print("[INFO] Switching to inference mode. Starting precomputation of unitaries...")
        self.q_fast.rx0_params = self.q_slow.rx0
        self.q_fast.ry0_params = self.q_slow.ry0
        self.q_fast.ry1_params = self.q_slow.ry1

        self.v_fast.rx0_params = self.v_slow.rx0
        self.v_fast.ry0_params = self.v_slow.ry0
        self.v_fast.ry1_params = self.v_slow.ry1

        self.q_fast._build_unitary(self.unitary_count, layer_id="q_fast",
                                   transformer_layer_id=self.transformer_layer_id, head_id=self.head_id)
        self.unitary_count += 1
        self.v_fast._build_unitary(self.unitary_count, layer_id="v_fast",
                                   transformer_layer_id=self.transformer_layer_id, head_id=self.head_id)
        self.unitary_count += 1

        print("[INFO] Precomputation completed. Fast branch matrices updated.")

# ------------------ Fast Branch: ValueLayerFast -----------------------

class ValueLayerFast(nn.Module):
    """
    Fast value layer for inference.
    Depending on the layer_type:
      - "query": pairs of states are formed for each token,
      - "value": standard quantum transformation with measurements.
    """
    def __init__(self, n_wires: int, ops: list[str], hidden_dim: int,
                 transformer_layer_id: int, head_id: int, layer_type: str = "value", qk: bool = False):
        super().__init__()
        self.layer_type = layer_type  # "query" or "value"
        self.n_wires = n_wires
        self.ops = ops
        self.hidden_dim = hidden_dim
        self.transformer_layer_id = transformer_layer_id
        self.head_id = head_id
        self.__qk = qk
        if self.layer_type == "query":
            self.dim = 1 << (2 * n_wires)
        else:
            self.dim = 1 << n_wires
        self.encoding = tq.AmplitudeEncoder()
        self.token_encoder = nn.Linear(hidden_dim, 1 << n_wires)
        self.register_buffer('U', torch.eye(self.dim, dtype=torch.complex64))
        self.unitary_counter = 0
        self.current_unitary_id = None

        self.rx0_params = tq.QuantumModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry0_params = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
        self.ry1_params = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])

    def _build_unitary(self, unitary_count: int, layer_id: str,
                       transformer_layer_id: int, head_id: int):
        n = self.n_wires
        if self.layer_type == "query":
            total_wires = 2 * n
        else:
            total_wires = n
        dim = 1 << total_wires
        device = self.rx0_params[0].params.device
        U_total = torch.eye(dim, dtype=torch.complex64, device=device)

        if self.layer_type == "value":
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
                expanded_cnot = expand_operator(cnot_matrix(), [j, (j+1) % n], n).to(device=device, dtype=torch.complex64)
                U_total = expanded_cnot @ U_total
            for j in range(n):
                ry1_val = self.ry1_params[j].params.clone()
                mat = ry_matrix(ry1_val).to('cpu')
                expanded_ry1 = expand_operator(mat, [j], n).to(device=device, dtype=torch.complex64)
                U_total = expanded_ry1 @ U_total
        else:
            U_total = torch.eye(dim, dtype=torch.complex64, device=device)
        unique_name = f"layer{transformer_layer_id}_head{head_id}_{layer_id}_U{unitary_count}"
        if hasattr(self, unique_name):
            getattr(self, unique_name).data.copy_(U_total)
        else:
            self.register_buffer(unique_name, U_total)
        self.current_unitary_id = unique_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        device = x.device
        if self.layer_type == "query":
            encoded_list = []
            for t in range(T):
                token = x[:, t, :]
                vec = self.token_encoder(token)
                vec = vec / vec.norm(dim=-1, keepdim=True)
                encoded_list.append(vec)
            paired_states = []
            for i in range(T):
                for j in range(T):
                    a = encoded_list[i]
                    b = encoded_list[j]
                    pair = (a.unsqueeze(2) * b.unsqueeze(1)).view(B, -1)
                    paired_states.append(pair)
            state_pairs = torch.stack(paired_states, dim=1)
            state_pairs = state_pairs / state_pairs.norm(dim=-1, keepdim=True)
            # Convert to complex type
            state_pairs = state_pairs.to(torch.complex64)
            state_pairs_flat = state_pairs.view(B * T * T, -1)
            U = getattr(self, self.current_unitary_id) if self.current_unitary_id is not None else self.U
            transformed = torch.matmul(state_pairs_flat, U)
            transformed = transformed.view(B, T * T, -1)
            return transformed
        else:
            x_flat = x.view(B * T, C)
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
            psi_out = tqf.apply_unitary_bmm(psi_reshaped, U, list(range(self.n_wires)))
            psi_out = psi_out.view(B * T, -1)
            qdev.set_states(psi_out)
            if self.__qk:
                out = tq.expval(qdev, 0, tq.PauliZ())
                measurements = [out] * len(self.ops)
            else:
                measurements = [expval_joint_analytical(qdev, op) for op in self.ops]
            out = torch.stack(measurements, dim=-1)
            return out.view(B, T, self.hidden_dim).float()

# ------------------ Slow Branch: ValueLayerSlow -----------------------

class ValueLayerSlow(nn.Module):
    """
    Slow value layer for training.
    If layer_type == "query", pairs of states are formed and a step-by-step simulation
    of the quantum circuit is performed on 2*n_wires qubits with a unitary transformation
    (analogous to the fast branch).
    For "value", a step-by-step simulation is performed on n_wires.
    """
    def __init__(self, n_wires: int, ops: list[str], hidden_dim: int,
                 layer_type: str = "value", qk: bool = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_wires = n_wires
        self.ops = ops
        self.layer_type = layer_type
        self.encoding = tq.AmplitudeEncoder()
        self.__qk = qk
        if self.layer_type == "query":
            self.token_encoder = nn.Linear(hidden_dim, 1 << n_wires)
            # Generate operators for 2*n_wires qubits
            self.rx0 = tq.QuantumModuleList([tq.RX(has_params=True, trainable=True) for _ in range(2 * n_wires)])
            self.ry0 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(2 * n_wires)])
            self.ry1 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(2 * n_wires)])
        else:
            self.rx0 = tq.QuantumModuleList([tq.RX(has_params=True, trainable=True) for _ in range(n_wires)])
            self.ry0 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])
            self.ry1 = tq.QuantumModuleList([tq.RY(has_params=True, trainable=True) for _ in range(n_wires)])

    def forward(self, x: torch.Tensor, qdev: tq.QuantumDevice = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        B, T, C = x.shape
        device = x.device

        if self.layer_type == "query":
            # Form pairs of states, similar to the fast branch
            encoded_list = []
            for t in range(T):
                token = x[:, t, :]
                vec = self.token_encoder(token)
                vec = vec / vec.norm(dim=-1, keepdim=True)
                encoded_list.append(vec)
            paired_states = []
            for i in range(T):
                for j in range(T):
                    a = encoded_list[i]
                    b = encoded_list[j]
                    pair = (a.unsqueeze(2) * b.unsqueeze(1)).view(B, -1)
                    paired_states.append(pair)
            state_pairs = torch.stack(paired_states, dim=1)
            state_pairs = state_pairs / state_pairs.norm(dim=-1, keepdim=True)
            # Convert to complex type
            state_pairs = state_pairs.to(torch.complex64)
            total_wires = 2 * self.n_wires
            dim = 1 << total_wires
            if qdev is None:
                qdev = tq.QuantumDevice(n_wires=total_wires, bsz=B * T * T, device=device)
            base_states = torch.zeros((B * T * T, dim), device=device, dtype=torch.complex64)
            base_states[:, 0] = 1.0
            qdev.set_states(base_states)
            # Replace states with the formed pairs
            qdev.set_states(state_pairs.view(B * T * T, -1))
            for j, (rx, ry) in enumerate(zip(self.rx0, self.ry0)):
                rx(qdev, wires=j)
                ry(qdev, wires=j)
            for j in range(total_wires):
                tqf.cnot(qdev, wires=[j, (j+1) % total_wires])
            for j, ry in enumerate(self.ry1):
                ry(qdev, wires=j)
            final_state = qdev.states
            return final_state.reshape(B, T * T, dim)
        else:
            # Value branch: step-by-step simulation of the quantum circuit on n_wires
            if qdev is None:
                qdev = tq.QuantumDevice(n_wires=self.n_wires, bsz=B * T, device=device)
            dim = 1 << self.n_wires
            base_states = torch.zeros((B, dim), device=device, dtype=torch.complex64)
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
            if self.__qk:
                out = tq.expval(qdev, 0, tq.PauliZ())
                measurements = [out] * len(self.ops)
            else:
                measurements = [expval_joint_analytical(qdev, op) for op in self.ops]
            out = torch.stack(measurements, dim=-1)
            return out.view(B, T, self.hidden_dim).float()
