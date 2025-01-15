import random
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical
import torch
import numpy as np

from torch import nn
from torch import Tensor
import torch.nn.functional as F



class QSA(tq.QuantumModule):
    class QueryKeyLayer(tq.QuantumModule):
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires

            self.encoding = tq.AmplitudeEncoder()
            self.measure = tq.MeasureAll(tq.PauliZ)
            # gates with trainable parameters
            self.rx0 = tq.QuantumModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.ry0 = tq.QuantumModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.ry1 = tq.QuantumModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )


        def forward(self, x: Tensor, qdev: tq.QuantumDevice) -> Tensor:
            """
            input: (B,C)
            otput: (B,)
            """
            x = self.encoding(qdev, x)
            base_states = torch.zeros_like(qdev.get_states_1d())
            base_states[:, 0] = 1.0

            qdev.set_states(base_states)

            for j, (rx, ry) in enumerate(zip(self.rx0, self.ry0)):
                rx(qdev, wires=j)
                ry(qdev, wires=j)
            
            for j in range(self.n_wires):
                for j in range(self.n_wires):
                    tqf.cnot(qdev, wires=[j, (j+1) % self.n_wires])

                for j, ry in enumerate(self.ry1):
                    ry(qdev, wires=j)
            
            out = tq.expval(qdev, 0, tq.PauliZ())

            return out


    class ValueLayer(tq.QuantumModule):
        def __init__(self, n_wires: int, ops: list[str], hidden_dim: int):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_wires = n_wires
            self.ops = ops

            self.encoding = tq.AmplitudeEncoder()
            self.rx0 = tq.QuantumModuleList(
                [tq.RX(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.ry0 = tq.QuantumModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )
            self.ry1 = tq.QuantumModuleList(
                [tq.RY(has_params=True, trainable=True) for _ in range(n_wires)]
            )

        def forward(self, x: Tensor, qdev: tq.QuantumDevice) -> Tensor:
            """
            input: (B, T, C) or (B, C) for single time step
            output: (B, T, C) or (B, C)
            """
            if x.dim() == 2:  # Handle single time step for backward compatibility
                x = x.unsqueeze(1)  # (B, 1, C)

            B, T, C = x.shape
            base_states = torch.zeros((B, 1 << self.n_wires), device=x.device)
            base_states[:, 0] = 1.0

            qdev.set_states(base_states)

            # Flatten batch and time for parallel processing
            x = x.view(B * T, C)
            self.encoding(qdev, x)

            for j, (rx, ry) in enumerate(zip(self.rx0, self.ry0)):
                rx(qdev, wires=j)
                ry(qdev, wires=j)

            for _ in range(self.n_wires):
                for j in range(self.n_wires):
                    tqf.cnot(qdev, wires=[j, (j + 1) % self.n_wires])

                for j, ry in enumerate(self.ry1):
                    ry(qdev, wires=j)

            measurements = [
                expval_joint_analytical(qdev, self.ops[i]).view(B, T, 1)
                for i in range(self.hidden_dim)
            ]
            out = torch.cat(measurements, dim=-1)  # (B, T, hidden_dim)

            return out



    def choose_op(self):
        a = random.randint(0, 3)
        op_s = 'IXYZ'
        op = op_s[a]

        op_elimated='I'
        for _ in range(1,self.n_wires):
            op_elimated = op_elimated + 'I'

        Select_wrong = True
        while Select_wrong:
            for _ in range(1,self.n_wires):
                a = random.randint(0, 3)
                op += op_s[a]
            if op != op_elimated:
                Select_wrong = False


        return op
    
    def __init__(self, n_embed: int, n_context: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_context = n_context
        self.n_wires = int(np.ceil(np.log2(n_embed)))
        self.n_states = 1 << self.n_wires
        self.register_buffer('tril', torch.tril(torch.ones(n_context, n_context)))
        self.drop = nn.Dropout(0.2)

        self.ops = [self.choose_op()[:self.n_wires] for _ in range(1 << self.hidden_dim)]

        self.encoder = tq.AmplitudeEncoder()
        
        self.k = self.ValueLayer(self.n_wires, self.ops, self.hidden_dim)#self.QueryKeyLayer(self.n_wires)
        self.q = self.ValueLayer(self.n_wires, self.ops, self.hidden_dim)#self.QueryKeyLayer(self.n_wires)
        self.v = self.ValueLayer(self.n_wires, self.ops, self.hidden_dim)


    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, embed_size) - input states
        returns: (B, T, embed_size)
        """
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0] * x.shape[1], device=x.device
        )

        q = self.q(x, qdev)  # (B, T, hidden_dim)
        k = self.k(x, qdev)  # (B, T, hidden_dim)
        v = self.v(x, qdev)  # (B, T, hidden_dim)

        q = q.repeat((1, 1, self.n_context // self.hidden_dim))
        k = k.repeat((1, 1, self.n_context // self.hidden_dim))

        alpha = torch.exp(-(q - k) ** 2)
        alpha = alpha.masked_fill(self.tril == 0, float('-inf'))

        normalized_alpha = F.softmax(alpha, dim=-1)
        out = normalized_alpha.permute(0, 2, 1) @ v

        return out
