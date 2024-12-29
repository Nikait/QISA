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
            otput: (B,C)
            """
            base_states = torch.zeros_like(qdev.get_states_1d())
            base_states[:, 0] = 1.0

            qdev.set_states(base_states)

            x = self.encoding(qdev, x)

            for j, (rx, ry) in enumerate(zip(self.rx0, self.ry0)):
                rx(qdev, wires=j)
                ry(qdev, wires=j)
            
            for j in range(self.n_wires):
                for j in range(self.n_wires):
                    tqf.cnot(qdev, wires=[j, (j+1) % self.n_wires])

                for j, ry in enumerate(self.ry1):
                    ry(qdev, wires=j)
            
            measurements = [expval_joint_analytical(qdev, self.ops[i]) for i in range(self.hidden_dim)]
            out = torch.stack(measurements, dim=-1)

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
        
        self.k = self.QueryKeyLayer(self.n_wires)
        self.q = self.QueryKeyLayer(self.n_wires)
        self.v = self.ValueLayer(self.n_wires, self.ops, self.hidden_dim)


    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B,T,embed_size) - input states
        returns: (B,T,hidden_size)
        """
        
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device
        )
        
        Q_output = torch.stack([self.q(x[:,i], qdev) for i in range(self.n_context)], dim=1)
        K_output = torch.stack([self.k(x[:,i], qdev) for i in range(self.n_context)], dim=1)
        V_output = torch.stack([self.v(x[:,i], qdev) for i in range(self.n_context)], dim=1)


        Q_output = Q_output.repeat((1, 1, self.n_context))
        K_output = K_output.repeat((1, 1, self.n_context))
        
        alpha = torch.exp(-(Q_output-K_output)**2)
        alpha = alpha.masked_fill(self.tril == 0, 0)
        
        # output = []

        # for i in range(self.n_context):
        #     Sum_a=torch.sum(alpha[:,i,:],-1)
        #     div_sum_a=(1 / Sum_a).repeat(self.hidden_dim, self.n_context,1).transpose(0,2)

        #     Sum_w=torch.sum(alpha[:,:,i].repeat((self.hidden_dim,1,1)).transpose(0,2).transpose(0,1)*V_output*div_sum_a,1)
        #     output.append(Sum_w)

        # out = torch.stack(output).transpose(0,1) ## Should we sum with x??

        # a shortcut version of the code above
        
        out = torch.sum(
            alpha.permute(0, 2, 1).unsqueeze(-1) * value.unsqueeze(1) * (1 / torch.sum(alpha, dim=-1, keepdim=True)).unsqueeze(-1),
            dim=2
        )

        return out
