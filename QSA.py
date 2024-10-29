import random
import torchquantum as tq
import torchquantum.functional as tqf
from torchquantum.measurement import expval_joint_analytical
import torch

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
        def __init__(self, n_wires: int):
            super().__init__()
            self.n_wires = n_wires

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

        def choose_op(self):
            a = random.randint(0, 3)
            op_s = 'IXYZ'
            op = op_s[a]

            op_elimated='I'
            for _ in range(1,self.n_wires):
                op_elimated=op_elimated+'I'

            Select_wrong=True
            while Select_wrong:
                for i in range(1,self.n_wires):
                    a = random.randint(0, 3)
                    op += op_s[a]
                if op!=op_elimated:
                    Select_wrong=False
            return op

        def forward(self, x: Tensor, qdev: tq.QuantumDevice) -> Tensor:
            """
            input: (B,C)
            otput: (B,C)
            """

            x = self.encoding(qdev, x)
            op = self.choose_op()[:self.n_wires]
            
            for j, (rx, ry) in enumerate(zip(self.rx0, self.ry0)):
                rx(qdev, wires=j)
                ry(qdev, wires=j)
            
            for j in range(self.n_wires):
                for j in range(self.n_wires):
                    tqf.cnot(qdev, wires=[j, (j+1) % self.n_wires])

                for j, ry in enumerate(self.ry1):
                    ry(qdev, wires=j)
            

            measurements = [expval_joint_analytical(qdev, op) for i in range(1 << self.n_wires)]
            out = torch.stack(measurements, dim=-1)

            return out


    def __init__(self, n_context: int, n_wires: int, n_var_layers: int) -> None:
        super().__init__()
        self.n_context = n_context
        self.n_wires = n_wires
        self.n_states = 1 << n_wires
        self.n_var_layers = n_var_layers
        self.encoder = tq.AmplitudeEncoder()
        
        self.k = tq.QuantumModuleList(
            [self.QueryKeyLayer(self.n_wires) for _ in range(self.n_context)]
        )
        self.q = tq.QuantumModuleList(
            [self.QueryKeyLayer(self.n_wires) for _ in range(self.n_context)]
        )
        self.v = tq.QuantumModuleList(
            [self.ValueLayer(self.n_wires) for _ in range(self.n_context)]
        )

        self.measure = tq.MeasureAll(tq.PauliZ)
        self.fc = torch.nn.Linear(self.n_wires, 1)
   

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B,T,C) - input states
        returns: (B,T,C)
        """
        qdev = tq.QuantumDevice(
            n_wires=self.n_wires, bsz=x.shape[0], device=x.device
        )

        Q_output = torch.stack([self.q[i](x[:,i], qdev) for i in range(self.n_context)])
        K_output = torch.stack([self.k[i](x[:,i], qdev) for i in range(self.n_context)])
        V_output = torch.stack([self.v[i](x[:,i], qdev) for i in range(self.n_context)])

        Q_output = Q_output.transpose(0,2).repeat((self.n_context, 1, 1))
        K_output = K_output.transpose(0,2).repeat((self.n_context, 1, 1)).transpose(0, 2)
        
        alpha=torch.exp(-(Q_output-K_output)**2)
        alpha=alpha.transpose(0,1)
        V_output=V_output.transpose(0,1)
        output=[]

        for i in range(self.n_context):
            Sum_a=torch.sum(alpha[:,i,:],-1)
            div_sum_a=(1/Sum_a).repeat(self.n_states, self.n_context,1).transpose(0,2)

            Sum_w=torch.sum(alpha[:,:,i].repeat((self.n_states,1,1)).transpose(0,2).transpose(0,1)*V_output*div_sum_a,1)
            output.append(Sum_w)

        out = x + torch.stack(output).transpose(0,1)

        return out
