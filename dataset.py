import numpy as np

import torch
from torch import Tensor
from torchpack.datasets.dataset import Dataset



class Data(Dataset):
    def __init__(
            self, 
            x: np.ndarray, 
            y: np.ndarray, 
        ) -> None:

        # TODO

        self.x_data = torch.from_numpy(x)
        self.y_data = torch.from_numpy(y)

        self._length = len(self.x_data)


    def __getitem__(self, ind: int) -> tuple[Tensor, ...]:
        return self.x_data[ind], self.y_data[ind]


    def __len__(self) -> int:
        return self._length
