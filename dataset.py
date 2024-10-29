import torch
from torch import Tensor, tensor

from typing import Dict, Union


class TextDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            text: str, 
            characters: int, 
            block_size: int, 
            train: bool=True
        ):
        """
        Construct character level encding-decoding
        """
        super().__init__()
        self.text = text
        self.characters = characters
        self.block_size = block_size
        self.train = train

        stoi = {s:i for i,s in enumerate(self.characters)}
        itos = {i:s for i,s in enumerate(self.characters)}

        self.enc = lambda s: [stoi[c] for c in s]
        self.dec = lambda l: ''.join([itos[i] for i in l])
        self.data = tensor(self.enc(self.text), dtype=torch.long)


    def __getitem__(self, index: int) -> Dict[str, Union[Tensor, str]]:
        idx = index # torch.randint(len(self.data) - self.block_size, size=(1,))
        if self.train:
            idx = torch.randint(len(self.data) - self.block_size, size=(1,))
        
        X = self.data[idx:idx + self.block_size]
        y = self.data[idx+1 : idx+self.block_size+1]

        return (X, y)

    def __len__(self) -> int:
        if self.train:
            return 5000
        return len(self.data) - self.block_size
