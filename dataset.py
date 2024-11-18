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

        self.stoi = {s:i for i,s in enumerate(self.characters)}
        self.itos = {i:s for i,s in enumerate(self.characters)}

        self.enc = lambda s: [self.stoi[c] for c in s]
        self.dec = lambda l: ''.join([self.itos[i] for i in l])

        self.data = tensor(self.enc(self.text), dtype=torch.long)

        chunk_len = int(len(self.data) * 0.8)
        if self.train:
            self.data = self.data[:chunk_len]
        else:
            self.data = self.data[chunk_len:]

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

    def __len__(self) -> int:
        return len(self.data) - self.block_size
