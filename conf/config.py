import torch
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class Train:
    bsz: int = 1024
    lr: float = 3e-3
    n_epoch: int = 30

    block_size: int = 16 # context length
    n_embd: int = 16
    # hidden_dim = n_embed // num_heads --> 4
    n_head: int = 1
    n_layer: int = 6
    dropout: float = 0.2

    vocab_size: int = 65


@dataclass
class Data:
    # path to save dataset
    data_path: str = "data/input.txt"
    # path to weights
    checkpoints: str = "weights/"
    # part for train
    train: float = 0.8

    load_checkpoint: bool = False
    load_checkpoint_path: str = "weights/checkpoint.pt"



@dataclass
class Config:
    train: Train = field(default_factory=Train)
    data: Data = field(default_factory=Data)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
