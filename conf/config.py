import torch
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class Train:
    bsz: int = 25
    lr: float = 3e-3
    n_epoch: int = 30

    block_size: int = 4
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    n_embd: int = 4
    n_head: int = 1
    n_layer: int = 6
    dropout: float = 0.2

    vocab_size: int = 65


@dataclass
class Data:
    # path to save dataset
    data_path: str = "QSA/h"
    # path to weights
    checkpoints: str = "/weights/"
    # part for train
    train: float = 0.8
    block_size: int = 6

    load_checkpoint: bool = False
    load_checkpoint_path: str = "/weights/checkpoint.pt"



@dataclass
class Config:
    train: Train = field(default_factory=Train)
    data: Data = field(default_factory=Data)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
