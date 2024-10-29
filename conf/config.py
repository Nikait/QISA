import torch
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class Train:
    bsz: int = 64
    lr: float = 3e-3
    n_epoch: int = 30

    train_sz: float = 0.8

    block_size = 256 # what is the maximum context length for predictions?
    max_iters = 5000
    eval_interval = 500
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_head = 6
    n_layer = 6
    dropout = 0.2

    vocab_size = 65
    block_size = 256
    n_embed = 24
    num_heads = 6
    n_layers = 6


@dataclass
class Data:
    # path to save dataset
    data_path: str = "data/IMDB Dataset.csv"
    # path to weights
    checkpoints: str = "/weights/"
    # part for train
    train: float = 0.8
    block_size: int = 256
    
    load_checkpoint: bool = False
    load_checkpoint_path: str = "/weights/chepoint.pt"



@dataclass
class Config:
    train: Train = field(default_factory=Train)
    data: Data = field(default_factory=Data)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
