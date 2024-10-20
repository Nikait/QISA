from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class Train:
    bsz: int = 16
    lr: float = 3e-3
    n_epoch: int = 30

    train_sz: float = 0.8


@dataclass
class Data:
    # path to save dataset
    data_path: str = "dataset/train"
    # path to weights
    


@dataclass
class Config:
    train: Train = field(default_factory=Train)
    data: Data = field(default_factory=Data)


cs = ConfigStore.instance()
cs.store(name="config", node=Config)
