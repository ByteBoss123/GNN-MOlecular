"""
config.py — Central hyperparameter & path configuration.
"""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class DataConfig:
    # Tox21 has ~8k molecules, 12 binary toxicity tasks
    dataset_name: Literal["tox21", "esol", "freesolv", "lipophilicity"] = "tox21"
    task: Literal["toxicity", "solubility"] = "toxicity"
    # Which Tox21 column to use as binary label
    tox_target_col: str = "NR-AR"
    val_split: float = 0.1
    test_split: float = 0.1
    seed: int = 42
    # Max atoms (molecules with more are dropped)
    max_atoms: int = 100


@dataclass
class FeatureConfig:
    # Atom features
    atom_feature_dim: int = 74          # see featurizer.py for breakdown
    # Bond (edge) features
    bond_feature_dim: int = 12


@dataclass
class ModelConfig:
    arch: Literal["gcn", "gat", "mpnn"] = "gcn"
    hidden_dim: int = 128
    num_layers: int = 3
    dropout: float = 0.3
    heads: int = 4                      # GAT attention heads
    # Readout pooling
    pooling: Literal["mean", "sum", "max", "attention"] = "mean"
    # Task head
    mlp_hidden: int = 64


@dataclass
class TrainConfig:
    epochs: int = 100
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # Learning rate scheduler
    lr_patience: int = 10
    lr_factor: float = 0.5
    # Early stopping
    early_stop_patience: int = 20
    # Class imbalance — used for toxicity BCE loss
    pos_weight: float = 3.0


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    device: str = "cuda"          # "cuda" | "mps" | "cpu"
    output_dir: str = "outputs"
    log_every: int = 5


# ---------------------------------------------------------------------------
# Quick presets
# ---------------------------------------------------------------------------

def toxicity_config() -> Config:
    cfg = Config()
    cfg.data.dataset_name = "tox21"
    cfg.data.task = "toxicity"
    return cfg


def solubility_config() -> Config:
    cfg = Config()
    cfg.data.dataset_name = "esol"
    cfg.data.task = "solubility"
    cfg.train.pos_weight = 1.0   # regression, no class imbalance
    return cfg
