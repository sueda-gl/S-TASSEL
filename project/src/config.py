from dataclasses import dataclass
import numpy as np

@dataclass
class Config:
    K: int      = 4          # tiers
    lambda_: float = 0.20    # solidarity slice λ
    beta:   float = 400.0    # Gini penalty weight
    zeta:   float = 50.0     # inventory penalty weight
    gini_target: float = 0.25
    token_expiry: int = 3    # epochs
    step_size:  float = 0.10 # η in mirrored-descent
    unit_stock: int = 30     # items per tier per epoch
    reserve_floor: bool = True
    seed: int = 42

cfg = Config()
rng = np.random.default_rng(cfg.seed)
