from dataclasses import dataclass
from typing import Optional

@dataclass
class SFTEarlyStoppingConfig:
    early_stopping_patience: int = 5
    early_stopping_threshold: float = 0.01

@dataclass
class GRPOEarlyStoppingConfig:
    inference_callback_freq: int = 100
    early_stopping_patience: Optional[int] = 5
    max_kl_divergence: float = 0.5
