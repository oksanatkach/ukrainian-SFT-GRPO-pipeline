from dataclasses import dataclass
from typing import Optional

@dataclass
class EarlyStoppingConfig:
    early_stopping_patience: Optional[int] = 5
    early_stopping_threshold: Optional[float] = 0.01
