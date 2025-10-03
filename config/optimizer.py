from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AdamConfig:
    optim: Optional[str] = field(
        default="paged_adamw_8bit",
        metadata={"help": "The optimizer to use."},
    )
    learning_rate: Optional[float] = field(default=2e-4)
    max_grad_norm: Optional[float] = field(default=1.0)
    weight_decay: Optional[int] = field(default=0.001)

    lr_scheduler_type: str = field(
        default="constant",
        metadata={"help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"},
    )
