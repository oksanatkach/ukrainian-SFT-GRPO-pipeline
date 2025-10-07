from dataclasses import dataclass
from config.SFT_train import SFTConfigBase
from config.GRPO_train import GRPOConfigBase


@dataclass
class AdamOptimizerConfig:
    """Adam optimizer parameters - can be mixed into any training config"""
    optim: str = "paged_adamw_8bit"
    learning_rate: float = 5e-5
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"

@dataclass
class AdamOptimizerConservativeConfig:
    optim: str = "paged_adamw_8bit"
    learning_rate: float = 1e-6
    max_grad_norm: float = 0.5
    weight_decay: float = 0.01
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1

@dataclass
class AdamSFTConfig(AdamOptimizerConfig, SFTConfigBase):
    """SFT config with Adam optimizer"""
    pass


@dataclass
class AdamGRPOConfig(AdamOptimizerConfig, GRPOConfigBase):
    """GRPO config with Adam optimizer"""
    pass
