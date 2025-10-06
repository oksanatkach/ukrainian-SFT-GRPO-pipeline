from dataclasses import dataclass
from config.SFT_train import SFTConfigBase
from config.GRPO_train import GRPOConfigBase

# @dataclass
# class AdamOptimizerConfig(SFTConfigBase):
#     """Adam optimizer config - inherits all SFT fields and overrides optimizer params"""
#     optim: str = "paged_adamw_8bit"
#     learning_rate: float = 2e-4
#     max_grad_norm: float = 1.0
#     weight_decay: float = 0.001
#     lr_scheduler_type: str = "constant"
#     # All other fields inherited from SFTConfigBase keep their defaults



@dataclass
class AdamOptimizerConfig:
    """Adam optimizer parameters - can be mixed into any training config"""
    optim: str = "paged_adamw_8bit"
    learning_rate: float = 2e-4
    max_grad_norm: float = 1.0
    weight_decay: float = 0.001
    lr_scheduler_type: str = "constant"


@dataclass
class AdamSFTConfig(AdamOptimizerConfig, SFTConfigBase):
    """SFT config with Adam optimizer"""
    pass


@dataclass
class AdamGRPOConfig(AdamOptimizerConfig, GRPOConfigBase):
    """GRPO config with Adam optimizer"""
    pass
