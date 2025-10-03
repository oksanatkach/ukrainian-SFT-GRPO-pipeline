from dataclasses import dataclass, field
from typing import List, Optional, Callable, Union, Dict

@dataclass
class GRPOConfig:
    _target_: str = "trl.GRPOConfig"
    output_dir: str = field(default="./outputs/GRPO")
    # use vllm to generate predictions from the model that's being aligned
    use_vllm: bool = field(default=True)
    # keep the vllm model on the same GPU
    vllm_mode: str = field(default="colocate")
    bf16: bool = field(default=True)
    gradient_checkpointing: bool = field(default=True)
