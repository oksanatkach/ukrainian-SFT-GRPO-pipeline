from dataclasses import dataclass
from typing import Dict, Optional
from omegaconf import MISSING

@dataclass
class GRPOConfigBase:
    _target_: str = "trl.GRPOConfig"

    optim: str = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    max_grad_norm: float = MISSING
    lr_scheduler_type: str = MISSING

    output_dir: str = "./outputs/GRPO"
    bf16: bool = True
    gradient_checkpointing: bool = True
    # gradient_checkpointing_kwargs: Optional[Dict] = {"use_reentrant": True}

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    per_device_eval_batch_size: int = 1

    # generation params
    num_generations: int = 2
    temperature: float = 0.9
    max_completion_length: int = 256
    max_prompt_length: int = 512

    model_init_kwargs: Optional[Dict] = None

    # vllm + LoRA are too buggy in GRPO
    # use_vllm: bool = True
    # vllm_mode: str = "colocate"
