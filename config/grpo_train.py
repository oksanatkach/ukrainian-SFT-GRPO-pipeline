from dataclasses import dataclass, field
from typing import Dict

@dataclass
class GRPOConfig:
    _target_: str = "trl.GRPOConfig"
    output_dir: str = "./outputs/GRPO"
    bf16: bool = True
    gradient_checkpointing: bool = True
    # gradient_checkpointing_kwargs : Dict ={"use_reentrant": True}

    # todo: inherit from optim config
    optim = "adamw_torch_fused"
    learning_rate: float = 1e-6

    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    per_device_eval_batch_size: int = 1

    # generation params
    num_generations: int = 4
    temperature: float = 0.9
    max_completion_length: int = 256
    max_prompt_length: int = 512

    # todo: get from model config
    model_init_kwargs: Dict = field(default_factory=lambda: {"dtype": "bfloat16",
                                                             'attn_implementation': 'eager'})

    # vllm + LoRA are too buggy in GRPO
    # use_vllm: bool = True
    # vllm_mode: str = "colocate"
