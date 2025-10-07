from dataclasses import dataclass, field
from typing import Optional, Dict
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
    gradient_accumulation_steps: int = 16
    per_device_eval_batch_size: int = 1

    # generation params
    num_generations: int = 4
    temperature: float = 0.8
    max_completion_length: int = 256
    max_prompt_length: int = 512
    num_train_epochs: int = 1
    logging_steps: int = 10
    save_strategy: str = "steps"
    save_steps: int = 100
    save_total_limit: int = 2

    evaluation_strategy: Optional[str] = "no"
    report_to: str = "wandb"

    # can later replace this with params from config.model if passing model path instead of object
    # model_init_kwargs: Optional[Dict] = None

    # vllm + LoRA are too buggy in GRPO :(
    # maybe would work better with vllm_mode = "server" but that needs a separate setup
    # use_vllm: bool = True
    # vllm_mode: str = "colocate"
