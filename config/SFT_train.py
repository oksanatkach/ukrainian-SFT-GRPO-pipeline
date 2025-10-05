from dataclasses import dataclass, field
from typing import List, Optional, Dict
from trl import SFTConfig as TRLSFTConfig
from omegaconf import MISSING


@dataclass
class SFTConfigBase:
    _target_: str = "trl.SFTConfig"

    optim: str = MISSING
    learning_rate: float = MISSING
    weight_decay: float = MISSING
    max_grad_norm: float = MISSING
    lr_scheduler_type: str = MISSING

    per_device_train_batch_size: Optional[int] = 8
    per_device_eval_batch_size: Optional[int] = 8
    gradient_accumulation_steps: Optional[int] = 8
    eval_accumulation_steps: Optional[int] = 4
    max_seq_length: Optional[int] = None  # max_seq_length is declared in dataset config and handled in dataset_prep
    fp16: Optional[bool] = False # Enables fp16 training
    bf16: Optional[bool] = True # Enables bf16 training
    packing: Optional[bool] = False # Use packing dataset_prep creating
    gradient_checkpointing: Optional[bool] = True # Enables gradient checkpointing
    gradient_checkpointing_kwargs: Optional[Dict] = field(default_factory=lambda: {"use_reentrant": False})
    max_steps: int = 2500 # How many optimizer update steps to take
    warmup_ratio: float = 0.03 #Fraction of steps to do a warmup for
    save_steps: int = 500 # Save checkpoint every X updates steps
    save_total_limit: int = 3  # Keep only best 3 checkpoints to save space
    logging_steps: int = 100
    eval_steps: int = 500
    eval_strategy: str  = "steps"
    logging_strategy: str  = "steps"
    report_to: List[str] = "wandb"
    logging_first_step: bool = True

    # Model selection parameters:
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "rouge_l"
    greater_is_better: bool = True
    save_strategy: str  = "steps"

    dataset_kwargs: Optional[Dict] = field(default_factory=lambda: {
                "add_special_tokens": False,  # We template with special tokens
                "append_concat_token": False,  # No need to add additional separator token
                'assistant_only_loss': True
            })
    dataloader_num_workers: int = 2
    dataloader_pin_memory: bool = True
    dataset_text_field: Optional[str] = None

    output_dir: str = "./outputs/SFT"


@dataclass
class SFTConfig(SFTConfigBase):
    """Default SFT config with actual default optimizer values"""
    defaults: List[str] = field(default_factory=lambda: ["/sft_train/optimizer@_here_: adam"])
