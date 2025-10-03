from dataclasses import dataclass, field
from typing import List, Optional, Callable, Union, Dict

@dataclass
class SFTConfig:
    _target_: str = "trl.SFTConfig"

    # will be inserted later from config/optimizer.py
    optim: str = "???"  # Required, comes from optimizer config
    learning_rate: float = "???"
    weight_decay: float = "???"
    max_grad_norm: float = "???"
    lr_scheduler_type: str = "???"

    early_stopping_patience: Optional[int] = field(default=5)
    early_stopping_threshold: Optional[float] = field(default=0.01)
    per_device_train_batch_size: Optional[int] = field(default=8)
    per_device_eval_batch_size: Optional[int] = field(default=8)
    gradient_accumulation_steps: Optional[int] = field(default=8)
    eval_accumulation_steps: Optional[int] = field(default=4)

    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    max_seq_length: Optional[int] = field(default=1024)
    max_length: Optional[int] = field(default=256)
    fp16: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables fp16 training."},
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables bf16 training."},
    )
    packing: Optional[bool] = field(
        default=False,
        metadata={"help": "Use packing dataset_prep creating."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True,
        metadata={"help": "Enables gradient checkpointing."},
    )
    gradient_checkpointing_kwargs: Optional[Dict] = field(default_factory=lambda: {"use_reentrant": False})
    use_flash_attention_2: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables Flash Attention 2."},
    )
    max_steps: int = field(default=2700, metadata={"help": "How many optimizer update steps to take"})
    warmup_ratio: float = field(default=0.03, metadata={"help": "Fraction of steps to do a warmup for"})
    save_steps: int = field(default=100, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: int = field(default=3)  # Keep only best 3 checkpoints to save space
    logging_steps: int = field(default=100, metadata={"help": "Log every X updates steps."})
    eval_steps: int = field(default=270)
    eval_strategy: str  = field(default="steps")
    logging_strategy: str  = field(default="steps")
    report_to: List[str] = field(default_factory=lambda: ["wandb"])
    logging_first_step: bool = field(default=True)

    # Model selection parameters:
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="rouge_l")
    greater_is_better: bool = field(default=True)
    save_strategy: str  = field(default="steps")

    dataset_kwargs: Optional[Dict] = field(default_factory=lambda: {
                "add_special_tokens": False,  # We template with special tokens
                "append_concat_token": False,  # No need to add additional separator token
                'assistant_only_loss': True
            })
    dataloader_num_workers: int = field(default=2)
    dataloader_pin_memory: bool = field(default=True)
    dataset_text_field: Union[Callable, None] = field(default=None)

    output_dir: str = field(
        default="./outputs/SFT",
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."},
    )
