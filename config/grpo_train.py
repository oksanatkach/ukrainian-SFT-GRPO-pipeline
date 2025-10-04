from dataclasses import dataclass

@dataclass
class GRPOConfig:
    _target_: str = "trl.GRPOConfig"
    output_dir: str = "./outputs/GRPO"
    # use vllm to generate predictions from the model that's being aligned
    use_vllm: bool = True
    # keep the vllm model on the same GPU
    vllm_mode: str = "colocate"
    bf16: bool = True
    gradient_checkpointing: bool = True
