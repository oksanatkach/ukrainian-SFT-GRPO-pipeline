from dataclasses import dataclass
from typing import Union, Any
import torch

@dataclass
class ModelConfig:
    model_name: str = "google/gemma-3-1b-pt"
    dtype: str = "bfloat16"
    attn_implementation: str =  "eager" # "sdpa", "flash_attention_2"
    device_map: Any = "auto"
