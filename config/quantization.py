from transformers import BitsAndBytesConfig
from dataclasses import dataclass
import torch


@dataclass
class QuantizationConfig:
    _target_: str = "config.quantization.create_quantization_config"  # Point to factory function
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True

def create_quantization_config(bnb_4bit_compute_dtype: str, **kwargs) -> BitsAndBytesConfig:
    """Factory function to create BitsAndBytesConfig with proper dtype conversion"""
    return BitsAndBytesConfig(
        bnb_4bit_compute_dtype=getattr(torch, bnb_4bit_compute_dtype),
        **kwargs  # Pass all other arguments through
    )
