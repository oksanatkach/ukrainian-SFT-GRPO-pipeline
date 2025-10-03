from dataclasses import dataclass, field
import torch

@dataclass
class QuantizationConfig:
    _target_: str = "transformers.BitsAndBytesConfig"
    load_in_4bit: bool =True
    bnb_4bit_compute_dtype: torch.dtype = torch.bfloat16
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True
