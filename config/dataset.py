from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    dataset_name: str = "csebuetnlp/xlsum"
    max_token_length: Optional[int] = 1536
    special_tokens_buffer: Optional[int] = 5
