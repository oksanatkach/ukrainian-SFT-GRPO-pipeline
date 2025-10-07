from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    dataset_name: str = "csebuetnlp/xlsum"
    max_seq_length: Optional[int] = 1536
    special_tokens_buffer: Optional[int] = 5
    alignment_subset_size: int = 1000
