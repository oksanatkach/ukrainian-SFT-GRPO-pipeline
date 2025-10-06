from dataclasses import dataclass

@dataclass
class RewardClassifierConfig:
    task: str = "classify"
    enforce_eager: bool = True
    max_input_len: int = 512
