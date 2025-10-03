from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RunConfig:
    do_sft: Optional[bool] = field(default=True)
    do_align: Optional[bool] = field(default=True)
