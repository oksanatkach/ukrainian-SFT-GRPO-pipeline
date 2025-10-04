from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RunConfig:
    do_sft: Optional[bool] = True
    do_align: Optional[bool] = True
