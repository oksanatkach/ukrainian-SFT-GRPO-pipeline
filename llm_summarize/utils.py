import transformers
import torch
import vllm
import random
import numpy as np

def set_seed(seed: int) -> None:
    transformers.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
