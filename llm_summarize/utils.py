import transformers
import torch
import vllm
import random
import numpy as np

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    transformers.set_seed(seed)
    vllm.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
