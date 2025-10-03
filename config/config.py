from dataclasses import dataclass, field
from omegaconf import OmegaConf
import torch
from datetime import datetime
from hydra.core.config_store import ConfigStore
from config.lora import LoRAConfig, LoRASmall
from config.quantization import QuantizationConfig
from config.SFT_train import SFTConfig
from config.optimizer import AdamConfig
from config.run import RunConfig
from config.grpo_train import GRPOConfig
from config.dataset import DatasetConfig
from config.custom_metrics import (CustomMetricsConfig,
                                   CustomMetricsROUGEConfig,
                                   CustomMetricsBLEUConfig,
                                   CustomMetricsFullConfig)


@dataclass
class MainConfig:
    # Nested groups
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    sft_train: SFTConfig = field(default_factory=SFTConfig)
    grpo_train: GRPOConfig = field(default_factory=GRPOConfig)
    run: RunConfig = field(default_factory=RunConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    custom_metrics: CustomMetricsConfig = field(default_factory=CustomMetricsConfig)

    model_name: str = "google/gemma-3-1b-pt"
    wandb_project_name: str = "gemma_FST"
    wandb_run_name: str = f"run_{datetime.now():%Y%m%d_%H%M%S}"
    seed: int = 42
    cpu_workers: int = 1
    best_fst_model: str = "???"
    best_grpo_model: str = "???"


def register_configs():
    """Register all configs with Hydra's ConfigStore"""
    cs = ConfigStore.instance()

    OmegaConf.register_new_resolver(
        "torch.dtype",
        lambda dtype_name: getattr(torch, dtype_name)
    )

    # Register main config
    cs.store(name="config", node=MainConfig)

    # Register model variants
    cs.store(group="lora", name="default", node=LoRAConfig)
    cs.store(group="lora", name="small", node=LoRASmall)

    cs.store(group="quantization", name="default", node=QuantizationConfig)

    cs.store(group="sft_train", name="default", node=SFTConfig)
    cs.store(group="sft_train/optimizer", name="adam", node=AdamConfig)

    cs.store(group="grpo_train", name="default", node=GRPOConfig)

    cs.store(group="dataset", name="default", node=DatasetConfig)

    cs.store(group="custom_metrics", name="default", node=CustomMetricsConfig)
    cs.store(group="custom_metrics", name="ROUGE", node=CustomMetricsROUGEConfig)
    cs.store(group="custom_metrics", name="BLEU", node=CustomMetricsBLEUConfig)
    cs.store(group="custom_metrics", name="full", node=CustomMetricsFullConfig)
