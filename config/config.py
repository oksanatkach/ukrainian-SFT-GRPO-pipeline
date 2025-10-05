from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from hydra.core.config_store import ConfigStore
from config.lora import LoRAConfig, LoRASmall
from config.quantization import QuantizationConfig
from config.SFT_train import SFTConfig, SFTConfigBase
from omegaconf import MISSING
from config.optimizer import AdamOptimizerConfig
from config.run import RunConfig
from config.reward_classifier import RewardClassifierConfig
from config.grpo_train import GRPOConfig
from config.model import ModelConfig
from config.dataset import DatasetConfig
from config.early_stopping import EarlyStoppingConfig
from config.custom_metrics import (CustomMetricsConfig,
                                   CustomMetricsROUGEConfig,
                                   CustomMetricsBLEUConfig,
                                   CustomMetricsFullConfig)


@dataclass
class MainConfig:
    # Nested groups
    model: ModelConfig = field(default_factory=ModelConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    quantization: QuantizationConfig = field(default_factory=QuantizationConfig)
    sft_train: SFTConfigBase = MISSING
    grpo_train: GRPOConfig = field(default_factory=GRPOConfig)
    run: RunConfig = field(default_factory=RunConfig)
    reward_classifier: RewardClassifierConfig = field(default_factory=RewardClassifierConfig)
    early_stopping: EarlyStoppingConfig = field(default_factory=EarlyStoppingConfig)
    custom_metrics: CustomMetricsConfig = field(default_factory=CustomMetricsConfig)

    wandb_project_name: str = "gemma_FST"
    wandb_run_name: str = f"run_{datetime.now():%Y%m%d_%H%M%S}"
    seed: int = 42
    cpu_workers: int = 1
    best_fst_model: Optional[str] = "./outputs/SFT"
    best_grpo_model: Optional[str] = None


def register_configs():
    """Register all configs with Hydra's ConfigStore"""
    cs = ConfigStore.instance()

    # Register main config
    cs.store(name="config", node=MainConfig)

    # Register model variants
    cs.store(group="lora", name="default", node=LoRAConfig)
    cs.store(group="lora", name="small", node=LoRASmall)

    cs.store(group="quantization", name="default", node=QuantizationConfig)

    cs.store(group="sft_train", name="default", node=SFTConfigBase)
    cs.store(group="sft_train", name="adam", node=AdamOptimizerConfig)

    cs.store(group="grpo_train", name="default", node=GRPOConfig)

    cs.store(group="dataset", name="default", node=DatasetConfig)

    cs.store(group="custom_metrics", name="default", node=CustomMetricsConfig)
    cs.store(group="custom_metrics", name="ROUGE", node=CustomMetricsROUGEConfig)
    cs.store(group="custom_metrics", name="BLEU", node=CustomMetricsBLEUConfig)
    cs.store(group="custom_metrics", name="full", node=CustomMetricsFullConfig)

    cs.store(group="early_stopping", name="default", node=EarlyStoppingConfig)
    cs.store(group="run", name="default", node=RunConfig)
    cs.store(group="reward_classifier", name="default", node=RewardClassifierConfig)
    cs.store(group="model", name="default", node=ModelConfig)
