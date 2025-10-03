#!/usr/bin/env python3
"""
Main entry point for the LLM summarization pipeline.
Usage: python scripts/run_pipeline.py [hydra overrides]
"""

import hydra
from llm_summarize.runner import run_pipeline
from config.config import MainConfig, register_configs


# Register configs before Hydra decorator
register_configs()

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: MainConfig) -> None:
    """Run the full pipeline: data prep -> SFT -> alignment -> eval."""
    run_pipeline(cfg)

if __name__ == "__main__":
    main()
