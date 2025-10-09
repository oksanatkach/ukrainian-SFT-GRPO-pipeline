## Usage

Run this once to set up the environment:
```commandline
uv sync
```

To run the entire pipeline:
```commandline
uv run python scripts/run_pipeline.py
```

The project requires huggingface key and wandb key as environment variables.
You can place them in an `.env` file and point to it when running the script:
```commandline
printf 'HF_KEY="your-hf-key"\nWANDB_API_KEY="your-wandb-key"' >> .env
uv run --env-file=.env python scripts/run_pipeline.py
```

To disable SFT and start with GRPO with a known SFT model just override hydra config values:
```commandline
uv run python scripts/run_pipeline.py best_fst_model=outputs/SFT/final_best_model run.do_sft=False
```
