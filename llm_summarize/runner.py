from transformers import AutoTokenizer, AutoModelForCausalLM
from config.config import MainConfig
from hydra.utils import instantiate
import logging
import os
import wandb
from huggingface_hub import login


log = logging.getLogger(__name__)

def _init_environment():
    """Authenticate Hugging Face and W&B safely, with logging."""
    hf_key = os.environ.get("HF_KEY")
    wandb_key = os.environ.get("WANDB_API_KEY")

    # Hugging Face login
    if not hf_key:
        log.error("HF_KEY environment variable not set. Cannot authenticate Hugging Face.")
        raise ValueError("HF_KEY environment variable not set")
    else:
        try:
            login(hf_key)
            log.info("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            log.error(f"Failed to log in to Hugging Face: {e}")
            raise

    # Weights & Biases login
    if not wandb_key:
        log.error("WANDB_API_KEY environment variable not set. W&B logging will not work.")
        # Non-critical: we can choose to raise or just warn depending on your preference
        # Here we just warn and continue
        log.warning("Continuing without W&B logging")
    else:
        try:
            wandb.login(key=wandb_key)
            log.info("Successfully logged in to Weights & Biases")
        except Exception as e:
            log.error(f"Failed to log in to W&B: {e}")
            # W&B failure might not stop training, so we log but do not raise
            log.warning("Continuing without W&B logging")


# Automatically initialize on import
_init_environment()


def run_pipeline(config: MainConfig) -> None:
    from llm_summarize.utils.utils import set_seed
    set_seed(config.seed)

    # 1) Data preparation
    log.info(f"Loading tokenizer {config.model.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    log.info(f"Formatting dataset {config.dataset.dataset_name}")
    from llm_summarize.dataset_prep import format_dataset
    train_dataset_formatted, eval_dataset_formatted = format_dataset.get_dataset(
        tokenizer=tokenizer,
        dataset_name=config.dataset.dataset_name,
        max_seq_length=config.dataset.max_seq_length,
        special_tokens_buffer=config.dataset.special_tokens_buffer,
        cpu_workers=config.cpu_workers
    )

    # 2) Load base model
    quantization_config = instantiate(config.quantization)
    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=quantization_config,
        dtype = config.model.dtype,
        attn_implementation=config.model.attn_implementation,
        device_map=config.model.device_map
    )
    model.gradient_checkpointing_enable()

    # 3) SFT
    if config.run.do_sft:
        log.info(f"Initializing SFT on model {config.model.model_name}")

        if config.best_fst_model:
            log.warning("You specify SFT model for alignment but also call SFT, best SFT model path will be rewritten")

        from llm_summarize.SFT import run_SFT

        config.best_fst_model = run_SFT(model=model,
                                        train_dataset=train_dataset_formatted,
                                        eval_dataset=eval_dataset_formatted,
                                        config=config)
        log.info(f"Finished SFT, final model path:\t{config.best_fst_model}")

    # 3) Alignment
    if config.run.do_align:
        if not config.best_fst_model:
            log.error("ALignment is enabled but best_fst_model path is not in the config")
            raise ValueError("Specify which model to align!")

        log.info(f"Aligning model {config.best_fst_model}")

        from llm_summarize.alignment import run_GRPO
        from llm_summarize.dataset_prep.format_dataset import format_ds_for_GRPO

        grpo_dataset = format_ds_for_GRPO(dataset=train_dataset_formatted, cpu_workers=config.cpu_workers)

        # pass last checkpoint path or model object
        config.best_grpo_model = run_GRPO(base_model=model,
                                          tokenizer=tokenizer,
                                          best_checkpoint_path=config.best_fst_model,
                                          dataset=grpo_dataset,
                                          config=config)
        log.info(f"Finished alignment, final model path:\t{config.best_grpo_model}")
