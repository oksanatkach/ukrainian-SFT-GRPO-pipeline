from transformers import AutoModelForCausalLM, EarlyStoppingCallback
from llm_summarize.SFT.custom_sfttrainer import CustomSFTTrainer
import wandb
from hydra.utils import instantiate
from config.config import MainConfig
import logging
from datasets import Dataset
from omegaconf import OmegaConf
from llm_summarize.utils.peft_patches import apply_peft_patches

log = logging.getLogger(__name__)
apply_peft_patches()


def run_SFT(model: AutoModelForCausalLM, train_dataset: Dataset, eval_dataset: Dataset, config: MainConfig) -> str:
    lora_config_dict = OmegaConf.to_container(config.sft_lora, resolve=True)
    lora_config = instantiate(lora_config_dict)

    # Force convert target_modules to plain list
    if hasattr(lora_config, 'target_modules'):
        lora_config.target_modules = list(lora_config.target_modules)

    train_config = instantiate(OmegaConf.to_container(config.sft_train, resolve=True))

    # Initialize wandb
    wandb.init(
        project=config.wandb_project_name,
        name=config.wandb_run_name,
        config=train_config.to_dict()
    )

    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=config.early_stopping.early_stopping_patience,
        early_stopping_threshold=config.early_stopping.early_stopping_threshold,
    )
    model.get_tokenizer().pad_token_id = model.get_tokenizer().eos_token_id
    model.get_tokenizer().padding_side = "left"
    trainer = CustomSFTTrainer(
        model=model,
        args=train_config,
        custom_metrics_args=config.custom_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        formatting_func=None,  # Disable formatting
        callbacks=[early_stopping_callback],
    )
    trainer.train()

    # The best model is now automatically loaded in trainer.model
    # Saving to your final output location for the alignment stage
    final_model_path = f"{config.sft_train.output_dir}/final_best_model"
    trainer.save_model(final_model_path)

    print("SFT is done.")
    print(f"Best model saved to: {final_model_path}")

    if config.run.do_sft_sanity_check:
        import torch

        model = trainer.model
        tokenizer = trainer.tokenizer

        model.eval()

        input_strings = [row['text'] for row in eval_dataset.select(range(3))]
        test_inputs = [f"Підсумуй цей текст: {el}" for el in input_strings]
        input_tokens = tokenizer(test_inputs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        with torch.no_grad():
            output_tokens = trainer.model.generate(**input_tokens, max_new_tokens=132)

        for ind, output in enumerate(output_tokens):
            prompt_length = input_tokens['input_ids'][ind].shape[0]
            output_string = tokenizer.decode(output[prompt_length:])

            print(f"Input: {input_strings[ind][:100]}...")
            print(f"Summary: {output_string}")
            print("---")

    return final_model_path
