from transformers import AutoModelForCausalLM, EarlyStoppingCallback
from llm_summarize.SFT.custom_sfttrainer import CustomSFTTrainer
import wandb
from hydra.utils import instantiate
from config.config import MainConfig
import logging
from datasets import Dataset


log = logging.getLogger(__name__)

def run_SFT(train_dataset: Dataset, eval_dataset: Dataset, config: MainConfig) -> str:
    lora_config = instantiate(config.lora)
    quantization_config = instantiate(config.quantization)
    train_config = instantiate(config.sft_train)

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        quantization_config=quantization_config,
        dtype = config.model.dtype,
        attn_implementation=config.model.attn_implementation,
        device_map=config.model.device_map
    )
    model.gradient_checkpointing_enable()

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
    model = model.merge_and_unload()
    model.save_pretrained(final_model_path)

    print("SFT is done.")
    print(f"Best model saved to: {final_model_path}")

    if config.run.do_sft_sanity_check:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(config.model.model_name)

        input_strings = [row['text'] for row in eval_dataset.select(range(3))]
        test_inputs = [f"Підсумуй цей текст: {el}" for el in input_strings]
        input_tokens = tokenizer(test_inputs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        output_tokens = model.generate(**input_tokens, max_new_tokens=132)

        for ind, output in enumerate(output_tokens):
            prompt_length = input_tokens['input_ids'][ind].shape[0]
            output_string = tokenizer.decode(output[prompt_length:])

            print(f"Input: {input_strings[ind][:100]}...")
            print(f"Summary: {output_string}")
            print("---")

    return final_model_path
