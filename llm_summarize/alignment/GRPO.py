from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import GRPOTrainer
from reward_functions import ToxicityClassifiers
from config.config import MainConfig
from hydra.utils import instantiate


def run_GRPO(best_model_path: str, dataset: Dataset, config: MainConfig) -> str:
    model = AutoModelForCausalLM.from_pretrained(best_model_path)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    lora_config = instantiate(config.lora)
    training_args = instantiate(config.grpo_train)
    classifiers = ToxicityClassifiers()

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[classifiers.get_rewards_classifier_1, classifiers.get_rewards_classifier_2],
        args=training_args,
        train_dataset=dataset,
        peft_config=lora_config,
    )

    trainer.train()
    final_model_path = f"{config.grpo_train.output_dir}/final_best_model"
    trainer.save_model(final_model_path)

    # todo: sanity check?

    return final_model_path
