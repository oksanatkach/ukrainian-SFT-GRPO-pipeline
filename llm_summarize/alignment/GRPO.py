from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from trl import GRPOTrainer
from peft import PeftModel, get_peft_model
from llm_summarize.alignment.reward_functions import ToxicityClassifiers
from config.config import MainConfig
from hydra.utils import instantiate
from llm_summarize.alignment.custom_inference_callback import InferenceCallback
from llm_summarize.utils import extract_text_from_html


def run_GRPO(base_model: AutoModelForCausalLM,
             tokenizer: AutoTokenizer,
             best_checkpoint_path: str,
             dataset: Dataset,
             config: MainConfig) -> str:

    lora_config = instantiate(config.grpo_lora)
    train_config = instantiate(config.grpo_train)

    tokenizer.chat_template = "{{ messages[0]['content'] }}"

    sft_model = PeftModel.from_pretrained(base_model, best_checkpoint_path)
    align_model = get_peft_model(sft_model, lora_config)

    classifiers = ToxicityClassifiers(cfg=config.reward_classifier)

    dirty_urls = [
        "https://sigwait.gitlab.io/les_podervyansky--plays/ch07.html",
        "https://sigwait.gitlab.io/les_podervyansky--plays/ch08.html",
        "https://sigwait.gitlab.io/les_podervyansky--plays/ch14.html",
        "https://sigwait.gitlab.io/les_podervyansky--plays/ch22.html"
    ]
    test_prompts = [extract_text_from_html(url) for url in dirty_urls]
    test_prompts = [f"Підсумуй цей текст: {el}" for el in test_prompts]

    inference_callback = InferenceCallback(test_prompts, tokenizer, every_n_steps=100)

    trainer = GRPOTrainer(
        model=align_model,
        reward_funcs=[classifiers.get_rewards_classifier_1, classifiers.get_rewards_classifier_2],
        args=train_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[inference_callback]
    )

    trainer.train()
    final_model_path = f"{config.grpo_train.output_dir}/final_best_model"
    trainer.save_model(final_model_path)

    return final_model_path
