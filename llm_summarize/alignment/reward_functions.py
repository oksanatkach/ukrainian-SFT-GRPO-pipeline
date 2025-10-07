from vllm import LLM
from typing import List
from config.reward_classifier import RewardClassifierConfig
from vllm import TokensPrompt
import torch

class ToxicityClassifiers:
    def __init__(self, cfg: RewardClassifierConfig):
        self.cfg = cfg

        self.toxicity_clf1 = self.init_model("ukr-detect/ukr-toxicity-classifier")
        self.tokenizer_clf1 = self.toxicity_clf1.get_tokenizer()

        self.toxicity_clf2 = self.init_model("textdetox/xlmr-large-toxicity-classifier-v2")
        self.tokenizer_clf2 = self.toxicity_clf2.get_tokenizer()

    def init_model(self, model_name):
        return LLM(
            model=model_name,
            task=self.cfg.task,
            enforce_eager=self.cfg.enforce_eager
        )

    def get_rewards(self, classifier: LLM, tokenizer, completions: List[str]) -> List[float]:
        input_ids = tokenizer(completions,
                              max_length=self.cfg.max_input_len,
                              truncation=True)["input_ids"]
        tokens_prompts = [TokensPrompt(prompt_token_ids=el) for el in input_ids]
        output = classifier.classify(tokens_prompts)
        output_probs = [el.outputs.probs for el in output]
        rewards: List[float] = [el[0] - el[1] for el in output_probs]
        return rewards

    def get_rewards_classifier_1(self, prompts: List[str], completions: List[str], **kwargs)-> List[float]:
        return self.get_rewards(self.toxicity_clf1, self.tokenizer_clf1, completions)


    def get_rewards_classifier_2(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        return self.get_rewards(self.toxicity_clf2, self.tokenizer_clf2, completions)
