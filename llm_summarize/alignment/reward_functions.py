from vllm import LLM, TokensPrompt
from typing import List
from config.reward_classifier import RewardClassifierConfig

class ToxicityClassifiers:
    def __init__(self, cfg: RewardClassifierConfig):
        self.cfg = cfg

        self.toxicity_clf1 = self.init_model("ukr-detect/ukr-toxicity-classifier")
        self.toxicity_clf2 = self.init_model("textdetox/xlmr-large-toxicity-classifier-v2")

    def init_model(self, model_name):
        return LLM(
            model=model_name,
            task=self.cfg.task,
            enforce_eager=self.cfg.enforce_eager,
            max_model_len=self.cfg.max_model_len
        )

    def get_rewards(self, classifier: LLM, completions: List[str]) -> List[float]:
        # todo: classify or reward?
        input_ids = classifier.get_tokenizer()(completions, max_length=self.cfg.max_model_len, truncation=True)
        output = classifier.classify(input_ids)
        output = [el.outputs.probs for el in output]
        return [el[0] - el[1] for el in output]

    def get_rewards_classifier_1(self, prompts: List[str], completions: List[str], **kwargs)-> List[float]:
        return self.get_rewards(self.toxicity_clf1, completions)

    def get_rewards_classifier_2(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        return self.get_rewards(self.toxicity_clf2, completions)
