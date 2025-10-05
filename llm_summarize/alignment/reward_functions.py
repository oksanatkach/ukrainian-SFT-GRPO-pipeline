from vllm import LLM
from typing import List

class ToxicityClassifiers:
    def __init__(self):
        self.toxicity_clf1 = LLM(
            model="ukr-detect/ukr-toxicity-classifier",
            task="classify",
            enforce_eager=True,
        )
        self.toxicity_clf2 = LLM(
            model="textdetox/xlmr-large-toxicity-classifier-v2",
            task="classify",
            enforce_eager=True,
        )

    @ staticmethod
    def get_rewards(classifier: LLM, completions: List[str]) -> List[float]:
        output = classifier.classify(completions)
        output = [el.outputs.probs for el in output]
        return [el[0] - el[1] for el in output]

    def get_rewards_classifier_1(self, prompts: List[str], completions: List[str], **kwargs)-> List[float]:
        return self.get_rewards(self.toxicity_clf1, completions)

    def get_rewards_classifier_2(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        return self.get_rewards(self.toxicity_clf2, completions)
