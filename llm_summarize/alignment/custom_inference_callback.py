from transformers import TrainerCallback
import torch


class InferenceCallback(TrainerCallback):
    def __init__(self, test_prompts, tokenizer, every_n_steps=50):
        self.test_prompts = test_prompts
        self.tokenizer = tokenizer
        self.every_n_steps = every_n_steps

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every_n_steps == 0:
            print(f"\n{'=' * 50}")
            print(f"Inference Check - Step {state.global_step}")
            print(f"{'=' * 50}\n")

            model.eval()
            for prompt in self.test_prompts[:2]:  # Test 2 prompts
                inputs = self.tokenizer(prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=128)
                print(f"Prompt: {prompt[:80]}...")
                print(f"Output: {self.tokenizer.decode(outputs[0], skip_special_tokens=True)}")
                print("-" * 50)
            model.train()
