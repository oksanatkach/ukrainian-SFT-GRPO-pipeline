from trl import SFTTrainer
from typing import Optional, Union
from torch.utils.data import DataLoader, Dataset
import time
import math
import torch
import numpy as np
from transformers.debug_utils import DebugOption
from transformers.integrations.tpu import tpu_spmd_dataloader
from transformers.trainer_utils import speed_metrics
from transformers.utils import is_torch_xla_available, XLA_FSDPV2_MIN_VERSION

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    from torch_xla import __version__ as XLA_VERSION

    IS_XLA_FSDPV2_POST_2_2 = version.parse(XLA_VERSION) >= version.parse(XLA_FSDPV2_MIN_VERSION)
else:
    IS_XLA_FSDPV2_POST_2_2 = False

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Subset
import evaluate
import random


rouge = evaluate.load('rouge')
bleu = evaluate.load("bleu")
bertscore = evaluate.load("bertscore")


class CustomSFTTrainer(SFTTrainer):
    def __init__(self, *args, custom_metrics_args=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.custom_metrics_args = custom_metrics_args

    def get_prompts_tensor(self, batch):
        return {
            "prompt_ids": pad_sequence(
                [torch.tensor(el['prompt_ids']) for el in batch],
                batch_first=True,
                padding_value=0,
                padding_side="left"
            ),
            "labels": [el['summary'] for el in batch]
        }

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dataset, dict[str, Dataset]]] = None,
        ignore_keys: Optional[list[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        # handle multiple eval datasets
        override = eval_dataset is not None
        eval_dataset = eval_dataset if override else self.eval_dataset
        if isinstance(eval_dataset, dict):
            metrics = {}
            for eval_dataset_name, _eval_dataset in eval_dataset.items():
                dataset_metrics = self.evaluate(
                    eval_dataset=_eval_dataset if override else eval_dataset_name,
                    ignore_keys=ignore_keys,
                    metric_key_prefix=f"{metric_key_prefix}_{eval_dataset_name}",
                )
                metrics.update(dataset_metrics)
            return metrics

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        if self.is_fsdp_xla_v2_enabled:
            eval_dataloader = tpu_spmd_dataloader(eval_dataloader)

        start_time = time.time()

        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if self.compute_metrics is None else None,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )



        #############################################
        indices = random.sample(range(len(self.eval_dataset)), k=self.custom_metrics_args.custom_eval_sample_size)
        val_subset = Subset(self.eval_dataset, indices)
        val_subset_loader = DataLoader(val_subset,
                                       batch_size=self.args.eval_batch_size,
                                       shuffle=False,
                                       collate_fn=self.get_prompts_tensor)

        all_preds, all_labels = [], []

        for batch in val_subset_loader:
            prompts_tensor = batch['prompt_ids'].to(self.model.device)
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=prompts_tensor,
                    max_new_tokens=self.custom_metrics_args.max_new_tokens
                )
            pred_ids = generated_ids[:, prompts_tensor.shape[1]:]
            preds = self.processing_class.batch_decode(pred_ids, skip_special_tokens=True)

            all_preds.extend(preds)
            all_labels.extend(batch['labels'])

        if any([self.custom_metrics_args.rouge_l,
                self.custom_metrics_args.rouge_1,
                self.custom_metrics_args.rouge_2]):
            rouge_scores = rouge.compute(predictions=all_preds, references=all_labels)
            if self.custom_metrics_args.rouge_l:
                output.metrics.update({
                    "eval_rouge_l": rouge_scores["rougeL"].item()
                })
            if self.custom_metrics_args.rouge_1:
                output.metrics.update({
                    "eval_rouge_1": rouge_scores["rouge1"].item()
                })
            if self.custom_metrics_args.rouge_2:
                output.metrics.update({
                    "eval_rouge_2": rouge_scores["rouge2"].item()
                })
            if self.custom_metrics_args.rouge_lsum:
                output.metrics.update({
                    "eval_rouge_2": rouge_scores["rougeLsum"].item()
                })
        if self.custom_metrics_args.bleu:
            bleu_scores = bleu.compute(predictions=all_preds, references=[[label] for label in all_labels])
            output.metrics.update({
                "eval_bleu": bleu_scores["bleu"]
            })
        if self.custom_metrics_args.bert_score_f1:
            bert_scores = bertscore.compute(predictions=all_preds, references=all_labels, lang="uk")
            bert_score_f1_mean = np.mean(bert_scores['f1']).item()
            output.metrics.update({
                "eval_bert_score_f1": bert_score_f1_mean
            })

        #############################################

        total_batch_size = self.args.eval_batch_size * self.args.world_size
        if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
        if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
            start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
        output.metrics.update(
            speed_metrics(
                metric_key_prefix,
                start_time,
                num_samples=output.num_samples,
                num_steps=math.ceil(output.num_samples / total_batch_size),
            )
        )

        self.log(output.metrics)

        if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
            # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
            xm.master_print(met.metrics_report())

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)

        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
