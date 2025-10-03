from dataclasses import dataclass


@dataclass
class CustomMetricsConfig:
    custom_eval_sample_size: int = 500
    rouge_1: bool = True
    rouge_2: bool = True
    rouge_l: bool = True
    rouge_lsum: bool = True
    bleu: bool = True
    bert_score_f1: bool = False

class CustomMetricsROUGEConfig:
    custom_eval_sample_size: int = 500
    rouge_1: bool = True
    rouge_2: bool = True
    rouge_l: bool = True
    rouge_lsum: bool = False
    bleu: bool = False
    bert_score_f1: bool = False

class CustomMetricsBLEUConfig:
    custom_eval_sample_size: int = 500
    rouge_1: bool = False
    rouge_2: bool = False
    rouge_l: bool = False
    rouge_lsum: bool = False
    bleu: bool = True
    bert_score_f1: bool = False

class CustomMetricsFullConfig:
    custom_eval_sample_size: int = 500
    rouge_1: bool = True
    rouge_2: bool = True
    rouge_l: bool = True
    rouge_lsum: bool = True
    bleu: bool = True
    bert_score_f1: bool = True
