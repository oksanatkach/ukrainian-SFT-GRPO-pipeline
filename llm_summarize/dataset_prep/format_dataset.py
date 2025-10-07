from datasets import load_dataset, Dataset
from transformers import AutoTokenizer
from typing import Tuple
import logging


log = logging.getLogger(__name__)

def formatting_func(row, tokenizer, max_length=2048, special_tokens_buffer=5):
    prompt_text = (
        "Коротко підсумуй цей текст:\n"
        f"{row['text']}\n\n"
    )
    completion_text = f"Підсумок тексту: {row['summary']}"

    prompt_ids = tokenizer(text=prompt_text, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(text=completion_text, add_special_tokens=False)["input_ids"]

    # Truncate prompt if needed
    if len(prompt_ids) + len(completion_ids) > (max_length - special_tokens_buffer):
        max_prompt_length = max_length - len(completion_ids) - special_tokens_buffer
        prompt_ids = prompt_ids[:max_prompt_length]

    # Concatenate
    input_ids = prompt_ids + completion_ids
    completion_mask = [0] * len(prompt_ids) + [1] * len(completion_ids)

    # Return tokenized data (SFTTrainer will detect "input_ids" and skip tokenization)
    return {
        "input_ids": input_ids,
        "completion_mask": completion_mask,
        "prompt_text": prompt_text,
        "prompt_ids": prompt_ids
    }

def format_ds(dataset: Dataset,
              tokenizer: AutoTokenizer,
              max_seq_length: int,
              special_tokens_buffer: int,
              cpu_workers: int = 1):
    formatted_dataset = dataset.map(
        formatting_func,
        fn_kwargs={"tokenizer": tokenizer,
                   "max_length": max_seq_length,
                   "special_tokens_buffer": special_tokens_buffer
                   },
        remove_columns=["title", "text", "url"],
        num_proc=cpu_workers
    )
    formatted_dataset.set_format(type="pt",
                                 columns=["input_ids", "completion_mask", "prompt_ids"],
                                 output_all_columns=True)
    return formatted_dataset

def format_ds_for_GRPO(dataset: Dataset, cpu_workers: int = 1):
    '''
    GRPO doesn't need the completion, so we simply map prompt_ids to input_ids.
    This avoids another tokenization run.
    '''
    return dataset.map(
        lambda row: {"prompt": row["prompt_text"]},
        remove_columns=["prompt_ids", "completion_mask", "input_ids"],
        num_proc=cpu_workers
    )

def get_dataset(dataset_name: str,
                tokenizer: AutoTokenizer,
                max_seq_length: int,
                special_tokens_buffer: int,
                cpu_workers: int) -> Tuple[Dataset, Dataset]:

    # a neater and more concise way to pass arguments
    format_ds_kwargs = locals().copy()
    format_ds_kwargs.pop("dataset_name")

    try:
        train_dataset = load_dataset(dataset_name, "ukrainian", split="train[:1%]")
        eval_dataset = load_dataset(dataset_name, "ukrainian", split="validation[:1%]")
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        raise  # critical, cannot continue

    return (
        format_ds(dataset=train_dataset, **format_ds_kwargs),
        format_ds(dataset=eval_dataset, **format_ds_kwargs),
    )
