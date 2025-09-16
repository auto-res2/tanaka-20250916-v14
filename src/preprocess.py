from typing import Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_tokenise(dataset_name: str, tokenizer_name: str, split: str = "train"):
    """Load a HF dataset and tokenize it for causal-LM training."""
    raw = load_dataset(dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    def _tok(batch):
        if "text" in batch:
            texts = batch["text"]
        elif "prompt" in batch and isinstance(batch["prompt"][0], dict):
            texts = [item["text"] for item in batch["prompt"]]
        else:
            raise ValueError(f"Cannot find text field in batch. Available keys: {list(batch.keys())}")
        
        tokenized = tokenizer(texts, truncation=True, max_length=128, padding=True)
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    tokenised = raw.map(_tok, batched=True, remove_columns=raw.column_names)
    tokenised.set_format(type="torch")
    return tokenised
