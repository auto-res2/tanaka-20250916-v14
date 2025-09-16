from typing import Tuple

from datasets import load_dataset
from transformers import AutoTokenizer


def load_and_tokenise(dataset_name: str, tokenizer_name: str, split: str = "train"):
    """Load a HF dataset and tokenize it for causal-LM training."""
    raw = load_dataset(dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def _tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=1024)

    tokenised = raw.map(_tok, batched=True, remove_columns=raw.column_names)
    tokenised.set_format(type="torch")
    return tokenised
