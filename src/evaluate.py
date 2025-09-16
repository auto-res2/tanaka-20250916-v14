import json
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

from .train import LASERTrainer, build_model_and_tokenizer, save_metrics


@torch.no_grad()
def compute_laser_ppl(trainer: LASERTrainer, texts: List[str], tokenizer: AutoTokenizer) -> float:
    """Compute LASERPPL (exp of spectral risk) over a list of generated texts."""
    device = next(trainer.model.parameters()).device

    losses = []
    for txt in texts:
        enc = tokenizer(txt, return_tensors="pt").to(device)
        labels = enc.input_ids.clone()
        labels[:, :-1] = -100  # predict next token only
        enc["labels"] = labels
        loss = trainer.compute_loss(trainer.model, enc)
        losses.append(loss.item())
    return float(torch.exp(torch.tensor(losses).mean()).item())


def evaluate(
    model_name: str,
    prompts: List[str],
    trainer_ckpt: Path,
    output_dir: Path,
):
    """Load a trained checkpoint and run the full evaluation pipeline.

    This is a greatly simplified stand-in for the much more extensive evaluation
    described in the paper, but it demonstrates the data-flow & results dumping
    that the orchestrator expects."""

    model, tokenizer = build_model_and_tokenizer(model_name)
    trainer = LASERTrainer(model=model, args=None)  # “args” unused for inference
    trainer.model.load_state_dict(torch.load(trainer_ckpt, map_location="cpu"))

    laser_ppl = compute_laser_ppl(trainer, prompts, tokenizer)
    metrics: Dict = {
        "LASERPPL": laser_ppl,
        "num_prompts": len(prompts),
    }

    save_metrics(metrics, output_dir, tag="eval")

    return metrics
