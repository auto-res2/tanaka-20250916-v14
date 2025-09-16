import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
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
        if isinstance(loss, tuple):
            loss = loss[0]
        losses.append(loss.item())
    return float(torch.exp(torch.tensor(losses).mean()).item())


def create_evaluation_plots(metrics: Dict, output_dir: Path) -> List[str]:
    """Create visualization plots for experimental results."""
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    plot_paths = []
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(['LASERPPL'], [metrics['LASERPPL']], color='skyblue')
    ax.set_ylabel('LASERPPL Score')
    ax.set_title('LASER Perplexity Evaluation Results')
    ax.grid(True, alpha=0.3)
    
    plot_path = images_dir / "laserppl_results.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_paths.append(str(plot_path))
    
    if len(metrics) > 2:
        fig, ax = plt.subplots(figsize=(10, 6))
        metric_names = [k for k in metrics.keys() if isinstance(metrics[k], (int, float))]
        metric_values = [metrics[k] for k in metric_names]
        
        ax.bar(metric_names, metric_values, color='lightcoral')
        ax.set_ylabel('Metric Values')
        ax.set_title('Experimental Metrics Summary')
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        
        plot_path = images_dir / "metrics_summary.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths.append(str(plot_path))
    
    return plot_paths


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
    trainer = LASERTrainer(model=model, args=None)  # "args" unused for inference
    trainer.model.load_state_dict(torch.load(trainer_ckpt, map_location="cpu"))

    laser_ppl = compute_laser_ppl(trainer, prompts, tokenizer)
    
    metrics: Dict = {
        "LASERPPL": laser_ppl,
        "num_prompts": len(prompts),
        "experiment_type": "LASER_evaluation",
        "model_name": model_name,
        "statistical_summary": {
            "mean_laserppl": float(laser_ppl),
            "evaluation_completed": True
        }
    }
    
    plot_paths = create_evaluation_plots(metrics, output_dir)
    metrics["plot_paths"] = plot_paths
    
    print("=" * 60)
    print("LASER EXPERIMENT EVALUATION RESULTS")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Number of prompts evaluated: {len(prompts)}")
    print(f"LASERPPL Score: {laser_ppl:.4f}")
    print(f"Evaluation plots saved to:")
    for plot_path in plot_paths:
        print(f"  - {plot_path}")
    print("=" * 60)

    save_metrics(metrics, output_dir, tag="eval")

    return metrics
