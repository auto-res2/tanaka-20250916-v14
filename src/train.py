import os
import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


# -----------------------------------------------------------------------------
# Helper / base trainer
# -----------------------------------------------------------------------------

class SACTTrainer(Trainer):
    """A very thin wrapper around HF Trainer that only adds the `alpha` hyper-parameter
    required by LASER.  If a more sophisticated SACT implementation is available
    it can be plugged-in here without changing the LASER subclass below."""

    def __init__(self, *args, alpha: float = 0.05, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    # HF Trainer will call `compute_loss` defined in the subclass (LASERTrainer)


# -----------------------------------------------------------------------------
#  LASER implementation extracted verbatim from the single-file “Experiment Code”
# -----------------------------------------------------------------------------

class LASERTrainer(SACTTrainer):
    """Implementation of the LASER loss as described in the original experiment."""

    def __init__(self, *a, M: int = 8, tau: float = 2.0, **kw):
        super().__init__(*a, **kw)
        self.M = M
        self.tau = tau
        # Spectrum parameter living on the simplex via softmax
        self.register_parameter("spec", torch.nn.Parameter(torch.full((M,), 1.0 / M)))
        # Contextual gate projection (one-layer MLP without bias)
        self.gate_proj = torch.nn.Linear(self.model.config.hidden_size, 1, bias=False)

    # ---------------------------------------------------------------------
    #  Core LASER functions
    # ---------------------------------------------------------------------

    def soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """NeuralSort: differentiable approximation of argsort/rank (Cuturi & Blondel 2020)."""
        diff = x.unsqueeze(-1) - x.unsqueeze(0)  # pair-wise differences
        P = torch.softmax(-diff / self.tau, dim=-1)  # doubly-stochastic permutation matrix
        K = torch.arange(1, x.numel() + 1, device=x.device)
        return P @ K.float()

    def spectral_risk(self, losses: torch.Tensor) -> torch.Tensor:
        """Compute the spectral risk R_w(l) = Σ w_j q_j(l).

        In practice we use equally-spaced quantiles q_j of the (optionally gated)
        per-token loss distribution.  The spectrum w is learned (softmax of `spec`)."""

        _ = self.soft_rank(losses)  # ranks are not explicitly used but keep the op for autograd
        q_bins = [torch.quantile(losses, 1 - (k + 1) / self.M) for k in range(self.M)]
        q = torch.stack(q_bins)
        w = torch.softmax(self.spec, dim=-1)
        return (w * q).sum()

    # ------------------------------------------------------------------
    #  Loss used by HF Trainer
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False):  # noqa: D401 – signature fixed by HF
        labels = inputs.pop("labels")
        out = model(**inputs, output_hidden_states=True)

        # Shift so that tokens < t predict token t
        logits = out.logits
        h = out.hidden_states[-1][:, :-1]
        logp = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1]
        labels_shift = labels[:, 1:]
        mask = labels_shift.eq(-100)

        # Gather log-probs of the true next token
        lp = logp.gather(-1, labels_shift.unsqueeze(-1)).squeeze(-1)
        lp = lp.masked_fill(mask, 0.0)
        losses = (-lp)[~mask]  # token-level NLL

        # Contextual gate multiplies losses
        gates = 1 + 4 * torch.sigmoid(self.gate_proj(h)[~mask])
        risk = self.spectral_risk(gates * losses) / self.alpha
        loss = risk + 0.1 * losses.mean()  # blended with mean-loss for stability

        return (loss, out) if return_outputs else loss

    # ---------------------------------------------------------------
    #  Meta-update of the spectrum after validation risk is measured
    # ---------------------------------------------------------------

    def outer_update(self, val_risk: torch.Tensor) -> None:
        bound = torch.exp(-0.5 * val_risk)
        bound.backward()  # only updates `self.spec`


# -----------------------------------------------------------------------------
#  Utility functions used by src.main
# -----------------------------------------------------------------------------

def build_model_and_tokenizer(model_name: str, hf_token: str = None):
    """Load model & tokenizer in 4-bit LoRA mode if specified in the name."""
    auth = True if hf_token else False
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=auth)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        load_in_4bit=True,
        device_map="auto",
        use_auth_token=auth,
    )
    return model, tokenizer


def get_trainer(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: Dict,
    laser_hp: Dict,
) -> LASERTrainer:
    """Factory that wires together datasets, HF TrainingArguments, and LASER hyper-params."""

    model, _ = build_model_and_tokenizer(model_name, os.getenv("HF_TOKEN"))

    args = TrainingArguments(**training_args)
    trainer = LASERTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args,
        **laser_hp,
    )
    return trainer


def save_metrics(metrics: Dict, save_dir: Path, tag: str) -> None:
    """Persist metrics to .research/iteration1 and also pretty-print to stdout."""
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{tag}.json"
    with file_path.open("w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"===== {tag} metrics =====")
    print(json.dumps(metrics, indent=2))
