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
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


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

    def __init__(self, *a, M: int = 8, tau: float = 2.0, outer_freq: int = 500, lambda_: float = 0.1, 
                 seed: int = 13, max_steps: int = 14000, report_to: str = 'none', **kw):
        laser_params = {'outer_freq', 'lambda_', 'seed', 'max_steps', 'report_to', 'M', 'tau'}
        trainer_kw = {k: v for k, v in kw.items() if k not in laser_params}
        
        super().__init__(*a, **trainer_kw)
        self.M = M
        self.tau = tau
        self.outer_freq = outer_freq
        self.lambda_ = lambda_
        self.seed = seed
        self.max_steps = max_steps
        self.report_to = report_to
        # Spectrum parameter living on the simplex via softmax
        self.spec = torch.nn.Parameter(torch.full((M,), 1.0 / M))
        # Contextual gate projection will be initialized after we know the hidden size
        self.gate_proj = None

    # ---------------------------------------------------------------------
    #  Core LASER functions
    # ---------------------------------------------------------------------

    def soft_rank(self, x: torch.Tensor) -> torch.Tensor:
        """NeuralSort: differentiable approximation of argsort/rank (Cuturi & Blondel 2020)."""
        x_flat = x.flatten()
        n = x_flat.numel()
        if n == 0:
            return torch.tensor([], device=x_flat.device, dtype=torch.float32)
        
        diff = x_flat.unsqueeze(-1) - x_flat.unsqueeze(0)  # pair-wise differences
        P = torch.softmax(-diff / self.tau, dim=-1)  # doubly-stochastic permutation matrix
        K = torch.arange(1, n + 1, device=x_flat.device, dtype=torch.float32)
        return P @ K

    def spectral_risk(self, losses: torch.Tensor) -> torch.Tensor:
        """Compute the spectral risk R_w(l) = Σ w_j q_j(l).

        In practice we use equally-spaced quantiles q_j of the (optionally gated)
        per-token loss distribution.  The spectrum w is learned (softmax of `spec`)."""

        losses_flat = losses.flatten()
        _ = self.soft_rank(losses_flat)  # ranks are not explicitly used but keep the op for autograd
        
        if losses_flat.numel() < self.M:
            mean_loss = losses_flat.mean()
            w = torch.softmax(self.spec.to(losses_flat.device), dim=-1)
            return mean_loss * w.sum()
        else:
            w = torch.softmax(self.spec.to(losses_flat.device), dim=-1)
            risk_sum = torch.tensor(0.0, device=losses_flat.device, dtype=losses_flat.dtype)
            for k in range(self.M):
                quantile_val = torch.quantile(losses_flat, 1 - (k + 1) / self.M)
                risk_sum = risk_sum + w[k] * quantile_val
            return risk_sum

    # ------------------------------------------------------------------
    #  Loss used by HF Trainer
    # ------------------------------------------------------------------

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # noqa: D401 – signature fixed by HF
        labels = inputs.pop("labels")
        out = model(**inputs, output_hidden_states=True)

        # Shift so that tokens < t predict token t
        logits = out.logits
        h = out.hidden_states[-1][:, :-1]
        logp = torch.nn.functional.log_softmax(logits, dim=-1)[:, :-1]
        labels_shift = labels[:, 1:]
        mask = labels_shift.eq(-100)

        # Gather log-probs of the true next token, but only for valid labels
        valid_labels = labels_shift.clone()
        valid_labels[mask] = 0  # Replace -100 with 0 for gathering
        lp = logp.gather(-1, valid_labels.unsqueeze(-1)).squeeze(-1)
        lp = lp.masked_fill(mask, 0.0)
        losses = (-lp)[~mask]  # token-level NLL

        if self.gate_proj is None:
            hidden_size = h.shape[-1]
            self.gate_proj = torch.nn.Linear(hidden_size, 1, bias=False).to(h.device)
            if hasattr(self.model, 'device'):
                self.gate_proj = self.gate_proj.to(self.model.device)
            else:
                model_device = next(self.model.parameters()).device
                self.gate_proj = self.gate_proj.to(model_device)

        # Contextual gate multiplies losses
        h_masked = h[~mask]
        if self.gate_proj is not None:
            h_masked = h_masked.to(next(self.gate_proj.parameters()).device)
        gates = 1 + 4 * torch.sigmoid(self.gate_proj(h_masked).squeeze(-1))
        
        gated_losses = gates * losses
        risk = self.spectral_risk(gated_losses) / self.alpha
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

def build_model_and_tokenizer(model_name: str, hf_token: str | None = None):
    """Load model & tokenizer, with simplified config for smoke test."""
    auth = hf_token if hf_token else None
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth)
    
    if model_name == "gpt2":
        model = AutoModelForCausalLM.from_pretrained(model_name, token=auth)
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=auth,
            quantization_config=bnb_config,
            device_map="auto",
        )
        
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        model = get_peft_model(model, lora_config)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    return model, tokenizer


def get_trainer(
    model_name: str,
    train_dataset: Dataset,
    val_dataset: Dataset,
    training_args: Dict,
    laser_hp: Dict,
) -> LASERTrainer:
    """Factory that wires together datasets, HF TrainingArguments, and LASER hyper-params."""

    model, tokenizer = build_model_and_tokenizer(model_name, os.getenv("HF_TOKEN"))

    args = TrainingArguments(**training_args)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
    )
    
    trainer = LASERTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=args,
        data_collator=data_collator,
        **laser_hp,
    )
    
    model_device = next(model.parameters()).device
    trainer.spec = trainer.spec.to(model_device)
    
    return trainer


def save_metrics(metrics: Dict, save_dir: Path, tag: str) -> None:
    """Persist metrics to .research/iteration7 and also pretty-print to stdout."""
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / f"{tag}.json"
    with file_path.open("w") as fp:
        json.dump(metrics, fp, indent=2)
    print(f"===== {tag} metrics =====")
    print(json.dumps(metrics, indent=2))
