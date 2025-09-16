import argparse
import sys
from pathlib import Path
from typing import Dict

import yaml

from .preprocess import load_and_tokenise
from .train import get_trainer, save_metrics

# -----------------------------------------------------------------------------
#  Configuration helpers
# -----------------------------------------------------------------------------

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
RESEARCH_DIR = Path(".research/iteration1")


def load_config(config_path: Path) -> Dict:
    with config_path.open() as fp:
        return yaml.safe_load(fp)


# -----------------------------------------------------------------------------
#  Orchestration helpers
# -----------------------------------------------------------------------------


def run_experiment(conf: Dict, tag: str):
    # ------------------ data ------------------
    train_ds = load_and_tokenise(conf["dataset"], conf["model"], split="train")
    val_ds = load_and_tokenise(conf["dataset"], conf["model"], split="validation")

    # ----------------- trainer -----------------
    trainer = get_trainer(
        model_name=conf["model"],
        train_dataset=train_ds,
        val_dataset=val_ds,
        training_args=conf["training_args"],
        laser_hp=conf["laser_hyperparams"],
    )

    # ------------------ train ------------------
    trainer.train()
    metrics = trainer.evaluate()
    save_metrics(metrics, RESEARCH_DIR, tag)


# -----------------------------------------------------------------------------
#  CLI
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser("LASER experiments orchestrator")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--smoke-test", action="store_true", help="run the small quick test configuration")
    g.add_argument("--full-experiment", action="store_true", help="run the full training+eval pipeline")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    try:
        if args.smoke_test:
            config_file = CONFIG_DIR / "smoke_test.yaml"
            run_experiment(load_config(config_file), tag="smoke_test")
        elif args.full_experiment:
            # Always run smoke test first; abort if it fails
            smoke_cfg = load_config(CONFIG_DIR / "smoke_test.yaml")
            run_experiment(smoke_cfg, tag="smoke_test")

            full_cfg = load_config(CONFIG_DIR / "full_experiment.yaml")
            run_experiment(full_cfg, tag="full_experiment")
    except Exception as e:
        print("[FATAL] experiment failed: ", e, file=sys.stderr)
        sys.exit(1)
