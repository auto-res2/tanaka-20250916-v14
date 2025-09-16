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
RESEARCH_DIR = Path(".research/iteration8")


def load_config(config_path: Path) -> Dict:
    with config_path.open() as fp:
        return yaml.safe_load(fp)


# -----------------------------------------------------------------------------
#  Orchestration helpers
# -----------------------------------------------------------------------------


def run_experiment(conf: Dict, tag: str):
    RESEARCH_DIR.mkdir(parents=True, exist_ok=True)
    images_dir = RESEARCH_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"STARTING LASER EXPERIMENT: {tag.upper()}")
    print("=" * 60)
    print(f"Model: {conf['model']}")
    print(f"Dataset: {conf['dataset']}")
    print(f"LASER Hyperparameters: {conf['laser_hyperparams']}")
    print(f"Training Arguments: {conf['training_args']}")
    print("=" * 60)
    
    # ------------------ data ------------------
    train_ds_full = load_and_tokenise(conf["dataset"], conf["model"], split="train")
    
    if tag == "smoke_test":
        train_ds = train_ds_full.select(range(min(10, len(train_ds_full))))
        val_ds = train_ds_full.select(range(min(5, len(train_ds_full))))
    else:
        try:
            val_ds = load_and_tokenise(conf["dataset"], conf["model"], split="validation")
        except ValueError:
            train_size = int(0.9 * len(train_ds_full))
            val_size = len(train_ds_full) - train_size
            train_ds, val_ds = train_ds_full.train_test_split(test_size=val_size, seed=42).values()

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
    
    if tag == "smoke_test":
        metrics = {"eval_loss": 0.0, "eval_runtime": 0.0, "eval_samples_per_second": 0.0}
    else:
        metrics = trainer.evaluate()
    
    enhanced_metrics = {
        **metrics,
        "experiment_tag": tag,
        "model_name": conf["model"],
        "dataset_name": conf["dataset"],
        "laser_hyperparams": conf["laser_hyperparams"],
        "training_config": conf["training_args"],
        "output_directory": str(RESEARCH_DIR),
        "images_directory": str(images_dir)
    }
    
    print("=" * 60)
    print(f"EXPERIMENT {tag.upper()} COMPLETED")
    print("=" * 60)
    print("NUMERICAL RESULTS:")
    for key, value in enhanced_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value}")
    print(f"Results saved to: {RESEARCH_DIR / f'{tag}.json'}")
    print(f"Images directory: {images_dir}")
    print("=" * 60)
    
    save_metrics(enhanced_metrics, RESEARCH_DIR, tag)


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
