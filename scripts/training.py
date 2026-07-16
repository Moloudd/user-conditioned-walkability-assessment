#!/usr/bin/env python3
"""
Model Training for Walkability Perception

Fits and saves a single MultiModalPredictor checkpoint per invocation. The
experiment (image backbone, tabular model, fusion method, loss function, and
whether tabular/demographic columns are used at all) is defined by a YAML
config file passed via --config. See scripts/configs/ for the final-model,
image-only, and ablation-grid configs; sweeping a grid means invoking this
script once per config file (e.g. a shell loop over scripts/configs/*.yaml).

Metrics are not computed here — use evaluation.py against the saved
checkpoint for that.

Config fields (see scripts/configs/*.yaml for examples):
  name: str                     # experiment name; used for logging and as the default --save-dir subfolder
  backbone_ckpt: str             # timm checkpoint name (model.timm_image.checkpoint_name)
  tabular: ft_transformer | mlp
  fusion: fusion_mlp | fusion_transformer
  loss: coral | corn | cross_entropy
  drop_tabular_columns: bool     # true = image-only (tabular/demographic columns dropped before loading)

Fixed settings (override via CLI flags):
- optim.max_epochs = 50
- env.per_gpu_batch_size = 32 (auto-reduced on --cpu for safety)
- env.num_workers = 8
- env.num_workers_inference = 8
- data.categorical.minimum_cat_count = 10
"""

import argparse
import os
import pkgutil
import shutil
import time
from datetime import datetime
from importlib.machinery import FileFinder, SourceFileLoader
from typing import Dict, List

import pandas as pd
import yaml

# Colab's Python 3.12 image can expose an older pkg_resources implementation
# that expects deprecated import APIs removed in Python 3.12.
if not hasattr(FileFinder, "find_module"):
    def _find_module_compat(self, fullname: str):
        spec = self.find_spec(fullname)
        return None if spec is None else spec.loader

    FileFinder.find_module = _find_module_compat  # type: ignore[attr-defined]

if not hasattr(pkgutil, "ImpImporter"):
    pkgutil.ImpImporter = FileFinder  # type: ignore[attr-defined]
if not hasattr(pkgutil, "ImpLoader"):
    pkgutil.ImpLoader = SourceFileLoader  # type: ignore[attr-defined]

from autogluon.multimodal import MultiModalPredictor

LABEL_COL = "rating"
IMAGE_COL = "image_path"
DROP_COLS = ["response_id"]

TABULAR_CHOICES = {"ft_transformer", "mlp"}
FUSION_CHOICES = {"fusion_mlp", "fusion_transformer"}
LOSS_CHOICES = {"coral", "corn", "cross_entropy"}

# All paths default relative to $WALKABILITY_DATA_DIR / $WALKABILITY_OUTPUT_DIR
# (see data/README.md for the expected dataset layout). Override individually
# with the matching --flag.
DATA_DIR = os.environ.get("WALKABILITY_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("WALKABILITY_OUTPUT_DIR", "./outputs")

DEFAULT_TRAIN_CSV = os.path.join(DATA_DIR, "splits", "train.csv")
DEFAULT_VAL_CSV = os.path.join(DATA_DIR, "splits", "val.csv")
DEFAULT_IMG_DIR = os.path.join(DATA_DIR, "images")

# CPU fallback for safety when --cpu is used.
CPU_SAFE_BATCH_SIZE = 8


def log(msg: str, level: str = "INFO") -> None:
    """Print a timestamped log line."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def to_abs(path: str, base: str) -> str:
    """Convert a path to absolute, relative to base if needed."""
    p = str(path).strip()
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base, p))


def add_abs_image_paths(df: pd.DataFrame, image_col: str, base_dir: str) -> pd.DataFrame:
    """Return a copy with image column converted to absolute paths."""
    out = df.copy()
    out[image_col] = out[image_col].astype(str).map(lambda p: to_abs(p, base_dir))
    return out


def load_config(config_path: str) -> Dict:
    """Load and validate an experiment config."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    required = ["name", "backbone_ckpt", "tabular", "fusion", "loss", "drop_tabular_columns"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config {config_path} is missing required fields: {missing}")

    if cfg["tabular"] not in TABULAR_CHOICES:
        raise ValueError(f"Config {config_path}: tabular must be one of {TABULAR_CHOICES}, got {cfg['tabular']!r}")
    if cfg["fusion"] not in FUSION_CHOICES:
        raise ValueError(f"Config {config_path}: fusion must be one of {FUSION_CHOICES}, got {cfg['fusion']!r}")
    if cfg["loss"] not in LOSS_CHOICES:
        raise ValueError(f"Config {config_path}: loss must be one of {LOSS_CHOICES}, got {cfg['loss']!r}")

    return cfg


def build_model_names(tabular: str, fusion: str) -> List[str]:
    """Return the ordered list of model names for AutoGluon."""
    if tabular == "ft_transformer":
        tabular_models = ["ft_transformer"]
    else:  # mlp
        tabular_models = ["categorical_mlp", "numerical_mlp"]
    return ["timm_image"] + tabular_models + [fusion]


def load_split(csv_path: str, img_dir: str, split_name: str, drop_tabular_columns: bool) -> pd.DataFrame:
    """Load one split, drop non-feature (or all tabular) cols, and normalize image paths."""
    log(f"Loading {split_name} split from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    missing_required = [c for c in [LABEL_COL, IMAGE_COL] if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"{split_name} split is missing required columns: {missing_required}. "
            f"Columns found: {list(df.columns)}"
        )

    if drop_tabular_columns:
        df = df[[IMAGE_COL, LABEL_COL]].copy()
    else:
        df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    df = add_abs_image_paths(df, IMAGE_COL, img_dir)

    if len(df) == 0:
        raise ValueError(f"{split_name} split is empty: {csv_path}")

    missing_img = (~df[IMAGE_COL].map(os.path.exists)).sum()
    if missing_img > 0:
        log(f"{split_name}: {missing_img}/{len(df)} image paths do not exist", "WARN")

    return df


def build_hyperparameters(
    cfg: Dict,
    use_cpu: bool,
    max_epochs: int,
    num_workers: int,
    num_workers_inference: int,
    effective_batch_size: int | None,
    per_gpu_batch_size: int | None,
) -> Dict:
    """Build the hyperparameter override dict for the configured experiment."""
    per_gpu_batch_size = 32 if per_gpu_batch_size is None else per_gpu_batch_size
    if use_cpu:
        per_gpu_batch_size = min(per_gpu_batch_size, CPU_SAFE_BATCH_SIZE)

    hparams = {
        "optim.max_epochs": max_epochs,
        "env.per_gpu_batch_size": per_gpu_batch_size,
        "env.num_workers": num_workers,
        "env.num_workers_inference": num_workers_inference,
        "env.precision": "bf16-mixed",
        "data.categorical.minimum_cat_count": 10,
        "optim.loss_func": cfg["loss"],
        "model.names": build_model_names(cfg["tabular"], cfg["fusion"]),
        "model.timm_image.checkpoint_name": cfg["backbone_ckpt"],
    }

    if effective_batch_size is not None:
        hparams["env.batch_size"] = effective_batch_size

    if use_cpu:
        hparams.update(
            {
                "env.num_gpus": 0,
                "env.accelerator": "cpu",
                "env.precision": 32,
                "env.strategy": "auto",
            }
        )

    return hparams


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one model from a YAML experiment config (see scripts/configs/).",
    )

    parser.add_argument("--config", type=str, required=True, help="Path to a YAML experiment config")
    parser.add_argument("--train-csv", type=str, default=DEFAULT_TRAIN_CSV, help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, default=DEFAULT_VAL_CSV, help="Path to validation CSV")
    parser.add_argument("--img-dir", type=str, default=DEFAULT_IMG_DIR, help="Base directory for image files")
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save the trained model (default: $WALKABILITY_OUTPUT_DIR/<config name>)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument(
        "--effective-batch-size",
        type=int,
        default=None,
        help="Optional effective/global batch size target (sets env.batch_size)",
    )
    parser.add_argument(
        "--per-gpu-batch-size",
        type=int,
        default=None,
        help="Optional micro-batch size per GPU. Useful for Colab memory limits.",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="Training data loader workers (default: 8)")
    parser.add_argument(
        "--num-workers-inference",
        type=int,
        default=8,
        help="Inference data loader workers (default: same as --num-workers)",
    )
    parser.add_argument("--time-limit", type=int, default=None, help="Optional fit time limit in seconds")
    parser.add_argument("--cpu", action="store_true", help="Force CPU mode. Batch size is auto-reduced for safety.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed passed to AutoGluon fit (default: 0)")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    num_workers_inference = args.num_workers if args.num_workers_inference is None else args.num_workers_inference

    cfg = load_config(args.config)
    save_dir = args.save_dir or os.path.join(OUTPUT_DIR, cfg["name"])

    log(f"Starting training: {cfg['name']}")
    log(f"config={args.config}")
    log(f"backbone={cfg['backbone_ckpt']}, tabular={cfg['tabular']}, fusion={cfg['fusion']}, loss={cfg['loss']}")
    log(f"drop_tabular_columns={cfg['drop_tabular_columns']}")
    log(f"train_csv={args.train_csv}")
    log(f"val_csv={args.val_csv}")
    log(f"img_dir={args.img_dir}")
    log(f"save_dir={save_dir}")
    log(f"epochs={args.epochs}")
    log(f"num_workers={args.num_workers}")
    log(f"num_workers_inference={num_workers_inference}")
    log(f"per_gpu_batch_size={args.per_gpu_batch_size}")
    log(f"effective_batch_size={args.effective_batch_size}")
    log(f"time_limit={args.time_limit}")
    log(f"seed={args.seed}")

    if args.cpu:
        log(
            f"CPU mode enabled: per_gpu_batch_size will be reduced from 32 to {CPU_SAFE_BATCH_SIZE} for safety",
            "WARN",
        )

    train_df = load_split(args.train_csv, args.img_dir, "train", cfg["drop_tabular_columns"])
    val_df = load_split(args.val_csv, args.img_dir, "val", cfg["drop_tabular_columns"])
    log(f"train: {len(train_df)} rows, val: {len(val_df)} rows")
    log(f"Feature columns: {[c for c in train_df.columns if c != LABEL_COL]}")

    if os.path.exists(save_dir):
        log(f"Removing existing directory: {save_dir}", "INFO")
        shutil.rmtree(save_dir)
    os.makedirs(os.path.dirname(os.path.abspath(save_dir)) or ".", exist_ok=True)

    hparams = build_hyperparameters(
        cfg=cfg,
        use_cpu=args.cpu,
        max_epochs=args.epochs,
        num_workers=args.num_workers,
        num_workers_inference=num_workers_inference,
        effective_batch_size=args.effective_batch_size,
        per_gpu_batch_size=args.per_gpu_batch_size,
    )
    log(f"model.names={hparams['model.names']}")

    log(
        "Using external validation via tuning_data=val_df; "
        "AutoGluon internal random split is bypassed."
    )

    t0 = time.time()
    predictor = MultiModalPredictor(
        label=LABEL_COL,
        problem_type="multiclass",
        eval_metric="quadratic_kappa",
        path=save_dir,
        verbosity=2,
    )

    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        hyperparameters=hparams,
        time_limit=args.time_limit,
        seed=args.seed,
    )

    train_time_sec = time.time() - t0
    log(f"Training finished in {train_time_sec:.1f}s")
    log(f"Saved model to: {save_dir}", "SUCCESS")
    log("Run scripts/evaluation.py --model-path <save-dir> --csv <split.csv> to compute metrics.")


if __name__ == "__main__":
    main()
