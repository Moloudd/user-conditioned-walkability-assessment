#!/usr/bin/env python3
"""
Model Evaluation for Walkability Perception

Loads an already-trained MultiModalPredictor checkpoint (output of
training.py) and computes
metrics against a labeled CSV split (must contain the `rating` column).

Metrics reported:
- accuracy
- within-one accuracy
- MAE
- quadratic weighted kappa (QWK)
- per-class recall, F1, and mean absolute error
- confusion matrix
"""

import argparse
import os
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, mean_absolute_error, recall_score

from autogluon.multimodal import MultiModalPredictor

LABEL_COL = "rating"
IMAGE_COL = "image_path"
DROP_COLS = ["response_id"]
LABELS = [1, 2, 3, 4, 5]

# All paths default relative to $WALKABILITY_DATA_DIR / $WALKABILITY_OUTPUT_DIR
# (see data/README.md for the expected dataset layout). Override individually
# with the matching --flag. --model-path has no hardcoded default and must
# be passed — point it at a checkpoint you trained yourself, or one
# downloaded from the Hugging Face model repo (see "Pretrained checkpoints"
# in the top-level README.md).
DATA_DIR = os.environ.get("WALKABILITY_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("WALKABILITY_OUTPUT_DIR", "./outputs")

DEFAULT_CSV = os.path.join(DATA_DIR, "splits", "test.csv")
DEFAULT_IMG_DIR = os.path.join(DATA_DIR, "images")
DEFAULT_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "evaluation_results.csv")


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


def resolve_predictor_path(model_path: str) -> str:
    """Accept either a predictor directory or a file path (e.g., model.ckpt)."""
    model_path = os.path.abspath(model_path)
    if os.path.isdir(model_path):
        return model_path
    if os.path.isfile(model_path):
        return os.path.dirname(model_path)
    raise FileNotFoundError(f"Model path does not exist: {model_path}")


def load_labeled_split(csv_path: str, img_dir: str) -> pd.DataFrame:
    """Load a labeled split, drop non-feature cols, and normalize image paths."""
    log(f"Loading CSV from: {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    missing_required = [c for c in [LABEL_COL, IMAGE_COL] if c not in df.columns]
    if missing_required:
        raise ValueError(
            f"CSV is missing required columns: {missing_required}. "
            f"Columns found: {list(df.columns)}"
        )

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = add_abs_image_paths(df, IMAGE_COL, img_dir)

    if len(df) == 0:
        raise ValueError(f"CSV is empty: {csv_path}")

    missing_img = (~df[IMAGE_COL].map(os.path.exists)).sum()
    if missing_img > 0:
        log(f"{missing_img}/{len(df)} image paths do not exist", "WARN")

    return df


def compute_ordinal_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: List[int] = LABELS) -> Dict:
    """Compute ordinal classification metrics, including per-class breakdowns and confusion matrix."""
    y_true_i = np.asarray(y_true, dtype=int)
    y_pred_i = np.asarray(y_pred, dtype=int)
    y_true_f = y_true_i.astype(np.float32)
    y_pred_f = y_pred_i.astype(np.float32)

    acc = accuracy_score(y_true_i, y_pred_i)
    within_one = float(np.mean(np.abs(y_true_f - y_pred_f) <= 1))
    mae = mean_absolute_error(y_true_f, y_pred_f)
    qwk = cohen_kappa_score(y_true_i, y_pred_i, weights="quadratic", labels=labels)

    recall_per_class = recall_score(y_true_i, y_pred_i, average=None, labels=labels, zero_division=0)
    f1_per_class = f1_score(y_true_i, y_pred_i, average=None, labels=labels, zero_division=0)

    abs_err_per_class = []
    for c in labels:
        mask = y_true_i == c
        abs_err_per_class.append(float(np.mean(np.abs(y_true_f[mask] - y_pred_f[mask]))) if mask.any() else float("nan"))

    cm = confusion_matrix(y_true_i, y_pred_i, labels=labels)

    metrics = {
        "accuracy": float(acc),
        "within_one_accuracy": within_one,
        "mae": float(mae),
        "qwk": float(qwk),
        "confusion_matrix": cm,
    }
    for c, r in zip(labels, recall_per_class):
        metrics[f"recall_class{c}"] = float(r)
    for c, f1 in zip(labels, f1_per_class):
        metrics[f"f1_class{c}"] = float(f1)
    for c, e in zip(labels, abs_err_per_class):
        metrics[f"abs_err_class{c}"] = e

    return metrics


def print_metrics(metrics: Dict, name: str, labels: List[int] = LABELS) -> None:
    """Pretty-print a metrics dict, including per-class breakdown and confusion matrix."""
    log(f"--- {name} metrics ---")
    log(f"  accuracy:            {metrics['accuracy']:.4f}")
    log(f"  within_one_accuracy: {metrics['within_one_accuracy']:.4f}")
    log(f"  mae:                 {metrics['mae']:.4f}")
    log(f"  qwk:                 {metrics['qwk']:.4f}")
    log("  Per-class Recall / F1 / AbsErr:")
    for c in labels:
        log(
            f"    class {c}: recall={metrics[f'recall_class{c}']:.4f} "
            f"f1={metrics[f'f1_class{c}']:.4f} abs_err={metrics[f'abs_err_class{c}']:.4f}"
        )
    log(f"  Confusion matrix (rows=true, cols=pred, labels={labels}):")
    for row in metrics["confusion_matrix"]:
        log(f"    {row.tolist()}")


def flatten_metrics(metrics: Dict) -> Dict:
    """Flatten a metrics dict into result-row columns."""
    flat = {}
    for k, v in metrics.items():
        if k == "confusion_matrix":
            flat[k] = str(v.tolist())
        else:
            flat[k] = v
    return flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute metrics for a trained MultiModalPredictor checkpoint on a labeled CSV split.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Predictor directory or model file path (e.g. output of training.py)",
    )
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Labeled CSV to evaluate (must contain `rating`)")
    parser.add_argument("--img-dir", type=str, default=DEFAULT_IMG_DIR, help="Base directory for image_path")
    parser.add_argument("--split-name", type=str, default="eval", help="Label used in log output for this run")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Where to append/write the metrics row")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    predictor_path = resolve_predictor_path(args.model_path)
    log(f"Loading model from: {predictor_path}")
    predictor = MultiModalPredictor.load(predictor_path)
    log("Model loaded.")

    df = load_labeled_split(args.csv, args.img_dir)
    log(f"{args.split_name}: {len(df)} rows")

    y_true = df[LABEL_COL].values
    y_pred = predictor.predict(df.drop(columns=[LABEL_COL])).values

    metrics = compute_ordinal_metrics(y_true, y_pred)
    print_metrics(metrics, name=args.split_name)

    row = {"split_name": args.split_name, "model_path": predictor_path, "csv": args.csv}
    row.update(flatten_metrics(metrics))

    out_path = args.output_csv
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    row_df = pd.DataFrame([row])
    if os.path.exists(out_path):
        row_df.to_csv(out_path, mode="a", header=False, index=False)
    else:
        row_df.to_csv(out_path, index=False)

    log(f"Saved metrics row to: {out_path}")


if __name__ == "__main__":
    main()
