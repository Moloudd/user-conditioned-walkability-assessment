#!/usr/bin/env python3
"""
Model Inference for Walkability Perception

Loads an already-trained MultiModalPredictor checkpoint (output of
training.py) and scores a CSV of samples, writing a copy with a
`predicted_rating` column added.
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings("ignore")

# All paths default relative to $WALKABILITY_DATA_DIR / $WALKABILITY_OUTPUT_DIR
# (see data/README.md for the expected dataset layout). Override individually
# with the matching --flag. --model-path has no hardcoded default and must
# be passed — point it at a checkpoint you trained yourself, or one
# downloaded from the Hugging Face model repo (see "Pretrained checkpoints"
# in the top-level README.md).
DATA_DIR = os.environ.get("WALKABILITY_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("WALKABILITY_OUTPUT_DIR", "./outputs")

DEFAULT_INPUT_CSV = os.path.join(DATA_DIR, "splits", "test.csv")
DEFAULT_IMG_DIR = os.path.join(DATA_DIR, "images")
DEFAULT_OUTPUT_CSV = os.path.join(OUTPUT_DIR, "predictions.csv")

LABEL_COL = "rating"
IMAGE_COL = "image_path"
DROP_COLS = ["response_id"]


def log(msg: str, level: str = "INFO") -> None:
    print(f"[{level}] {msg}")


def to_abs(path: str, base: str) -> str:
    p = str(path).strip()
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base, p))


def add_abs_image_paths(df: pd.DataFrame, image_col: str, base_dir: str) -> pd.DataFrame:
    out = df.copy()
    out[image_col] = out[image_col].astype(str).map(lambda p: to_abs(p, base_dir))
    return out


def align_label_scales(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Best-effort fix-up for checkpoints trained when labels were 0-indexed
    instead of the current 1-5 scale.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    true_labels = sorted(np.unique(y_true).tolist())
    pred_labels = sorted(np.unique(y_pred).tolist())

    if true_labels != pred_labels and len(true_labels) == 5 and len(pred_labels) == 5:
        t = np.array(true_labels)
        p = np.array(pred_labels)
        if np.array_equal(t + 1, p):
            log("Detected label offset (true: 0-4, pred: 1-5). Shifting true labels to 1-5.")
            y_true = y_true + 1
        elif np.array_equal(p + 1, t):
            log("Detected label offset (pred: 0-4, true: 1-5). Shifting predicted labels to 1-5.")
            y_pred = y_pred + 1

    if sorted(np.unique(y_true).tolist()) == [0, 1, 2, 3, 4] and sorted(np.unique(y_pred).tolist()) == [0, 1, 2, 3, 4]:
        log("Detected 0-4 labels. Shifting both to 1-5.")
        y_true = y_true + 1
        y_pred = y_pred + 1

    return y_true, y_pred


def resolve_model_dir(path: str) -> str:
    if path.endswith(".ckpt"):
        return str(Path(path).resolve().parent)
    return str(Path(path).resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference with a trained MultiModalPredictor checkpoint on a CSV of samples.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Predictor directory or model file path (e.g. output of training.py)",
    )
    parser.add_argument("--input-csv", type=str, default=DEFAULT_INPUT_CSV, help="CSV of samples to score")
    parser.add_argument("--img-dir", type=str, default=DEFAULT_IMG_DIR, help="Base directory for image_path")
    parser.add_argument("--output-csv", type=str, default=DEFAULT_OUTPUT_CSV, help="Where to write the input CSV plus a predicted_rating column")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model_dir = resolve_model_dir(args.model_path)
    log(f"Loading model from: {model_dir}")
    predictor = MultiModalPredictor.load(model_dir)
    log("Model loaded.")

    log(f"Loading input CSV from: {args.input_csv}")
    input_df = pd.read_csv(args.input_csv, index_col=0)
    output_df = input_df.copy()

    model_df = input_df.drop(columns=[c for c in DROP_COLS if c in input_df.columns])
    model_df = add_abs_image_paths(model_df, IMAGE_COL, args.img_dir)

    missing = (~model_df[IMAGE_COL].map(os.path.exists)).sum()
    log(f"Missing image files: {missing}/{len(model_df)}")
    if missing == len(model_df):
        raise FileNotFoundError(f"All image paths are missing. Check --img-dir='{args.img_dir}'.")

    X = model_df.drop(columns=[LABEL_COL]) if LABEL_COL in model_df.columns else model_df

    log(f"Running inference on {len(X)} samples...")
    preds = predictor.predict(X)
    y_pred = preds.values if hasattr(preds, "values") else np.asarray(preds)

    if LABEL_COL in input_df.columns:
        y_true = input_df[LABEL_COL].values.astype(int)
        _, y_pred = align_label_scales(y_true, y_pred)
    else:
        y_pred = np.asarray(y_pred, dtype=int)

    output_df["predicted_rating"] = y_pred

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(out_path)
    log(f"Saved predictions to: {out_path}")
    print(output_df["predicted_rating"].value_counts().sort_index())


if __name__ == "__main__":
    main()
