#!/usr/bin/env python3
"""
Permutation-based feature importance for a fixed trained  user-conditioned multimodal model of walkability perception.

Workflow:
1) Load a pre-trained model (no re-training)
2) Load test dataframe
3) Compute baseline metrics
4) For each selected feature, shuffle user-level values across unique response_id
5) Re-predict and compute metric drop vs baseline
6) Repeat N times and average drops
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, mean_absolute_error

from autogluon.multimodal import MultiModalPredictor

# All paths default relative to $WALKABILITY_DATA_DIR / $WALKABILITY_OUTPUT_DIR
# (see data/README.md for the expected dataset layout). Override individually
# with the matching --flag. --model-path has no hardcoded default and must
# be passed — point it at a checkpoint you trained yourself, or one
# downloaded from the Hugging Face model repo (see "Pretrained checkpoints"
# in the top-level README.md).
DATA_DIR = os.environ.get("WALKABILITY_DATA_DIR", "./data")
OUTPUT_DIR = os.environ.get("WALKABILITY_OUTPUT_DIR", "./outputs")

DEFAULT_TEST_CSV = os.path.join(DATA_DIR, "splits", "test.csv")
DEFAULT_OUTPUT = os.path.join(OUTPUT_DIR, "permutation_importance.csv")
DEFAULT_RAW_OUTPUT = os.path.join(OUTPUT_DIR, "permutation_importance_raw.csv")
DEFAULT_IMG_DIR = os.path.join(DATA_DIR, "images")

LABEL_COL = "rating"
IMAGE_COL = "image_path"
USER_ID_COL = "response_id"

DEFAULT_COLUMNS = [
    "age",
    "gender",
    "childhood_country",
    "childhood_area",
    "disability",
    "walking_frequency",
    "residence_type",
]

# Metrics where larger is better (drop = baseline - permuted)
HIGHER_IS_BETTER = {"within_one_accuracy", "quadratic_kappa"}

# Metrics where smaller is better (drop = permuted - baseline)
LOWER_IS_BETTER = {"mae"}


def log(msg: str, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {msg}")


def to_abs(path: str, base: str) -> str:
    p = str(path).strip()
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(base, p))


def add_abs_image_paths(df: pd.DataFrame, image_col: str, base_dir: str) -> pd.DataFrame:
    out = df.copy()
    out[image_col] = out[image_col].astype(str).map(lambda p: to_abs(p, base_dir))
    return out


def compute_ordinal_metrics(y_true, y_pred) -> dict:
    classes = [1, 2, 3, 4, 5]

    y_true_int = np.array(y_true, dtype=int)
    y_pred_int = np.array(y_pred, dtype=int)
    y_true_float = np.array(y_true, dtype=np.float32)
    y_pred_float = np.array(y_pred, dtype=np.float32)

    within_one = np.mean(np.abs(y_true_float - y_pred_float) <= 1)
    mae = mean_absolute_error(y_true_float, y_pred_float)
    qwk = cohen_kappa_score(y_true_int, y_pred_int, weights="quadratic", labels=classes)

    return {
        "within_one_accuracy": float(within_one),
        "mae": float(mae),
        "quadratic_kappa": float(qwk),
    }


def resolve_predictor_path(model_path: str) -> str:
    """
    Accept either predictor directory or a file path (e.g., model.ckpt).
    If a file is provided, use its parent directory for loading.
    """
    model_path = os.path.abspath(model_path)
    if os.path.isdir(model_path):
        return model_path
    if os.path.isfile(model_path):
        return os.path.dirname(model_path)
    raise FileNotFoundError(f"Model path does not exist: {model_path}")


def permute_feature_by_user(df: pd.DataFrame, feature_col: str, user_col: str, rng: np.random.Generator) -> pd.DataFrame:
    out = df.copy()

    if user_col not in out.columns:
        raise KeyError(f"Missing user id column: {user_col}")
    if feature_col not in out.columns:
        raise KeyError(f"Missing feature column: {feature_col}")

    # Use first observed value per user and shuffle across users.
    # This enforces user-level consistency for the permuted column.
    per_user_values = out.groupby(user_col)[feature_col].first()
    user_ids = per_user_values.index.to_numpy()
    values = per_user_values.to_numpy()
    shuffled_values = rng.permutation(values)
    mapping = dict(zip(user_ids, shuffled_values))

    out[feature_col] = out[user_col].map(mapping)
    return out


def compute_drop(baseline: dict, permuted: dict) -> dict:
    drops = {}
    for metric, b_val in baseline.items():
        p_val = permuted.get(metric, np.nan)
        if pd.isna(b_val) or pd.isna(p_val):
            drops[f"drop_{metric}"] = np.nan
            continue

        if metric in HIGHER_IS_BETTER:
            drops[f"drop_{metric}"] = float(b_val - p_val)
        elif metric in LOWER_IS_BETTER:
            drops[f"drop_{metric}"] = float(p_val - b_val)
        else:
            drops[f"drop_{metric}"] = np.nan

    return drops


def main() -> None:
    parser = argparse.ArgumentParser(description="Permutation-based feature importance for fixed AutoGluon model")
    parser.add_argument("--model-path", type=str, required=True, help="Predictor directory or model file path (e.g. output of training.py)")
    parser.add_argument("--test-csv", type=str, default=DEFAULT_TEST_CSV, help="Test split CSV")
    parser.add_argument("--img-dir", type=str, default=DEFAULT_IMG_DIR, help="Base image directory for image_path")
    parser.add_argument("--columns", type=str, nargs="+", default=DEFAULT_COLUMNS, help="Columns to permute")
    parser.add_argument("--repeats", type=int, default=30, help="Permutation repeats per column")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT, help="CSV path for averaged drops")
    parser.add_argument("--raw-output", type=str, default=DEFAULT_RAW_OUTPUT, help="CSV path for all repeats")
    args = parser.parse_args()

    log("Loading fixed model...")
    predictor_path = resolve_predictor_path(args.model_path)
    predictor = MultiModalPredictor.load(predictor_path)
    log(f"Loaded predictor from: {predictor_path}")

    log("Loading test dataframe...")
    test_df = pd.read_csv(args.test_csv)

    missing = [c for c in [LABEL_COL, IMAGE_COL, USER_ID_COL] if c not in test_df.columns]
    if missing:
        raise KeyError(f"Missing required columns in test CSV: {missing}")

    test_df = add_abs_image_paths(test_df, IMAGE_COL, args.img_dir)

    invalid_cols = [c for c in args.columns if c not in test_df.columns]
    if invalid_cols:
        raise KeyError(f"Columns requested in --columns not found in test CSV: {invalid_cols}")

    # Baseline prediction with original test set.
    log("Computing baseline metrics...")
    baseline_features = test_df.drop(columns=[USER_ID_COL])
    baseline_preds = predictor.predict(baseline_features.drop(columns=[LABEL_COL]))
    baseline_metrics = compute_ordinal_metrics(test_df[LABEL_COL].values, baseline_preds.values)

    log("Baseline metrics:")
    for k, v in baseline_metrics.items():
        log(f"  {k}: {v}")

    rng = np.random.default_rng(args.seed)
    raw_rows = []
    tracked_metrics = list(baseline_metrics.keys())

    for col in args.columns:
        log(f"Running permutation importance for column: {col}")
        for rep in range(args.repeats):
            perm_df = permute_feature_by_user(test_df, col, USER_ID_COL, rng)

            # Remove response_id before prediction.
            pred_df = perm_df.drop(columns=[USER_ID_COL])
            preds = predictor.predict(pred_df.drop(columns=[LABEL_COL]))

            perm_metrics = compute_ordinal_metrics(perm_df[LABEL_COL].values, preds.values)
            drop_metrics = compute_drop(baseline_metrics, perm_metrics)

            row = {
                "feature": col,
                "repeat": rep,
            }
            for metric in tracked_metrics:
                row[f"permuted_{metric}"] = perm_metrics.get(metric, np.nan)
            row.update(drop_metrics)
            raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)

    # Build summary with baseline metric, median permuted metric, median drop, IQR of drop.
    summary_rows = []
    for feature, group in raw_df.groupby("feature"):
        row = {"feature": feature}
        for metric in tracked_metrics:
            baseline_val = baseline_metrics.get(metric, np.nan)
            median_permuted = group[f"permuted_{metric}"].median() if f"permuted_{metric}" in group else np.nan
            median_drop = group[f"drop_{metric}"].median() if f"drop_{metric}" in group else np.nan
            iqr_drop = (group[f"drop_{metric}"].quantile(0.75) - group[f"drop_{metric}"].quantile(0.25)) if f"drop_{metric}" in group else np.nan

            row[f"baseline_{metric}"] = baseline_val
            row[f"median_permuted_{metric}"] = median_permuted
            row[f"median_drop_{metric}"] = median_drop
            row[f"iqr_drop_{metric}"] = iqr_drop

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.raw_output)), exist_ok=True)

    raw_df.to_csv(args.raw_output, index=False)
    summary_df.to_csv(args.output, index=False)

    log(f"Saved raw repeat-level drops to: {args.raw_output}")
    log(f"Saved averaged permutation importance to: {args.output}")

    # Console-friendly view sorted by median drop in QWK.
    sort_col = "median_drop_quadratic_kappa"
    if sort_col in summary_df.columns:
        summary_df = summary_df.sort_values(sort_col, ascending=False)

    print("\nAveraged permutation drops (higher positive drop = more important):")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
