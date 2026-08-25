# Walkable to Whom? Capturing Subjective Variability in Walkability Perception Using Multimodal Deep Learning

This repository contains the official implementation of **[Walkable to Whom? Capturing Subjective Variability in Walkability Perception Using Multimodal Deep Learning](https://arxiv.org/abs/2608.06934)**  

The proposed multimodal deep learning framework combines sidewalk-view imagery with respondent demographic and mobility characteristics to predict user-conditioned walkability ratings on a 1–5 ordinal scale. Rather than assigning a single aggregated score to each environment, the model captures variation in how different individuals perceive and rate the same pedestrian environment.


Built on [AutoGluon](https://github.com/autogluon/autogluon)'s
`MultiModalPredictor`. This repository vendors a **modified fork** of
AutoGluon (`third_party/autogluon/`) that adds CORAL and CORN ordinal-
regression losses, which upstream AutoGluon does not support — see
`third_party/autogluon/CHANGES.md`.

## Repository structure

```
.
├── scripts/
│   ├── training.py                      # train one model from a YAML config (final model, image-only, or an ablation variant)
│   ├── configs/
│   │   ├── final_model.yaml              # Swin-Tiny + FT-Transformer + fusion transformer + CORAL
│   │   ├── image_only.yaml               # same as final_model, tabular/demographic columns dropped
│   │   ├── component_ablation/           # single-component swaps against final_model (fusion, tabular, loss)
│   │   └── backbone_ablation/            # image backbone swaps against final_model (resnet50, convnext_tiny, caformer_b36)
│   ├── evaluation.py                    # compute metrics for a trained checkpoint against a labeled split
│   ├── permutation_feature_importance.py  # per-feature importance for a trained model
│   └── inference.py                     # score a CSV of samples with a trained checkpoint
├── third_party/autogluon/               # modified AutoGluon fork (CORAL + CORN losses); see CHANGES.md
├── data/README.md                       # expected dataset layout & how to obtain data
├── install.sh
├── requirements.txt
└── .env.example
```

## Installation

Requires Python (developed against the versions pinned by
`third_party/autogluon/*/setup.py`) and a CUDA-capable GPU for realistic
training times (CPU works via `--cpu` but is slow).

```bash
git clone <this-repo-url>
cd user-conditioned-walkability-assessment
python -m venv .venv && source .venv/bin/activate   
./install.sh
```

`install.sh` editable-installs the modified AutoGluon fork from
`third_party/autogluon/` (do **not** `pip install autogluon` from PyPI — it
lacks the CORN/CORAL loss support these scripts depend on), then installs
the remaining direct dependencies from `requirements.txt`.

Verify the install:

```bash
python -c "from autogluon.multimodal import MultiModalPredictor"
```

## Data setup

See [`data/README.md`](data/README.md)
for the expected directory layout, CSV schema, and how to obtain/prepare the
dataset. Once ready, either place data under `./data` or point
`WALKABILITY_DATA_DIR` (see `.env.example`) at your dataset location.

## Pretrained checkpoints

Two trained checkpoints are hosted on Hugging Face:
[`mdamandeh/user-conditioned-walkability-assessment-model`](https://huggingface.co/mdamandeh/user-conditioned-walkability-assessment-model).

| Checkpoint (subfolder) | Config                             | Description                                                                                    |
|-------------------------|--------------------------------------|--------------------------------------------------------------------------------------------------|
| `final_model`             | `scripts/configs/final_model.yaml` | Swin-Tiny + FT-Transformer + fusion transformer + CORAL — the best performing configuration    |
| `image_only`               | `scripts/configs/image_only.yaml`  | Same as `final_model`, tabular/demographic columns dropped — quantifies how much those features contribute |

Download either with the Hugging Face CLI:

```bash
hf download mdamandeh/user-conditioned-walkability-assessment-model final_model \
  --repo-type model \
  --local-dir outputs/final_model

hf download mdamandeh/user-conditioned-walkability-assessment-model image_only \
  --repo-type model \
  --local-dir outputs/image_only
```

or with `huggingface_hub` in Python:

```python
from huggingface_hub import snapshot_download
snapshot_download("mdamandeh/user-conditioned-walkability-assessment-model", repo_type="model", allow_patterns="final_model/*", local_dir="outputs")
snapshot_download("mdamandeh/user-conditioned-walkability-assessment-model", repo_type="model", allow_patterns="image_only/*", local_dir="outputs")
```

Once downloaded, point `--model-path` at `outputs/final_model` or
`outputs/image_only` for `evaluation.py`, `inference.py`, or
`permutation_feature_importance.py` (see below) — no training required.

## Training / ablations

`training.py` trains and saves exactly one model per invocation; which
experiment it runs is entirely determined by the `--config` YAML file you
pass it (see `scripts/configs/`). It does not compute metrics — it only fits
and saves a checkpoint. Use `evaluation.py` against the saved checkpoint to
get metrics.

**Final model** (Swin-Tiny + FT-Transformer + fusion transformer + CORAL —
the configuration used in the paper):

```bash
python scripts/training.py --config scripts/configs/final_model.yaml
```

**Image-only baseline** (same as final_model, tabular/demographic columns
dropped — tests how much those features actually contribute):

```bash
python scripts/training.py --config scripts/configs/image_only.yaml
```

**Component ablation** (single-component swaps against `final_model.yaml` —
tabular model, fusion method, or loss function each varied one at a time):

```bash
for cfg in scripts/configs/component_ablation/*.yaml; do
  python scripts/training.py --config "$cfg"
done
```

**Backbone ablation** (image backbone swapped against `final_model.yaml` —
resnet50, convnext_tiny, caformer_b36):

```bash
for cfg in scripts/configs/backbone_ablation/*.yaml; do
  python scripts/training.py --config "$cfg"
done
```

Each config's `name` field sets the default `--save-dir` subfolder under
`$WALKABILITY_OUTPUT_DIR` (e.g. `outputs/component_ablation/ft_transformer__fusion_mlp__coral`).
Run `python scripts/training.py --help` for the full flag list (batch size,
epochs, `--cpu`, etc).

## Evaluation

Compute metrics (accuracy, within-one accuracy, MAE, QWK, per-class recall/F1/
absolute-error, confusion matrix) for an already-trained checkpoint against a
labeled CSV split:

```bash
python scripts/evaluation.py \
  --model-path outputs/final_model \
  --csv data/splits/test.csv \
  --split-name test
```

## Inference

Score a CSV of samples (with or without a `rating` column) using an
already-trained checkpoint from any of the training scripts above:

```bash
python scripts/inference.py \
  --model-path outputs/final_model \
  --input-csv data/splits/test.csv \
  --output-csv outputs/predictions.csv
```

Writes the input CSV back out with a `predicted_rating` column added. 

## Interpretability

**Permutation feature importance** for an already-trained model: shuffles
one demographic column at a time and measures the resulting metric drop.

```bash
python scripts/permutation_feature_importance.py \
  --model-path outputs/final_model \
  --columns age gender disability walking_frequency
```


## License

Original code in this repository is MIT-licensed — see `LICENSE`. The
vendored AutoGluon fork under `third_party/autogluon/` is Apache-2.0 — see
`third_party/autogluon/LICENSE` and `third_party/autogluon/NOTICE`.

## Citation

See [`CITATION.cff`](CITATION.cff).
