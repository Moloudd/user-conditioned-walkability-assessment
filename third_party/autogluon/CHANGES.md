# Modifications to AutoGluon

This directory vendors modified copies of
[`autogluon.multimodal`](https://github.com/autogluon/autogluon)
and its required AutoGluon dependencies:
`autogluon.core`, `autogluon.common`, and `autogluon.features`.

AutoGluon is licensed under the Apache License 2.0. See the `LICENSE` and
`NOTICE` files in this directory.

Only the subpackages required by this project are included. Other components
of the upstream monorepo, including `tabular`, `timeseries`, `eda`,
documentation, examples, and CI configuration, were omitted because they are
not used by this project.

## Upstream base

The `VERSION`/`VERSION.minor` files in this fork are not reliable fork-point
markers — they're rewritten to the *current date* every time `setup.py` is
run (a normal AutoGluon nightly-versioning mechanism), not the date this fork
diverged. 

This repository is based on AutoGluon commit
[`db3cca184028729ab459db4e3b9af505d5bcf44d`](https://github.com/autogluon/autogluon/commit/db3cca184028729ab459db4e3b9af505d5bcf44d)
from 1 November 2025. The `common/` and `features/` subpackages are unchanged from upstream, while `core/` and `multimodal/` contain the project-specific modifications described below.


## What was changed

Upstream AutoGluon does not support ordinal-regression loss functions for
`MultiModalPredictor`. This fork adds two — **CORAL** (COnsistent RAnk
Logits) and **CORN** (Conditional Ordinal Regression for Neural networks) —
for the ordinal walkability-rating prediction task in this project, along
with the plumbing needed to route their logits through prediction and
probability conversion, plus a generalized per-model random-seed mechanism
in `autogluon.core` used to make training runs reproducible. 10 files
changed across 2 subpackages:

**`autogluon.core`**:

- `core/src/autogluon/core/models/abstract/abstract_model.py` — adds
  `seed_name` / `seed_name_alt` class attributes so a model subclass can
  declare which hyperparameter key holds its random seed;
  `_get_random_seed_from_hyperparameters` now returns `(seed_value,
  seed_name)` instead of just the value, and `init_random_seed` writes the
  resolved seed back into the hyperparameters dict under that key (returning
  the updated dict, which callers now assign back to `self.params`). Also
  adds type annotations throughout `AbstractModel.__init__` (cosmetic, no
  behavior change). Separately, extracts a `_get_model_path()` helper
  (`os.path.join(path_root, path_suffix)`) used by `__init__` and the
  `_path_v2` property instead of duplicating that path computation inline —
  unrelated to the seed/ordinal-loss work, just a small cleanup.
- `core/src/autogluon/core/models/ensemble/bagged_ensemble_model.py` —
  replaces a hardcoded `_get_random_seed_from_hyperparameters` override
  (`hyperparameters.get("model_random_seed", "N/A")`) with the class
  attribute `seed_name = "model_random_seed"`, relying on the new
  generalized base-class logic above.

**`autogluon.multimodal`** (CORAL/CORN loss support):

- `multimodal/src/autogluon/multimodal/optim/losses/coral_loss.py` — CORAL
  loss implementation (new file). The per-sample BCE term is computed via
  PyTorch's built-in `F.binary_cross_entropy_with_logits` rather than a
  manual `log_sigmoid`-based formula, for numerical stability — the two are
  mathematically equivalent.
- `multimodal/src/autogluon/multimodal/optim/losses/corn_loss.py` — CORN loss
  implementation (new file).
- `multimodal/src/autogluon/multimodal/optim/losses/__init__.py` — export the
  CORAL/CORN loss symbols.
- `multimodal/src/autogluon/multimodal/optim/losses/utils.py` — wire up
  `get_loss_func` to construct a `CoralLoss`/`CornLoss` when `optim.loss_func`
  names "coral"/"corn". 

- `multimodal/src/autogluon/multimodal/models/utils.py` — post-processing
  path (logits → probabilities) for CORAL/CORN outputs; when a CORAL/CORN
  loss is active, the model head's output dimension is reduced by one
  (`num_classes - 1`), which is the actual mechanism implementing ordinal
  regression (each output is a cumulative/conditional threshold logit rather
  than a per-class logit).
- `multimodal/src/autogluon/multimodal/data/preprocess_dataframe.py` — accept
  a `loss_func_name` parameter in `transform_prediction` to disambiguate
  CORAL vs. CORN logit conversion (both produce `num_classes - 1` outputs).
- `multimodal/src/autogluon/multimodal/learners/base.py` — pass the active
  `loss_func_name` through to prediction/probability conversion call sites.
- `multimodal/src/autogluon/multimodal/utils/misc.py` — `logits_to_prob`
  takes an optional `loss_func_name` to pick CORAL- vs. CORN-style
  conversion.


