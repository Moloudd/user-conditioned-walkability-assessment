#!/usr/bin/env bash
# Install this project's dependencies, including the modified AutoGluon
# fork vendored under third_party/autogluon/. Run from the repo root.
set -euo pipefail
script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

PY="${PYTHON:-python}"

echo "[1/2] Installing the modified AutoGluon fork (editable) from third_party/autogluon/ ..."
"$PY" -m pip install -e third_party/autogluon/common
"$PY" -m pip install -e third_party/autogluon/core
"$PY" -m pip install -e third_party/autogluon/features
"$PY" -m pip install -e third_party/autogluon/multimodal

echo "[2/2] Installing remaining requirements ..."
"$PY" -m pip install -r requirements.txt

echo "Done. Verify with: $PY -c 'from autogluon.multimodal import MultiModalPredictor'"
