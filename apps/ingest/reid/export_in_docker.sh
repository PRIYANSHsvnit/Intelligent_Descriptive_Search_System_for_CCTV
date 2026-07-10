#!/usr/bin/env bash
# One-time: export the FastReID VeRi checkpoint to ONNX in a throwaway old-torch
# container (keeps fast-reid OUT of the pipeline venv, per plan.md [6]).
# Produces veri_reid.onnx in this dir; move it to ../models/veri_reid.onnx.
set -euo pipefail
cd "$(dirname "$0")"

docker run --rm -v "$PWD":/w -w /w python:3.9-slim bash -lc '
  set -e
  apt-get update -qq && apt-get install -y -qq git build-essential >/dev/null
  pip install -q torch==1.13.1 torchvision==0.14.1 --index-url https://download.pytorch.org/whl/cpu
  pip install -q yacs termcolor scikit-learn tabulate "numpy<2" Pillow onnx
  git clone -q --depth 1 https://github.com/JDAI-CV/fast-reid.git
  PYTHONPATH=/w/fast-reid python export_reid_onnx.py
'
echo "---"
ls -la veri_reid.onnx
