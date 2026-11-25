#!/usr/bin/env bash
# One-time setup on a fresh Linux GPU machine (tested target: Ubuntu with an NVIDIA driver).
# Run from anywhere inside the repository:  bash cloud/setup.sh
#
# Creates .venv (Python 3.14 via uv, the same versions as requirements.txt), downloads
# Qwen3-0.6B into models/, checks the data files and CUDA, and runs the unit tests.
#   TORCH_INDEX  PyTorch wheel index (default CUDA 13.0); use an older CUDA index if the
#                driver is too old for CUDA 13 (nvidia-smi shows "CUDA Version" < 13.0)
#   PY_VERSION   Python version for the venv (default 3.14)
set -euo pipefail
cd "$(dirname "$0")/.."
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}"
PY_VERSION="${PY_VERSION:-3.14}"
MODEL_SHA256=f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b  # local copy used so far

echo "== GPU"
if ! command -v nvidia-smi >/dev/null; then
  echo "nvidia-smi not found: install the NVIDIA driver first (cloud/GUIDE.zh.md, step 3)." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi | grep -o "CUDA Version: [0-9.]*" || true

echo "== Python environment"
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv python install "$PY_VERSION"
[ -x .venv/bin/python ] || uv venv --python "$PY_VERSION" .venv
uv pip install --python .venv/bin/python "torch==2.12.1" --index-url "$TORCH_INDEX"
grep -v '^torch==' requirements.txt > /tmp/requirements-no-torch.txt
uv pip install --python .venv/bin/python -r /tmp/requirements-no-torch.txt
PY=.venv/bin/python

echo "== Model"
if [ ! -f models/Qwen3-0.6B/model.safetensors ]; then
  "$PY" -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-0.6B', local_dir='models/Qwen3-0.6B')"
fi
got=$(sha256sum models/Qwen3-0.6B/model.safetensors | cut -d' ' -f1)
if [ "$got" = "$MODEL_SHA256" ]; then
  echo "model.safetensors matches the local copy"
else
  echo "WARNING: model.safetensors sha256 $got differs from the local copy ($MODEL_SHA256)" >&2
fi

echo "== Data"
missing=0
for f in data/uf_train4096.jsonl data/uf_dev500.jsonl data/uf_test1486.jsonl; do
  [ -f "$f" ] || { echo "missing $f"; missing=1; }
done
[ -f data/arena-hard-v0.1/question.jsonl ] || echo "note: data/arena-hard-v0.1/question.jsonl missing (only needed for --arena)"
if [ "$missing" = 1 ]; then
  echo "Upload cloud_data.tgz and extract it in the repository root (cloud/GUIDE.zh.md, step 4)." >&2
fi

echo "== CUDA check"
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; x = torch.ones(4, device='cuda', dtype=torch.bfloat16); print('torch', torch.__version__, '|', torch.cuda.get_device_name(0), '|', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GiB | bf16 ok', bool((x * 2).sum() == 8))"

echo "== Unit tests"
"$PY" -m unittest discover -s tests -t .
[ "$missing" = 0 ] && echo "Setup done. Next: bash cloud/run.sh (see cloud/GUIDE.zh.md, step 6)."
