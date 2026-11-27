#!/usr/bin/env bash
# Build SimPO-style on-policy preference data on the GPU VM (onpolicy/decode.py, onpolicy/annotate.py):
# Qwen3-0.6B answers every prompt of the train / dev / test files once per seed (13 21 42 79 100;
# vLLM, at most 4096 new tokens, Qwen3 model-card sampling: temperature 0.7, top-p 0.8, top-k 20;
# SimPO used temperature 0.8, top-p 0.95), prompts with identical answers under
# every seed are dropped, RLHFlow/ArmoRM-Llama3-8B-v0.1 scores each answer, and the highest / lowest
# scoring answers become chosen / rejected. Writes data/uf_onpolicy_{train4096,test2000}.jsonl
# (the test set is every test_prefs prompt, data/uf_test2000.jsonl; there is no dev split).
#
#   bash cloud/onpolicy.sh            # a rerun skips finished seeds and already-scored prompts
#
# vLLM and the reward model run in their own environment, .venv-gen (Python 3.12): vLLM pins its own
# torch and transformers, and ArmoRM's custom model code (trust_remote_code) targets transformers 4.x.
# Env: VLLM_SPEC (pip spec for vllm, default "vllm"), SPLITS (default "train4096 test2000").
set -euo pipefail
cd "$(dirname "$0")/.."
GEN_PY=.venv-gen/bin/python
RM=models/ArmoRM-Llama3-8B-v0.1
SPLITS="${SPLITS:-train4096 test2000}"

if [ ! -x "$GEN_PY" ]; then
  echo "=== $(date '+%F %T') creating .venv-gen with ${VLLM_SPEC:-vllm}"
  uv venv --python 3.12 .venv-gen
  uv pip install --python "$GEN_PY" "${VLLM_SPEC:-vllm}"
fi
"$GEN_PY" -c "import vllm, transformers, torch; print('[onpolicy] vllm', vllm.__version__, '| transformers', transformers.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"
if [ ! -f "$RM/config.json" ]; then
  echo "=== $(date '+%F %T') downloading RLHFlow/ArmoRM-Llama3-8B-v0.1 to $RM"
  "$GEN_PY" -c "from huggingface_hub import snapshot_download; snapshot_download('RLHFlow/ArmoRM-Llama3-8B-v0.1', local_dir='$RM')"
fi

for split in $SPLITS; do
  echo "=== $(date '+%F %T') decoding $split"
  "$GEN_PY" -u onpolicy/decode.py --data "data/uf_$split.jsonl" --output-dir "data/onpolicy/$split" \
    --temperature 0.7 --top-p 0.8 --top-k 20
done
for split in $SPLITS; do
  echo "=== $(date '+%F %T') scoring $split with ArmoRM"
  "$GEN_PY" -u onpolicy/annotate.py --input-dir "data/onpolicy/$split" --reward-model "$RM" \
    --output "data/uf_onpolicy_$split.jsonl"
done
echo "=== $(date '+%F %T') done"
wc -l data/uf_onpolicy_*.jsonl
