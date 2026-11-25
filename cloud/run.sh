#!/usr/bin/env bash
# Train, score and generate on a Linux GPU machine: the cloud counterpart of run_experiment.ps1.
# GPT judging uses the local Codex app, so it is not run here: download the run directory
# (the script packs it as outputs/<run>.tar.gz) and judge locally (cloud/GUIDE.zh.md, step 9).
#
#   bash cloud/run.sh                                     # fixed:0,1,2 + calibrated, 384 tokens
#   bash cloud/run.sh --variants "fixed:2 calibrated" --no-length-limit --alpaca 805 --arena 500
#   bash cloud/run.sh --run-dir outputs/run-X --skip-train --alpaca 805   # reuse trained adapters
#   bash cloud/run.sh ... --shutdown                      # power the VM off at the end
#
# Options (defaults = the reported setup):
#   --variants "SPEC ..."   space-separated fixed:<gamma> | calibrated   ["fixed:0 fixed:1 fixed:2 calibrated"]
#   --run-dir DIR           [outputs/run-bs<B>-lr<LR>-ep<E>-seed<S>-L<len>-<time>]
#   --lr --epochs --batch-size --seed --beta                [1e-4, 1, 16, 42, 2]
#   --max-length N --max-prompt N                           [384, 192]
#   --no-length-limit       same as --max-length 8192 --max-prompt 8192 (longer than every pair)
#   --alpaca N              AlpacaEval prompts to generate, 0 = none; 805 = full set   [128]
#   --arena N               Arena-Hard v0.1 prompts, 0 = none; 500 = full set          [0]
#   --group N --max-new-tokens N                            [16, 4096]
#   --baseline NAME         fixed variant to generate against [best fixed on dev+test]
#   --skip-train --skip-gen --shutdown
#   --train-data --dev-data --test-data FILE                [data/uf_train4096 / uf_dev500 / uf_test1486]
# Env: PY (interpreter, default .venv/bin/python), EXTRA_TRAIN_ARGS (e.g. "--max-steps 2" for a smoke test).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

VARIANTS="fixed:0 fixed:1 fixed:2 calibrated"
RUN_DIR=""; LR=0.0001; EPOCHS=1; BS=16; SEED=42; BETA=2
MAX_LEN=384; MAX_PROMPT=192
ALPACA=128; ARENA=0; GROUP=16; MAX_NEW=4096; BASELINE=""
SKIP_TRAIN=0; SKIP_GEN=0; SHUTDOWN=0
TRAIN_DATA=data/uf_train4096.jsonl; DEV_DATA=data/uf_dev500.jsonl; TEST_DATA=data/uf_test1486.jsonl

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --variants) VARIANTS="$2"; shift ;;
    --run-dir) RUN_DIR="$2"; shift ;;
    --lr) LR="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --batch-size) BS="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
    --beta) BETA="$2"; shift ;;
    --max-length) MAX_LEN="$2"; shift ;;
    --max-prompt) MAX_PROMPT="$2"; shift ;;
    --no-length-limit) MAX_LEN=8192; MAX_PROMPT=8192 ;;
    --alpaca) ALPACA="$2"; shift ;;
    --arena) ARENA="$2"; shift ;;
    --group) GROUP="$2"; shift ;;
    --max-new-tokens) MAX_NEW="$2"; shift ;;
    --baseline) BASELINE="$2"; shift ;;
    --skip-train) SKIP_TRAIN=1 ;;
    --skip-gen) SKIP_GEN=1 ;;
    --shutdown) SHUTDOWN=1 ;;
    --train-data) TRAIN_DATA="$2"; shift ;;
    --dev-data) DEV_DATA="$2"; shift ;;
    --test-data) TEST_DATA="$2"; shift ;;
    -h|--help) usage ;;
    *) echo "unknown option: $1" >&2; usage 2 ;;
  esac
  shift
done

if [ "$SHUTDOWN" = 1 ]; then
  # Runs on success and on failure, so an unattended VM never keeps billing for GPU time.
  trap 'echo "[run] powering off in 120 s (Ctrl-C to cancel)"; sleep 120; sudo shutdown -h now' EXIT
fi
[ -x "$PY" ] || [ -f "$PY" ] || { echo "missing interpreter $PY (run cloud/setup.sh)" >&2; exit 1; }
for f in "$TRAIN_DATA" "$DEV_DATA" "$TEST_DATA"; do
  [ -f "$f" ] || { echo "missing data file $f (cloud/GUIDE.zh.md, step 4)" >&2; exit 1; }
done
ARENA_DATA=data/arena-hard-v0.1/question.jsonl
if [ "$ARENA" -gt 0 ] && [ "$SKIP_GEN" = 0 ] && [ ! -f "$ARENA_DATA" ]; then
  echo "missing $ARENA_DATA (needed for --arena)" >&2; exit 1
fi
if [ -z "$RUN_DIR" ]; then
  [ "$SKIP_TRAIN" = 0 ] || { echo "--skip-train needs --run-dir" >&2; exit 2; }
  RUN_DIR="outputs/run-bs$BS-lr$LR-ep$EPOCHS-seed$SEED-L$MAX_LEN-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

names=(); train_args=()
for spec in $VARIANTS; do
  case "$spec" in
    calibrated) names+=(calibrated); train_args+=("--margin calibrated") ;;
    fixed:*) g=$(printf '%g' "${spec#fixed:}"); names+=("fixed-g$g"); train_args+=("--margin fixed --gamma $g") ;;
    *) echo "unknown variant $spec (use fixed:<gamma> or calibrated)" >&2; exit 2 ;;
  esac
done
echo "[run] $(date '+%F %T')  run dir $RUN_DIR"
echo "[run] variants ${names[*]} | lr $LR, epochs $EPOCHS, batch $BS, seed $SEED, beta $BETA, max length $MAX_LEN, max prompt $MAX_PROMPT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[run] torch', torch.__version__, torch.cuda.get_device_name(0))"

# ---- train (variants that already have margin_stats.json are skipped, so a rerun resumes)
if [ "$SKIP_TRAIN" = 0 ]; then
  for i in "${!names[@]}"; do
    name=${names[$i]}; out="$RUN_DIR/$name"
    if [ -f "$out/margin_stats.json" ]; then echo "[run] skip trained $name"; continue; fi
    rm -rf "$out"
    echo "=== $(date '+%F %T') training $name"
    # shellcheck disable=SC2086
    "$PY" -u train.py --data "$TRAIN_DATA" --batch-size "$BS" --seed "$SEED" --lr "$LR" --beta "$BETA" \
      --epochs "$EPOCHS" --max-length "$MAX_LEN" --max-prompt "$MAX_PROMPT" ${train_args[$i]} \
      ${EXTRA_TRAIN_ARGS:-} --output "$out"
  done
fi

# ---- score preference pairs (lengths come from each adapter's training_args.json)
echo "=== $(date '+%F %T') scoring"
"$PY" -u score_pairs.py "$RUN_DIR" --split dev --data "$DEV_DATA" --adapters "${names[@]}"
"$PY" -u score_pairs.py "$RUN_DIR" --split test --data "$TEST_DATA" --adapters "${names[@]}"
"$PY" -u summarize.py "$RUN_DIR"

fixed=(); for n in "${names[@]}"; do [[ $n == fixed-g* ]] && fixed+=("$n"); done
if [ ${#fixed[@]} -gt 0 ]; then
  # "test" in reports = dev + test merged (1986 pairs); the fixed baseline is the best one there.
  DEVTEST="$RUN_DIR/devtest1986_scores.json"
  "$PY" research/devtest_scores.py "$RUN_DIR" --baseline "${fixed[0]}" --dev-data "$DEV_DATA" \
    --test-data "$TEST_DATA" --out "$DEVTEST" > /dev/null
  best=$("$PY" - "$DEVTEST" <<'EOF'
import json, sys
rows = [r for r in json.load(open(sys.argv[1]))["results"] if r["adapter"].startswith("fixed-g")]
rows.sort(key=lambda r: (-r["correct"], float(r["adapter"][7:])))
print(rows[0]["adapter"])
EOF
)
  BASELINE=${BASELINE:-$best}
  echo "=== dev+test accuracy (baseline $BASELINE; best fixed on dev+test: $best)"
  "$PY" research/devtest_scores.py "$RUN_DIR" --baseline "$BASELINE" --dev-data "$DEV_DATA" \
    --test-data "$TEST_DATA" --out "$DEVTEST"
fi

# ---- generate: the baseline and every non-fixed variant answer the same prompts
if [ "$SKIP_GEN" = 0 ] && { [ "$ALPACA" -gt 0 ] || [ "$ARENA" -gt 0 ]; }; then
  gen=()
  for n in "${names[@]}"; do
    if [[ $n != fixed-g* || $n == "$BASELINE" ]]; then gen+=(--adapter "$n=$RUN_DIR/$n"); fi
  done
  tag=""; [ "$GROUP" != 8 ] && tag="-g$GROUP"
  common=(--train-data "$TRAIN_DATA" --seed "$SEED" --group "$GROUP" --max-new-tokens "$MAX_NEW")
  if [ "$ALPACA" -gt 0 ]; then
    echo "=== $(date '+%F %T') AlpacaEval generation ($ALPACA prompts)"
    "$PY" -u generate.py "${gen[@]}" "${common[@]}" --num-prompts "$ALPACA" \
      --output "$RUN_DIR/generations-ae$ALPACA-seed$SEED$tag"
  fi
  if [ "$ARENA" -gt 0 ]; then
    echo "=== $(date '+%F %T') Arena-Hard generation ($ARENA prompts)"
    "$PY" -u generate.py "${gen[@]}" "${common[@]}" --num-prompts "$ARENA" --eval-data "$ARENA_DATA" \
      --output "$RUN_DIR/generations-ah$ARENA-seed$SEED$tag"
  fi
fi

pack="outputs/$(basename "$RUN_DIR").tar.gz"
tar -czf "$pack" -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
echo "[run] $(date '+%F %T') done. Results: $RUN_DIR/results.md, ${DEVTEST:-} | download $pack ($(du -h "$pack" | cut -f1))"
