#!/usr/bin/env bash
# Train, score and generate on a Linux GPU machine: the cloud counterpart of run_experiment.ps1.
# GPT judging uses the local Codex app, so it is not run here: download the run directory
# (the script packs it as outputs/<run>.tar.gz) and judge locally (cloud/GUIDE.zh.md, step 9).
#
#   bash cloud/run.sh                                     # fixed:0,1,2 + calibrated, 384 tokens
#   bash cloud/run.sh --variants "fixed:2 calibrated" --no-length-limit --alpaca 805 --arena 500
#   bash cloud/run.sh --full --lr "3e-7 1e-6 3e-6 1e-5" ...  # full fine-tuning, every variant at 4 lrs
#   bash cloud/run.sh --run-dir outputs/run-X --skip-train --alpaca 805   # reuse trained models
#   bash cloud/run.sh ... --shutdown                      # power the VM off at the end
#   bash cloud/run.sh --run-dir outputs/run-X <same options>   # resume after an interruption
#
# Options (defaults = the reported setup):
#   --variants "SPEC ..."   space-separated fixed:<gamma> | calibrated[-logistic][-ema:<decay>]   ["fixed:0 fixed:1 fixed:2 calibrated"]
#                           (calibrated-ema:0.9 = train.py --k-ema 0.9, named calibrated-ema0.9;
#                            calibrated-logistic = --k-family logistic; calibrated-logistic-ema:0.9 both)
#   --lr "LR ..."           one or more learning rates; with several, every variant is trained at each
#                           and named <variant>-lr<LR>, e.g. fixed-g2-lr1e-05            [1e-4]
#   --full                  full fine-tuning instead of LoRA (fp32 master weights, see train.py)
#   --fp32-master MODE      full fine-tuning: cpu (master weights + Adam moments in RAM, ~1 s/step slower) | gpu (peak 20.8 of 22 GiB on an L4)  [cpu]
#   --run-dir DIR           [outputs/run-<full->bs<B>-lr<LR|grid>-ep<E>-seed<S>-L<len>-<time>]
#   --epochs --batch-size --seed --beta                     [1, 16, 42, 2]
#   --max-length N --max-prompt N                           [384, 192]
#   --no-length-limit       same as --max-length 8192 --max-prompt 4096 (longer than every pair)
#   --truncation MODE       always (cut every prompt to --max-prompt) | simpo (cut only pairs over --max-length, as SimPO)  [always]
#   --simpo-length          SimPO's Mistral-7B-Base setting: --max-length 1024 --max-prompt 512 --truncation simpo
#   --alpaca N              AlpacaEval prompts to generate, 0 = none; 805 = full set   [128]
#   --arena N               Arena-Hard v0.1 prompts, 0 = none; 500 = full set          [0]
#   --group N --max-new-tokens N                            [16, 4096]
#   --baseline NAME         fixed variant to generate against [best fixed on dev+test]
#   --checkpoint-every N    training checkpoint interval in steps (a rerun resumes from it) [32]
#   --skip-train --skip-gen --shutdown
#   --train-data --dev-data --test-data FILE                [data/uf_train4096 / uf_dev500 / uf_test1486]
# Generation: the baseline and the calibrated variant (with several lrs, the calibrated one with
# the best dev+test accuracy). Full-model weights stay on the VM: the packed archive leaves them out.
# Env: PY (interpreter, default .venv/bin/python), EXTRA_TRAIN_ARGS (e.g. "--max-steps 2" for a smoke test).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"

VARIANTS="fixed:0 fixed:1 fixed:2 calibrated"
RUN_DIR=""; LRS=0.0001; EPOCHS=1; BS=16; SEED=42; BETA=2; FULL=0; MASTER=cpu
MAX_LEN=384; MAX_PROMPT=192; TRUNC=always
ALPACA=128; ARENA=0; GROUP=16; MAX_NEW=4096; BASELINE=""; CKPT_EVERY=32
SKIP_TRAIN=0; SKIP_GEN=0; SHUTDOWN=0
TRAIN_DATA=data/uf_train4096.jsonl; DEV_DATA=data/uf_dev500.jsonl; TEST_DATA=data/uf_test1486.jsonl

usage() { sed -n '2,34p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --variants) VARIANTS="$2"; shift ;;
    --run-dir) RUN_DIR="$2"; shift ;;
    --lr) LRS="$2"; shift ;;
    --full) FULL=1 ;;
    --fp32-master) MASTER="$2"; shift ;;
    --epochs) EPOCHS="$2"; shift ;;
    --batch-size) BS="$2"; shift ;;
    --seed) SEED="$2"; shift ;;
    --beta) BETA="$2"; shift ;;
    --max-length) MAX_LEN="$2"; shift ;;
    --max-prompt) MAX_PROMPT="$2"; shift ;;
    --no-length-limit) MAX_LEN=8192; MAX_PROMPT=4096 ;;
    --truncation) TRUNC="$2"; shift ;;
    --simpo-length) MAX_LEN=1024; MAX_PROMPT=512; TRUNC=simpo ;;
    --alpaca) ALPACA="$2"; shift ;;
    --arena) ARENA="$2"; shift ;;
    --group) GROUP="$2"; shift ;;
    --max-new-tokens) MAX_NEW="$2"; shift ;;
    --baseline) BASELINE="$2"; shift ;;
    --checkpoint-every) CKPT_EVERY="$2"; shift ;;
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
case "$MASTER" in gpu|cpu) ;; *) echo "--fp32-master must be gpu or cpu" >&2; exit 2 ;; esac
[ -x "$PY" ] || [ -f "$PY" ] || { echo "missing interpreter $PY (run cloud/setup.sh)" >&2; exit 1; }
for f in "$TRAIN_DATA" "$DEV_DATA" "$TEST_DATA"; do
  [ -f "$f" ] || { echo "missing data file $f (cloud/GUIDE.zh.md, step 4)" >&2; exit 1; }
done
ARENA_DATA=data/arena-hard-v0.1/question.jsonl
if [ "$ARENA" -gt 0 ] && [ "$SKIP_GEN" = 0 ] && [ ! -f "$ARENA_DATA" ]; then
  echo "missing $ARENA_DATA (needed for --arena)" >&2; exit 1
fi
read -r -a lrs <<< "$LRS"
[ ${#lrs[@]} -gt 0 ] || { echo "--lr needs at least one learning rate" >&2; exit 2; }
if [ -z "$RUN_DIR" ]; then
  [ "$SKIP_TRAIN" = 0 ] || { echo "--skip-train needs --run-dir" >&2; exit 2; }
  lrtag="lr${lrs[0]}"; [ ${#lrs[@]} -gt 1 ] && lrtag=lrgrid
  mode=""; [ "$FULL" = 1 ] && mode=full-
  RUN_DIR="outputs/run-${mode}bs$BS-$lrtag-ep$EPOCHS-seed$SEED-L$MAX_LEN$([ "$TRUNC" = simpo ] && echo -simpo || true)-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

names=(); train_args=()
for lr in "${lrs[@]}"; do
  suffix=""; [ ${#lrs[@]} -gt 1 ] && suffix="-lr$(printf '%g' "$lr")"
  for spec in $VARIANTS; do
    case "$spec" in
      calibrated) names+=("calibrated$suffix"); train_args+=("--lr $lr --margin calibrated") ;;
      calibrated-ema:*) b=$(printf '%g' "${spec#calibrated-ema:}"); names+=("calibrated-ema$b$suffix")
               train_args+=("--lr $lr --margin calibrated --k-ema $b") ;;
      calibrated-logistic) names+=("calibrated-logistic$suffix")
               train_args+=("--lr $lr --margin calibrated --k-family logistic") ;;
      calibrated-logistic-ema:*) b=$(printf '%g' "${spec#calibrated-logistic-ema:}")
               names+=("calibrated-logistic-ema$b$suffix")
               train_args+=("--lr $lr --margin calibrated --k-family logistic --k-ema $b") ;;
      fixed:*) g=$(printf '%g' "${spec#fixed:}"); names+=("fixed-g$g$suffix")
               train_args+=("--lr $lr --margin fixed --gamma $g") ;;
      *) echo "unknown variant $spec (use fixed:<gamma> or calibrated[-logistic][-ema:<decay>])" >&2; exit 2 ;;
    esac
  done
done
model_args=""
[ "$FULL" = 1 ] && model_args="--no-lora --fp32-master $MASTER"
echo "[run] $(date '+%F %T')  run dir $RUN_DIR"
echo "[run] ${#names[@]} variants ${names[*]}"
echo "[run] $([ "$FULL" = 1 ] && echo "full fine-tuning (fp32 masters on $MASTER)" || echo LoRA) | lr ${lrs[*]}, epochs $EPOCHS, batch $BS, seed $SEED, beta $BETA, max length $MAX_LEN, max prompt $MAX_PROMPT, truncation $TRUNC"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
"$PY" -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('[run] torch', torch.__version__, torch.cuda.get_device_name(0))"

# ---- train. A rerun skips variants that already have margin_stats.json and continues an
# interrupted one from its last checkpoint (train.py saves one every --checkpoint-every steps).
if [ "$SKIP_TRAIN" = 0 ]; then
  for i in "${!names[@]}"; do
    name=${names[$i]}; out="$RUN_DIR/$name"
    if [ -f "$out/margin_stats.json" ]; then echo "[run] skip trained $name"; continue; fi
    echo "=== $(date '+%F %T') training $name ($((i + 1))/${#names[@]})"
    # shellcheck disable=SC2086
    "$PY" -u train.py --data "$TRAIN_DATA" --batch-size "$BS" --seed "$SEED" --beta "$BETA" \
      --epochs "$EPOCHS" --max-length "$MAX_LEN" --max-prompt "$MAX_PROMPT" --truncation "$TRUNC" ${train_args[$i]} $model_args \
      --checkpoint-every "$CKPT_EVERY" --resume ${EXTRA_TRAIN_ARGS:-} --output "$out"
  done
fi

# ---- score preference pairs (lengths come from each model's training_args.json)
echo "=== $(date '+%F %T') scoring"
"$PY" -u score_pairs.py "$RUN_DIR" --split dev --data "$DEV_DATA" --adapters "${names[@]}"
"$PY" -u score_pairs.py "$RUN_DIR" --split test --data "$TEST_DATA" --adapters "${names[@]}"
"$PY" -u summarize.py "$RUN_DIR"

gen_names=()
fixed=(); for n in "${names[@]}"; do [[ $n == fixed-g* ]] && fixed+=("$n"); done
calibrated=(); for n in "${names[@]}"; do [[ $n == calibrated* ]] && calibrated+=("$n"); done
if [ ${#fixed[@]} -gt 0 ]; then
  # "test" in reports = dev + test merged (1986 pairs); the fixed baseline is the best one there.
  DEVTEST="$RUN_DIR/devtest1986_scores.json"
  "$PY" research/devtest_scores.py "$RUN_DIR" --baseline "${fixed[0]}" --dev-data "$DEV_DATA" \
    --test-data "$TEST_DATA" --out "$DEVTEST" > /dev/null
  read -r best best_cal < <("$PY" - "$DEVTEST" <<'EOF'
import json, re, sys
rows = json.load(open(sys.argv[1]))["results"]
gamma = lambda n: float(re.match(r"fixed-g([0-9.]+)", n).group(1))
fixed = sorted((r for r in rows if r["adapter"].startswith("fixed-g")),
               key=lambda r: (-r["correct"], gamma(r["adapter"]), r["adapter"]))
cal = sorted((r for r in rows if r["adapter"].startswith("calibrated")),
             key=lambda r: (-r["correct"], r["adapter"]))
print(fixed[0]["adapter"], cal[0]["adapter"] if cal else "-")
EOF
)
  BASELINE=${BASELINE:-$best}
  echo "=== dev+test accuracy (baseline $BASELINE; best fixed on dev+test: $best; best calibrated: $best_cal)"
  "$PY" research/devtest_scores.py "$RUN_DIR" --baseline "$BASELINE" --dev-data "$DEV_DATA" \
    --test-data "$TEST_DATA" --out "$DEVTEST"
  gen_names+=("$BASELINE")
  if [ ${#calibrated[@]} -gt 1 ]; then gen_names+=("$best_cal"); else gen_names+=("${calibrated[@]}"); fi
else
  gen_names=("${calibrated[@]}")
fi

# ---- generate: the baseline and the calibrated variant answer the same prompts
if [ "$SKIP_GEN" = 0 ] && [ ${#gen_names[@]} -gt 0 ] && { [ "$ALPACA" -gt 0 ] || [ "$ARENA" -gt 0 ]; }; then
  gen=(); for n in "${gen_names[@]}"; do gen+=(--adapter "$n=$RUN_DIR/$n"); done
  tag=""; [ "$GROUP" != 8 ] && tag="-g$GROUP"
  common=(--train-data "$TRAIN_DATA" --seed "$SEED" --group "$GROUP" --max-new-tokens "$MAX_NEW")
  if [ "$ALPACA" -gt 0 ]; then
    echo "=== $(date '+%F %T') AlpacaEval generation ($ALPACA prompts: ${gen_names[*]})"
    "$PY" -u generate.py "${gen[@]}" "${common[@]}" --num-prompts "$ALPACA" \
      --output "$RUN_DIR/generations-ae$ALPACA-seed$SEED$tag"
  fi
  if [ "$ARENA" -gt 0 ]; then
    echo "=== $(date '+%F %T') Arena-Hard generation ($ARENA prompts: ${gen_names[*]})"
    "$PY" -u generate.py "${gen[@]}" "${common[@]}" --num-prompts "$ARENA" --eval-data "$ARENA_DATA" \
      --output "$RUN_DIR/generations-ah$ARENA-seed$SEED$tag"
  fi
fi

# Full-model weights (model*.safetensors, ~1.2 GB each) stay on the VM; LoRA adapters are packed.
pack="outputs/$(basename "$RUN_DIR").tar.gz"
tar -czf "$pack" --exclude='model*.safetensors' --exclude='checkpoint*' \
  -C "$(dirname "$RUN_DIR")" "$(basename "$RUN_DIR")"
echo "[run] $(date '+%F %T') done. Results: $RUN_DIR/results.md, ${DEVTEST:-} | download $pack ($(du -h "$pack" | cut -f1))"
