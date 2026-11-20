# AdaptiveSimPO

AdaptiveSimPO tests whether SimPO generates better answers when each preference pair gets its own target margin based on how hard the pair currently is. The comparison holds everything else fixed: the same model, training data, shuffling order, LoRA initialisation and optimisation budget. The baseline is a tuned fixed-margin SimPO.

The theoretical derivation (in Chinese) is in [`discussion.txt`](discussion.txt). This README summarises it and documents the code.

## Method

### Reward gap and margin

For prompt $x_i$, chosen answer $y_i^+$ and rejected answer $y_i^-$, let $\bar\ell_\theta(y\mid x)$ be the mean log-probability of the answer tokens. The reward gap is

$$
r_i=\beta\left[\bar\ell_\theta(y_i^+\mid x_i)-\bar\ell_\theta(y_i^-\mid x_i)\right].
$$

$r_i>0$ means the pair is ranked correctly and $r_i<0$ means it is misranked. The per-pair SimPO loss is

$$
L_i=\operatorname{softplus}(\gamma_i-r_i).
$$

$\gamma_i$ is detached after it is computed, so

$$
\frac{\partial L_i}{\partial r_i}=-\operatorname{sigmoid}(\gamma_i-r_i).
$$

At the same $r_i$, a larger $\gamma_i$ puts more gradient weight on raising that pair's reward gap. A pair with $\gamma_i=0$ still trains. The hypothesis under test: giving extra margin to harder pairs works better than giving every pair the same margin.

### Margin rules

| Variant | Rule | CLI |
|---|---|---|
| `fixed-g<γ>` | $\gamma_i=\gamma$ (SimPO) | `--margin fixed --gamma γ` |
| `erf-zero-g<γ_max>` | $\gamma_i=\gamma_{\max}\operatorname{erf}\!\left(\dfrac{(-r_i)_+}{\sqrt2\,s_B}\right)$ | `--margin adaptive --gamma-max γ_max` |

Here $(a)_+=\max(a,0)$ and $s_B$ is the standard deviation of the reward gaps in the current batch (divisor $B$ by default, `--std-correction 1` for $B-1$; `--eps` replaces a zero std). Only misranked pairs get a positive margin. The margin rises with the size of the error and never exceeds $\gamma_{\max}$.

### The normal assumption behind the erf mapping

Assume the reward gaps within one training stage are approximately normal, $R\sim\mathcal N(\mu,\sigma^2)$. Standardising one side of the distribution gives the difficulty $u=(t-R)_+/\sigma$ for a hardness threshold $t$. Define $Z=(t-R)/\sigma$. When $t=\mu$, $Z\sim\mathcal N(0,1)$ and $Z>0$ is the hard half. Require the margin to be proportional to a pair's percentile within the hard half: a pair harder than 80% of hard pairs gets 80% of the maximum margin. That percentile is

$$
P(Z\le u\mid Z>0)=\frac{\Phi(u)-1/2}{1/2}=2\Phi(u)-1=\operatorname{erf}\!\left(\frac{u}{\sqrt2}\right),
$$

so under this requirement erf is the unique mapping. For example, a pair one standard deviation below the threshold gets $0.683\,\gamma_{\max}$. Compared with the earlier heuristic $\tanh(u)$:

| Position below the threshold | erf (percentile) | tanh |
|---|---:|---:|
| 0 | 0 | 0 |
| $0.5\sigma$ | 0.383 | 0.462 |
| $1\sigma$ | 0.683 | 0.762 |
| $2\sigma$ | 0.954 | 0.964 |

The two mappings agree on very hard pairs and differ mainly at moderate difficulty.

Consequences used in this project:

- **Choice of normaliser.** Under normality the natural scale is the standard deviation. A median/IQR normaliser differs only by a constant: for a normal population the median is $\mu$ and $\mathrm{IQR}\approx1.349\,\sigma$. In small batches the two estimators give different values; the code uses the batch mean and std.
- **Expected margin.** If the threshold equals the mean and the parameters are known, half the pairs get zero margin and the other half have uniformly distributed percentiles, so $\mathbb E[\gamma_i]=\gamma_{\max}/4$. For example, $\gamma_{\max}=0.8$ gives an average extra margin of 0.2. If you care about the maximum increase in gradient weight instead, $\gamma_{\max}=4\operatorname{arctanh}(\delta)$ controls that directly.
- **Zero threshold.** This project uses $t=0$, so only misranked pairs are hard. Under normality the expected fraction of pairs with a nonzero margin is $P(R<0)=\Phi(-\mu/\sigma)$. The erf value is still the half-normal percentile of the error $(-r_i)/\sigma$, and it equals the percentile within the hard half exactly when $\mu=0$, where $\mathbb E[\gamma_i]=\gamma_{\max}/4$ again.

These rest on two separate assumptions: the reward gaps are approximately normal, and the margin should be proportional to the hardness percentile. Neither implies that generation quality improves; that has to be measured. Training estimates $(\mu,\sigma)$ from a batch of only 8 pairs. So every step logs the observed fraction of nonzero margins and mean margin, alongside the values the normal model predicts from that batch's mean and std (`ref_active`, `ref_gamma_mean`). These logs show how far the batch estimates deviate from the population reference.

## Experiment

| Setting | Value |
|---|---|
| Policy | Qwen3-0.6B, LoRA r=16 (α=32) on `q_proj`/`v_proj`, BF16, gradient checkpointing |
| Training data | 256 UltraFeedback preference pairs (`data/uf_original_train.jsonl`) |
| Optimisation | AdamW, lr 5e-5 constant, β=2, batch 8 pairs, 1 epoch (32 updates), seed 42 |
| Lengths | prompt ≤192 tokens (keeps the end), prompt + answer ≤384 tokens (keeps the start); EOS counts as an answer token |
| Variants | fixed γ ∈ {0, 0.25, 0.5, 0.75, 1} and zero-threshold erf with γ_max = 2 |
| Baseline selection | 64 held-out preference pairs (`data/uf_original_test.jsonl`): highest pairwise accuracy ($r>0$), ties broken by lower mean $\operatorname{softplus}(-r)$, then smaller γ |
| Generation | 64 AlpacaEval prompts sampled with seed 42 (exact training-prompt matches excluded), Qwen chat template with thinking disabled, temperature 0.9, top-p 1, ≤4096 new tokens. Every policy uses the same sampling seed per prompt group |
| Judge | PairRM, run in both answer orders: a win needs both orders to agree, anything else is a tie. Win rate $=(W+0.5T)/N$ with a 95% prompt-bootstrap interval. Reported head to head against the dev-selected fixed baseline and against the stored GPT-4 reference answers |

The development pairs are used only to choose the fixed γ; generation results are never used for tuning. PairRM win rates on 64 prompts are a local proxy, not official AlpacaEval 2 scores.

## Results

Run `outputs/run-bs8-seed42-20261119-233225` (2026-11-19, default settings above; full table in its `results.md`).

**Development pairs (baseline selection).**

| Variant | Train mean γ | Normal-ref mean γ | Train active | Normal-ref active | Dev accuracy | Dev log-loss |
|---|---:|---:|---:|---:|---:|---:|
| base (untrained) | — | — | — | — | 33/64 | 0.7161 |
| fixed-g0 | 0 | 0 | 0% | 0% | 33/64 | **0.7195** |
| fixed-g0.25 | 0.25 | 0.25 | 100% | 100% | 33/64 | 0.7232 |
| fixed-g0.5 | 0.5 | 0.5 | 100% | 100% | 33/64 | 0.7217 |
| fixed-g0.75 | 0.75 | 0.75 | 100% | 100% | 33/64 | 0.7228 |
| fixed-g1 | 1 | 1 | 100% | 100% | 33/64 | 0.7223 |
| erf-zero-g2 | 0.302 | 0.383 | 41.8% | 39.8% | 33/64 | 0.7199 |

All fixed margins tie on accuracy, so the log-loss tie-break selects `fixed-g0` as the baseline.

**Generation (PairRM, 64 AlpacaEval prompts).**

| Comparison | Wins / ties / losses | Win rate (95% CI) |
|---|---:|---:|
| erf-zero-g2 vs fixed-g0 | 23 / 15 / 26 | 47.66% (36.72%–58.59%) |
| fixed-g0 vs GPT-4 reference | 2 / 1 / 61 | 3.91% |
| erf-zero-g2 vs GPT-4 reference | 2 / 1 / 61 | 3.91% |

Mean answer length was 268 tokens for fixed-g0 and 276 for erf-zero-g2. No answer hit the 4096-token limit, and 10 of the 64 answer pairs were identical.

**Reading the results.**

- **No detectable difference.** The head-to-head interval contains 50%, so this run shows no difference between the adaptive and the tuned fixed margin.
- **Training barely moved the policy.** On every development pair, each trained adapter ranks the chosen and rejected answers exactly as the untrained base model does. Every trained adapter also has a slightly higher dev log-loss than the base model. The choice of fixed γ therefore rests on log-loss differences of about 0.004, and with 32 LoRA updates at lr 5e-5 the comparison has little power.
- **Normal assumption check.** Averaged over the 32 steps, the observed fraction of nonzero margins is within 2 points of the normal prediction (step-level SD 10 points). The mean margin is 0.08 lower than predicted (0.302 vs 0.383). The batch std averages 2.34 and some batches contain large outliers (std up to about 11). This is consistent with heavy tails inflating $s_B$ and shrinking the standardised errors, but this run does not test that explanation.

## Repository layout

| Path | Contents |
|---|---|
| `adaptive_simpo/margins.py` | Margin rules (`compute_margins`), SimPO loss (`simpo_loss`), normal reference (`normal_reference`) |
| `adaptive_simpo/modeling.py` | Prompt templating, truncation, length-normalised answer log-probabilities (answer positions only, chunked through `lm_head`), model/adapter loading |
| `train.py` | Trains one variant; writes LoRA weights, `metrics.jsonl` (per step), `margin_stats.json`, `training_args.json` |
| `score_pairs.py` | Scores adapters and the base model on the dev pairs; writes `dev_scores.{json,csv}` and `selection.json` |
| `generate.py` | Samples answers for several adapters on the same prompts; caches each prompt group |
| `judge.py` | PairRM judging; writes `judgments.json` and `win_rates.json` |
| `summarize.py` | Collects everything into `RUN/results.md` |
| `run_experiment.ps1` | Runs the whole pipeline end to end |
| `tests/` | Unit tests for the margin formulas and gradients; CPU end-to-end smoke test with a tiny random Qwen3 |
| `data/` | Training pairs, development pairs, AlpacaEval instructions with GPT-4 reference answers |
| `third_party/llm_blender/` | PairRM model definition and its license |
| `discussion.txt` | Theoretical derivation (Chinese) |
| `models/`, `papers/`, `.venv/`, `outputs/` | Local only (git-ignored): Qwen3-0.6B and PairRM checkpoints, the SimPO paper, the Python environment, experiment outputs |

## Setup

On a machine without the local folders, download [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) to `models/Qwen3-0.6B` and [llm-blender/PairRM](https://huggingface.co/llm-blender/PairRM) to `models/PairRM`, then create the environment (Windows, Python 3.14, NVIDIA GPU with CUDA 13):

```powershell
py -3.14 -m venv .venv
.\.venv\Scripts\python.exe -m pip install "torch==2.12.1+cu130" --index-url https://download.pytorch.org/whl/cu130
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Usage

```powershell
# Tests (formula checks + CPU pipeline smoke test)
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v

# Full experiment: train -> dev selection -> generate -> judge -> results.md
.\run_experiment.ps1 -WhatIf
.\run_experiment.ps1

# Other variant sets, more prompts, or evaluating an existing run
.\run_experiment.ps1 -Variants "fixed:0.5","erf-zero:1","erf-zero:2" -NumPrompts 128
.\run_experiment.ps1 -RunDir outputs\run-bs8-seed42-YYYYMMDD-HHMMSS -EvalOnly
```

Re-running with an existing `-RunDir` trains only the missing variants and reuses cached generations. The individual scripts can be run directly; see `--help` on each, for example:

```powershell
.\.venv\Scripts\python.exe train.py --margin adaptive --gamma-max 2 --output outputs\custom\erf-zero-g2
```

Custom data is UTF-8 JSONL with one pair per line: `{"prompt": "...", "chosen": "...", "rejected": "..."}`. A string prompt is wrapped as a user message in the chat template; a message list is used as is.

## References

- Yu Meng, Mengzhou Xia, Danqi Chen. *SimPO: Simple Preference Optimization with a Reference-Free Reward.* NeurIPS 2024. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
- Ruibo Deng, Duanyu Feng, Wenqiang Lei. *AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment.* AAAI 2026. [arXiv:2511.09385](https://arxiv.org/abs/2511.09385). Related work: an instance-wise adaptive margin that also shifts learning effort toward misranked pairs, using Z-normalisation and exponential scaling.
- Dongfu Jiang, Xiang Ren, Bill Yuchen Lin. *LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion.* ACL 2023 (PairRM).
