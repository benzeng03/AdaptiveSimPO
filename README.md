# AdaptiveSimPO

AdaptiveSimPO tests whether SimPO generates better answers when each preference pair gets its own target margin based on how hard the pair currently is. The comparison holds everything else fixed: the same model, training data, shuffling order, LoRA initialisation and optimisation budget. The baseline is a tuned fixed-margin SimPO.

## Method

### SimPO reward gap and loss

For prompt $x_i$, chosen answer $y_i^+$ and rejected answer $y_i^-$, SimPO uses the length-normalised log-likelihood of the policy $\pi_\theta$ as an implicit reward:

$$
\frac{\beta}{|y|}\log\pi_\theta(y\mid x)=\frac{\beta}{|y|}\sum_{k=1}^{|y|}\log\pi_\theta(y_k\mid x,y_{<k}),
$$

where $|y|$ is the number of answer tokens. The reward gap of pair $i$ is

$$
r_i=\frac{\beta}{|y_i^+|}\log\pi_\theta(y_i^+\mid x_i)-\frac{\beta}{|y_i^-|}\log\pi_\theta(y_i^-\mid x_i).
$$

$r_i>0$ means the pair is ranked correctly and $r_i<0$ means it is misranked. With a target margin $\gamma_i$, the per-pair SimPO loss is

$$
L_i=-\log\sigma(r_i-\gamma_i)=\operatorname{softplus}(\gamma_i-r_i),\qquad
\sigma(a)=\frac{1}{1+e^{-a}},\quad \operatorname{softplus}(a)=\log\left(1+e^{a}\right),
$$

and the training objective is the batch mean of $L_i$. Original SimPO uses one constant $\gamma_i=\gamma$ for all pairs.

$\gamma_i$ is detached after it is computed, so

$$
\frac{\partial L_i}{\partial r_i}=-\sigma(\gamma_i-r_i)=-\frac{1}{1+e^{\,r_i-\gamma_i}}.
$$

At the same $r_i$, a larger $\gamma_i$ puts more gradient weight on raising that pair's reward gap. A pair with $\gamma_i=0$ still trains. The hypothesis under test: giving extra margin to harder pairs works better than giving every pair the same margin.

### Margin rules

| Variant | Rule | CLI |
|---|---|---|
| `fixed-g<γ>` | $\gamma_i=\gamma$ (SimPO) | `--margin fixed --gamma γ` |
| `erf-zero-g<γ_max>` | $\gamma_i=\gamma_{\max}\operatorname{erf}\!\left(\dfrac{(-r_i)_+}{\sqrt2\,s_B}\right)$ | `--margin adaptive --gamma-max γ_max` |

Here $(a)_+=\max(a,0)$, $\operatorname{erf}(z)=\frac{2}{\sqrt\pi}\int_0^z e^{-t^2}\,dt$, and $s_B$ is the standard deviation of the reward gaps in the current batch. The divisor is $B$ by default; `--std-correction 1` uses $B-1$, and `--eps` replaces a zero std. Only misranked pairs get a positive margin. The margin rises with the size of the error and never exceeds $\gamma_{\max}$.

### Why erf: a normal model of the reward gaps

Assume the reward gaps within one training stage are approximately normal, $R\sim\mathcal N(\mu,\sigma^2)$, and call a pair hard when its gap falls below a threshold $t$. The one-sided standardised difficulty is

$$
u_i=\frac{(t-r_i)_+}{\sigma}.
$$

The direction is the same for every threshold: $t=0$ gives the numerator $(-r_i)_+$, and $t=\mu$ gives $(\mu-r_i)_+$. Take $t=\mu$ and define $Z=(\mu-R)/\sigma$. Then $Z\sim\mathcal N(0,1)$ and $Z>0$ is the hard half. Require the margin to be proportional to a pair's percentile within the hard half: a pair harder than 80% of the hard pairs gets 80% of the maximum margin. For difficulty $u>0$ that percentile is

$$
P(Z\le u\mid Z>0)=\frac{\Phi(u)-1/2}{1/2}=2\Phi(u)-1=\operatorname{erf}\!\left(\frac{u}{\sqrt2}\right),
$$

where $\Phi$ is the standard normal CDF. Under this requirement erf is the unique mapping:

$$
\gamma_i=\operatorname{stopgrad}\!\left[\gamma_{\max}\operatorname{erf}\!\left(\frac{u_i}{\sqrt2}\right)\right].
$$

For example, a pair one standard deviation below the threshold is harder than about 68.3% of the hard half and gets $0.683\,\gamma_{\max}$; at $2\sigma$ it gets $0.954\,\gamma_{\max}$.

Consequences:

- **Expected margin.** With $t=\mu$ and known parameters, half the pairs get zero margin and the other half have percentiles uniformly distributed on $[0,1]$. So $\mathbb E[\gamma_i]=\tfrac12\cdot\tfrac12\gamma_{\max}=\gamma_{\max}/4$; for example, $\gamma_{\max}=0.8$ gives an average extra margin of 0.2. This gives $\gamma_{\max}$ a direct interpretation.
- **Zero threshold (used here).** This project uses $t=0$, so only misranked pairs count as hard. Under the normal model, the expected fraction of pairs with a nonzero margin is $P(R<0)=\Phi(-\mu/\sigma)$. The erf value is the half-normal percentile of the standardised error $-r_i/\sigma$. It equals the percentile within the hard half exactly when $\mu=0$, where $\mathbb E[\gamma_i]=\gamma_{\max}/4$ again.

These results rest on two separate assumptions: the reward gaps are approximately normal, and the margin should be proportional to the hardness percentile. Neither implies that generation quality improves; that has to be measured. Training estimates $(\mu,\sigma)$ from a batch of only 8 pairs, using the batch mean $\bar r_B$ and std $s_B$. So every step logs the observed fraction of nonzero margins and the mean margin. Alongside them it logs the values the normal model predicts from that batch's $(\bar r_B, s_B)$ (`ref_active`, `ref_gamma_mean`), which shows how far the batch estimates deviate from the population reference.

## Data

SimPO trains its Base-model settings (Mistral-7B-Base, Llama-3-8B-Base) on [HuggingFaceH4/ultrafeedback_binarized](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized), using the `train_prefs` split for training and `test_prefs` for evaluation during training (`dataset_splits: [train_prefs, test_prefs]` in the official training configs). Its Instruct settings use on-policy UltraFeedback variants. In these, the prompts are UltraFeedback's, the responses are sampled from the SFT model and the pairs are ranked by PairRM: [princeton-nlp/mistral-instruct-ultrafeedback](https://huggingface.co/datasets/princeton-nlp/mistral-instruct-ultrafeedback) and [princeton-nlp/llama3-ultrafeedback](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback). The later v0.2 release uses [princeton-nlp/llama3-ultrafeedback-armorm](https://huggingface.co/datasets/princeton-nlp/llama3-ultrafeedback-armorm), annotated by ArmoRM.

| File | Contents |
|---|---|
| `data/ultrafeedback_binarized/train_prefs-00000-of-00001.parquet` | Official `train_prefs` split, 61,135 pairs (local only, 226 MB) |
| `data/ultrafeedback_binarized/test_prefs-00000-of-00001.parquet` | Official `test_prefs` split, 2,000 pairs (local only) |
| `data/uf_original_train.jsonl` | 256 training pairs: an exact subset of `train_prefs` (prompts whitespace-stripped) |
| `data/uf_original_test.jsonl` | 64 development pairs: an exact subset of `test_prefs`, prompt-disjoint from the training pairs |
| `data/alpaca_eval_gpt4_baseline.json` | 805 AlpacaEval instructions with GPT-4 reference answers, used for generation |

The JSONL files keep only the final chosen and rejected assistant messages, as `{"prompt", "chosen", "rejected"}`.

## Experiment

| Setting | Value |
|---|---|
| Policy | Qwen3-0.6B, LoRA r=16 (α=32) on `q_proj`/`v_proj`, BF16, gradient checkpointing |
| Training data | `data/uf_original_train.jsonl` (256 pairs) |
| Optimisation | AdamW with constant lr, β=2, batch 8 pairs (32 updates per epoch), seed 42; lr and epochs per run are listed under Results |
| Lengths | prompt ≤192 tokens (keeps the end), prompt + answer ≤384 tokens (keeps the start); EOS counts as an answer token |
| Variants | fixed γ ∈ {0, 0.25, 0.5, 0.75, 1} and zero-threshold erf with γ_max = 2 |
| Baseline selection | `data/uf_original_test.jsonl` (64 pairs): highest pairwise accuracy ($r>0$), ties broken by lower mean $\operatorname{softplus}(-r)$, then smaller γ |
| Generation | 64 AlpacaEval prompts sampled with seed 42 (exact training-prompt matches excluded), Qwen chat template with thinking disabled, temperature 0.9, top-p 1, ≤4096 new tokens. Every policy uses the same sampling seed per prompt group |
| Judge | PairRM, run in both answer orders: a win needs both orders to agree, anything else is a tie. Win rate $=(W+0.5T)/N$ with a 95% prompt-bootstrap interval. Reported head to head against the dev-selected fixed baseline and against the stored GPT-4 reference answers |

The development pairs are used only to choose the fixed γ; generation results are never used for tuning. PairRM win rates on 64 prompts are a local proxy, not official AlpacaEval 2 scores.

## Results

### Run 1: lr 5e-5, 1 epoch (32 updates)

Run `outputs/run-bs8-seed42-20261119-233225` (2026-11-19; full table in its `results.md`).

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
| `data/` | Training and development pairs, official UltraFeedback splits, AlpacaEval instructions (see Data) |
| `third_party/llm_blender/` | PairRM model definition and its license |
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

`-LearningRate` and `-Epochs` set the optimisation budget (for example `-LearningRate 0.0001 -Epochs 2`); the run directory name records both. Re-running with an existing `-RunDir` trains only the missing variants and reuses cached generations. The individual scripts can be run directly; see `--help` on each, for example:

```powershell
.\.venv\Scripts\python.exe train.py --margin adaptive --gamma-max 2 --output outputs\custom\erf-zero-g2
```

Custom data is UTF-8 JSONL with one pair per line: `{"prompt": "...", "chosen": "...", "rejected": "..."}`. A string prompt is wrapped as a user message in the chat template; a message list is used as is.

## References

- Yu Meng, Mengzhou Xia, Danqi Chen. *SimPO: Simple Preference Optimization with a Reference-Free Reward.* NeurIPS 2024. [arXiv:2405.14734](https://arxiv.org/abs/2405.14734)
- Ruibo Deng, Duanyu Feng, Wenqiang Lei. *AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment.* AAAI 2026. [arXiv:2511.09385](https://arxiv.org/abs/2511.09385). Related work: an instance-wise adaptive margin that also shifts learning effort toward misranked pairs, using Z-normalisation and exponential scaling.
- Dongfu Jiang, Xiang Ren, Bill Yuchen Lin. *LLM-Blender: Ensembling Large Language Models with Pairwise Ranking and Generative Fusion.* ACL 2023 (PairRM).
