# AdaptiveSimPO — 项目总结（供后续 agent 与协作者阅读）

> STATS 425（大模型课）final project。**每次修改代码、跑完实验或得出新结论后，都要同步更新本文件**（尤其是“实验记录”和“状态 / 待办”两节），并写明日期。

最后更新：2026-11-19

## 1. 研究问题

固定 SimPO 对所有偏好对使用同一个目标 margin γ。本项目检验：在**模型、训练数据、shuffle 顺序、LoRA 初始化和优化预算都相同**的条件下，按样本难度分配 margin，能否比**调过参的固定 margin**得到更好的生成回答。两个设计选择：

1. **怎样算困难样本**：零阈值 t=0（只看模型当前排错的样本）还是均值阈值 t=μ（奖励差低于当前 batch 均值的样本）。
2. **怎样把难度映射成 margin**：erf（困难半区百分位，有理论推导）还是 tanh（早期启发式）。

理论推导见 [discussion.txt](discussion.txt)。

## 2. 背景论文

- **SimPO**（Meng, Xia, Chen, NeurIPS 2024，本地 `../SimPO.pdf`，代码 `../SimPO`）：不使用 reference model，奖励取长度归一化的对数似然 `r = β/|y| · log π(y|x)`，损失为 `-log σ(r⁺ - r⁻ - γ)`，γ 是固定的目标奖励 margin。论文的网格：β∈{2, 2.5}，γ∈[0.3, 1.6]，lr∈{3e-7…1e-6}，有效 batch 128，1 epoch。
- **AMaPO**（Deng, Feng, Lei, AAAI 2026，arXiv:2511.09385，代码 `../AMaPO`）：指出 DPO/SimPO 存在“过拟合–欠拟合困境”：已排对的样本仍拿到多余梯度，排错的样本矫正信号不足。它提出逐样本自适应 margin。已发布的 trainer（`../AMaPO/scripts/amapo_trainer.py`）实际使用的公式（Δ 为长度归一化对数比，μ、s 为 batch 均值和**无偏**标准差）是
  `γ_i = β · exp( (μ−Δ_i)/s · μ ) · 1[Δ_i < μ]`，没有上限。
  它与本项目的均值阈值方案同样只给低于均值的样本 margin，但用指数缩放，且在阈值处有跳变（γ 从 0 跳到 β）。本项目把它实现为对照 `--margin amapo`。

## 3. 方法（代码中的统一记号）

奖励差 `r_i = β[ℓ̄(y⁺|x) − ℓ̄(y⁻|x)]`，r>0 表示排对。损失 `L_i = softplus(γ_i − r_i)`，γ_i 计算后 stop-grad，因此 `∂L_i/∂r_i = −sigmoid(γ_i − r_i)`：γ 越大，该样本的梯度权重越大。γ=0 的样本仍参与训练。

| 规则 | 公式 | CLI |
|---|---|---|
| 固定 SimPO | γ_i = γ | `--margin fixed --gamma G` |
| 零阈值 erf（RobustSimPO 已跑） | γ_max·erf((−r_i)₊ / (√2 s_B)) | `--margin adaptive --threshold zero --mapping erf` |
| **均值阈值 erf（理论候选）** | γ_max·erf((r̄_B − r_i)₊ / (√2 s_B)) | `--margin adaptive --threshold mean --mapping erf` |
| 均值阈值 tanh（映射消融） | γ_max·tanh((r̄_B − r_i)₊ / s_B) | `--mapping tanh` |
| AMaPO 对照 | 见第 2 节 | `--margin amapo` |

- s_B 默认是 batch 总体标准差（除以 B，`--std-correction 0`）；std=0 时用 eps。
- 正态参照：若 R~N(μ,σ²)，均值阈值 erf 的激活比例为 1/2，E[γ]=γ_max/4。因此默认 γ_max=2 对应的平均 margin **0.5**，与固定 γ=0.5 的平均 margin 匹配（fixed-g0.5 是“平均 margin 匹配”对照，生成评测总会包含它）。
- batch size 只有 8，所以每一步都记录实际的 `active`、`gamma_mean`，以及用该 batch 的 (r̄, s) 算出的正态参照 `ref_active`、`ref_gamma_mean`。参照值通过数值积分计算，零阈值和 AMaPO 也适用。

## 4. 代码结构

| 文件 | 作用 |
|---|---|
| `adaptive_simpo/margins.py` | 各 margin 规则 `compute_margins`、损失 `simpo_loss`、正态参照 `normal_reference` |
| `adaptive_simpo/modeling.py` | 对 prompt 套 chat 模板（关闭 thinking），对回答做截断，计算长度归一化 logp（只对回答位置分块经过 lm_head），加载 base 模型与 LoRA；`adapter_key` 把 `.` 替换掉，因为 PEFT 的模块名不能含点 |
| `train.py` | 训练单个变体，输出 `metrics.jsonl`（逐步记录）、`margin_stats.json`（汇总）、`training_args.json` 和 LoRA 权重 |
| `score_pairs.py` | 在 64 个 dev 偏好对上计算 acc / logloss / 平均 r，按 acc→logloss→γ 的顺序选出最佳固定 γ，写 `selection.json` |
| `generate.py` | 多个 adapter 在同一组 AlpacaEval 提示上采样（每组用相同的种子），按组缓存，可断点续跑 |
| `judge.py` | PairRM 双向判定：两个顺序结论一致才算胜负，否则算平；比较各变体 vs 基线和 vs GPT-4 参考答案，给出 bootstrap 95% CI、平均 token 数和未结束回答数 |
| `summarize.py` | 汇总为 `RUN/results.md` |
| `run_experiment.ps1` | 一键完成训练→dev 选择→生成→评判→汇总；支持 `-WhatIf`、`-Variants`、`-SkipEval`、`-EvalOnly -RunDir` |
| `tests/` | `test_margins.py`（公式、梯度、AMaPO 对齐、正态参照）、`test_pipeline.py`（tiny Qwen3 在 CPU 上的端到端冒烟测试） |
| `data/` | 从 RobustSimPO 复制的 UltraFeedback：训练 256 对、dev 64 对；以及 AlpacaEval GPT-4 参考答案 |
| `third_party/llm_blender/` | PairRM 模型定义（含 LICENSE） |

外部依赖（不在仓库内）：`../Qwen3-0.6B`（策略模型）、`../PairRM`（评判模型）、Python 环境 `../RobustSimPO/.venv`（Python 3.14，torch 2.12.1+cu130，transformers 5.23，peft 0.21.3）。路径都可以通过参数覆盖。

## 5. 运行

```powershell
# 单元测试 + CPU 冒烟测试
..\RobustSimPO\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
# 预览，然后跑完整实验（默认 8 个变体）
.\run_experiment.ps1 -WhatIf
.\run_experiment.ps1
# 只评测已训练好的 run
.\run_experiment.ps1 -RunDir outputs\run-... -EvalOnly
```

默认设置（与 RobustSimPO 保持一致，便于对比）：Qwen3-0.6B + LoRA（r=16，q/v），bf16，β=2，lr 5e-5 常数，batch 8，1 epoch，共 32 步，seed 42，prompt≤192 / 总长≤384 token。生成：64 个 AlpacaEval 提示（排除与训练提示完全相同的），temperature 0.9，top-p 1，最多 4096 个新 token。默认变体：`fixed:0,0.25,0.5,1`、`erf-zero:2`、`erf-mean:2`、`tanh-mean:2`、`amapo`。

## 6. 实验记录

### 前序项目 RobustSimPO（`../RobustSimPO`，PairRM，64 个提示）
- EMA median/IQR + tanh，bs2：动态 vs 固定 WR 54.69%（CI 42.97–65.63%）；bs8：46.88%（36.72–57.81%）。
- 零阈值 erf（γ_max=2）vs 固定 γ=0.5，bs8（run `qwen-uf256-bs8-20261119-221553-049`，vLLM 生成）：20 胜 / 13 平 / 31 负，WR 41.41%（CI 30.47–52.34%）；两者对 GPT-4 参考的 WR 分别约为 5.5% 和 4.7%。
- **教训**：在 dev 偏好对上，固定 γ∈{0.29, 0.5, 1} 的准确率全部是 33/64，准确率无法区分各候选。因此本项目在准确率打平时用 dev logloss 决胜。

### AdaptiveSimPO
- 2026-11-19：完成代码，单元测试和 CPU 冒烟测试通过；GPU 冒烟测试（2 步、2 个提示、PairRM 在 cuda 上）跑通。尚无正式结果。

## 7. 状态 / 待办

- [ ] 跑 `run_experiment.ps1` 默认实验，把 `results.md` 的主要数字写回第 6 节。
- [ ] 64 个提示的 CI 宽约 ±11 个百分点：考虑增加提示数（`-NumPrompts`）或多跑几个种子（`-Seed`）。
- [ ] 检查实际激活比例和平均 γ 与正态参照的偏差（`metrics.jsonl`），以此检验小 batch 下的正态假设。
- [ ] 可选：用 γ_max=0.8 让平均 margin 为 0.2；也可以按梯度权重上限选 γ_max=4·arctanh(δ)。
- [ ] 注意：dev 集只用于选择固定 γ，不要用 AlpacaEval 的结果调参。

## 8. 给 agent 的约定

- 代码注释和 CLI 用英文；本文件和面向用户的总结用中文。
- `outputs/` 不入库（见 `.gitignore`）。实验结论要摘要写进本文件。
- 修改 margin 公式时，必须同时更新 `tests/test_margins.py` 和第 3 节的表格。
