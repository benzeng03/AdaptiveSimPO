# AdaptiveSimPO — 项目总结（供后续 agent 与协作者阅读）

> STATS 425（大模型课）final project。**每次修改代码、跑完实验或得出新结论后，都要同步更新本文件**（尤其是“实验记录”和“状态 / 待办”两节），并写明日期。

最后更新：2026-11-20

## 1. 研究问题

固定 SimPO 对所有偏好对使用同一个目标 margin γ。本项目检验：在**模型、训练数据、shuffle 顺序、LoRA 初始化和优化预算都相同**的条件下，按样本难度分配 margin（**零阈值 erf**：只给模型当前排错的样本额外 margin，大小按困难百分位确定），能否比**调过参的固定 margin**（γ∈{0, 0.25, 0.5, 0.75, 1}，在 dev 集上选）得到更好的生成回答。

理论推导的原始讨论在 [discussion.txt](discussion.txt)（中文），完整理论现在写在 [README.md](README.md) 的 Method 一节。**README.md 用英文写，不提前序项目，也不提 discussion.txt，每次改动后同步更新。**

## 2. 背景论文

- **SimPO**（Meng, Xia, Chen, NeurIPS 2024，本地 `../SimPO.pdf`，代码 `../SimPO`）：不使用 reference model，奖励取长度归一化的对数似然 `r = β/|y| · log π(y|x)`，损失为 `-log σ(r⁺ - r⁻ - γ)`，γ 是固定的目标奖励 margin。论文的网格：β∈{2, 2.5}，γ∈[0.3, 1.6]，lr∈{3e-7…1e-6}，有效 batch 128，1 epoch。
- **AMaPO**（Deng, Feng, Lei, AAAI 2026，arXiv:2511.09385，代码 `../AMaPO`）：指出 DPO/SimPO 存在“过拟合–欠拟合困境”：已排对的样本仍拿到多余梯度，排错的样本矫正信号不足。它提出逐样本自适应 margin。已发布的 trainer（`../AMaPO/scripts/amapo_trainer.py`）实际使用的公式（Δ 为长度归一化对数比，μ、s 为 batch 均值和**无偏**标准差）是
  `γ_i = β · exp( (μ−Δ_i)/s · μ ) · 1[Δ_i < μ]`，没有上限。
  它与本项目的均值阈值方案同样只给低于均值的样本 margin，但用指数缩放，且在阈值处有跳变（γ 从 0 跳到 β）。本项目曾把它实现为对照 `--margin amapo`，已于 2026-11-19 按用户要求删除，现在只作为相关工作。

## 3. 方法（代码中的统一记号）

奖励差 `r_i = β/|y^w|·log π_θ(y^w|x) − β/|y^l|·log π_θ(y^l|x)`（记号与 SimPO 论文一致：y^w 为胜出回答，y^l 为落败回答），r>0 表示排对。损失 `L_i = softplus(γ_i − r_i)`，γ_i 计算后 stop-grad，因此 `∂L_i/∂r_i = −sigmoid(γ_i − r_i)`。γ=0 的样本仍参与训练。

| 变体名 | 公式 | CLI |
|---|---|---|
| `fixed-g<γ>` | γ_i = γ | `--margin fixed --gamma G` |
| `erf-zero-g<γ_max>` | γ_max·erf((−r_i)₊ / (√2 s_B)) | `--margin adaptive --gamma-max M` |

- s_B 默认为 batch 总体标准差（除以 B，`--std-correction 0`）；std=0 时用 eps。
- 2026-11-19 按用户要求删除了均值阈值 erf、均值阈值 tanh 和 AMaPO 三个选项，代码中只保留以上两种规则。
- 正态假设：R~N(μ,σ²) 时，要求 margin 与困难百分位成正比，就唯一确定了 erf 映射；均值阈值下 E[γ]=γ_max/4。在零阈值下，激活比例为 Φ(−μ/σ)，只有 μ=0 时才回到 γ_max/4。
- batch size 只有 8，所以每一步都记录实际的 `active`、`gamma_mean`，以及用该 batch 的 (r̄, s) 算出的正态参照 `ref_active`、`ref_gamma_mean`（数值积分）。

## 4. 代码结构

| 文件 | 作用 |
|---|---|
| `adaptive_simpo/margins.py` | `compute_margins`、`simpo_loss`、`normal_reference` |
| `adaptive_simpo/modeling.py` | 套 chat 模板（关闭 thinking）和截断，计算长度归一化 logp（只对回答位置分块经过 lm_head），加载模型与 LoRA；`adapter_key` 把 `.` 替换掉，因为 PEFT 的模块名不能含点 |
| `train.py` | 训练单个变体，输出 `metrics.jsonl`、`margin_stats.json`、`training_args.json` 和 LoRA 权重 |
| `score_pairs.py` | 在 64 个 dev 对上计算 acc / logloss / 平均 r，按 acc→logloss→γ 选出固定基线；`--adapters` 限定要打分的变体 |
| `generate.py` | 多个 adapter 在同一组 AlpacaEval 提示上采样（每组用相同的种子），按组缓存 |
| `judge.py` | PairRM 双向判定，比较各变体 vs 基线和 vs GPT-4 参考，给出 bootstrap 95% CI |
| `summarize.py` | 汇总为 `RUN/results.md` |
| `run_experiment.ps1` | 一键完成全流程（`-LearningRate`、`-Epochs`，run 目录名会记录 lr 和 epoch）；所有 Python 调用都经过 `Invoke-Python`（stderr 转成文本，只按退出码判断成败）。生成阶段只包括 dev 选出的固定基线和全部 erf-zero 变体 |
| `tests/` | 公式和梯度的单元测试；tiny Qwen3 在 CPU 上的端到端冒烟测试 |
| `data/`、`third_party/llm_blender/` | 数据（训练 256 对 / dev 64 对 / AlpacaEval 参考答案）；PairRM 模型定义 |
| `models/`、`papers/`、`.venv/`、`outputs/` | 本地文件，不入库：Qwen3-0.6B 和 PairRM 权重、SimPO.pdf、Python 环境（3.14，torch 2.12.1+cu130，transformers 5.23，peft 0.21.3，见 `requirements.txt`）、实验输出 |

**所有依赖都在本目录内**（2026-11-19 已从上级目录复制），默认路径指向 `models/` 和 `.venv/`。

## 5. 运行

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -t . -v
.\run_experiment.ps1 -WhatIf
.\run_experiment.ps1
# 后台运行并写日志（agent 用这种方式）：
#   powershell -NoProfile -ExecutionPolicy Bypass -File run_experiment.ps1 > outputs\experiment-<time>.log 2>&1
```

- 不要用 `*>&1` 包裹脚本：在 PS 5.1 下，Python 的 stderr 会变成终止错误。
- 在 PS 5.1 下用 `-File` 启动时，param 默认值里的 `$PSScriptRoot` 为空，所以默认路径要在脚本正文里解析。

默认设置：Qwen3-0.6B + LoRA（r=16，q/v），bf16，β=2，lr 5e-5 常数，batch 8，1 epoch，共 32 步，seed 42，prompt≤192 / 总长≤384 token。生成：64 个 AlpacaEval 提示，temperature 0.9，top-p 1，最多 4096 个新 token。默认变体：`fixed:0,0.25,0.5,0.75,1`、`erf-zero:2`。

## 6. 实验记录

### 前序项目 RobustSimPO（`../RobustSimPO`，PairRM，64 个提示）
- EMA median/IQR + tanh，bs2：动态 vs 固定 WR 54.69%（CI 42.97–65.63%）；bs8：46.88%（36.72–57.81%）。
- 零阈值 erf（γ_max=2）vs 固定 γ=0.5，bs8（run `qwen-uf256-bs8-20261119-221553-049`，vLLM 生成）：20 胜 / 13 平 / 31 负，WR 41.41%（CI 30.47–52.34%）；两者对 GPT-4 参考的 WR 分别约为 5.5% 和 4.7%。
- **教训**：在 dev 偏好对上，固定 γ∈{0.29, 0.5, 1} 的准确率全部是 33/64，准确率无法区分各候选。因此本项目在准确率打平时用 dev logloss 决胜。

### AdaptiveSimPO
- 2026-11-19：完成代码，测试通过。第一次正式运行用的是旧的 8 个变体，被用户中途停止，结果作废（旧目录 `outputs/run-bs8-seed42-20261119-231419` 保留未删）。
- **2026-11-19 正式运行** `outputs/run-bs8-seed42-20261119-233225`（日志 `outputs/experiment-20261119-233224.log`）：
  - dev：所有变体（包括未训练的 base）准确率都是 33/64；按 logloss 决胜选中 `fixed-g0`（0.7195；erf-zero-g2 为 0.7199，base 为 0.7161，训练后的模型 logloss 全部比 base 略高）。
  - 生成（PairRM，64 个提示）：erf-zero-g2 vs fixed-g0 为 23 胜 / 15 平 / 26 负，WR 47.66%（CI 36.72–58.59%），**没有显著差异**。两者对 GPT-4 参考的 WR 都是 3.91%；平均长度 276 / 268 token；没有回答被 4096 上限截断；64 对回答中有 10 对完全相同。
  - **关键诊断**：每个 adapter 在 64 个 dev 对上的排序方向都和 base 模型完全一致。32 步 LoRA（lr 5e-5）几乎没有改变策略，所以这次比较的统计功效很低，固定 γ 的选择只取决于约 0.004 的 logloss 噪声。
  - 正态假设检验（erf-zero-g2，32 步）：实际激活比例比正态参照平均高 2.0 个百分点（逐步 SD 10.4）；实际平均 γ 比参照平均低 0.081（0.302 vs 0.383）。batch std 平均为 2.34，偶尔出现 std≈11 的离群 batch，这与重尾分布抬高 s_B、压低 u 的解释一致（未检验）。
- **2026-11-20 第二次运行（进行中）**：lr 1e-4，2 epoch（64 步），其余设置不变。目录 `outputs/run-bs8-lr0.0001-ep2-seed42-20261120-000613`，日志 `outputs/experiment-lr1e-4-ep2-20261120-000612.log`。结果待补。

### 数据来源（2026-11-20 核实）
- SimPO 的 Base 设置用 `HuggingFaceH4/ultrafeedback_binarized`：`train_prefs` 用于训练，`test_prefs` 用于评估（官方配置 `dataset_splits: [train_prefs, test_prefs]`）。Instruct 设置用 on-policy 数据 `princeton-nlp/mistral-instruct-ultrafeedback`、`princeton-nlp/llama3-ultrafeedback`（由 PairRM 标注）；v0.2 用 `princeton-nlp/llama3-ultrafeedback-armorm`。
- 官方两个分片（61,135 / 2,000 对）已从 `../SemanticAMaPO` 复制到 `data/ultrafeedback_binarized/`（不入库，train 有 226 MB）。
- **训练并没有用全量数据**：目前所有实验只用 256 对的子集（约占 `train_prefs` 的 0.4%）。要用全量的 61,135 对，需要写一个把 parquet 转成 JSONL 的脚本。按现在的速度（约 2.1 秒/步，bs8，RTX 4060），1 个 epoch 约 7,640 步，每个变体大约 4.5 小时。
- 已核实：`data/uf_original_train.jsonl` 的 256 对和 `uf_original_test.jsonl` 的 64 对，都是对应官方分片的精确子集（prompt 去掉了首尾空白）。Instruct 用的 on-policy 数据集本地没有，下载需要用户同意。

## 7. 状态 / 待办

- [x] 第一次正式结果已写回第 6 节和 README（2026-11-19）。
- [ ] **优先**：加大训练强度，让策略真正发生变化（更高 lr、多个 epoch、更多数据），再比较 margin 规则。可以用“dev 排序是否与 base 不同、logloss 是否下降”作为训练是否生效的检查。
- [ ] 64 个提示的 CI 宽约 ±11 个百分点：考虑增加提示数（`-NumPrompts`）或多跑几个种子（`-Seed`）。
- [ ] 检查实际激活比例和平均 γ 与正态参照的偏差（`metrics.jsonl`）。
- [ ] 可选：扫描 γ_max（例如 1、2、4），或按梯度权重上限选 γ_max=4·arctanh(δ)。
- [ ] 注意：dev 集只用于选择固定 γ，不要用 AlpacaEval 的结果调参。

## 8. 给 agent 的约定

- 代码注释和 CLI 用英文；本文件和面向用户的总结用中文。
- `outputs/` 不入库（见 `.gitignore`）。实验结论要摘要写进本文件。
- 修改 margin 公式时，必须同时更新 `tests/test_margins.py`、第 3 节的表格和 README 的 Method 一节。
- README.md 保持全英文，不提前序项目 RobustSimPO，不提 discussion.txt，不讨论 tanh 和归一化方式的选择（2026-11-20 用户要求）。奖励写成 β/|y|·log π_θ(y|x)，softplus 和 sigmoid 都写出表达式。记号用 y^w / y^l（与 SimPO 论文一致）。README 有一节 “Advantage: no per-setting γ search”：引用 SimPO 的 Table 8（四个设置的 γ 分别为 1.6、0.3、1.0、1.4，网格为 {0.3, 0.5, 1.0, 1.2, 1.4, 1.6}）和 §4.3（margin 过大时性能下降）；同时说明 γ_max 仍是超参数，只是固定为 2、不调。
