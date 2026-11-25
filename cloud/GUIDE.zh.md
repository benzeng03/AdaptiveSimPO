# 在 Google Cloud 上训练 AdaptiveSimPO

云上负责训练、偏好对打分和生成。GPT 判定要用本机的 Codex 桌面版，所以生成完要把结果下载回本机再判。

整个流程分三处操作：

- **本机**：PowerShell，在仓库根目录下。
- **云主机**：通过 SSH 登录的 Linux 终端。
- **网页控制台**：<https://console.cloud.google.com>。

下面以一块 L4（24 GB 显存）为例。界面上的菜单名可能和这里写的略有不同。

---

## 1. 账号与额度（网页控制台）

1. **确认账号类型。** 免费试用（Free Trial）账号不能挂 GPU。要在 **Billing** 页面点 **Activate full account / 升级**，升级后剩下的试用额度仍然可用。如果你的额度来自课程或教育优惠券，挂在哪个结算账号上，项目就要关联到那个结算账号。
2. **新建一个项目**，比如 `simpo`，并确认它关联到有额度的结算账号。
3. **设置预算提醒。** 在 **Billing → Budgets & alerts** 里设一个金额，比如额度的 50% 和 90%，花到这个数时会发邮件提醒。
4. **启用 Compute Engine API。** 第一次打开 **Compute Engine** 时会提示启用。

## 2. 申请 GPU 配额（网页控制台）

新项目的 GPU 配额通常是 0，不申请就创建不了带 GPU 的虚拟机。

1. 打开 **IAM & Admin → Quotas & System Limits**。
2. 搜索 `GPUs (all regions)`，把上限申请为 **1**。
3. 再搜索 `NVIDIA L4 GPUs`，在你要用的区域（比如 `us-central1`）申请 **1**。如果用 A100，就搜 `NVIDIA A100 GPUs`。
4. 审批可能几分钟就通过，也可能要一两天，结果会发邮件通知。

## 3. 创建虚拟机（网页控制台）

**Compute Engine → VM instances → Create instance**：

| 设置 | 选择 |
|---|---|
| Name | `simpo-l4` |
| Region / Zone | 有 L4 的区域，比如 `us-central1` 下的某个 zone；提示资源不足（`ZONE_RESOURCE_POOL_EXHAUSTED`）就换一个 zone |
| Machine configuration | **GPUs** → GPU type **NVIDIA L4**，数量 1；机型 **g2-standard-8**（8 vCPU、32 GB 内存） |
| Boot disk | 点 **Change** → Operating system 选 **Deep Learning on Linux** → 选 CUDA 版本最新的那个镜像；大小 **100 GB** |
| VM provisioning model | **Standard**；或选 **Spot**，便宜很多，但随时可能被回收，见第 7 步 |

其余保持默认，点 **Create**。

驱动的安装：
- **Deep Learning 镜像**：第一次 SSH 登录时会问是否安装 NVIDIA 驱动，输入 `y`，等几分钟。装完后运行 `nvidia-smi`，能看到 L4 就说明装好了。
- **普通 Ubuntu 镜像**：要按 Google 文档 “Install GPU drivers” 自己安装。

`nvidia-smi` 右上角会显示 `CUDA Version`：
- **13.0 或更高**：可以直接用默认的 PyTorch（cu130）。
- **低于 13.0**：第 5 步要换成对应的 PyTorch 索引，比如 `TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash cloud/setup.sh`。具体有哪些版本，以 pytorch.org 上列出的为准。

## 4. 上传代码和数据

### 连接方式（二选一）

- **浏览器 SSH**：在 VM 列表里点 **SSH** 按钮。窗口右上角有 **Upload file / Download file**，可以用来传文件。
- **本机 Google Cloud CLI**：安装 <https://cloud.google.com/sdk/docs/install>，然后运行 `gcloud init` 登录，并选择项目和 zone。之后可以这样登录和传文件：

```powershell
gcloud compute ssh simpo-l4 --zone us-central1-a
```

### 打包数据（本机）

训练和评测用的 JSONL 不在 git 里，需要手动传上去：

```powershell
tar -czf cloud_data.tgz data/uf_train4096.jsonl data/uf_dev500.jsonl data/uf_test1486.jsonl data/arena-hard-v0.1/question.jsonl
```

```powershell
gcloud compute scp cloud_data.tgz simpo-l4:~ --zone us-central1-a
```

### 取代码并解压数据（云主机）

```bash
git clone https://github.com/benzeng03/AdaptiveSimPO.git
cd AdaptiveSimPO
tar -xzf ~/cloud_data.tgz
```

`cloud/` 目录要先提交并推送到 GitHub，`git clone` 才能拿到。

## 5. 安装环境并测显存（云主机）

```bash
bash cloud/setup.sh
```

这个脚本会做以下几件事，大约 5–10 分钟：

1. 用 uv 装 Python 3.14 和 `requirements.txt` 里的版本，环境放在 `.venv/`。
2. 从 Hugging Face 下载 Qwen3-0.6B，并和本机那份的 sha256 对比。
3. 检查数据文件和 CUDA。
4. 跑单元测试。

如果要放宽或去掉长度限制，先测一下这台机器放不放得下：

```bash
.venv/bin/python cloud/len_probe.py              # LoRA，不限长度
.venv/bin/python cloud/len_probe.py --mode full  # 全参数（只看显存，见文末说明）
```

脚本会对 batch 宽度 448、1108、1624、2234、2507 各跑一步，这几个宽度对应训练集里 batch 宽度的中位数、p90、p99 和最大值，每步输出耗时和显存峰值。

- **全部不 OOM**：可以直接用 `--no-length-limit`。
- **大宽度 OOM**：用 `--max-length` 设一个放得下的上限，或者换显存更大的 GPU。

## 6. 训练（云主机）

一定要在 tmux 里跑，这样 SSH 断开后训练还会继续：

```bash
tmux new -s simpo
```

**第一步：在 384 长度下复现，检查新机器的结果和本机是否接近。**

```bash
bash cloud/run.sh --variants "fixed:2 calibrated" --alpaca 0
```

本机在 dev+test（1986 对）上的准确率是 fixed γ=2 66.06%、calibrated 65.61%。换了 GPU，结果不会逐位一致，相差 1 个百分点左右属于正常。

**第二步：正式实验。** 比如不限长度，跑完整的 γ 网格，然后在完整的 AlpacaEval 和 Arena-Hard 上生成，跑完自动关机：

```bash
bash cloud/run.sh --no-length-limit \
  --variants "fixed:0 fixed:0.5 fixed:1 fixed:1.5 fixed:2 fixed:2.5 calibrated" \
  --alpaca 805 --arena 500 --shutdown
```

全参数微调加学习率网格（见文末说明）：

```bash
bash cloud/run.sh --full --no-length-limit --lr "3e-7 1e-6 3e-6 1e-5" \
  --variants "fixed:0.5 fixed:1 fixed:1.5 fixed:2 calibrated" \
  --alpaca 805 --arena 500 --shutdown
```

`run.sh` 依次做这些事：

1. 逐个训练变体，已训练完的会跳过。
2. 在 dev 和 test 上打分，再合并成 1986 对的结果。
3. 取合并集上最好的 fixed γ 作基线。
4. 基线和 calibrated 生成回答。
5. 把整个 run 目录打包成 `outputs/<run>.tar.gz`。

日志在 `outputs/<run>/run.log`。所有参数可以用 `bash cloud/run.sh --help` 查看。

tmux 的基本操作：
- **离开**：按 `Ctrl-b`，再按 `d`。
- **回来**：

```bash
tmux attach -t simpo
```

## 7. 监控与断点续跑（云主机）

```bash
tail -f outputs/run-*/run.log                       # 实时日志
grep -c . outputs/run-*/calibrated/metrics.jsonl    # 某个变体已经训练了多少步（共 256 步）
nvidia-smi                                          # 显存与 GPU 占用
```

每步耗时可以这样算：先看 `run.log` 里 `=== ... training` 的开始时间，过几分钟再数一下 `metrics.jsonl` 的行数。

如果虚拟机被停了（比如 Spot 被回收，或者手动停机），重新启动后用**同一个 run 目录**再跑一次：

```bash
bash cloud/run.sh --run-dir outputs/<run> <和原来相同的参数>
```

续跑时：
- 已训练完的变体会跳过。
- 正在训练的那个变体从最近一次 checkpoint 接着训（每 32 步存一次，可用 `--checkpoint-every` 改），日志里会出现 `resuming from checkpoint at step N`。最多重训 31 步：384 长度约 2 分钟，不限长度约 8 分钟。
- 打分会整个重跑。
- 已经生成完的组会直接复用。

用 Spot 时，**On VM termination** 一定要选 **Stop**：选 Delete 的话，被回收时磁盘连同 checkpoint 一起删掉。Spot 被回收后不会自动重启，要在 VM 列表里手动点 **Start**，再开 tmux 重跑上面的命令。

## 8. 下载结果（本机）

```powershell
gcloud compute scp <用户名>@simpo-l4:AdaptiveSimPO/outputs/<run>.tar.gz outputs/ --zone us-central1-a
```

```powershell
tar -xzf outputs/<run>.tar.gz -C outputs
```

偏好对的结果已经在 `outputs/<run>/results.md` 和 `devtest1986_scores.json` 里。

adapter 的 `training_args.json` 里记的是云主机上的路径，所以不要在本机对这些 adapter 重新运行 `score_pairs.py`。打分已经在云上做完了。

`<用户名>` 是浏览器 SSH 登录时用的那个 Linux 用户（代码在它的家目录下）；不写的话，gcloud 会用本机的 Windows 用户名登录，找不到文件。Windows 上远程路径不要写 `~`。

## 9. GPT 判定（本机）

以完整的 AlpacaEval 为例。基线名以 `run.log` 里写的为准：

```powershell
.\.venv\Scripts\python.exe research\llm_judge.py outputs\<run>\generations-ae805-seed42-g16 --variant calibrated --baseline fixed-g2
```

```powershell
.\.venv\Scripts\python.exe research\lc_winrate.py outputs\<run>\generations-ae805-seed42-g16\llm-judge-gpt-6.2-sol-medium-calibrated-vs-fixed-g2
```

```powershell
.\.venv\Scripts\python.exe summarize.py outputs\<run>
```

Arena-Hard 同理，把目录换成 `generations-ah500-seed42-g16`。

## 10. 关机与清理（网页控制台）

- **停止计费 GPU**：不用时在 VM 列表里点 **Stop**。停机后 GPU 和 CPU 不再计费，但 100 GB 磁盘照样收费。用了 `--shutdown` 的话，跑完会自动停机。
- **彻底删除**：实验全部做完、结果也下载了以后，点 **Delete** 删掉虚拟机和磁盘。
- **核对花费**：在 **Billing → Reports** 里看实际花了多少。

---

### 说明

- **全参数微调**：`run.sh` 加 `--full`（训练时传 `--no-lora`）。
  - 模型权重仍是 bf16，但 AdamW 更新的是 fp32 主权重，所以 3e-7 这样的小学习率不会被舍入掉。
  - 主权重和 Adam 状态默认放在 CPU 内存里（`--fp32-master cpu`，约 7 GB 内存）。在 L4 上不限长度时，GPU 显存峰值约 13.8 GB，每步只比放在 GPU 上慢约 1 秒；放在 GPU 上（`--fp32-master gpu`）时峰值 20.8 GB，太紧。
  - checkpoint 约 7 GB，每次保存约 23 秒。
  - 每个训练好的模型约 1.1 GB，留在云主机上，打包时不包含。偏好对打分和生成都在云上做完了，所以本机只需要那个小压缩包。
- **学习率网格**：`--lr "3e-7 1e-6 3e-6 1e-5"` 让每个变体在每个学习率下各训一次，名字带后缀，比如 `fixed-g2-lr1e-06`。基线取 1986 对上最好的 fixed（γ 和学习率一起选），calibrated 也取 1986 对上最好的那个学习率，生成只用这两个。
- 改了长度设置以后，fixed γ 基线必须在同一台机器、同样的长度下重训，才能和 calibrated 比较。上面的正式实验命令已经包含了整个网格。
