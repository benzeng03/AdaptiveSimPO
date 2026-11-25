# 在 Google Cloud 上训练 AdaptiveSimPO

云上负责训练、偏好对打分和生成。GPT 判定要用本机的 Codex 桌面版，所以生成完要把结果下载回本机再判。

整个流程分三处操作：

- **本机**：PowerShell，在仓库根目录下。
- **云主机**：通过 SSH 登录的 Linux 终端。
- **网页控制台**：<https://console.cloud.google.com>。

下面以一块 L4（24 GB 显存）为例。界面上的菜单名可能和这里写的略有不同。命令里的 `us-central1-a` 换成你的虚拟机实际所在的 zone。

---

## 1. 账号与额度（网页控制台）

1. **确认账号类型。** 免费试用（Free Trial）账号不能挂 GPU。要在 **Billing** 页面点 **Activate full account / 升级**，升级后剩下的试用额度仍然可用。如果你的额度来自课程或教育优惠券，挂在哪个结算账号上，项目就要关联到那个结算账号。
2. **新建一个项目**，比如 `simpo`，并确认它关联到有额度的结算账号。
3. **设置预算提醒。** 在 **Billing → Budgets & alerts** 里设一个金额，比如额度的 50% 和 90%，花到这个数时会发邮件提醒。提醒不会自动停止花费。
4. **启用 Compute Engine API。** 第一次打开 **Compute Engine** 时会提示启用。

## 2. 申请 GPU 配额（网页控制台）

新项目的 GPU 配额通常是 0，不申请就创建不了带 GPU 的虚拟机。

1. 打开 **IAM & Admin → Quotas & System Limits**。
2. 搜索 `GPUs (all regions)`，把上限申请为 **1**。
3. 再搜索 `NVIDIA L4 GPUs`，在你要用的区域申请 **1**；用 Spot 的话还要 `Preemptible NVIDIA L4 GPUs`。配额是按区域给的，换区域前先看那个区域的配额是不是 0。
4. 审批可能几分钟就通过，也可能要一两天，结果会发邮件通知。

## 3. 创建虚拟机（网页控制台）

**Compute Engine → VM instances → Create instance**：

| 设置 | 选择 |
|---|---|
| Name | `simpo-l4` |
| Region / Zone | 有 L4 的区域，比如 `us-central1`、`us-east1`、`us-east4`、`us-west1`、`us-west4` 下的某个 zone；提示 `ZONE_RESOURCE_POOL_EXHAUSTED` 或 “currently unavailable” 就换一个 zone |
| Machine configuration | **GPUs** → GPU type **NVIDIA L4**，数量 1；机型 **g2-standard-8**（8 vCPU、32 GB 内存） |
| Boot disk | 点 **Change** → Operating system 选 **Deep Learning on Linux** → 选 CUDA 版本最新的那个镜像；大小 **100 GB** |
| VM provisioning model | **Spot**（便宜约 2/3，但随时可能被回收），**On VM termination 选 Stop**；或 **Standard** |

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
- **本机 Google Cloud CLI**：安装 <https://cloud.google.com/sdk/docs/install>，然后运行 `gcloud init` 登录，并选择项目。之后可以这样登录：

```powershell
gcloud compute ssh <用户名>@simpo-l4 --zone us-central1-a
```

`<用户名>` 是浏览器 SSH 登录时的那个 Linux 用户，代码就放在它的家目录下。不写的话，gcloud 会用本机的 Windows 用户名登录，看不到那份代码。

### 打包数据（本机）

训练和评测用的 JSONL 不在 git 里，需要手动传上去：

```powershell
tar -czf cloud_data.tgz data/uf_train4096.jsonl data/uf_dev500.jsonl data/uf_test1486.jsonl data/arena-hard-v0.1/question.jsonl
```

```powershell
gcloud compute scp cloud_data.tgz <用户名>@simpo-l4:cloud_data.tgz --zone us-central1-a
```

Windows 上远程路径不要写 `~`，不带目录的文件名就是家目录。

### 取代码并解压数据（云主机）

```bash
git clone https://github.com/benzeng03/AdaptiveSimPO.git
cd AdaptiveSimPO
tar -xzf ~/cloud_data.tgz
```

以后代码有更新，在 `~/AdaptiveSimPO` 里运行 `git pull` 即可。

## 5. 安装环境并测显存（云主机）

```bash
bash cloud/setup.sh
```

这个脚本会做以下几件事，大约 5–10 分钟：

1. 用 uv 装 Python 3.14 和 `requirements.txt` 里的版本，环境放在 `.venv/`。
2. 从 Hugging Face 下载 Qwen3-0.6B，并和本机那份的 sha256 对比。
3. 检查数据文件和 CUDA。
4. 跑单元测试。

换了长度设置或训练方式时，可以先测一步训练的耗时和显存（学习率为 0，不会训练）：

```bash
.venv/bin/python cloud/len_probe.py --mode full --max-length 1024 --max-prompt 512 --truncation simpo --widths 448 800 1024
.venv/bin/python cloud/len_probe.py              # LoRA，不限长度
```

每个宽度跑一步，输出耗时和显存峰值，放不下的显示 `OOM`。

## 6. 训练（云主机）

一定要在 tmux 里跑，这样 SSH 断开后训练还会继续：

```bash
tmux new -s simpo
```

```bash
cd ~/AdaptiveSimPO
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**第一步（可选）：在 384 长度下复现，检查新机器的结果和本机是否接近。**

```bash
bash cloud/run.sh --variants "fixed:2 calibrated" --alpaca 0
```

本机在 dev+test（1986 对）上的准确率是 fixed γ=2 66.06%、calibrated 65.61%；L4 上复现为 65.86% 和 65.96%。换了 GPU，结果不会逐位一致，相差 1 个百分点左右属于正常。

**第二步：正式实验。** 全参数微调，SimPO 的长度设置，4 个学习率 × 5 个 margin，然后在完整的 AlpacaEval 和 Arena-Hard 上生成，跑完自动关机：

```bash
bash cloud/run.sh --full --simpo-length --fp32-master gpu --lr "3e-7 1e-6 3e-6 1e-5" \
  --variants "fixed:0.5 fixed:1 fixed:1.5 fixed:2 calibrated" \
  --alpaca 805 --arena 500 --shutdown
```

训练配置见文末的“正式实验的设置”。按 L4 上的实测估计（未跑完整实验）：每个模型训练约 50 分钟，20 个约 16–17 小时；加上打分和生成，总共约 20 小时。

`run.sh` 依次做这些事：

1. 逐个训练变体，已训练完的会跳过，训到一半的从 checkpoint 接着训。
2. 在 dev 和 test 上打分，再合并成 1986 对的结果。
3. 取合并集上最好的 fixed（γ 和学习率一起选）作基线，calibrated 也取合并集上最好的学习率。
4. 这两个模型生成回答。
5. 把 run 目录打包成 `outputs/<run>.tar.gz`（全参数模型的权重不打包）。

日志在 `outputs/<run>/run.log`，开头一行写着 run 目录名，记下来，续跑时要用。所有参数可以用 `bash cloud/run.sh --help` 查看。

tmux 的基本操作：
- **离开**：按 `Ctrl-b`，再按 `d`。注意在 tmux 里按 `Ctrl-c` 会中断程序。
- **回来**：

```bash
tmux attach -t simpo
```

## 7. 监控与断点续跑（云主机）

```bash
tail -f outputs/run-*/run.log                                 # 实时日志
grep -c . outputs/run-*/calibrated-lr1e-06/metrics.jsonl      # 某个变体已经训练了多少步（共 256 步）
grep "=== .* training" outputs/run-*/run.log                  # 每个变体的开始时间和序号
nvidia-smi                                                    # 显存与 GPU 占用
```

如果虚拟机被停了（比如 Spot 被回收，或者手动停机），在 VM 列表里点 **Start**（Spot 不会自动重启），登录后开 tmux，用**同一个 run 目录和相同的参数**再跑一次：

```bash
bash cloud/run.sh --run-dir outputs/<run> <和原来相同的参数>
```

续跑时：
- 已训练完的变体会跳过。
- 正在训练的那个变体从最近一次 checkpoint 接着训（每 32 步存一次，可用 `--checkpoint-every` 改），日志里会出现 `resuming from checkpoint at step N`。最多重训 31 步。参数和 checkpoint 不一致时会报错，不会混着训。
- 打分会整个重跑。
- 已经生成完的组会直接复用。

**On VM termination** 一定要是 **Stop**：选 Delete 的话，被回收时磁盘连同 checkpoint 一起删掉。

### 自动续跑和看门狗

不想手动重启的话，可以让虚拟机开机时自己续跑，再让本机定时检查：

1. **启动脚本**（云主机，每次开机以 root 运行，对每个家目录里有 `~/autoresume.sh` 的用户执行它）。在本机安装：

```powershell
gcloud compute instances add-metadata simpo-l4 --zone us-central1-a --metadata-from-file startup-script=cloud/startup-script.sh
```

2. **`~/autoresume.sh` 和 `~/start_full.sh`**（云主机）：把 `cloud/autoresume.example.sh` 复制成 `~/autoresume.sh`，`RUN` 改成你的 run 目录；`~/start_full.sh` 里写正式实验的 `run.sh` 命令，并把第一个参数作为 `--run-dir` 传进去。开机后它会等 GPU 驱动就绪，没有 tmux 会话 `simpo` 就新开一个续跑；结果已经打包（`<run>.tar.gz` 存在）就什么都不做。记录在 `~/autoresume.log`。

3. **看门狗**（本机）：每 5 分钟查一次虚拟机。被回收（Spot preemption）就重新启动；连续 6 次（30 分钟）都因为没货启动不了，就给磁盘做快照，在 `-Zones` 列表里第一个有货的 zone 用快照新建同名虚拟机（带上启动脚本），新机器上确认 tmux 会话起来以后，删掉旧机器和快照。如果虚拟机是正常停下的（`run.sh --shutdown` 跑完，或者你手动点了 Stop），看门狗就退出。

```powershell
Start-Process powershell -WindowStyle Hidden -ArgumentList '-NoProfile','-ExecutionPolicy','Bypass','-File','cloud\watchdog.ps1'
```

```powershell
Get-Content outputs\watchdog.log -Tail 20
```

看门狗运行时会阻止 Windows 空闲睡眠（退出后恢复），但合上盖子或手动睡眠还是会睡，电脑睡着时它不工作。只查一次、不做任何操作：`cloud\watchdog.ps1 -Once`。停止：在任务管理器里结束对应的 powershell 进程，或者 `Stop-Process -Id (Get-Content outputs\watchdog.pid)`（如果启动时记下了 PID）。

### Spot 起不来时换 zone（本机）

点 Start 时报 “currently unavailable” 说明这个 zone 暂时没有 L4。可以过一会儿再试，也可以用磁盘快照在别的 zone 重建（磁盘里的环境、代码和结果都会带过去）：

```powershell
gcloud compute snapshots create simpo-snap --source-disk <磁盘名> --source-disk-zone <旧 zone>
```

```powershell
gcloud compute instances create simpo-l4 --zone <新 zone> --machine-type g2-standard-8 --accelerator count=1,type=nvidia-l4 --provisioning-model SPOT --instance-termination-action STOP --maintenance-policy TERMINATE --create-disk boot=yes,auto-delete=yes,source-snapshot=simpo-snap,size=100,type=pd-balanced
```

磁盘名用 `gcloud compute disks list` 查。新机器建好、确认数据都在以后，再删掉旧机器和快照：

```powershell
gcloud compute instances delete simpo-l4 --zone <旧 zone>
```

```powershell
gcloud compute snapshots delete simpo-snap
```

## 8. 下载结果（本机）

```powershell
gcloud compute scp <用户名>@simpo-l4:AdaptiveSimPO/outputs/<run>.tar.gz outputs/ --zone us-central1-a
```

```powershell
tar -xzf outputs/<run>.tar.gz -C outputs
```

偏好对的结果已经在 `outputs/<run>/results.md` 和 `devtest1986_scores.json` 里。

模型的 `training_args.json` 里记的是云主机上的路径，全参数模型的权重也不在压缩包里，所以不要在本机对它们重新运行 `score_pairs.py`。打分已经在云上做完了。

## 9. GPT 判定（本机）

基线和 calibrated 的名字以 `run.log` 里 `=== dev+test accuracy (baseline ...; best calibrated: ...)` 那一行为准，比如 `fixed-g1.5-lr3e-06` 和 `calibrated-lr1e-06`。以完整的 AlpacaEval 为例：

```powershell
.\.venv\Scripts\python.exe research\llm_judge.py outputs\<run>\generations-ae805-seed42-g16 --variant <calibrated 名> --baseline <基线名>
```

```powershell
.\.venv\Scripts\python.exe research\lc_winrate.py outputs\<run>\generations-ae805-seed42-g16\llm-judge-gpt-6.2-sol-medium-<calibrated 名>-vs-<基线名>
```

```powershell
.\.venv\Scripts\python.exe summarize.py outputs\<run>
```

Arena-Hard 同理，把目录换成 `generations-ah500-seed42-g16`。

## 10. 关机与清理（网页控制台）

- **停止计费 GPU**：不用时在 VM 列表里点 **Stop**。停机后 GPU 和 CPU 不再计费，但 100 GB 磁盘照样收费。用了 `--shutdown` 的话，跑完会自动停机。
- **彻底删除**：实验全部做完、结果也下载了以后，点 **Delete** 删掉虚拟机和磁盘。全参数模型的权重只在磁盘上，要保留的话先下载。
- **核对花费**：在 **Billing → Reports** 里看实际花了多少。

---

### 正式实验的设置

| 项目 | 设置 |
|---|---|
| 模型 | Qwen3-0.6B，**全参数微调**（596,049,920 个参数），bf16 前向和反向，梯度检查点 |
| 优化器 | AdamW，**fp32 主权重和 fp32 动量**（fused AdamW，放在 GPU 上），每步把主权重写回 bf16 模型；weight decay 0，学习率常数，不做梯度裁剪 |
| 学习率 | **3e-7、1e-6、3e-6、1e-5** |
| 损失 | SimPO，β=2 |
| margin | fixed γ ∈ **{0.5, 1, 1.5, 2}**，加 calibrated（γ_i=(1−k̂)r_i，k̂=2μ/σ² 取当前 batch 的 16 个 gap，下界 0.01） |
| 模型数 | 5 个 margin × 4 个学习率 = **20 个** |
| 数据 | `uf_train4096`，batch 16（一次前向，不累积），1 epoch = 256 步，seed 42（各变体的数据顺序相同） |
| 长度 | **SimPO 的 Mistral-7B-Base 设置**：总长 ≤ 1024，prompt ≤ 512，按 SimPO 的规则截断：只有 prompt 加较长回答超过 1024 时才把 prompt 截到最后 512 个 token，仍然超长再把两个回答都截到 1024−512=512 个 token |
| checkpoint | 每 32 步一次（约 7 GB，约 23 秒；最后一步不存） |
| 实测（L4） | 截断后训练集 3.5% 的对截了 prompt、3.5% 截了回答；batch 宽度平均 924，256 个 batch 里 45 个顶到 1024；宽度 448 / 800 / 1024 时 4.8 / 9.2 / 11.4 秒/步，显存峰值 13.2 GiB |
| 评测 | dev 500 + test 1486 合并成 1986 对，打分用同样的长度和截断；基线取合并集上最好的 fixed（γ 和学习率一起事后选，对基线有利），calibrated 也取最好的学习率 |
| 生成 | 基线和 calibrated；AlpacaEval 805 题（排除与训练集重复的 1 题，实际 804）、Arena-Hard v0.1 500 题；temperature 0.9、top-p 1、最多 4096 个新 token、seed 42、每组 16 题 |
| 判定 | 本机，GPT-6.2-Sol 双向判定 + LC 胜率（第 9 步） |

### 说明

- **全参数微调**（`--full`）：模型权重仍是 bf16，但 AdamW 更新的是 fp32 主权重，所以 3e-7 这样的小学习率不会被舍入掉。主权重和 Adam 状态约 7 GB：
  - `--fp32-master gpu`：放在显存里。SimPO 长度（1024）下峰值 13.2 GiB，L4 放得下，正式实验用这个。不限长度时峰值 20.8 GiB，在 L4 上太紧。
  - `--fp32-master cpu`（`run.sh` 的默认值）：放在内存里，每步慢约 1 秒，不限长度时显存峰值只有 13.8 GiB。
  - 每个训练好的模型约 1.1 GB，留在云主机上。
- **学习率网格**（`--lr` 给多个值）：每个变体在每个学习率下各训一次，名字带后缀，比如 `fixed-g2-lr1e-06`、`calibrated-lr3e-07`。只给一个学习率时名字不带后缀。
- **长度设置**：
  - 默认：总长 ≤ 384、prompt ≤ 192，每个 prompt 都截到最后 192 个 token（`--truncation always`）。这是之前本机实验用的设置。
  - `--simpo-length`：等于 `--max-length 1024 --max-prompt 512 --truncation simpo`。
  - `--no-length-limit`：总长 ≤ 8192、prompt ≤ 4096，数据里没有这么长的，等于不截断。
- 改了长度设置或训练方式以后，fixed γ 基线必须在同一台机器、同样的设置下重训，才能和 calibrated 比较。上面的正式实验命令已经包含了整个网格。
