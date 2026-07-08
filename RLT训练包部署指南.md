# RLT 训练包部署指南

> 给 Mickeyy 的说明：本机（工控机）**无可用 GPU**，仅用于 Franka 遥操与数据采集；RLT 训练须在**另一台带 NVIDIA 驱动的主机**上完成。

## 1. 背景

| 机器 | 角色 | 能力 |
|------|------|------|
| **本机**（`vla`，`/home/host5010/workspaces/yjr`） | Ray head + Franka2 真机环境 | 遥操采集数据；`nvidia-smi` 不可用，无法本地 GPU 训练 |
| **GPU 训练主机** | 离线 / 在线 RL 训练 | 需 `nvidia-smi` 正常、`torch.cuda.is_available() == True` |

本压缩包来自 `smq&jgy/rlt_project`，实现 Physical Intelligence [RLT (RL Token)](https://www.pi.website/research/rlt) 论文的社区复现。**官方未开源**，此处为基于 openpi 的自研复现代码。

## 2. 压缩包内容

文件名：`rlt_training_package_20260623.tar.gz`（位于 `smq&jgy/` 目录）

```
rlt_project/
├── README.md                          # 项目总览
├── PACKAGE_README.md                  # 包内快速说明（与本文档互补）
└── rlt_reproduce/
    ├── configs/default.yaml           # 超参（对齐论文 Appendix B）
    ├── pyproject.toml                 # Python 包定义
    ├── scripts/
    │   ├── setup_env.sh               # 一键 conda 环境
    │   ├── setup_gpu.sh               # GPU 驱动就绪后安装 CUDA PyTorch
    │   └── download_checkpoint.py     # 下载 openpi VLA checkpoint
    ├── src/rlt/
    │   ├── rl_token/                  # RL Token encoder-decoder（论文 IV-A）
    │   ├── rl/                        # Actor-Critic + TD3 learner（论文 IV-B）
    │   ├── vla/                       # openpi 封装 / MockVLA
    │   ├── sim/                       # CPU mock 环境（无机器人时自测）
    │   └── scripts/
    │       ├── train_rl_token.py      # 阶段一：RL Token 预训练
    │       ├── train_online_rl.py     # 阶段二：在线 RL
    │       ├── smoke_test.py          # 端到端冒烟测试
    │       └── verify_env.py          # 环境检查
    └── third_party/openpi/            # Physical Intelligence VLA 源码
```

**未打入包的内容**（需在 GPU 主机上重新生成）：

- `.git` / `.venv`（减小体积；GPU 主机用 `setup_env.sh` 重建）
- `checkpoints/`、`logs/`、`outputs/`（训练产物）

## 3. 在 GPU 主机上部署

### 3.1 传输压缩包

```bash
# 在本机执行（示例：传到 GPU 服务器）
scp "/home/host5010/workspaces/smq&jgy/rlt_training_package_20260623.tar.gz" \
    user@GPU_HOST:~/rlt_training_package.tar.gz
```

### 3.2 解压

```bash
cd ~
tar -xzf rlt_training_package.tar.gz
cd rlt_project/rlt_reproduce
```

> 目录名含 `&` 时务必加引号：`cd "/path/to/smq&jgy/rlt_project/rlt_reproduce"`

### 3.3 安装环境

```bash
# 需要 conda（Python 3.11）
bash scripts/setup_env.sh
conda activate rlt
python -m rlt.scripts.verify_env
python -m rlt.scripts.smoke_test
```

### 3.4 启用 GPU

```bash
bash scripts/setup_gpu.sh
# 修改 configs/default.yaml 中 device: cuda
python scripts/download_checkpoint.py --config pi05_base
```

### 3.5 训练流程

**阶段一：RL Token 预训练**（在演示数据的 VLA embedding 上训练 encoder-decoder）

```bash
conda activate rlt
cd ~/rlt_project/rlt_reproduce

# 使用 mock VLA（无 checkpoint 时仅验证流程）
python -m rlt.scripts.train_rl_token --config configs/default.yaml

# 有 openpi checkpoint 后，在 default.yaml 中设置：
#   vla.checkpoint: checkpoints/pi05_base
#   device: cuda
```

**阶段二：在线 RL**（TD3 风格 actor-critic，论文 Algorithm 1）

```bash
python -m rlt.scripts.train_online_rl --config configs/default.yaml --episodes 50
```

当前 `train_online_rl.py` 默认使用 **MockPrecisionEnv**；接入真机需替换为机器人接口或与 HIL-SERL gRPC learner/actor 对接。

## 4. 与本机数据采集的协作

本机采集的数据在 `yjr/data/`（如 `first-20`、`second-20`、`third-20` 等），由 ConsVLA/RLinf 的 `collect_data.sh` 产生，格式与 RLT 复现**尚未直接打通**。

建议协作方式：

1. **本机**：继续用 `yjr/ConsVLA/RLinf` 做遥操采集，数据留在 `yjr/data/`。
2. **GPU 主机**：将需要训练的 demo 目录 `rsync`/`scp` 到训练机，例如 `rlt_reproduce/data/demos/`。
3. **后续开发**（在 GPU 主机上）：编写数据加载器，把 RLinf 采集格式转为 RLT `train_rl_token.py` 所需的 VLA embedding 输入。

```bash
# 示例：同步采集数据到 GPU 主机
rsync -avz /home/host5010/workspaces/yjr/data/first-20/ \
    user@GPU_HOST:~/rlt_project/rlt_reproduce/data/demos/first-20/
```

## 5. 关键超参（`configs/default.yaml`）

| 模块 | 参数 | 默认值 | 说明 |
|------|------|--------|------|
| RL Token | `train_steps` | 5000 | 预训练步数 |
| RL Token | `embed_dim` / `token_dim` | 2048 | 对齐 pi0.5 VLA |
| Online RL | `chunk_length` | 10 | 动作块长度 C |
| Online RL | `update_to_data_ratio` | 5 | 每步环境交互后的梯度更新次数 G |
| Online RL | `policy_constraint_beta` | 1.0 | BC 正则 β |
| Online RL | `reference_dropout` | 0.5 | reference action dropout |

螺丝任务等困难场景可将 `actor_hidden` / `critic_hidden` 改为 `[512, 512, 512]`（见论文 Appendix B）。

## 6. 验收清单（GPU 主机）

- [ ] `nvidia-smi` 正常
- [ ] `conda activate rlt` 后 `python -c "import torch; print(torch.cuda.is_available())"` 为 `True`
- [ ] `python -m rlt.scripts.verify_env` 通过
- [ ] `python -m rlt.scripts.smoke_test` 通过
- [ ] `download_checkpoint.py` 成功下载 checkpoint（可选，真实训练需要）
- [ ] `train_rl_token.py` 能保存 `checkpoints/rl_token.pt`

## 7. 与本机 yjr/RLinf 训练的关系

| 项目 | 路径 | 用途 | 本机能否跑 |
|------|------|------|-----------|
| **RLinf 真机 SAC** | `yjr/ConsVLA/RLinf` | Franka 在线 RL（Ray 多机） | 仅 env 节点；GPU actor 在远程 |
| **RLT 复现** | `smq&jgy/rlt_project` | RL Token + TD3 离线/在线 | **不能**；整包需在 GPU 主机运行 |

两套代码相互独立；本压缩包仅包含 **RLT 复现**，不含 RLinf。

## 8. 参考链接

- RLT 论文：https://www.pi.website/research/rlt
- arXiv：https://arxiv.org/html/2604.23073v1
- openpi：https://github.com/Physical-Intelligence/openpi
- 社区复现笔记：https://villekuosmanen.medium.com/research-notes-from-reproducing-rl-token-f375ecfd3c28

---

*生成日期：2026-06-23 | 仅修改于 `smq&jgy` 目录*
