# RLT Training Package（GPU 主机专用）

本目录为 **RLT (RL Token)** 训练代码包，须在带 NVIDIA GPU 的机器上运行。

## 快速开始

```bash
cd rlt_reproduce
bash scripts/setup_env.sh
conda activate rlt
python -m rlt.scripts.smoke_test
```

GPU 就绪后：

```bash
bash scripts/setup_gpu.sh
# 编辑 configs/default.yaml → device: cuda
python scripts/download_checkpoint.py --config pi05_base
python -m rlt.scripts.train_rl_token
python -m rlt.scripts.train_online_rl --episodes 50
```

## 命令入口

| 命令 | 说明 |
|------|------|
| `python -m rlt.scripts.verify_env` | 检查 Python / torch / 核心模块 |
| `python -m rlt.scripts.smoke_test` | CPU 端到端冒烟（mock VLA + mock env） |
| `python -m rlt.scripts.train_rl_token` | RL Token 预训练 → `checkpoints/rl_token.pt` |
| `python -m rlt.scripts.train_online_rl` | 在线 RL（当前为 mock 环境） |

也可使用 pip 安装后的 CLI：`rlt-verify`、`rlt-smoke`、`rlt-train-token`、`rlt-train-rl`。

## 目录说明

- `src/rlt/rl_token/` — 论文 Section IV-A，Eq. (1)(2)
- `src/rlt/rl/` — TD3 learner、actor-critic、replay buffer，Eq. (3)(5) + Algorithm 1
- `src/rlt/vla/` — openpi 策略加载；无 GPU/checkpoint 时用 `MockVLAWrapper`
- `third_party/openpi/` — VLA 骨干；`setup_env.sh` 会尝试 `uv sync`

完整部署说明见包外同级文档 **`RLT训练包部署指南.md`**。
