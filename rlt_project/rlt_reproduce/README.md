# RLT (RL Token) 复现项目

基于 [Physical Intelligence RLT 论文](https://www.pi.website/research/rlt) 的社区复现。**官方未开源**，本仓库在 [openpi](https://github.com/Physical-Intelligence/openpi) VLA 基础上实现论文中的 RL Token + 在线 TD3 流程。

> **部署位置**：本代码应在 **GPU 训练主机**上运行，不在本工控机上训练。  
> 传输与安装见 `smq&jgy/RLT训练包部署指南.md`。

## 现状

| 组件 | 状态 |
|------|------|
| RL Token encoder-decoder (Eq. 1-2) | 已实现 |
| Actor-Critic + TD3 (Eq. 3-5, Alg. 1) | 已实现 |
| openpi VLA 集成 | 依赖 openpi 安装与 checkpoint |
| 真机四任务 (螺丝/扎带/网线/充电器) | 需机器人 + HIL 接口 |
| GPU 推理/训练 | **在 GPU 主机执行**（本机无驱动） |

## 快速开始（无 GPU，冒烟测试）

```bash
cd rlt_reproduce   # 或解压后的等价路径
bash scripts/setup_env.sh
conda activate rlt
python -m rlt.scripts.smoke_test
```

## GPU 主机完整流程

```bash
bash scripts/setup_gpu.sh
# configs/default.yaml → device: cuda
python scripts/download_checkpoint.py --config pi05_base
python -m rlt.scripts.train_rl_token --config configs/default.yaml
python -m rlt.scripts.train_online_rl --config configs/default.yaml --episodes 50
```

## 目录结构

```
rlt_reproduce/
├── configs/default.yaml      # 超参与论文 Appendix B 对齐
├── src/rlt/
│   ├── rl_token/             # VLA embedding → RL token
│   ├── rl/                   # Actor-Critic, replay, TD3 learner
│   ├── vla/                  # openpi wrapper (mock / real)
│   └── sim/                  # CPU mock env
├── third_party/openpi/       # Physical Intelligence VLA
└── scripts/
    ├── setup_env.sh
    ├── setup_gpu.sh
    └── download_checkpoint.py
```

## 方法概要

1. **RL Token 预训练**：在冻结 VLA 的 final-layer embeddings 上训练 encoder-decoder，瓶颈向量 `z_rl` 作为 RL 状态。
2. **在线 RL**：小型 MLP actor/critic，actor 以 VLA reference action chunk 为条件并加 BC 正则 (β)；50% reference dropout。
3. **部署**：VLA 处理粗粒度阶段，RL policy 接管 critical phase（与 HIL-SERL 类似的人机切换）。

参考资源：
- 论文 PDF: https://www.pi.website/download/rlt.pdf
- arXiv: https://arxiv.org/html/2604.23073v1
- 社区复现笔记: [Ville Kuosmanen](https://villekuosmanen.medium.com/research-notes-from-reproducing-rl-token-f375ecfd3c28)

## 与本机 yjr 数据采集

本机 `yjr/data/` 下的 demo 可通过 `rsync`/`scp` 拷到 GPU 主机的 `data/demos/`，再扩展数据加载器接入 `train_rl_token.py`（当前版本使用 mock embedding，待接 openpi forward hook）。

## 与 HIL-SERL 环境

若已有 `hil-serl` conda 环境，可复用其 gRPC learner/actor 通信；本仓库 `online_rl` 模块可对接 LeRobot HIL-SERL 实现。

## 许可证

复现代码 MIT；openpi 遵循其原仓库许可证。
