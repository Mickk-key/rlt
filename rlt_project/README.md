# RLT 复现工作区

Physical Intelligence 的 [RLT (RL Token)](https://www.pi.website/research/rlt) **官方未开源**。本目录为社区复现实现。

## 重要：训练不在本机进行

当前工控机（`vla`）**没有可用 NVIDIA 驱动**，只能做 Franka 遥操与数据采集（见 `yjr/ConsVLA/RLinf`）。  
**RLT 训练代码须拷贝到 GPU 主机**运行；已打包为：

```
../rlt_training_package_20260623.tar.gz
```

部署步骤见：**[`../RLT训练包部署指南.md`](../RLT训练包部署指南.md)**

## 路径注意

目录名含 `&`，请始终加引号：

```bash
cd "/home/host5010/workspaces/smq&jgy/rlt_project/rlt_reproduce"
```

## 本地（仅 CPU 开发 / 冒烟）

```bash
bash scripts/setup_env.sh
conda activate rlt
python -m rlt.scripts.smoke_test
```

## GPU 主机训练

```bash
bash scripts/setup_gpu.sh
python scripts/download_checkpoint.py --config pi05_base
python -m rlt.scripts.train_rl_token
python -m rlt.scripts.train_online_rl
```

详见 [rlt_reproduce/README.md](rlt_reproduce/README.md) 与 [PACKAGE_README.md](PACKAGE_README.md)。
