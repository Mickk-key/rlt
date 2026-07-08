# SMQ&JGY — Franka FR3 真机 + RLT / Online RL

工控机工作目录，封装 deoxys 机械臂/夹爪启动、数据采集、Online RL actor。

## 快速启动

```bash
cd "/home/host5010/workspaces/smq&jgy"

# 终端 1 — 机械臂 + 夹爪
bash scripts/start_robot.sh

# 终端 2 — 数据采集
bash scripts/collect_data.sh

# Online RL（见 ONLINE_RL_TASKS.md）
bash scripts/gpu/start_ssh_tunnel.sh
CONFIRM=1 bash scripts/run_deoxys_actor.sh
```

## 首次部署依赖

| 依赖 | 恢复方式 |
|------|----------|
| `third_party/deoxys` (~9GB) | `bash scripts/robot/copy_deoxys_to_smq.sh` |
| conda 环境 `franka_mani` | 见 `机械臂启动说明.md` |
| openpi (GPU 训练) | `rlt_project/rlt_reproduce/scripts/setup_env.sh` |

## 文档

- [机械臂启动说明.md](机械臂启动说明.md) — 真机操作入口
- [ONLINE_RL_TASKS.md](ONLINE_RL_TASKS.md) — 双机 Online RL 任务清单
- [docs/ONLINE_RL_ROBOT.md](docs/ONLINE_RL_ROBOT.md) — 工控机 rollout 指南
- [docs/SFT_DATA_COLLECTION.md](docs/SFT_DATA_COLLECTION.md) — SFT 数据采集

## Git 说明

本仓库用 git 跟踪**源码与配置**；大体积目录（`third_party/deoxys`、`data/`、`logs/`）在 `.gitignore` 中排除。误删文件时可 `git checkout -- <path>` 恢复。
