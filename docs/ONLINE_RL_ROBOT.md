# 工控机 Online RL Rollout 指南

> 工作目录：`/home/host5010/workspaces/smq&jgy`  
> GPU server：`fvl08` `192.168.110.18:8765`（`inference.mode` 以 **GPU deploy 树 yaml** 为准）  
> 工控机 actor 配置：`configs/plug_insertion.yaml`（仅 robot / 网络 / demo reset）

---

## 工控机还需要做什么

**Reward / episode 终止**：已有 `RewardLogger`（`s`=成功、`f`=失败、`q`=退出、timeout），**不必新写代码**。

**必须做的：**

| 项 | 说明 |
|----|------|
| **同步代码** | 已用 `src/rlt/` 覆盖 GPU deploy 的 `actor_loop.py`、`gpu_client.py`；`scripts/_env.sh` 把 `src/` 置于 `PYTHONPATH` 最前 |
| **SSH 隧道** | 无 NAT/sudo 时：`bash scripts/gpu/start_ssh_tunnel.sh`（终端 1，常驻） |
| **连本地端口** | `export GPU_SERVER_HOST=127.0.0.1`（终端 2，跑 actor 前） |
| **TTY 终端** | 实机 rollout 必须在**真实 SSH 终端**跑 actor，终端需聚焦才能按 `s`/`f` |
| **GPU server** | fvl08 上 `bash scripts/run_rl_server.sh configs/plug_insertion.yaml` |

**不必做的：**

- 不必在工控机再实现 reward 逻辑
- 不必改 msgpack 栈（协议是 JPEG WebSocket，见 `ONLINE_RL_TASKS.md`）

---

## 代码路径（canonical）

```
smq&jgy/
├── configs/plug_insertion.yaml       # 工控机 yaml（gpu_server.host 可保留边界 IP）
├── src/rlt/scripts/actor_loop.py     # GPU-synced actor
├── src/rlt/rl/gpu_client.py          # GPU-synced client
├── scripts/run_deoxys_actor.sh       # 入口
└── rlt_project/rlt_reproduce/src/rlt/   # reward_logger、reset、env 等
```

---

## 标准启动（三终端）

**终端 A — 机械臂（真机）**

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/start_robot.sh
```

**终端 B — SSH 隧道（联调全程保持）**

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/gpu/start_ssh_tunnel.sh
# 看到 [OK] Tunnel up 后不要关
```

**终端 C — Actor（必须有 TTY，用于 s/f）**

```bash
cd "/home/host5010/workspaces/smq&jgy"
export GPU_SERVER_HOST=127.0.0.1

# 通路检查（可选）
bash scripts/ping_gpu_server.sh

# 见下文分阶段命令
CONFIRM=1 bash scripts/run_deoxys_actor.sh ...
```

---

## Rollout 分阶段

| 阶段 | GPU `inference.mode` | 工控机命令 | 目的 |
|------|----------------------|------------|------|
| **1** | `reference` | `GPU_SERVER_HOST=127.0.0.1 MOCK=1 ... --no-cameras --episodes 1 --max-steps 5` | 通路 |
| **2** | `reference` | `GPU_SERVER_HOST=127.0.0.1 CONFIRM=1 ...` 真机 | 验 VLA reference 动作 |
| **3** | `reference` | 多 episode，按 `s`/`f` | 攒 replay ≥ 500 步 |
| **4** | `policy`（GPU 改 yaml 并重启 server） | 同上 | online RL 梯度更新 |

- **阶段 3 之前**：动作应跟 VLA **reference**（GPU meta 里 `policy_mode=reference`）
- **阶段 4 起**：actor 参与决策（`policy_mode=policy`）；GPU 端改 `configs/plug_insertion.yaml` 里 `inference.mode: policy` 后重启 `run_rl_server.sh`

### 阶段 1 — 通路（不连机械臂）

```bash
export GPU_SERVER_HOST=127.0.0.1
MOCK=1 bash scripts/run_deoxys_actor.sh --no-cameras --episodes 1 --max-steps 5
```

期望：`GPU server {'type': 'pong', ...}`，`GPU infer meta {'policy_mode': 'reference', ...}`

### 阶段 2 — Reference 真机试跑

```bash
export GPU_SERVER_HOST=127.0.0.1
CONFIRM=1 EPISODES=1 RESET_MODE=home MAX_STEPS=100 bash scripts/run_deoxys_actor.sh
```

观察机械臂是否按 VLA reference 运动；在合适时机按 `s` 或 `f` 结束 episode。

### 阶段 3 — 积累 replay

```bash
export GPU_SERVER_HOST=127.0.0.1
CONFIRM=1 EPISODES=10 RESET_MODE=demo bash scripts/run_deoxys_actor.sh
```

每个 episode 结束时按 `s`（成功）或 `f`（失败）。GPU replay buffer **≥ 500 step** 后 learner 才开始更新。

### 阶段 4 — Policy online RL

1. GPU 端：`inference.mode: policy`，重启 `run_rl_server.sh`
2. 工控机：同阶段 3 命令，继续 `s`/`f`

---

## Episode 内操作

1. 等 **reset** 完成（`demo` ~30s；`home` 更快）
2. 机械臂按 GPU 动作运动
3. 按键（**本终端焦点**）：
   - **`s`** — 成功，`reward=1`，episode 结束
   - **`f`** — 失败，`reward=0`
   - **`q`** — 放弃，`reward=0`
4. 不按键 → `max_steps_per_episode`（默认 200）后 timeout fail

---

## 日志位置

| 路径 | 内容 |
|------|------|
| `logs/online_rl/rewards/` | 每 episode reward JSON |
| `logs/online_rl/transitions/` | 逐步 jsonl |
| GPU replay buffer | 每 step transition；warmup 500 后 learner 更新 |

---

## 环境变量

| 变量 | 含义 | 隧道联调 |
|------|------|----------|
| `GPU_SERVER_HOST` | GPU WebSocket 地址 | **`127.0.0.1`** |
| `GPU_SERVER_PORT` | 端口 | `8765` |
| `MOCK=1` | mock env（不连机械臂） | 阶段 1 |
| `CONFIRM=1` | 实机确认 | 阶段 2+ |
| `EPISODES` | episode 数 | 默认 yaml `50` |
| `MAX_STEPS` | 超时步数 | 默认 `200` |
| `RESET_MODE` | `demo` / `home` / `none` | 默认 `demo` |

---

## 相关文档

- [`ONLINE_RL_TASKS.md`](../ONLINE_RL_TASKS.md) — 双机架构、隧道、协议
- [`docs/GPU_SERVER_START.md`](GPU_SERVER_START.md) — GPU 端启动
