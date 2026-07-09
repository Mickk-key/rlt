# RLT 在线 RL 双机拆分 — 任务清单

> **工控机** `10.162.132.11` → **GPU fvl08** `192.168.110.18:8765`  
> **推荐通路**：SSH 隧道 `127.0.0.1:8765`（**无需边界 NAT**）  
> 工作目录：`/home/host5010/workspaces/smq&jgy`  
> **工控机 rollout 步骤**：[`docs/ONLINE_RL_ROBOT.md`](docs/ONLINE_RL_ROBOT.md)

最后更新：**2026-07-10**（reset 已改为 SFT init-cube external；详见 [`docs/ONLINE_RL_ROBOT.md`](docs/ONLINE_RL_ROBOT.md)）

---

## 推荐联调方式：SSH 隧道（可长期使用）

**可以一直用 SSH 隧道，不必等边界 NAT。**

| | SSH 隧道 | 边界 NAT |
|--|----------|----------|
| 是否需要网管 | ❌ 不需要 | ✅ 需 root 配 DNAT |
| 工控机连谁 | `127.0.0.1:8765` | `10.176.53.120:8765` |
| 稳定性 | 依赖 SSH 会话；断线需重开隧道 | 更「正规」，少一层 |
| 当前状态 | ✅ **已验证双机 infer** | ❌ 未配置 |

隧道原理：工控机 `localhost:8765` → SSH 经 fvl05（`10.176.53.120:26570`）→ fvl08 `192.168.110.18:8765`。

---

## 联调时要开什么、要不要一直开着

### 必须长期运行（联调 / rollout 期间勿关）

| 位置 | 脚本 | 终端 | 说明 |
|------|------|------|------|
| **GPU fvl08** | `bash scripts/run_rl_server.sh configs/plug_insertion.yaml` | GPU 上 1 个 | JPEG rl_server，`0.0.0.0:8765` |
| **工控机** | `bash scripts/gpu/start_ssh_tunnel.sh` | 工控机 1 个 | **必须一直开着**；关了这个 actor 连不上 GPU |
| **工控机（真机）** | `bash scripts/start_arm.sh` + `start_gripper.sh` | 各 1 个或 `start_robot.sh` | 仅 mock 测试不需要 |

### 每次联调 / 每个 episode 运行（跑完可关）

| 脚本 | 何时 |
|------|------|
| `GPU_SERVER_HOST=127.0.0.1 bash scripts/run_deoxys_actor.sh ...` | 跑 actor rollout |
| `GPU_SERVER_HOST=127.0.0.1 bash scripts/ping_gpu_server.sh` | 测通路（可选） |

### 停止

```bash
# 工控机 — 关隧道（actor 结束后若不再联调）
bash scripts/gpu/stop_ssh_tunnel.sh
```

---

## 工控机标准联调流程（复制即用）

```bash
cd "/home/host5010/workspaces/smq&jgy"
```

**终端 A — 机械臂（真机才需要，mock 可跳过）**

```bash
bash scripts/start_robot.sh    # 或 start_arm.sh + start_gripper.sh
```

**终端 B — SSH 隧道（联调全程保持运行）**

```bash
bash scripts/gpu/start_ssh_tunnel.sh
# 看到 [OK] Tunnel up 后不要关这个终端
# 若已启动会提示 Tunnel already running
```

**终端 C — 验证 + actor**

```bash
# 1) 通路检查
GPU_SERVER_HOST=127.0.0.1 bash scripts/ping_gpu_server.sh

# 2) Mock 冒烟（不连机械臂）
GPU_SERVER_HOST=127.0.0.1 MOCK=1 bash scripts/run_deoxys_actor.sh \
  --no-cameras --episodes 1 --max-steps 5

# 3) 真机 rollout（需终端 A + 本终端按 s/f）
GPU_SERVER_HOST=127.0.0.1 CONFIRM=1 bash scripts/run_deoxys_actor.sh \
  --episodes 1 --reset-mode home
```

期望 log：

```
GPU server {'type': 'pong', 'buffer_size': 0, 'device': 'cuda'}
GPU infer meta {'policy_mode': 'reference', 'z_rl_norm': ...}
```

**说明**

- 联调时 **`GPU_SERVER_HOST` 必须设为 `127.0.0.1`**（走隧道），不要设 `10.176.53.120`（直连仍 refused）。
- `scripts/robot/deoxys_actor.env` 默认是 `10.176.53.120`；用隧道时在命令前加 `GPU_SERVER_HOST=127.0.0.1`，或：
  ```bash
  export GPU_SERVER_HOST=127.0.0.1   # 写入 ~/.bashrc 或联调前 export 一次
  ```
- 人工 reward（`s`/`f`）需在**有 TTY 的真实 SSH 终端**里跑 actor，Cursor 后台 shell 只能 timeout 结束。

---

## GPU 端（fvl08，只需起一次 server）

在 **fvl08** 上（`192.168.110.18`，rl_server 已跑则跳过）：

```bash
cd ~/smq_jgy/smq\&jgy/rlt_project/rlt_reproduce   # 解压路径按实际
conda activate rlt
bash scripts/run_rl_server.sh configs/plug_insertion.yaml
```

期望：

```
Inference mode: reference
RL server listening on ws://0.0.0.0:8765
```

详见 [`docs/GPU_SERVER_START.md`](docs/GPU_SERVER_START.md)。

---

## 协议（smq&jgy 栈）

| 工控机 | GPU |
|--------|-----|
| `run_deoxys_actor.sh` → `actor_loop.py` | `run_rl_server.sh` → `rl_server.py` |
| JPEG base64 WebSocket | `decode_images_jpeg` |
| meta: `policy_mode`, `z_rl_norm` | 与 loopback 一致 |

不要用工控机 actor 连 rlt_reproduce 独立仓的 **msgpack** server。

---

## 网络对照

| 路径 | 状态 |
|------|------|
| fvl08 `:8765` | ✅ server 正常 |
| 工控机 → `10.176.53.120:8765` 直连 | ❌ refused（无 NAT） |
| 工控机 → `127.0.0.1:8765` SSH 隧道 | ✅ **推荐** |

---

## 可选：边界 NAT（不用隧道时才需要）

若以后网管配好 DNAT，可把 `GPU_SERVER_HOST` 改回 `10.176.53.120`，无需再开隧道。

脚本（需 root，在拥有 `10.176.53.120` 的网关上）：`scripts/gpu/setup_port_forward_8765.sh`（已 scp 到 fvl05 `~/setup_port_forward_8765.sh`）。

---

## 本地-only 测试（不连 GPU 物理机）

```bash
bash scripts/test_gpu_loopback.sh          # 工控机本机起临时 server
MOCK=1 bash scripts/run_deoxys_actor.sh --mock --episodes 1   # 全 mock
```

---

## Online RL 真机运行（手动 s / f）

联调成功后，进入**带人工 reward 的在线 rollout**。每个 episode 你按键结束，GPU 端 learner 从 transition 里学。

### 一次 session 要开什么

| 终端 | 位置 | 命令 | 常驻？ |
|------|------|------|--------|
| GPU | fvl08 | `bash scripts/run_rl_server.sh configs/plug_insertion.yaml` | ✅ |
| B | 工控机 | `bash scripts/gpu/start_ssh_tunnel.sh` | ✅ |
| A | 工控机 | `bash scripts/start_robot.sh` | ✅ |
| **C** | 工控机 **SSH（有键盘）** | 下面 actor 命令 | 每轮 episode |

> **终端 C 必须是真实 SSH**，不能是 Cursor 内置终端，否则 `s`/`f` 无效，只会 timeout。

### 启动命令（终端 C）

```bash
cd "/home/host5010/workspaces/smq&jgy"
export GPU_SERVER_HOST=127.0.0.1

# 建议先 1–2 个 episode 试手感（reset 同 bash scripts/reset_to_init.sh）
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 2 --max-steps 200

# 阶段 A：多 episode 标 s/f
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 10 --max-steps 200
```

可选参数：

| 变量 / 参数 | 含义 | 默认 |
|-------------|------|------|
| `--episodes N` | 连续 N 个 episode | yaml `max_episodes`（50） |
| `--max-steps M` | 超时自动 fail（reward=0） | yaml `200` |
| `MOCK=1` | 不连机械臂 | — |
| `CONFIRM=1` | 实机确认 | 真机必须 |

Reset 由 yaml `online_rl.reset_method: external` 控制（**不再使用** `RESET_MODE=demo`）。

### Episode 里你要做什么

1. **等 external reset 完成**（子进程 `reset_to_init_pose`，init cube z≈0.202m）
2. **机械臂开始动** — GPU 发 reference action（config 里 `inference.mode: reference`）
3. **任务进行中**：观察插入是否成功
4. **按键**（焦点在本终端）：
   - **`s`** — 成功 → `reward=1`，本 episode 结束
   - **`f`** — 失败 → `reward=0`，本 episode 结束
   - **`q`** — 放弃 → `reward=0`，结束
5. 若不按键，**200 step 后自动 fail**（timeout）
6. 自动进入下一个 episode（若 `EPISODES>1`）

### 数据去哪了

| 位置 | 内容 |
|------|------|
| 工控机 `logs/online_rl/rewards/` | 每个 episode 的 reward JSON |
| 工控机 `logs/online_rl/transitions/` | 逐步 jsonl |
| **GPU replay buffer** | 每 step 的 transition；**warmup 500 step 后** learner 才开始更新 |

### 阶段建议

| 阶段 | GPU `inference.mode` | 工控机做什么 |
|------|----------------------|--------------|
| **1. Reference 试跑** | `reference`（当前默认） | 熟悉 reset + s/f；动作=VLA 参考轨迹 |
| **2. 积累数据** | `reference` | 多 episode，成功/失败都标；GPU buffer 到 500+ |
| **3. 上 RL policy** | GPU 改 yaml 为 `policy`，重启 server | actor 执行 RL 修正后的 action |
| **4. 持续在线训练** | `policy` | 继续 s/f；GPU 定期存 `checkpoints/rl_actor.pt` |

GPU 改 inference 模式：编辑 `configs/plug_insertion.yaml` 里 `inference.mode: policy`，重启 `run_rl_server.sh`。

### 第一次真机建议流程

```bash
# 1) 终端 B 确认隧道
bash scripts/gpu/start_ssh_tunnel.sh

# 2) 终端 C — reference 试跑
export GPU_SERVER_HOST=127.0.0.1
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 2 --max-steps 200
# 等日志 OK to start VLA 后观察臂动；适时按 s 或 f

# 3) 确认日志
ls logs/online_rl/rewards/
cat logs/online_rl/rewards/ep_0000.json

# 4) 阶段 A — 多 episode 攒 buffer
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 5 --max-steps 200
```

### 安全提醒

- 第一次 reference rollout 前确认工作空间、急停可用
- external reset 会移动机械臂到 init cube，勿靠近
- 异常时终端 `Ctrl+C`，必要时物理急停

---

| Phase | 状态 |
|-------|------|
| 工控机 actor + loopback | ✅ |
| 双机 websocket（SSH 隧道） | ✅ |
| 真机 + 手动 s/f online RL | ⬜ 按上文「Online RL 真机运行」 |

---

## 相关脚本

| 脚本 | 作用 |
|------|------|
| `scripts/gpu/start_ssh_tunnel.sh` | 开隧道（**联调时常驻**） |
| `scripts/gpu/stop_ssh_tunnel.sh` | 关隧道 |
| `scripts/gpu/setup_port_forward_8765.sh` | 边界 NAT（可选） |
| `scripts/run_deoxys_actor.sh` | 工控机 actor 入口 |
| `scripts/ping_gpu_server.sh` | 测 GPU 连通 |

隧道 pid：`logs/gpu_tunnel_8765.pid`
