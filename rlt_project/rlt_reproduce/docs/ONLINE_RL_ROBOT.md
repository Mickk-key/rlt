# Online RL — robot host checklist

> **⚠️ SUPERSEDED (2026-07-19).** The authoritative RLT online-RL runbook is now
> [`ONLINE_RL_RUNBOOK.md`](ONLINE_RL_RUNBOOK.md). Follow that file for the full
> workflow; this doc is kept for history.

> **最后更新**：2026-07-10  
> Dual-machine stack: **JPEG WebSocket** (`actor_loop.py` ↔ `rl_server.py`).

## Config（✅ 已分离）

| 侧 | 文件 | 用途 |
|----|------|------|
| 工控机 | `configs/plug_insertion.yaml` | **actor_loop 读此文件** |
| GPU | `configs/plug_insertion_gpu.yaml` | rl_server；工控机 **不要** 改此文件 |

## Network (no sudo)

Until boundary NAT forwards `10.176.53.120:8765` → fvl08, use SSH tunnel:

```bash
# Terminal 1 — keep open
bash scripts/gpu/start_ssh_tunnel.sh

# Terminal 2
export GPU_SERVER_HOST=127.0.0.1
```

## Robot-side requirements

| Item | Status in smq&jgy |
|------|-------------------|
| `actor_loop.py` + `RewardLogger` (s/f/q keys) | ✅ built-in |
| Workspace reset | ✅ `online_rl.reset_mode: workspace`（2026-07-10 使用） |
| RealSense cameras | ✅ `deoxys_realsense` |
| Transition → GPU replay | ✅ sends encoded `state` + `next_proprio` (+ JPEG) |
| GPU infer timeout | ✅ `gpu_server.infer_timeout_sec: 180` |

### Controls during rollout

- **s** = success → reward=1, episode ends
- **f** = fail → reward=0, episode ends
- **q** = quit run
- Terminal must be **focused** (TTY); otherwise only step timeout ends episodes

### Sync code from GPU deploy tree

After GPU updates, rsync `src/rlt/` + **both** yaml files from fvl08 deploy path:

- `configs/plug_insertion.yaml`（工控机）
- `configs/plug_insertion_gpu.yaml`（仅 GPU 需要）

## Rollout phases（2026-07-10 进度）

### Phase 1 — reference（validate VLA + arm） — 🟡 部分完成

GPU **`configs/plug_insertion_gpu.yaml`**:

```yaml
inference:
  mode: reference
```

```bash
# 通路 smoke（可选）
MOCK=1 bash scripts/run_deoxys_actor.sh --no-cameras --episodes 1 --max-steps 5

# 真机（2026-07-10 已跑 ~20+ episode）
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 10 --max-steps 200
```

**已观察到**：`policy_mode=reference`、双路 JPEG infer 成功；success 比例低；motion 因 GPU infer 慢而卡顿（预期）。

**首步 infer 请等 2–3 分钟。**

### Phase 2 — collect replay + TD3（Stage-3） — 🟡 已启动

保持 GPU `mode: reference`。2026-07-10 收工 GPU 侧：`buffer_size=578`，`train_steps=395`，actor/critic ckpt 已存于 GPU deploy 树。

工控机继续多 episode 按 **s/f** 标 reward。

### Phase 3 — policy mode（Stage-4） — ⬜ 未开始

**勿提前切换。** 条件：reference 动作验稳、`train_steps` 足够、团队确认。

On **GPU only**, edit **`configs/plug_insertion_gpu.yaml`** and restart `start_gpu_rl_server.sh`:

```yaml
inference:
  mode: policy
  deterministic: true
```

Robot command unchanged (`plug_insertion.yaml`).

## Nothing extra to implement on robot

Reward, episode termination, reset, and transition protocol are already in `actor_loop.py`. Only ensure:

1. SSH tunnel (or NAT) to GPU
2. Latest `actor_loop.py` / `gpu_client.py` + **工控机 yaml** synced
3. TTY for HIL keys during real episodes
4. `CONFIRM=1` for real robot runs
