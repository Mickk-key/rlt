# RLT 在线 RL 双机拆分 — 任务清单

> **工控机** `10.162.132.11` → **GPU** `10.176.53.120:8765`（JPEG WebSocket）  
> 工作目录：`/home/host5010/workspaces/smq&jgy`

最后更新：**2026-07-10**

---

## 当前进度（2026-07-10 收工）

| 阶段 | 状态 | 说明 |
|------|------|------|
| Stage-2c 真实 VLA + 双路相机 infer | ✅ | Libero + SFT 5000 |
| Stage-2d reference 真机 rollout | 🟡 | ~20+ ep；success 少；motion 卡（infer 慢） |
| Stage-3 replay + TD3 | 🟡 | 收工 buffer **578**，train **395**；ckpt 已存 |
| Stage-4 policy 控臂 | ⬜ | 仍为 `reference` |

**配置（✅ 已分离）**：GPU 用 `configs/plug_insertion_gpu.yaml`；工控机用 `configs/plug_insertion.yaml`。详见 `rlt_project/rlt_reproduce/docs/ONLINE_RL_WORK_PLAN.md`。

**下次续跑**：GPU `CUDA_VISIBLE_DEVICES=1 bash scripts/start_gpu_rl_server.sh` → 工控机隧道 + `CONFIRM=1 bash scripts/run_deoxys_actor.sh`。⚠️ 重启 server 会清空 replay buffer。

---

## 协议确认（与 GPU 端一致）

**双机联调 = smq&jgy 栈（JPEG base64 JSON WebSocket）**

| 工控机 | GPU |
|--------|-----|
| `run_deoxys_actor.sh` → `actor_loop.py` | smq&jgy 的 `run_rl_server.sh` → `rl_server.py` |
| `ws_protocol` JPEG base64 | `decode_images_jpeg` |
| meta: `policy_mode=reference`, `z_rl_norm` | 与 loopback 一致 |

**不要**用工控机 actor 连 rlt_reproduce 独立仓的 **msgpack** server。  
smq&jgy 内 `rlt_reproduce/scripts/run_rl_server.sh` 已指向 **JPEG 版 `rl_server.py`**（正确）。

离线 RL Token 训练仍可用 rlt_reproduce；**在线双机只认 smq&jgy 同步到 10.176.53.120 的那份**。

---

## 网络实测（工控机侧）

| 目标 | ping | :8765 | 说明 |
|------|------|-------|------|
| **10.176.53.120** | ✅ ~5ms | 🟡 需 server 或 SSH 隧道 | 2026-07-10 双机 infer 已通 |
| 192.168.110.18 (fvl08) | ❌ 超时 | ❌ | 工控机不可达，勿改 IP |
| 192.168.130.18 | ❌ 超时 | — | 同上 |

**推荐**：工控机 `scripts/gpu/start_ssh_tunnel.sh` + `GPU_SERVER_HOST=127.0.0.1`（2026-07-10 已验证）。

---

## 当前待办（2026-07-10 起）

| 项 | 状态 |
|----|------|
| GPU deploy 树 + server 启动 | ✅ 见 `docs/GPU_SERVER_START.md` |
| 双机 reference 真机 | 🟡 继续攒 episode / reward |
| Stage-3 在线 TD3 | 🟡 已启动；续跑注意 buffer 重启清空 |
| Stage-4 policy | ⬜ 勿提前切换 |
| NAT 直连 :8765 | 🟡 可选；隧道可用 |

---

## 工控机 — 你现在能跑的

```bash
cd "/home/host5010/workspaces/smq&jgy"

# 本地 loopback（不依赖 GPU 物理机）
bash scripts/test_gpu_loopback.sh

# 打包给 GPU
bash scripts/export_smq_for_gpu.sh

# GPU server 起来之后
bash scripts/ping_gpu_server.sh
MOCK=1 bash scripts/run_deoxys_actor.sh --no-cameras --episodes 1 --max-steps 5
```

---

## GPU — 请在 10.176.53.120 上手动执行

完整步骤：**[`docs/GPU_SERVER_START.md`](docs/GPU_SERVER_START.md)**

**最短命令（GPU 终端）：**

```bash
# 解压 smq&jgy 后
cd ~/smq_jgy/smq\&jgy/rlt_project/rlt_reproduce
conda activate rlt   # 或你的 GPU conda 环境
export CUDA_VISIBLE_DEVICES=1
bash scripts/start_gpu_rl_server.sh
# 等价: bash scripts/run_rl_server.sh configs/plug_insertion_gpu.yaml
```

期望日志：

```
Inference mode: reference
RL server listening on ws://0.0.0.0:8765
```

---

## 工控机配置（✅ 已设）

- `configs/plug_insertion.yaml` → `gpu_server.host`（隧道时用 `127.0.0.1`）
- GPU 侧 **`configs/plug_insertion_gpu.yaml`** — 工控机勿改

---

## 进度

| Phase | 状态 |
|-------|------|
| 工控机 actor + loopback | ✅ |
| 双机 websocket + 真机 infer | ✅ |
| reference rollout + Stage-3 TD3 | 🟡 进行中 |
| policy 控臂 (Stage-4) | ⬜ |
