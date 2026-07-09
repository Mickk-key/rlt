# Plug Insertion Online RL — 启动流程与当前进度

> **任务**：Franka 插充电器 critical phase，双机 split online RL（RLT 论文 Algorithm 1）  
> **最后更新**：2026-07-10  
> **GPU 主机**：fvl08（内网 `192.168.110.18`，对外 `10.176.53.120`）  
> **工控机**：`10.162.132.11`，工作目录 `/home/host5010/workspaces/smq&jgy`

---

## 1. 架构一览

```
工控机 10.162.132.11                    GPU fvl08 / 10.176.53.120
┌────────────────────────────┐          ┌─────────────────────────────┐
│ RealSense D435 ×2 (USB)    │  JPEG    │ start_gpu_rl_server.sh      │
│ Deoxys 臂 + 夹爪 (ZMQ)     │  WS      │  rl_server.py               │
│ actor_loop.py              │ ───────► │  pi05 VLA + RL token        │
│ RewardLogger (s/f/q)       │  :8765   │  actor/critic + replay      │
└────────────────────────────┘          └─────────────────────────────┘
```

| 组件 | 协议 | 说明 |
|------|------|------|
| 工控机 actor | `run_deoxys_actor.sh` → `actor_loop.py` | 本地采图，WebSocket 发 GPU |
| GPU server | `scripts/start_gpu_rl_server.sh` → `rl_server.py` | **JPEG base64 JSON**，不是 msgpack |
| 图像 | `external` + `wrist` → `images_jpeg` | GPU **不接 USB 相机** |
| 控制 | Deoxys ZMQ 5555/5557 | 仅臂/夹爪，不传图 |

**GPU 需求**：**1 张整卡**（建议 24GB 基本空闲）。现有代码 **不支持多卡**；工控机 **0 GPU**。

---

## 2. 当前进度总表（2026-07-10）

| 阶段 | 内容 | 状态 | 备注 |
|------|------|------|------|
| **Stage-0** | 数据采集 51 success + 50 fail | ✅ 完成 | `data/plug_insertion/` |
| **Stage-1** | RL Token 离线训练 | ✅ 完成 | `rl_token.pt` → sft5000 软链 |
| **Stage-2a** | 双机 WebSocket 通路 | ✅ 完成 | SSH 隧道 + 真机 infer |
| **Stage-2b** | GPU 部署 smq&jgy + 真实 VLA 代码 | ✅ 完成 | deploy 树 fvl08 |
| **Stage-2c** | 真实 VLA + 双路相机 infer | ✅ **完成** | Libero 格式 + SFT 5000 ckpt |
| **Stage-2d** | reference 真机 rollout | 🟡 **部分完成** | ~20+ ep；success 少；motion 卡（infer 慢） |
| **Stage-3** | replay warmup 500 + TD3 | 🟡 **进行中** | 收工：buffer **578**，train **395**，有 actor/critic ckpt |
| **Stage-4** | `inference.mode: policy` | ⬜ 未开始 | 仍为 reference |
| **网络** | `10.176.53.120:8765` 直连 | 🟡 | 建议 SSH 隧道 |
| **GPU 资源** | 独占 1×3090 | ✅ | `CUDA_VISIBLE_DEVICES=1` 已验证 |

### 收工 checkpoint（deploy 树）

```
checkpoints/rl_actor.pt          # 2026-07-10 06:29
checkpoints/rl_critic.pt
checkpoints/online_rl/rl_*_step000150~350_*.pt
checkpoints/rl_actor.pt.presft_bak   # 旧备份，保留
```

---

## 2-old. 历史进度（2026-07-03）

| 阶段 | 内容 | 状态 | 备注 |
|------|------|------|------|
| **Stage-2c** | 真实 VLA + 双路相机 infer | ⬜ **未通过** | 无图 infer → KeyError；或 GPU OOM |
| **Stage-3** | replay warmup 500 步 | ⬜ 未开始 | |
| **GPU 资源** | 独占 1×24GB | ❌ **阻塞** | 8 卡被 simvla 占满 |

---

## 3. 已知运行现象（2026-07-10）

- **首步 / 周期性 infer 1–2 分钟**：正常（pi05 + JAX）；工控机需耐心等待。
- **Motion 卡顿**：每轮 infer 后只执行 10 步 @ 20Hz（~0.5s）再停；双机架构预期行为。
- **GPU 终端 `keepalive ping timeout`**：infer 阻塞导致；buffer 仍涨则可持续。
- **重启 server 会清空 replay buffer**（actor/critic 权重会从 ckpt 重载）。

---

## 3-old. 历史阻塞（2026-07-03）

### 3.1 GPU 显存不足（**首要**）

- fvl08 **8×3090** 均被他人 `simvla` 占用（每卡约 15–23GB）
- `start_gpu_rl_server.sh` 在 **RL token `.to(cuda)`** 阶段即 **CUDA OOM**
- **8765 当前未监听** → 工控机无法连 server
- **解决**：等实验结束，或协调 **让出 1 张整卡**（24GB 基本空闲），再：

  ```bash
  CUDA_VISIBLE_DEVICES=<空闲卡号> bash scripts/start_gpu_rl_server.sh
  ```

### 3.2 边界 NAT 未配置

- 工控机 ping `10.176.53.120` ✅，但 `:8765` refused（无 DNAT）
- **临时方案**：工控机 `scripts/gpu/start_ssh_tunnel.sh` + `GPU_SERVER_HOST=127.0.0.1` ✅ 已验证

### 3.3 真实 VLA 必须带相机图

- `--no-cameras` / `MOCK=1` 仅用于 **Mock VLA 通路测试**
- GPU 已切 **真实 pi05** 后，infer **必须**有 `external` + `wrist` JPEG
- 缺图报错：`KeyError: 'observation/exterior_image_1_left'`（不是加载失败）

---

## 4. 启动流程（按顺序）

### 前置条件

- [ ] GPU 有 **≥1 张空闲 24GB 卡**
- [ ] Stage-1 `rl_token.pt`、`pi05_base` checkpoint 在 GPU deploy 树 `checkpoints/`
- [ ] 工控机 Deoxys 臂/夹爪 SOP 可用；RealSense serial 与 yaml 一致
- [ ] 工控机已同步最新 `actor_loop.py`、`gpu_client.py`（见 §6）

---

### Step 0 — 确认 GPU 有空闲（GPU 上）

```bash
nvidia-smi
# 选一张 memory.free 接近 24GB 的卡，记下 index
```

---

### Step 1 — GPU server（GPU 终端 A，**一直挂着**）

```bash
cd "/sdb/private_folders/shimingqi/smq_jgy_deploy/smq&jgy/rlt_project/rlt_reproduce"
# 或工控机解压路径 ~/smq_jgy/smq\&jgy/rlt_project/rlt_reproduce

export CUDA_VISIBLE_DEVICES=<空闲卡号>   # 2026-07-10 使用 1
bash scripts/start_gpu_rl_server.sh
# 默认 config: configs/plug_insertion_gpu.yaml（勿用工控机 plug_insertion.yaml）
```

**等到**（首次加载 pi05 可能 2–5 分钟，UserWarning 后可继续等；日志可能滞后）：

```text
Loaded RL token from checkpoints/rl_token.pt
Inference mode: reference
RL server listening on ws://0.0.0.0:8765
```

另开终端确认：

```bash
ss -tlnp | grep 8765
# 应见 LISTEN 0.0.0.0:8765
```

> 终端挂着不动 = server 在等连接，**正常**。不要关此窗口。

---

### Step 2 — SSH 隧道（工控机终端 B，无 NAT 时，**一直挂着**）

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/gpu/start_ssh_tunnel.sh
```

---

### Step 3 — 工控机 Deoxys（按项目 SOP）

起臂、夹爪等（`start_arm.sh` / `start_robot.sh`，以 smq&jgy 文档为准）。

---

### Step 4 — 工控机 actor（工控机终端 C）

**必须在 Step 1 出现 `RL server listening` 之后**再跑（不必同一秒，但 server 要先就绪）。

```bash
cd "/home/host5010/workspaces/smq&jgy"
export GPU_SERVER_HOST=127.0.0.1    # 走隧道时

# 真机 + 双路相机（真实 online RL 用这个）
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 1 --max-steps 30
```

**不要**用于真实 VLA：

```bash
MOCK=1 ... --no-cameras   # 仅 Mock/通路测试；真实 pi05 会 KeyError
```

运行中（终端需聚焦）：

| 键 | 含义 |
|----|------|
| **s** | 成功，reward=1，结束 episode |
| **f** | 失败，reward=0，结束 episode |
| **q** | 退出 |

期望 actor 首步 log：`GPU infer meta {'policy_mode': 'reference', ...}`

---

## 5. Online RL 分阶段（reference vs policy）

这是 **两件不同的事**：

| 概念 | 配置 | 含义 |
|------|------|------|
| **执行模式** | `inference.mode` | `reference` = 跟 VLA 参考动作；`policy` = RL actor 输出动作 |
| **何时开始训练** | `online_rl.warmup_steps: 500` | replay 满 **500 条 transition** 后 GPU 才开始 TD3 梯度更新 |

### 推荐顺序

| 阶段 | GPU `inference.mode` | 工控机 | 目的 |
|------|----------------------|--------|------|
| **A** | `reference`（当前默认） | 真机 + 相机，短 episode | 验 VLA 动作 + 双路图 |
| **B** | `reference` | 多 episode，按 **s/f** 标 reward | 攒 replay → `buffer_size` → 500 |
| **C** | 改 **`policy`**，**重启 server** | 同上 | actor 控臂 + 继续收 transition + 训练 |

切 policy 时只改 **GPU** 上 `configs/plug_insertion_gpu.yaml`：

```yaml
inference:
  mode: policy
  deterministic: true
```

工控机命令不变。

---

## 6. 代码与路径

### GPU deploy（fvl08 当前使用）

```text
/sdb/private_folders/shimingqi/smq_jgy_deploy/smq&jgy/rlt_project/rlt_reproduce/
├── scripts/start_gpu_rl_server.sh    # 推荐入口
├── scripts/run_rl_server.sh           # → rl_server.py
├── configs/plug_insertion.yaml
├── checkpoints/rl_token.pt            # 软链至 rlt_reproduce 训练产物
├── checkpoints/pi05_base/
└── src/rlt/scripts/rl_server.py       # JPEG WS + 真实 VLA + replay
```

### 工控机需同步的文件（GPU 更新后）

- `src/rlt/scripts/actor_loop.py`（transition 带 `state` + `next_proprio`+JPEG）
- `src/rlt/rl/gpu_client.py`
- `configs/plug_insertion.yaml`（工控机侧 host/tunnel 相关项）

### 相关文档

| 文档 | 位置 |
|------|------|
| 工控机任务清单 | `smq&jgy/ONLINE_RL_TASKS.md` |
| 工控机 rollout 要点 | `docs/ONLINE_RL_ROBOT.md` |
| GPU 部署包 | `smq_jgy_rl_server_20260702.tar.gz` |

---

## 7. 常见报错对照

| 现象 | 原因 | 处理 |
|------|------|------|
| `CUDA error: out of memory`（启动时） | GPU 被占满 | 换空闲卡或等别人实验结束 |
| `KeyError: observation/exterior_image_1_left` | infer **无相机图** | 去掉 `--no-cameras`，开 RealSense |
| `connection handler failed` 紧跟 UserWarning | server **已起来**，infer 失败 | 看栈底 KeyError/OOM，不是「没加载完」 |
| 工控机 `:8765` refused | server 未 listen 或 NAT 未配 | 查 `ss -tlnp`；无 NAT 用隧道 |
| `policy_mode=reference` | 正常 | 尚未切 policy；不是 bug |
| `stdin 非 TTY` | 无键盘 reward | 仅 timeout 结束 episode；实机用交互终端 |

---

## 8. 下一步行动清单

**GPU 侧（你）**

1. [ ] 协调 **1 张空闲 3090**
2. [ ] `CUDA_VISIBLE_DEVICES=<N> bash scripts/start_gpu_rl_server.sh`，确认 `RL server listening`
3. [ ] 保持 server 终端不关

**工控机侧**

1. [ ] `start_ssh_tunnel.sh` 常开
2. [ ] 同步最新 actor/gpu_client（若尚未同步）
3. [ ] `CONFIRM=1 run_deoxys_actor.sh`（**无** MOCK / **无** `--no-cameras`）
4. [ ] reference 模式下多跑 episode，**s/f** 标 reward，直到 GPU `buffer_size ≥ 500`
5. [ ] GPU 改 `mode: policy` 并重启 server → 继续 rollout

**网络（可选，需管理员 sudo）**

- [ ] 边界 NAT：`10.176.53.120:8765` → `192.168.110.18:8765`（`setup_port_forward_8765.sh` on fvl05）

---

## 9. 一句话状态（2026-07-03）

**离线 RL Token 已完成；双机 JPEG WebSocket 在隧道 + MOCK 下已通；真实 pi05 + 真机 online RL 卡在 GPU 无空闲显存（OOM），且真机跑时必须开双路相机、不能用 `--no-cameras`。**
