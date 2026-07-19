# 工控机 Online RL Rollout 指南

> **⚠️ SUPERSEDED (2026-07-19).** The authoritative RLT online-RL runbook is now
> [`ONLINE_RL_RUNBOOK.md`](../rlt_project/rlt_reproduce/docs/ONLINE_RL_RUNBOOK.md).
> Follow that file for the full workflow; this doc is kept for history.

> 工作目录：`/home/host5010/workspaces/smq&jgy`  
> 最后更新：**2026-07-10**  
> Actor 配置：`configs/plug_insertion.yaml`  
> GPU server：deploy 树 yaml（`inference.mode` 以 GPU 端为准）

---

## 当前状态（2026-07-10）

| 项 | 状态 |
|----|------|
| 双机 WebSocket 通路 | ✅ 已通（隧道或本机 `127.0.0.1`） |
| Reset | ✅ **SFT init-cube**（`reset_method: external`，同 `bash scripts/reset_to_init.sh`） |
| 动作尺度 | ✅ `action_is_physical: true`（VLA 物理增量，不再二次缩放） |
| 首帧 VLA | ✅ reset 后 settle + 排空相机缓存 + warmup |
| ZMQ 5555 冲突 | ✅ 外部 reset 前停相机子进程 |
| 外部 reset 子进程 cwd | ✅ 已修（`cwd=rlt_project/rlt_reproduce`） |
| Reference 真机多 episode | ⏳ 待跑（见下文阶段 A） |
| GPU 代码同步 | ⏳ 需 `export_smq_for_gpu.sh`（单次 VLA infer、checkpoint 保存） |

**已废弃、文档不再写：** 旧 101 条 NPZ demo reset（`ep_00022` / `RESET_MODE=demo`）。数据已迁至 `legacy_*` 路径，仅作归档。

---

## 工控机要做什么

**Reward / episode 终止**：`RewardLogger` 已有（`s`=成功、`f`=失败、`q`=退出、timeout），不必新写代码。

| 必须 | 说明 |
|------|------|
| **SSH 隧道**（远程 GPU） | `bash scripts/gpu/start_ssh_tunnel.sh`，跑 actor 前 `export GPU_SERVER_HOST=127.0.0.1` |
| **本机 GPU**（联调） | 直接 `export GPU_SERVER_HOST=127.0.0.1`，无需隧道 |
| **TTY 终端** | 实机 rollout 须在真实 SSH 终端跑 actor，终端聚焦才能按 `s`/`f` |
| **GPU server** | GPU 上 `bash scripts/run_rl_server.sh configs/plug_insertion_gpu.yaml`（或 deploy 等价 yaml） |

---

## Reset（与 SFT 采集一致）

Online actor 每 episode 开头：

1. **停相机**（释放 ZMQ 5555）
2. **子进程** `python -m rlt.scripts.reset_to_init_pose --config configs/sft_plug_insertion.yaml --random`（`cwd=rlt_reproduce`）
3. **重启相机** → settle 1s → 排空缓存 → warmup 25 步
4. 日志应出现：`[external_reset] verified init pose ... z≈0.202m — OK to start VLA`

手动测 reset（不跑 actor）：

```bash
bash scripts/reset_to_init.sh          # 随机 init cube
FIXED=1 bash scripts/reset_to_init.sh  # 固定底面中心
```

Init 立方体底面中心 z ≈ **0.202 m**（见 `configs/plug_insertion.yaml` → `sft_collection.workspace_randomization`）。

---

## 代码路径

```
smq&jgy/
├── configs/plug_insertion.yaml              # 工控机 actor yaml
├── configs/sft_plug_insertion.yaml          # reset 与 SFT 共用
├── src/rlt/scripts/actor_loop.py
├── src/rlt/rl/gpu_client.py
├── scripts/run_deoxys_actor.sh
└── rlt_project/rlt_reproduce/src/rlt/
    ├── hardware/deoxys/collection_reset.py  # 外部 reset 子进程
    ├── hardware/deoxys/reset_manager.py
    └── scripts/rl_server.py                 # GPU 端
```

---

## 标准启动

**终端 A — 机械臂**

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/start_robot.sh
```

**终端 B — SSH 隧道**（远程 GPU 时保持运行）

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/gpu/start_ssh_tunnel.sh
```

**终端 C — Actor**

```bash
cd "/home/host5010/workspaces/smq&jgy"
export GPU_SERVER_HOST=127.0.0.1

bash scripts/ping_gpu_server.sh   # 可选

CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 2 --max-steps 200
```

---

## Rollout 分阶段

| 阶段 | GPU `inference.mode` | 工控机 | 目的 |
|------|----------------------|--------|------|
| **A** | `reference` | `CONFIRM=1 ... --episodes 5+` | 验 reset + VLA reference，按 `s`/`f` 标 reward |
| **B** | `reference` | 继续 rollout | replay buffer ≥ **warmup+500**（默认 warmup=500） |
| **C** | `policy` | GPU 改 yaml 并重启 server，同上 | online RL 梯度更新 |

**阶段 A 通过标准：**

- reset 后 `ee_z ≈ 0.202 m`，`pos_err < 3 cm`
- `GPU infer meta` 里 `policy_mode=reference`
- 机械臂有可见运动（非「蹭」不动）
- `logs/online_rl/` 有 transitions / rewards

**阶段 B → C：** GPU `pong` 里 `buffer_size ≥ 750` 且 loss 稳定后，再将 `inference.mode: policy` 并重启 server。

### Mock 通路（不连机械臂）

```bash
export GPU_SERVER_HOST=127.0.0.1
MOCK=1 bash scripts/run_deoxys_actor.sh --no-cameras --episodes 1 --max-steps 5
```

期望：`GPU server {'type': 'pong', ...}`，`policy_mode: reference`。

---

## Episode 内操作

1. 等 **external reset** 完成（约 10–30 s，视随机 xy）
2. 等日志 **OK to start VLA**
3. 机械臂按 GPU 动作运动
4. 按键（**本终端焦点**）：
   - **`s`** — 成功，`reward=1`
   - **`f`** — 失败，`reward=0`
   - **`q`** — 放弃
5. 不按键 → `--max-steps`（默认 yaml `200`）后 timeout fail

---

## 日志

| 路径 | 内容 |
|------|------|
| `logs/online_rl/rewards/` | 每 episode reward JSON |
| `logs/online_rl/transitions/` | 逐步 jsonl |
| GPU | replay buffer；warmup 后 `training: true`；`checkpoints/online_rl/` 周期快照 |

---

## 环境变量

| 变量 | 含义 | 隧道联调 |
|------|------|----------|
| `GPU_SERVER_HOST` | WebSocket 地址 | **`127.0.0.1`** |
| `GPU_SERVER_PORT` | 端口 | `8765` |
| `MOCK=1` | mock env | 通路冒烟 |
| `CONFIRM=1` | 实机确认 | 阶段 A+ |
| `CONFIRM=1 bash ... --episodes N --max-steps M` | 覆盖 yaml 默认 | 推荐写法 |

`RESET_MODE=demo` **已无效**；reset 由 yaml `online_rl.reset_mode: workspace` + `reset_method: external` 控制。

---

## 相关文档

- [`docs/desktop/online_rl_worklog.md`](desktop/online_rl_worklog.md) — 进度与故障记录
- [`ONLINE_RL_TASKS.md`](../ONLINE_RL_TASKS.md) — 双机架构、隧道
- [`docs/GPU_SERVER_START.md`](GPU_SERVER_START.md) — GPU 端启动
- [`docs/SFT_DATA_COLLECTION.md`](SFT_DATA_COLLECTION.md) — init-cube reset 标定
