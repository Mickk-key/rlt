# Online RL 工作计划（整合版）— 插充电器 Critical Phase

> **任务**：Franka 插充电器，双机 split online RL（RLT Algorithm 1）  
> **最后更新**：2026-07-11  
> **整合自**：
> - **[工控机]** `docs/desktop/online_rl_worklog.md` — 工控机 `10.162.132.11`（用户 `host5010`），工作目录 `/home/host5010/workspaces/smq&jgy`
> - **[主机/GPU]** 原 `rlt_project/rlt_reproduce/docs/ONLINE_RL_WORK_PLAN.md` — GPU `10.176.53.120`（fvl08 / 内网 `192.168.110.18`），部署树 `smq_jgy_deploy/smq&jgy/rlt_project/rlt_reproduce/`
>
> **来源标记**：`[工控机]` / `[主机]` / `[共通]` 标注信息最初记录方；整合后两边应同步 git commit。  
> **相关文档**：`ONLINE_RL_RUNBOOK.md`（启动命令）、`ONLINE_RL_ROBOT.md`（工控机 checklist）、工控机桌面 `docs/desktop/online_rl_worklog.md`（摘要入口）

---

## 0. 文档说明

| 标记 | 含义 | 典型内容 |
|------|------|----------|
| **[工控机]** | 真机侧记录 | reset、deoxys、RealSense、ZMQ、actor 配置、reset 排错 |
| **[主机]** | GPU 部署树记录 | `rl_server`、SFT ckpt、RL Token、TD3、buffer、算法分析 |
| **[共通]** | 双机协作 | 架构、阶段定义、执行流程、s/f 协议 |

**配置勿混用**：

| 机器 | 配置文件 |
|------|----------|
| **[主机]** GPU `rl_server` | `configs/plug_insertion_gpu.yaml`（含 `inference.mode`、`vla`、`learner`） |
| **[工控机]** `actor_loop` | `configs/plug_insertion.yaml`（reset、相机、`gpu_server`；**无** `inference.mode`） |

---

## 1. 系统架构 `[共通]`

```text
SFT 采集 / reset 标定（工控机）
  init-cube 随机 reset → JSONL+PNG → GPU 离线 VLA SFT + RL Token
       ↓
Online RL
  工控机 external reset（同 reset_to_init.sh）→ actor_loop（双相机 + proprio）
       → WebSocket（隧道 127.0.0.1:8765 或直连）→ GPU rl_server
       → embedding + reference/policy action + TD3 更新
       → 回 action → 真机执行 → 人工 s/f
```

| 机器 | IP / 角色 | 职责 |
|------|-----------|------|
| **[工控机]** `vla` | `10.162.132.11` | deoxys、SFT 采集、`actor_loop`、SSH 隧道 |
| **[主机/GPU]** `fvl08` | `10.176.53.120` | `rl_server`（VLA + RL Token + TD3） |

**[工控机]** 工控机 **不跑 VLA**；每步传 proprio + 双相机 JPEG，embedding 在 GPU 算。

---

## 2. 速查 — 核心概念、算法问题与对策 `[主机]`（2026-07-10）

> 组内 FAQ：Online RL / Reference / Policy 概念、算法问题、优化方向、续跑 checklist。

### 2.A 核心概念

#### 2.A.1 四个阶段别混

| 层级 | 含义 | 当前（7/10） |
|------|------|--------------|
| 双机通路 | WebSocket infer / transition 通 | ✅ |
| **Reference rollout** | 真臂 + SFT VLA；臂执行 **reference** | 🟡 ~20+ ep，success 低 |
| **Stage-3 训练** | buffer ≥ warmup 后 TD3 更新 | 🟡 buffer 578，train 395 |
| **Policy 控臂（Stage-4）** | `mode: policy`，**RL Actor** 出动作 | ⬜ 未开始 |

- **Online RL「训练开始」** = replay 满 warmup + GPU 出现 `learner metrics`（**不必等 policy**）。
- **Policy 阶段** = 改 **[主机]** GPU yaml `inference.mode: policy` 并重启 server。

#### 2.A.2 Rollout / Reference / Policy

| 词 | 一句话 |
|----|--------|
| **Rollout** | 真机跑完一整局 episode（reset → 多步 → s/f/timeout） |
| **Reference** | SFT VLA 输出的动作；`inference.mode: reference` 时 **臂听 VLA** |
| **Policy 控臂** | RL Actor 输出动作；VLA 仍提供 reference 作条件，但 **臂听 Actor** |
| **Buffer** | GPU 上存 transition 的经验池（`buffer_size` = 条数，不是 episode 数） |
| **TD3** | 从 buffer 抽样更新 Actor/Critic 的算法（`RLTLearner`） |
| **Transition** | 一步经验，见下表 |

#### 2.A.3 Transition 记什么？

| 字段 | 含义 |
|------|------|
| `state` | 当前 `[z_rl, proprio]`（GPU 算） |
| `action` | **本步实际执行**的 7 维动作 |
| `reference_action` | 本步 VLA reference |
| `reward` | 通常仅按 **s/f** 那步为 1/0，中间步为 0 |
| `next_state` | **下一步状态向量**（非「下一步动作」） |
| `done` | 本步后 episode 是否结束 |

Reference 模式下 action 来自 **VLA infer**，不是工控机本地 `f(state)`。chunk 内逐步执行；chunk 用完时 **下一次 `gpu.infer()`** 用新相机图 + proprio 再算。

#### 2.A.4 「收敛」分别指什么？

| 对象 | Reference 阶段 | Policy 阶段 |
|------|----------------|-------------|
| SFT VLA 权重 | ❌ 冻结 | ❌ 不变 |
| Reset/接近进工作区（Jerry：~100%） | ✅ 应稳定 | 仍应稳定 |
| 终局插入 success（按 s） | ❌ 不要求 100% | ✅ **行为应变好并稳定** |
| TD3 / buffer | ✅ 在积累、可训练 | ✅ 继续训 |
| **~100 episode** | online 流程 **数据量**目标 | 切 policy 后 often **再 50–200 ep** |

- Jerry「100 个左右才能收敛」≈ 跑够 **~100 局 rollout**，非终局 success 100%。
- Jerry「接近矩形 ~100%」= reset 后进工作区（`xy_half_range_m: 0.05`），**非**整段插入成功 100%。
- **Reference 不学习**；VLA 无反向传播；攒的数据训 **Actor/Critic**。

#### 2.A.5 卡顿：平滑 vs 超时

| 原因 | 现象 | 来源 |
|------|------|------|
| Action chunk 无平滑 | 块边界 jerk | Jerry |
| GPU infer 慢 | 动 ~0.5s、停 1–2 min | **[主机]** 实测 |
| WebSocket timeout | 断连、丢 transition | Mickeyy |

#### 2.A.6 Policy 阶段时长

经验：切 policy 后 **50–200 episode** + **train_steps 再多几百～几千**；以 success 率高于 reference 且稳定为准。

---

### 2.B 算法与流程 — 问题分析 `[主机]`

#### 2.B.1 总体判断

| 类别 | 阻止开跑？ | 让训练学不到东西？ |
|------|------------|-------------------|
| GPU OOM / server 未 listen | ✅ | — |
| infer 过慢 + WebSocket timeout | 可能断连 | ✅ 丢 transition |
| VLA 每 transition 重复全量 infer | — | ✅ 卡顿 |
| Action chunk 无时间平滑 | — | ⚠️ 观感差 |
| 终局 success 极低（如 1/20） | — | ✅ 负样本多 |
| Replay 重启不持久化 | — | ⚠️ 需重新 warmup |
| 过早 `mode: policy` | ✅ 真机风险 | ✅ replay 污染 |

#### 2.B.2 已修复 / 已改善（7/5 → 7/10）

| 项 | 来源 | 状态 |
|----|------|------|
| `actor_loop` 首步 `action_chunk` 顺序 | [主机] | ✅ |
| GPU yaml `inference.mode: reference` | [主机] | ✅ |
| SFT VLA + Libero + sft5000 RL Token 对齐 | [主机] | ✅ |
| init-cube external reset（与 SFT 一致） | [工控机] | ✅ |
| `action_is_physical: true` | [工控机] | ✅ |
| `rl_actor.pt` / `rl_critic.pt` 保存 | [主机] | ✅ |
| GPU/工控机 yaml 分离 | [主机] | ✅ |
| VLA infer 超时 180s；`rl_server` 合并单次 VLA | [工控机] | ✅ |
| Reset 弃用 101 demo → SFT init-cube external | [工控机] | ✅ |
| 外部 reset 前 suspend 相机子进程（ZMQ 5555） | [工控机] | ✅ |
| `post_reset_settle` + RealSense 缓存排空 + warmup | [工控机] | ✅ |
| `franka_hand.yaml` 子进程 `cwd` 修正 | [工控机] | ✅ |

#### 2.B.3 仍须关注

1. **VLA 调用过多**：每 chunk infer + 每步 transition 算 `next_state` 再 infer → ~11 次/chunk。
2. **Chunk 无平滑**：块边界 jerk + infer 等待 →「动一下停很久」。
3. **稀疏 reward + 低 success**：TD3 成功信号少。
4. **Replay 不持久化**：重启 server 清空 buffer。
5. **`max_steps_per_episode: 200`**（工控机 yaml）：易 timeout → 用 `--max-steps 2400`。

---

### 2.C 优化方向（按优先级）`[主机]`

| 优先级 | 方向 | 实现状态 |
|--------|------|----------|
| **P0** | 续跑 checklist（§2.D） | 流程 |
| **P1** | 降低 `next_state` 的 VLA 开销 / infer 异步 / chunk 平滑 | 待开发 |
| **P2** | reference ~100 ep；加大 `--max-steps`；加快 infer | 操作+运维 |
| **P3** | replay buffer 持久化 | 待开发 |
| **P4** | reference 稳定后再切 `policy` | 操作 |

**[工控机] 原待办（已部分完成，保留备查）**：

| P | 项 | 状态 |
|---|-----|------|
| P0 | 阶段 A：`run_deoxys_actor.sh --episodes 5 --max-steps 200` 验 reset→VLA→臂动→s/f | 🟡 通路已通；建议 `--max-steps 1200+` |
| P1 | `bash scripts/export_smq_for_gpu.sh` 同步 GPU | 按需 |
| P2 | buffer ≥ 750 后切 `policy`，smoke 1 ep | ⬜ 未做（[主机] warmup 500 已达；切 policy 前建议验稳） |
| P3 | 插座/相机治具长期固定；评估姿态对齐 | 长期 |

---

### 2.D 续跑 checklist `[共通]`

#### 2.D.1 开跑前三端

| 终端 | 机器 | 动作 |
|------|------|------|
| A | **[主机]** GPU | `CUDA_VISIBLE_DEVICES=1 bash scripts/start_gpu_rl_server.sh`（**`plug_insertion_gpu.yaml`**） |
| B | **[工控机]** | `bash scripts/gpu/start_ssh_tunnel.sh` 常开 |
| C | **[工控机]** | `export GPU_SERVER_HOST=127.0.0.1` + `CONFIRM=1 bash scripts/run_deoxys_actor.sh ...`（**`plug_insertion.yaml`**） |

#### 2.D.2 续跑前必查

- [ ] **[主机]** SFT ckpt、`rl_token.pt`（sft5000 软链）、`rl_server listening`
- [ ] **[工控机]** `configs/sft_plug_insertion.yaml` 存在；Deoxys + 双 RealSense；TTY 可 s/f
- [ ] 两边 **git 同一 commit**（如 `gpu-sync-20260710`）
- [ ] **勿切 policy**，直到 reference 稳定且 `train_steps` 持续增长

#### 2.D.3 推荐运行参数 `[工控机]`

```bash
# 单局试跑
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 1 --max-steps 1200

# 攒数据（~100 ep 量级）
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 20 --max-steps 2400
```

#### 2.D.4 阶段目标

```
1. Reset/接近（init 立方体）          → ~100% 稳定
2. Reference rollout + 攒 replay     → ~50–100 ep
3. TD3（Stage-3）                     → train_steps 涨
4. 切 policy（Stage-4）               → 短试 5–10 ep → 再 50–200 ep
5. Policy 收敛                        → success 高于 reference 且稳定
```

#### 2.D.5 速查对策表 `[主机]`

| 现象 | 原因 | 处理 |
|------|------|------|
| 一顿一顿 | chunk 无平滑 + infer 慢 | 减 infer 重复；后续加平滑 |
| WebSocket timeout | infer > 客户端等待 | `infer_timeout_sec: 180` |
| buffer 重启变 0 | 无持久化 | 重攒 warmup 或做 buffer save |
| success 1/20 | 稀疏 reward | 继续攒；policy 阶段再判 |
| 臂乱动 | 误切 policy | 保持 reference |

---

## 3. 最新进度（2026-07-10）`[共通]`

### 3.1 GPU / 在线训练状态 `[主机]`

**收工状态**（GPU server 已退出；shutdown 时已保存 ckpt）：

| 项 | 数值 / 状态 |
|----|-------------|
| Stage-2c 真实 VLA + 双路相机 infer | ✅ |
| Stage-2d reference 真机 rollout | 🟡 ~20+ ep；success 少；motion 卡 |
| Stage-3 replay warmup + TD3 | 🟡 buffer **578**，train **395**，`training=true` |
| Stage-4 policy 控臂 | ⬜ `inference.mode: reference` |
| 在线 ckpt | ✅ `rl_actor.pt`、`rl_critic.pt`（07-10 06:29）；`online_rl/rl_*_step150~350_*.pt` |

**部署树（勿与工控机 yaml 混用）**：

| 用途 | 文件 | 状态 |
|------|------|------|
| GPU `rl_server` | `configs/plug_insertion_gpu.yaml` | ✅ |
| 工控机 `actor_loop` | `configs/plug_insertion.yaml` | ✅ |
| SFT VLA | `.../pi05_plug_insertion/.../5000` + Libero | ✅ gpu yaml |
| RL Token | `checkpoints/rl_token.pt` → sft5000 软链 | ✅ |
| 旧 actor 备份 | `checkpoints/rl_actor.pt.presft_bak` | ✅ |

**运行观察（预期，非 bug）**：

- GPU infer **1–2 min/chunk** → 臂 execute_prefix=10 @ 20Hz 仅动 ~0.5s。
- 偶发 `keepalive ping timeout`；`buffer_size` 仍涨可继续。
- reference 阶段 success 低正常；fail/timeout 也进 replay。

**下次续跑 `[主机]`**：

1. GPU 起 server → **重载** actor/critic，但 **buffer 内存清空**（未做持久化）。
2. 工控机隧道 + actor；config 用 `plug_insertion.yaml`。
3. **勿切 policy**，直到 reference 验稳。

---

### 3.2 工控机真机修复明细 `[工控机]`（2026-07-10）

| 问题 | 处理 |
|------|------|
| 臂不动 / 位移极小 | `action_is_physical: true` |
| VLA infer 超时 | `rl_server` 合并单次 VLA；`gpu_client` timeout **180s** |
| Reset 位姿偏差大（~3 cm） | 弃用 101 demo reset → **SFT init-cube + external 子进程** |
| Reset 与 `reset_to_init.sh` 不一致 | `reset_method: external` |
| ZMQ 5555 占用 | 外部 reset 前 **suspend** actor 相机子进程 |
| 首帧 VLA 位姿偏低 / 画面旧 | `post_reset_settle_sec` + 排空 RealSense 缓存 + warmup |
| `franka_hand.yaml` 找不到 | 子进程 `cwd` → `rlt_project/rlt_reproduce` |

---

### 3.3 数据与配置变更 `[工控机]` + `[主机]`

| 项 | 说明 |
|----|------|
| 旧 critical demo 101 eps | → `legacy_critical_demo_json_101eps` / `legacy_critical_npz_101eps`（**不再用于 reset 或训练**） |
| 当前 SFT 数据 | `rlt_project/rlt_reproduce/data/sft/plug_insertion`（**122+ eps**） |
| Init cube 底面中心 z | ≈ **0.202 m** |
| Actor reset | `reset_mode: workspace`，`reset_config: configs/sft_plug_insertion.yaml` |
| Stage-0 legacy | 51 success + 50 fail NPZ（历史；fail **不进** SFT 监督） |

---

## 4. SFT VLA 与 RL Token 切换 `[主机]`（2026-07-08，务必先读）

**背景**：pi0.5 已在 plug_insertion 上 **SFT**；`rl_token.pt` 在 **SFT 版 VLA embedding** 上训练。旧非 SFT RL Token 已弃用。上真机前 **先用 reference 验证 SFT**（学长建议）。

**GPU 部署树变更**（`rlt_project/rlt_reproduce/`，`rl_server` 实际读取）：

| 变更 | 文件 | 内容 |
|------|------|------|
| VLA 换 SFT | `configs/plug_insertion_gpu.yaml` | `checkpoint → .../5000`；`config_name: pi05_plug_insertion`；`input_format: libero` |
| 执行模式 | 同上 | `inference.mode: reference` |
| Libero obs | `obs_builder.py` 等 | `ws_obs_to_libero` + `build_observation_for_format` |
| 软链 | `checkpoints/pi05_plug_insertion` | 指向训练树 |
| 旧 actor | `rl_actor.pt` | → `rl_actor.pt.presft_bak` |

> **为何 5000 非 4000**：orbax `max_to_keep=1` + `keep_period=5000`，4000 已删；5000 embedding 与 4000 近乎一致。

**✅ 2026-07-08 后续**：step-5000 SFT 重抽 embedding + 重训 RL Token。

- 训练配置：`configs/plug_insertion_sft5000.yaml`；脚本 `scripts/run_rl_token_sft5000.sh`（4 卡 precompute → GPU1 训练）。
- 新 token：`checkpoints/sft5000_rltoken/rl_token.pt`（run `rl_token_run_20260708T075448Z`，`best_val_L_ro≈0.2363`、`final_train_L_ro≈0.2325`）。比 sft4000 版（`rl_token_run_20260707T225233Z`，best val L_ro≈0.247）略好。
- **已切换在线 token 软链**：
  - 旧：sft4000 实体 `…/checkpoints/rl_token.pt`（保留未删）
  - 新：`…/checkpoints/sft5000_rltoken/rl_token.pt`
  - 命令：`ln -sfn "…/checkpoints/sft5000_rltoken/rl_token.pt" "…/smq_jgy_deploy/smq&jgy/rlt_project/rlt_reproduce/checkpoints/rl_token.pt"`
- **未改动**：SFT ckpt 软链 `pi05_plug_insertion`；**[工控机]** `configs/plug_insertion.yaml`（不含 `rl_token`/learner）。
- **回滚 sft4000**：`ln -sfn "…/checkpoints/rl_token.pt" "…/smq_jgy_deploy/.../checkpoints/rl_token.pt"`（指回 sft4000 实体文件）。

**一致性铁律**：在线 VLA（checkpoint + config_name + input_format）必须与训 `rl_token.pt` 时 **完全一致**。

**验证 SFT = Stage-2d**：GPU `mode: reference` → 工控机 `run_deoxys_actor.sh --episodes 1 --max-steps 1200` → 看 `policy_mode=reference`、`ref_norm`、臂 motion。

---

## 5. 阶段总览 `[共通]`

| 阶段 | 内容 | 状态 |
|------|------|------|
| Stage-0 | 数据采集 51 success + 50 fail | ✅ legacy |
| Stage-1 | RL Token 离线 `rl_token.pt` | ✅ sft5000 |
| Stage-2a | 双机 WebSocket（MOCK + 隧道） | ✅ |
| Stage-2b | GPU 部署 smq&jgy 代码 | ✅ |
| Stage-2c | 真实 pi05 + 双相机 infer | ✅ |
| Stage-2d | reference 真机 rollout | 🟡 通路 OK；success 少；motion 卡 |
| **Stage-3** | replay warmup + TD3 | 🟡 buffer 578，train 395 |
| **Stage-4** | `mode: policy` 控臂 | ⬜ |

**当前阻塞 / 风险 `[主机]`**：

1. Infer 延迟 1–2min → motion 卡、WS timeout。
2. Replay 不持久化 → 重启丢 buffer。
3. 网络 → 继续 SSH 隧道 + `127.0.0.1`。
4. Policy 勿早切。

---

## 6. 为什么 Online RL「还没开始 / 已在训」？ `[主机]`

| 层级 | 含义 | 状态 |
|------|------|------|
| 双机通路 | ping / infer 通 | ✅ |
| Reference rollout | 真臂 + pi05 + 双相机 | 🟡 |
| Replay warmup + TD3 | buffer ≥ warmup | ✅ 578/500 |
| Policy 控臂 | Actor 出动作 | ⬜ |

**训练开始** = Stage-3（`learner metrics`）。**Policy 阶段** = Stage-4。

> 工控机 yaml 无 `inference.mode`；以 **[主机]** gpu yaml 与日志为准。

---

## 7. 各阶段「怎样才算通过」`[共通]`

### 7.1 Reference rollout（Stage-2d）

**[主机] GPU 终端应出现**：

```text
Loaded RL token from checkpoints/rl_token.pt
Loaded openpi VLA from ...
Inference mode: reference
RL server listening on ws://0.0.0.0:8765
```

**[工控机] 命令**（无 MOCK、无 `--no-cameras`）：

```bash
export GPU_SERVER_HOST=127.0.0.1
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 1 --max-steps 1200
```

**通过标准**：

- [x] 首步 meta：`policy_mode=reference`、`z_rl_norm`、`ref_norm`
- [x] 无 `KeyError: observation/exterior_image_1_left`、无 OOM
- [x] 臂 motion 大致合理
- [x] s/f 结束 episode
- [x] 下一 ep 前 **自动 external init-cube reset**（2026-07-10 起；**非** legacy demo reset）

### 7.2 Online RL 训练开始（Stage-3）

- GPU 保持 `reference`（或 `reference_noise`）
- 工控机多 ep，s/f 标 reward
- `buffer_size` 涨；≥ warmup 后出现 `learner metrics`

### 7.3 Policy 接管（Stage-4）`[主机]`

```yaml
inference:
  mode: policy
  deterministic: true
```

重启 server；infer meta 为 `policy_mode=policy`；臂执行 Actor 输出。

---

## 8. 与 RLT 论文差异 `[工控机]`（仍成立）

| 维度 | 论文 | 我们 |
|------|------|------|
| 粗阶段 | VLA 从较远接近 | **init-cube reset 直接到 critical 高度** |
| Critical 起点 | 演示初始 | **SFT 标定 init 立方体底面**（z≈0.202m） |
| VLA→RL 切换 | 自动 | **改 [主机] GPU yaml** `reference` → `policy` |
| Reward | 人工 sparse | 同：**s/f** |

---

## 9. Reference rollout 常见阻塞 `[主机]`

| 原因 | 现象 | 处理 |
|------|------|------|
| GPU OOM | server/infer OOM | 换空闲 24GB 卡 |
| 无相机图 | KeyError image | 开 RealSense |
| 网络 | `:8765` refused | 隧道 + `127.0.0.1` |
| MOCK | 非真实 VLA | 必须真实 pi05 + JPEG |

**MOCK 通路通 ≠ reference 真机通过。**

---

## 10. 代码与配置风险审查 `[主机]`（2026-07-05，7/10 复核）

### 10.1 总体判断

| 类别 | 阻止开跑？ | 学不到东西？ |
|------|------------|-------------|
| GPU OOM | ✅ | — |
| `actor_loop` 首步 bug | ✅（已修） | — |
| `mode: policy` + 随机 Actor | ✅ | ✅ |
| VLA action / proprio mismatch | 可能 | ✅ **很可能** |
| 超时太短 | — | ✅ timeout fail |
| 无 ckpt save | — | ⚠️（actor/critic 已可存；buffer 仍不持久） |

### 10.2 🔴 严重（多数已修，留档）

**（1）`actor_loop.py` 首步 `action_chunk` 顺序** — ✅ 已修。原 bug（`[主机]` 留档）：

```python
# 问题：action_chunk 在打印之后才赋值
if step == 0 and result.meta:
    console.print(... action_chunk[0] ...)  # ← 未定义
action_chunk = result.action_chunk           # ← 应挪到打印之前
```

**（2）`inference.mode: policy` 与进度不符** — ✅ 已改 reference；旧 actor → `.presft_bak`。

**（3）VLA 动作空间 vs OSC** — 仍须 reference 真机观察；domain gap 表（`[主机]`）：

| | 采集 / 真机执行 | openpi `pi05_droid` 输出 |
|--|----------------|-------------------------|
| 控制 | Deoxys **OSC 笛卡尔增量** × `action_scale` | DROID 预训练空间 |
| proprio | 8 维末端位姿 + 夹爪 | 硬映射为 `joint_position`（`obs_builder.py`） |

代码中 **没有** 将 VLA 输出再映射回 OSC 空间；reference 直接 `env.step(action)`。不合理则需 VLA 微调、换 config 或加 action 变换层。

### 10.3 🟠 高影响

- **Episode 超时**：见 §11；工控机 yaml 曾 200/600，推荐 1200–2400。
- **Replay action 单步 tile 成 chunk**：工控机每步发 **7 维单步**；GPU `rl_server._as_action_chunk()` 将单步 **复制** 为 `(chunk_length, 7)` 再进 Critic。Policy 模式 Actor 输出 **完整 chunk**，与 replay 存储形式不完全一致（实现简化，可能拖慢收敛）。
- **稀疏 reward + batch 256**：仅 s/f 一步 1/0；`warmup_steps: 256` 且 `batch_size: 256` → 方差大；论文常用 warmup **500**（当前 578 已超）。
- **在线训练 ckpt**：✅ actor/critic 已 save；❌ buffer 仍不 save。`rl_server.py` 原仅 load、无 save（7/5 审查）；7/10 已落盘 `rl_actor.pt`/`rl_critic.pt` 及 `online_rl/rl_*_step*.pt`。

### 10.4 🟡 运维

| 问题 | 影响 |
|------|------|
| GPU OOM | server 起不来 |
| SSH 隧道 | 断则断连 |
| 双代码树 | 改一边忘同步 |
| `--no-cameras` | KeyError |
| MockGPUClient | 不能代表真链路 |

### 10.5 🟢 设计限制

- HIL 稀疏 reward；强依赖 reference 质量
- RL Token 离线 success embedding vs 在线 live VLA：同 ckpt OK；光照 shift 会有
- **fail NPZ 不进监督训练**；legacy 仅 demo reset（已废弃 reset 路径）
- VLA 微调非 RLT 硬性前提
- Reference **无反向传播**；数据训 Actor/Critic

### 10.6 修复优先序（真机前）

| 优先级 | 项 | 状态 |
|--------|-----|------|
| P0 | `action_chunk` 顺序 | ✅ |
| P0 | `inference.mode: reference` | ✅ |
| P1 | `max_steps` → 1800–2400 | 推荐 CLI/`yaml` |
| P1 | reference 真机验证 motion | 🟡 进行中 |
| P2 | warmup 500+；再切 policy | buffer 已 578 |
| P3 | buffer 持久化 | 待开发 |

---

## 11. Episode 超时与人机反应时间 `[主机]`

### 11.1 机制

- `online_rl.max_steps_per_episode`（工控机 yaml 当前可能 **200**；历史文档写 600）
- `control_hz: 20` → 每步 sleep 0.05s
- `step >= max_steps` → timeout，reward=0，自动 reset 下一 ep

### 11.2 墙钟 vs 步数

```text
墙钟 ≈ (步数 × 0.05s) + (infer 次数 × 1~2min)
```

| 场景 | 建议 max_steps @20Hz |
|------|---------------------|
| 首次 reference 试跑 | **1200** (~60s sleep) |
| 正常 HIL | **1800–2400** |
| 工控机阶段 A smoke | **200**（仅验通路；不够 HIL） |
| CLI 短测 | `--max-steps 60` |

```bash
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 5 --max-steps 2400
```

### 11.3 操作提示

- 终端 **TTY 聚焦**，否则 s/f 无效 → 只剩 timeout
- timeout = fail，污染 replay；尽早按 s/f
- demo 录制 critical 仅 2.5–8s，但 **online 需人观察**，必须留余量

---

## 12. 推荐执行流程 `[共通]`

```
Phase 0 — 解除阻塞 + 代码修复（§10.6）
  □ GPU：CUDA_VISIBLE_DEVICES=N → start_gpu_rl_server.sh（reference）
  □ 工控机：start_ssh_tunnel.sh；GPU_SERVER_HOST=127.0.0.1

Phase 1 — Reference rollout（Stage-2d）
  □ Deoxys + 双 RealSense
  □ run_deoxys_actor.sh --episodes 1 --max-steps 1200
  □ 验 infer、臂动、s/f、external reset

Phase 2 — 攒 replay + TD3（Stage-3）
  □ 保持 reference；--max-steps 2400；多 ep s/f
  □ buffer ≥ warmup → learner metrics

Phase 3 — Policy（Stage-4）
  □ [主机] gpu yaml: mode: policy，重启 server
  □ [工控机] 命令不变

Phase 4 — 可选
  □ 再采 success + VLA 微调；重训 RL Token；NAT 去隧道
  □ [工控机] export_smq_for_gpu.sh 同步代码
```

---

## 13. 三终端分工 `[共通]`

| 终端 | 机器 | 命令 |
|------|------|------|
| A | GPU | `CUDA_VISIBLE_DEVICES=1 bash scripts/start_gpu_rl_server.sh` |
| B | 工控机 | `bash scripts/gpu/start_ssh_tunnel.sh` |
| C | 工控机 | `export GPU_SERVER_HOST=127.0.0.1` + `CONFIRM=1 bash scripts/run_deoxys_actor.sh ...` |

---

## 14. Episode 生命周期与故障排查

### 14.1 生命周期 `[主机]`

```
[自动] external init-cube reset
[自动] GPU infer → 执行 action chunk（循环）
[你]   s / f ；或 step ≥ max_steps → timeout（fail）
[自动] transition（done=true）→ 下一 ep
```

| 键 | reward | 结束 ep |
|----|--------|---------|
| s | 1.0 | 是 |
| f | 0.0 | 是 |
| q | 0.0 | 退出 run |
| timeout | 0.0 | 是 |

### 14.2 Reset 故障排查 `[工控机]`

| 日志 / 现象 | 处理 |
|-------------|------|
| `FileNotFoundError: franka_hand.yaml` | 确认子进程 `cwd=.../rlt_reproduce` |
| `ZMQ port 5555 in use` | 外部 reset 前停相机；或 `free_deoxys_client.sh` |
| `pos_err` > 3 cm 或 z 远离 0.202 | 单独跑 `reset_to_init.sh`；查碰撞 / `xy_half_range_m` |
| 子进程 `CalledProcessError` | 手动 `reset_to_init.sh` 看 traceback |

### 14.3 Online infer 故障排查 `[工控机]` + `[主机]`

| 现象 | 处理 | 来源 |
|------|------|------|
| 动一下停很久 | 正常；每 chunk 一次 GPU infer | 工控机 |
| `TimeoutError` on infer | 首帧可达 180s；查 server、隧道 | 工控机 |
| 位移很小像「蹭」 | 查 reset 后 z≈0.202；buffer 够后试 policy | 工控机 |
| warmup 后更卡 | GPU `train_step` 叠加；可减 `update_to_data_ratio` | 工控机 |
| 一顿一顿 | chunk 无平滑 + infer 慢 | 主机 |
| `keepalive ping timeout` | infer 阻塞；buffer 仍涨可继续 | 主机 |
| buffer 重启变 0 | 无持久化；重攒 warmup | 主机 |
| 臂乱动 | 误切 `mode: policy` | 主机 |

---

## 15. 关键配置摘要

### 15.1 工控机 Actor `[工控机]` — `configs/plug_insertion.yaml`

```yaml
online_rl:
  reset_mode: workspace
  reset_method: external
  reset_config: configs/sft_plug_insertion.yaml
  post_reset_settle_sec: 1.0
  post_reset_warmup_steps: 25
  post_reset_max_pos_err_m: 0.03
  action_is_physical: true
  # max_steps_per_episode: 建议 1200–2400（当前可能仍为 200，CLI 可覆盖）
  # infer_timeout_sec: 180
```

### 15.2 GPU Server `[主机]` — `configs/plug_insertion_gpu.yaml`

- `vla.checkpoint` → SFT 5000
- `vla.input_format: libero`
- `inference.mode: reference`（Stage-4 改 `policy`）
- `online_rl.warmup_steps`（500 或 256，以 yaml 为准）

> GPU 用 **gpu yaml** 的 `inference.mode`；工控机 **plug_insertion.yaml** 无此字段。

---

## 16. 关键文件 `[共通]`

| 用途 | 路径 | 来源 |
|------|------|------|
| Actor 配置 | `configs/plug_insertion.yaml` | 工控机 |
| GPU 配置 | `configs/plug_insertion_gpu.yaml` | 主机 |
| Reset 配置 | `configs/sft_plug_insertion.yaml` | 工控机 |
| 外部 reset | `src/rlt/hardware/deoxys/collection_reset.py` | 工控机 |
| 工控机 actor | `src/rlt/scripts/actor_loop.py` | 共通 |
| GPU client | `src/rlt/rl/gpu_client.py` | 共通 |
| GPU server | `src/rlt/scripts/rl_server.py` | 主机 |
| embedding | `src/rlt/vla/embedding_extractor.py` | 主机 |
| Reset 脚本 | `scripts/reset_to_init.sh` | 工控机 |
| Actor 入口 | `scripts/run_deoxys_actor.sh` | 工控机 |
| GPU 启动 | `scripts/start_gpu_rl_server.sh` | 主机 |
| 同步 GPU | `scripts/export_smq_for_gpu.sh` | 工控机→主机 |

---

## 17. 数据与训练要点 `[共通]`

| topic | 结论 |
|-------|------|
| success vs fail NPZ | **训练只用 success**；fail 曾用于 demo reset（已弃用该 reset 路径） |
| 1:1 采集 | **非必须**；多 success 更有价值 |
| 微调 VLA 目的 | 让 frozen teacher 会任务；online 改进靠 TD3 |
| reference 是否学习 | **否**；无 backward；数据训 Actor/Critic |
| 能否 skip reference 模式 | 代码可 policy；仍须攒 transition；reference 用 VLA 安全采数 |
| z_rl | GPU 在线算；满 warmup 喂 TD3 |
| online embedding | `rl_server` → `embedding_extractor` → `extract_embeddings()` |
| SSH 隧道 | **隧道 + 127.0.0.1 即可** |
| 切 policy 阈值 | [主机] warmup 500 已达；[工控机] 原建议 buffer **750** 再 smoke policy |

---

## 18. 下一步行动清单（2026-07-11）`[共通]`

**[主机] GPU**

- [x] 3090 跑通；warmup 满 500；actor/critic ckpt
- [ ] 续跑 reference；增大 `train_steps`
- [ ] 勿早切 policy
- [ ] buffer 持久化、infer 优化（可选）

**[工控机]**

- [ ] SSH 隧道常开
- [ ] 同步 deploy 树 + 双 yaml
- [ ] reference 多 ep；TTY 按 s/f
- [ ] `max_steps` 1200–2400
- [ ] `export_smq_for_gpu.sh` 按需同步

**[共通] 工程**

- [ ] infer 加速 / 减重复 forward
- [ ] chunk 平滑
- [ ] 治具固定（工控机 P3）

---

## 19. 常见报错速查 `[主机]`

| 现象 | 处理 |
|------|------|
| CUDA OOM | 换空闲卡 |
| `KeyError: exterior_image_1_left` | 开相机 |
| `:8765` refused | 隧道 / 查 server |
| `stdin 非 TTY` | 交互终端 |
| 突然 reset | timeout → 加大 max_steps |
| `NameError: action_chunk` | 已修 |
| 臂乱动 | 改 reference |
| `keepalive ping timeout` | 等 2–3min；buffer 涨可继续 |

---

## 20. 历史备忘

<details>
<summary>[工控机] 2026-07-05：demo reset（ep_00022）— 已废弃</summary>

曾用 101 条 NPZ critical demo 的 `proprio[0]` 做 pin reset（`ep_00022`），容差 6 cm 偏大、与 SFT 采集不一致。  
**2026-07-10 起全面改为 SFT init-cube external reset**，见 §3.2。

</details>

---

## 21. 一句话总结 `[共通]`

**2026-07-10：双机 reference 真机已跑通，Stage-3 TD3 已启动（buffer 578、train 395、actor/critic ckpt）；仍为 reference；success 少与 motion 卡由 infer 延迟导致。续跑：[主机] 起 GPU server（`plug_insertion_gpu.yaml`），[工控机] 隧道 + actor 继续 reference 攒数据；验稳后再切 policy（Stage-4）。**

---

## 22. 附录 — 历史行动清单（2026-07-05，`[主机]` 留档）

<details>
<summary>Phase 0 详细 checklist（多数已完成）</summary>

```
Phase 0 — 解除阻塞 + 代码修复
  □ 修 actor_loop.py action_chunk bug          ✅
  □ yaml: inference.mode → reference             ✅
  □ yaml: max_steps_per_episode → 2400           进行中
  □ GPU：协调 1× 空闲 24GB                       ✅ CUDA_VISIBLE_DEVICES=1
  □ GPU：start_gpu_rl_server.sh                  ✅
  □ 确认 ss -tlnp | grep 8765
  □ 工控机：start_ssh_tunnel.sh
  □ export GPU_SERVER_HOST=127.0.0.1
```

</details>
