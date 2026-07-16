# Plug Insertion Online RL — 工作文档

> 最后更新：**2026-07-16**  
> 任务：Franka FR3 + Franka Hand + 2× RealSense，插插头 online RL  
> 操作指南：[`docs/ONLINE_RL_ROBOT.md`](../ONLINE_RL_ROBOT.md)

---

## 0. 2026-07-16 — RLT 对齐修复（Phase 1–4 + 安全）务必先读

现象：`reference` 模式正常，切 `policy` 后臂大幅乱动、急停。根因不是「actor 从零初始化」（论文本就如此），而是缺少论文对 actor 的**约束**：动作空间硬边界、reference 锚定、warmup 门控、正确的 chunk transition。以下为已落地的修复。**不改 SFT / RL Token 流水线，不把 actor 改成 residual policy。**

### 0.1 运行时源码树映射（关键，避免改错树）

两份 `rlt` 源码树属于**同一个 git 仓库**（`smq&jgy`），但运行时分工不同：

| 组件 | 运行时加载的树 | 路径 | 说明 |
|------|----------------|------|------|
| `actor_loop.py`（工控机 rollout） | **Tree C** | `smq&jgy/src/rlt/scripts/actor_loop.py` | `_env.sh` 把 `smq&jgy/src` 放 `PYTHONPATH` **最前**，覆盖 Tree B 同名文件 |
| `gpu_client.py`（工控机→GPU 通信） | **Tree C** | `smq&jgy/src/rlt/rl/gpu_client.py` | 同上，运行时权威副本 |
| `rl_server.py`（GPU 服务） | **Tree B** | `rlt_project/rlt_reproduce/src/rlt/scripts/rl_server.py` | `run_rl_server.sh` 只用 Tree B `src` |
| `learner.py`（TD3） | **Tree B** | `…/rlt_reproduce/src/rlt/rl/learner.py` | GPU 侧 |
| `replay_buffer.py` | **Tree B** | `…/rlt_reproduce/src/rlt/rl/replay_buffer.py` | GPU 侧 |
| `inference_policy.py`（reference/policy 门控 + 锚定） | **Tree B** | `…/rlt_reproduce/src/rlt/rl/inference_policy.py` | GPU 侧 |
| `deoxys_env.py`（安全 clamp） | **Tree B** | `…/rlt_reproduce/src/rlt/hardware/deoxys/deoxys_env.py` | 工控机执行时 `cd` 到 Tree B（`RLT_ROOT`），env 从 Tree B 加载 |

> **同步铁律**：`actor_loop.py` 与 `gpu_client.py` 的**功能代码**在 Tree B / Tree C 两份保持一致（仅 docstring 与 `main()` 里 `Path(__file__).parents[N]` 深度不同）。改其一必须同步另一份。GPU 配置 `plug_insertion_gpu.yaml` 在两树各有一份，`run_rl_server.sh` 读 **Tree B** 那份；Tree C 那份为镜像，已同步以防误用。

### 0.2 Phase 1 — 机器人硬安全 clamp（`deoxys_env.py`）

`DeoxysEnv.step` 在 `FrankaInterface.control()` 之前对**物理 EE 增量**做硬限幅（`_apply_safety_limits`），**reference / policy 两种模式都生效**，是最后一道防线：

- 平移每步 ≤ `max_trans_delta_m = 0.02 m`
- 旋转（axis-angle，弧度）每步 ≤ `max_rot_delta_rad = 0.1 rad`
- 夹爪 clip 到 `[-1, 1]`
- NaN/Inf → 保持当前位姿（发零增量），不下发危险指令
- 记录 raw / clipped norm 与是否触发 clamp

单位与机器人 action space 一致（OSC_POSE 的笛卡尔增量，弧度制，非四元数/角度）。默认值即 `DeoxysEnvConfig` 默认，已在**工控机实际加载的** `configs/plug_insertion.yaml` 用显式 `safety:` 块写出（见 §0.7）。

### 0.3 Phase 2 — 论文式 policy 动作锚定（`inference_policy.py`）

actor 架构不变，仍输出**绝对**动作 chunk `a ~ pi(a | x, ref)`（**不是** reference+residual）。部署时把 actor 相对 reference 的偏移**限幅**，语义即论文「actor 在 VLA 行为上做局部改进」：

```
action = reference + clip(actor - reference, ±delta)
```

- `max_dev_trans_m = 0.01 m`，`max_dev_rot_rad = 0.05 rad`，`max_dev_grip = 1.0`
- 记录 ref_norm / actor_norm / 偏移距离 / 被限幅量

### 0.4 Phase 3 — warmup 门控 + policy ramp（`inference_policy.act_gated` / `rl_server`）

执行模式**由 replay buffer transition 数**决定，忠实 Algorithm 1（不再只看 config 开关）：

```
buffer_size < warmup_steps                 → reference（alpha=0）
warmup ≤ buffer < warmup+ramp              → 线性 ramp：alpha=(buffer-warmup)/ramp
buffer_size ≥ warmup+ramp                  → anchored policy（alpha=1）
```

- `warmup_steps = 500`，`ramp_steps = 500`（保持，勿降到 250）
- 手动 override 仍保留：`inference.mode = reference | reference_noise | policy`；设为 `reference` 时**无视 buffer 恒执行 reference**（首跑用它做数据 QA 门）。
- 日志：buffer_size / warmup 阈值 / 当前 alpha / 执行模式 / ref&policy 动作统计。

### 0.5 Phase 4 — 真实 chunk transition（`actor_loop.py` / `rl_server.py` / `replay_buffer.py` / `learner.py`）

修复前（错误）：工控机每步发单步动作，GPU 把它 **tile** 成 `(C,7)`；`next_state` 只差 1 步却用 `γ^C` 折扣；无 `next_reference_action`。

修复后（论文式），每条 transition 为：

```
(x_s, a_{s:s+C}, ref_{s:s+C}, ref_{s+C:s+2C}, R=Σ_{k=0}^{C-1} γ^k r_{s+k}, x_{s+C}, done)
```

- `actor_loop` 按步收集整段 episode 流，episode 结束时用 `build_chunk_transitions` 按 `subsample_stride=2` 组装**真实** chunk（无 tile）。
- 新增 `gpu.encode()` 消息：对中间步做 per-step 状态编码，得到精确相隔 C 步的 `x_s` 与 `x_{s+C}`。
- `rl_server` 严格校验 chunk 形状 `(C=10, 7)`（不合法即报错），把 per-step reward 聚合成 `R`。
- `replay_buffer.Transition` 增加 `next_reference_action`；新增 `clear()`。
- `learner`：critic target `a' ~ pi(x', ref')` 用**下一状态**的 reference；折扣 `γ^C` 与真实 C 步时间间隔一致。
- **终局 chunk padding**：动作/reference 用**最后一个有效步**补齐，reward 用 **0** 补齐——保证稀疏 success reward 不被污染。

### 0.6 checkpoint 保留 / 丢弃

| 保留（不动） | 丢弃（重训前删） |
|--------------|------------------|
| `checkpoints/rl_token.pt`（→ sft5000 真 token 软链） | `checkpoints/rl_actor.pt` |
| `checkpoints/pi05_base`（SFT base 软链） | `checkpoints/rl_critic.pt` |
| `checkpoints/pi05_plug_insertion`（SFT VLA 软链） | `checkpoints/online_rl/`（全部快照） |

原因：transition 定义变了，旧 actor/critic 与旧 replay 快照按错误语义训练，必须从零重训 actor/critic（论文允许 from scratch）；SFT VLA 与 RL Token 未受影响，保留。删除前先整目录备份到 `checkpoints`（gitignored）下的 `backups/phase5_pre_reset_*`。

### 0.7 安全重启流程（重训准备）

1. **备份**：`backups/phase5_pre_reset_<ts>/` 已存全部待删 ckpt + 反解引用的 `rl_token_REAL.pt`。
2. **删除**：仅删 `rl_actor.pt` / `rl_critic.pt` / `online_rl/`（路径含 `&`，命令必须加引号）。
3. **从零初始化**：`rl_server` 见不到 `rl_actor.pt`/`rl_critic.pt` 即随机初始化 actor/critic；`rl_token.pt`、SFT VLA 正常加载。
4. **清空 buffer**：replay buffer 仅内存，server 启动即为空。
5. **首跑强制 reference**：GPU 两份 `plug_insertion_gpu.yaml` 均设 `inference.mode: reference`；直到 (a) ≥500 条有效 chunk transition、(b) 形状/reward/done 校验通过、(c) 至少 2–3 个成功 reference episode，再改回 `inference.mode: auto` 并**重启** server（重启会清空 buffer→auto 从 0 重新 warmup 500 后 alpha 0→1 平滑 ramp）。
6. **启动命令**（路径含 `&` 已加引号）：
   - GPU：`cd "…/smq&jgy/rlt_project/rlt_reproduce" && bash scripts/run_rl_server.sh configs/plug_insertion_gpu.yaml`
   - 工控机：`cd "…/smq&jgy" && CONFIRM=1 bash scripts/run_actor_loop.sh`
   - 健康检查：`cd "…/smq&jgy" && bash scripts/ping_gpu_server.sh`（看 `buffer_size` 与 `exec_state: override:reference`）

---

## 1. 系统架构（当前）

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

| 机器 | 角色 |
|------|------|
| 工控机 `vla` | deoxys、SFT 采集、`actor_loop`、SSH 隧道 |
| GPU | `rl_server`（VLA + RL Token + TD3） |

工控机 **不跑 VLA**；每步传 proprio + 双相机 JPEG，embedding 在 GPU 算。

---

## 2. 2026-07-10 进度

### 已完成

| 问题 | 处理 |
|------|------|
| 臂不动 / 位移极小 | `action_is_physical: true`，VLA 物理增量只除 controller scale |
| VLA infer 超时 | `rl_server` 合并为单次 VLA；`gpu_client` 默认 timeout 180s |
| Reset 位姿偏差大（~3 cm） | 弃用 101 条 demo reset；改用 **SFT init-cube** + **external 子进程** |
| Reset 与 `reset_to_init.sh` 不一致 | `reset_method: external`，与手动脚本同流程 |
| ZMQ 5555 占用 | 外部 reset 前 `suspend` actor 相机子进程 |
| 首帧 VLA 位姿偏低 / 画面旧 | `post_reset_settle_sec` + 排空 RealSense 缓存 + warmup |
| `franka_hand.yaml` 找不到 | 子进程 `cwd` 修正为 `rlt_project/rlt_reproduce` |

### 数据 / 配置变更

- 旧 critical demo（101 eps）→ `legacy_critical_demo_json_101eps` / `legacy_critical_npz_101eps`（**不再用于 reset 或训练**）
- 当前 SFT 数据：`rlt_project/rlt_reproduce/data/sft/plug_insertion`（122+ eps）
- Actor yaml：`online_rl.reset_mode: workspace`，`reset_config: configs/sft_plug_insertion.yaml`
- Init cube 底面中心 z ≈ **0.202 m**

### 待做（按优先级）

| P | 项 |
|---|-----|
| **P0** | 跑通阶段 A：`CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 5 --max-steps 200`，确认 reset → VLA → 臂动 → s/f |
| **P1** | `bash scripts/export_smq_for_gpu.sh` 同步 GPU（单次 infer、checkpoint 保存） |
| **P2** | buffer ≥ 750 后切 `inference.mode: policy`，smoke 1 episode |
| **P3** | 插座/相机治具长期固定；评估是否加姿态对齐 |

---

## 3. 关键配置摘要

`configs/plug_insertion.yaml`（工控机 actor）：

```yaml
online_rl:
  reset_mode: workspace
  reset_method: external
  reset_config: configs/sft_plug_insertion.yaml
  post_reset_settle_sec: 1.0
  post_reset_warmup_steps: 25
  post_reset_max_pos_err_m: 0.03
  action_is_physical: true
```

---

## 4. 与 RLT 论文差异（仍成立）

| 维度 | 论文 | 我们 |
|------|------|------|
| 粗阶段 | VLA 从较远接近 | **init-cube reset 直接到 critical 高度** |
| Critical 起点 | 演示初始 | **SFT 标定的 init 立方体底面**（z≈0.202m） |
| VLA→RL 切换 | 自动 | **改 GPU yaml** `reference` → `policy` |
| Reward | 人工 sparse | 同：**s/f** |

---

## 5. 关键文件

| 用途 | 路径 |
|------|------|
| Actor 配置 | `configs/plug_insertion.yaml` |
| Reset 配置 | `configs/sft_plug_insertion.yaml` |
| 外部 reset | `rlt/.../deoxys/collection_reset.py` |
| 工控机 actor | `src/rlt/scripts/actor_loop.py` |
| GPU client | `src/rlt/rl/gpu_client.py` |
| GPU server | `rlt_project/rlt_reproduce/src/rlt/scripts/rl_server.py` |
| Reset 脚本 | `scripts/reset_to_init.sh` |
| Actor 入口 | `scripts/run_deoxys_actor.sh` |

---

## 6. 故障排查

### Reset

| 日志 / 现象 | 处理 |
|-------------|------|
| `FileNotFoundError: franka_hand.yaml` | 已修；若再现，确认子进程 log 里 `cwd=.../rlt_reproduce` |
| `ZMQ port 5555 in use` | 确认 actor 在外部 reset 前已停相机；或 `bash scripts/free_deoxys_client.sh` |
| `pos_err` > 3 cm 或 z 远离 0.202 | 查 `reset_to_init.sh` 单独是否 OK；查碰撞 / `xy_half_range_m` |
| 子进程 `CalledProcessError` | 先手动 `bash scripts/reset_to_init.sh` 看完整 traceback |

### Online infer

| 现象 | 处理 |
|------|------|
| 动一下停很久 | 正常：每 chunk 一次 GPU infer（VLA 重）；隧道 / `nvidia-smi` |
| `TimeoutError` on infer | 首帧可达 180s；确认 GPU server 存活、`GPU_SERVER_HOST=127.0.0.1` |
| 位移很小像「蹭」 | 查 reset 后 z≈0.202、external 画面与采集一致；buffer 够后试 policy |
| warmup 后更卡 | GPU 端 `train_step` 叠加；可暂减 `update_to_data_ratio` |

---

## 7. 历史备忘（已解决，仅留档）

<details>
<summary>2026-07-05：demo reset（ep_00022）讨论 — 已废弃</summary>

曾用 101 条 NPZ critical demo 的 `proprio[0]` 做 pin reset（`ep_00022`），容差 6 cm 偏大、与 SFT 采集起点不一致。  
**2026-07-10 起全面改为 SFT init-cube external reset**，见上文 §2。

</details>
