# Plug Insertion Online RL — 工作文档

> 最后更新：**2026-07-10**  
> 任务：Franka FR3 + Franka Hand + 2× RealSense，插插头 online RL  
> 操作指南：[`docs/ONLINE_RL_ROBOT.md`](../ONLINE_RL_ROBOT.md)

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

---

## 8. 算法审查 & Policy 切换备忘（2026-07-10 23:17）

> 只读代码审查结论；目标：**policy 能跑起来不报错**，并明确算法侧风险。  
> 相关代码：`rl_server.py`、`inference_policy.py`、`learner.py`、`actor_loop.py`、`plug_insertion_gpu.yaml`。

### 8.1 一句话结论

- **切 `policy` 本身不会让 Python 走新的崩溃路径**；`reference` 已稳定跑通时，`policy` 大概率也不会因「模式切换」而 exception。
- 真正要担心的是：**未充分训练的 Actor 输出乱动作**（真机危险，但不是报错），以及 **TD3 实现与论文假设不完全一致**（影响学得好不好，不直接报错）。

### 8.2 Policy 切换机制（代码层面已通）

```text
工控机 infer → GPU VLA 编码 state → inference.mode 分支
  reference      → 执行 ref_chunk
  reference_noise → 执行 action_chunk（ref + noise）
  policy         → 执行 action_chunk（Actor 输出）
→ env.step → transition（写 buffer + 可选 TD3）
```

| 组件 | 行为 |
|------|------|
| GPU `RLTInferencePolicy.act` | `policy` 模式：`actor(state, ref)` → `(10, 7)` |
| 工控机 `actor_loop` | `policy_mode == "reference"` 用 ref，否则用 `action_chunk` |
| 门控 | **无** buffer / train_steps 门控；改 yaml 重启后 buffer=0 也会走 Actor |

### 8.3 切 Policy 前必须满足（避免报错 / 踩坑）

| 条件 | 原因 |
|------|------|
| **Reference 模式已稳定跑通** | 与 policy 共用 infer/transition 链路 |
| **只改 GPU** `configs/plug_insertion_gpu.yaml` | `inference.mode` 仅在 `rl_server` 启动时读入 |
| **改完后必须重启 `rl_server`** | 不重启仍会是 `reference` |
| **GPU 有匹配 checkpoint** | `rl_token.pt` + `rl_actor.pt`；缺 actor 则随机初始化（不 crash，动作不可控） |
| **勿加载旧随机 actor** | 非 SFT 时代 `rl_actor.pt` 应已挪为 `presft_bak` |
| **双相机 JPEG 必传** | 缺图 → `_encode_state` ValueError |
| **`ping` 确认** | 应见 `inference_mode: policy`；首步 `policy_mode=policy` |

GPU 修改：

```yaml
inference:
  mode: policy
```

工控机 actor **不用改**。

### 8.4 算法问题（按对 policy 的影响排序）

#### 🔴 1. Policy 无训练质量门控

| 情况 | 报错？ | 后果 |
|------|--------|------|
| 有训练过的 `rl_actor.pt` | 否 | 动作应接近 VLA（BC 约束） |
| 无 ckpt / 随机 Actor | 否 | 臂可能乱动，replay 污染 |
| ckpt 维度不匹配 | **server 启动失败** | `load_state_dict` 报错 |

GPU 文档曾记录 **buffer≈578、train_steps≈395** 时可尝试 policy；工控机 `logs/online_rl/` 步数 ≠ GPU buffer，**以 GPU `ping` 为准**。

#### 🔴 2. TD3 折扣 `γ^chunk_length` 与单步 transition 不匹配

- 工控机每步存 **7 维单步** transition。
- Critic bootstrap 用 **γ^10**（`discount ** chunk_length`）。
- 稀疏 reward 仅在按 `s`/`f` 那步非零。

**影响**：Q 目标有偏差，Actor 学得慢/不稳；**不阻止 policy 运行**。

#### 🟠 3. Replay 里 action「单步 tile 成 chunk」

- Policy infer：Actor 输出完整 `(10, 7)` chunk。
- 写 buffer：每步 7 维经 `_as_action_chunk` **复制 10 遍** 进 Critic。

**影响**：与论文 chunk-level MDP 不完全一致，收敛变慢；**不阻止 policy 运行**。

#### 🟠 4. 重启 server 清空内存 buffer

- `ReplayBuffer` 仅内存，不落盘。
- 重启切 policy：`rl_actor.pt` 从磁盘加载 ✓；buffer 归零。
- 需重新攒满 **500**（`warmup_steps`）才再跑 TD3；**policy infer 不依赖 buffer**。

#### 🟠 5. buffer ≥ warmup 后每步训练很重

配置：`warmup_steps=500`，`batch_size=256`，`update_to_data_ratio=5`。  
buffer ≥ 500 后每步 transition 最多 5× `train_step` + 1× VLA 编码 → 控制频率骤降，易 **timeout**（非 Python 报错）。

#### 🟢 6. VLA 观测格式（当前配置 OK）

GPU yaml 已用 `input_format: libero`，8 维 EE proprio 与 SFT 一致。Reference 已有臂动 + 1/10 成功，state/reference 链路基本可用。

#### 🟢 7. BC 约束合理

`policy_constraint_beta=1.0` 把 policy 锚在 VLA reference；policy infer 用 `actor.forward`（均值）而非 `sample`，较稳。充分训练后 `policy_norm` 应与 `ref_norm` 同量级。

### 8.5 仍可能导致报错（与模式无关）

| 风险 | 表现 |
|------|------|
| `send_transition` 不检查 server error | `KeyError: next_state` |
| GPU infer/transition 超时 | `TimeoutError` |
| 缺双相机图 | server 返回 `error` |
| `train_step` CUDA OOM | transition 失败（buffer≥500 后更常见） |

### 8.6 建议切换顺序

```text
reference（已做）
  → reference_noise（可选，验证 action_chunk 可执行）
  → policy（确认 train_steps > 0、有 rl_actor.pt）
  → smoke 1 episode（--episodes 1 --max-steps 80）
```

**GPU 切 policy 前：**

```bash
# ping：buffer_size, train_steps, inference_mode
ls checkpoints/rl_actor.pt checkpoints/rl_token.pt
# 改 plug_insertion_gpu.yaml → mode: policy → 重启 rl_server
# 再 ping，必须 inference_mode: policy
```

**工控机 smoke：**

```bash
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 1 --max-steps 80
```

首步期望：

```text
GPU infer meta {'policy_mode': 'policy', 'policy_norm': ..., 'ref_norm': ...}
first action (policy) pos=[...]
```

`policy_norm` 与 `ref_norm` 同量级（reference 时 ref_norm 约 0.05～0.15）→ 尺度正常。

### 8.7 常见问题速查

| 问题 | 答案 |
|------|------|
| 算法有致命 bug 导致 policy 必挂？ | **无**；切换逻辑完整，shape 对齐 |
| 算法影响 policy 质量？ | **有**：γ^10、单步 tile、稀疏 reward、无训练门控 |
| 现在能切 policy 且不报错？ | **可试**；需 GPU reference 已通、有训练过的 `rl_actor.pt`、`train_steps > 0` |
| 文档 P2 写 buffer≥750？ | **代码只认 warmup=500**；750 为经验值，非硬门槛 |
| policy 成功率低算不算 bug？ | 算法上预期内；实现近似 + 稀疏 reward 会拖质量 |

### 8.8 与 §2 P2 的对应关系

- §2 写「buffer ≥ 750 后切 policy」：代码侧训练启动门槛是 **`warmup_steps: 500`** + **`batch_size: 256`**。
- 更稳妥的 Stage-4 条件：**GPU `ping` 显示 `train_steps > 0`**、reference rollout 臂动正常、**确认非 `presft_bak` 的 `rl_actor.pt`**，再改 `inference.mode: policy` 并重启 server。
