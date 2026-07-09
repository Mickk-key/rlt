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
