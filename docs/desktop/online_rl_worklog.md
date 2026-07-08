# Plug Insertion Online RL — 工作文档

> 整理日期：2026-07-05  
> 任务：Franka FR3 + Franka Hand + 2× RealSense，复现 RLT 插插头 online RL  
> 相关：`configs/plug_insertion.yaml`、`ONLINE_RL_TASKS.md`、`docs/ONLINE_RL_ROBOT.md`、`docs/GPU_SERVER_START.md`

---

## 1. 系统架构（当前认知）

```text
采集（工控机）
  SpaceMouse 遥操 → 对准插座 → 按 r（critical 起点）→ 录 NPZ + JSON
       ↓ export / rsync
GPU 离线
  VLA 微调 → extract_vla_embeddings → train_rl_token → checkpoints/rl_token.pt
       ↓
Online RL
  工控机 demo_reset(ep_00022 xyz) → actor_loop（相机 + proprio）
       → SSH 隧道 127.0.0.1:8765 → GPU rl_server
       → extract_embeddings + reference_action + RL 更新
       → 回 action → 真机执行 → 人工 s/f
```

| 机器 | IP / 角色 | 主要进程 |
|------|-----------|----------|
| 工控机 `vla` | `10.162.132.11` | deoxys、采集、`actor_loop`、SSH 隧道 |
| 跳板 | `10.176.53.120:26570` | SSH 入口（`yangjiarui` + `~/.ssh/id_rsa`） |
| GPU `fvl08` | `192.168.110.18:8765` | `rl_server`（VLA + RL Token + TD3） |

**通信两层：**

1. **SSH 隧道**（推荐）：`scripts/gpu/start_ssh_tunnel.sh` → 本机 `127.0.0.1:8765` → fvl08 `8765`
2. **WebSocket JSON**：`src/rlt/rl/gpu_client.py` ↔ `rlt_project/rlt_reproduce/src/rlt/scripts/rl_server.py`  
   消息：`ping` / `infer` / `transition`；图像 JPEG base64（`ws_protocol.py`）

工控机 **不跑 VLA**；每步只传 **proprio + 双相机 JPEG**，embedding 在 **GPU** 上算。

---

## 2. 今日讨论问题汇总

### 2.1 Reset / 采集 / proprio

- `proprio`：8 维 = xyz(3) + 四元数(4) + 夹爪宽度(1)，Franka base 系，**不含图像**
- 采集 NPZ：**GPU 离线**（VLA 微调、抽 embedding、训 RL Token）；**本机 demo reset** 用 `proprio[0]`
- `ep_00022`：`proprio[0]` xyz ≈ `(0.677, -0.016, 0.189)` m，距 51 条 success demo 均值约 **0.4 cm**，适合作 **pin 固定 reset**
- Reset **可随机** demo 池，当前 yaml **`demo_reset_pin_episode: ep_00022`** → **固定**
- Reset **只对齐 xyz**，**不强制姿态**（`demo_reset_trim_orientation: false`）；结束时日志会打 `rot=...deg (not enforced)`
- **容差**：见 **§9.0**（`rlt_project/.../plug_insertion.yaml` 仍为 6 cm；根目录 `configs/` 已改为 3～4 cm，需确认 actor 实际读哪份）
- 代码路径：`rlt_project/rlt_reproduce/src/rlt/hardware/deoxys/demo_reset.py`

### 2.2 相机 / 插座 / 环境一致性

- Online **必须双相机**（`wrist` + `external`），与 openpi 键名映射一致
- **插座、相机支架应固定**；动支架会改变 VLA 输入分布
- RealSense **当前只用 RGB**（`enable_depth=False`），**未用深度**做区域判断

### 2.3 Online RL / reward / 训练

- **稀疏终端 reward**：大部分 step `reward=0`；插好按 **`s`**（1），失败按 **`f`**（0）；超时 `max_steps_per_episode`（默认 600）自动 fail
- **反向传播只在 GPU**：VLA 冻结；warmup 后更新 actor/critic；`infer` 用 `torch.no_grad()`
- `inference.mode`：`reference`（VLA ref）→ 攒 buffer → 改 **`policy`** 并重启 `rl_server`
- Step 计数：episode 结束日志 / `logs/online_rl/transitions/*.jsonl` / `logs/online_rl/rewards/*.json`

### 2.4 与 RLT 论文差异

| 维度 | 论文 | 我们 |
|------|------|------|
| 粗阶段 | VLA 从较远接近 | **demo_reset 直接到 critical 起点**（无粗 VLA rollout） |
| Critical 起点 | 演示初始 / 区域切换 | **`proprio[0]`，pin ep_00022** |
| VLA→RL 切换 | **自动**（区域/距离等） | **demo_reset 即进入 RL 段**；reference→policy 靠 **改 yaml** |
| Reward | 人工 sparse | 同：**s/f** |

- **粗 VLA 可不采数据、直接用 openpi 基座试**，但不保证在 FR3+本场景好用；**critical 微调数据帮不了粗阶段**
- **「判断离插座距离」** 需 base 下 **标定插座/approach 点**；不能单靠 z 区分


---

## 3. 关键结论表

| 话题 | 结论 |
|------|------|
| demo reset | pin `ep_00022`，只对 xyz；姿态 incidental；**容差建议 2～3 cm**（§9.0） |
| 采集数据用途 | GPU 离线训练；本机 reset 用 JSON/NPZ 的 `proprio[0]` |
| 工控机↔GPU | 隧道 + WebSocket；`GPU_SERVER_HOST=127.0.0.1` |
| Reward | Episode 末 s/f，非每步 |
| 粗 VLA | 可跳过（当前方案）；论文完整流程需粗阶段+自动切换 |
| Embedding | GPU 新版已 hook；本机 repo 待同步 |

---

## 4. 接下来工作（优先级）

### P0 — 跑通 Online RL 通路

```bash
# GPU（fvl08）：一次启动
bash scripts/run_rl_server.sh configs/plug_insertion.yaml
# 期望：Loaded openpi VLA；listening ws://0.0.0.0:8765

# 工控机 终端 1：隧道常驻
bash scripts/gpu/start_ssh_tunnel.sh

# 工控机 终端 2：测通路
export GPU_SERVER_HOST=127.0.0.1
bash scripts/ping_gpu_server.sh

# 工控机：机械臂 + actor（reference 试跑）
bash scripts/start_robot.sh   # 或 start_arm + start_gripper
export GPU_SERVER_HOST=127.0.0.1
CONFIRM=1 bash scripts/run_deoxys_actor.sh
# episode 中：s=成功  f=失败  q=退出
```

检查：

- [ ] 隧道 pid：`logs/gpu_tunnel_8765.pid`
- [ ] ping 返回 `type: pong`
- [ ] demo reset 日志：`pos` **< 3 cm**（见 §9.0），`rot` 是否可接受
- [ ] `logs/online_rl/transitions/`、`rewards/` 有输出

### P1 — 确认 GPU RL 栈完整

- [ ] GPU 上 `rl_token.pt` 是否用 **真实 embedding** 训练（否则重跑 `extract_vla_embeddings` → `train_rl_token`）
- [ ] NPZ 单帧测 `extract_embeddings` 非零、shape `(M, 2048)` 量级
- [ ] **sync GPU 代码 → 本机 repo**（`embedding_extractor.py`、`obs_builder.py`、`openpi_wrapper.py`）

### P2 — 实验质量

- [ ] 插座/相机治具固定；reset 后目视 external 与采集一致
- [ ] replay ≥ `warmup_steps`（yaml 256）后 GPU 开始 learner 更新
- [ ] 改 `inference.mode: policy`，重启 server，继续 s/f

### P3 — 可选（更贴论文）

- [ ] 标定 `p_approach`，粗 openpi + 自动切 RL（不必先上深度）
- [ ] reset 改为 success demo **随机**池
- [ ] 评估是否增加 **姿态对齐**（当前仅 xyz）

---

## 5. 明日最小 Checklist（单页）

| # | 位置 | 动作 |
|---|------|------|
| 1 | GPU | `run_rl_server.sh` 已跑，非 Mock VLA |
| 2 | 工控机 | `start_ssh_tunnel.sh`，`GPU_SERVER_HOST=127.0.0.1` |
| 3 | 工控机 | `ping_gpu_server.sh` 通 |
| 4 | 工控机 | `start_robot.sh`，工作空间/急停 OK |
| 5 | 工控机 | `CONFIRM=1 run_deoxys_actor.sh`，reference 试 1 episode |
| 6 | 工控机 | 看 reset `pos`/`rot`，适时 `s` 或 `f` |
| 7 | GPU | pong 里 `buffer_size` 增长；warmup 后 `training: true` |

---

## 6. 关键文件索引

| 用途 | 路径 |
|------|------|
| 任务配置 | `configs/plug_insertion.yaml` |
| demo reset | `rlt_project/rlt_reproduce/src/rlt/hardware/deoxys/demo_reset.py` |
| 工控机 actor | `src/rlt/scripts/actor_loop.py` |
| GPU client | `src/rlt/rl/gpu_client.py` |
| GPU server | `rlt_project/rlt_reproduce/src/rlt/scripts/rl_server.py` |
| SSH 隧道 | `scripts/gpu/start_ssh_tunnel.sh` |
| actor 入口 | `scripts/run_deoxys_actor.sh` → `scripts/run_actor_loop.sh` |
| 联调说明 | `ONLINE_RL_TASKS.md` |
| deoxys 本机流程 | `docs/desktop/deoxys.txt` |

---

## 7. 今日代码改动说明

**2026-07-05 对话整理日：未对仓库提交新的代码 patch。**

对话中完成的工作主要为：阅读配置与源码、统计 demo 位姿、`ep_00022` 分析、梳理通信与论文差异、对照 GPU 与本机 `embedding_extractor` 版本。

更早同项目会话中已完成（非本日）：Franka Hand 迁移、`SpaceMouse` 右键 reset 修复、采集/online 脚本重组等（见 git 历史）。

---

## 8. 待确认项（需与 GPU 管理员核对）

- [ ] `vla_rsa` / `shimingqi` 登记在哪台机、哪个端口（Windows SSH 与工控机隧道是 **两套账号**）
- [ ] GPU 部署目录与本机 `smq&jgy` 的 **sync 策略**（单向 rsync 还是 git）
- [ ] `rl_token.pt`、`rl_actor.pt` 当前版本是否对应 **真实 embedding** 流水线

---

## 9. 故障排查：Reset 卡住 & Online 一卡一卡

> 两类现象 **原因不同**。写法：**每个现象下面紧跟「怎么办」**。

### 9.0 demo reset 容差是否太大？

**结论：对插插头 critical phase，6 cm 偏大；建议收到 2～3 cm。**

| 参数 | `rlt_project/.../yaml` | 根目录 `configs/plug_insertion.yaml` | 含义 |
|------|------------------------|--------------------------------------|------|
| `demo_reset_pos_tol_m` | **0.06** (6 cm) | **0.04** (4 cm) | reset **结束**时允许的位置误差 |
| `demo_reset_skip_if_within_m` | **0.06** | **0.03** (3 cm) | 起点已接近则 **跳过整个 reset motion** |

参考：101 条 success demo 起点 xyz 聚类 **std ≈ 0.5～1.5 cm**；`ep_00022` 在簇中心 **~0.4 cm**。

| 容差 | 问题 |
|------|------|
| **skip 6 cm** | 上一 episode 停在孔旁 5 cm → 下一 episode **不 reset**，像「卡住」、起点漂移 |
| **pos_tol 6 cm** | reset 结束可离 demo 还有 6 cm，**远比人手按 `r` 的分布宽**，VLA 起点偏 |
| **只平移不控姿态** | xyz 进了 3 cm，**rot 仍可能 >5°**，视觉与采集不一致 |

**建议值：**

```yaml
demo_reset_pos_tol_m: 0.03
demo_reset_position_tol_m: 0.03
demo_reset_skip_if_within_m: 0.02   # 或 0 强制每轮都 reset
```

**怎么办：**

- [ ] 确认 actor 实际读的 yaml（`RLT_COLLECT_CONFIG` / `run_actor_loop.sh`）。  
- [ ] 若仍用 `rlt_project/...` 的 **0.06**，改为 **0.03**（或与根目录 `configs/` 对齐）。  
- [ ] 改后看 reset 日志 `pos=...cm`：timeout 变多 → 略放宽到 **0.04**，不要回到 6 cm。  
- [ ] `rot` 经常很大 → 容差再小也解决不了，见 **现象 B**。

---

### 9.1 Reset：现象 → 怎么办

#### 现象 A：日志 `skip motion — already within X.Xcm`，臂几乎不动

**原因：** 已在 `demo_reset_skip_if_within_m` 内，跳过 motion（§9.0）。

**怎么办：**

- [ ] 将 `skip_if_within` 改为 **0.02** 或 **0**（强制每轮 reset）。  
- [ ] 上一 episode 停在孔口 → 可 **`RESET_MODE=home`** 再 demo，或 episode 结束后人工挪远。  
- [ ] 不要误当成 GPU/actor 死机。

---

#### 现象 B：`batch stalled` / `timed out ... err=...cm`

**原因：** OSC 分段 xy→z 走不到 demo xyz（碰撞、奇异、姿态不对、线缆拉扯）。

**怎么办：**

- [ ] 读完整 `[demo_reset]`，记下 **err(cm)** 和哪一段 `posN/M`。  
- [ ] 人离开工作空间；查 external 是否顶桌/插座/线绷紧。  
- [ ] 试 **`RESET_MODE=home`** 再 demo，或换更近的 success demo。  
- [ ] 反复 timeout → 暂 **`pos_tol: 0.04`**，并记录 **`rot=...deg`**。  
- [ ] `rot` 持续 >10° → 长期需 **姿态对齐**（当前未实现）。

---

#### 现象 C：在动但很慢，段间顿一下（20～60 s）

**原因：** 多段 waypoint + 每段最多 350 step；**不是死机**。

**怎么办：**

- [ ] 第一次 online **预留 30～60 s** 等 reset。  
- [ ] 日志 segment 逐个 `ok` → **正常**，继续 actor。  
- [ ] 熟悉流程后再考虑 yaml 里更快的 `demo_fast` 选项。

---

#### 现象 D：采集/遥操 SpaceMouse 右键 home 后不能再动

**原因：** 未 `acknowledge_spacemouse_reset`（**采集/teleop**，非 actor 主路径）。

**怎么办：**

- [ ] 用已修复的 `teleop_test.sh` / `collect_data.sh`。  
- [ ] Online actor **通常可排除此条**。

---

#### Reset 日志速查

| 日志 | 立刻做什么 |
|------|------------|
| `skip motion — already within` | 调小 skip 或 home 后再 demo |
| `batch stalled — incremental` | 查碰撞，读 err(cm) |
| `timed out ... err=` | home / 换 demo / 略放 tol |
| `reached ep_00022 pos=... rot=` | 看 pos<3 cm、rot 是否可接受 |

---

### 9.2 Online：现象 → 怎么办

#### 现象 E：动一下、停很久（一卡一卡）

**原因：** 每 chunk 一次 `infer`（约 2× VLA），**每步** `send_transition`（再 1× VLA encode）+ WebSocket + JPEG；**实际频率 ≪ 20 Hz**。

```text
每 10 步：gpu.infer → 10×(step + send_transition + sleep 50ms) → 再 infer
```

**怎么办（短期）：**

- [ ] **预期管理**：reference online **不会像 20 Hz 遥操顺滑**。  
- [ ] GPU `nvidia-smi`；隧道 `GPU_SERVER_HOST=127.0.0.1`。  
- [ ] `ping_gpu_server.sh`；transition 常 **>500 ms** 先查 GPU/网络。

**怎么办（中期，改代码）：**

- [ ] `rl_server.handle_infer`：**合并** encode 与 reference 为 **一次** VLA。  
- [ ] 避免每步 transition 都 `_encode_state(next)`。  
- [ ] 代码：`actor_loop.py`、`rl_server.py`。

---

#### 现象 F：每步位移很小，像「蹭」

**原因：** `reference` 在孔附近 **输出小增量** + 现象 E 的 GPU 等待。

**怎么办：**

- [ ] 固定插座/相机；reset 后构图接近采集时。  
- [ ] 看 infer meta 的 `ref_norm`；reset **`rot`** 大则先改善 §9.0/9.1。  
- [ ] buffer 够后试 **`inference.mode: policy`**。

---

#### 现象 G：约每 10 步多顿一下

**原因：** `chunk_length` / `execute_prefix` = 10，边界重新 `infer`。

**怎么办：**

- [ ] 属设计行为；减小 `execute_prefix` 会更频繁 infer、**往往更慢**。  
- [ ] 根本缓解靠 **现象 E 中期方案**。

---

#### 现象 H：warmup 后比刚开始更卡

**原因：** `handle_transition` 内 **`train_step`（TD3）** 叠加。

**怎么办：**

- [ ] 可暂减 yaml 的 `update_to_data_ratio` / `critic_updates_per_actor`。  
- [ ] 长期仍靠减少每步 VLA 次数。

---

### 9.3 快速对照（查表）

| 现象 | 阶段 | 首要动作 |
|------|------|----------|
| A skip 不动 | Reset | 缩小 skip / home 后再 demo |
| B timeout | Reset | 碰撞、home、§9.0 容差 |
| C 慢但在动 | Reset | 等 30～60 s |
| D 遥操死 | 采集 | `acknowledge_spacemouse_reset` |
| E 一卡一卡 | Online | GPU/隧道；中期改 server |
| F 微动 | Online | 起点/相机；试 policy |
| G 每 10 步顿 | Online | chunk 边界，可接受 |
| H warmup 后更卡 | Online | 减 learner 更新频率 |

**一句话：** Reset → **§9.0 容差** + 日志；Online → **等 GPU 是主因**。
