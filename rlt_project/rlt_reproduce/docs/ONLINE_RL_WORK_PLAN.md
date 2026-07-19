# Online RL 工作计划 — 插充电器 Critical Phase

> **⚠️ SUPERSEDED (2026-07-19).** The authoritative RLT online-RL runbook is now
> [`ONLINE_RL_RUNBOOK.md`](ONLINE_RL_RUNBOOK.md). Follow that file for the full
> workflow; this plan is kept for history/background.

> **任务**：Franka 插充电器，双机 split online RL（RLT Algorithm 1）  
> **最后更新**：2026-07-10  
> **工控机**：`10.162.132.11`，工作目录 `smq&jgy`  
> **GPU**：`10.176.53.120`（fvl08 / 内网 `192.168.110.18`）  
> **相关文档**：`ONLINE_RL_RUNBOOK.md`（启动命令）、`ONLINE_RL_ROBOT.md`（工控机 checklist）

---

## 更新 — 2026-07-16 RLT 对齐修复（Phase 1–4 + 安全）

> **权威说明见** [`docs/desktop/online_rl_worklog.md` §0](../../../docs/desktop/online_rl_worklog.md)。下文 §5「代码与配置风险审查」多为 7/5 旧状态，其中若干已在 7/16 修复：
>
> - **运行时源码树映射**：`actor_loop.py` / `gpu_client.py` 运行时用 **Tree C**（`smq&jgy/src`，`_env.sh` 置于 PYTHONPATH 最前）；`rl_server.py`、`learner.py`、`replay_buffer.py`、`inference_policy.py`、`deoxys_env.py` 用 **Tree B**（`rlt_reproduce/src`）。两树同属一个 git 仓库；`actor_loop`/`gpu_client` 两份功能代码保持同步。
> - **Phase 1 安全 clamp**（`deoxys_env.py`）：物理 EE 增量硬限幅（平移 ≤0.02m、旋转 ≤0.1rad、夹爪 clip、NaN→保持位姿），reference/policy 均生效。
> - **Phase 2 policy 锚定**（`inference_policy.py`）：`action = reference + clip(actor-reference, ±delta)`（0.01m/0.05rad），actor 架构不变、非 residual。
> - **Phase 3 warmup/ramp 门控**（`act_gated`）：按 buffer transition 数决定 reference→ramp→policy；`warmup_steps=500`、`ramp_steps=500`。
> - **Phase 4 真实 chunk transition**：`(x_s, a_{s:s+C}, ref_{s:s+C}, ref_{s+C:s+2C}, R=Σγ^k r, x_{s+C}, done)`，无 tile；`γ^C` 与 C 步间隔一致；终局 padding：动作/reference 补最后有效步、reward 补 0。**这修复了下文 §5.3(5) 的「单步 tile 成 chunk」。**
> - **checkpoint**：保留 `rl_token.pt` + SFT VLA（`pi05_base` / `pi05_plug_insertion`）；丢弃 `rl_actor.pt` / `rl_critic.pt` / `online_rl/`（删除前已备份到 `checkpoints/backups/phase5_pre_reset_*`）。
> - **安全重启**：首跑强制 `inference.mode: reference`，待 ≥500 有效 transition + 形状/reward/done 校验 + 2–3 成功 reference episode 后改回 `auto` 并重启 server。

---

## 速查 — 核心概念、算法问题与对策（2026-07-10）

> 本节为组内 FAQ：汇总 **Online RL / Reference / Policy** 概念，以及 **算法问题分析、优化方向、解决方法**。下文原有章节（进度、SFT 切换、启动流程等）保持不变。

### A. 核心概念（Online RL / Reference / Policy）

#### A.1 四个阶段别混

| 层级 | 含义 | 当前（7/10） |
|------|------|--------------|
| 双机通路 | WebSocket infer / transition 通 | ✅ |
| **Reference rollout** | 真臂 + SFT VLA；臂执行 **reference** | 🟡 ~20+ ep，success 低 |
| **Stage-3 训练** | buffer ≥ warmup 后 TD3 更新 | 🟡 buffer 578，train 395 |
| **Policy 控臂（Stage-4）** | `mode: policy`，**RL Actor** 出动作 | ⬜ 未开始 |

- **Online RL「训练开始」** = replay 满 warmup + GPU 出现 `learner metrics`（**不必等 policy**）。
- **Policy 阶段** = 改 GPU yaml `inference.mode: policy` 并重启 server。

#### A.2 Rollout / Reference / Policy

| 词 | 一句话 |
|----|--------|
| **Rollout** | 真机跑完一整局 episode（reset → 多步 → s/f/timeout） |
| **Reference** | SFT VLA 输出的动作；`inference.mode: reference` 时 **臂听 VLA** |
| **Policy 控臂** | RL Actor 输出动作；VLA 仍提供 reference 作条件，但 **臂听 Actor** |
| **Buffer** | GPU 上存 transition 的经验池（`buffer_size` = 条数，不是 episode 数） |
| **TD3** | 从 buffer 抽样更新 Actor/Critic 的算法（`RLTLearner`） |
| **Transition** | 一步经验，见 A.3 |

#### A.3 Transition 记什么？

| 字段 | 含义 |
|------|------|
| `state` | 当前 `[z_rl, proprio]`（GPU 算） |
| `action` | **本步实际执行**的 7 维动作 |
| `reference_action` | 本步 VLA reference |
| `reward` | 通常仅按 **s/f** 那步为 1/0，中间步为 0 |
| `next_state` | **下一步状态向量**（非「下一步动作」） |
| `done` | 本步后 episode 是否结束 |

**「动作为下一步 infer 时另算」**：chunk 内逐步执行；chunk 用完或需新动作时，**下一次 `gpu.infer()`** 用 **新相机图 + proprio** 再算，不在 `next_state` 里。

Reference 模式下 action 来自 **VLA infer**，不是工控机本地 `f(state)`。

#### A.4 「收敛」分别指什么？

| 对象 | Reference 阶段 | Policy 阶段 |
|------|----------------|-------------|
| SFT VLA 权重 | ❌ 冻结，不收敛 | ❌ 不变 |
| Reset/接近进工作区（Jerry：~100%） | ✅ 应稳定 | 仍应稳定 |
| 终局插入 success（按 s） | ❌ 不要求收敛到 100% | ✅ **行为应变好并稳定** |
| TD3 / buffer | ✅ 在积累、可训练 | ✅ 继续训 |
| **~100 episode** | 整条 online 流程 **数据量**目标（含 reference） | 切 policy 后 often **再 50–200 ep** |

- Jerry **「100 个左右才能收敛」**：主要指 online 流程跑够 **~100 局 rollout / 足够 transition**，不是终局 success 100%。
- Jerry **「接近矩形 ~100%」**：reset 后进入插头正上方工作区（config `xy_half_range_m: 0.05`），**不是**整段插入成功 100%。
- **Reference 阶段没有「VLA reference 越跑越准」**；Policy 阶段才有 **插入 success 行为收敛** 可言。

#### A.5 卡顿：平滑 vs 超时（可同时存在）

| 原因 | 现象 | 谁提的 |
|------|------|--------|
| **Action chunk 无平滑** | 块边界 jerk，「一顿一顿」 | Jerry：预测单元是动作块，需平滑 |
| **GPU infer 慢** | 动 ~0.5s、停 1–2 min | 文档 / 实测 |
| **WebSocket timeout** | 断连、部分 transition 未写入 replay | Mickeyy |

平滑（chunk 首尾插值、指令低通）**减轻块内/块边界 jerk**，**不能消除 infer 等待**。超时是 **网络/延迟** 问题，与平滑是不同层。

#### A.6 Policy 阶段要跑多久？

无固定公式。经验：**切 policy 后再 50–200 episode** + **train_steps 再多几百～几千**；以 **success 率高于 reference 且稳定** 为准。infer 慢则 **墙钟远长于 episode 数**（同样 100 ep 可能需数天）。

---

### B. 算法与流程 — 问题分析

> 框架（WebSocket、RL Token、TD3、replay）整体可用；以下为 **影响真机 / 训练质量** 的主要问题（7/10 复核）。

#### B.1 总体判断

| 类别 | 阻止开跑？ | 让训练学不到东西？ |
|------|------------|-------------------|
| GPU OOM / server 未 listen | ✅ | — |
| infer 过慢 + WebSocket timeout | 可能断连 | ✅ 丢 transition |
| VLA 每步/每 transition 重复全量 infer | — | ✅ 卡顿、有效 step 少 |
| Action chunk 无时间平滑 | — | ⚠️ 观感差、控制不连贯 |
| 终局 success 极低（如 1/20） | — | ✅ replay 负样本多 |
| Replay 重启不持久化 | — | ⚠️ 需重新 warmup |
| 过早 `mode: policy` | ✅ 真机风险 | ✅ replay 污染 |

#### B.2 已修复 / 已改善（相对 7/5）

| 项 | 状态 |
|----|------|
| `actor_loop` 首步 `action_chunk` 顺序 | ✅ 已修 |
| GPU yaml `inference.mode: reference` | ✅ |
| SFT VLA + Libero + sft5000 RL Token 对齐 | ✅ |
| init-cube external reset（与 SFT 一致） | ✅ |
| `action_is_physical: true` | ✅ |
| `rl_actor.pt` / `rl_critic.pt` 保存 | ✅ |
| GPU/工控机 yaml 分离 | ✅ |

#### B.3 仍须关注的问题

**（1）VLA 调用次数过多（卡顿主因之一）**

- 每次 `infer`：`extract_embeddings` + `reference_action`（`embedding_extractor.py`）。
- 每步 `transition` 算 `next_state` 时再次 `infer_from_proprio`（`rl_server._encode_state`）。
- 粗算：1 次 chunk infer + 10 步 transition ≈ **11 次** VLA 相关前向/chunk → 墙钟极长。

**（2）Chunk 边界无平滑**

- VLA 一次输出 `chunk_length=10` 步；块间无插值/滤波 → Jerry 所述「一顿一顿」。
- 与 infer 等待叠加 → 「动一下停很久」。

**（3）稀疏 reward + 低 success**

- 仅 s/f 一步有 1/0；1/20 终局 success → TD3 成功信号极少。

**（4）Replay 不持久化**

- 重启 `rl_server` → buffer 清空；权重可 reload，transition 需重攒。

**（5）工控机 `max_steps_per_episode: 200`**

- @20Hz 约 10s sleep 上限；易 timeout fail（可与 `--max-steps 2400` 临时加大）。

---

### C. 优化方向（按优先级）

| 优先级 | 方向 | 预期效果 | 实现状态 |
|--------|------|----------|----------|
| **P0** | 续跑 checklist 打满（§D） | 少报错、少断连 | 流程 |
| **P1** | 降低 transition 算 `next_state` 的 VLA 开销（仅 embed / 复用 infer 缓存） | **大幅减卡顿** | 待开发 |
| **P1** | infer 异步：臂执行 chunk 时 GPU 算下一 chunk | 减停顿感 | 待开发 |
| **P1** | chunk 间 action 平滑（插值/EMA） | 减 jerk（Jerry 建议） | 待开发 |
| **P2** | reference 多跑 episode（目标 ~100），`--max-steps` 加大 | 更多有效 transition | 操作 |
| **P2** | 区分指标：reset/接近成功率 vs 终局插入 success | 与 Jerry 对齐评估 | 操作 |
| **P2** | 加快 infer（GPU 独占、预热、减少重复 forward） | 减 timeout、增 step/局 | 运维+代码 |
| **P3** | replay buffer 持久化 | 重启不丢数据 | 待开发 |
| **P4** | reference 稳定 + buffer 够后再切 `policy` | 安全进入 Stage-4 | 操作 |

---

### D. 解决方法与续跑 checklist

#### D.1 开跑前三端

| 终端 | 机器 | 动作 |
|------|------|------|
| A | GPU | `CUDA_VISIBLE_DEVICES=1 bash scripts/start_gpu_rl_server.sh`（读 **`plug_insertion_gpu.yaml`**） |
| B | 工控机 | `bash scripts/gpu/start_ssh_tunnel.sh` 常开 |
| C | 工控机 | `export GPU_SERVER_HOST=127.0.0.1` + `CONFIRM=1 bash scripts/run_deoxys_actor.sh ...`（读 **`plug_insertion.yaml`**） |

#### D.2 续跑前必查

- [ ] GPU：SFT ckpt、`rl_token.pt`（sft5000 软链）、`rl_server listening`
- [ ] 工控机：`configs/sft_plug_insertion.yaml` 存在（external reset）
- [ ] Deoxys 臂/夹爪 + 双 RealSense；**无** `--no-cameras`
- [ ] actor 终端 **TTY**（能按 s/f）
- [ ] 两边 **git 同一 commit**（如 `gpu-sync-20260710`）
- [ ] **勿切 policy**，直到 reference/接近稳定且 `train_steps` 持续增长

#### D.3 推荐运行参数

```bash
# 工控机 — 单局试跑
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 1 --max-steps 1200

# 工控机 — 攒数据（学长 ~100 ep 量级）
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 20 --max-steps 2400
# 或: export MAX_STEPS=2400
```

#### D.4 阶段目标（与学长对齐）

```
1. Reset/接近（进 init 立方体/矩形区）     → 目标 ~100% 稳定
2. Reference rollout + 攒 replay          → ~50–100 ep，buffer 持续增长
3. TD3 后台训练（Stage-3）                → train_steps 涨，loss 大致稳定
4. 切 policy（Stage-4）                   → 短试 5–10 ep → 再 50–200 ep
5. Policy 收敛                            → success 率高于 reference 且稳定
```

#### D.5 常见问题对策

| 现象 | 原因 | 处理 |
|------|------|------|
| 一顿一顿 | chunk 无平滑 + infer 慢 | 预期部分现象；优先减 infer 重复；后续加平滑 |
| WebSocket timeout | infer > 客户端等待 | `infer_timeout_sec: 180`；减 VLA 调用次数 |
| buffer 重启变 0 | 无持久化 | 接受重攒 warmup，或后续做 buffer save |
| success 1/20 | 任务难 + 稀疏 reward | 正常可继续攒；policy 阶段再判是否变好 |
| 臂乱动 | 误切 `mode: policy` | 保持 `reference` |

---

## 最新进度（2026-07-10）

**今日收工状态**（GPU server 已正常退出；`rl_server` shutdown 时已保存 ckpt）：

| 项 | 数值 / 状态 |
|----|-------------|
| Stage-2c 真实 VLA + 双路相机 infer | ✅ 通过 |
| Stage-2d reference 真机 rollout | 🟡 **部分通过**（~20+ episode；success 少；动作因 infer 延迟呈「动一下停很久」） |
| Stage-3 replay warmup + TD3 | 🟡 **已启动**（收工时 `buffer_size=578`，`train_steps=395`，`training=true`） |
| Stage-4 policy 控臂 | ⬜ 未开始（仍为 `inference.mode: reference`） |
| 在线 ckpt | ✅ `checkpoints/rl_actor.pt`、`rl_critic.pt`（2026-07-10 06:29）；`checkpoints/online_rl/rl_*_step000150~350_*.pt` |

**部署树已就绪（勿与工控机 yaml 混用）**：

| 用途 | 文件 | 状态 |
|------|------|------|
| GPU `rl_server` | `configs/plug_insertion_gpu.yaml` | ✅ 已添加；`start_gpu_rl_server.sh` **默认读此文件** |
| 工控机 `actor_loop` | `configs/plug_insertion.yaml` | ✅ 工控机专用（reset / 相机 / gpu_server） |
| SFT VLA | `.../pi05_plug_insertion/.../5000` + Libero `input_format` | ✅ 已在 **gpu yaml** 配置 |
| RL Token | `checkpoints/rl_token.pt` → sft5000 软链 | ✅ 未改离线权重 |
| 旧 actor 备份 | `checkpoints/rl_actor.pt.presft_bak` | ✅ 保留 |

**运行中观察（预期现象，非 bug）**：

- 每次 GPU infer 约 **1–2 分钟** → 臂 **execute_prefix=10 @ 20Hz** 仅动 ~0.5s 再停 → 观感卡顿。
- GPU 终端偶发 `keepalive ping timeout` / `connection handler failed`；只要 ping 里 `buffer_size` 仍上涨可继续跑。
- reference 阶段 **success 比例低** 正常；HIL 以 s/f 标 reward，fail/timeout 也会进 replay。

**下次开机续跑**：

1. GPU：`CUDA_VISIBLE_DEVICES=1 bash scripts/start_gpu_rl_server.sh`（会 **重载** `rl_actor.pt`/`rl_critic.pt`，但 **replay buffer 内存清空**，需重新攒或接受从 0 开始 warmup——**今日未做 buffer 持久化**）。
2. 工控机：隧道 + `CONFIRM=1 bash scripts/run_deoxys_actor.sh ...`，config 用 `plug_insertion.yaml`。
3. **勿切 policy**，直到 reference 动作与 success 率满意且 `train_steps` 继续增长。

---

## 0. 2026-07-08 更新 — 切换到 SFT VLA（务必先读）

**背景**：pi0.5 已在 plug_insertion 上做完 **SFT**；新的 `rl_token.pt` 是在 **SFT 版 VLA 的 embedding** 上训练的（`checkpoints/rl_token_sft4000_112ep.pt`，run `rl_token_run_20260707T225233Z`，best val L_ro≈0.247）。之前不带 SFT 的 RL Token 效果差，已弃用。上真机跑 online RL 前，**先用 reference 模式验证 SFT**（学长建议）。

**为此对 GPU 部署树（**`smq_jgy_deploy/smq&jgy/rlt_project/rlt_reproduce/`**，即** `rl_server` **实际读取的那份）做了以下变更**：


| 变更                  | 文件                                                                  | 内容                                                                                                                                                                                                      |
| ------------------- | ------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| VLA 换成 SFT          | **`configs/plug_insertion_gpu.yaml`**（GPU）                          | `vla.checkpoint → checkpoints/pi05_plug_insertion/plug_insertion_112ep_success_8gpu/5000`；`config_name → pi05_plug_insertion`；`asset_id → local/plug_insertion_112ep_success`；`input_format: libero` |
| 执行模式                | **`configs/plug_insertion_gpu.yaml`**                               | `inference.mode: reference`（warmup / 训练进行中仍保持 reference；Stage-4 再改 `policy`）                                                                                                                        |
| 在线 obs 支持 Libero 格式 | `src/rlt/vla/obs_builder.py`                                        | 新增 `ws_obs_to_libero`（`observation/image`+`observation/wrist_image`+`observation/state`）与 `build_observation_for_format` 分发器                                                                            |
| 传递 input_format     | `src/rlt/vla/embedding_extractor.py`、`src/rlt/scripts/rl_server.py` | 从 config 的 `vla.input_format` 选择 obs 构造器                                                                                                                                                                |
| checkpoint 软链       | `checkpoints/pi05_plug_insertion`                                   | 指向训练树同名目录，使 `.../5000` 在部署树可解析                                                                                                                                                                          |
| 清理旧 actor           | `checkpoints/rl_actor.pt`                                           | 重命名为 `rl_actor.pt.presft_bak`（旧的非-SFT 随机 actor，避免 policy 模式误加载）                                                                                                                                         |


> ⚠️ **为什么用 5000 而不是 4000**：RL Token 原本是在 **step-4000** 的 SFT ckpt 上训的，但 SFT 用 orbax `max_to_keep=1` + `keep_period=5000`，**4000 已被自动删除**，磁盘只剩 `5000/10000/10500`。选了同一次 SFT run 里最接近的 **5000**（val L_ro 在 4000≈0.2517、~5000≈0.2467，embedding 近乎一致）。

**✅ 2026-07-08（18:02）后续：已用 step-5000 SFT ckpt 重抽 embedding + 重训 RL Token，做到 VLA 与 RL Token 完全同源。**

- 训练配置：`rlt_project/rlt_reproduce/configs/plug_insertion_sft5000.yaml`（隔离产物，未动现有）；运行脚本 `scripts/run_rl_token_sft5000.sh`（4 卡 sharded precompute → GPU1 训练）。
- 新 token：`rlt_project/rlt_reproduce/checkpoints/sft5000_rltoken/rl_token.pt`（run `rl_token_run_20260708T075448Z`，5000 步完成，`best_val_L_ro≈0.2363`、`final_train_L_ro≈0.2325`，strict 加载通过、权重全 finite）。比原 sft4000 版（0.2467）略好。
- **已切换在线 token 软链**：
  - 旧目标：`…/rlt_project/rlt_reproduce/checkpoints/rl_token.pt`（sft4000 版，实体文件，保留未删）
  - 新目标：`…/rlt_project/rlt_reproduce/checkpoints/sft5000_rltoken/rl_token.pt`
  - 命令：`ln -sfn "…/checkpoints/sft5000_rltoken/rl_token.pt" "…/smq_jgy_deploy/smq&jgy/rlt_project/rlt_reproduce/checkpoints/rl_token.pt"`
  - 校验：部署 `checkpoints/rl_token.pt` → resolves 到 sft5000 token（`rl_server` 读的就是这个）。
- **未改动**：SFT ckpt 软链 `pi05_plug_insertion`；工控机 **`configs/plug_insertion.yaml`**（不含 `rl_token`/learner 段，由 gpu yaml 承担）。
- 回滚：`ln -sfn "…/checkpoints/rl_token.pt" "…/smq_jgy_deploy/.../checkpoints/rl_token.pt"` 即可切回 sft4000 版。

**一致性铁律**：在线抽 embedding 的 VLA（`vla.checkpoint` + `config_name` + `input_format`）必须与训练 `rl_token.pt` 时**完全一致**，否则 `z_rl` 失真、online RL 学不到东西。改 RL Token 训练配置时，务必同步本部署 config。

**如何验证 SFT（= Stage-2d reference rollout）**：GPU 起 `rl_server`（`mode: reference`）→ 工控机 `CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 1 --max-steps 1200`（无 MOCK、无 `--no-cameras`）→ 看首步 `policy_mode=reference`、`ref_norm` 合理、臂在 critical phase 的插入动作方向/幅度是否像模像样。这一步跑通即 SFT 验证通过，且正好是 online RL 的第一步。

---



## 1. 当前状态（2026-07-10）


| 阶段           | 内容                          | 状态        |
| ------------ | --------------------------- | --------- |
| Stage-0      | 数据采集 51 success + 50 fail   | ✅         |
| Stage-1      | RL Token 离线训练 `rl_token.pt` | ✅         |
| Stage-2a     | 双机 WebSocket（MOCK + SSH 隧道） | ✅         |
| Stage-2b     | GPU 部署 smq&jgy 代码           | ✅         |
| Stage-2c     | 真实 pi05 + 双路相机 infer        | ✅ **已通过** |
| Stage-2d     | reference 真机 rollout        | 🟡 **部分通过**（通路 OK；success 少；motion 卡顿） |
| **Stage-3**  | replay warmup + TD3 梯度更新    | 🟡 **进行中**（收工：buffer 578，train 395 步，有 ckpt） |
| **Stage-4**  | `inference.mode: policy` 控臂 | ⬜ **未开始** |


**当前阻塞 / 风险（按优先级）**

1. **Infer 延迟**：远程 VLA 单次 1–2min → 真机 motion 卡顿、WebSocket 偶发 timeout；buffer 增长慢于 episode 数。
2. **Replay 不持久化**：重启 `rl_server` 会 **清空 buffer**；今日 ckpt 仅 actor/critic 权重。
3. **网络**：工控机直连 `:8765` 可能仍 refused → 继续 **SSH 隧道** + `GPU_SERVER_HOST=127.0.0.1`。
4. **Policy 勿早切**：`train_steps≈400` 且 reference 未验稳前 **保持 reference**。

---



## 2. 为什么 Online RL「还没开始」？

把四件事分开，不要混为一谈：


| 层级                    | 含义                                   | 状态        |
| --------------------- | ------------------------------------ | --------- |
| 双机通路                  | WebSocket ping / infer 通             | ✅ 真机 JPEG infer 已通 |
| **Reference rollout** | 真臂 + 真 pi05 + 双相机跑起来                 | 🟡 已跑通；success 少、motion 卡 |
| **Replay warmup**     | `buffer_size ≥ warmup_steps` 后开始 TD3 | ✅ 今日已达（578/500） |
| **Policy 控臂**         | `mode: policy`，Actor 输出动作            | ⬜ 未开始     |


**Online RL「训练开始」** = Stage-3：replay 满 warmup 且 GPU 打印 `learner metrics` / `updated=True`。  
**Online RL「policy 阶段」** = Stage-4：`policy_mode=policy`，Actor 接管控制。

> **注意**：`configs/plug_insertion.yaml` 里若写了 `inference.mode: policy`，以 **实际 buffer 与日志** 为准；未跑通 reference rollout 前，应改回 `reference`。

---



## 3. 各阶段「怎样才算通过」



### 3.1 Reference rollout 通过（Stage-2d）

**GPU 终端应出现：**

```text
Loaded RL token from checkpoints/rl_token.pt
Loaded openpi VLA from .../checkpoints/pi05_base
Inference mode: reference
RL server listening on ws://0.0.0.0:8765
```

**工控机命令（无 MOCK、无** `--no-cameras`**）：**

```bash
export GPU_SERVER_HOST=127.0.0.1   # 走 SSH 隧道时
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 1 --max-steps 60
```

**通过标准：**

- [x] 首步 `GPU infer meta` 含 `policy_mode=reference`、`z_rl_norm`、`ref_norm`
- [x] 无 `KeyError: observation/exterior_image_1_left`
- [x] 无 infer 时 CUDA OOM
- [x] 臂 motion 大致合理（不要求一次插成功）
- [x] 按 **s/f** 可结束 episode，终端打印 `Episode ep_XXXX done ...`
- [x] 下一 episode 前 **自动 demo reset**（无需手动 SpaceMouse 回 home）

---



### 3.2 Online RL 训练开始（Stage-3）

- GPU 保持 `inference.mode: reference`（或 `reference_noise`）
- 工控机多 episode rollout，**s=成功 / f=失败**
- GPU `buffer_size` 持续增长
- 当 `buffer_size ≥ online_rl.warmup_steps`（论文 500；当前配置可能为 256）时出现：

```text
learner metrics={'critic_loss': ..., 'actor_loss': ...}
```

---



### 3.3 Policy 接管（Stage-4）

GPU 修改并 **重启 server**：

```yaml
inference:
  mode: policy
  deterministic: true
```

- [ ] infer meta 为 `policy_mode=policy`
- [ ] 臂执行 RL Actor 输出（非纯 VLA reference）
- [ ] replay 继续增长，learner 持续更新

---



## 4. Reference rollout 为什么没通过？


| 原因      | 现象                                            | 处理                                       |
| ------- | --------------------------------------------- | ---------------------------------------- |
| GPU OOM | server 起不来或 infer OOM                         | 协调 1 张空闲 24GB 卡；`CUDA_VISIBLE_DEVICES=N` |
| 无相机图    | `KeyError: observation/exterior_image_1_left` | 去掉 `--no-cameras`；开 RealSense            |
| 网络      | `:8765` refused                               | SSH 隧道 + `GPU_SERVER_HOST=127.0.0.1`     |
| 误用 MOCK | 通路通但非真实 VLA                                   | 真机 reference 必须真实 pi05 + JPEG            |


**MOCK 通路通 ≠ reference 真机通过。**

---



## 5. 代码与配置风险审查（2026-07-05）

> 双机框架（WebSocket、RL Token、TD3、replay）整体是齐的；下面列出 **会挡真机 / 拖训练质量** 的问题，按严重程度排序。



### 5.1 总体判断


| 类别                            | 会不会阻止「开始训练」 | 会不会让训练「学不到东西」     |
| ----------------------------- | ----------- | ----------------- |
| GPU OOM / server 未起           | ✅ 会         | —                 |
| `actor_loop` 首步 bug           | ✅ 可能        | —                 |
| `mode: policy` + 随机 Actor     | ✅ 真机危险      | ✅ replay 污染       |
| VLA action / proprio mismatch | 可能能跑        | ✅ **很可能**         |
| 超时太短                          | —           | ✅ 很多 timeout fail |
| batch/warmup / 无 ckpt save    | —           | ⚠️ 不稳定、难复现        |


---



### 5.2 🔴 严重：建议先修再真机训练



#### （1）`actor_loop.py` 首步可能崩溃（代码 bug）

文件：`src/rlt/scripts/actor_loop.py`

在 `step == 0` 打印 `first action` 时使用了 `action_chunk`，但该变量在 **后面** 才从 `result.action_chunk` 赋值。第一步 infer 若返回 `meta`（正常情况会返回），会触发 `NameError`。

```python
# 问题：action_chunk 在打印之后才赋值
if step == 0 and result.meta:
    console.print(... action_chunk[0] ...)  # ← 未定义
action_chunk = result.action_chunk           # ← 应挪到打印之前
```

**影响**：真机 reference rollout 可能在第一步就挂（MOCK 走 GPU server 时也会踩到）。  
**修复**：将 `action_chunk = result.action_chunk` 移到打印语句之前。

---



#### （2）配置 `inference.mode: policy` 与真实进度不符

文件：`configs/plug_insertion.yaml`

当前 yaml 注释写「warmup 已完成」，但 Stage-3/4 **实际未开始**。且无 `checkpoints/rl_actor.pt` 时，Actor 为 **随机初始化**。

**影响**：

- `policy` 模式会让臂执行 **随机策略**，不是 VLA reference  
- replay 会被无效/危险 transition 污染

**修复**：在 reference rollout + replay warmup 完成前，保持 `inference.mode: reference`；仅 Stage-4 再改 `policy` 并重启 server。  
✅ **2026-07-08 已处理**：部署 config 已改回 `mode: reference`；旧的非-SFT `rl_actor.pt` 已挪为 `rl_actor.pt.presft_bak`（见 §0）。

---



#### （3）VLA 动作空间 vs 真机 OSC 控制（domain gap，影响最大之一）


|         | 采集 / 真机执行                             | openpi `pi05_droid` 输出                         |
| ------- | ------------------------------------- | ---------------------------------------------- |
| 控制      | Deoxys **OSC 笛卡尔增量** × `action_scale` | DROID 预训练空间的 action                            |
| proprio | 8 维末端位姿 + 夹爪                          | 硬映射为 `joint_position`（语义不对，见 `obs_builder.py`） |


代码中 **没有** 将 VLA 输出再映射回采集时的 OSC 空间；reference 直接 `env.step(action)`。

**影响**：reference 可能方向/幅度完全不对，在线 RL 的 BC 锚点与 reward 学习都会变差——这比「要不要 VLA 微调」更底层。  
**验证**：先做 reference 真机短跑，观察 `ref_norm` 与臂 motion；不合理则需 VLA 微调、换 config 或加 action 变换层。

---



### 5.3 🟠 高影响：能跑但训练质量可能很差



#### （4）Episode 超时 600 步偏短

见 **§6**（原超时专节）。infer 快时约 30s sleep，人工来不及按 s/f → timeout fail 进 replay。

#### （5）Replay 里 action 是「单步 tile 成 chunk」 — ✅ 2026-07-16 已修复（Phase 4）

~~工控机每步发 **7 维单步** action；GPU `rl_server._as_action_chunk()` 将单步 **复制** 为 `(chunk_length, 7)` 再进 Critic。~~

现状：`actor_loop` 按步收集整段 episode，`build_chunk_transitions` 组装**真实** chunk `a_{s:s+C}`（无 tile，`subsample_stride=2`）；`rl_server._as_action_chunk` 严格校验 `(C,7)` 形状；`next_state = x_{s+C}` 且 `learner` 用 `γ^C`。详见 worklog §0.5。

#### （6）稀疏 reward + `batch_size=256`

- 仅按 **s/f** 那一步有 reward 1/0，中间步全为 0（HIL 设计如此）  
- `warmup_steps: 256` 且 `batch_size: 256` → 第一批梯度即吃满整个 buffer，方差大  
- 论文常用 warmup **500**

**影响**：能训，但 sample 效率差、不稳定；timeout fail 多会更糟。

#### （7）在线训练不保存 `rl_actor.pt`

`rl_server.py` 仅 **加载** `checkpoints/rl_actor.pt`，训练过程中 **无 save 逻辑**。Server 重启后 Actor 权重丢失。

**影响**：不阻止第一次训练，但无法复现/续训。

---



### 5.4 🟡 中等：运维/环境（非逻辑 bug 但会挡路）


| 问题                                         | 影响                          |
| ------------------------------------------ | --------------------------- |
| GPU OOM，8765 未 listen                      | 根本起不来 server                |
| 必须 SSH 隧道 + `GPU_SERVER_HOST=127.0.0.1`    | 隧道断则断连（**隧道通即可，不必等直连 NAT**） |
| 双代码树（`rlt_project` vs `smq_jgy_deploy`）    | 改了一边忘同步                     |
| 真实 infer 必须双相机 JPEG                        | `--no-cameras` → KeyError   |
| `MockGPUClient` 不返回 `state` / `next_state` | 仅适合极简 smoke，不能代表真链路         |


---



### 5.5 🟢 设计限制（心里有数即可）

- **HIL 稀疏 reward**：学习慢，强依赖 reference 质量  
- **RL Token 离线训于 success embedding，在线 live VLA 抽 embedding**：同 checkpoint 下 OK；光照/相机变化会有 shift  
- **fail NPZ 不进监督训练**：只参与 demo reset，合理  
- **VLA 微调非 RLT 硬性前提**：reference 太差再考虑  
- **单卡、无多卡**：基础设施限制

---



### 5.6 代码修复优先序（真机前必做）


| 优先级 | 项                                     | 文件/位置                         |
| --- | ------------------------------------- | ----------------------------- |
| P0  | 修 `action_chunk` 赋值顺序                 | `actor_loop.py`               |
| P0  | `inference.mode` → `reference`        | `configs/plug_insertion.yaml` |
| P1  | `max_steps_per_episode` → 1800～2400   | `configs/plug_insertion.yaml` |
| P1  | reference 真机短跑，验证 VLA 动作是否合理          | 真机 + 双相机                      |
| P2  | warmup 对齐论文 500；攒够 replay 再切 `policy` | yaml + 操作流程                   |
| P3  | 增加 `rl_actor.pt` 定期 save              | `rl_server.py`（待实现）           |


---



## 6. Episode 超时与人机反应时间



### 6.1 超时机制（代码行为）

- 配置项：`online_rl.max_steps_per_episode`（当前 **600**）
- 控制频率：`control_hz: 20` → 每步 `sleep(0.05s)`
- **超时按步数计，不是按墙钟单独计时**

```text
RewardLogger.poll(step):
  step >= max_steps  →  reward=0, done=True, reason=timeout
```

超时后：

1. 本 episode 以 **失败（reward=0）** 结束
2. 写 transition / 日志
3. **自动 demo reset**，进入下一 episode（若 `--episodes` 还有剩余）

**不会**在超时后还停在同一 episode 等你按键；若没来得及按 **s**，会被记为 timeout 失败并 reset。

### 6.2 600 步 ≈ 30 秒是否太短？

**仅计控制步 sleep**：`600 ÷ 20 Hz = 30 秒`。

但每一 chunk（默认 10 步）会调用一次 **GPU VLA infer**（真实 pi05 可能 2–5 秒/次）：

```text
墙钟时间 ≈ (步数 × 0.05s) + (infer 次数 × infer 延迟)
         ≈ 30s + (600/10 × 2~5s) ≈ 2.5 ~ 5.5 分钟   （真实 VLA、infer 较慢时）
```

因此：


| 场景                       | 600 步实际感受                                 |
| ------------------------ | ----------------------------------------- |
| Mock GPU / infer 极快      | **约 30 秒**，人工反应可能偏紧                       |
| 真实 pi05                  | 墙钟往往 **更长**，步数上限可能不是首要矛盾                  |
| 插孔 critical phase + 人眼确认 | 建议仍 **加大 max_steps**，避免 infer 变快后突然变 30 秒 |


采集 demo 里 success episode 录制时长约 **2.5～8 秒**（critical 段本身不长），但 **online rollout 需要人观察、判断、按键**，应留足余量。

### 6.3 建议配置


| 场景                | 建议 `max_steps_per_episode` | 约 sleep 时间 @20Hz |
| ----------------- | -------------------------- | ---------------- |
| 首次 reference 真机试跑 | **1200**                   | ~60s             |
| 正常 HIL online RL  | **1800～2400**              | ~90–120s         |
| 调试 / 短测           | CLI `--max-steps 60`       | ~3s（仅 smoke）     |


**修改方式（任选）：**

```bash
# 临时（不改 yaml）
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 5 --max-steps 2400

# 或环境变量
export MAX_STEPS=2400
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 10
```

```yaml
# configs/plug_insertion.yaml — 推荐长期值
online_rl:
  max_steps_per_episode: 2400   # 20Hz 下 ~120s sleep；+HIL 反应 + VLA infer 余量
```



### 6.4 操作提示

- 终端必须 **聚焦**（TTY），否则 **s/f 无效**，只剩 timeout 结束  
- 看到即将成功/失败时 **尽早按 s/f**，不要等最后一秒  
- timeout 记为 **fail（reward=0）**，会污染 replay；步数够长可减少误 timeout  
- 若仍担心来不及：可提高到 **3600（~3 分钟 sleep 下限）**

---



## 7. 推荐执行流程（按顺序，勿跳步）

```
Phase 0 — 解除阻塞 + 代码修复（§5.6）
  □ 修 actor_loop.py action_chunk bug
  □ yaml: inference.mode → reference
  □ yaml: max_steps_per_episode → 2400
  □ GPU：协调 1× 空闲 24GB
  □ nvidia-smi 选卡 → export CUDA_VISIBLE_DEVICES=N
  □ GPU：bash scripts/start_gpu_rl_server.sh（inference.mode: reference）
  □ 确认 ss -tlnp | grep 8765
  □ 工控机：bash scripts/gpu/start_ssh_tunnel.sh（无 NAT 时）
  □ export GPU_SERVER_HOST=127.0.0.1

Phase 1 — Reference 真机 rollout（Stage-2d）
  □ Deoxys 臂/夹爪/双 RealSense 就绪
  □ CONFIRM=1 run_deoxys_actor.sh --episodes 1 --max-steps 1200
  □ 验证 infer、臂动、s/f、demo reset

Phase 2 — 攒 replay + 开始 TD3（Stage-3）
  □ 保持 mode: reference
  □ --max-steps 2400，多 episode，认真 s/f
  □ 盯 buffer_size → ≥ warmup_steps（500 或配置值）
  □ 出现 learner metrics → 训练已开始

Phase 3 — Policy 接管（Stage-4）
  □ GPU yaml: inference.mode: policy，重启 server
  □ 工控机命令不变，继续 rollout

Phase 4 — 可选优化
  □ 再采 30~50 条 success + VLA 微调
  □ 重算 embedding / 可选重训 RL Token
  □ 配置边界 NAT，去掉隧道依赖
```

---



## 8. 三终端分工


| 终端    | 机器  | 命令                                                                                  | 说明            |
| ----- | --- | ----------------------------------------------------------------------------------- | ------------- |
| **A** | GPU | `CUDA_VISIBLE_DEVICES=N bash scripts/start_gpu_rl_server.sh`                        | 常驻            |
| **B** | 工控机 | `bash scripts/gpu/start_ssh_tunnel.sh`                                              | 无 NAT 时常驻     |
| **C** | 工控机 | `export GPU_SERVER_HOST=127.0.0.1` `CONFIRM=1 bash scripts/run_deoxys_actor.sh ...` | rollout + s/f |


---



## 9. Episode 生命周期（人机）

```
[自动] demo reset → critical 起始位姿
[自动] GPU infer → 执行 action chunk（循环）
[你]   观察 → 按 s（成功）或 f（失败）
       └─ 若步数 ≥ max_steps → timeout（reward=0，等同失败）
[自动] 写日志、发 transition（done=true）
[自动] 打印 Episode ... done
[自动] 若还有剩余 episode → demo reset → 下一轮
```

按键含义：


| 键       | reward | 结束 episode |
| ------- | ------ | ---------- |
| **s**   | 1.0    | 是          |
| **f**   | 0.0    | 是          |
| **q**   | 0.0    | 退出整个 run   |
| timeout | 0.0    | 是（自动 fail） |


---



## 10. 数据与训练要点（讨论摘要）


| topic               | 结论                                                                                |
| ------------------- | --------------------------------------------------------------------------------- |
| success vs fail NPZ | 训练只用 **success**；fail 主要用于 **demo reset 起始 pose**                                 |
| 1:1 采集              | **非必须**；多采 success 比多采 fail 更有价值                                                  |
| VLA 微调              | **非 RLT 硬性前提**；reference 太差再考虑                                                    |
| z_rl                | GPU `rl_server` 在线算；replay 满 warmup 后喂给 TD3                                       |
| online embedding 代码 | `rl_server.py` → `embedding_extractor.py` → `openpi_wrapper.extract_embeddings()` |
| SSH 隧道 vs 直连        | **隧道 +** `127.0.0.1` **即可**；直连 `10.176.53.120:8765` 未配 NAT 时可不用                   |


---



> **注意**：GPU 用 **`configs/plug_insertion_gpu.yaml`** 的 `inference.mode`；工控机 **`plug_insertion.yaml`** 无此字段。未跑稳 reference 前保持 **`reference`**。

---

## 11. 下一步行动清单（2026-07-10 起）

**GPU**

- [x] 协调 1 张空闲 3090（`CUDA_VISIBLE_DEVICES=1` 已跑通）
- [x] 启动 server + 真机 infer + warmup 满 500
- [x] 在线 `rl_actor.pt` / `rl_critic.pt` 已落盘
- [ ] 续跑 reference，提高 success 样本、继续增大 `train_steps`
- [ ] **勿** 在未验稳前改 `inference.mode: policy`

**工控机**

- [ ] SSH 隧道常开（或配 NAT）
- [ ] 同步 deploy 树 `src/rlt/` + 双 yaml（见上表）
- [ ] reference 多 episode；终端聚焦按 **s/f**
- [ ] 首步 infer 等 **2–3 分钟** 属正常

**配置 / 工程（可选后续）**

- [ ] infer 加速或降频（缓解卡顿与 WS timeout）
- [ ] replay buffer 持久化（避免重启丢 buffer）
- [ ] `max_steps_per_episode` 按需调到 1200–2400

---

## 11-old. 下一步行动清单（历史）

---



## 12. 常见报错


| 现象                                | 原因                        | 处理                            |
| --------------------------------- | ------------------------- | ----------------------------- |
| CUDA OOM（启动）                      | GPU 被占满                   | 换空闲卡                          |
| `KeyError: exterior_image_1_left` | 无相机图                      | 开 RealSense，去掉 `--no-cameras` |
| `:8765` refused                   | server 未 listen / NAT     | 查 `ss`；用隧道                    |
| `stdin 非 TTY`                     | 无法 s/f                    | 用交互终端                         |
| 突然 reset、没来得及按键                   | **timeout**               | 增大 `max_steps`；提前按 s/f        |
| `NameError: action_chunk`         | `actor_loop` 首步 bug       | 见 §5.2(1)，修赋值顺序               |
| 臂乱动 / 不合理                         | `mode: policy` + 随机 Actor | 改 `reference`；见 §5.2(2)       |


---

| `keepalive ping timeout`            | infer 阻塞过久                | 等 2–3min；buffer 仍涨可继续；后续可优化 infer |
| infer 卡顿                          | 远程 VLA 慢 + execute_prefix=10 | 预期；非控制器故障              |

---

## 13. 一句话

**2026-07-10：双机 reference 真机已跑通，Stage-3 在线 TD3 已启动（buffer 578、train 395 步、actor/critic ckpt 已存）；仍为 reference 模式，success 少与 motion 卡顿由 infer 延迟导致。下次续跑先起 GPU server（`plug_insertion_gpu.yaml`），工控机继续 reference 攒数据；满训且动作验稳后再切 policy（Stage-4）。**