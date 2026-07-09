# SFT / Imitation Learning 数据采集指南

> Franka FR3 + Franka Hand + RealSense D435×2  
> 输出：**JSONL + PNG 图像目录**，可直接用于 VLA SFT、RL Token embedding 抽取、RLT 在线 RL 训练。

---

## 快速开始

### 1. 启动硬件（3 个终端）

```bash
cd "/home/host5010/workspaces/smq&jgy"

# 终端 1 — 机械臂
bash scripts/start_arm.sh

# 终端 2 — 夹爪
bash scripts/start_gripper.sh

# 终端 3 — SFT 采集
bash scripts/collect_sft_data.sh
```

### 2. 采集流程

**先试 reset + 遥操（不写盘）：**

```bash
MOCK=1 bash scripts/collect_sft_data.sh
# 或: bash scripts/collect_sft_data.sh --  # 加 python --mock
```

| 按键 | MOCK=1 | 正式采集 |
|------|--------|----------|
| **r** | 快速 reset + 试遥操 | reset + **开始录制** |
| **s** / **f** | 无效（提示去掉 mock） | 保存成功/失败 episode |
| **q** | 退出 | 退出 |

**正式采集（写 JSONL + PNG）：**

```bash
bash scripts/collect_sft_data.sh
# 或 MOCK=0 bash scripts/collect_sft_data.sh
```

### 3. 输出位置

```
data/sft/plug_insertion/
  ep_00000/
    meta.json              # episode 元数据
    transitions.jsonl      # 每步一条 JSON
    images/
      rgb_front/000000.png # external 相机（固定机位）
      rgb_wrist/000000.png # wrist 相机
  ep_00001/
    ...
```

---

## 数据结构

### `meta.json`

```json
{
  "episode_id": "ep_00000",
  "task": "plug_insertion",
  "language": "plug the charger into the power socket",
  "success": true,
  "fps": 50.0,
  "num_steps": 312,
  "dropped_steps": 0,
  "max_sync_delta_ms": 28.4,
  "workspace_offset_xy": [0.023, -0.041],
  "reset_target_xyz": [0.700, -0.057, 0.189]
}
```

### `transitions.jsonl`（每行一条）

| 字段 | 类型 | 说明 |
|------|------|------|
| `step` | int | 步序号，从 0 开始 |
| `timestamp` | float | episode 内相对时间 (s) |
| `robot_timestamp` | float | 主机 wall-clock 时间 |
| `robot_state_timestamp` | float \| null | libfranka 状态时间（若有） |
| `ee_pos` | [x,y,z] | 末端位置 (m)，**Franka base 系** |
| `ee_quat` | [w,x,y,z] | 末端姿态四元数，**Franka base 系** |
| `gripper_width` | float | 夹爪开口宽度 (m) |
| `action` | [7] | 7D delta 命令（见下） |
| `rgb_front_path` | str | 相对路径 → external/front 相机 |
| `rgb_wrist_path` | str | 相对路径 → wrist 相机 |
| `camera_rgb_front_timestamp` | float | front 帧时间戳 |
| `camera_rgb_wrist_timestamp` | float | wrist 帧时间戳 |
| `sync_delta_ms` | float | 本步最大相机-robot 同步误差 |
| `per_camera_sync_delta_ms` | dict | 各相机单独误差 |

### 动作表示 `action[7]`

| 索引 | 含义 | 单位 |
|------|------|------|
| 0–2 | Δx, Δy, Δz | 米 (m)，Franka base 系平移增量 |
| 3–5 | Δrx, Δry, Δrz | 轴角向量 (axis-angle)，弧度 |
| 6 | gripper | `+1` 闭合，`-1` 张开 |

缩放系数（写入 jsonl 前已乘）：`action_scale = [0.05, 0.02, 1.0]`（见 `configs/sft_plug_insertion.yaml`）。

### 坐标系

| 数据 | 坐标系 |
|------|--------|
| `ee_pos`, `ee_quat`, `action[:6]` | **Franka 机器人 base link**（libfranka `O_T_EE`） |
| `rgb_front`, `rgb_wrist` | **RealSense 光学坐标系**（与 robot base 独立） |

---

## 任务设定

- **插座固定**，相机治具固定
- 每次按 **r** 时，`reset_random_workspace()` 在 **init 立方体底面**（10 cm × 10 cm）内随机采样 EE 初始位置
- 控制频率 **50 Hz**；相机 **30 Hz**（D435 稳定值），逐步 timestamp 对齐

**Init 立方体定义**（须先标定 `bottom_center`）：

```bash
# 1) 遥操到插座正上方（你现在这个位置）
# 2) Ctrl+C 停 teleop
# 3) 记录并写入 yaml
WRITE=1 bash scripts/record_bottom_center.sh
```

`bottom_center_xyz` = 你标定的插座正上方 EE 点 = **立方体底面中心**（当前约 z≈0.202 m）。  
**Online RL actor 与 SFT 采集共用同一套 reset**（`configs/sft_plug_insertion.yaml`）。

### 单独测 reset（不采集）

先启动机械臂：

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/start_robot.sh
```

```bash
bash scripts/reset_to_init.sh           # 随机 init cube（默认）
FIXED=1 bash scripts/reset_to_init.sh   # 固定底面中心
```

标定 / 更新底面中心（换插座时）：

```bash
bash scripts/record_bottom_center.sh          # 预览
WRITE=1 bash scripts/record_bottom_center.sh  # 写入 yaml
```

`workspace_randomization`（`configs/sft_plug_insertion.yaml`）：

```yaml
workspace_randomization:
  bottom_center_xyz: [...]  # record 脚本写入
  xy_half_range_m: 0.05     # 底面 ±5 cm
  z_range_m: 0.0            # 仅底面；>0 则向上随机
```

**Reset 运动**：先 **xy 平移**（保持当前高度）→ 再 **仅降 z**，且 **不会低于 bottom_center.z**。

---

## 核心模块

| 模块 | 路径 | 函数 |
|------|------|------|
| 快速 reset | `rlt/hardware/deoxys/fast_reset.py` | `reset_to_collection_init()` |
| 随机 init | `rlt/hardware/workspace_reset.py` | `reset_random_workspace()` |
| 时间同步 | `rlt/data/sync_sampler.py` | `sync_camera_robot_timestamp()` |
| 逐步保存 | `rlt/data/sft_io.py` | `save_transition()` |
| 录制循环 | `rlt/data/sft_recorder.py` | `SFTRecorder.record_episode()` |
| 主入口 | `rlt/scripts/collect_sft_plug_insertion.py` | CLI |

---

## 下游训练用法

### VLA SFT（openpi）

直接读取 JSONL + PNG；`rgb_front` 映射为 `observation/exterior_image_1_left`，`rgb_wrist` 映射为 `observation/wrist_image_left`。

### RL Token embedding

若需沿用现有 NPZ 工具链：

```bash
cd "/home/host5010/workspaces/smq&jgy/rlt_project/rlt_reproduce"
conda activate rlt   # 或 franka_mani
python -m rlt.scripts.export_sft_to_npz --config configs/sft_plug_insertion.yaml
python -m rlt.scripts.extract_vla_embeddings --config configs/sft_plug_insertion.yaml
```

### 与旧版 NPZ 采集的关系

| 脚本 | 格式 | 频率 | 用途 |
|------|------|------|------|
| `scripts/collect_data.sh` | NPZ | 20 Hz | RLT critical-phase 原始流程 |
| `scripts/collect_sft_data.sh` | JSONL+PNG | 50 Hz | **SFT / 全轨迹 imitation** |

---

## 常见 Bug Checklist

### 启动前

- [ ] CPU governor 为 **performance**（`cpupower frequency-info`）
- [ ] `bash scripts/start_arm.sh` 已运行（`pgrep -f franka-interface`）
- [ ] `bash scripts/start_gripper.sh` 显示 **Gripper homing complete**
- [ ] 相机 serial 与 `configs/sft_plug_insertion.yaml` 中 `cameras.mapping` 一致（`bash scripts/detect_cameras.sh`）
- [ ] 工作空间无障碍，急停可用

### 相机

- [ ] 若报 **Device busy**：先 `bash scripts/free_cameras.sh`
- [ ] SSH 无 DISPLAY 时不要用 `deoxys/examples/test_camera.py`；用 `bash scripts/test_camera.sh`
- [ ] `dropped_steps` 持续增加 → 检查 USB 带宽、降低 `image_size` 或略放宽 `max_sync_delta_ms`
- [ ] 两路相机 timestamp 差 > 50 ms → 检查线缆/Hub，重启相机服务

### 机械臂 / 遥操

- [ ] SpaceMouse 右键后不动 → 已修复（`acknowledge_spacemouse_reset`）；若仍卡住，重启采集脚本
- [ ] reset 超时 → 缩小 `xy_half_range_m` 或检查碰撞；先单独跑 `bash scripts/reset_to_init.sh` 看日志
- [ ] 夹爪不闭合 → 确认 Terminal 2 夹爪服务正常；按 **g** 或 SpaceMouse 左键

### 数据质量

- [ ] 每个 episode 的 `meta.json` 中 `dropped_steps` 应为 **0**（或可接受的小值）
- [ ] `max_sync_delta_ms` < 配置的 `max_sync_delta_ms`（默认 50 ms）
- [ ] 录制前按 **r** 触发 reset；不要手动挪臂后开始录（起点不一致）
- [ ] 插孔前 1–3 s 开始录 vs 全轨迹：SFT 默认录 **全轨迹**（从 reset 后到 s/f）

### 多进程冲突

- [ ] 勿同时运行 `collect_data.sh`、`run_deoxys_actor.sh`、`teleop_test.sh`
- [ ] 采集前自动执行 `free_deoxys_client.sh`；若仍报 zmq 冲突，手动 kill 旧客户端

### 路径

- [ ] 目录名含 `&`，所有 `cd` 必须加引号：`cd "/home/host5010/workspaces/smq&jgy"`

---

## 配置文件

主配置：`configs/sft_plug_insertion.yaml`

环境变量：

```bash
export SFT_COLLECT_CONFIG="/home/host5010/workspaces/smq&jgy/configs/sft_plug_insertion.yaml"
bash scripts/collect_sft_data.sh
```

---

## 相关文档

- [机械臂启动说明.md](../机械臂启动说明.md) — 硬件 bring-up
- [ONLINE_RL_TASKS.md](../ONLINE_RL_TASKS.md) — 在线 RL 联调
- [RLT训练包部署指南.md](../RLT训练包部署指南.md) — GPU 端训练
