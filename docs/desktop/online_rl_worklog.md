# Plug Insertion Online RL — 工作文档

> ⚠️ **已整合**：完整内容（含主机/GPU 侧）见  
> **`rlt_project/rlt_reproduce/docs/ONLINE_RL_WORK_PLAN.md`**（整合版，带来源标记 `[工控机]` / `[主机]`）。  
> 本文档保留作工控机桌面快捷入口；**请以整合版为准**，避免双份维护分叉。

> 工控机路径：`/home/host5010/workspaces/smq&jgy/docs/desktop/online_rl_worklog.md`  
> 最后更新：**2026-07-11**

---

## 工控机侧速查（摘要）

| 项 | 内容 |
|----|------|
| 角色 | deoxys、SFT 采集、`actor_loop`、SSH 隧道；**不跑 VLA** |
| Actor 配置 | `configs/plug_insertion.yaml` |
| Reset | `reset_mode: workspace`，`reset_method: external`，`reset_config: configs/sft_plug_insertion.yaml` |
| Init cube 底面 z | ≈ **0.202 m** |
| 开跑 | 隧道 + `export GPU_SERVER_HOST=127.0.0.1` + `CONFIRM=1 bash scripts/run_deoxys_actor.sh ...` |

**故障排查（工控机）** → 见整合版 **§14.2 Reset**、**§14.3 Online infer**。

**详细进度、算法 FAQ、GPU 配置、SFT/RL Token** → 见整合版全文。
