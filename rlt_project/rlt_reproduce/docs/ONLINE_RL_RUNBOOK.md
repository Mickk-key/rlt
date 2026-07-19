# RLT Online-RL — Authoritative Runbook (Plug Insertion)

> **THIS IS THE SINGLE SOURCE OF TRUTH** for the complete dual-machine RLT online-RL
> workflow after the Phase 1–5 fixes. If any other doc disagrees, follow this file.
>
> **Last updated:** 2026-07-19 · **Repo revision baseline:** `origin/main @ 07b9a8e`
> **Task:** Franka plug insertion, split online RL (RLT paper, Algorithm 1)
> **GPU host:** fvl08 (`10.176.53.120`, internal `192.168.110.18`) — runs `rl_server`
> **Robot PC:** `10.162.132.11`, workspace `.../smq&jgy` — runs `actor_loop` + Deoxys
>
> **Superseded by this runbook** (kept for history only): `docs/GPU_SERVER_START.md`,
> `docs/ONLINE_RL_ROBOT.md`, `docs/desktop/online_rl_worklog.md`,
> `docs/ONLINE_RL_WORK_PLAN.md`, `ONLINE_RL_TASKS.md`.

---

## 0. What changed (Phase 1–5) — read first

| Phase | Change | File (runtime tree) | Effect |
|-------|--------|---------------------|--------|
| **1** | Hard EE-delta safety clamp | `deoxys_env.py` (Tree B) | Bounds physical per-step delta for **both** reference and policy; NaN→hold pose |
| **2** | Paper-faithful deployment anchor | `inference_policy.py` (Tree B) | `action = reference + clip(actor − reference, ±δ)`; actor stays absolute `a~π(a\|x,ref)`, **not** residual |
| **3** | Warmup gating + linear ramp | `inference_policy.act_gated` / `rl_server.py` (Tree B) | Execution keyed on replay-buffer transition count, not a manual switch |
| **4** | Real chunk transitions | `actor_loop.py` (Tree C) + `rl_server.py`/`learner.py`/`replay_buffer.py` (Tree B) | `(x_s, a_{s:s+C}, ref_{s:s+C}, ref_{s+C:s+2C}, R=Σγ^k r, x_{s+C}, done)`; no tiling; `γ^C` |
| **5** | Fresh actor/critic reset | checkpoints + config | Actor/critic/replay restart from scratch; SFT VLA + RL token **unchanged** |

**Reuse (never delete/retrain here):** SFT VLA (`pi05_base`, `pi05_plug_insertion`) and RL token (`rl_token.pt`).
**Online-RL components that get reset/restarted:** actor (`rl_actor.pt`), critic (`rl_critic.pt`), and the in-memory replay buffer.

---

## 1. Runtime source-tree mapping (critical)

Both source trees live in **one git repo** (`smq&jgy`). At runtime, which copy loads depends on `PYTHONPATH`:

| Component | Runtime tree | Exact path | Why |
|-----------|--------------|-----------|-----|
| `actor_loop.py` (robot rollout) | **Tree C** | `smq&jgy/src/rlt/scripts/actor_loop.py` | `_env.sh` puts `smq&jgy/src` **first** on `PYTHONPATH`, overriding Tree B |
| `gpu_client.py` (robot↔GPU) | **Tree C** | `smq&jgy/src/rlt/rl/gpu_client.py` | same |
| `rl_server.py` (GPU) | **Tree B** | `rlt_project/rlt_reproduce/src/rlt/scripts/rl_server.py` | `run_rl_server.sh` uses only Tree B `src` |
| `learner.py` (TD3) | **Tree B** | `.../rlt_reproduce/src/rlt/rl/learner.py` | GPU side |
| `replay_buffer.py` | **Tree B** | `.../rlt_reproduce/src/rlt/rl/replay_buffer.py` | GPU side |
| `inference_policy.py` (gating + anchor) | **Tree B** | `.../rlt_reproduce/src/rlt/rl/inference_policy.py` | GPU side |
| `deoxys_env.py` (safety clamp) | **Tree B** | `.../rlt_reproduce/src/rlt/hardware/deoxys/deoxys_env.py` | robot execs with `cd RLT_ROOT`=Tree B |

**Config paths:**

| Config | Loaded by | Path |
|--------|-----------|------|
| GPU server | `run_rl_server.sh` (Tree B) | `rlt_project/rlt_reproduce/configs/plug_insertion_gpu.yaml` |
| Robot actor | `_env.sh` → `RLT_COLLECT_CONFIG` | `smq&jgy/configs/plug_insertion.yaml` (Tree C) |

> **Sync rule:** `actor_loop.py` and `gpu_client.py` transition logic is kept **byte-identical**
> across Tree B and Tree C (only docstrings + `main()` bootstrap differ). Edit one → edit both.
> A Tree C mirror of `plug_insertion_gpu.yaml` exists for parity but is **not** the one `rl_server` loads.

---

## 2. Architecture

```
Robot PC 10.162.132.11                     GPU fvl08 / 10.176.53.120
┌────────────────────────────┐             ┌─────────────────────────────┐
│ RealSense D435 ×2 (USB)     │   JPEG WS   │ start_gpu_rl_server.sh       │
│ Deoxys arm + gripper (ZMQ)  │  ─────────► │  rl_server.py                │
│ actor_loop.py (Tree C)      │   :8765     │  pi05 VLA + RL token         │
│ RewardLogger (s/f/q keys)   │             │  actor/critic + replay (TD3) │
└────────────────────────────┘             └─────────────────────────────┘
```

- GPU needs **1 whole GPU** (~24 GB free). No multi-GPU. Robot PC uses **0 GPU**.
- Transport is JPEG-base64 JSON over WebSocket. GPU has **no** cameras.

---

## 3. Prerequisites

- [ ] GPU: ≥1 free ~24 GB card; `checkpoints/rl_token.pt`, `pi05_base`, `pi05_plug_insertion` present.
- [ ] GPU env: launch **only** via `start_gpu_rl_server.sh` (it `source`s `activate_rlt.sh`, which activates the `rlt` conda env **and** adds the openpi `.venv` to `PYTHONPATH` for `jax`/`openpi`/`websockets`). Running `run_rl_server.sh` directly from `(base)` → `ModuleNotFoundError: websockets`.
- [ ] Robot: Deoxys arm/gripper SOP works; RealSense serials match yaml; terminal is a real TTY (for s/f keys).
- [ ] Robot ↔ GPU reachable (`:8765` direct, or SSH tunnel + `GPU_SERVER_HOST=127.0.0.1`).

---

## 4. Startup commands

Paths contain `&` — **always quote** them (`&` is a shell operator).

### 4.1 GPU server (GPU terminal, keep open)

```bash
cd "/sdb/private_folders/shimingqi/smq_jgy_deploy/smq&jgy/rlt_project/rlt_reproduce"
CUDA_VISIBLE_DEVICES=<free_gpu> bash scripts/start_gpu_rl_server.sh configs/plug_insertion_gpu.yaml
```

Wait for (first pi05 load can take minutes):

```text
Loaded RL token from checkpoints/rl_token.pt
Inference mode: reference
RL server listening on ws://0.0.0.0:8765
```

Confirm: `ss -tlnp | grep 8765` → `LISTEN 0.0.0.0:8765`.

### 4.2 SSH tunnel (robot terminal, if no NAT, keep open)

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/gpu/start_ssh_tunnel.sh
export GPU_SERVER_HOST=127.0.0.1
```

### 4.3 Ping / health check (robot terminal)

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/ping_gpu_server.sh
```

Expected fields: `buffer_size`, `exec_state`, `inference_mode`, `warmup_steps=500`, `ramp_steps=500`, `train_steps`, `training`.
- Forced reference run → `exec_state: override:reference`.

### 4.4 Franka arm + gripper (robot terminal)

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/start_arm.sh
bash scripts/start_gripper.sh
```

### 4.5 Robot actor loop (robot terminal, after server is `listening`)

```bash
cd "/home/host5010/workspaces/smq&jgy"
export GPU_SERVER_HOST=127.0.0.1     # if tunneling
CONFIRM=1 bash scripts/run_deoxys_actor.sh --episodes 1 --max-steps 200
```

- `run_deoxys_actor.sh` layers robot env then execs `run_actor_loop.sh` → `actor_loop.py` (Tree C), config `smq&jgy/configs/plug_insertion.yaml`.
- Never use `MOCK=1` / `--no-cameras` for real VLA (→ `KeyError: observation/exterior_image_1_left`).
- Keys (terminal focused): **s** = success (reward=1, done), **f** = fail (reward=0, done), **q** = quit run. Timeout at `max_steps_per_episode=200` = fail.

---

## 5. Reference smoke test (do this first, every fresh run)

Goal: confirm the reference pipeline + safety are healthy **before** collecting warmup data.

Run 1 short episode (§4.5, `--episodes 1 --max-steps 200`). **Check:**

- [ ] Actor first-step log: `GPU infer meta {'policy_mode': 'reference', ...}` with sane `z_rl_norm`, `ref_norm`.
- [ ] Arm moves smoothly toward the socket; magnitude looks like the SFT demo (not jerky/huge).
- [ ] `s`/`f` ends the episode; auto reset runs before the next.
- [ ] Reward JSON written: `logs/online_rl/rewards/ep_XXXX.json`.

**Safety / abnormal-motion STOP conditions** (Phase 1 clamp in `deoxys_env.step`, applies to reference too):

| Symptom | What it means | Action |
|---------|---------------|--------|
| `[safety] clamp` firing on almost every step with large raw norms | reference deltas exceed 0.02 m / 0.1 rad often | **STOP** (Ctrl-C). Check `action_is_physical: true`, VLA/action-space alignment |
| `NaN`/`Inf` action logged, arm holds pose | bad model output | **STOP**, inspect VLA/RL-token load; do not continue |
| Large-amplitude / unexpected motion, E-stop | pipeline mismatch | **STOP**, hit E-stop, do not switch to policy |
| Per-step translation > 0.02 m or rotation > 0.1 rad reaching the arm | clamp bypassed | **STOP**, verify safety block present in `configs/plug_insertion.yaml` |

Safety limits (robot config, `smq&jgy/configs/plug_insertion.yaml`): `max_trans_delta_m: 0.02`, `max_rot_delta_rad: 0.1`, gripper `[-1, 1]`.

---

## 6. Reference warmup (forced reference)

Fill the replay buffer with **reference-only** trajectories. Execution is forced to reference regardless of buffer count.

- GPU config `inference.mode: reference` (already set for the fresh run) → `act_gated` returns `alpha=0` always.
- `online_rl.warmup_steps: 500`, `chunk_length: 10`, robot `online_rl.subsample_stride: 2`.
- Each finished episode emits real chunk transitions at stride 2: `(x_s, a_{s:s+C}, ref_{s:s+C}, ref_{s+C:s+2C}, R, x_{s+C}, done)`. Terminal chunk padding: action/reference = last valid step, reward = 0.

Run multiple episodes (§4.5, e.g. `--episodes 20 --max-steps 200`), pressing **s/f** honestly.

**Do NOT proceed to auto until ALL of these hold:**

1. [ ] `buffer_size >= 500` (via `ping`).
2. [ ] Transition shapes/rewards/done verified — guaranteed by construction (robot `_validate_chunk_transition` and server strict `_as_action_chunk` both **raise** on malformed `(C=10,7)` / wrong gap; a running server with growing `buffer_size` and no shape errors = OK). Spot-check reward JSONs.
3. [ ] **At least 2–3 successful reference episodes** (`reason: success_key`). Buffer≥500 that is almost all failures is **not** enough.

Quick success/quality check (robot):

```bash
python3 - <<'PY'
import glob, json, os
base="/home/host5010/workspaces/smq&jgy"
dirs=[base+"/rlt_project/rlt_reproduce/logs/online_rl/rewards", base+"/logs/online_rl/rewards"]
fs=sum((sorted(glob.glob(d+"/*.json")) for d in dirs if os.path.isdir(d)),[])
succ=sum(json.load(open(f))["reason"]=="success_key" for f in fs)
print(f"episodes={len(fs)} successes={succ} -> {'PASS' if succ>=2 else 'NOT YET'}")
PY
```

---

## 7. Actor/critic checkpoint verification (before switching modes)

During forced reference, once `buffer_size >= warmup_steps` the learner trains and checkpoints. Before switching to auto, confirm:

```bash
cd "/sdb/private_folders/shimingqi/smq_jgy_deploy/smq&jgy/rlt_project/rlt_reproduce"
ls -la checkpoints/rl_actor.pt checkpoints/rl_critic.pt
ls -1 checkpoints/online_rl/ | tail
```

- [ ] `rl_actor.pt` / `rl_critic.pt` exist with a **recent** mtime.
- [ ] `online_rl/` snapshots exist with increasing `stepNNNN_bufNNN` (via `ping`: `train_steps` > 0, `training: true`).
- [ ] Load sanity (optional): weights finite, shapes match config (`actor_hidden`, `critic_hidden`, `action_dim=7`).

If actor/critic are missing/stale, keep collecting reference data — do not switch.

---

## 8. Switch reference → auto (enable policy ramp)

The gate is **buffer transition count**, so a clean 0→1 ramp requires the buffer to start from 0. Since restarting `rl_server` resets the in-memory buffer, the switch is naturally clean.

**Procedure:**

1. Edit GPU config `rlt_project/rlt_reproduce/configs/plug_insertion_gpu.yaml`: `inference.mode: reference` → `auto`. (Keep the Tree C mirror in parity.)
2. **Restart** `rl_server` (§4.1). Restart **resets replay buffer to 0**; trained `rl_actor.pt`/`rl_critic.pt` are reloaded from disk.
3. Auto gating (`act_gated`), with `warmup_steps=500`, `ramp_steps=500`:

| `buffer_size` | Executed action | `alpha` |
|---------------|-----------------|---------|
| `0 – 499` | VLA reference | 0 |
| `500 – 999` | linear blend reference→anchored policy | `(buffer−500)/500` |
| `≥ 1000` | anchored policy `ref + clip(actor−ref, ±δ)` | 1 |

Anchor limits (GPU config `policy_anchor`): `max_dev_trans_m: 0.01`, `max_dev_rot_rad: 0.05`, `max_dev_grip: 1.0`.

> Restart implication: auto re-warms 500 reference transitions (safe) before any policy blending. This re-collection is intentional, not waste. If you build buffer persistence later, set `warmup_steps` to the loaded buffer count so the ramp still starts at `alpha=0`.

---

## 9. Monitoring during ramp / online RL

**Watch (GPU `rl_server` log + `ping`):**

- `buffer_size`, `exec_state` (`warmup_reference` → `ramp(alpha=…)` → `policy`), `alpha`.
- `train_steps` increasing, `training: true`.
- Learner metrics: `critic_loss`, `actor_loss`, `bc_loss` (BC regularization). Should stay bounded.
- Per-step action stats: `ref_norm`, `actor_norm`, `deviation = ‖actor − ref‖`, clipped-deviation amount.

**Watch (robot log):**

- `[safety] clamp` frequency and raw vs clipped norms.
- Motion smoothness, reward keypresses, reset success.

**STOP conditions (Ctrl-C / E-stop, revert to `mode: reference`, investigate):**

- Any `NaN`/`Inf` action, or safety clamp firing hard every step as `alpha` rises.
- Actor deviation **constantly saturating** the anchor clip (0.01 m / 0.05 rad) → actor pulling away from reference; lower ramp rate or collect more warmup.
- `critic_loss` / `actor_loss` diverging (orders-of-magnitude growth).
- Success rate drops **below reference** as `alpha` increases, or motion becomes unsafe.
- `buffer_size` not growing (dropped transitions / WS timeouts) — fix connectivity before trusting training.

---

## 10. Checkpoint / restart / recovery

**Keep (never delete):** `checkpoints/rl_token.pt` (→ sft5000 real token), `checkpoints/pi05_base`, `checkpoints/pi05_plug_insertion`.

**Online-RL, resettable:** `checkpoints/rl_actor.pt`, `checkpoints/rl_critic.pt`, `checkpoints/online_rl/`, in-memory replay buffer.

**Fresh reset (Phase 5) — back up first, quote `&` paths:**

```bash
cd "/sdb/private_folders/shimingqi/smq_jgy_deploy/smq&jgy/rlt_project/rlt_reproduce"
mkdir -p "backups/reset_$(date +%Y%m%d_%H%M%S)"
cp -a "checkpoints/rl_actor.pt" "checkpoints/rl_critic.pt" "checkpoints/online_rl" "backups/reset_$(date +%Y%m%d_%H%M%S)/" 2>/dev/null || true
rm -f "checkpoints/rl_actor.pt" "checkpoints/rl_critic.pt"
rm -rf "checkpoints/online_rl"
```

- On next start, `rl_server` finds no `rl_actor.pt`/`rl_critic.pt` → **random init from scratch** (paper-faithful); `rl_token.pt` + SFT VLA load normally; buffer starts empty.
- Set GPU config `inference.mode: reference` for the fresh run (see §6).

**Recovery / continue (no reset):** just restart `rl_server` — actor/critic reload from `rl_actor.pt`/`rl_critic.pt`; buffer re-warms from 0 (reference) before any policy.

---

## 11. End-to-end checklist

**Fresh run, first time:**

- [ ] GPU: free card chosen (`nvidia-smi`); `rl_token.pt` + SFT VLA present.
- [ ] (If resetting) backup + delete `rl_actor.pt`/`rl_critic.pt`/`online_rl/`; GPU config `inference.mode: reference`.
- [ ] GPU: `start_gpu_rl_server.sh` → `RL server listening`; `ss -tlnp | grep 8765`.
- [ ] Robot: tunnel up (if needed) + `GPU_SERVER_HOST=127.0.0.1`; `ping` OK, `exec_state: override:reference`.
- [ ] Robot: arm + gripper started.
- [ ] **§5 smoke test** (1 episode) passes; no NaN / no hard clamp / no abnormal motion.
- [ ] **§6 warmup**: `buffer_size ≥ 500` **AND** ≥ 2–3 `success_key` episodes; shapes/rewards/done verified.
- [ ] **§7**: `rl_actor.pt`/`rl_critic.pt` recent; `train_steps > 0`.
- [ ] **§8 switch**: config `mode: auto`; **restart** `rl_server` (buffer→0).
- [ ] **§9 monitor** ramp `alpha 0→1` over buffer 500→1000; stop on any listed condition.
- [ ] Policy (`alpha=1`, buffer ≥ 1000): success ≥ reference and stable → continue; else revert to `reference` and investigate.

**Do NOT:** retrain SFT/RL-token here; make the actor a residual policy; run policy before warmup+QA; skip `activate_rlt.sh` on the GPU; run real VLA with `--no-cameras`.
