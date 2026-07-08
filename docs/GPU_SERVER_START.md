# GPU 端启动 smq&jgy JPEG WebSocket Server

> **工控机连的是 `10.176.53.120:8765`**（工控机 `10.162.132.11` 能 ping 通）。  
> **fvl08（192.168.110.18）与工控机不在同一可达网段**，在 fvl08 上起 server 工控机连不上。

---

## 1. 从工控机同步代码（在工控机执行）

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/export_smq_for_gpu.sh
scp data/export_for_gpu_host/smq_jgy_rl_server_*.tar.gz YOUR_USER@10.176.53.120:~/
```

或 rsync 整包：

```bash
rsync -av --exclude '.git' --exclude 'logs' --exclude 'third_party/deoxys' \
  --exclude 'data/episodes' \
  "/home/host5010/workspaces/smq&jgy/" \
  YOUR_USER@10.176.53.120:~/smq\&jgy/
```

---

## 2. 在 GPU（10.176.53.120）上解压并装环境

```bash
mkdir -p ~/smq_jgy && cd ~/smq_jgy
tar -xzf ~/smq_jgy_rl_server_*.tar.gz
cd smq\&jgy/rlt_project/rlt_reproduce   # 或解压后的等价路径

# conda 环境（与工控机 franka_mani 类似，或 rlt GPU 环境）
conda activate rlt   # 或 franka_mani + pip install websockets pyyaml rich
pip install -e . 2>/dev/null || export PYTHONPATH=$PWD/src
```

---

## 3. 启动 JPEG WebSocket server（不是 msgpack inference_policy）

```bash
cd ~/smq_jgy/smq\&jgy/rlt_project/rlt_reproduce

# 必须看到: Inference mode: reference
# 必须看到: RL server listening on ws://0.0.0.0:8765
bash scripts/run_rl_server.sh configs/plug_insertion.yaml

# 或显式 CPU 冒烟:
DEVICE=cpu bash scripts/run_rl_server.sh configs/plug_insertion.yaml
```

**正确栈**：`run_rl_server.sh` → `python -m rlt.scripts.rl_server` → JSON + `images_jpeg` base64。

**错误栈**：rlt_reproduce 独立仓里旧的 msgpack `{type: act|transition}` handler。

---

## 4. 防火墙 / 监听

```bash
ss -tlnp | grep 8765          # 0.0.0.0:8765
sudo ufw allow 8765/tcp       # 若启用 ufw
```

从工控机验证：

```bash
cd "/home/host5010/workspaces/smq&jgy"
bash scripts/ping_gpu_server.sh
```

---

## 5. 工控机双机 smoke（GPU server 起来后）

```bash
cd "/home/host5010/workspaces/smq&jgy"
MOCK=1 bash scripts/run_deoxys_actor.sh --no-cameras --episodes 1 --max-steps 5
```

期望工控机 log：

```
GPU server {'type': 'pong', ...}
GPU infer meta {'policy_mode': 'reference', 'z_rl_norm': ..., ...}
```

---

## IP 对照

| 机器 | IP | 工控机可达 | 用途 |
|------|-----|-----------|------|
| 工控机 | 10.162.132.11 | — | actor |
| GPU（配置） | **10.176.53.120** | ✅ ping OK | **在此起 server** |
| fvl08 | 192.168.110.18 | ❌ ping 超时 | 勿改 yaml 指此 IP |
