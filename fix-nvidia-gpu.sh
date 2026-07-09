#!/bin/bash
# NVIDIA GPU 驱动修复脚本
# 问题：当前 PREEMPT_RT 实时内核不支持 NVIDIA 专有驱动
# 解决：切换到标准 generic 内核 + 安装预编译驱动模块

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "请使用 sudo 运行: sudo bash $0"
    exit 1
fi

CURRENT_KERNEL="$(uname -r)"
echo "当前内核: ${CURRENT_KERNEL}"

if [[ "${CURRENT_KERNEL}" == *"rt"* ]] || [[ "${CURRENT_KERNEL}" == *"realtime"* ]]; then
    echo ""
    echo "错误: 仍在实时(RT)内核上运行，NVIDIA 驱动无法在此内核加载。"
    echo "请先重启，在 GRUB 菜单选择: Advanced options -> Ubuntu, with Linux 6.8.0-124-generic"
    echo "进入 generic 内核后再运行本脚本。"
    exit 1
fi

echo ""
echo "=== 步骤 1: 安装预编译 NVIDIA 内核模块 (无需 DKMS 编译) ==="
apt-get update
apt-get install -y \
    linux-modules-nvidia-595-open-generic-hwe-22.04 \
    nvidia-driver-595

echo ""
echo "=== 步骤 2: 修复未完成的包配置 ==="
apt-get install -f -y
dpkg --configure -a

echo ""
echo "=== 步骤 3: 加载 NVIDIA 内核模块 ==="
modprobe nvidia || true
modprobe nvidia_uvm || true

echo ""
echo "=== 步骤 4: 验证 GPU ==="
if nvidia-smi; then
    echo ""
    echo "GPU 驱动安装成功!"
else
    echo ""
    echo "nvidia-smi 仍失败，请重启后再试: sudo reboot"
    exit 1
fi

echo ""
echo "=== 可选: 将默认启动内核改为 generic (保留 RT 内核供实时任务使用) ==="
GRUB_FILE="/etc/default/grub"
if grep -q '^GRUB_DEFAULT=0' "${GRUB_FILE}"; then
    # 找到 6.8.0-124-generic 在 grub 菜单中的序号
    MENU_ENTRY=$(grep -n "menuentry.*6.8.0-124-generic" /boot/grub/grub.cfg | head -1 | cut -d: -f1 || true)
    if [[ -n "${MENU_ENTRY}" ]]; then
        echo "如需每次默认启动 generic 内核，可手动编辑 ${GRUB_FILE}"
        echo "将 GRUB_DEFAULT 改为对应菜单项序号，然后运行: update-grub"
    fi
fi

echo ""
echo "完成。注意: 实时内核 (5.15.0-*-realtime) 下仍无法使用 NVIDIA GPU，"
echo "需要 GPU 时请启动 6.8.0-*-generic 内核。"
