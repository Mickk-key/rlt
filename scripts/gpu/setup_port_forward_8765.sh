#!/usr/bin/env bash
# Run on the host that owns 10.176.53.120 (or campus border router) — requires root.
# Forwards 10.176.53.120:8765 -> fvl08 192.168.110.18:8765 (JPEG rl_server).
#
# Robot subnet: 10.162.0.0/16 (工控机 10.162.132.11)
set -euo pipefail

ROBOT_CIDR="${ROBOT_CIDR:-10.162.0.0/16}"
PUBLIC_IP="${PUBLIC_IP:-10.176.53.120}"
PUBLIC_PORT="${PUBLIC_PORT:-8765}"
BACKEND_IP="${BACKEND_IP:-192.168.110.18}"
BACKEND_PORT="${BACKEND_PORT:-8765}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root: sudo bash $0" >&2
  exit 1
fi

sysctl -w net.ipv4.ip_forward=1
grep -q '^net.ipv4.ip_forward=1' /etc/sysctl.conf 2>/dev/null || \
  echo 'net.ipv4.ip_forward=1' >> /etc/sysctl.conf

# Remove old rules (ignore errors)
iptables -t nat -D PREROUTING -d "$PUBLIC_IP" -p tcp --dport "$PUBLIC_PORT" -j DNAT --to-destination "${BACKEND_IP}:${BACKEND_PORT}" 2>/dev/null || true
iptables -t nat -D POSTROUTING -d "$BACKEND_IP" -p tcp --dport "$BACKEND_PORT" -j MASQUERADE 2>/dev/null || true

iptables -t nat -A PREROUTING -d "$PUBLIC_IP" -p tcp --dport "$PUBLIC_PORT" \
  -j DNAT --to-destination "${BACKEND_IP}:${BACKEND_PORT}"
iptables -t nat -A POSTROUTING -d "$BACKEND_IP" -p tcp --dport "$BACKEND_PORT" -j MASQUERADE

iptables -A FORWARD -p tcp -d "$BACKEND_IP" --dport "$BACKEND_PORT" -j ACCEPT
iptables -A FORWARD -p tcp -s "$BACKEND_IP" --sport "$BACKEND_PORT" -m state --state ESTABLISHED,RELATED -j ACCEPT

# Allow robot subnet to reach public port (adjust if INPUT chain default is DROP)
iptables -C INPUT -p tcp -s "$ROBOT_CIDR" -d "$PUBLIC_IP" --dport "$PUBLIC_PORT" -j ACCEPT 2>/dev/null || \
  iptables -A INPUT -p tcp -s "$ROBOT_CIDR" -d "$PUBLIC_IP" --dport "$PUBLIC_PORT" -j ACCEPT

echo "[OK] DNAT ${PUBLIC_IP}:${PUBLIC_PORT} -> ${BACKEND_IP}:${BACKEND_PORT}"
echo "     robot allow: ${ROBOT_CIDR}"
iptables -t nat -L PREROUTING -n | grep "$PUBLIC_PORT" || true
