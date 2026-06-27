#!/bin/bash
# macOS host network setup for the corridor deploy: auto-detect which USB-Ethernet
# adapter goes to the mocap host (Motive) and which goes to the G1, configure both,
# and add the NatNet multicast route on the mocap NIC.
#
# Why a script: the two USB adapters share the hardware-port name "USB 10/100/1000
# LAN", so `networksetup` can't tell them apart, and macOS routes multicast to the
# default (Wi-Fi) interface unless a host route pins it to the wired NIC. This probes
# each adapter on each subnet and pins everything correctly regardless of which cable
# is in which port.
#
#   sudo bash genedynamics/deploy/localization/setup_host_net_macos.sh
#
# Re-run after any reboot / cable reseat (ifconfig + route changes are not persistent).

set -u

MOCAP_HOST=192.168.0.77;     MOCAP_IP=192.168.0.100      # Motive PC / this Mac on its subnet
ROBOT_HOST=192.168.123.161;  ROBOT_IP=192.168.123.222    # G1 onboard PC / this Mac on its subnet
MASK=255.255.255.0
MCAST=239.255.42.99                                       # NatNet multicast group
CANDS="en7 en8 en3 en4 en5 en6"                           # USB/Ethernet adapters to probe

if [ "$(id -u)" != "0" ]; then echo "请用 sudo 运行: sudo bash $0"; exit 1; fi
ping_ok(){ ping -c1 -t1 -S "$1" "$2" >/dev/null 2>&1; }
clear_if(){ ifconfig "$1" inet "$MOCAP_IP" delete 2>/dev/null; ifconfig "$1" inet "$ROBOT_IP" delete 2>/dev/null; }

echo "== 探测网卡 ↔ 设备映射 =="
MOCAP_IF=""; ROBOT_IF=""
for IF in $CANDS; do
  ifconfig "$IF" >/dev/null 2>&1 || continue
  ifconfig "$IF" | grep -q "status: active" || { echo "  $IF 无连线，跳过"; continue; }
  role=""
  if [ -z "$MOCAP_IF" ]; then
    ifconfig "$IF" inet "$MOCAP_IP" netmask "$MASK" up; sleep 1
    if ping_ok "$MOCAP_IP" "$MOCAP_HOST"; then MOCAP_IF=$IF; role=mocap; echo "  ✅ 动捕  在 $IF (ping $MOCAP_HOST 通)"; fi
  fi
  if [ -z "$role" ] && [ -z "$ROBOT_IF" ]; then
    ifconfig "$IF" inet "$MOCAP_IP" delete 2>/dev/null
    ifconfig "$IF" inet "$ROBOT_IP" netmask "$MASK" up; sleep 1
    if ping_ok "$ROBOT_IP" "$ROBOT_HOST"; then ROBOT_IF=$IF; role=robot; echo "  ✅ 机器人 在 $IF (ping $ROBOT_HOST 通)"; fi
  fi
  [ -z "$role" ] && { clear_if "$IF"; echo "  ❓ $IF 两个网段都没应答"; }
done

echo
echo "== 落实配置 =="
if [ -n "$MOCAP_IF" ]; then
  ifconfig "$MOCAP_IF" inet "$MOCAP_IP" netmask "$MASK" up
  route -n delete -host "$MCAST" >/dev/null 2>&1
  route -n add -host "$MCAST" -interface "$MOCAP_IF" >/dev/null 2>&1
  echo "  动捕:   $MOCAP_IF = $MOCAP_IP/24   (+多播 $MCAST → $MOCAP_IF)"
else
  echo "  ❌ 没找到通往动捕 $MOCAP_HOST 的网卡 —— 查：Motive 机开没开 / 网线插没插 / Motive 没崩"
fi
if [ -n "$ROBOT_IF" ]; then
  ifconfig "$ROBOT_IF" inet "$ROBOT_IP" netmask "$MASK" up
  echo "  机器人: $ROBOT_IF = $ROBOT_IP/24"
else
  echo "  ❌ 没找到通往机器人 $ROBOT_HOST 的网卡 —— 查：G1 开没开 / 网线插没插"
fi

echo
echo "== 最终验证 =="
[ -n "$MOCAP_IF" ] && { ping_ok "$MOCAP_IP" "$MOCAP_HOST" && echo "  动捕   $MOCAP_HOST ✅" || echo "  动捕   $MOCAP_HOST ❌"; }
[ -n "$ROBOT_IF" ] && { ping_ok "$ROBOT_IP" "$ROBOT_HOST" && echo "  机器人 $ROBOT_HOST ✅" || echo "  机器人 $ROBOT_HOST ❌"; }

echo
echo "== 跑的时候用这些参数 =="
[ -n "$MOCAP_IF" ] && echo "  动捕订阅: --natnet --local-ip $MOCAP_IP --server $MOCAP_HOST --rigid-body-id 5"
[ -n "$ROBOT_IF" ] && echo "  机器人:   --network-interface $ROBOT_IF"
