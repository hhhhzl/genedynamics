# 动捕(OptiTrack/Motive · NatNet)→ Mac 接入手册 + 踩坑全记录

> **2026-06-24 端到端实测打通**(无机器人):`Motive(NatNet 多播)→ Mac en7 → natnet_shm_writer --natnet → mocap_state_shm → ViconShmPlugin`,读出的 G1 base 位姿与 Motive 里完全一致。
>
> 这份文档专门讲**怎么从一台 Mac 连到动捕、怎么测、踩过哪些坑**。实机走廊全流程见 [`../REAL_DEPLOY_G1_CN.md`](../REAL_DEPLOY_G1_CN.md);这里是它 §4 的详细底稿。

---

## 0. 最终架构(一张图 + 三个关键结论)

```
[Motive 动捕主机 192.168.0.77]
   Streaming: Enable / Local Interface=192.168.0.77 / Multicast 239.255.42.99
              data:1511 cmd:1510 / Rigid Bodies=ON / Up Axis=Z / 米
        │  NatNet 多播(UDP),直接发,没有 ROS
        ▼  以太网(交换机/网线)
[Mac  en7  192.168.0.100]
   natnet_shm_writer.py --natnet      ← 直连解析,无 ROS
        │  写
        ▼
   mocap_state_shm  (q13d: utime + pos3 + quat_xyzw4 + vel3 + omega3)
        │  读
        ▼
   ViconShmPlugin   == run_real_g1.py / run_twin_server.py 的 --localization vicon
```

**三个关键结论(整份文档的精华):**
1. **没有 ROS。** 之前以为要 `natnet_ros2` + UDP 桥 —— 全错。Motive 直接发 NatNet,Mac 直接解析。`mocap_udp_bridge.py` 那条路对本套设置作废。
2. **机器人用的是 NatNet 流,不是 Motive 屏幕显示。** 屏幕显示是 Y-up 给人看的;流是 Z-up 给机器人用的,两者坐标可能差一个轴向约定(见 §6)。**标定/验证一律以流为准。**
3. **网络的坑全在 Mac 和 Motive 两侧的"接口选择",不在代码。** 一旦接口选对,解析是稳的(毫米级)。

---

## 1. 一步步从零连通(happy path)

> 这是把所有坑都避开后的"干净流程"。每次重启 Mac / 重插网线,**第 2、3 步会失效要重做**(临时设置);Motive 侧(第 4、5 步)持久存在配置里。

### 第 1 步 · 物理连线
Mac 的 USB 有线网卡(本机是 **`en7`**)→ 交换机 →(同一以太段)Motive 主机 +(可选)G1 机器人。
- 只测动捕:Mac ←→ Motive 一根线即可。
- 闭环跑机器人:Mac、Motive(`192.168.0.77`)、G1(`192.168.123.161`)在**同一以太段**(一个交换机),靠 Mac 一个 `en7` 同时够到两边(见第 2 步用 /16)。

### 第 2 步 · Mac 网卡设静态 IP(临时,重启失效)
```bash
# 只测动捕（/24 够到 192.168.0.x）：
sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.0.100 255.255.255.0

# 闭环跑机器人（/16 同时够到 .0.x 动捕 和 .123.x 机器人）：
sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.0.100 255.255.0.0
```
- ⚠️ **绝不要把 Mac 设成 `192.168.0.77`** —— 那是 Motive 自己的 IP,撞 IP(我们踩过)。Mac 用 `.100` 之类的空闲地址。
- 验证:`ping 192.168.0.77` 通(Motive 主机);闭环时 `ping 192.168.123.161` 也要通(机器人)。

### 第 3 步 · 把多播路由掰到有线口(临时,重启失效)
```bash
sudo route -n add -host 239.255.42.99 -interface en7
```
- **为什么必须做**:macOS 默认把多播(`239.x`)路由到 `en0`(WiFi)。不加这条,即使网卡 IP 对、ping 通,**en7 上也收不到任何 NatNet 包**(我们卡这里很久)。
- 我们的 `--natnet` 代码还会用 `local_ip` 主动 `IP_ADD_MEMBERSHIP` 在 en7 上加入组 —— 双保险,但路由这条仍建议加。

### 第 4 步 · Motive 侧设置(持久,一次配好)
Motive → Settings → **Streaming**:
| 项 | 值 | 说明 |
|---|---|---|
| **Enable** | ON | 开 NatNet |
| **Local Interface** | **`192.168.0.77`** | ⚠️ **必须选朝向 Mac 那张网卡的 IP** —— 这是我们的**最后一把钥匙**:之前选错接口,Mac 一个包都收不到(只有空帧) |
| **Transmission Type** | **Multicast** | 有线推荐多播;无线建议改 Unicast(见 §5) |
| **Rigid Bodies** | **ON** | ⚠️ 关着的话流里没有刚体(全空帧)。我们只需要这个,其它 Markers/Skeletons 都可关 |
| **Up Axis** | **Z-Axis** | 和我们世界系(z 朝上、米)一致,**不用做轴变换** |
| Multicast / 端口 | `239.255.42.99` / data `1511` cmd `1510` | NatNet 默认值,和代码默认一致 |

### 第 5 步 · 在 Motive 里建/确认 G1 刚体 + 记下 streaming ID
1. G1 放进捕捉区,Motive 实时视图能看到它身上的 marker。
2. 选中这些 marker → 右键 **Create Rigid Body**(命名如 `g1_base`)。
3. **本体 +x 轴对齐机器人前进方向、原点设在骨盆中心**(否则航向有常量偏置,见 §6)。
4. 记下它的 **Streaming ID**(刚体属性里;本机 G1 = **`5`**)—— 写进 `--rigid-body-id`。

### 第 6 步 · 跑订阅(写 shm)
```bash
REPO=/Users/zhilinhe/Desktop/hhhhzl/EduGetRicher/CMU/projects/enerdynamics
PYTHONPATH=$REPO $HOME/g1/venv/bin/python -m genedynamics.deploy.localization.natnet_shm_writer \
    --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5
```
- `--local-ip` = Mac 在动捕段的 IP(用于在正确网卡上加入多播组,**必填**)。
- `--rigid-body-id` 不填 = 自动取第一个被跟踪的刚体(只有 G1 在场时也行)。
- Motive 是 Unicast 就加 `--unicast`。

### 第 7 步 · 验证(用机器人真正读的那条路)
```bash
PYTHONPATH=$REPO $HOME/g1/venv/bin/python - <<'PY'
import time, numpy as np
from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin
p=None
for _ in range(60):
    try: p=ViconShmPlugin({}); break
    except Exception: time.sleep(0.1)
for _ in range(5):
    pose,twist=p.get_state()
    print("pose=",np.round(pose,3)," twist=",np.round(twist,3))
    time.sleep(0.25)
PY
```
站着不动应读到稳定的 `pose=[x y z | qw qx qy qz]`、`twist≈0`。**到这步就闭环就绪**,`run_real_g1.py --localization vicon` 直接吃这块 shm。

---

## 2. 怎么"探针"调试(连不上时按这个顺序查)

每一层独立验证,定位最快:

```bash
# (a) 网络层：能不能 ping 通 Motive 主机
ping -c3 192.168.0.77

# (b) 收包层：en7 上到底有没有 NatNet 多播包（不解析，只数包）
$HOME/g1/venv/bin/python - <<'PY'
import socket,struct,time
s=socket.socket(socket.AF_INET,socket.SOCK_DGRAM); s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
s.bind(("",1511)); s.setsockopt(socket.IPPROTO_IP,socket.IP_ADD_MEMBERSHIP,
    struct.pack("4s4s",socket.inet_aton("239.255.42.99"),socket.inet_aton("192.168.0.100")))
s.settimeout(3); n=0; t=time.time()
while time.time()-t<3:
    try: d,a=s.recvfrom(65536); n+=1
    except socket.timeout: break
print(f"3秒收到 {n} 个包" + ("" if n else "  ← 0包：查 Motive Enable/Local Interface 或第3步路由"))
PY

# (c) 解析层：能不能解出刚体（用 writer 自己跑，看它打印 tracking rigid body id=…）
```
- (a) 不通 → 第 1/2 步(线、IP、掩码、撞 IP)。
- (a) 通但 (b) 收 0 包 → 第 3 步多播路由 / 第 4 步 Motive **Local Interface** 选错。
- (b) 有包但 (c) 刚体数=0 → 第 4 步 **Rigid Bodies=OFF** / 第 5 步刚体没建 / 机器人不在捕捉区。

---

## 3. NatNet 4.5 帧格式要点(为什么一开始解析全是 0)

Motive(NatNet 4.5)的 `FrameOfData`(msgID=7)里,**每个顶层段是 `[count int32][byteSize int32][payload]`**(3.0 起新增的 size 前缀,让客户端能跳过不懂的段)。我们一开始按老版固定偏移读,把 size 前缀当成了计数,于是 markersets/rigidbodies 全读成 0。正确解析:

- 帧号(int32)→ markersets 段(count+size,按 size 跳)→ legacy/other markers 段(count+size,跳)→ **rigid bodies 段(count+size,逐个解)**。
- 每个刚体 **38 字节**:`id(i32) x,y,z(3f) qx,qy,qz,qw(4f) meanError(f) flags(i16)`,`flags` bit0 = tracking valid。
- 实现见 [`natnet_shm_writer.py`](natnet_shm_writer.py) 的 `parse_frame_of_data` / `NatNetDirectReceiver`。

---

## 4. 我们踩过的坑(按发生顺序)

| # | 症状 | 根因 | 修复 |
|---|---|---|---|
| 1 | Mac 上没有 `192.168.0.x` 的接口,`ping 192.168.0.77` 100% 丢 | Mac 没配动捕段的 IP | 第 2 步给 en7 设 `192.168.0.100` |
| 2 | 设了 IP 还是不对 / 冲突 | 一度把 Mac 设成 `192.168.0.77`(= Motive 自己) | Mac 用空闲地址 `.100`,绝不用 `.77` |
| 3 | IP 对、ping 通,但 en7 上**一个 NatNet 包都收不到** | macOS 默认把多播路由到 en0(WiFi) | 第 3 步 `route add -host 239.255.42.99 -interface en7` |
| 4 | 路由也加了,**还是收不到包** | Motive 的 **Local Interface 选错网卡** | 第 4 步把 Local Interface 改成 `192.168.0.77`(朝 Mac 那张) ← **最后一把钥匙** |
| 5 | 终于收到包,但解析出 **markerset=0 / 刚体=0**(全空帧) | Streaming 里 **Rigid Bodies=OFF**(且没建刚体) | 第 4 步开 Rigid Bodies + 第 5 步建 G1 刚体 |
| 6 | 收到刚体段但**按固定偏移解出来全是 0** | NatNet 4.5 段头多了 `byteSize` 前缀(见 §3) | 改成"按段 count+size 跳"解析 |
| 7 | 多个刚体时不知道哪个是 G1 | FrameOfData 只有 ID 没有名字 | 第 5 步在 Motive 记下 G1 的 streaming ID(=5),用 `--rigid-body-id` 锁定 |

---

## 5. 无线(WiFi)接动捕可行吗?

**可行,但要分场景 —— 调试可以无线,真机闭环用有线。**

- **只看位置 / 调试(低频、不进控制环):** WiFi 没问题。建议:
  - Motive **Transmission Type 改 `Unicast`**(WiFi 上多播常被 AP 丢包/限速,Unicast 更稳),Mac 用 `--unicast`。
  - Mac 和 Motive 在**同一 WiFi 子网**(或可路由);`--server` 填 Motive 的 WiFi IP,`--local-ip` 填 Mac 的 WiFi IP。
  - **代码不用改**,`--natnet --unicast` 已支持。
- **真机闭环(120Hz 定位喂进平衡/行走环):** **不推荐 WiFi。** WiFi 的抖动 / 丢包 / 延迟会直接劣化控制(机器人可能走斜、晃)。有线低延迟稳定,真机跑用有线。
- **关于"SSH":** SSH 只是远程登录,**不会**用来传位姿流。同事说的"WiFi 连"应理解为:Mac 与 Motive 在同一 WiFi 网,NatNet 走 WiFi 传(Unicast)。要 SSH 隧道转发 1511/1510 端口也能通,但徒增延迟,没必要。
- **一句话:** 平时工位调试用 WiFi+Unicast 很方便;**正式跑机器人用有线**。

---

## 6. 坐标系:Motive 显示 vs NatNet 流(很重要)

- **Motive 屏幕显示**通常是 **Y-up**(给人看);**NatNet 流**因我们设了 **Up Axis=Z**,是 **Z-up**(给机器人用)。两者差一个轴向约定(Y-up→Z-up 约为 `(X,Y,Z)→(X,−Z,Y)`),所以**屏幕上看到的 (x,y) 和流里的 (x,y) 可能某个轴反号**。
- 例:流里读到 `pos=(0.184, 0.991, 0.791)`,Motive 屏幕上看像 `(-0.2, -1)` —— **量级一致(≈0.2、≈1.0)就是同一个点**,符号差是上面的约定差。
- **机器人只用流。** 所以:
  - 标定 `T_world_scene`、判断机器人朝向,**一律以流里的 (x,y,yaw) 为准**,别用屏幕读数。
  - **刚体本体 +x 必须 = 机器人前进方向**(第 5 步)。否则流里的 yaw 有个常量偏置,机器人会走斜 —— 要么在 Motive 里重置刚体朝向,要么把偏置量折进 `--t-world-scene` 的 yaw。
  - 上线前做一次**走向验证**:让机器人朝物理 +x 走一小段,看流里的 x 是不是在增大、y 基本不变;对不上就调刚体朝向 / TWS。

---

## 7. 速查

| 项 | 值 / 命令 |
|---|---|
| Motive 主机 | `192.168.0.77`,NatNet 多播 `239.255.42.99` data`1511` cmd`1510` |
| Mac 网卡 | `en7`;测动捕 `192.168.0.100/24`,闭环 `192.168.0.100/16`(够到 `.123.161` 机器人) |
| 多播路由 | `sudo route -n add -host 239.255.42.99 -interface en7`(每次重启重加) |
| Motive 必备 | Enable / Local Interface=`192.168.0.77` / Multicast / **Rigid Bodies=ON** / Up Axis=Z |
| G1 刚体 streaming ID | **5** |
| 起订阅 | `python -m genedynamics.deploy.localization.natnet_shm_writer --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5` |
| 无线 | Motive 改 `Unicast` + 加 `--unicast`;仅调试用,真机闭环用有线 |
| 验证读取 | `ViconShmPlugin({}).get_state()`(== `--localization vicon`) |
| 代码 | `natnet_shm_writer.py`(`parse_frame_of_data`/`NatNetDirectReceiver`/`run_natnet`) |
