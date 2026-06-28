# G1 走廊 2GO · 实机 + AR 部署手册

> **重写于 2026-06-23,基于真机 bring-up 实测。** 机器人**起身 / 站立 / 行走已在硬件上验证通过**;走廊全程 + 收臂 + 动捕闭环 + AR 闭环为**待验证**(本文逐项标注)。这是从零到跑通的完整 runbook。
>
> 控制路线已确定并实测:**高层 sport(不是低层 lowcmd)** —— 腿走 `LocoClient.Move`(sport 内置 RL 步态),臂走 `rt/arm_sdk`,起身走 sport FSM `SetFsmId(4)→(200)`。参考 CMU SPARK `g1_real_agent`。

---

## 0. 总览:三台设备 + 一个原理

你的真实拓扑:
```
[iPhone]    Unity→Xcode AR app  ── WiFi ──► 连 Mac 孪生服务

[Mac]  算法/控制(~/g1 原生 env)
   └─ 网线 en7 ─► [交换机] ─┬─► [G1 机器人  192.168.123.161]
                            └─► [动捕主机 Motive  192.168.0.77]
                                  NatNet 多播,直接发,无 ROS
```
> Mac 一张 `en7`(设 /16)同时够到机器人(`.123.x`)和动捕(`.0.x`),两者在同一以太段。**动捕走有线 NatNet,不是 WiFi、不是 ROS。** 详细动捕接入 + 踩坑见 [`localization/mocap.md`](localization/mocap.md)。

**原理 —— 共享数字孪生:** 一份 `SceneSource`(唯一障碍源)+ 一个**动捕世界系** + 一个 `T_world_scene`。**AR 渲染的障碍 = governor 避的障碍 = 同一物理位置。** 机器人靠**自身动捕定位 + 共享障碍几何**"感知"虚拟障碍 —— 没有视觉感知栈,障碍是虚拟的。

**机器分工:**

| 机器 | 跑什么 |
|---|---|
| **Mac** | 控制(`run_real_g1`)+ 孪生服务(`run_twin_server`)+ **动捕直连解析(`natnet_shm_writer --natnet`)** + 规划(Docker,离线) |
| **动捕主机** | **只跑 Motive**(NatNet 多播直接发,无 ROS、无桥) |
| **iPhone** | Unity ARKit AR app(连 Mac 的孪生服务渲染) |
| **G1** | 网线连 Mac;sport 高层被 SDK 驱动 |

---

## 1. 一次性:Mac 原生环境 + 网络

### 1.1 为什么是 Mac 原生(不用 Docker / fedguide / 机载 PC)
真机控制要 `unitree_sdk2py` + `mujoco`(arm64)+ DDS 直连机器人。`fedguide` 是 Rosetta x86(跑不了 arm64 mujoco);Docker-on-Mac 跑不了到机器人的 DDS 多播;机载 PC 没外网 + py3.8。→ 用一个独立的 **Mac 原生 arm64 venv(`~/g1/venv`)+ 源码编译的 CycloneDDS**。

### 1.2 装(一次性)
**CycloneDDS**(macOS 上 `cyclonedds==0.10.2` 无轮子、无 brew formula,而 `unitree_sdk2py` 硬 pin 这个版本 → 必须源码编译):
```bash
git clone --depth 1 -b 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cdds
cmake -S /tmp/cdds -B /tmp/cdds/build -DCMAKE_INSTALL_PREFIX=$HOME/g1/cyclonedds -DBUILD_IDLC=ON -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/cdds/build --target install -j4    # 产出 libddsc + idlc
```
**venv + 依赖**(homebrew py3.12 原生 arm64):
```bash
/opt/homebrew/bin/python3 -m venv $HOME/g1/venv
CYCLONEDDS_HOME=$HOME/g1/cyclonedds $HOME/g1/venv/bin/pip install "cyclonedds==0.10.2"
CYCLONEDDS_HOME=$HOME/g1/cyclonedds $HOME/g1/venv/bin/pip install "git+https://github.com/unitreerobotics/unitree_sdk2_python.git"
$HOME/g1/venv/bin/pip install "mujoco" "numpy<2" scipy "jax[cpu]" "websockets>=12" "flatbuffers>=2.0"
```
**每次运行必带的环境变量**(建议做成 shell 别名):
```bash
export CYCLONEDDS_HOME=$HOME/g1/cyclonedds DYLD_LIBRARY_PATH=$HOME/g1/cyclonedds/lib
```

### 1.3 网络
- 网线 Mac↔G1。Mac 的 USB 网卡(本机是 **`en7`**)设静态 IP:
  ```bash
  sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.123.222 255.255.255.0   # 路由器留空
  ```
  验证:`ping 192.168.123.161`(机器人主控板)通。`--network-interface en7`(Mac 上是 en7,机载 PC 上才是 eth0)。
- **Mac 的 WiFi 接实验室网**(够到动捕主机 + iPhone)。

---

## 2. 已验证 vs 待验证(诚实清单)

✅ **已在硬件验证(2026-06-23):**
- Mac↔G1 DDS 通信:读 `rt/lowstate`(35 路电机,**0–28 为活动关节**,29–34 手部/保留;`mode_machine=5`)。
- **起身**:`GetFsmId → SetFsmId(4)[LockStand,站起锁定] → SetFsmId(200)[MainMode,可行走]`。
  - ⚠️ **pip SDK 的 `Start()`=SetFsmId(500)、`Squat2StandUp()`=706 是错的 id**,本固件拒(Invalid FSM ID 7302)。**必须用通用 `SetFsmId` + `4`/`200`**(SPARK 的 id)。
- **站立平衡**:MainMode 下机器人"活的",会自主小幅踏步保持平衡 —— **正常,不是失控**。
- **行走**:`Move(vx)` 迈步。**脚必须承重**(sport 步态接触门控,完全离地只会站不会走)。`vx=0.1` 太慢只原地碎步;`0.2+` 明显向前迈步。
- **动捕直连(2026-06-24,无机器人):** `Motive NatNet 多播 → natnet_shm_writer --natnet → mocap_state_shm → ViconShmPlugin` 端到端读出 G1 base 位姿(毫米级稳定)。**全程无 ROS。** 详见 [`localization/mocap.md`](localization/mocap.md)。

⏳ **待验证:**
- `run_real_g1` 全程走廊(起身 + governor + 走 + 收臂一起跑)。
- **arm_sdk 收臂**(第一次真机);若乱甩,调 `arm_sdk` 的 `motor_cmd.mode`(SPARK 留默认 0,官方 arm_sdk 例子设 1)。
- **动捕闭环**:机器人 + 动捕一起,`run_real_g1 --localization vicon`(修掉 mock 开环转身飘逸);含刚体 +x 朝向校验。
- AR 闭环(iPhone 注册 + 机器人绕虚拟障碍)。

**deploy 已覆盖的代码**(`genedynamics/deploy/io/unitree_g1_io.py`):`_bring_up_sport()`(按当前 FSM 走 `…→4→200`,已修为完整路径)、`_send_arm_sdk()`(`rt/arm_sdk` + `kNotUsedJoint=29` 权重位)、`LowCmd_` 默认工厂构造 + `mode_machine`。所以 `run_real_g1` 不用改,reset 自动起身、腿 Move、臂 arm_sdk。

---

## 3. 机器人运动验证阶梯(每次开机照走)

> **安全(实验室无 e-stop):所有运动在你自己终端跑(Ctrl-C 在手),机器人吊龙门 / 自承重,手放电源开关附近。** 运动指令用 1 秒自动过期的速度,脚本一停机器人 ~1s 内停。

环境 + 网络就绪(`ping .161` 通)后:

### 3.0 只读通信自检(无运动)
```bash
$HOME/g1/venv/bin/python - <<'PY'
import time
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
ChannelFactoryInitialize(0, "en7")
got={}; ChannelSubscriber("rt/lowstate", LowState_).Init(lambda m: got.__setitem__('m',m), 10)
t=time.time()
while time.time()-t<10 and 'm' not in got: time.sleep(0.2)   # 刚上电要等 DDS 服务起来(~数十秒)
m=got.get('m')
print("lowstate:", "OK" if m else "无(机器人还没启动完?)", "| mode_machine=", m.mode_machine if m else None)
from unitree_sdk2py.g1.loco.g1_loco_client import LocoClient
lc=LocoClient(); lc.SetTimeout(3.0); lc.Init(); print("current FSM:", lc.GetFsmId())
PY
```
> 刚上电时 `ping` 先通,但 `rt/lowstate` 的运动服务要几十秒才起 —— **收不到先等一会再试**。

### 3.1 起身(`~/g1/bringup_test.py`,每步等回车)
开机机器人在 **ZeroTorque(FSM 0,软)**。脚踩地有空间 / 吊龙门:
```bash
$HOME/g1/venv/bin/python $HOME/g1/bringup_test.py
```
它读当前 FSM,走 `0→1→4→200`(每步回车确认)。**`SetFsmId(4)` 那步机器人发力站起来。** 到 `FSM 200` = 成功。放软:`… bringup_test.py --damp`。
> ⚠️ 自承重站着时**别直接 `--damp`**(一软腿就塌);要放软先把它降到支撑/吊住。

### 3.2 小步走(`~/g1/move_test.py`)
**脚要承重**(接触门控)。已在 MainMode 则跳过起身,直接 Move:
```bash
$HOME/g1/venv/bin/python $HOME/g1/move_test.py --vx 0.2 --sec 3.0
```
看是否从"原地碎步"变"明显向前迈步"。还不迈 → `--vx 0.3`。

### 3.3 全程 `run_real_g1`(起身 + governor + 走 + 收臂)
```bash
$HOME/g1/venv/bin/python scripts/tasks/robot/humanoid/run_real_g1.py \
    --plan results/humanoid/corridor_2d/main/twogo_zone_a/level_1/seed_0/trajectory/trajectory.json \
    --preset zone_a --localization mock --network-interface en7 \
    --plan-speed 0.35 --max-steps 150
```
- `--localization mock` = 无动捕,固定起点位姿驱动 governor → 先验**软件全链**(起身 + governor + Move + arm_sdk)。
- reset 自动 `4→200` 起身(已在 MainMode 则 no-op)。
- **⚠️ 盯手臂第一次收拢:平稳收向收拢姿态 = 好;乱甩 = 立刻 Ctrl-C**(然后调 arm_sdk 的 mode)。

---

## 4. 动捕(NatNet)接入 —— 直连,无 ROS

> **2026-06-24 端到端实测通过。完整步骤 + 踩坑全记录见 [`localization/mocap.md`](localization/mocap.md);这里是要点。**

数据流:`Motive(NatNet 多播,Z-up,Rigid Bodies)→ Mac en7 → natnet_shm_writer --natnet → mocap_state_shm → --localization vicon`。**Motive 直接发 NatNet,Mac 直接解析,全程无 ROS、无 UDP 桥**(之前的 `mocap_udp_bridge.py` / `natnet_ros2` 路线对本套设置作废)。

### 4.1 网络(临时,重启重做)
```bash
sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.0.100 255.255.0.0  # /16 同时够到动捕+机器人
sudo route -n add -host 239.255.42.99 -interface en7                          # 多播路由掰到有线口
ping 192.168.0.77   # Motive 主机通
```
⚠️ **别把 Mac 设成 `.0.77`(撞 Motive 自己);多播路由这条必加**(macOS 默认走 WiFi,不加 en7 收不到包)。

### 4.2 Motive 侧(持久,一次配好)
Settings → Streaming:**Enable** / **Local Interface=`192.168.0.77`**(朝 Mac 那张网卡,关键) / **Multicast** / **Rigid Bodies=ON** / **Up Axis=Z**。在 Motive 里给 G1 建刚体(`+x = 机器人前向`,原点在骨盆),记下它的 **Streaming ID(本机 = `5`)**。

### 4.3 跑订阅 + 验证
```bash
# 写 shm（一直跑）
PYTHONPATH=<repo> $HOME/g1/venv/bin/python -m genedynamics.deploy.localization.natnet_shm_writer \
    --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5
# 验证（机器人真正读的那条路）
python -c "from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin; \
import time,numpy as np; p=ViconShmPlugin({}); time.sleep(0.3); print(np.round(p.get_state()[0],3))"
```
站着不动读到稳定位姿 = 闭环就绪。之后 `run_twin_server` / `run_real_g1` 用 `--localization vicon`。
> **无线?** Motive 改 `Unicast` + 加 `--unicast` 即可,**仅调试用;真机闭环用有线**(WiFi 抖动/丢包会劣化控制)。见 [`localization/mocap.md`](localization/mocap.md) §5。

### 4.4 `T_world_scene`(场景系 ↔ 动捕世界系)
**Motive 世界原点已固定(你已确认)—— 不去改它**,只把"场景系"和它对上。两条:
- **对齐法(最省事 → `TWS = "0 0 0"`,不用量):** 去 Motive 里看世界原点的位置和 +x 方向,**贴地面走廊时让场景原点 O 压在 Motive 原点上、中线 +x 对齐 Motive +x**。
- **两点法(场景不方便压在 Motive 原点时):** 放一个反光 marker 到 O 读 Motive 世界坐标 `(X0,Y0)`、放场景 (1,0) 读 `(X1,Y1)` → `x0=X0, y0=Y0, yaw0=atan2(Y1-Y0, X1-X0)`,传 `--t-world-scene x0 y0 yaw0`(**孪生和真机用同一个**)。
- 量点用一个 marker 读 Motive 坐标即可,**不是用校准杆(那是校相机的)**。

---

## 5. AR(iPhone Unity→Xcode)接入

### 5.1 让机器人"吃"AR 障碍
`run_twin_server` 渲染的 scene 和 `run_real_g1` 吃的 scene **必须同一份**(同一 `SceneSource`):
- **静态障碍**:两边传**同一个 `--preset`**(如 `zone_a`)。
- **改过 / 动态**:把孪生当前 contract 存成 JSON → `run_real_g1 --scene-file <ar_scene.json>`。

governor 经 `body_sdf_scene = corridor_scene_to_dict(scene)` 自动避(`run_real_g1` 已接好,见 [run_real_g1.py](../../scripts/tasks/robot/humanoid/run_real_g1.py) 第 77–81 行)。**机器人是解析感知:共享几何 + 自身动捕定位 → governor 收紧命令包络横向 steer。没有视觉栈。**

### 5.2 把 AR 标定到机器人起始位(详细)

> **先记住一句、别焦虑:AR 注册误差只影响"看着准不准",不影响机器人避不避。** 机器人靠**动捕定位 + 共享障碍列表**避障,跟 AR 对齐**无关**(见 [ar/clients/README.md](ar/clients/README.md))。所以注册"差不多对上"就够,不必追求完美。

**坐标链(因你确认 Motive 原点 = 场景原点而大幅简化):**
- Motive 世界原点 = 场景原点 O、+x 沿走廊 → **`T_world_scene` 恒等(`"0 0 0"`),场景坐标 = 世界坐标**。
- 障碍在 contract 里是**场景=世界坐标**;渲染时按世界→Unity 轴映射 **`(-y, z, x)`**(一次 RH→LH 翻转,都在 [ar/clients/unity/Scripts/FrameRegistration.cs](ar/clients/unity/Scripts/FrameRegistration.cs))。
- **AR 要做的只有一件:把 `WorldOriginAnchor` 放到"动捕世界原点(=场景原点 O)在 iPhone AR 会话里的位姿"**。障碍是它的子物体,自动跟着对齐。

iPhone(Unity→Xcode ARKit)有两条注册路 —— **A 是你现在的路、最快;B 最稳(动捕室推荐升级)。**

#### 方法 A — fiducial 图像注册(你现在的 Unity ARKit 路,最快)
原理:放一张**已知世界位姿**的打印图,ARKit 认出它 → 反推世界原点。**因 TWS 恒等,最干净的是把 fiducial 正好贴在场景原点 O。**
1. **打印 fiducial**:高对比、纹理丰富、**非对称**的图(当 ARKit Reference Image),A3 裱硬板,量出实际宽高(米)。
2. **贴在原点 O、压平、对齐场景轴**(图 +x 沿走廊 +x,图平面贴地)→ 这样 **fiducial 世界位姿 = 世界原点本身(`T_world_fiducial` 恒等)**。
   - O 处不便放图就放别处,但要知道它的世界位姿 —— 最省事:**给图板也贴 marker,在 Motive 里当刚体**,Motive 直接给 `T_world_fiducial`。
3. **Unity 工程**(AR Foundation + ARKit XR Plugin + Newtonsoft Json):
   - `AR Session` + `XR Origin (AR)`;加 `ARTrackedImageManager` + Reference Image Library(放这张图、填实际尺寸)。
   - 空物体 `CorridorTwin` 挂:`WorldClient`(`Url = ws://<Mac的WiFi_IP>:8766/`)、`FrameRegistration`(`WorldOriginAnchor` 拖一个空子物体)、`WorldRenderer`(`Client`/`Frame` 连好,occluder 材质)。
4. **写一个注册脚本**(仓库只有 OpenXR/Quest 的,iPhone 这条要你自己加,给骨架):在 `trackedImagesChanged` 里,认到图就把 `WorldOriginAnchor` 设成图的位姿:
   ```csharp
   // ARKitImageRegistration.cs —— 挂在带 ARTrackedImageManager 的物体上
   public FrameRegistration Frame;          // 它的 WorldOriginAnchor 我们来设
   public string FiducialName = "corridor_origin";
   void Apply(ARTrackedImage img) {
       if (img.referenceImage.name != FiducialName) return;
       if (img.trackingState != TrackingState.Tracking) return;
       // fiducial 贴在世界原点 O → 它在 AR 里的位姿 ≈ 世界原点的位姿
       Frame.WorldOriginAnchor.SetPositionAndRotation(img.transform.position, img.transform.rotation);
       // 若 fiducial 不在 O：anchor = imgPose * inverse(T_world_fiducial)
   }
   ```
   - ⚠️ **唯一要调的是"朝向约定"**:ARKit 给的图朝向,和我们 世界轴(x前/y左/z上)+ Unity 映射 `(-y,z,x)` 之间,可能差一个**固定旋转**(取决于图怎么贴、ARKit 图坐标约定)。**先按上面跑,看 AR 障碍是不是整体转了个角**;转了就在 rotation 上乘一个固定补偿(贴图时让图 +x 对齐走廊 +x 能把它降到最小)。
5. iPhone 和 Mac 同 WiFi → 开 app → **对准 fiducial 一下**完成注册 → 障碍锁地面。

#### 方法 B — iPhone 当动捕刚体(最稳,推荐升级 —— 真正"直接用世界原点")
不用 fiducial、不漂、连续校正:
1. iPhone 贴 3–4 个 marker,Motive 建刚体(如 `ar_phone`,本地原点=相机)→ NatNet 直接发它世界位姿(同 §4 直连方式读取)。
2. 把这个位姿喂给孪生服务的 `populate_headset`([ar/producers/tracker_pose.py](ar/producers/tracker_pose.py))→ 它作为 `viewer` 实体(`headset/base`)随同一条流推给客户端。
3. Unity 端用现成的 [ar/clients/unity/Scripts/OpenXRAnchorProvider.cs](ar/clients/unity/Scripts/OpenXRAnchorProvider.cs) —— 它**厂商无关、只依赖一个 Camera + FrameRegistration,iPhone 的 AR 相机也能用**:`HeadPoseSource` = AR 相机,`Frame` = FrameRegistration。它每帧解 `anchor = camera_unity ∘ L⁻¹` 并 EMA 平滑 → **连续把全息钉到动捕真值,自动校正 ARKit 漂移**。
4. `WorldRenderer` 的 `AnchorProvider` 连它、`LocalHeadsetId = headset/base`(自己那台不画自己的代理)。
- 代价:多一步把 iPhone 动捕位姿喂进来;好处:长时间不漂、亚毫米,比一次性 fiducial 稳。

#### 验证对齐
- 站到起点绿十字(场景 (0.5,0)):AR 的起点标记 / 中线墙应和地面胶带重合,**偏差 < 几 cm**。
- 沿中线走,AR 中线跟着;AR 墙立在 x=1.7–2.0。
- 偏大 → fiducial 贴歪 / 尺寸填错 / 朝向补偿没加(方法 A),或 iPhone 刚体没喂进来(方法 B)。
- 再说一遍:**注册误差只是视觉的,机器人避障不受它影响** —— 演示对齐差几 cm 完全能跑。

### 5.3 撞上 AR 障碍会摔么? —— **不会因虚拟障碍物理摔**
- 障碍是虚拟的,**无物理接触**,撞不到全息影像、绊不倒。
- governor **提前 steer 避开**(实测:盲跟冲进 −0.267m,governor 感知后避到 +0.004m)。
- 万一避不够 → 机器人**穿过**虚拟障碍(仍不摔),**唯一后果是日志 body-SDF 违例**(`certified_safe=False`),是软 / 逻辑失败。
- 真摔风险来自**步态不稳 / sim-to-real**,与虚拟障碍无关。
- ⚠️ tight 场景(zone_d/b)执行间隙上限 ~**+0.08m**(`m_track≈0.08`);**首次选最宽的 zone_a**。
- **这正是 AR 的价值:安全测避障 —— 失败只是日志,不撞坏机器人。**

---

## 6. 动捕闭环联调(从 mock 台阶到 vicon)

> **跑完整走廊前,先单独验证"动捕 → 闭环"这一跳。** `mock` 是**开环**(固定起点位姿,转身后机器人飘出走廊);`vicon` 是**闭环**(真实位姿)。但闭环还有**另一个飘逸源:刚体朝向**——本节核心就是把它校掉。

### 6.1 前置(网络 + 两进程就绪)
- en7 设 **/16**:`sudo networksetup -setmanual "USB 10/100/1000 LAN" 192.168.0.100 255.255.0.0` → `ping 192.168.0.77`(动捕)和 `ping 192.168.123.161`(机器人)都通;多播路由加好(§4.1)。
- **终端 A**:`natnet_shm_writer --natnet …`(一直跑,写 shm)。
- **终端 B**:`ViconShmPlugin({}).get_state()` 能读到稳定位姿(§4.3)。

### 6.2 刚体 +x 朝向校验(关键,首次必做)
**为什么:** 刚体本体 **+x 必须 = 机器人前进方向**。否则流里的 `yaw` 与真实朝向差一个常量 α → governor 把"机体系速度命令"整体转了 α → 机器人走斜 / 飘(这是 mock 开环之外的**第二个飘逸源**)。

**怎么测(走一小段,比对"位移朝向"和"刚体 yaw"):** 存成 `~/g1/heading_check.py`,`PYTHONPATH=<repo>` 跑:
```python
import time, math, numpy as np
from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin

def yaw_deg(pose):                       # pose=[x,y,z, qw,qx,qy,qz]（z-up）
    w,x,y,z = pose[3],pose[4],pose[5],pose[6]
    return math.degrees(math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))

p = ViconShmPlugin({})
def read():
    pose,_ = p.get_state(); return np.array(pose[:2]), yaw_deg(pose)

t0=time.time(); P=[]; Y=[]                # 静态 2s：起点 + yaw
while time.time()-t0<2.0:
    xy,yw=read(); P.append(xy); Y.append(yw); time.sleep(0.05)
p0=np.mean(P,axis=0); yaw0=float(np.mean(Y))
print(f"[静态] 位置=({p0[0]:+.3f},{p0[1]:+.3f})  yaw_world={yaw0:+.1f}°")
input("→ 让机器人【朝走廊 +x】向前走 ~0.5m（另开终端: move_test.py --vx 0.2 --sec 3），走完按回车…")

far=p0.copy(); d=0.0; t0=time.time()      # 取离起点最远的一帧
while time.time()-t0<1.0:
    xy,_=read()
    if np.linalg.norm(xy-p0)>d: d=np.linalg.norm(xy-p0); far=xy
    time.sleep(0.05)
disp=far-p0; dist=float(np.linalg.norm(disp))
if dist<0.15:
    print(f"⚠️ 只走了 {dist:.2f}m，太短不可靠，重来（走够 0.3m+）"); raise SystemExit
heading=math.degrees(math.atan2(disp[1],disp[0]))
delta=(heading-yaw0+180)%360-180
print(f"[行走] 位移=({disp[0]:+.3f},{disp[1]:+.3f}) |{dist:.2f}m|  位移朝向={heading:+.1f}°")
print(f"刚体 +x 与机器人前向偏差 Δ = {delta:+.1f}°  →  " +
      ("✅ 对齐(|Δ|<10°)" if abs(delta)<10 else "⚠️ 不对齐：Motive 转正刚体朝向，或把 Δ 折进 --t-world-scene 的 yaw"))
print(f"若机器人此刻就朝场景 +x 站：--t-world-scene 的 yaw 设为 {math.radians(yaw0):+.3f} rad（={yaw0:+.1f}°）")
```
**判读:** `|Δ|<~10°` → 对齐,OK;`Δ≈常量` → 去 Motive 把刚体朝向转正(最好),或把 Δ 折进 `--t-world-scene` 的 yaw;`Δ≈±90/180°` → 刚体建歪,重建朝向。脚本最后一行给的 `yaw0` 是 §6.3 注册的输入。

### 6.3 注册 `T_world_scene`(把机器人摆到场景起点)
机器人**物理摆在场景起点**(zone_a:`(0.5,0)`)、**朝场景 +x**,让它的世界位姿映射到 scene `(0.5,0,0)`:
- **(a) Motive 原点=场景原点**(你已定)→ 物理把机器人摆到对应世界点,`--t-world-scene 0 0 0`。
- **(b) 机器人就地不便挪** → 用读到的世界位姿 `(Xr,Yr,yaw_r)` 反算:`yaw0=yaw_r`;`(x0,y0)=(Xr,Yr) − R(yaw0)·(0.5,0)`,传 `--t-world-scene x0 y0 yaw0`。
- 公式 / 两点法细节见 §4.4。

### 6.4 live frame-glue 核对(不动电机,读真实动捕)
用 `run_real_g1` 同款 `SceneFrameLocalization` 读**实时**动捕,确认机器人在场景里的位姿对:
```python
import time, math
from genedynamics.deploy.localization.vicon_shm_plugin import ViconShmPlugin
from genedynamics.deploy.localization.scene_frame_plugin import SceneFrameLocalization
TWS={"x":0.0,"y":0.0,"yaw":0.0}          # 改成你的 --t-world-scene
loc=SceneFrameLocalization(ViconShmPlugin({}), TWS)
for _ in range(6):
    pose,_=loc.get_state(); w,x,y,z=pose[3],pose[4],pose[5],pose[6]
    yaw=math.degrees(math.atan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    print(f"scene pose=({pose[0]:+.3f},{pose[1]:+.3f})  yaw_scene={yaw:+.1f}°")
    time.sleep(0.3)
```
机器人摆在场景起点朝 +x 时,应读到 **≈ `(0.5, 0)`、`yaw_scene≈0°`**。对不上 → TWS 没设对 / 机器人没摆正 / 刚体朝向没校(回 §6.2)。
> `run_real_g1 --dry-run` 是用**固定起点**验 frame-glue **数学**(不读 shm);本脚本是验**实时动捕→场景**的真实映射,两者互补。

### 6.5 首次闭环跑(最小,无障碍)
确认 §6.2/§6.4 都过后,跑最小闭环(先别上 AR/障碍):
```bash
# 终端A：writer 一直跑（§4.3）
# 终端B：
$HOME/g1/venv/bin/python scripts/tasks/robot/humanoid/run_real_g1.py \
    --plan results/humanoid/corridor_2d/main/twogo_zone_a/level_1/seed_0/trajectory/trajectory.json \
    --preset zone_a --localization vicon --t-world-scene 0 0 0 \
    --network-interface en7 --plan-speed 0.3 --max-steps 150
```
- **安全**:龙门 / 手扶,Ctrl-C 在手,手放电源(同 §3 红线)。
- **看什么**:不再像 mock 那样转身飘出走廊 —— 机器人按**真实动捕位姿**闭环跟踪;`Pelvis z range` 是真实高度(不再恒定 0.793)。
- 顺了再进 §7 完整 zone_a(加 AR + 中线墙)。

---

## 7. zone_a 端到端(动捕室全流程)

**zone_a 几何:** 走廊长 **4.0m**、宽 **1.6m**(墙 y=±0.8);起点**场景 (0.5,0)**、终点 (3.5,0);**一道中线薄墙** x∈[1.7,2.0]、y≈0(机器人要侧绕)。

> **首次先过 §6 联调**(刚体 +x 朝向校验 + 最小闭环跑)——本节是在此基础上加 AR + 中线墙的完整流程。

1. **贴地面场景系**:红十字定 O(0,0);粉线/激光拉 +x 中线 4m,每 0.5m 标刻度;**起点绿十字 x=0.5 + 朝 +x 箭头**;终点蓝十字 x=3.5;**障碍位置**标 x=1.7–2.0(不放实物);两壁 y=±0.8 参考线。
2. **`TWS`**:Motive 原点已固定 → 贴地面时让场景 O 对齐它(`TWS="0 0 0"`),或两点法量(§4.4)。
3. **把 G1 放进捕捉区、启动 Motive** → NatNet 多播自动推刚体 pose(记下 streaming ID,本机=`5`)。
4. **动捕订阅(Mac)**:`natnet_shm_writer --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5`(详见 §4 / [`localization/mocap.md`](localization/mocap.md))。
5. **孪生**:Mac `run_twin_server.py --preset zone_a --t-world-scene <TWS> --localization vicon`。
6. **iPhone**:连 `ws://<Mac>:8766/` + fiducial 注册 → 看到全息走廊 + 中线墙锁在地面 x=1.85 处。
7. **干跑核 frame glue**(不动电机):`run_real_g1.py --plan <…twogo_zone_a…trajectory.json> --preset zone_a --t-world-scene <TWS> --localization vicon --dry-run` → 应打印 `world start → scene pose (0.500, 0.000)`。
8. **落地跑**:机器人踩绿十字朝 +x,去掉 `--dry-run`(留 `--plan-speed 0.35 --max-steps` 先短):真机**侧绕中线墙**走,AR 同步显示绕开,occluder 让真机正确遮挡墙。

> 计划来源:`results/humanoid/corridor_2d/main/twogo_zone_a/level_1/seed_*/trajectory/trajectory.json`(已有)。重规划 / 标定到 AR 场景:Docker 跑 `replan_from_scene.py` / `governor_calibrate.py`(`m_track` 实测最优 ≈0.08;未标定时 run_real_g1 回退 0.08)。

---

## 8. 安全 / 故障排查 / 速查

### 安全红线
- **无 e-stop**:运动你自己终端跑、Ctrl-C 在手、手放电源;首次吊龙门 / 自承重 + 防摔。
- 自承重站着时**别突然 Damp**(会塌);先支撑。
- 速度只升不跳;先验最易 zone_a;tight 场景(zone_d/b)留足物理裕度。
- arm_sdk 第一次跑盯手臂,乱甩立刻 Ctrl-C。

### 故障排查
| 现象 | 排查 |
|---|---|
| `ping .161` 不通 | 网线 / en7 静态 IP(`192.168.123.222/24`) |
| `ping` 通但读不到 `rt/lowstate` | 机器人还没启动完,等几十秒再试 |
| `SetFsmId(4)/(200)` 被拒 | 确认用的是 **4/200**(不是 pip 的 500/706);或遥控器/App 先到 LockStand |
| 站起来但 Move 不迈步 | **脚没承重**(接触门控)/ vx 太小(提到 0.2–0.3) |
| `--localization mock` 跑 run_real_g1 第 1 步就 `Fell over` | mock 无真高度、报 z=0 → 摔倒检测误判;**已修**(mock 报 0.793 站立高度)。真机用 `vicon` 是真高度。verdict 里 "SparkRL policy diverged" 是 sim 文案,真机忽略 |
| 手臂乱甩 | arm_sdk 的 `motor_cmd.mode`(SPARK 0 / 官方例子 1)—— 改 `_send_arm_sdk` |
| `--localization vicon` 读不到 | `natnet_shm_writer --natnet` 没起 / Motive **Local Interface 选错** / **Rigid Bodies 没开** / 多播路由没加(`route add 239.255.42.99 en7`)→ 按 [`localization/mocap.md`](localization/mocap.md) §2 分层探针 |
| AR 障碍漂移 / 不对齐 | `T_world_scene` 或 fiducial 世界位姿量不准 |
| 干跑 pose 不是 (0.5,0) | `T_world_scene` 错 / 机器人没摆在起点 / 刚体方向错 |

### 速查
| 项 | 值 |
|---|---|
| Mac 原生 env | `~/g1/venv`;`CYCLONEDDS_HOME=$HOME/g1/cyclonedds DYLD_LIBRARY_PATH=$HOME/g1/cyclonedds/lib` |
| 网卡 | Mac `en7`(192.168.123.222);机器人 `.161` |
| 起身 FSM | `GetFsmId → SetFsmId(4) → SetFsmId(200)`(LockStand → MainMode) |
| 起身/走 脚本 | `~/g1/bringup_test.py`、`~/g1/move_test.py` |
| 动捕订阅 | `natnet_shm_writer --natnet --local-ip 192.168.0.100 --server 192.168.0.77 --rigid-body-id 5`(详见 [`localization/mocap.md`](localization/mocap.md))|
| 真机入口 | `scripts/tasks/robot/humanoid/run_real_g1.py`(`--dry-run` 先验) |
| 孪生服务 | `scripts/tasks/robot/humanoid/run_twin_server.py` → `ws://<Mac>:8766/` |
| 真机 IO | `genedynamics/deploy/io/unitree_g1_io.py`(`_bring_up_sport` / `_send_arm_sdk`) |
| 计划 | `results/humanoid/corridor_2d/main/twogo_zone_a/.../trajectory.json` |
| 参考 | CMU SPARK `g1_real_agent`(`enerdynamics/spark/`);Unitree G1 SDK 文档 |
