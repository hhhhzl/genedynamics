# G1 走廊 2GO 上实机操作步骤（M0–M5）

把 humanoid corridor 的 2GO 规划轨迹部署到**真 Unitree G1**的实战 runbook。
分 M0–M5 六个阶段,**每阶段有明确的进入条件、操作命令、验收标准、回退方式**。
原则:**先在 sim 把一切验通,再换 IO 上真机,全程 governor 兜底,逐步放开速度。**

> 现状(2026-06-14):M0–M4 的**软件**已就绪并验过(`diagnose()` IO 可插拔、AR 场景服务、动捕 frame glue、`run_real_g1.py --dry-run`)。本文聚焦**操作步骤**;M5 需要真硬件。

> **本文 vs `ar/instruction.md` 的分工**:本文是**分阶段安全闸门计划**(M0→M5,先 sim 验通再换 IO,逐步放速,每阶段有进入条件/验收/回退),适合**真机首次 bring-up 的安全顺序**。`ar/instruction.md` 是**AR 实验的逐步操作手册**(动捕地面标定、四种 AR 客户端的 build、每次实验启动顺序、排查)。**两者配合用**:按本文的 M0–M5 闸门走;到 M3(定位)/M5(障碍+AR)的**具体操作**(怎么贴坐标系、测 `T_world_scene`、起 `run_twin_server`、AR 配准)直接照 `ar/instruction.md` 的 Part A/B/C。本文覆盖**真实障碍**路径;纯 AR(虚拟障碍)路径以 `ar/instruction.md` 为准。

---

## 0. 验证过的部署配置(不要乱改)

sim 里 twogo 四个 zone × 5 seed 全部 **eSSR=1.00**(不摔 AND 全身无碰 AND 到达≤0.20)的配置,已写进 `run_sport_mode_zones.py` 默认值:

| 参数 | 值 | 作用 |
|---|---|---|
| `PLAN_SPEED` | **0.5** | 把参考轨迹 time-scale 到 SparkRL ~0.3 m/s 能力内,tracker 不再落后卡住 |
| `GOAL_HOLD_SEC` | **5.0** | plan 放完后继续追终点,让滞后的机器人走完 |
| 手臂收拢 | shoulder_roll **0.15** / elbow_base **0.60** | 手腕不擦侧墙(`HumanoidUpperBodyMapperConfig` 默认) |
| governor | 开 | body-SDF 运行时安全网(`m_track`、activation_band） |
| `xy_kp` / `xy_correction_cap` | 1.5 / 0.25 | 位置闭环把开环速度复现变成位置跟踪 |

这套配置**已是 `run_sport_mode_zones.py` / `run_real_g1.py` 的默认值**(`PLAN_SPEED=0.5`、`GOAL_HOLD_SEC=5.0`、轻收手在 mapper 默认),无需手动传。

> **部署哪个方法?用 twogo。** 同一套执行配置下,5 方法的部署 eSSR(不摔 AND 全身真实几何无碰 AND 到达):**twogo 1.00** > mdcoas 0.70 > mdoc 0.60 > ebmbd 0.40 > mbd 0.25。twogo 四个 zone 全满分,且优势在**全身可执行性**(脚/腿/臂都不擦),正是真机最看重的。
>
> **运行时机器人会"慢走 + 走完后多走 ~5s"**(plan_speed 0.5 + goal-hold)——这是刻意的(让 tracker 跟得上、走到终点),**别误以为卡住或失控**;首次可把 `--plan-speed` 调更小(0.35)更稳。

> ⚠️ **sim eSSR=1.00 ≠ 真机 1.00**。sim 用 MuJoCo 完美真值定位、sim 训的策略。真机有定位噪声/延迟、策略 sim-to-real gap、接触/摩擦差异。本流程的目的就是**安全地跨过这道 gap**。

---

## 0.5 你的机器与网络拓扑(1 台 Mac + G1,照 Unitree「External PC」接线)

> **2026-06-22 实测更新**:**拓扑 A(Mac 原生直连)已端到端验证通过**(通信/读状态全通,见 §0.6),比 B 省事,**优先走 A**。本节下面"推荐 B"是基于装包难度的事前判断;实测 A 只需把 CycloneDDS 在 macOS 源码编译一次即可,且**真机走高层 sport(腿 `LocoClient.Move` + 臂 `rt/arm_sdk`,对齐 SPARK),不依赖机载 PC**。详细实测路径、踩的坑、确定路线见 **§0.6**。

你的实际硬件:**一台 Mac**(装 Docker,跑实验/规划)+ **G1 本体**(自带机载 Linux 上位机)。Mac 与机器人用**网线直连**。

### 拓扑决策:控制进程跑在哪台机器?

`run_real_g1.py` 的**真机路径**需要 `mujoco`(取作动器规格)+ `jax`(import 链上需要,运行时不算)+ `unitree_sdk2py` + DDS 直连机器人;**governor/执行本身是纯 numpy**(已核实,无 jax/brax)。两种放法:

| | **B. 机载上位机控制(推荐)** | A. Mac 直接控制(备选) |
|---|---|---|
| 控制进程跑在 | 机器人**机载 Linux PC**(arm64) | Mac **原生 arm64 env** |
| DDS 到 `.161` | 机内本地网,稳 | 跨网线,需对网卡/防火墙 |
| 依赖 | Jetson 装 `mujoco`+`jax[cpu]`+`scipy`+`unitree_sdk2py`(都有 arm64) | macOS 上 `unitree_sdk2py` 官方不支持,风险高 |
| 用 Docker? | — | **不行**:Docker Desktop on Mac 跑不了到机器人的 DDS 多播 |
| 用 fedguide? | — | **不行**:fedguide 是 Rosetta x86_64,跑不了 arm64 `mujoco` |
| Mac 的角色 | 规划(Docker)+ scp plan + SSH 进去运行 | 规划 + 控制都在 Mac |

**推荐 B**:Mac 只做离线规划(Docker)产出 `trajectory.json`,scp 到机器人,SSH 进机载 PC 在那儿跑控制。理由:DDS 本地最稳、依赖都能在 arm64 Linux 装、避开 macOS-SDK / Docker-DDS / Rosetta-mujoco 三个坑。**下面默认按 B。**

### 在哪台机器跑什么

| 机器 | 跑什么 | 命令/产物 |
|---|---|---|
| **Mac(Docker `dev-cpu:torch`)** | M0 sim 复核、M1 标定、规划/重规划 | 产出 `…/trajectory.json` |
| **Mac(终端)** | 配网、`ping`、`ssh`、`scp` plan 到机器人 | 见下 |
| **机器人机载 PC(SSH 进)** | M2 dry-run、M4 龙门空跑、M5 落地(`run_real_g1.py` 真机路径) | `--network-interface eth0` |
| (可选)动捕机 | `natnet_ros2` + `natnet_shm_writer` → `mocap_state_shm` | 见 `ar/instruction.md` §A7 |

> `--dry-run` 在真机路径 import **之前**就返回,所以 dry-run 可在**任意机器**(含 Mac/Docker/fedguide)跑;**真机运行只能在控制机(机载 PC)上**。

### 接线 + 配网(照 Unitree「External PC Commanding」图)

1. 网线:Mac ↔ G1。
2. Mac 有线网卡设**静态 IP**:`192.168.123.222`(同网段任一,避开 `.161`),掩码 `255.255.255.0`。
   - macOS:系统设置 → 网络 →(USB/雷电网卡)→ 详细信息 → TCP/IP → 手动 → IP `192.168.123.222` / 子网 `255.255.255.0`。
3. 验证能通:`ping 192.168.123.161`(机器人主控板默认 IP)。**ping 不通别往下走**(查网线/IP/网卡名)。

### 穿梭进机器人(SSH + scp)

机载上位机(开发计算单元)在机内网,IP 视型号而定(常见 `192.168.123.164`,用户名 `unitree`;**以你机器实际为准**——可 `for i in 161 162 164 18; do ping -c1 -W1 192.168.123.$i; done` 探活)。

```bash
# 1) 在 Mac 上:把标定好的 plan 拷进机器人
scp results/.../trajectory.json unitree@192.168.123.164:~/plans/zone_a.json
# 2) SSH 进机载 PC
ssh unitree@192.168.123.164
# 3)(机载 PC,首次)取仓库 + 装依赖
#    git clone <本仓库> && cd enerdynamics      # 或从 Mac rsync 过去
#    pip install mujoco "jax[cpu]" scipy numpy unitree_sdk2py
# 4)(机载 PC)dry-run 验证 import/接线/frame(不碰电机)
python scripts/tasks/robot/humanoid/run_real_g1.py \
  --plan ~/plans/zone_a.json --preset zone_a --dry-run
```

> **备选拓扑 A(Mac 控制)**:跳过 SSH/scp,在 Mac **原生 arm64** env(**非** fedguide、**非** Docker)装 `mujoco`+`jax`+`unitree_sdk2py`;`--network-interface` 填 **Mac 网卡名**(如 `en7`,`ifconfig` 查,**不是** `eth0`)。能否在 macOS 跑通 `unitree_sdk2py` 需自行验证。

---

## 0.6 实机 bring-up 实测记录与确定路线(2026-06-22)

> 首次真机 bring-up(Mac + G1 网线直连)。**结论:拓扑 A 的通信链路全部验证通过;机器人最终没走起来,根因是 FSM id 用错了 ——pip SDK 的 `500/706`,而本机器人(按 SPARK)用 `LockStand=4 / MainMode=200`,从没进可行走态。读 SPARK 源码后路线已修正为「高层 sport」(腿 `LocoClient.Move`、臂 `rt/arm_sdk`),不是低层 lowcmd。** 下面是验过的环境、踩的坑、和确定的正确路线。

### A. 已验证通过 ✅(拓扑 A:Mac 原生直连)
- **网络**:Mac 网卡(本机是 USB 网卡 `en7`)设静态 `192.168.123.222`(掩码 `255.255.255.0`/24;实测填成 /16 也通,但建议 /24),`ping 192.168.123.161` 通 ~0.6ms;机载上位机 `192.168.123.164`(Ubuntu 20.04 aarch64,SSH 开,但本路线用不到它)。
- **CycloneDDS 在 macOS 必须源码编译**(无 `cyclonedds==0.10.2` 的 mac 轮子、无 brew formula;`unitree_sdk2py` 硬 pin 这个版本):
  ```bash
  git clone --depth 1 -b 0.10.2 https://github.com/eclipse-cyclonedds/cyclonedds /tmp/cdds
  cmake -S /tmp/cdds -B /tmp/cdds/build -DCMAKE_INSTALL_PREFIX=$HOME/g1/cyclonedds -DBUILD_IDLC=ON -DCMAKE_BUILD_TYPE=Release
  cmake --build /tmp/cdds/build --target install -j4   # 产出 libddsc + idlc
  ```
- **原生 arm64 venv**(homebrew py3.12;`fedguide` 是 Rosetta x86,装不了 arm mujoco,必须独立 venv)+ 依赖:
  ```bash
  /opt/homebrew/bin/python3 -m venv $HOME/g1/venv
  CYCLONEDDS_HOME=$HOME/g1/cyclonedds $HOME/g1/venv/bin/pip install "cyclonedds==0.10.2"
  CYCLONEDDS_HOME=$HOME/g1/cyclonedds $HOME/g1/venv/bin/pip install "git+https://github.com/unitreerobotics/unitree_sdk2_python.git"
  $HOME/g1/venv/bin/pip install "mujoco" "numpy<2" scipy "jax[cpu]"
  ```
  > `unitree_sdk2py` 不在 PyPI,从 git 装(import 名 `unitree_sdk2py`)。
- **运行环境变量(每次必带)**:`CYCLONEDDS_HOME=$HOME/g1/cyclonedds DYLD_LIBRARY_PATH=$HOME/g1/cyclonedds/lib`;网卡 Mac 上是 `en7`(机载 PC 上才是 `eth0`)。
- **通信只读自检通过**:订阅 `rt/lowstate` 收到实时 IMU + 35 路电机(索引 **0–28 为活动关节**,29–34 为手部/保留全 0),`mode_machine=5` → Mac↔G1 DDS 完全打通。
- **自由度对齐**:打包 MJCF `third_party/mujoco_menagerie/unitree_g1/scene.xml` = **29 作动器**(腿12+腰3+臂14),与真机活动的 0–28 一一对应。sport 高层不用逐关节映射;低层 lowcmd 按索引 0–28 直接对应。
- SDK 符号在 macOS 原生全部导入成功;`MotionSwitcherClient` 在新版 SDK 位于 `comm.motion_switcher`(旧版 `g1.motion_switcher`)—— `unitree_g1_io` 已改为两条路径都兼容。
- 测试脚本放在 `~/g1/`(`lowcmd_hold.py` 等),非仓库内容。

### B. 没走起来的根因 ❌(读了 SPARK 源码后修正)
1. **FSM id 用错了 —— 首要根因。** 今天用的是 **pip 版 SDK**:`Start()=SetFsmId(500)`、`Squat2StandUp()=706`。但 **SPARK(你们这台机器的权威参考)用的 FSM id 是:`Damp=1`、`Squat=2`、`Sit=3`、`LockStand(起身锁定)=4`、`MainMode(可行走主控)=200`**。`500/706` 对本固件是 "Invalid FSM ID"(7302)→ 被拒(`None`)→ 机器人一直停在 Damp(1),从没进 MainMode(200)。**不是控制模式问题,是 FSM id 不对 + 没走起身序列。**
2. **SPARK 用高层 sport,不是低层 lowcmd**(`level="low"` 在 SPARK 里直接 `NotImplementedError`)。它的实机控制 = **腿:`LocoClient.Move(vx,vy,vyaw)`(sport 内置 RL 步态);手臂:`rt/arm_sdk` 通道**(不是 `rt/lowcmd`)。所以"走低层"是之前带偏了 —— genedynamics 的 `RealLocoClient`(sport)方向本来就对。
3. SPARK vendored 了自己的 loco client,**补上了 pip SDK 没有的 `GetFsmId`**,据此读当前 FSM 再按 `…→LockStand(4)→MainMode(200)` 切。它 TODO 提示"先手动 LockStand",原因:pip SDK 没 GetFsmId + 它自动切换有个 `code==0` 拿来比元组的 bug;**不代表必须遥控器**。
4. 昨天试的 lowcmd 那条线对 SPARK/本机器人**根本不是主路**;它"没反应"同样是因为机器人没进 MainMode、没使能。

### C. 确定的正确路线(下次照此走 —— 高层 sport,对齐 SPARK)
1. **拓扑 A:Mac 原生直连**(已验证);不用机载 PC / Docker。
2. **高层 sport(不是 lowcmd)**:腿用 sport 内置 RL 步态,手臂用 arm_sdk。
   - 用 **SPARK 的 loco client**(`spark/module/spark_agent/spark_agent/real/g1/loco/g1_loco_client.py`,带 `GetFsmId` + 正确 FSM id),或把它搬进我们仓库。
   - **起身序列(纯 SDK,可能不用遥控器)**:`GetFsmId` 读当前 → `SetFsmId(4)`(LockStand,站起来锁定)→ `SetFsmId(200)`(MainMode,进可行走态)。**今天从没试过 4/200,只试了错的 500/706。**
   - **腿(走廊行走)**:`LocoClient.Move(vx,vy,vyaw, continuous_move=True)`;**手臂**:发 `rt/arm_sdk`(配 `kNotUsedJoint` 权重位 `motor_cmd[kNotUsedJoint].q = weight²`),**不要**发 `rt/lowcmd`。
3. **遥控器:不一定需要。** 先试纯 SDK 的 `SetFsmId(4)→(200)`;若被拒(固件要求),再用遥控器/App 把它弄到 LockStand,SDK 再接管到 MainMode。
4. **验证阶梯(修订)**:通信只读 ✅ → `SetFsmId(4)`(看是否站起来锁定)→ `SetFsmId(200)`(进 MainMode)→ `Move` 小速度(脚承重、龙门防摔)→ 走廊。
   > 注:sport 行走需脚承重才迈步(接触门控);龙门**完全离地只能测站,测不了走**。

### D. 仓库代码现状(deploy 已覆盖 SPARK 实机功能)
- **✅ 已做(本次,`unitree_g1_io.py`):**
  - `_bring_up_sport()`:起身序列 `GetFsmId → SetFsmId(4) LockStand → SetFsmId(200) MainMode`,接进 `_reset_robot`(sport);sport 不再调失败的 `SelectMode("normal")`。
  - 手臂改走 `rt/arm_sdk`(`_send_arm_sdk`,带 `kNotUsedJoint=29` 权重位 + 默认工厂构造),不再发 `rt/lowcmd`(避免和 sport 冲突)。
  - 修 `LowCmd_()` 无参构造 bug(改用 `unitree_hg_msg_dds__LowCmd_` 默认工厂)+ `rt/lowcmd` 补 `mode_machine`。
  - 早先:`MotionSwitcherClient` import 双路径兼容;`--localization ros2` 类名/键名;`BaseRegistry` 挪 `genedynamics/registry_base`。
  - **⇒ `run_real_g1.py` 现在 reset 就会自己 `4→200` 起身、腿走 `Move`、臂走 `arm_sdk`,代码层面已自给自足。**
- **⏳ 待在机器人上验证(全未验证):** FSM `4/200` 是否真起身;`Move` 是否迈步;`arm_sdk` 是否收臂。若 `SetFsmId(4)` 被固件拒,遥控器/App 先到 LockStand,再让它进 200。
- **可选微调(若上机不对):** `arm_sdk` 的 `motor_cmd.mode`(SPARK 留默认 0,官方 arm_sdk 例子设 1)。

### E. 下次最短复现(环境 `~/g1` 已在则跳过安装)
```bash
# 1) Mac en7 = 192.168.123.222/24；ping 192.168.123.161 通
# 2) 只读通信自检（订阅 rt/lowstate 看 IMU + 电机）
# 3) 用 SPARK 的 loco client（带 GetFsmId + 正确 FSM id）起身：
#    GetFsmId 读当前 → SetFsmId(4) 起身锁定 → SetFsmId(200) 进 MainMode
#    （被拒就用遥控器/App 先到 LockStand，再 SDK 切 200）
# 4) LocoClient.Move(vx=0.1,0,0, continuous_move=True) 小速度试走（脚承重 + 龙门防摔）
# 5) 手臂如需控制：发 rt/arm_sdk（不要 rt/lowcmd）
```

参考:SPARK `module/spark_agent/spark_agent/real/g1/g1_real_agent.py`;宇树 G1 SDK 文档;`unitree_sdk2_python` Issue #43(勿进 debug 模式)。

---

## M0 — sim 复核(必过,纯软件)

**进入条件**:有目标 zone 的 plan(`results/humanoid/corridor_2d/main/twogo_zone_<z>/level_1/seed_<s>/trajectory/trajectory.json`)。

**操作**(docker `genedynamics/dev-cpu:torch`):
```bash
# 单 zone 部署 + 看 certified + reach
python scripts/tasks/robot/humanoid/run_sport_mode_zones.py \
  --zones results/humanoid/corridor_2d/main/twogo_zone_a/level_1/seed_0/trajectory/trajectory.json \
  --out-root results/humanoid/corridor_2d/deploy/governed/twogo_zone_a/level_1 --plot
# 全身几何 + reach 审计(真实 G1 link vs 障碍 SDF)
python scripts/tasks/robot/humanoid/audit_exec_collision.py \
  --deploy-root results/humanoid/corridor_2d/deploy/governed --zones twogo_zone_a
```

**验收**:该 seed `eSSR=Y`(不摔 + `true_sdf≥0` + `endpoint≤0.20`)。看一眼 `trajectory_mujoco.gif` / `motion_strip.png` 确认姿态正常。

**回退**:不过就别上真机;先调 `plan_speed`(更慢)、换 seed、或回 M0 的规划侧。

---

## M1 — plan 标定 + 准备(纯软件)

真机执行的是**烤进 trajectory.json 的 `best_idx`(执行安全模式)+ `m_track`**。新规划/重规划的 plan 是未标定的,必须先标定。

**操作**:
```bash
# 对要部署的 zone/seed 标定(Step-1 选执行最安全的 mode,Step-4 推 m_track)
docker run ... genedynamics/dev-cpu:torch \
  python scripts/tasks/robot/humanoid/governor_calibrate.py --zones twogo_zone_a
# 若障碍是 AR/现场实测的,用 replan_from_scene 重规划 + 标定到该场景
python scripts/tasks/robot/humanoid/replan_from_scene.py --preset zone_a --scene-file <ar_scene.json>
```

**验收**:`trajectory.json` 里 `best_idx_source` 非空、`m_track` 有值;标定报告 `CERTIFIED`。

> 提醒:tight 场景(如 zone_d)执行间隙受 SparkRL ±0.3 m/s 限制,`m_track` 最优≈0.08(再大 governor 会把机器人挤离轨迹);**首次上实机选最容易的 zone_a**。

---

## M2 — dry-run 接线校验(无硬件)

不接机器人,验证真机 IO / 定位 / governor 的接线和 frame 变换都对。

**操作**:
```bash
python scripts/tasks/robot/humanoid/run_real_g1.py \
  --plan <calibrated trajectory.json> \
  --preset zone_a \
  --localization vicon \
  --t-world-scene 0 0 0 \
  --dry-run
```
`run_real_g1.py` 组装 `UnitreeG1RobotIO + RealLocoClient + SceneFrameLocalization(vicon, T_world_scene) → diagnose(governor)`,SDK 是 lazy import,`--dry-run` 不碰硬件。

**验收**:dry-run 打印 wiring OK、plan/scene/frame 参数无误、无异常。

**前置软件**:真机还需 `unitree_sdk2py`(机载 PC 装好);动捕侧 `natnet_ros2` 驱动 + 本仓库 `natnet_shm_writer` 往 `mocap_state_shm` 写 `q13d`(见 `ar/instruction.md` §A7)。

---

## M3 — 定位 bring-up(接动捕 NatNet,不动腿)

> **flag 名说明**:`--localization vicon` 是**历史遗留名**,实际语义是"读 `mocap_state_shm`"——不管那块共享内存是 Vicon DataStream 还是 **NatNet(`natnet_shm_writer`)** 写的,flag 都用 `vicon`。你的系统是 NatNet,起 `natnet_ros2`+`natnet_shm_writer` 即可,命令仍写 `--localization vicon`。

**进入条件**:G1 在**动捕捕捉区**,身上贴好 marker(在 Motive 里建成刚体 `g1_base`),`T_world_scene`(动捕世界→场景坐标的 x/y/yaw)已标定。

**操作**:
1. 启动 `natnet_ros2` 驱动 + `natnet_shm_writer`(`ar/instruction.md` §A7),确认 `mocap_state_shm` 在更新(`python -m genedynamics.deploy.localization.natnet_shm_writer --rigid-body g1_base`)。
2. 先用 `--localization mock` 跑通流程,再切 `--localization vicon`(= 读上一步的 `mocap_state_shm`)。
3. 静止状态下打印 `SceneFrameLocalization` 输出的 **scene-frame base pose**,人工比对机器人实际站位(应与 plan 起点一致)。
4. (可选)起 AR 数字孪生:`run_twin_server.py`,在 AR 端/网页看障碍+机器人位姿是否对齐。

**验收**:静止时 scene-frame pose 与真实站位误差 < 几 cm(取决于 tracker 精度,OptiTrack/NatNet 通常 <1cm);手动挪机器人,pose 跟着动且方向对。

**回退**:定位不准 → 重标 `T_world_scene` / 检查 marker / 标定坐标系手性映射。**定位不过坚决不放腿。**

---

## M4 — 吊装/龙门低速测试(放腿,空场,有保护)

**进入条件**:M2 dry-run 过。机器人**吊装或龙门悬挂**(脚能瞬间离地)、**空场无障碍**、**物理 e-stop 在手**。M3(动捕)**非必须**——先做无动捕的 mock 空跑(看步态/通信/急停),动捕好了再做闭环。

> 所有命令都在**机载 PC**(SSH 进去)上跑;`--network-interface` 填机载 PC 连内部 DDS 网的网卡(常见 `eth0`,`ifconfig` 确认)。

### M4.a 先跑通(无动捕,mock 空跑)— 推荐第一步
不接动捕,只验:能否**站立平衡 → 起步 → 步态正常 → governor cmd 平滑 → 急停可用**。
```bash
python scripts/tasks/robot/humanoid/run_real_g1.py \
  --plan ~/plans/zone_a.json --preset zone_a \
  --localization mock \
  --plan-speed 0.35 --max-steps 40 \
  --network-interface eth0 \
  --out-dir results/g1_corridor/real
```
- `--localization mock`:base 固定在起点、**不闭环**。**预期是原地/开环迈步,不是沿走廊行进**——空跑就是看这个,**别误判为卡住/失控**。
- 盯:`pelvis_z`(别塌)、迈步对称、loco cmd 不饱和、**急停能瞬停**。**任何异常立即 e-stop**。
- 稳了再逐步:`--max-steps` 加长 → `--plan-speed` 往 `0.5` 收。

### M4.b 再跑(接动捕 NatNet,闭环空跑)
动捕起好(`ar/instruction.md` §A7,`mocap_state_shm` 在更新)、`T_world_scene` 标好后,换成闭环位置跟踪:
```bash
python scripts/tasks/robot/humanoid/run_real_g1.py \
  --plan ~/plans/zone_a.json --preset zone_a \
  --localization vicon --t-world-scene <x> <y> <yaw> \
  --plan-speed 0.35 --max-steps 80 \
  --network-interface eth0 \
  --out-dir results/g1_corridor/real
```
- **从极慢开始**:`--plan-speed` 先 `0.35`、`--max-steps` 设小,先验证站立→起步→几步跟踪。
- 逐步放开:`max_steps` 加长 → `plan_speed` 往 0.5 收 → 走完全程。
- 全程盯:pelvis_z(别塌)、tracking 误差、governor 是否在乱动、cmd 是否饱和。

**验收**:空场低速能站稳、按 plan 走、不摔;落地数据 `sport_mode.json` 的 `fell_over=false`、tracking 合理。

**回退**:抖/塌/跟不上 → 降 `plan_speed`、降 `xy_kp`、回 sim 重调;**任何异常立即 e-stop**。这一步是 sim-to-real 重调的主战场(`plan_speed / xy_kp / goal_hold / m_track` 都可能要在真机上重整定)。

---

## M5 — 落地部署(着地,真/AR 障碍,全程 governor 兜底)

**进入条件**:M4 空场低速稳定、全程无摔、参数在真机上重调收敛。

**操作**:
1. **空场着地全程**:先在地面、无障碍跑完整条 plan,确认着地动力学 OK。
2. **加障碍**:
   - 真实障碍:按 `corridor_scene` 几何在**动捕世界系**摆好,`--t-world-scene` 对齐。
   - AR 障碍(共享数字孪生):`--scene-file <ar_scene.json>`,障碍只在虚拟世界,机器人靠**自身定位 + governor 感知**避让(无感知栈)。
3. **governor 是运行时安全网**:遇到未规划/移动的障碍,`diagnose(body_sdf_scene=...)` 注入场景,governor 在 ±0.3 m/s 横向能力内 steer 避让(M3 验证过:盲跟会撞进 −0.27m,governor 避到 +0.004m)。
4. 一次一个 zone,从 **zone_a(最易)→ c → d → b(U 墙最难)**;每个 zone 先单 seed 再扩。

**验收**:着地、真/AR 障碍下走完,`certified_safe`、不摔、endpoint 到位;多次重复稳定。

**安全红线(全程)**:
- e-stop 常备,操作员手不离;
- 速度只升不跳,异常先降速再排查;
- governor 必须开(它是唯一运行时避障兜底,别关);
- tight 场景(zone_d/b)执行间隙薄(~+0.08m 上限),留足物理安全距离 / 先用软障碍。

---

## 关键文件 / 模块

| 用途 | 路径 |
|---|---|
| 真机入口 | `scripts/tasks/robot/humanoid/run_real_g1.py`(`--dry-run` 先验) |
| sim 执行 + governor | `scripts/tasks/robot/humanoid/run_sport_mode_zones.py` → `sport_mode_corridor.diagnose()` |
| 标定(Step-1/4) | `scripts/tasks/robot/humanoid/governor_calibrate.py` |
| AR 重规划到场景 | `scripts/tasks/robot/humanoid/replan_from_scene.py` |
| AR 数字孪生服务 | `scripts/tasks/robot/humanoid/run_twin_server.py` |
| eSSR / 碰撞审计 | `scripts/tasks/robot/humanoid/audit_exec_collision.py` |
| 真机 IO | `genedynamics/deploy/io/unitree_g1_io.py` + `controllers/sport_mode/real_loco_client.py` |
| 定位(scene frame) | `genedynamics/deploy/localization/{scene_frame_plugin,vicon_shm_plugin}.py` |
| AR 架构 | `genedynamics/deploy/ar/ARCHITECTURE.md` |

## 真机还缺的硬件件(M5 前置)
- G1 本体 + `unitree_sdk2py`(机载/上位机);
- NatNet/OptiTrack 动捕(`natnet_ros2`)+ 本仓库 `natnet_shm_writer` 往 `mocap_state_shm` 写 `q13d`(见 `ar/instruction.md` §A7);
- `T_world_scene` 标定;
- (AR 可选)AR 端(Vision Pro / Android tablet)接 `run_twin_server`。

> sim 已验证"配置正确性"(twogo eSSR 1.00);真机就绪 = 走完 M3(定位)→M4(吊装低速重调)→M5(着地+障碍)。**不要跳级。**
