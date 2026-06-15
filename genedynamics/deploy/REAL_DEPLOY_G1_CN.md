# G1 走廊 2GO 上实机操作步骤（M0–M5）

把 humanoid corridor 的 2GO 规划轨迹部署到**真 Unitree G1**的实战 runbook。
分 M0–M5 六个阶段,**每阶段有明确的进入条件、操作命令、验收标准、回退方式**。
原则:**先在 sim 把一切验通,再换 IO 上真机,全程 governor 兜底,逐步放开速度。**

> 现状(2026-06-14):M0–M4 的**软件**已就绪并验过(`diagnose()` IO 可插拔、AR 场景服务、Vicon frame glue、`run_real_g1.py --dry-run`)。本文聚焦**操作步骤**;M5 需要真硬件。

> **本文 vs `ar/instruction.md` 的分工**:本文是**分阶段安全闸门计划**(M0→M5,先 sim 验通再换 IO,逐步放速,每阶段有进入条件/验收/回退),适合**真机首次 bring-up 的安全顺序**。`ar/instruction.md` 是**AR 实验的逐步操作手册**(Vicon 地面标定、四种 AR 客户端的 build、每次实验启动顺序、排查)。**两者配合用**:按本文的 M0–M5 闸门走;到 M3(定位)/M5(障碍+AR)的**具体操作**(怎么贴坐标系、测 `T_world_scene`、起 `run_twin_server`、AR 配准)直接照 `ar/instruction.md` 的 Part A/B/C。本文覆盖**真实障碍**路径;纯 AR(虚拟障碍)路径以 `ar/instruction.md` 为准。

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

**前置软件**:真机还需 `unitree_sdk2py`(在机载/上位机装好)、Vicon 往 shm `mocap_state_shm` 写 `q13d` 的 writer。

---

## M3 — 定位 bring-up(接 Vicon,不动腿)

**进入条件**:G1 在 Vicon 捕捉区,身上贴好 marker,`T_world_scene`(Vicon 世界→场景坐标的 x/y/yaw)已标定。

**操作**:
1. 启动 Vicon → shm writer,确认 `mocap_state_shm` 在更新。
2. 先用 `--localization mock` 跑通流程,再切 `vicon`。
3. 静止状态下打印 `SceneFrameLocalization` 输出的 **scene-frame base pose**,人工比对机器人实际站位(应与 plan 起点一致)。
4. (可选)起 AR 数字孪生:`run_twin_server.py`,在 AR 端/网页看障碍+机器人位姿是否对齐。

**验收**:静止时 scene-frame pose 与真实站位误差 < 几 cm(取决于 tracker 精度,Vicon 通常 <1–2cm);手动挪机器人,pose 跟着动且方向对。

**回退**:定位不准 → 重标 `T_world_scene` / 检查 marker / 标定坐标系手性映射。**定位不过坚决不放腿。**

---

## M4 — 吊装/龙门低速测试(放腿,空场,有保护)

**进入条件**:M2/M3 全过。机器人**吊装或龙门悬挂**(能瞬间离地),**空场无障碍**,e-stop 在手。

**操作**:
```bash
python scripts/tasks/robot/humanoid/run_real_g1.py \
  --plan <calibrated zone_a trajectory.json> \
  --preset zone_a --localization vicon \
  --t-world-scene <x> <y> <yaw> \
  --max-steps 80          # 先短,只走几步
  --out-dir results/g1_corridor/real
```
- **从极慢开始**:可临时把 `--plan-speed` 调更小(如 0.35)、`--max-steps` 设小,先验证站立→起步→几步跟踪。
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
   - 真实障碍:按 `corridor_scene` 几何在 Vicon 世界摆好,`--t-world-scene` 对齐。
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
- Vicon/OptiTrack 捕捉 + 往 `mocap_state_shm` 写 `q13d` 的 writer;
- `T_world_scene` 标定;
- (AR 可选)AR 端(Vision Pro / Android tablet)接 `run_twin_server`。

> sim 已验证"配置正确性"(twogo eSSR 1.00);真机就绪 = 走完 M3(定位)→M4(吊装低速重调)→M5(着地+障碍)。**不要跳级。**
