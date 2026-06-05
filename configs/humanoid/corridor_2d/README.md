# Humanoid Corridor 2D — Plan + Deploy 现状

本目录的任务：用 TwoGO planner 在 14D 状态空间里规划 G1 humanoid 的 corridor
轨迹 (xy + yaw + 手臂收缩 + torso pitch/yaw)，然后在 MuJoCo 里用 SparkRL
PPO policy 跟踪，并最终落到实机。

本 README 不解释代码结构，只回答两个问题：
1. spark 的 policy 是怎么训出来的？为什么它不适合我们的 plan？
2. 想跟得上 medium plan、并能跑实机，应该怎么重新训？

---

## 1. SparkRL policy 现状分析

### 1.1 我们手上拿到的东西

`genedynamics/deploy/policies/spark_rl/` 下只有一个文件：

- `motion.pt` — TorchScript checkpoint (47-D obs → 12-D action MLP)

**没有训练代码、没有 task 配置、没有 reward 定义、没有 curriculum。** Spark
仓库本身只发布了 inference checkpoint，对应的训练 pipeline 在 unitree_rl_lab
/ legged_gym 这一支生态里。所以我们看不到 reward shaping 和 episode 终止
条件，只能从 policy 接口反推。

### 1.2 从 policy 接口反推训练设置

通过 47-D obs 的拼装顺序 (定义在 `SparkRLPolicy.observe()` 里) 可以反推
training task：

```
obs = concat[
    omega          (3)  — base ang vel
    gravity_in_base(3)  — projected gravity
    cmd * scale    (3)  — [vx, vy, wz] in body frame, vx/vy scale=2.0, wz scale=0.25
    qj             (12) — leg joint pos relative to default
    dqj            (12) — leg joint vel
    last_action    (12) — previous policy output
    sin/cos phase  (2)  — gait clock, period=0.8s
]
```

action：

```
target_lower_body_pos = action * 0.25 + default_pos_lower_body
default_pos_lower_body = [-0.1, 0, 0, 0.3, -0.2, 0]  (left leg, mirror right)
```

PD gains (来自 spark 仓库的 MuJoCo 训练配置)：

| 关节        | kp  | kd |
|-------------|-----|----|
| hip_*       | 100 | 2  |
| knee_*      | 150 | 4  |
| ankle_*     |  40 | 2  |
| waist_*     | 300 | 5  |
| shoulder_*  |  90 | 2  |
| elbow_*     |  60 | 1  |
| wrist_*     |  20 | 1  |

**关键事实**：

- obs 里**没有任何 plan 信息**。policy 只看当前 base 状态 + 一个瞬时
  velocity command (`cmd`) + 一个 8 周期的 gait phase。它不知道前方 0.5 秒
  之后该往哪走，不知道场景里有没有墙，不知道走到哪一步了。
- `cmd` 在训练里被 clip 到 ±0.3 m/s (vx, vy 各自，wz=±0.5)。这就是为什么
  我们 plan 的 `vx_max=0.3` 是合适的边界——再大 policy 就走不动。
- gait phase 是固定时钟 0.8s，**不与 plan 对齐**。policy 自己内部决定什么
  时候迈左脚、什么时候迈右脚。
- task 是 **blind velocity tracking on flat ground**。reward 大概率是
  `track_lin_vel + track_ang_vel + alive + smooth + ...` 这一套
  legged_gym 模板。**没有 obstacle、没有路径、没有 footstep target。**
- 这是 spark 的"基础版"——一个通用 walker，不是 corridor / planner-aware
  policy。

### 1.3 为什么短 zone (a/b/c/d) 跑得动、medium 跑不下来

| 因素                 | 短 zone (5–8 m, ~12 s)              | medium (~22 m, ~37 s)                |
|----------------------|-------------------------------------|--------------------------------------|
| 总时长 vs 训练 ep    | 12 s 一般在 episode horizon 内      | 37 s 远超训练 episode horizon        |
| 误差累积             | 1–2 次微调就修正回来                | 累积偏离越来越大，xy P-control 救不动|
| cmd 饱和             | 主要直线，vx≈0.3                    | 转弯/绕障要 vx+vy 同时给到 0.3，对角方向合速度被 ±0.3 box clip 砍掉 √2 |
| gait phase 错相      | 短时间内相位漂移有限                | 长时间下 policy 内部 0.8s 时钟跟 plan 完全对不上 |
| PD gain 错配         | G1 XML 默认 kp=500 vs 训练 kp=100/150/40，5–12× 偏硬。短时间靠 P-control 还能压住 | 长时间下高刚度让微小 cmd 误差被放大成大幅抖动 |

所以表现是：

- 短 zone：**deploy 成功**(`spark_rl_final_03x.gif`)，但身体抖动明显——这
  个抖动是 policy gait 本身的特征，不是 cmd 噪声 (LPF 实验已经证伪)。
- medium：早期还能跟住，后段视觉上看起来"上障碍物了"——其实是 pelvis 一直
  在 gap 里，只是相机被半透明的 squeeze 几何挡住造成视觉欺骗 (顶视
  elev=-45° 重渲染后已经能看清楚)。但姿态稳定性边际确实在恶化。

### 1.4 已经做过、有效的补救（仍然在用 spark policy 的前提下）

这些是当前 deploy pipeline 在 inference 时干的事，写在
`scripts/tasks/robot/humanoid/sport_mode_corridor.py` 里：

1. **Body-frame velocity 转换**：plan 输出的 `(vx_world, vy_world)` 在送给
   policy 前先按当前 yaw 反转，这样 policy 看到的 cmd 永远是身体系——和
   它训练时看到的一致。
2. **XY P-control 闭环**：`v_cmd = v_plan + xy_kp * (xy_plan - xy_actual)`，
   再做 `xy_correction_cap` 限幅。把开环 velocity replay 改成闭环位置
   tracker。
3. **Cmd warmup ramp**：前 0.5 s 把 cmd 从 0 线性 ramp 到 `plan[0].v`。直接
   给 0 cmd 会导致 policy 在零点附近"倒走" (零速度是训练分布的奇异点)。
4. **Spark MuJoCo PD gain patch**：deploy 时运行时改写
   `model.actuator_gainprm/biasprm`，把默认 kp=500 替换为
   hip=100/knee=150/ankle=40——和 spark 训练 PD 一致。这是 medium 能跑通的
   关键修复 (之前 5–12× 刚度错配造成结构性 sim2sim failure)。
5. **Plan 增加手臂收缩 + torso 信息**：14D plan 里第 9–14 维是
   `[arm_tuck, arm_yaw, arm_pitch, torso_yaw, torso_pitch, torso_roll]`，
   deploy 时由 `SportModeController` 的上身 PD 通道直接执行，不经过 policy。

WBC controller 也试过 (`run_humanoid_corridor_g1_wbc.py`)，4 个 zone 全部
在 t≈1.56 s 第 2→3 swing 切换时跌倒，伴随 ankle torque 饱和到 21.5 N·m。
没有继续调下去。

---

## 2. 为 medium + 实机重训 policy 的方案

短结论：**用 Isaac Lab + unitree_rl_lab 这一套训练栈**。理由：

- spark 的 inference 接口 (47-D obs / 12-D leg action / 0.8 s 固定相位) 就是
  从这个生态出来的。重训出来的 checkpoint 可以直接替换 `motion.pt`，部署
  pipeline 改动最小。
- unitree_rl_lab 默认带 G1 walk task、PPO baseline、URDF/XML 双 sim、ROS
  桥，对实机最友好。
- Isaac Lab (基于 IsaacSim / GPU rigid body) 训练吞吐比 legged_gym 高 5–10
  倍，且 domain randomization、terrain curriculum、actuator 模型都更细。

### 2.1 训练任务怎么改才能跟得上 medium plan

如果只是直接拿 spark 的 walk task 重训，问题不会变好——它仍然是 blind
velocity tracking。要想 long-horizon plan-following 稳定，至少改三件事：

**(a) Observation 扩到 plan-conditioned**

把未来一段 plan 喂进 obs：

```
obs += [
    plan_xy_rel_body[:, 0:H_obs],   # 未来 H_obs 步 (例如 H_obs=10) 的目标位置 in body frame
    plan_yaw_rel_body[:, 0:H_obs],
]
```

这样 policy 能"看到"前方 1–2 秒的 reference，长 horizon 就不会一直只靠
瞬时 vel cmd 死扛。这是 OmniH2O / HumanoidPolicy / RoboMimic 这一类做法。

**(b) Reward 加 plan-tracking 项**

```
r_track_xy   = exp(- (xy - xy_plan(t))^2 / sigma_xy^2)
r_track_yaw  = exp(- wrap(yaw - yaw_plan(t))^2 / sigma_yaw^2)
r_alive      = ... (legged_gym 默认那套)
r_smooth     = ... (action rate, joint accel)
r_obstacle   = (可选) 离 SDF wall 距离的 barrier
```

reward weight 上 plan-tracking 别压过 alive/smooth，否则 policy 会为了
追位置而摔。

**(c) Curriculum**

- 训练初期：xy_plan 是直线 + 慢速，policy 学到 "follow xy"。
- 中期：引入 zone_a/c/d 这种短弯道，episode 长 12–15 s。
- 后期：medium 这种长 plan，episode 30–40 s，并加 obstacle SDF 进 reward。
- 全程开 domain randomization (mass ±15%, friction 0.6–1.2, motor strength
  ±10%, latency 0–30 ms, base init perturbation ±5 cm)。

### 2.2 实机部署的补充考虑

要从 sim 跨到 G1 实机，下面这些事必须在训练阶段就锁死：

- **Action space**：保持和 spark 一样的 12-D leg delta + `default_pos`，
  上身用 PD 直驱。这样 G1 SDK 那边的 motor command pipeline 不用改。
- **Control rate**：训练就用 50 Hz policy / 500 Hz physics，和实机一致。
- **PD gain**：训练时就把 hip=100/knee=150/ankle=40 写死，**不要**等到部
  署再 patch (我们当前 spark 那个 runtime patch 是补救，不是设计)。
- **Latency randomization**：obs 延迟 / action 延迟各自 sample 0–30 ms，
  这是 sim2real gap 最大的来源之一。
- **Actuator model**：用 unitree_rl_lab 自带的 G1 motor model (含 torque
  limit + ω-τ 曲线)，不要用 ideal PD。
- **Terrain noise**：训练地形里加 ±2 cm 高度扰动 + 摩擦随机化，否则平地
  policy 一上实机就抖。
- **Push perturbation**：每隔 N 步给 base 一个 sample 自 [−30, 30] N 的
  脉冲，提升鲁棒性。
- **Sensor noise**：IMU ω 加 σ=0.05 rad/s，joint pos 加 σ=0.001 rad，
  joint vel 加 σ=0.05 rad/s 的高斯噪声。

### 2.3 具体推荐的下一步

明天就能动手的事：

1. `git clone` unitree_rl_lab，跑通它默认的 G1 walk task (不改 reward / obs)，
   验证 Isaac Lab 装好了、pipeline 通了。
2. 把它训出来的 `motion.pt` 灌进我们的 `SparkRLPolicy`，跑
   `humanoid/sport_mode_corridor.py twogo_zone_a` 验证接口 1:1 兼容
   (47-D obs / 12-D action / 0.25 scale / 0.8 s phase)。
3. 在 unitree_rl_lab 的 task 里 fork 出一个 `g1_corridor_follow`，按 2.1
   改 obs 和 reward。先在直线 plan 上验证 plan-conditioned 比 blind cmd 强。
4. 加 zone_a/c/d 短 plan 训练，最后 episode 拉到 40 s + medium plan +
   obstacle SDF reward。
5. DR + actuator model + latency 上齐之后再考虑实机。

---

## 3. 目录结构 + canonical gif 位置

```
configs/humanoid/corridor_2d/
├── plan/      # 5 个 TwoGO planner config (输出 trajectory.json)
│   ├── twogo_zone_a.yaml
│   ├── twogo_zone_b.yaml
│   ├── twogo_zone_c.yaml
│   ├── twogo_zone_d.yaml
│   └── twogo_long.yaml          # 之前叫 twogo_medium
├── deploy/    # 5 个 deploy + render config (final 版本参数固化)
│   ├── twogo_zone_a.yaml
│   ├── twogo_zone_b.yaml
│   ├── twogo_zone_c.yaml
│   ├── twogo_zone_d.yaml
│   └── twogo_long.yaml
└── smoke/     # 仅留 mbd / mdcoas 的早期 smoke (非 2GO)
```

最后一轮、`cam-mode global` + `azimuth=90` + `elevation=-45` + `every_n=4` +
`1200×600` + `speed=0.3`、用 spark PD patch + warmup-ramp + body-frame XY
P-control 渲染的 5 个 gif：

```
results/deploy/spark_rl/twogo_zone_a/spark_rl_final_03x.gif
results/deploy/spark_rl/twogo_zone_b/spark_rl_final_03x.gif
results/deploy/spark_rl/twogo_zone_c/spark_rl_final_03x.gif
results/deploy/spark_rl/twogo_zone_d/spark_rl_final_03x.gif
results/deploy/spark_rl/twogo_long/spark_rl_final_03x.gif
```

对应 plan 配置：

| Scene  | Plan config                          | vx_max | horizon | 备注 |
|--------|--------------------------------------|--------|---------|------|
| zone_a | `plan/twogo_zone_a.yaml`             | 0.3    | 50      | gate_dynamics on |
| zone_b | `plan/twogo_zone_b.yaml`             | 0.8    | 20      | 唯一一个回到原始 v/h，0.3 走不通 U-wall gap |
| zone_c | `plan/twogo_zone_c.yaml`             | 0.3    | 50      | M_k=1024, Ndiffuse=100 (默认就够) |
| zone_d | `plan/twogo_zone_d.yaml`             | 0.3    | 50      | gate_dynamics on |
| long   | `plan/twogo_long.yaml`               | 0.3    | 150     | 8/8 mode 全成功 |

复现命令：

```bash
# Plan
python -m genedynamics.experiments.runner configs/humanoid/corridor_2d/plan/twogo_zone_a.yaml

# Deploy + render (参数从 deploy yaml 读，CLI flag 仍可覆盖)
python scripts/tasks/robot/humanoid/sport_mode_corridor.py \
    --deploy-config configs/humanoid/corridor_2d/deploy/twogo_zone_a.yaml \
    --out-dir results/deploy/spark_rl/twogo_zone_a
# 上面运行结束会打印对应的 render 命令，可以直接 copy 跑：
python scripts/visualizations/render_spark_rl_corridor_gif.py \
    --npz results/deploy/spark_rl/twogo_zone_a/sport_mode.npz \
    --out results/deploy/spark_rl/twogo_zone_a/spark_rl_final_03x.gif \
    --cam-mode global --cam-azimuth 90 --cam-elevation -45 \
    --every-n 4 --width 1200 --height 600 --speed 0.3

# 一次跑 5 个 zone
python scripts/tasks/robot/humanoid/run_sport_mode_zones.py
```
