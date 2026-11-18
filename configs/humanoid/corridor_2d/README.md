# Humanoid Corridor 2D — Current Planning and Deployment State

This directory uses the TwoGO planner to generate G1 humanoid corridor trajectories
in a 14D state space (xy, yaw, arm retraction, and torso pitch/yaw), tracks them in
MuJoCo with a SparkRL PPO policy, and ultimately targets hardware deployment.

This README addresses two questions rather than explaining the code structure:
1. How was the Spark policy trained, and why does it not fit our plans?
2. How should it be retrained to follow medium plans and run on hardware?

---

## 1. Current SparkRL Policy Analysis

### 1.1 Available Artifacts

`genedynamics/deploy/policies/spark_rl/` contains just one file:

- `motion.pt` — TorchScript checkpoint (47-D observation → 12-D action MLP)

**No training code, task configuration, reward definition, or curriculum is available.**
The Spark repository publishes an inference checkpoint; its training pipeline belongs
to the unitree_rl_lab / legged_gym ecosystem. Reward shaping and episode termination
conditions cannot be inspected directly; we can only infer them from the interface.

### 1.2 Inferring Training Setup from the Policy Interface

The 47-D observation assembly in `SparkRLPolicy.observe()` suggests the training task:

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

Action:

```
target_lower_body_pos = action * 0.25 + default_pos_lower_body
default_pos_lower_body = [-0.1, 0, 0, 0.3, -0.2, 0]  (left leg, mirror right)
```

PD gains (from the Spark repository's MuJoCo training configuration):

| Joint | kp | kd |
|-------------|-----|----|
| hip_* | 100 | 2 |
| knee_* | 150 | 4 |
| ankle_* | 40 | 2 |
| waist_* | 300 | 5 |
| shoulder_* | 90 | 2 |
| elbow_* | 60 | 1 |
| wrist_* | 20 | 1 |

**Key observations:**

- Observations contain **no plan information**. The policy sees current base state,
  an instantaneous velocity command (`cmd`), and an eight-cycle gait phase. It does
  not know where to go 0.5 seconds ahead, whether walls are present, or its plan progress.
- Training clips `cmd` to ±0.3 m/s independently for vx/vy, and wz to ±0.5. This explains
  why `vx_max=0.3` is an appropriate plan limit; larger commands exceed policy capability.
- The gait phase follows a fixed 0.8s clock, **not aligned with the plan**. The policy
  internally determines left/right foot timing.
- The task is **blind velocity tracking on flat ground**. The reward is likely the
  legged_gym template `track_lin_vel + track_ang_vel + alive + smooth + ...`.
  **No obstacles, paths, or footstep targets.**
- This is Spark's basic general walker, not a corridor/planner-aware policy.

### 1.3 Why Short Zones (a/b/c/d) Work but Medium Fails

| Factor | Short zone (5–8 m, ~12 s) | Medium (~22 m, ~37 s) |
|---|---|---|
| Duration vs training episode | 12 s usually fits the episode horizon | 37 s far exceeds the training episode horizon |
| Error accumulation | One or two small corrections restore tracking | Drift accumulates beyond xy P-control's ability to recover |
| Command saturation | Mostly straight, vx≈0.3 | Turns/avoidance require vx+vy at 0.3; the ±0.3 box clip cuts diagonal resultant speed by √2 |
| Gait phase mismatch | Limited short-term drift | Over long runs, the internal 0.8s clock becomes entirely misaligned with the plan |
| PD gain mismatch | G1 XML defaults kp=500 vs training kp=100/150/40, 5–12× too stiff; short-term P-control can compensate | High stiffness amplifies small command errors into strong shaking over long runs |

Observed behavior:

- Short zones: **deployment succeeds** (`spark_rl_final_03x.gif`), but body shaking is
  visible. This is a property of the policy gait, not command noise (LPF experiments
  ruled that out).
- Medium: tracking works initially; later the robot appears to climb an obstacle.
  The pelvis actually stays in the gap; translucent squeeze geometry occludes the
  camera and creates a misleading view, clarified by a top-view elev=-45° rerender.
  However, the posture-stability margin does deteriorate.

### 1.4 Effective Mitigations Already Applied with the Spark Policy

The current deployment pipeline applies these inference-time changes in
`scripts/tasks/robot/humanoid/sport_mode_corridor.py`:

1. **Body-frame velocity conversion:** rotate planned `(vx_world, vy_world)` by inverse
   current yaw before policy input, matching the training command frame.
2. **Closed-loop XY P-control:** `v_cmd = v_plan + xy_kp * (xy_plan - xy_actual)`, bounded
   by `xy_correction_cap`, converts open-loop velocity replay into closed-loop position tracking.
3. **Command warmup ramp:** linearly ramp from 0 to `plan[0].v` over the first 0.5 s.
   Direct zero commands make the policy walk backward near zero, a singular point in
   its training distribution.
4. **Spark MuJoCo PD gain patch:** modify `model.actuator_gainprm/biasprm` at runtime,
   replacing default kp=500 with hip=100/knee=150/ankle=40 to match training PD.
   This was the key fix for medium execution; the previous 5–12× stiffness mismatch
   caused a structural sim-to-sim failure.
5. **Arm retraction and torso information in plans:** dimensions 9–14 of the 14D plan
   are `[arm_tuck, arm_yaw, arm_pitch, torso_yaw, torso_pitch, torso_roll]`.
   `SportModeController` executes them directly through upper-body PD, bypassing the policy.

A WBC controller was also tried (`run_humanoid_corridor_g1_wbc.py`). All four zones
fell at t≈1.56 s during the second-to-third swing transition, with ankle torque
saturating at 21.5 N·m. Further tuning was not pursued.

---

## 2. Retraining for Medium Plans and Hardware

Use the **Isaac Lab + unitree_rl_lab training stack**. Rationale:

- Spark's inference interface (47-D observations / 12-D leg actions / fixed 0.8 s phase)
  comes from this ecosystem. A retrained checkpoint can replace `motion.pt` with
  minimal deployment-pipeline changes.
- unitree_rl_lab includes a G1 walk task, PPO baseline, URDF/XML dual simulation,
  and a ROS bridge, making it suitable for hardware.
- Isaac Lab (IsaacSim/GPU rigid-body physics) offers 5–10× legged_gym training throughput,
  with finer domain randomization, terrain curricula, and actuator models.

### 2.1 Training Changes Needed to Follow Medium Plans

Simply retraining Spark's walk task leaves the problem unchanged: blind velocity
tracking. Stable long-horizon plan following requires at least three changes:

**(a) Expand observations to include the plan**

Supply a future plan window:

```
obs += [
    plan_xy_rel_body[:, 0:H_obs],   # Next H_obs targets (e.g. H_obs=10), in body frame
    plan_yaw_rel_body[:, 0:H_obs],
]
```

The policy can then see 1–2 seconds of reference instead of relying exclusively on
instantaneous velocity commands over long horizons. This follows approaches such as
OmniH2O / HumanoidPolicy / RoboMimic.

**(b) Add plan-tracking rewards**

```
r_track_xy   = exp(- (xy - xy_plan(t))^2 / sigma_xy^2)
r_track_yaw  = exp(- wrap(yaw - yaw_plan(t))^2 / sigma_yaw^2)
r_alive      = ... (standard legged_gym terms)
r_smooth     = ... (action rate, joint accel)
r_obstacle   = (optional) barrier on distance to the SDF wall
```

Do not weight plan tracking above alive/smooth terms so heavily that the policy
falls while chasing position.

**(c) Curriculum**

- Early: straight, slow xy_plan; learn to follow xy.
- Middle: short turns such as zone_a/c/d; episodes of 12–15 s.
- Late: long medium plans; episodes of 30–40 s and obstacle SDF rewards.
- Throughout: domain randomization (mass ±15%, friction 0.6–1.2, motor strength
  ±10%, latency 0–30 ms, base-initialization perturbation ±5 cm).

### 2.2 Additional Hardware-Deployment Considerations

For simulation-to-G1 hardware transfer, fix these during training:

- **Action space:** Spark-compatible 12-D leg deltas + `default_pos`, with direct
  upper-body PD. The G1 SDK motor-command pipeline then remains unchanged.
- **Control rate:** 50 Hz policy / 500 Hz physics, matching hardware.
- **PD gains:** hip=100/knee=150/ankle=40 from training; **do not** wait until deployment
  to patch them. The current Spark runtime patch is a remedy, not the intended design.
- **Latency randomization:** sample observation/action delays separately at 0–30 ms,
  one of the largest sim-to-real gap sources.
- **Actuator model:** use unitree_rl_lab's G1 motor model, including torque limits and
  the ω-τ curve, instead of ideal PD.
- **Terrain noise:** add ±2 cm height variation and friction randomization; otherwise,
  a flat-ground policy shakes immediately on hardware.
- **Push perturbations:** every N steps, apply a base impulse sampled from [−30, 30] N
  to improve robustness.
- **Sensor noise:** Gaussian IMU ω noise σ=0.05 rad/s, joint-position σ=0.001 rad,
  and joint-velocity σ=0.05 rad/s.

### 2.3 Recommended Next Actions

Concrete actions to begin next:

1. Clone unitree_rl_lab and run its default G1 walk task without changing rewards or
   observations, validating Isaac Lab installation and the pipeline.
2. Load its trained `motion.pt` into `SparkRLPolicy`; run
   `humanoid/sport_mode_corridor.py twogo_zone_a` to verify 1:1 compatibility
   (47-D observations / 12-D actions / 0.25 scale / 0.8 s phase).
3. Fork a `g1_corridor_follow` task in unitree_rl_lab; modify observations/rewards per
   Section 2.1. First test whether plan conditioning improves on blind commands on straight plans.
4. Train on short zone_a/c/d plans, then extend to 40 s episodes, medium plans,
   and obstacle SDF rewards.
5. Consider hardware only after adding domain randomization, actuator models, and latency.

---

## 3. Directory Structure and Canonical GIFs

```
configs/humanoid/corridor_2d/
├── plan/      # 5 TwoGO planner configs (output: trajectory.json)
│   ├── twogo_zone_a.yaml
│   ├── twogo_zone_b.yaml
│   ├── twogo_zone_c.yaml
│   ├── twogo_zone_d.yaml
│   └── twogo_long.yaml          # Previously twogo_medium
├── deploy/    # 5 deployment/render configs (final parameters fixed)
│   ├── twogo_zone_a.yaml
│   ├── twogo_zone_b.yaml
│   ├── twogo_zone_c.yaml
│   ├── twogo_zone_d.yaml
│   └── twogo_long.yaml
└── smoke/     # Early mbd/mdcoas smoke tests only (not 2GO)
```

The latest five GIFs use `cam-mode global`, `azimuth=90`, `elevation=-45`, `every_n=4`,
`1200×600`, `speed=0.3`, Spark PD patching, warmup ramps, and body-frame XY P-control:

```
results/deploy/spark_rl/twogo_zone_a/spark_rl_final_03x.gif
results/deploy/spark_rl/twogo_zone_b/spark_rl_final_03x.gif
results/deploy/spark_rl/twogo_zone_c/spark_rl_final_03x.gif
results/deploy/spark_rl/twogo_zone_d/spark_rl_final_03x.gif
results/deploy/spark_rl/twogo_long/spark_rl_final_03x.gif
```

Corresponding plan configurations:

| Scene | Plan config | vx_max | horizon | Notes |
|---|---|---|---|---|
| zone_a | `plan/twogo_zone_a.yaml` | 0.3 | 50 | gate_dynamics on |
| zone_b | `plan/twogo_zone_b.yaml` | 0.8 | 20 | Only scene restored to original v/h; 0.3 cannot pass the U-wall gap |
| zone_c | `plan/twogo_zone_c.yaml` | 0.3 | 50 | M_k=1024, Ndiffuse=100 (defaults suffice) |
| zone_d | `plan/twogo_zone_d.yaml` | 0.3 | 50 | gate_dynamics on |
| long | `plan/twogo_long.yaml` | 0.3 | 150 | All 8/8 modes succeed |

Reproduction commands:

```bash
# Plan
python -m genedynamics.experiments.runner configs/humanoid/corridor_2d/plan/twogo_zone_a.yaml

# Deploy + render (parameters come from deployment YAML; CLI flags can override)
python scripts/tasks/robot/humanoid/sport_mode_corridor.py \
    --deploy-config configs/humanoid/corridor_2d/deploy/twogo_zone_a.yaml \
    --out-dir results/deploy/spark_rl/twogo_zone_a
# The run prints the corresponding rendering command, ready to copy and execute.
# It produces tracking / sport_mode / motion_strip (green line) / trajectory_mujoco.gif:
python scripts/visualizations/render_deploy_humanoid.py \
    --npz results/deploy/spark_rl/twogo_zone_a/sport_mode.npz \
    --which all

# Run all 5 zones
python scripts/tasks/robot/humanoid/run_sport_mode_zones.py
```
