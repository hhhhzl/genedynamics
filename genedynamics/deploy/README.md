# Deploy: Sim / Shadow / Replay / Real

Unified execution layer for UAV 3D, Quadruped, and Humanoid.

## Prerequisites

- JAX, mujoco, mujoco-mjx; gymnasium; PyYAML

## Unified CLI

```bash
# Sim (UAV, Quadruped, Humanoid)
genedynamics-deploy --config configs/uav3d/sim.yaml
genedynamics-deploy --config configs/quadruped/flat/mbd_deploy.yaml
genedynamics-deploy --robot humanoid --model g1 --mode sim --episodes 2

# Shadow (full telemetry)
genedynamics-deploy --config configs/quadruped/env/shadow.yaml
genedynamics-deploy --config configs/uav3d/shadow.yaml

# Replay
genedynamics-deploy --replay-dir 'results/deploy/uav3d_sim/episodes/ep_0001_*' --list

# Real (stub or hardware)
genedynamics-deploy --config configs/quadruped/env/real.yaml --mode real
```

### Shortcuts
```bash
genedynamics-deploy --config configs/quadruped/flat/mbd_deploy_quick.yaml --episodes 1 --max-steps 30
genedynamics-deploy --config configs/quadruped/flat/mbd_deploy.yaml --viz --viz-serve
```

## Phase 2: MBD + MD-COAS (cfsmbd_full)

Quadruped closed-loop with obstacle avoidance:

```bash
genedynamics-deploy --config configs/quadruped/flat/mdcoas_deploy.yaml
genedynamics-deploy --config configs/quadruped/obstacle_avoid/mdcoas_deploy.yaml
```

## Phase 3: Three Quadruped Tasks

| Task | Config | Description |
|------|--------|-------------|
| **obstacle_avoid** | `obstacle_avoid/mdcoas_deploy.yaml` | 3D spherical obstacles, CFS-MBD full QP |
| **rough_terrain** | `rough_terrain/mbd_deploy.yaml` | Ant on rough terrain |
| **push_recovery** | `push_recovery/mbd_deploy.yaml` | External impulse perturbation |

```bash
genedynamics-deploy --config configs/quadruped/obstacle_avoid/mdcoas_deploy.yaml
genedynamics-deploy --config configs/quadruped/rough_terrain/mbd_deploy.yaml
genedynamics-deploy --config configs/quadruped/push_recovery/mbd_deploy.yaml
```

## Phase 4: Closed-loop validation and fair comparison

```bash
# Full validation (MBD, MD-COAS, obstacle_avoid, rough_terrain, push_recovery)
python scripts/tasks/soft_robot/run_phase4_validation.py

# Or: bash scripts/tasks/robot/run_phase4_validation.sh

# Fair comparison only (MBD vs MD-COAS, same env)
bash scripts/tasks/soft_robot/run_phase4_fair_comparison.sh

# Custom venv (e.g. conda fedguide)
VENV=conda_run_fedguide python scripts/tasks/soft_robot/run_phase4_validation.py
```

## Report Generation (Phase 4)

Unified industrial-grade report from all results:

```bash
python -m genedynamics.reports --input results --output reports
python -m genedynamics.reports -i results -o reports -f html pdf
```

Reports auto-generate after `run_all` (config: `auto_report: true`).

## Output

- `results/deploy/uav3d_sim/episodes/` - UAV episode data
- `results/deploy/quadruped_sim/episodes/` - quadruped episode data
- `results/deploy/humanoid_sim/episodes/` - humanoid episode data
- `reports/report.html` - unified experiment + deploy report
- Replay-ready format for failure analysis and sim2real comparison

## Viewing and Validating Deploy Results

### 1. List episode contents
```bash
genedynamics-deploy --replay-dir 'results/deploy/uav3d_sim/episodes/ep_0001_*' --list
```

### 2. Analyze episodes (distance to target, crash detection)
```bash
python scripts/visualizations/analyze_deploy.py results/deploy/uav3d_sim --robot uav3d
python scripts/visualizations/analyze_deploy.py results/deploy/quadruped_sim --robot quadruped
python scripts/visualizations/analyze_deploy.py results/deploy/uav3d_sim/episodes/ep_0001_20260303-144543
```

### 3. Visualize deploy simulation (3D trajectory)
```bash
# Interactive plot
python scripts/visualizations/plot_deploy_episode.py results/deploy/uav3d_sim/episodes/ep_0001_20260303-144543

# Save trajectory_3d.png to episode dir
python scripts/visualizations/plot_deploy_episode.py results/deploy/uav3d_mbd_sim/episodes/ep_0001_* --save --no-show
```

### 4. MuJoCo GIF (true 3D simulation render)
```bash
# UAV: Render episode as MuJoCo simulation GIF
python scripts/visualizations/render_deploy_mujoco_gif.py results/deploy/uav3d_mbd_sim/episodes/ep_0001_*
# Output: trajectory_mujoco.gif in episode dir

# Quadruped: Render with auto model (Ant for flat, Go2 when available)
python scripts/visualizations/render_deploy_quadruped_gif.py results/deploy/quadruped_go2_mbd_sim --episodes 3 --model auto

# Quick one-shot: deploy + render (uses mbd_deploy_quick.yaml with flat/Ant)
python scripts/tasks/robot/run_quadruped_quick.py
```

### 4.1 Unified motion replay (shared by deploy + experiments)
```bash
# From deploy episode dir -> GIF + HTML
python scripts/visualizations/render_motion_web.py \
  --episode-dir results/deploy/quadruped_go2_mbd_sim/episodes/ep_0001_* \
  --name quadruped_deploy

# From experiments seed dir (uses trajectory/trajectory.json best_idx) -> GIF + HTML
python scripts/visualizations/render_motion_web.py \
  --seed-dir results/quadruped/flat/mbd_plan/level_0/seed_0 \
  --name quadruped_plan
```

### 5. Generate unified report
```bash
python -m genedynamics.reports --input results --output reports
# Open reports/report.html (includes deploy section)
```

### 6. Expected validation criteria
- **UAV**: End position within 0.35 of target (0, 0, 1); z should not go negative (crash)
- **Quadruped**: End position within 0.35 of target (2, 0, 0.5)
