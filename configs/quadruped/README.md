# Quadruped configs

Structure: `{task}/{method}_plan.yaml` | `{method}_deploy.yaml`

Uses `scheduler_config` (diffusion_schedulers, constraint_schedulers) aligned with single_2d.

## Tasks

| Task | Plan | Deploy |
|------|------|--------|
| **flat** | mbd, mdcoas, mbd_plan_go2 | mbd, mdcoas (+ _quick) |
| **obstacle_avoid** | - | mdcoas (+ _quick) |
| **rough_terrain** | - | mbd (+ _quick) |
| **push_recovery** | - | mbd (+ _quick) |

## Plan: Extend trajectory (horizon)

Trajectory length = `horizon * dt`. Increase `env_params.horizon` to lengthen:

```yaml
env_params:
  horizon: 64   # 64*0.05=3.2s (default 48=2.4s)
  # or 80, 96 for longer paths
```

## Plan: Go2 model (real Unitree Go2)

```bash
# Requires MUJOCO_MENAGERIE_PATH or mujoco-menagerie
python -m genedynamics.experiments.runner configs/quadruped/flat/mbd_plan_go2.yaml
```

Output: `results/quadruped/flat/mbd_plan_go2/level_*/seed_0/trajectory_3d.png`

Render as MuJoCo GIF (when episode data available):
```bash
python scripts/visualizations/render_deploy_quadruped_gif.py results/... --model go2
```

## Usage

```bash
# Plan (experiment runner)
python -m genedynamics.experiments.runner configs/quadruped/flat/mbd_plan.yaml
python -m genedynamics.experiments.runner configs/quadruped/flat/mdcoas_plan.yaml
python -m genedynamics.experiments.runner configs/quadruped/flat/mbd_plan_go2.yaml

# Deploy
genedynamics-deploy --config configs/quadruped/flat/mbd_deploy.yaml
genedynamics-deploy --config configs/quadruped/flat/mbd_deploy_quick.yaml
genedynamics-deploy --config configs/quadruped/flat/mdcoas_deploy_quick.yaml
genedynamics-deploy --config configs/quadruped/obstacle_avoid/mdcoas_deploy_quick.yaml
genedynamics-deploy --config configs/quadruped/rough_terrain/mbd_deploy_quick.yaml
genedynamics-deploy --config configs/quadruped/push_recovery/mbd_deploy_quick.yaml

# Env configs (sim/real/shadow)
genedynamics-deploy --config configs/quadruped/env/shadow.yaml
genedynamics-deploy --config configs/quadruped/env/real.yaml --mode real
```
