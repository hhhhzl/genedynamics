# Quadruped configs

Structure: `{task}/{method}_plan.yaml` | `{method}_deploy.yaml`

## Tasks

| Task | Plan | Deploy |
|------|------|--------|
| **flat** | mbd, mdcoas | mbd, mdcoas (+ _quick) |
| **obstacle_avoid** | - | mdcoas (+ _quick) |
| **rough_terrain** | - | mbd (+ _quick) |
| **push_recovery** | - | mbd (+ _quick) |

## Usage

```bash
# Plan (experiment runner)
python -m genedynamics.experiments.runner configs/quadruped/flat/mbd_plan.yaml
python -m genedynamics.experiments.runner configs/quadruped/flat/mdcoas_plan.yaml

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
