# Scripts

## Directory Structure

| Directory | Purpose |
|-----------|---------|
| **setup/** | Environment setup. Optional third-party: d3il, 3dgs |
| **tasks/** | Task-specific execution scripts by domain (3dgs, robot, soft_robot, mdcoas) |
| **debug/** | Debug/verification scripts. Safe to delete after use |
| **visualizations/** | Plotting, rendering, deploy analysis |
| **docker/** | Future Docker scripts (placeholder) |
| **baseline/** | Baseline regression (unchanged) |

---

## Setup

```bash
# Full setup (d3il + PyTorch + genedynamics)
./scripts/setup/setup.sh

# Optional: setup specific third-party env
./scripts/setup/setup.sh d3il
./scripts/setup/setup.sh 3dgs
./scripts/setup/setup.sh all
```

---

## Tasks

See `scripts/tasks/README.md` for task-specific scripts:
- **3dgs/** - 3D Gaussian Splatting experiments
- **robot/** - Quadruped, UAV, acceptance tests
- **soft_robot/** - MRMFMBD soft-robot co-design (jax_mpm)
- **mdcoas/** - MD-COAS experiments

---

## DPCC & SafeDiffuser (canonical paths)

DPCC/SafeDiffuser tools live under `genedynamics/solvers/single/`:

**Train 4D/9D, test diffusion:**
```bash
python -m genedynamics.solvers.single.dpcc.tools.train_9d --config-file configs/d3il_avoiding/dpcc_diffusion_train.py --dataset avoiding-d3il-9d
python -m genedynamics.solvers.single.dpcc.tools.test_diffusion --env-name d3il_avoiding_9d --dataset avoiding-d3il-9d --diffusion-loadpath diffusion/H8_K20_Dmodels.GaussianDiffusion --epoch best
```

**SafeDiffuser checkpoint conversion:**
```bash
python -m genedynamics.solvers.single.safediffuser.tools.convert_checkpoint --input-dir ... --output-dir ... --dataset-data-dir ...
```

See `genedynamics/solvers/single/dpcc/README.md` for full docs.

---

## Visualization

```bash
python scripts/visualizations/visualize_env_levels_seeds.py [--config CONFIG] [--outdir OUTDIR]
python scripts/visualizations/visualize_d3il_trajectory_all_seeds.py --method mdcoas
python scripts/visualizations/plot_deploy_episode.py results/deploy/...
python scripts/visualizations/render_deploy_quadruped_gif.py results/deploy/...
```
