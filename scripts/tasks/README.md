# Task Scripts

Execution scripts organized by task domain.

## 3dgs/

3D Gaussian Splatting experiments.

| Script | Purpose |
|--------|---------|
| `run_full_experiment.py` | Full experiment: multi-seed posterior + metrics + figures |
| `run_all.sh` | Batch run lego+chair iid/corr |
| `run_3dgs_map.sh` | 3DGS-MAP baseline |
| `run_mbd3d_iid.sh` | Ours-MBD (iid) |
| `run_mbd3d_corr.sh` | Ours-MBD (corr) |
| `run_baseline_experiment.py` | gsplat baseline |
| `run_quality_recovery.sh` | Quality recovery pipeline |
| `train_gsplat.py` | Train gsplat |
| `eval_3dgs_metrics.py` | PSNR/LPIPS/NLL |
| `export_3dgs_figures.py` | Export figures |
| `render_uncertainty_panel.py` | Uncertainty heatmap |

## robot/

Stepping-stones plan → governor → walker. One entry point with subcommands:
`quadruped/stepping_tones/run_stepping_execution.py`.

| Subcommand | Purpose |
|------------|---------|
| `exec --config <yaml>` | Run a deploy config (or `--seed-dir`) through governor+walker; write full `res` (qpos/qvel/ctrl + summary + governed step-stats + executed footholds + ranking) and a GIF |
| `eval --methods ... --mode both` | Planner-SSR vs execution-SSR across methods/seeds |
| `validate [--no-sim]` | Staged validation: flat-straight → straight-stones → 2GO → baselines |
| `viz --res-dir <…/governed>` | Paper figures from an exec result: front-view ghosted motion strip + execution-vs-reference tracking + gait contact/speed diagram (`exec --figures` emits them automatically) |

Needs native MuJoCo for the sim parts — run inside `genedynamics/dev-cpu:torch` (arm64).
Deploy configs: `configs/quadruped/stepping_stones_2d/deploy/*.yaml`.

## soft_robot/

MRMFMBD soft-robot co-design (jax_mpm backend). Organized under `co_design/`:

### co_design/main/
| Script | Purpose |
|--------|---------|
| `run_co_design.py` | Co-design from YAML config |
| `run_co_design_with_gif.py` | Co-design + render best-theta as GIF |
| `run_all_experiments.sh` | Sequential run of all 7 main/ablation/baseline configs |

### co_design/analysis/
| Script | Purpose |
|--------|---------|
| `analyze_main_result.py` | Analyze main run (morphology + reward + cross-mode) |
| `analyze_baseline_run.py` | Analyze CMA-ES / CEM baseline runs |
| `analyze_diffusion_run.py` | Diffusion evolution figures + cross-mode |
| `compare_4_methods.py` | Reward curves: MRMFMBD vs ablation vs CMA-ES vs CEM |
| `compare_reward_curves.py` | Generic reward-curve comparison |
| `compare_fidelity_ablation.py` | Fidelity-ladder ablation comparison |

### co_design/viz/
| Script | Purpose |
|--------|---------|
| `regenerate_all_visuals.sh` | Re-render every GIF + figure (no retraining) |

## mdcoas/

MD-COAS experiments.

| Script | Purpose |
|--------|---------|
| `run_mdcoas.sh` | All MDCOAS experiments/baselines |
