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

Quadruped, UAV, acceptance tests.

| Script | Purpose |
|--------|---------|
| `run_quadruped_plan.sh` | MBD + MD-COAS + Go2 plans |
| `run_stepping_stones_baselines.sh` | Stepping-stones baselines (MBD/MDOC/MD-COAS) |
| `run_stepping_stones_main.sh` | Stepping-stones full main suite (MBD/MDOC/MD-COAS/2GO) |
| `run_stepping_stones_ablations.sh` | 2GO ablations on stepping-stones |
| `run_stepping_stones_smoke.sh` | Stepping-stones smoke checks |
| `summarize_stepping_stones.py` | Aggregate success/CVaR/time from results |
| `run_quadruped_quick.py` | One-shot deploy + render |
| `run_acceptance.sh` | Acceptance tests |
| `run_phase4_validation.sh` | Phase 4 closed-loop validation |

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
| `compare_s3_ablation.py` | S3 fidelity-ladder ablation comparison |
| `plot_co_design_results.py` | Plot co-design results from results.json |

### co_design/viz/
| Script | Purpose |
|--------|---------|
| `regenerate_all_visuals.sh` | Re-render every GIF + figure (no retraining) |

## mdcoas/

MD-COAS experiments.

| Script | Purpose |
|--------|---------|
| `run_mdcoas.sh` | All MDCOAS experiments/baselines |
