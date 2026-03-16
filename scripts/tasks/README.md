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
| `run_quadruped_quick.py` | One-shot deploy + render |
| `run_acceptance.sh` | Acceptance tests |
| `run_phase4_validation.sh` | Phase 4 closed-loop validation |

## soft_robot/

MRMFMBD, co-design, Phase A/B.

| Script | Purpose |
|--------|---------|
| `run_mrmfmbd.sh` | MRMFMBD SoftZoo experiments |
| `run_mrmfmbd_posterior.py` | MRMFMBD posterior bridge |
| `run_co_design.py` | Co-design from YAML config |
| `run_phase_a.sh` | Phase A smoke |
| `run_phase_b.sh` | Phase B S1 validation |
| `run_phase4_fair_comparison.sh` | Fair comparison MBD vs MD-COAS |
| `run_phase4_validation.py` | Phase 4 validation (Python) |

## mdcoas/

MD-COAS experiments.

| Script | Purpose |
|--------|---------|
| `run_mdcoas.sh` | All MDCOAS experiments/baselines |
