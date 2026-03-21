# Experiment 1 — Part A (sanity)

Object-centric NeRF Synthetic `lego` with a fixed protocol:

- **MBD iid**: `lego_sanity_mbd_iid.yaml` → `run_full_experiment.py`
- **MBD corr**: `lego_sanity_mbd_corr.yaml` → `run_full_experiment.py`
- **gsplat MAP**: `lego_sanity_gsplat.yaml` → `run_baseline_experiment.py` or `train_gsplat.py` directly

Perturbations are **off** here. Stress tests live under `configs/3dgs/stress_tests/` and `run_stress_experiment.py`.

Environment knobs (pose bias / exposure) are documented in `configs/3dgs/_base_nerf_synth.yaml`.
