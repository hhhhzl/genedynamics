### DPCC in `enerdynamics`

This directory contains an integration of **DPCC (Diffusion Predictive Control with Constraints)** into the `enerdynamics` framework.

DPCC paper/repo context (original codebase in this workspace): `dpcc/` (L4DC 2025).

---

### 1) Original DPCC pipeline (upstream `dpcc/` repo)

DPCC has two main entry points:

- **Training**: `dpcc/scripts/train.py`
  - Trains a diffusion policy (here we focus on `GaussianDiffusion`).
  - Saves checkpoints and logs under the DPCC logging scheme (typically under `logs/`).

- **Evaluation**: `dpcc/scripts/eval.py`
  - Loads a trained diffusion model via DPCC’s loader utilities.
  - Constructs constraints (halfspace/bounds/obstacles + dynamics `deriv`) and a `Projector`.
  - Runs closed-loop rollouts in the D3IL Avoiding environment, reporting success/violations/time and saving plots/results.

Key DPCC runtime components:

- **`GaussianDiffusion` sampling loop**: calls a **`Projector`** during denoising (projection mode) or adds gradient corrections (gradient mode).
- **`Projector`**: solves a constrained projection (MVP uses SciPy/SLSQP).
- **`Policy`**: wraps the diffusion model and supports trajectory selection:
  - `random`
  - `temporal_consistency` (dpcc-t)
  - `minimum_projection_cost` (dpcc-c)

---

### 2) How DPCC is built inside `enerdynamics`

We keep the integration modular and backend-extensible:

- **Solver wrapper**
  - `dpcc.py`: `DPCCSolver` (a `SamplingSolver`) delegates to a backend planner.

- **Backend implementation**
  - `backends/dpcc_plan_torch.py`: `DPCCBackendTorch`
  - Runs a closed-loop rollout in D3IL Avoiding by calling the DPCC-style `Policy`.
  - Implements DPCC `eval.py`-aligned behavior:
    - parses `variant` → `gradient`, `dt*` multipliers, `model_free` + `tightened` constraint selection, and disables the projector for `diffuser`
    - computes `n_violations`, `total_violations`, `collision_free_completed`, `pos_tracking_errors` exactly like `dpcc/scripts/eval.py`
    - supports `n_trials` and returns DPCC-style per-trial arrays in `Trajectory.info`

- **DPCC “patch” layer (kept local to this solver)**
  - `patch/diffusion.py`: patched `GaussianDiffusion` that supports `projector`/`constraints` hooks during sampling.
  - `patch/projector.py`: DPCC `Projector` (MVP: SciPy solver).
  - `patch/policy.py`: DPCC `Policy` (trajectory selection + projector injection).
  - `patch/constraints_helpers.py`: DPCC constraint construction helpers (halfspace/bounds/dynamics deriv).
  - `patch/avoiding_adapter.py`: D3IL Avoiding observation/action + constraint building adapter.

This structure avoids creating a new standalone package per algorithm while still isolating DPCC-specific logic.

---

### 3) Experiments integration (D3IL Avoiding)

DPCC is wired into the experiment framework via a method plugin:

- `enerdynamics/experiments/plugins/methods/dpcc.py`: `DPCCMethodPlugin`
  - Resolves DPCC root from the vendored module (`enerdynamics/solvers/single/dpcc/config`) or nearby `../dpcc`.
  - Loads the trained diffusion model via `diffuser.utils.load_diffusion(...)`.
  - Creates a `DPCCSolver` for evaluation inside the standard experiment runner.

The environment plugin used is:

- `enerdynamics/experiments/plugins/environments/d3il_avoiding.py`: `D3ILAvoidingPlugin`

---

### 4) How to run D3IL Avoiding (4D and 9D)

All commands below are from repository root (`/home/lbw/mbd-project`).

#### 4.1 Train 4D DPCC diffusion

```bash
python scripts/dpcc_train.py \
  --config-file configs/d3il_avoiding/dpcc_diffusion_train.py \
  --dataset avoiding-d3il \
  --device cuda
```

Default training config: `dpcc/config/avoiding-d3il.py`.

#### 4.2 Train 9D DPCC diffusion (state `[x, y, q1..q7]`, action `[qdot1..qdot7]`)

```bash
python scripts/dpcc_train_9d.py \
  --config-file configs/d3il_avoiding/dpcc_diffusion_train.py \
  --dataset avoiding-d3il-9d \
  --device cuda
```

Default training config: `enerdynamics/configs/d3il_avoiding/dpcc_diffusion_train.py`.

#### 4.3 Eval 4D with experiment runner

```bash
python enerdynamics/experiments/runner.py \
  enerdynamics/configs/d3il_avoiding/dpcc_train_eval.yaml
```

This uses:
- env plugin: `d3il_avoiding` (4D)
- method plugin: `dpcc`
- diffusion checkpoint from `method_params.loadbase/dataset/diffusion_loadpath/seed`.

#### 4.4 Eval 9D with experiment runner

```bash
python enerdynamics/experiments/runner.py \
  enerdynamics/configs/d3il_avoiding/dpcc_train_eval_9d.yaml
```

#### 4.5 Eval 9D trained diffusion (quick rollout script)

Use the dedicated test script for 9D checkpoints:

```bash
python scripts/dpcc_test_diffusion.py \
  --env-name d3il_avoiding_9d \
  --dataset avoiding-d3il-9d \
  --exp avoiding-d3il-9d \
  --seed 5 \
  --device cuda \
  --diffusion-loadpath diffusion/H8_K20_Dmodels.GaussianDiffusion \
  --epoch best \
  --variant diffuser \
  --n-trials 5
```

Optional: save eval summary JSON

```bash
python scripts/dpcc_test_diffusion.py \
  --env-name d3il_avoiding_9d \
  --dataset avoiding-d3il-9d \
  --exp avoiding-d3il-9d \
  --seed 5 \
  --device cuda \
  --diffusion-loadpath diffusion/H8_K20_Dmodels.GaussianDiffusion \
  --epoch best \
  --save-json results/d3il_avoiding/dpcc_test_seed5.json
```

Notes for strict DPCC parity:

- `halfspace_variant`: set `method_params.halfspace_variant` to one of:
  - `top-left-hard`, `top-right-hard`, `both-hard`
  - If omitted, we default to the first entry in `dpcc/config/projection_eval.yaml`’s `avoiding_halfspace_variants`.
- `dt` / `enlarge_constraints` lookup: DPCC config keys are split:
  - `dt` and `enlarge_constraints` are keyed by `robot_name` (e.g. `avoiding`)
  - `halfspace_constraints`, `obstacle_constraints`, `bounds` are keyed by `exp` (e.g. `avoiding-d3il`)
- Avoiding reset parity: DPCC `eval.py` uses `env.reset()` (no per-trial seeding). We match that.
- `Trajectory` vs DPCC buffers:
  - DPCC’s `obs_buffer` does **not** include the initial state.
  - `enerdynamics` `Trajectory` requires `len(states) = len(actions) + 1`.
  - We return framework-valid `states` (with initial state prepended) and also expose DPCC-exact trial-0 buffer in `Trajectory.info["states_dpcc_trial0"]`.

---

### 5) What files are generated (training vs evaluation)

#### Training outputs
The train scripts write checkpoints/logs under `logs/`:
- 4D: `logs/avoiding-d3il/...`
- 9D: `logs/avoiding-d3il-9d/...`

Your eval config points to these fields:

- `method_params.loadbase`: base directory (default: `logs`)
- `method_params.diffusion_loadpath`: diffusion subdir (default: `diffusion`)
- `method_params.dataset`: dataset name (e.g. `avoiding-d3il` or `avoiding-d3il-9d`)
- `method_params.seed`: run seed subfolder used by the loader

#### Evaluation outputs (enerdynamics experiments)
Evaluation results are written to:

- `output_dir` in the YAML (default: `results/d3il_avoiding/dpcc`)

The experiment runner will write standard experiment JSON summaries and any configured visualizations/metrics outputs into that directory.

Additionally, DPCC-eval-aligned statistics are returned in `Trajectory.info`, including:

- per-trial arrays: `n_success`, `n_success_and_constraints`, `n_steps`, `n_violations`, `total_violations`, `avg_time_all`, `collision_free_completed`, `pos_tracking_errors`
- sampled rollouts: `sampled_trajectories_all`
- exact DPCC buffer for trial 0: `states_dpcc_trial0`

---

### 6) Artifact store (optional, for MLOps readiness)

We also provide an extendable artifact store implementation:

- `artifacts.py`: `DPCCArtifactStore`

It defines a stable on-disk schema with:

- `checkpoints/` (supports `best.pt` and `checkpoint_{step}.pt`)
- `configs/` (`train.yaml`, `plan.yaml`, `indices.yaml`)
- `normalizer.pkl`
- `meta.json` (schema version, timestamps, etc.)
- `manifest.json` (provenance: env versions, git sha, diffuser sha, etc.)
- `external_uris.json` (hooks for W&B / MLflow / S3 URIs)

Today, the DPCC training script is not yet wired to write into this store automatically; it exists so you can migrate toward a unified artifact/MLOps workflow without changing solver code.
