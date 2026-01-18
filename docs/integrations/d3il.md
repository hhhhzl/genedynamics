## D3IL integration (vendor / submodule)

### Why we keep the full D3IL repo

D3IL tasks (avoiding / pushing / sorting / stacking / aligning / inserting / …) share:
- `d3il_sim/` (scene, robot, controllers, MuJoCo backend)
- `models/` assets (XML, meshes, textures)

For long-term multi-task support, **do not “trim” D3IL down to only avoiding**. Missing assets will break other tasks later, often with late/opaque runtime errors.

### Repository layout (recommended)

- **Pinned third-party code** lives under `third_party/`.
- D3IL is expected to be importable as **`environments.d3il.*`**.

Current vendored layout:

```
third_party/
  environments/
    __init__.py
    d3il/
      __init__.py
      d3il_sim/
      envs/
      models/
      ...
```

This matches D3IL’s internal imports like:
- `from environments.d3il.d3il_sim... import ...`

### Alternative: full upstream repo

If you want **D3IL agents/configs/simulation scripts** (not only the MuJoCo envs),
vendor/pin the full upstream repo (e.g. as a submodule from
[`ALRhub/d3il`](https://github.com/ALRhub/d3il#)) and use this layout:

```
third_party/
  d3il/                # full upstream repo
    environments/
      d3il/
        d3il_sim/
        envs/
        models/
```

Our bootstrapping helper supports **both** layouts.

## TaskSpec pattern (recommended for multi-task scaling)

For each D3IL task, we recommend implementing a **spec** instead of duplicating wrapper code.

- **Spec**: task-specific glue (obs/state definition, action formatting, success/collision parsing)
- **Generic wrapper**: `D3ILTaskEnv(spec)` handles lifecycle (`start/reset/step/close`)

To add a new task (e.g. pushing):
- Create `D3ILPushingSpec` next to `D3ILAvoidingSpec`
- Create an EnvironmentPlugin `D3ILPushingPlugin` that instantiates `D3ILTaskEnv(D3ILPushingSpec(...))`

This keeps task extensions isolated and avoids copy/paste bugs.

### Python import path

Because D3IL imports are rooted at `environments.*`, the directory **containing** the `environments/` package must be on `sys.path` / `PYTHONPATH`:

- Add `<repo_root>/third_party` to `PYTHONPATH`, or
- Call `enerdynamics.envs.external.d3il.ensure_d3il_on_path()` early (preferred inside adapters).

### MuJoCo assets / path resolution

D3IL uses its own path helper (`d3il_sim/utils/sim_path.py`) to locate XML and meshes relative to the package location.

So the two key rules are:
- Keep D3IL’s directory structure intact under `third_party/environments/d3il/`
- Ensure imports resolve as `environments.d3il` (so D3IL can locate `models/` reliably)

### Submodule recommendation (future improvement)

For reproducibility, replace the vendored copy with a **git submodule** pinned to a specific commit:
- `third_party/environments/d3il` (or a repo that contains `environments/d3il`)

This keeps `enerdynamics` clean while allowing controlled upgrades.

## A → E summary (what we changed) and how to run

This section documents the current integration status, end-to-end:

- **A**: Third-party wiring (D3IL vendored/submodule + import bootstrap)
- **B**: `d3il_avoiding` env wrapper + experiment env plugin
- **C**: `edoc_mpc` / `mbd_mpc` (plan on JAX model, execute on MuJoCo)
- **D**: Standard outputs + minimal metrics for rollout baselines
- **E3**: TaskSpec abstraction for scaling to other D3IL tasks

### A — Third-party wiring

- **Vendored D3IL env stack**: `third_party/environments/d3il/`
- **`environments.*` package root**: `third_party/environments/__init__.py`
- **Bootstrap**: `enerdynamics/envs/external/d3il/bootstrap.py` (`ensure_d3il_on_path()` supports:
  - `third_party/environments/d3il` (vendored subtree)
  - `third_party/d3il/environments/d3il` (full upstream repo)
  )
- **Setup helpers**:
  - `scripts/setup_d3il.sh` (prints export commands)
  - `scripts/setup.sh` (points to setup_d3il.sh)
- **Dependencies**:
  - `requirements.txt` includes `gym>=0.26.2` and `gin-config>=0.5.0`
- **Packaging fix**:
  - `setup.py` now reads `requirements.txt` into `install_requires`

### B — D3IL Avoiding env integration

- **Wrapper env**: `enerdynamics/envs/external/d3il/avoiding_env.py`
- **Env plugin**: `enerdynamics/experiments/plugins/environments/d3il_avoiding.py`
  - `env_name: d3il_avoiding`
- **State/action convention (avoiding)**:
  - `state = [x_des, y_des, x, y]`
  - `action = [dx, dy]`

### C — EDOC/MBD on D3IL Avoiding (MPC)

- **Planning model**: `enerdynamics/envs/external/d3il/avoiding_plan_env.py` (`AvoidingPlanEnv`)
- **MPC executor**: `enerdynamics/experiments/common/d3il_mpc.py`
- **Method plugins**:
  - `method: edoc_mpc` → `enerdynamics/experiments/plugins/methods/edoc_mpc.py`
  - `method: mbd_mpc` → `enerdynamics/experiments/plugins/methods/mbd_mpc.py`

### D — Standard outputs + minimal metrics/viz

- **`results.json` (rollout methods)** now includes standardized fields when available:
  - `exec_states`, `exec_actions`
  - `success`, `collision`
  - `planning_time_per_step`, `avg_planning_time_per_step`
- **Metric plugin**:
  - `metric: episode_outcome` → `enerdynamics/experiments/plugins/metrics/episode_outcome.py`
- **SSR now supports non-standard state layout** via `env_plugin.extract_position(...)`

### E3 — TaskSpec abstraction (for future multi-task scaling)

- **Spec interface**: `enerdynamics/envs/external/d3il/specs/base.py`
- **Avoiding spec**: `enerdynamics/envs/external/d3il/specs/avoiding.py`
- **Generic wrapper**: `enerdynamics/envs/external/d3il/task_env.py` (`D3ILTaskEnv`)
- **Avoiding wrapper is spec-driven** (API unchanged): `D3ILAvoidingEnv` delegates to `D3ILTaskEnv(D3ILAvoidingSpec)`

## How to run

### 0) Install deps + set PYTHONPATH for D3IL

```bash
pip install -r requirements.txt

# show the PYTHONPATH export lines
./scripts/setup_d3il.sh

# vendored layout:
export PYTHONPATH="$(pwd)/third_party:${PYTHONPATH:-}"
```

Quick import check:

```bash
python3 -c "import environments.d3il; import environments.d3il.d3il_sim; print('OK')"
```

### 1) Offline/open-loop planning (no MuJoCo execution)

We added an offline planning environment:
- `env_name: avoiding_plan`

Example config:
- `configs/avoiding_plan/mbd.yaml`

Run:

```bash
python3 -m enerdynamics.experiments.run_experiment_from_config configs/avoiding_plan/mbd.yaml
```

### 2) MPC on real D3IL MuJoCo (plan in JAX model, execute in env)

Use:
- `env_name: d3il_avoiding`
- `method: edoc_mpc` or `mbd_mpc`

Minimal YAML fields:

```yaml
env_name: d3il_avoiding
env_params:
  render: true
method: edoc_mpc   # or mbd_mpc
method_params:
  horizon: 20
  max_episode_length: 150
  action_limit: 0.05
metrics:
  - episode_outcome
  - ssr
visualizations:
  - trajectory
```

### 3) Optional tests (auto-skip if deps missing)

```bash
pytest -q test/integration/test_d3il_avoiding_env_optional.py
pytest -q test/integration/test_d3il_avoiding_mpc_optional.py
```


