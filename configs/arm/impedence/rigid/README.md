# Arm surface-scan · impedance control · RIGID surface

Experiment configs for the Franka-Panda contact-rich surface-scan comparison
(idea.txt Exp I). A 7-DoF Panda scans a parametric surface under the impedance
primitive `u = (Δξ, Δη, Δψ, S_h, F_n)`, on a **rigid** mjx contact surface
(soft-body deformation is a later tier — RIGID first).

## Layout — subdivided by environment

```
rigid/
  _base.yaml                       # shared task / seeds / horizon / sampling budget
  <environment>/                   # one surface family per environment
    main/      mdac.yaml
    baseline/  dial.yaml  mppi.yaml
    ablation/  mdac_no_stiffness.yaml  mdac_fixed_stiffness.yaml
               mdac_euclid_stiffness.yaml  mdac_no_softfeas.yaml
               mdac_no_tangent.yaml  mdac_no_retraction.yaml  mdac_no_anneal.yaml
```

**Environments** (surface families): `plane`, `cylinder` (S1 analytic), `convex`
(S2 NURBS), `bumpy` (S3 NURBS), `unseen` (S4 NURBS + domain randomization).

**Roles**: `main` = `mdac` (the full method); `baseline` = `dial` / `mppi`;
`ablation` = the seven single-component knockouts.

Each method yaml sets only its `level` (environment) + `method` + `output_dir` and
inherits `_base.yaml` via `base: ../../_base.yaml`, so **every** method in **every**
environment runs at an **identical** sampling budget (`assert_fair`) — a fair
comparison differs only by the method's component flags.

Each ablation toggles **exactly one** MDAC component off relative to `mdac`
(`method_registry.assert_single_flag_ablation`). Two ablations are intentionally
**absent**: `mdac_no_rl_prior` (this run uses `prior=None`, so it is byte-identical
to `mdac`) and `mdac_no_mb_rollout` (inert by design — the parallel kernel *is* the
model-based rollout). `stiffness_mode` is derived from the method (none / log_spd /
euclid / fixed); do not set it in `env_params`.

## Run

`solvers/single/mdac/run_experiment.py` walks this tree, runs every
environment × method × seed at the shared budget, and writes per-method records to
each config's `output_dir`, which **mirrors the config path 1:1**:
`configs/arm/impedence/rigid/<env>/<role>/<method>.yaml`
→ `results/arm/impedence/rigid/<env>/<role>/<method>/metrics.json`. The render/plot
scripts (`scripts/tasks/robot/arm/`) write per-environment alongside it
(`<env>/arm_scan.gif`, `<env>/arm_render.mp4`, …). Needs real brax/mjx → run in
docker (`genedynamics/dev-cpu:torch`):

```bash
docker run --rm -v $(pwd):/workspace -w /workspace --user $(id -u):$(id -g) \
  -e HOME=/tmp -e MUJOCO_GL=egl -e PYTHONPATH=/workspace genedynamics/dev-cpu:torch \
  bash -lc "pip install -q 'setuptools<81' jax_cosmo >/dev/null 2>&1; \
            python -m genedynamics.solvers.single.mdac.run_experiment"
```

Point at a different config root with `MDAC_CONFIG_DIR=<dir>`.
