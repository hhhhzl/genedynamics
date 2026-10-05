# Your first planner

Start with planar navigation: it runs on CPU with the core package and needs no
robot simulator. Follow the [installation guide](installation.md) first, then
run from the repository root.

## Generate a trajectory

```bash
genedynamics-run configs/single_2d/mbd.yaml \
  --device cpu \
  --seed 0 \
  --level 0 \
  --development-root results/_development/quickstart
```

The run saves its resolved configuration, per-seed `results.json`, trajectory,
metrics, status, and figures under `results/_development/quickstart`. Open the
trajectory visualization to inspect the motion and use the saved metrics to
compare runs. The development root gives this example its own result tree.

## Try another planner

Keep the task and runner; switch the configuration to EB-MBD:

```bash
genedynamics-run configs/single_2d/ebmbd.yaml \
  --device cpu --seed 0 --level 0 \
  --development-root results/_development/quickstart
```

Browse [task recipes](../recipes/index.md) for constrained planning, robot tasks,
and learned planners. Their setup requirements are listed alongside each recipe.

## Resume a matrix

```bash
genedynamics-run configs/single_2d/mbd.yaml \
  --device cpu \
  --seeds 0 1 2 \
  --development-root results/_development/quickstart \
  --resume
```

`--resume` reuses complete entries whose configuration identity still matches
and executes missing or mismatched entries.

## Inspect configuration before a run

```bash
genedynamics-run configs/single_2d/mbd.yaml \
  --device cpu --seed 0 --level 0 --dry-run
```

This resolves and validates the selected config without allocating a planner
run. The [configuration guide](../guides/configuration.md) explains inheritance,
task matrices, and plugin selection.

Contributors can validate the entire release catalogue with
`python scripts/release/validate_release.py`; see
[testing](../development/testing.md) for the release checks.
