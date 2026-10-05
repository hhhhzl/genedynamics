# Planar CPU smoke

This is the smallest supported V1 recipe. It needs the core package only and
is the first command to run after installation.

## Validate without allocating a run

```bash
genedynamics-run configs/single_2d/mbd.yaml \
  --device cpu --seed 0 --level 0 --dry-run
```

The command must exit with status `0` and identify these resolved values:

```text
Configuration is valid (dry run mode)
Environment: single_integrator_box_2d
Method: mbd
Levels: [0]
Seeds: [0]
```

## Execute the smoke case

```bash
genedynamics-run configs/single_2d/mbd.yaml \
  --device cpu --seed 0 --level 0 \
  --development-root results/_development/v1-planar
```

The output contract is a resolved configuration plus the seed/level result,
trajectory, metrics, and execution status under
`results/_development/v1-planar`. Treat the command's nonzero status or an
aborted execution status as a failed smoke run; do not infer success from a
partially written directory.
