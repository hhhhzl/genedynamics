# Experiment configuration

Every published run begins with YAML loaded into `ExperimentConfig`.

```yaml
name: single2d_mbd_demo
output_dir: results/single2d/mbd_demo
env_name: single_integrator_box_2d
method: mbd
backend: jax
device: cpu
seeds: [0]
n_steps: 64

env_params:
  dt: 0.05
  horizon: 64

method_params:
  Nsample: 64
  Ndiffuse: 20

obstacle_levels: [0]
obstacle_config:
  generator: box2d

metrics: [ssr]
visualizations: [trajectory]
```

## Inheritance

Use a relative `base:` path for task families:

```yaml
base: ../_base.yaml
name: mga_ablation_no_retraction
method_params:
  controller_method: mga_no_retraction
```

Nested dictionaries merge. Lists and scalar values replace the base value.
Cyclic or missing bases fail during loading.

## Formal and development outputs

Use `--development-root` for probes, local changes, or incomplete protocols.
Formal configurations may declare a frozen protocol and required checkpoint
hashes. The runner blocks formal execution when those readiness conditions are
not satisfied.

## Matrix selection

- `--seed N` selects one seed.
- `--seeds N ...` selects several seeds.
- `--level N` selects one obstacle level.
- `--suite NAME` and `--suites NAME ...` select named task/physics conditions.
- `--device cpu|gpu|cuda` overrides the configuration for that invocation.
- `--resume` executes only missing or configuration-mismatched entries.

Validate the full catalogue after changing any config:

```bash
python scripts/release/validate_release.py
```
