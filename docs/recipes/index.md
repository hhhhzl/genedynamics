# Algorithm recipes

Start with one of our four algorithms, then use the
[planner catalogue](../reference/planners.md) to choose comparisons. Each recipe
is a configuration you can inspect and adapt.

For a small simulator-free integration, run the
[direct Python example](../getting-started/python-api.md). The recipes below
use the experiment runner to save configurations, metrics and trajectories.

## MDOC

Planar constrained planning:

```bash
genedynamics-run configs/single_2d/mdoc.yaml \
  --device cpu --seed 0 --level 0 \
  --development-root results/_development/mdoc
```

The V1 configuration exposes MDOC's single-robot planning component. The
[original MDOC project](https://github.com/hhhhzl/mdoc) contains the paper's
multi-robot coordination application.

## MD-COAS

Start with `configs/single_2d/mdcoas.yaml` for constrained diffusion with
adaptive scheduling. Install the `optimization` extra for the QP integrations.

```bash
genedynamics-run configs/single_2d/mdcoas.yaml \
  --device cpu --seed 0 --level 0 \
  --development-root results/_development/mdcoas
```

Then explore `configs/d3il_avoiding/mdcoas.yaml` for seven-joint arm avoidance;
follow the [D3IL setup guide](../integrations/d3il.md) first. The historical
`cfsmbd_full` key in these files identifies MD-COAS, not a separate planner.

## 2GO

| Task | Starting configuration | Setup |
| --- | --- | --- |
| Quadruped stepping stones | `configs/quadruped/stepping_stones_2d/main/twogo.yaml` | Simulation + optimization extras |
| Humanoid corridor | `configs/humanoid/corridor_2d/main/twogo_zone_a.yaml` | Core + optimization for planning; deploy extras for execution |

The [humanoid corridor recipe](humanoid-corridor.md) explains a complete runner
command and its output contract. Configuration filenames use `twogo`; the
registered solver key is `2go`.

## MGA

| Task | Starting configuration | Setup |
| --- | --- | --- |
| Peg insertion | `configs/arm/peg_insert/main/mga.yaml` | Simulation extras; frozen learned assets for formal runs |
| Compliant surface scan | `configs/arm/surface_scan/main/mga.yaml` | Simulation extras; prior assets depend on the selected settings |

See [installation](../getting-started/installation.md) for simulator and asset
setup. V1 qualifies the CPU path; GPU qualification and humanoid walking/push
are [follow-up priorities](../reference/roadmap.md).

## Compare and scale

The [planner catalogue](../reference/planners.md) lists the baseline
configurations and their dependency requirements. For a runner introduction,
the [planar CPU smoke](planar-cpu.md) pins the CLI output and artifact contract.

Use `--dry-run` to inspect a configuration, then a development root and one
seed for an initial execution. Inspect status, metrics and trajectory artifacts
before scaling to the formal matrix. The [first-run guide](../getting-started/quickstart.md)
shows how to resume multi-seed runs.
