# Quadruped Stepping Stones 2D

This task family models quadruped stepping as paired 2D footholds:

- State: `[pL_x, pL_y, pR_x, pR_y]`
- Action: `[dL_x, dL_y, dR_x, dR_y]`
- Goal: mid-foot reaches target while staying on stepping stones.

## Layout

- `main/`: primary experiment configs (same task distribution, different methods)
- `ablations/`: 2GO ablations on representative levels
- `smoke/`: fast validation runs

Difficulty is controlled by `obstacle_levels` and interpreted by the
`stepping_stones_2d` generator:

- `1-2`: easy
- `3-4`: medium
- `5-6`: hard

## Run

```bash
python -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/main/mbd.yaml
python -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/main/mdoc.yaml
python -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/main/mdcoas.yaml
python -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/main/2go.yaml
```

## Ablations

```bash
python -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/ablations/2go_no_gate.yaml
python -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/ablations/2go_no_tail.yaml
python -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/ablations/2go_no_probe.yaml
python -m genedynamics.experiments.runner configs/quadruped/stepping_stones_2d/ablations/2go_no_retract.yaml
```

