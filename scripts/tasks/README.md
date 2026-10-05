# Task Scripts

Execution scripts organized by task domain.

## robot/

Stepping-stones plan → governor → walker. One entry point with subcommands:
`quadruped/stepping_tones/run_stepping_execution.py`.

| Subcommand | Purpose |
|------------|---------|
| `exec --config <yaml>` | Run a deploy config (or `--seed-dir`) through governor+walker; write full `res` (qpos/qvel/ctrl + summary + governed step-stats + executed footholds + ranking) and a GIF |
| `eval --methods ... --mode both` | Planner-SSR vs execution-SSR across methods/seeds |
| `validate [--no-sim]` | Staged validation: flat-straight → straight-stones → 2GO → baselines |
| `viz --res-dir <…/governed>` | Paper figures from an exec result: front-view ghosted motion strip + execution-vs-reference tracking + gait contact/speed diagram (`exec --figures` emits them automatically) |

Needs native MuJoCo for the sim parts — run inside `genedynamics/dev-cpu:torch` (arm64).
Deploy configs: `configs/quadruped/stepping_stones_2d/deploy/*.yaml`.

### arm/

The formal contact-control experiments are orchestrated from
`scripts/paper/mga/`; task directories retain training utilities only.

| Script | Purpose |
|--------|---------|
| `train_rl_baseline.py --config <formal-base.yaml>` | Train the shared PPO or ATACOM policy from the frozen `metadata.training.rl` contract |
| `train_mga_reliability.py` | Fit the frozen reliability model from development-only result splits |

Formal configs live in `configs/arm/{surface_scan,peg_insert}` and run through
`python -m genedynamics.experiments.runner`. Saved trajectories are rendered by
`python -m genedynamics.experiments.utils.vis`.

## mdcoas/

MD-COAS experiments.

| Script | Purpose |
|--------|---------|
| `run_mdcoas.sh` | All MDCOAS experiments/baselines |
