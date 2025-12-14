# EDOC Experiment on Double Integrator 2D with Obstacles

This experiment tests EDOC (Energy-Guided Diffusion with Plug-and-Play Soft/Hard Constraints) on a 2D double integrator environment with obstacles.

## Experiment Setup

### Obstacle Levels (10 levels)

1. **Levels 1-3**: Convex obstacles only (low, medium, high density)
   - Level 1: 2 obstacles
   - Level 2: 5 obstacles
   - Level 3: 8 obstacles

2. **Levels 4-6**: Mixed convex obstacles (low, medium, high density)
   - Level 4: 3 obstacles (varied types)
   - Level 5: 6 obstacles
   - Level 6: 10 obstacles

3. **Levels 7-10**: Non-convex unions (increasing complexity)
   - Level 7: 2 unions, 2 primitives each
   - Level 8: 3 unions, 3 primitives each
   - Level 9: 4 unions, 4 primitives each
   - Level 10: 5 unions, 5 primitives each

### Constraints

1. **Soft Constraints**: Barrier energy for obstacle avoidance
   - S(τ) = Σ_h Σ_m α * exp(-β * sdf(x^h))
   - Scheduled: alpha decreases from 1.0 → 0.0 during diffusion

2. **Hard Constraints**:
   - Obstacle collision avoidance (clearance ≥ 0.1)
   - Acceleration bounds (|u| ≤ 1.0)
   - Scheduled: clearance relaxes from 0.5 → 0.1 during diffusion

3. **Feasibility Operator**: CFS-QP projection for non-convex obstacles
   - Late-stage application (last 30% of diffusion steps)

### Metrics

- **SSR (Safety Success Rate)**: Fraction of trajectories that are:
  - Safe (no collisions)
  - Feasible (satisfies acceleration constraints)
  - Successful (reaches target within 0.1 distance)

## Usage

### Run All Experiments

```bash
python enerdynamics/experiments/double2d/run_experiment.py \
    --output_dir results/double2d \
    --num_seeds 10
```

This will:
- Run 10 levels × 10 seeds = 100 experiments
- Save results to `results/double2d/`
- Generate visualizations for each experiment
- Compute SSR for each level and overall

### Run Single Experiment

```bash
python enerdynamics/experiments/double2d/run_experiment.py \
    --output_dir results/double2d \
    --level 1 \
    --seed 0
```

### Output Structure

```
results/double2d/
├── level_1/
│   ├── seed_0/
│   │   ├── trajectory.png      # Final trajectory visualization
│   │   ├── energy_reward.png   # Energy and reward over time
│   │   ├── states.png          # State components over time
│   │   └── results.json        # Detailed results
│   ├── seed_1/
│   │   └── ...
│   └── summary.json            # Level summary (SSR)
├── level_2/
│   └── ...
├── ssr_by_level.png            # SSR bar chart
└── overall_summary.json        # Overall statistics
```

## Visualization

Each experiment generates:

1. **Trajectory Plot**: Shows final trajectory, obstacles, start/end positions, target
2. **Energy/Reward Plot**: Energy and reward over timesteps
3. **States Plot**: Position and velocity components over time
4. **Diffusion Steps** (planned): Rollouts at 90%, 50%, 10% diffusion progress

## Results

The `results.json` file contains:

```json
{
  "level": 1,
  "seed": 0,
  "ssr": 1.0,
  "safe": true,
  "accel_feasible": true,
  "task_success": true,
  "distance_to_target": 0.05,
  "trajectory": {
    "states": [...],
    "actions": [...]
  },
  "energies": [...],
  "rewards": [...]
}
```

The `summary.json` files contain aggregated statistics per level.
