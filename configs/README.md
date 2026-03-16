# Experiment Configuration Files

This directory contains YAML configuration files for running experiments with the unified experiment framework.

## Configuration File Format

Configuration files use YAML format and define all parameters needed to run experiments:
- Environment settings (type, parameters)
- Method settings (solver, parameters)
- Obstacle configuration (generator, levels, parameters)
- Experiment execution (seeds, backend, device)
- Metrics and visualizations
- Constraint configuration

## Available Configurations

### Double Integrator 2D
- `double_integrator_box_2d/edoc_default.yaml`: Default EDOC configuration

### Single Integrator 2D
- `single_integrator_box_2d/edoc_default.yaml`: Default EDOC configuration

### MRMFMBD (Soft-robot S1+S3)
- `mrmfmbd/softzoo_flat.yaml`: SoftZoo caterpillar on flat terrain

## Usage

Run experiments from a configuration file:

```bash
python -m genedynamics.experiments.run_experiment_from_config configs/double_integrator_box_2d/edoc_default.yaml
```

Or with specific level/seed:

```bash
python -m genedynamics.experiments.run_experiment_from_config configs/double_integrator_box_2d/edoc_default.yaml --level 5 --seed 0
```

Validate configuration without running:

```bash
python -m genedynamics.experiments.run_experiment_from_config configs/double_integrator_box_2d/edoc_default.yaml --dry-run
```

## Creating Custom Configurations

1. Copy an existing configuration file as a template
2. Modify parameters as needed
3. Ensure all required fields are present (use `--dry-run` to validate)
4. Run with the new configuration file

## Configuration Schema

See `genedynamics/experiments/framework/config.py` for the full `ExperimentConfig` dataclass definition.

