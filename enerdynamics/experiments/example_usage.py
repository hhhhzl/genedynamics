"""
Example usage of the unified experiment framework.

This script demonstrates how to use the new experiment framework to run
experiments with different methods, environments, and configurations.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enerdynamics.experiments.framework import ExperimentRunner, ExperimentConfig
from enerdynamics.experiments.plugins import (
    EDOCMethodPlugin,
    SingleIntegrator2DPlugin,
    DoubleIntegrator2DPlugin,
    SSRMetricsPlugin,
    ObstacleDensityMetricsPlugin,
    NonconvexityMetricsPlugin,
    TrajectoryVisualizationPlugin,
    DiffusionVisualizationPlugin,
    EnergyRewardVisualizationPlugin,
    StatesVisualizationPlugin,
    Box2DObstacleGeneratorPlugin,
)


def run_experiment_example():
    """Example: Run experiment using YAML configuration."""
    
    # Load configuration from YAML file
    # Get project root directory (go up from experiments/example_usage.py)
    project_root = Path(__file__).parent.parent.parent.parent
    config_path = project_root / "configs" / "double_integrator_box_2d" / "edoc_default.yaml"
    config = ExperimentConfig.from_yaml(config_path)
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Register plugins
    runner.register_plugin(EDOCMethodPlugin(), 'method')
    runner.register_plugin(DoubleIntegrator2DPlugin(), 'environment')
    runner.register_plugin(SSRMetricsPlugin(), 'metric')
    runner.register_plugin(ObstacleDensityMetricsPlugin(), 'metric')
    runner.register_plugin(NonconvexityMetricsPlugin(), 'metric')
    runner.register_plugin(TrajectoryVisualizationPlugin(), 'visualization')
    runner.register_plugin(DiffusionVisualizationPlugin(), 'visualization')
    runner.register_plugin(EnergyRewardVisualizationPlugin(), 'visualization')
    runner.register_plugin(StatesVisualizationPlugin(), 'visualization')
    runner.register_plugin(Box2DObstacleGeneratorPlugin(), 'obstacle_generator')
    
    # Run all experiments
    results = runner.run_all()
    
    print(f"\nCompleted {len(results)} experiments")
    return results


def run_experiment_programmatic():
    """Example: Run experiment with programmatic configuration."""
    
    # Create configuration programmatically
    config = ExperimentConfig(
        name="test_experiment",
        output_dir=Path("results/test"),
        env_name="single_integrator_box_2d",
        env_params={
            'dt': 0.05,
            'horizon': 80,
            'p_max': 2.0,
        },
        method="edoc",
        method_params={
            'action_diffuse_steps': 50,
            'action_nsample': 128,
        },
        obstacle_levels=[0, 1],
        obstacle_config={
            'generator': 'box2d',
            'robot_radius': 0.05,
            'map_bounds': {
                'x_min': -1.5,
                'x_max': 1.0,
                'y_min': -2.0,
                'y_max': 0.5,
            },
        },
        seeds=[0, 1],
        backend="jax",
        device="cpu",
        metrics=["ssr"],
        visualizations=["trajectory"],
        constraint_config={
            'action_constraint_type': 'speed',
        },
    )
    
    # Create runner and register plugins
    runner = ExperimentRunner(config)
    runner.register_plugin(EDOCMethodPlugin(), 'method')
    runner.register_plugin(SingleIntegrator2DPlugin(), 'environment')
    runner.register_plugin(SSRMetricsPlugin(), 'metric')
    runner.register_plugin(TrajectoryVisualizationPlugin(), 'visualization')
    runner.register_plugin(Box2DObstacleGeneratorPlugin(), 'obstacle_generator')
    
    # Run single experiment
    result = runner.run_single_experiment(level=1, seed=0)
    
    print(f"\nExperiment completed: SSR={result['metrics'].get('ssr', {}).get('ssr', 'N/A')}")
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Example usage of experiment framework")
    parser.add_argument("--mode", type=str, default="example", 
                       choices=["example", "programmatic"],
                       help="Run mode: example (uses YAML config) or programmatic (uses code config)")
    
    args = parser.parse_args()
    
    if args.mode == "example":
        run_experiment_example()
    else:
        run_experiment_programmatic()

