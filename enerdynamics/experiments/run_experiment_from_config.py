"""
Run experiment from YAML configuration file.

This script demonstrates how to use the new experiment framework
to run experiments from YAML configuration files.
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from enerdynamics.experiments.framework import ExperimentRunner, ExperimentConfig
from enerdynamics.experiments.plugins import (
    EDOCMethodPlugin,
    SingleIntegrator2DPlugin,
    DoubleIntegrator2DPlugin,
    DroneEnvironmentPlugin,
    ManipulatorEnvironmentPlugin,
    SSRMetricsPlugin,
    ObstacleDensityMetricsPlugin,
    NonconvexityMetricsPlugin,
    TrajectoryVisualizationPlugin,
    DiffusionVisualizationPlugin,
    EnergyRewardVisualizationPlugin,
    StatesVisualizationPlugin,
    Box2DObstacleGeneratorPlugin,
)


def main():
    """Main entry point for running experiments from config files."""
    parser = argparse.ArgumentParser(
        description="Run experiment from YAML configuration file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'config',
        type=str,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--level',
        type=int,
        default=None,
        help='Run single obstacle level (overrides config)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Run single seed (overrides config)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without running experiments'
    )
    
    args = parser.parse_args()
    
    # Load configuration with smart path resolution
    config_path = Path(args.config)
    
    # If path is relative and doesn't exist, try relative to project root
    if not config_path.is_absolute() and not config_path.exists():
        # Try to find project root (where setup.py or pyproject.toml exists)
        script_dir = Path(__file__).parent
        project_root = script_dir.parent.parent  # enerdynamics/experiments -> enerdynamics -> project_root
        candidate_path = project_root / config_path
        if candidate_path.exists():
            config_path = candidate_path
    
    # Convert to absolute path for clarity
    config_path = config_path.resolve()
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        print(f"Current working directory: {Path.cwd()}")
        sys.exit(1)
    
    try:
        if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
            config = ExperimentConfig.from_yaml(config_path)
        else:
            config = ExperimentConfig.from_json(config_path)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        sys.exit(1)
    
    # Override level and seed if specified
    if args.level is not None:
        config.obstacle_levels = [args.level]
    if args.seed is not None:
        config.seeds = [args.seed]
    
    # Validate configuration
    errors = config.validate()
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    
    if args.dry_run:
        print("Configuration is valid (dry run mode)")
        print(f"Environment: {config.env_name}")
        print(f"Method: {config.method}")
        print(f"Levels: {config.obstacle_levels}")
        print(f"Seeds: {config.seeds}")
        return
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Register all plugins
    register_all_plugins(runner)
    
    # Run experiments
    print(f"Starting experiments from {config_path}")
    print(f"Environment: {config.env_name}")
    print(f"Method: {config.method}")
    print(f"Levels: {config.obstacle_levels}")
    print(f"Seeds: {config.seeds}")
    print(f"Output directory: {config.output_dir}")
    print()
    
    results = runner.run_all()
    
    print(f"\nCompleted {len(results)} experiments")
    print(f"Results saved to: {config.output_dir}")


def register_all_plugins(runner: ExperimentRunner):
    """
    Register all available plugins with the experiment runner.
    
    Args:
        runner: Experiment runner instance
    """
    # Method plugins
    runner.register_plugin(EDOCMethodPlugin(), 'method')
    
    # Environment plugins
    runner.register_plugin(SingleIntegrator2DPlugin(), 'environment')
    runner.register_plugin(DoubleIntegrator2DPlugin(), 'environment')
    runner.register_plugin(DroneEnvironmentPlugin(), 'environment')
    runner.register_plugin(ManipulatorEnvironmentPlugin(), 'environment')
    
    # Metrics plugins
    runner.register_plugin(SSRMetricsPlugin(), 'metric')
    runner.register_plugin(ObstacleDensityMetricsPlugin(), 'metric')
    runner.register_plugin(NonconvexityMetricsPlugin(), 'metric')
    
    # Visualization plugins
    runner.register_plugin(TrajectoryVisualizationPlugin(), 'visualization')
    runner.register_plugin(DiffusionVisualizationPlugin(), 'visualization')
    runner.register_plugin(EnergyRewardVisualizationPlugin(), 'visualization')
    runner.register_plugin(StatesVisualizationPlugin(), 'visualization')
    
    # Obstacle generator plugins
    runner.register_plugin(Box2DObstacleGeneratorPlugin(), 'obstacle_generator')


if __name__ == "__main__":
    main()

