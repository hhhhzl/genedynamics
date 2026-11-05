"""
Run experiment from YAML configuration file.

This script demonstrates how to use the new experiment framework
to run experiments from YAML configuration files.
"""

import argparse
import hashlib
import os
import warnings
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Headless/CI: Linux use egl; Darwin/Windows use default (cgl/glfw)
if "MUJOCO_GL" not in os.environ:
    import platform
    if platform.system() == "Linux":
        os.environ.setdefault("MUJOCO_GL", "egl")

# Suppress GLFW DISPLAY/init warnings when using headless (MUJOCO_GL=egl/osmesa)
warnings.filterwarnings("ignore", message=".*[Dd]isplay|GLFW.*", module="glfw")

from genedynamics.experiments.framework import ExperimentRunner, ExperimentConfig
from genedynamics.experiments.plugins import (
    MDOCMethodPlugin,
    EBMBDMethodPlugin,
    MBDMethodPlugin,
    D3ILUnifiedMethodPlugin,
    CFSMBDMethodPlugin,
    CFSMBDFullMethodPlugin,
    MPPIMethodPlugin,
    TwoGOMethodPlugin,
    DPCCMethodPlugin,
    SafeDiffuserMethodPlugin,
    SingleIntegrator2DPlugin,
    DoubleIntegrator2DPlugin,
    QuadrupedFlatMjxPlugin,
    QuadrupedGo2MjxPlugin,
    QuadrupedGo2BraxPlugin,
    QuadrupedSteppingStones2DPlugin,
    HumanoidSimplifiedMjxPlugin,
    HumanoidRunBraxPlugin,
    HumanoidG1MjxPlugin,
    DroneEnvironmentPlugin,
    DroneBox3DPlugin,
    DroneFull3DPlugin,
    DroneFull3DPhysicsPlugin,
    ManipulatorEnvironmentPlugin,
    D3ILAvoidingPlugin,
    D3ILAvoiding9DPlugin,
    AvoidingPlanEnvironmentPlugin,
    HumanoidCorridor2DPlugin,
    SSRMetricsPlugin,
    ObstacleDensityMetricsPlugin,
    NonconvexityMetricsPlugin,
    EpisodeOutcomeMetricsPlugin,
    SteppingStonesMetricsPlugin,
    CorridorMetricsPlugin,
    TrajectoryVisualizationPlugin,
    Trajectory3DVisualizationPlugin,
    TrajectoryModesVisualizationPlugin,
    DiffusionVisualizationPlugin,
    Diffusion3DVisualizationPlugin,
    EnergyRewardVisualizationPlugin,
    StatesVisualizationPlugin,
    SchedulerParamsVisualizationPlugin,
    SteppingStonesTrajectoryVisualizationPlugin,
    SteppingStonesModesVisualizationPlugin,
    CorridorTrajectoryVisualizationPlugin,
    Box2DObstacleGeneratorPlugin,
    Box3DObstacleGeneratorPlugin,
    D3ILAvoidingFixedGeneratorPlugin,
    SteppingStones2DObstacleGeneratorPlugin,
    Corridor2DObstacleGeneratorPlugin,
    EmptyObstacleGeneratorPlugin,
    ManipulatorSurfaceScanPlugin,
    ManipulatorPegInsertPlugin,
    HumanoidBoxPushPlugin,
    MGAMethodPlugin,
    ModelBasedOnlyMethodPlugin,
    DIALContactMethodPlugin,
    PegasusFlowContactMethodPlugin,
    ISSAContactMethodPlugin,
    ATACOMContactMethodPlugin,
    StandaloneRLMethodPlugin,
    transfer_method_plugins,
    transfer_environment_plugins,
    TransferTaskMetricsPlugin,
)


def _formal_readiness_errors(config: ExperimentConfig) -> list[str]:
    """Return execution blockers for a canonical formal result directory.

    Development runs use the same resolved algorithm config under an isolated
    output root.  A ``run_class: development`` label inside a canonical config
    is not isolation and must never bypass this gate.
    """
    metadata = config.metadata or {}
    if not metadata.get("formal_experiment", True):
        return []
    errors: list[str] = []
    if metadata.get("benchmark") == "four_task_transfer":
        if config.env_name != "obstacle_transfer":
            pointer = metadata.get("shared_task_contract")
            if not isinstance(pointer, dict):
                errors.append("four-task transfer config has no shared_task_contract binding")
            else:
                project_root = config._find_project_root()
                path = Path(str(pointer.get("path", ""))).expanduser()
                if not path.is_absolute() and project_root is not None:
                    path = project_root / path
                expected = str(pointer.get("sha256", ""))
                if not path.is_file():
                    errors.append(f"shared_task_contract is missing: {path}")
                elif not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    errors.append("shared_task_contract failed SHA256 validation")
        frozen = metadata.get("frozen_design")
        if not isinstance(frozen, dict):
            errors.append("four-task transfer config has no frozen_design binding")
        else:
            expected_design = {
                "admission_scope": "pre_refactor_design_and_preaccess",
                "migration_rule": "rebind_and_prove_equivalence_before_confirmation",
                "main_table_rows": 9,
                "ablation_cells": 6,
                "training_or_calibration_instances": 8,
            }
            for key, expected in expected_design.items():
                if frozen.get(key) != expected:
                    errors.append(
                        f"frozen_design.{key} must be {expected}, got "
                        f"{frozen.get(key)!r}"
                    )
            if frozen.get("confirmation_accessed") is not False:
                errors.append("frozen_design must declare confirmation_accessed: false")
            artifacts = frozen.get("artifacts")
            if not isinstance(artifacts, dict) or not artifacts:
                errors.append("frozen_design has no published artifact bindings")
            else:
                project_root = config._find_project_root()
                for label, pointer in artifacts.items():
                    if not isinstance(pointer, dict):
                        errors.append(f"frozen_design artifact {label!r} is malformed")
                        continue
                    path = Path(str(pointer.get("path", ""))).expanduser()
                    if not path.is_absolute() and project_root is not None:
                        path = project_root / path
                    expected = str(pointer.get("sha256", ""))
                    if not path.is_file():
                        errors.append(f"frozen_design artifact {label!r} is missing: {path}")
                    elif not expected or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                        errors.append(f"frozen_design artifact {label!r} failed SHA256 validation")
        if config.method != "task_reference":
            binding = config.method_params.get("actor_binding")
            if (
                not isinstance(binding, dict)
                or binding.get("version") != "transfer-actor-binding-v1"
            ):
                errors.append(
                    "method has no transfer-actor-binding-v1; bind and hash all "
                    "package-owned artifacts before any task reset"
                )
            elif "manifest" in binding:
                try:
                    from genedynamics.tasks.transfer.controllers.bindings import (
                        resolve_transfer_binding,
                        validate_transfer_actor_manifest,
                    )

                    validate_transfer_actor_manifest(binding["manifest"])
                    resolved = resolve_transfer_binding(config.method_params)
                    if resolved.get("status") == "instances_ready":
                        errors.append(
                            "learned actor selection requires one explicit "
                            "instance_01 through instance_08 before execution"
                        )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    errors.append(f"actor binding manifest failed validation: {exc}")

    isolated = bool(metadata.get("development_output_root"))
    if isolated:
        return errors
    if metadata.get("run_class") == "development":
        errors.append(
            "canonical output cannot use run_class 'development'; pass "
            "--development-root to create an isolated development run"
        )
    status = str(metadata.get("protocol_status", "")).strip()
    if not status.startswith("frozen"):
        errors.append(
            f"formal protocol is {status!r}; use --development-root for probes "
            "and freeze the shared protocol before writing canonical results"
        )
    if metadata.get("formal_seeds") is not True:
        errors.append("canonical formal execution requires formal_seeds: true")
    if (
        metadata.get("benchmark") == "four_task_transfer"
        and metadata.get("architecture_equivalence") != "passed"
    ):
        errors.append("four-task transfer architecture equivalence is not passed")
    return errors


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
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Run single seed (overrides config)'
    )
    seed_group.add_argument(
        '--seeds',
        type=int,
        nargs='+',
        default=None,
        help='Run multiple seeds in one manifest/summary (overrides config)'
    )
    suite_group = parser.add_mutually_exclusive_group()
    suite_group.add_argument(
        '--suite',
        type=str,
        default=None,
        help='Run one named suite from config.suites'
    )
    suite_group.add_argument(
        '--suites',
        type=str,
        nargs='+',
        default=None,
        help='Run multiple named suites in one manifest/summary'
    )
    parser.add_argument(
        '--development-root',
        type=str,
        default=None,
        help=(
            'Mirror the canonical project-relative output under this root and '
            'mark the run as development'
        ),
    )
    parser.add_argument(
        '--device',
        choices=('cpu', 'gpu', 'cuda'),
        default=None,
        help='Override the configured JAX device for this run',
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help=(
            'Reuse complete matching results already present in output_dir and '
            'run only missing or config-mismatched matrix entries'
        ),
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
        project_root = script_dir.parent.parent  # genedynamics/experiments -> genedynamics -> project_root
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
    
    if args.development_root is not None:
        try:
            config.use_development_output_root(Path(args.development_root))
        except ValueError as exc:
            parser.error(str(exc))
    if args.device is not None:
        config.device = args.device

    # Override level, seeds, and suites if specified.
    if args.level is not None:
        config.obstacle_levels = [args.level]
    if args.seed is not None:
        config.seeds = [args.seed]
    elif args.seeds is not None:
        config.seeds = list(dict.fromkeys(args.seeds))
    requested_suites = (
        [args.suite] if args.suite is not None else args.suites
    )
    if requested_suites is not None:
        requested_suites = list(dict.fromkeys(requested_suites))
        available_map = {str(s.get('name')): s for s in config.suites}
        unknown = [name for name in requested_suites if name not in available_map]
        if unknown:
            available = ', '.join(available_map)
            parser.error(
                f"unknown suites {unknown!r}; available: {available or '<none>'}"
            )
        requested = set(requested_suites)
        matches = [s for s in config.suites if str(s.get('name')) in requested]
        config.suites = matches
    
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
        print(f"Suites: {[s['name'] for s in config.suites]}" if config.suites
              else f"Levels: {config.obstacle_levels}")
        print(f"Seeds: {config.seeds}")
        print(f"Output directory: {config.output_dir}")
        print(f"Run class: {config.metadata.get('run_class', 'unspecified')}")
        return

    readiness_errors = _formal_readiness_errors(config)
    if readiness_errors:
        print("Formal execution is not ready:")
        for error in readiness_errors:
            print(f"  - {error}")
        raise SystemExit(2)

    if (config.metadata or {}).get("require_checkpoint_files", False):
        missing = []
        for key, value in config.method_params.items():
            if not isinstance(key, str) or not key.endswith("_ckpt") or not value:
                continue
            checkpoint = Path(str(value)).expanduser()
            if not checkpoint.is_absolute():
                checkpoint = Path.cwd() / checkpoint
            if not checkpoint.is_file():
                missing.append((key, value))
        if missing:
            print("Required learned components are not ready:")
            for key, value in missing:
                print(f"  - {key}: {value}")
            print(
                "Train and freeze the PegInsert components before running "
                "the formal matrix; archived checkpoints are not resumed."
            )
            raise SystemExit(2)
    
    # Create experiment runner
    runner = ExperimentRunner(config)
    
    # Register all plugins
    register_all_plugins(runner)
    
    # Run experiments
    print(f"Starting experiments from {config_path}")
    print(f"Environment: {config.env_name}")
    print(f"Method: {config.method}")
    print(f"Suites: {[s['name'] for s in config.suites]}" if config.suites
          else f"Levels: {config.obstacle_levels}")
    print(f"Seeds: {config.seeds}")
    print(f"Output directory: {config.output_dir}")
    print()
    
    results = runner.run_all(resume=args.resume)

    failures = runner.failures
    print(
        f"\nCompleted {len(results)} experiments; "
        f"failed {len(failures)} experiments"
    )
    print(f"Results saved to: {config.output_dir}")
    if failures:
        print("Failed experiments:")
        for failure in failures:
            print(
                f"  - level={failure['level']}, seed={failure['seed']}: "
                f"{failure['error_type']}: {failure['error']}"
            )
        raise SystemExit(1)


def register_all_plugins(runner: ExperimentRunner):
    """
    Register all available plugins with the experiment runner.
    
    Args:
        runner: Experiment runner instance
    """
    # Method plugins
    runner.register_plugin(MDOCMethodPlugin(), 'method')
    runner.register_plugin(EBMBDMethodPlugin(), 'method')
    runner.register_plugin(MBDMethodPlugin(), 'method')
    runner.register_plugin(D3ILUnifiedMethodPlugin(), 'method')
    runner.register_plugin(CFSMBDMethodPlugin(), 'method')
    runner.register_plugin(CFSMBDFullMethodPlugin(), 'method')
    runner.register_plugin(MPPIMethodPlugin(), 'method')
    runner.register_plugin(MGAMethodPlugin(), 'method')
    runner.register_plugin(ModelBasedOnlyMethodPlugin(), 'method')
    runner.register_plugin(DIALContactMethodPlugin(), 'method')
    runner.register_plugin(PegasusFlowContactMethodPlugin(), 'method')
    runner.register_plugin(ISSAContactMethodPlugin(), 'method')
    runner.register_plugin(ATACOMContactMethodPlugin(), 'method')
    runner.register_plugin(StandaloneRLMethodPlugin(), 'method')
    for plugin in transfer_method_plugins():
        runner.register_plugin(plugin, 'method')
    runner.register_plugin(TwoGOMethodPlugin(), 'method')
    if DPCCMethodPlugin is not None:
        runner.register_plugin(DPCCMethodPlugin(), 'method')
    if SafeDiffuserMethodPlugin is not None:
        runner.register_plugin(SafeDiffuserMethodPlugin(), 'method')
    
    # Environment plugins
    runner.register_plugin(SingleIntegrator2DPlugin(), 'environment')
    runner.register_plugin(DoubleIntegrator2DPlugin(), 'environment')
    runner.register_plugin(QuadrupedFlatMjxPlugin(), 'environment')
    runner.register_plugin(QuadrupedGo2MjxPlugin(), 'environment')
    runner.register_plugin(QuadrupedGo2BraxPlugin(), 'environment')
    runner.register_plugin(QuadrupedSteppingStones2DPlugin(), 'environment')
    runner.register_plugin(HumanoidSimplifiedMjxPlugin(), 'environment')
    runner.register_plugin(HumanoidRunBraxPlugin(), 'environment')
    runner.register_plugin(HumanoidG1MjxPlugin(), 'environment')
    runner.register_plugin(DroneEnvironmentPlugin(), 'environment')
    runner.register_plugin(DroneBox3DPlugin(), 'environment')
    runner.register_plugin(DroneFull3DPlugin(), 'environment')
    runner.register_plugin(DroneFull3DPhysicsPlugin(), 'environment')
    runner.register_plugin(ManipulatorEnvironmentPlugin(), 'environment')
    runner.register_plugin(D3ILAvoidingPlugin(), 'environment')
    runner.register_plugin(D3ILAvoiding9DPlugin(), 'environment')
    runner.register_plugin(AvoidingPlanEnvironmentPlugin(), 'environment')
    runner.register_plugin(HumanoidCorridor2DPlugin(), 'environment')
    runner.register_plugin(ManipulatorSurfaceScanPlugin(), 'environment')
    runner.register_plugin(ManipulatorPegInsertPlugin(), 'environment')
    runner.register_plugin(HumanoidBoxPushPlugin(), 'environment')
    for plugin in transfer_environment_plugins():
        runner.register_plugin(plugin, 'environment')

    # Metrics plugins
    runner.register_plugin(SSRMetricsPlugin(), 'metric')
    runner.register_plugin(ObstacleDensityMetricsPlugin(), 'metric')
    runner.register_plugin(NonconvexityMetricsPlugin(), 'metric')
    runner.register_plugin(EpisodeOutcomeMetricsPlugin(), 'metric')
    runner.register_plugin(SteppingStonesMetricsPlugin(), 'metric')
    runner.register_plugin(CorridorMetricsPlugin(), 'metric')
    runner.register_plugin(TransferTaskMetricsPlugin(), 'metric')
    # General-metrics plugins (shared evaluation library + a tiny per-task
    # extractor). brax tasks; the factories construct without importing brax.
    from genedynamics.experiments.plugins.metrics.extractors import (
        arm_surface_scan_metrics_plugin, humanoid_box_push_metrics_plugin,
        peg_insert_metrics_plugin,
    )
    runner.register_plugin(arm_surface_scan_metrics_plugin(), 'metric')
    runner.register_plugin(peg_insert_metrics_plugin(), 'metric')
    runner.register_plugin(humanoid_box_push_metrics_plugin(), 'metric')

    # Visualization plugins
    runner.register_plugin(TrajectoryVisualizationPlugin(), 'visualization')
    runner.register_plugin(Trajectory3DVisualizationPlugin(), 'visualization')
    runner.register_plugin(TrajectoryModesVisualizationPlugin(), 'visualization')
    runner.register_plugin(DiffusionVisualizationPlugin(), 'visualization')
    runner.register_plugin(Diffusion3DVisualizationPlugin(), 'visualization')
    runner.register_plugin(EnergyRewardVisualizationPlugin(), 'visualization')
    runner.register_plugin(StatesVisualizationPlugin(), 'visualization')
    runner.register_plugin(SchedulerParamsVisualizationPlugin(), 'visualization')
    runner.register_plugin(SteppingStonesTrajectoryVisualizationPlugin(), 'visualization')
    runner.register_plugin(SteppingStonesModesVisualizationPlugin(), 'visualization')
    runner.register_plugin(CorridorTrajectoryVisualizationPlugin(), 'visualization')

    # Obstacle generator plugins
    runner.register_plugin(Box2DObstacleGeneratorPlugin(), 'obstacle_generator')
    runner.register_plugin(Box3DObstacleGeneratorPlugin(), 'obstacle_generator')
    runner.register_plugin(D3ILAvoidingFixedGeneratorPlugin(), 'obstacle_generator')
    runner.register_plugin(SteppingStones2DObstacleGeneratorPlugin(), 'obstacle_generator')
    runner.register_plugin(Corridor2DObstacleGeneratorPlugin(), 'obstacle_generator')
    runner.register_plugin(EmptyObstacleGeneratorPlugin(), 'obstacle_generator')


if __name__ == "__main__":
    main()
