"""
Test script to verify the new constraint pipeline integration with EDOC.

This script runs a simple experiment to ensure:
1. create_constraint_pipeline works correctly
2. EDOC can use the new pipeline
3. Results are reasonable
"""

import numpy as np
from pathlib import Path

# Import experiment framework
from enerdynamics.experiments.framework.experiment import ExperimentRunner
from enerdynamics.experiments.framework.config import ExperimentConfig
from enerdynamics.experiments.common.constraints import create_constraint_pipeline
from enerdynamics.core.backends.runtime import RuntimeBackendManager

# Import EDOC
from enerdynamics.solvers.single.edoc import EDOCPlanner
from enerdynamics.envs.factories import make_env, make_energy
from enerdynamics.experiments.plugins.obstacles.box2d import Box2DObstacleGeneratorPlugin

# Import plugins for experiment framework
from enerdynamics.experiments.plugins import (
    EDOCMethodPlugin,
    SingleIntegrator2DPlugin,
    SSRMetricsPlugin,
)

# Import plugins for experiment framework
from enerdynamics.experiments.plugins import (
    EDOCMethodPlugin,
    SingleIntegrator2DPlugin,
    Box2DObstacleGeneratorPlugin as Box2DPlugin,
    SSRMetricsPlugin,
)


def test_pipeline_creation():
    """Test that create_constraint_pipeline works correctly."""
    print("\n" + "="*80)
    print("Test 1: Pipeline Creation")
    print("="*80)
    
    # Setup
    RuntimeBackendManager.set_backend("numpy", device="cpu")
    env = make_env("single_integrator_box_2d")
    env.dt = 0.05
    env.horizon = 64
    
    # Create obstacles
    start_pos = np.array([-1.0, -2.0], dtype=np.float32)
    target_pos = np.array([0.0, 0.0], dtype=np.float32)
    obstacle_gen = Box2DObstacleGeneratorPlugin()
    obstacles = obstacle_gen.generate(
        level=6,
        seed=0,
        start_pos=start_pos,
        target_pos=target_pos,
        config={
            'robot_radius': 0.05,
            'obstacle_radius_scale': 1.3,
            'p_max': 2.0,
            'map_bounds': {
                'x_min': -1.5,
                'x_max': 1.0,
                'y_min': -2.0,
                'y_max': 0.5,
            },
        }
    )
    
    # Build SDF texture
    obstacles.build_sdf_texture_2d(
        x_min=-1.5, x_max=1.0, y_min=-2.0, y_max=0.5, res=0.01, force_rebuild=True
    )
    
    # Create constraint config
    constraint_config = {
        'soft_constraint': {'enabled': True, 'alpha': 1.0, 'beta': 10.0},
        'hard_constraint': {'enabled': True, 'clearance': 0.2},
        'schedule': {
            'enabled': True,
            'type': 'soft_to_hard',
            'soft_alpha_start': 1.0,
            'soft_alpha_end': 0.0,
            'hard_clearance_start': 0.5,
            'hard_clearance_end': 0.1,
            'schedule_type': 'linear',
            'reverse_mode': True,
        },
        'cfs': {
            'enabled': True,
            'use_trajectory_qp': True,
            'max_iterations': 30,
            'use_late_stage_only': True,
            'late_stage_ratio': 0.2,
            'reconstruct_velocity': False,
            'smoothness_weight': 1.0,
        },
        'action_constraint_type': 'speed',
    }
    
    obstacle_config = {
        'robot_radius': 0.05,
    }
    
    # Test pipeline creation
    try:
        pipeline = create_constraint_pipeline(
            obstacles=obstacles,
            level=6,
            env=env,
            config=constraint_config,
            backend_name="numpy",
            obstacle_config=obstacle_config,
        )
        
        if pipeline is None:
            print("❌ Pipeline creation returned None")
            return False
        
        print(f"✅ Pipeline created successfully")
        print(f"   - Type: {type(pipeline).__name__}")
        print(f"   - Backend: {pipeline.config.backend}")
        print(f"   - Convexifier: {type(pipeline.convexifier).__name__}")
        print(f"   - Operator: {type(pipeline.operator).__name__}")
        print(f"   - Scheduler: {type(pipeline.scheduler).__name__}")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_edoc_with_pipeline():
    """Test that EDOC can use the new pipeline."""
    print("\n" + "="*80)
    print("Test 2: EDOC with Pipeline")
    print("="*80)
    
    # Setup
    RuntimeBackendManager.set_backend("numpy", device="cpu")
    env = make_env("single_integrator_box_2d")
    env.dt = 0.05
    env.horizon = 64
    energy = make_energy("single_integrator_box_2d")
    
    # Create obstacles
    start_pos = np.array([-1.0, -2.0], dtype=np.float32)
    target_pos = np.array([0.0, 0.0], dtype=np.float32)
    obstacle_gen = Box2DObstacleGeneratorPlugin()
    obstacles = obstacle_gen.generate(
        level=6,
        seed=0,
        start_pos=start_pos,
        target_pos=target_pos,
        config={
            'robot_radius': 0.05,
            'obstacle_radius_scale': 1.3,
            'p_max': 2.0,
            'map_bounds': {
                'x_min': -1.5,
                'x_max': 1.0,
                'y_min': -2.0,
                'y_max': 0.5,
            },
        }
    )
    
    # Build SDF texture
    obstacles.build_sdf_texture_2d(
        x_min=-1.5, x_max=1.0, y_min=-2.0, y_max=0.5, res=0.01, force_rebuild=True
    )
    
    # Create constraint config
    constraint_config = {
        'soft_constraint': {'enabled': True, 'alpha': 1.0, 'beta': 10.0},
        'hard_constraint': {'enabled': True, 'clearance': 0.2},
        'schedule': {
            'enabled': True,
            'type': 'soft_to_hard',
            'soft_alpha_start': 1.0,
            'soft_alpha_end': 0.0,
            'hard_clearance_start': 0.5,
            'hard_clearance_end': 0.1,
            'schedule_type': 'linear',
            'reverse_mode': True,
        },
        'cfs': {
            'enabled': True,
            'use_trajectory_qp': True,
            'max_iterations': 30,
            'use_late_stage_only': True,
            'late_stage_ratio': 0.2,
            'reconstruct_velocity': False,
            'smoothness_weight': 1.0,
        },
        'action_constraint_type': 'speed',
    }
    
    obstacle_config = {
        'robot_radius': 0.05,
    }
    
    # Create pipeline
    pipeline = create_constraint_pipeline(
        obstacles=obstacles,
        level=6,
        env=env,
        config=constraint_config,
        backend_name="numpy",
        obstacle_config=obstacle_config,
    )
    
    if pipeline is None:
        print("❌ Pipeline is None, cannot test EDOC")
        return False
    
    # Create EDOC planner with pipeline
    try:
        planner = EDOCPlanner(
            env=env,
            energy=energy,
            horizon=64,
            dt=0.05,
            action_space=True,
            diffusion_mode="reverse",
            action_diffuse_steps=50,  # Reduced for faster testing
            action_nsample=32,  # Reduced for faster testing
            use_antithetic=True,
            action_score_mode="energy",
            constraint_pipeline=pipeline,
            use_constraint_in_scoring=True,
            lambda_energy=1.0,
            terminal_energy_weight=50.0,
            show_tqdm=False,
        )
        
        print("✅ EDOC planner created successfully with pipeline")
        
        # Run a quick planning test
        print("   Running quick planning test...")
        result = planner.plan(0)
        
        if result is None:
            print("❌ Planning returned None")
            return False
        
        print(f"✅ Planning completed successfully")
        print(f"   - States shape: {np.asarray(result['states']).shape}")
        print(f"   - Actions shape: {np.asarray(result['actions']).shape}")
        print(f"   - Final state: {result['states'][-1]}")
        energies = result.get('energies')
        if energies is not None and len(energies) > 0:
            final_cost = energies[-1] if isinstance(energies, (list, np.ndarray)) else energies
            print(f"   - Final cost: {final_cost}")
        else:
            print(f"   - Final cost: N/A")
        
        return True
        
    except Exception as e:
        print(f"❌ EDOC with pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_experiment_framework():
    """Test that experiment framework works with new pipeline."""
    print("\n" + "="*80)
    print("Test 3: Experiment Framework Integration")
    print("="*80)
    
    # Create a minimal config
    config_dict = {
        'name': 'test_pipeline',
        'output_dir': 'test_results/pipeline_test',
        'env_name': 'single_integrator_box_2d',
        'env_params': {
            'dt': 0.05,
            'horizon': 32,  # Reduced for faster testing
            'p_max': 2.0,
            'control_limit': 1.0,
        },
        'method': 'edoc',
        'method_params': {
            'action_diffuse_steps': 20,  # Reduced for faster testing
            'action_nsample': 16,  # Reduced for faster testing
            'use_antithetic': True,
            'action_score_mode': 'energy',
            'terminal_energy_weight': 50.0,
        },
        'obstacle_levels': [3],  # Lower level for faster testing
        'obstacle_config': {
            'generator': 'box2d',
            'robot_radius': 0.05,
            'obstacle_radius_scale': 1.3,
            'p_max': 2.0,
            'map_bounds': {
                'x_min': -1.5,
                'x_max': 1.0,
                'y_min': -2.0,
                'y_max': 0.5,
            },
            'enable_connectivity_check': True,
            'enable_nonconvexity_check': True,
        },
        'seeds': [0],
        'backend': 'numpy',
        'device': 'cpu',
        'constraint_config': {
            'soft_constraint': {'enabled': True, 'alpha': 1.0, 'beta': 10.0},
            'hard_constraint': {'enabled': True, 'clearance': 0.2},
            'schedule': {
                'enabled': True,
                'type': 'soft_to_hard',
                'soft_alpha_start': 1.0,
                'soft_alpha_end': 0.0,
                'hard_clearance_start': 0.5,
                'hard_clearance_end': 0.1,
                'schedule_type': 'linear',
                'reverse_mode': True,
            },
            'cfs': {
                'enabled': True,
                'use_trajectory_qp': True,
                'max_iterations': 10,  # Reduced for faster testing
                'use_late_stage_only': True,
                'late_stage_ratio': 0.2,
                'reconstruct_velocity': False,
                'smoothness_weight': 1.0,
            },
            'action_constraint_type': 'speed',
        },
        'metrics': ['ssr'],
        'visualizations': [],
    }
    
    try:
        # Create config directly (ExperimentConfig doesn't have from_dict)
        config = ExperimentConfig(**config_dict)
        runner = ExperimentRunner(config)
        
        # Register required plugins
        runner.register_plugin(EDOCMethodPlugin(), 'method', 'edoc')
        runner.register_plugin(SingleIntegrator2DPlugin(), 'environment', 'single_integrator_box_2d')
        runner.register_plugin(Box2DObstacleGeneratorPlugin(), 'obstacle_generator', 'box2d')
        runner.register_plugin(SSRMetricsPlugin(), 'metric', 'ssr')
        
        print("✅ Experiment runner created successfully")
        
        # Run a single experiment
        print("   Running single experiment...")
        result = runner.run_single_experiment(level=3, seed=0)
        
        if result is None:
            print("❌ Experiment returned None")
            return False
        
        print(f"✅ Experiment completed successfully")
        print(f"   - Success: {result.get('success', False)}")
        print(f"   - Planning time: {result.get('planning_time', 'N/A')}")
        if 'trajectory' in result:
            traj = result['trajectory']
            print(f"   - Trajectory states: {len(traj.states) if hasattr(traj, 'states') else 'N/A'}")
            print(f"   - Trajectory actions: {len(traj.actions) if hasattr(traj, 'actions') else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Experiment framework test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("Testing New Constraint Pipeline Integration")
    print("="*80)
    
    results = []
    
    # Test 1: Pipeline creation
    results.append(("Pipeline Creation", test_pipeline_creation()))
    
    # Test 2: EDOC with pipeline
    results.append(("EDOC with Pipeline", test_edoc_with_pipeline()))
    
    # Test 3: Experiment framework
    results.append(("Experiment Framework", test_experiment_framework()))
    
    # Summary
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result[1] for result in results)
    print("\n" + "="*80)
    if all_passed:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*80)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

