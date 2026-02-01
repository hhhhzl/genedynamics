"""
Experiment execution framework.

This module provides the ExperimentRunner class that orchestrates
experiment execution using the plugin system.
"""

import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np

from enerdynamics.core.types import Trajectory
from enerdynamics.core.backends.runtime import RuntimeBackendManager

from .config import ExperimentConfig
from .registry import PluginRegistry
from ..common.constraints import create_constraint_manager, create_constraint_pipeline


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types and other non-JSON-serializable types to Python native types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8, np.uint16,
                        np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


class ExperimentRunner:
    """
    Unified experiment execution framework.
    
    This class orchestrates the execution of experiments using the plugin system.
    It handles environment setup, obstacle generation, planning, metrics computation,
    and result saving.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.registry = PluginRegistry()
        self.results: List[Dict[str, Any]] = []
        
        # Validate configuration
        errors = config.validate()
        if errors:
            raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
    
    def register_plugin(self, plugin: Any, plugin_type: str, name: Optional[str] = None) -> None:
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance
            plugin_type: Type of plugin ('method', 'environment', 'metric', etc.)
            name: Optional custom name (uses plugin.name if not provided)
        """
        self.registry.register(plugin, plugin_type, name)
    
    def run_single_experiment(self, level: int, seed: int) -> Dict[str, Any]:
        """
        Run single experiment instance.
        
        Args:
            level: Obstacle difficulty level
            seed: Random seed for reproducibility
            
        Returns:
            Dictionary containing experiment results
        """
        experiment_start_time = time.time()
        
        # Reset global random state for reproducibility
        # This ensures that each experiment run starts from the same random state
        np.random.seed(seed)
        
        # 1. Setup backend
        RuntimeBackendManager.set_backend(self.config.backend, device=self.config.device)
        backend = RuntimeBackendManager.get_backend()
        
        # 2. Setup environment
        env_plugin = self.registry.get_plugin('environment', self.config.env_name)
        
        # 3. Generate start/target positions (before obstacles, for obstacle generation)
        # We need a temporary env to get target position
        temp_env = env_plugin.create_env(self.config.env_params)
        target_pos = np.asarray(temp_env.target, dtype=np.float32)
        start_pos = self._generate_start_position(level, seed, temp_env, env_plugin)
        
        # 4. Generate obstacles
        obstacle_gen_name = self.config.obstacle_config.get('generator', 'box2d')
        obstacle_gen = self.registry.get_plugin('obstacle_generator', obstacle_gen_name)
        # Pass env_name to obstacle config for environment-specific generation
        obstacle_config_with_env = {
            **self.config.obstacle_config,
            'env_name': self.config.env_name,
        }
        obstacles = obstacle_gen.generate(level, seed, start_pos, target_pos, obstacle_config_with_env)
        
        # 5. Create environment with obstacles (for physics backends that need obstacles in model)
        env_params_with_obstacles = {**self.config.env_params}
        # Add obstacles to env_params if using physics backend that needs them
        physics_backend = env_params_with_obstacles.get('physics_backend', None)
        if physics_backend in ['mujoco', 'isaac'] and len(obstacles) > 0:
            env_params_with_obstacles['obstacles'] = obstacles
        
        env = env_plugin.create_env(env_params_with_obstacles)
        energy = env_plugin.create_energy()
        
        # Build SDF texture if needed (only for 2D environments)
        # For 3D environments, skip 2D SDF texture building
        if level > 0 and len(obstacles) > 0:
            # Check if this is a 3D environment by checking env_name or obstacle generator
            physics_backend = self.config.env_params.get('physics_backend', None)
            is_3d_env = (
                self.config.env_name in ['drone_box_3d', 'drone', 'drone_full_3d', 'drone_full_3d_physics', 
                                         'drone_full_3d_mujoco', 'drone_full_3d_isaac'] or
                self.config.obstacle_config.get('generator', '') == 'box3d' or
                physics_backend in ['mujoco', 'isaac']
            )
            
            if not is_3d_env:
                # Only build 2D SDF texture for 2D environments
                map_bounds = self.config.obstacle_config.get('map_bounds', {})
                obstacles.build_sdf_texture_2d(
                    x_min=float(map_bounds.get('x_min', -2.0)),
                    x_max=float(map_bounds.get('x_max', 2.0)),
                    y_min=float(map_bounds.get('y_min', -2.0)),
                    y_max=float(map_bounds.get('y_max', 2.0)),
                    res=0.01,
                    force_rebuild=True,
                )
        
        # 6. Setup constraints
        constraint_config = self.config.constraint_config or {}
        
        # Use new pipeline architecture (preferred)
        constraint_pipeline = create_constraint_pipeline(
            obstacles, level, env, constraint_config, self.config.backend,
            obstacle_config=self.config.obstacle_config,
            method_params=self.config.method_params,
        )
        
        # Also create legacy constraint_manager for backward compatibility
        # (only if explicitly requested or if pipeline is None)
        constraint_manager = None
        if constraint_config.get('use_legacy', False) or constraint_pipeline is None:
            constraint_manager = create_constraint_manager(
                obstacles, level, env, constraint_config, self.config.backend,
                obstacle_config=self.config.obstacle_config,
            )
        
        # 7. Create scheduler (if configured)
        scheduler = None
        scheduler_config = getattr(self.config, 'scheduler_config', None)
        if scheduler_config:
            from ..common.constraints import create_scheduler_from_config
            scheduler = create_scheduler_from_config(
                scheduler_config, 
                self.config.backend,
                method_params=self.config.method_params,
                obstacle_config=self.config.obstacle_config,  # Fix B3: Pass obstacle_config for robot_radius
            )
        
        # 8. Create planner
        method_plugin = self.registry.get_plugin('method', self.config.method)
        method_config = {
            **self.config.method_params,
            'constraint_manager': constraint_manager,  # Legacy (for backward compatibility)
            'constraint_pipeline': constraint_pipeline,  # New architecture (preferred)
            'scheduler': scheduler,  # New scheduler system
            'obstacles': obstacles,  # Provide obstacles to methods that can use fast SDF (e.g., EB-MBD)
            'obstacle_config': self.config.obstacle_config,  # Provide robot_radius/map bounds, etc.
            'np_random_seed': seed,  # Pass seed for reproducibility
        }
        # Merge first diffusion_scheduler's M_k / Ndiffuse / T_k / beta into method_config
        # so method plugins (MDOC, MBD, etc.) use YAML diffusion_schedulers values instead of defaults
        if scheduler is not None and getattr(scheduler, 'diffusion_schedulers', None):
            ds_list = scheduler.diffusion_schedulers
            if ds_list:
                from enerdynamics.core.constraints.core.types import ScheduleState
                params = ds_list[0].diffusion_params(ScheduleState(k=0, K=1))
                if params:
                    if 'M_k' in params:
                        method_config['Nsample'] = int(params['M_k'])
                        method_config['action_nsample'] = int(params['M_k'])
                    if 'Ndiffuse' in params and params.get('Ndiffuse') is not None:
                        method_config['Ndiffuse'] = int(params['Ndiffuse'])
                        method_config['action_diffuse_steps'] = int(params['Ndiffuse'])
                    if 'T_k' in params:
                        method_config['temp_sample'] = float(params['T_k'])
                    if params.get('beta0') is not None:
                        method_config['beta0'] = float(params['beta0'])
                    if params.get('betaT') is not None:
                        method_config['betaT'] = float(params['betaT'])
        planner = method_plugin.create_planner(env, energy, method_config)
        
        # 8. Run planning
        rng = backend.create_rng(seed)
        
        # Warmup call to exclude JIT compilation time from planning time measurement
        # Only needed for JAX backend with JIT enabled (NumPy backend doesn't need warmup)
        needs_warmup = False
        if self.config.backend == "jax":
            # Check if planner uses JIT
            if hasattr(planner, '_backend_impl') and hasattr(planner._backend_impl, 'use_jit'):
                if planner._backend_impl.use_jit:
                    needs_warmup = True
            
            # Also check if constraint manager uses JIT
            if not needs_warmup and constraint_manager is not None:
                if hasattr(constraint_manager, 'feasibility_operator'):
                    feas_op = constraint_manager.feasibility_operator
                    if feas_op is not None:
                        if hasattr(feas_op, '_backend_impl') and hasattr(feas_op._backend_impl, 'use_jit'):
                            if feas_op._backend_impl.use_jit:
                                needs_warmup = True
        
        if needs_warmup:
            try:
                warmup_result = method_plugin.plan(planner, start_pos, rng)
                # For JAX, ensure computation is complete before timing
                import jax
                jax.block_until_ready(warmup_result)
            except Exception:
                pass  # If warmup fails, continue anyway
            
            # Recreate rng to ensure same random seed for actual planning
            rng = backend.create_rng(seed)
        
        planning_start = time.time()
        result = method_plugin.plan(planner, start_pos, rng)
        planning_time = time.time() - planning_start
        
        # Check if planning returned a valid result
        if result is None:
            raise ValueError("Planning returned None. Planning may have failed.")
        
        # 9. Extract trajectory
        trajectory = self._extract_trajectory(result, env)
        
        # 10. Compute metrics
        metrics = self._compute_metrics(trajectory, env, obstacles, constraint_manager, level, env_plugin=env_plugin)
        
        # 11. Prepare results
        # Add obstacle statistics for backward compatibility
        num_obstacles = len(obstacles) if obstacles else 0
        num_union_obstacles = 0
        num_primitives_total = 0
        if obstacles and num_obstacles > 0:
            from enerdynamics.envs.obstacles.nonconvex import UnionObstacle
            for obs in list(obstacles):
                if isinstance(obs, UnionObstacle):
                    num_union_obstacles += 1
                    num_primitives_total += len(obs.obstacles)
                else:
                    num_primitives_total += 1
        
        # Check if CFS is enabled
        cfs_enabled = False
        if constraint_manager and constraint_manager.feasibility_operator is not None:
            cfs_enabled = True
        
        experiment_result = {
            'level': level,
            'seed': seed,
            'trajectory': trajectory,
            'metrics': metrics,
            'planning_time': float(planning_time),
            'result': result,
            'config_snapshot': self.config.to_dict(),
            # Additional fields for backward compatibility
            'num_obstacles': num_obstacles,
            'num_union_obstacles': num_union_obstacles,
            'num_primitives_total': num_primitives_total,
            'cfs_enabled': cfs_enabled,
        }
        
        # 12. Generate visualizations
        if self.config.visualizations:
            self._generate_visualizations(experiment_result, env, obstacles, env_plugin)
        
        total_time = time.time() - experiment_start_time
        experiment_result['total_time'] = float(total_time)
        
        return experiment_result
    
    def run_all(self) -> List[Dict[str, Any]]:
        """
        Run all experiments according to configuration.
        
        Returns:
            List of experiment result dictionaries
        """
        all_results = []
        
        for level in self.config.obstacle_levels:
            for seed in self.config.seeds:
                try:
                    result = self.run_single_experiment(level, seed)
                    all_results.append(result)
                    self.results.append(result)
                    self._save_result(result)
                except Exception as e:
                    print(f"Error in level={level}, seed={seed}: {e}")
                    import traceback
                    traceback.print_exc()
        
        self._save_summary(all_results)
        return all_results
    
    def _generate_start_position(self, level: int, seed: int, env: Any, env_plugin: Any) -> np.ndarray:
        """
        Generate start position based on level and seed.
        
        Args:
            level: Obstacle level
            seed: Random seed
            env: Environment instance
            env_plugin: Environment plugin
            
        Returns:
            Start position array (may be full state vector with velocities)
        """
        np.random.seed(seed)
        target_pos = np.asarray(env.target, dtype=np.float32)
        p_max = getattr(env, 'p_max', 2.0)
        
        # Extract target position (first 2 or 3 elements depending on environment)
        target_pos_flat = target_pos.flatten()
        pos_dim = min(len(target_pos_flat), 3)  # Support up to 3D positions
        target_pos_only = target_pos_flat[:pos_dim]
        
        # Distance from target increases with level
        if pos_dim == 2:
            max_distance = 2.0 * np.sqrt(2.0)
        elif pos_dim == 3:
            max_distance = 2.0 * np.sqrt(3.0)
        else:
            max_distance = 2.0
        min_dist = 0.5 + (level / 10.0) * 1.0
        max_dist = 1.0 + (level / 10.0) * (max_distance - 1.0)
        distance = np.random.uniform(min_dist, max_dist)
        
        # Generate start position
        if pos_dim == 2:
            # 2D: random angle in xy plane
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)
            start = target_pos_only + distance * direction
        elif pos_dim == 3:
            # 3D: random direction on sphere
            # Use spherical coordinates
            theta = np.random.uniform(0, 2 * np.pi)  # azimuth angle
            phi = np.random.uniform(0, np.pi)  # polar angle
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ], dtype=np.float32)
            start = target_pos_only + distance * direction
        else:
            # 1D: just add/subtract distance
            sign = np.random.choice([-1, 1])
            start = target_pos_only + sign * distance * np.array([1.0], dtype=np.float32)
        
        # Clip to bounds
        margin = 0.2
        if pos_dim == 1:
            start = np.clip(start, -p_max + margin, p_max - margin)
        else:
            start = np.clip(start, -p_max + margin, p_max - margin)
        
        # For environments with state_dim > pos_dim, add zero velocity
        state_dim = env_plugin.get_state_dim()
        if state_dim > pos_dim:
            start_full = np.concatenate([start, np.zeros(state_dim - pos_dim, dtype=np.float32)])
        else:
            start_full = start
        
        return start_full.astype(np.float32)
    
    def _extract_trajectory(self, result: Dict[str, Any], env: Any) -> Trajectory:
        """
        Extract trajectory from planning result.
        
        Args:
            result: Planning result dictionary (must not be None)
            env: Environment instance
            
        Returns:
            Trajectory object
        """
        if result is None:
            raise ValueError("Cannot extract trajectory from None result")
        
        states = result.get('states', [])
        actions = result.get('actions', None)
        
        if actions is None or len(actions) == 0:
            # Generate dummy actions
            act_dim = getattr(env, 'act_dim', 2)
            actions = [np.zeros(act_dim, dtype=np.float32) for _ in range(len(states) - 1)]
        
        states_list = [np.asarray(s, dtype=np.float32) for s in states]
        actions_list = [np.asarray(a, dtype=np.float32) for a in actions]
        
        # Attach optional per-step execution info if present (useful for rollout-based methods)
        info = {}
        if "infos" in result:
            info["infos"] = result.get("infos")
        if "costs" in result:
            info["costs"] = result.get("costs")
        if "success" in result:
            info["success"] = result.get("success")
        if "collision" in result:
            info["collision"] = result.get("collision")

        return Trajectory(states=states_list, actions=actions_list, info=info or None)
    
    def _compute_metrics(
        self,
        trajectory: Trajectory,
        env: Any,
        obstacles: Any,
        constraints: Any,
        level: int,
        env_plugin: Any = None,
    ) -> Dict[str, Any]:
        """
        Compute all requested metrics.
        
        Args:
            trajectory: Computed trajectory
            env: Environment instance
            obstacles: Obstacle manager
            constraints: Constraint manager
            level: Obstacle level
            
        Returns:
            Dictionary mapping metric names to values
        """
        metrics_result = {}
        
        # Get robot radius from obstacle config
        robot_radius = float(self.config.obstacle_config.get('robot_radius', 0.05))
        
        for metric_name in self.config.metrics:
            try:
                metric_plugin = self.registry.get_plugin('metric', metric_name)
                metric_value = metric_plugin.compute(
                    trajectory, env, obstacles, constraints,
                    robot_radius=robot_radius,
                    level=level,
                    obstacle_config=self.config.obstacle_config,
                    env_plugin=env_plugin,
                )
                metrics_result[metric_name] = convert_to_json_serializable(metric_value)
            except Exception as e:
                print(f"Warning: Failed to compute metric '{metric_name}': {e}")
                metrics_result[metric_name] = None
        
        return metrics_result
    
    def _generate_visualizations(self, result: Dict[str, Any], env: Any, 
                                obstacles: Any, env_plugin: Any) -> None:
        """
        Generate all requested visualizations.
        
        Args:
            result: Experiment result dictionary
            env: Environment instance
            obstacles: Obstacle manager
            env_plugin: Environment plugin
        """
        viz_config = self.config.visualization_config or {}
        
        for viz_name in self.config.visualizations:
            try:
                viz_plugin = self.registry.get_plugin('visualization', viz_name)
            except KeyError:
                print(f"Warning: Visualization plugin '{viz_name}' not found in registry")
                continue
            
            try:
                # Create figure based on visualization type
                import matplotlib.pyplot as plt
                
                # Visualization-specific figure creation
                if viz_name == 'trajectory':
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    viz_plugin.visualize(
                        fig, ax,
                        {
                            'trajectory': result['trajectory'],
                            'env': env,
                            'obstacles': obstacles,
                            'env_plugin': env_plugin,
                        },
                        {**viz_config.get(viz_name, {}), 'config': self.config}
                    )
                    output_path = self._get_output_path(result['level'], result['seed']) / f"{viz_name}.png"
                    viz_plugin.save(output_path, fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                elif viz_name == 'trajectory_3d':
                    from mpl_toolkits.mplot3d import Axes3D
                    fig = plt.figure(figsize=(10, 8))
                    ax = fig.add_subplot(111, projection='3d')
                    viz_plugin.visualize(
                        fig, ax,
                        {
                            'trajectory': result['trajectory'],
                            'env': env,
                            'obstacles': obstacles,
                            'env_plugin': env_plugin,
                        },
                        {**viz_config.get(viz_name, {}), 'config': self.config}
                    )
                    output_path = self._get_output_path(result['level'], result['seed']) / f"{viz_name}.png"
                    viz_plugin.save(output_path, fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                elif viz_name == 'trajectory_modes':
                    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                    viz_plugin.visualize(
                        fig, ax,
                        {
                            'result': result['result'],
                            'env': env,
                            'obstacles': obstacles,
                            'env_plugin': env_plugin,
                        },
                        {**viz_config.get(viz_name, {}), 'config': self.config}
                    )
                    output_path = self._get_output_path(result['level'], result['seed']) / f"{viz_name}.png"
                    viz_plugin.save(output_path, fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                elif viz_name == 'diffusion':
                    # Multiple subplots for diffusion steps
                    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
                    viz_plugin.visualize(
                        fig, axes,
                        {
                            'result': result['result'],
                            'env': env,
                            'obstacles': obstacles,
                            'initial_state': result['result'].get('initial_state'),
                            'env_plugin': env_plugin,
                        },
                        {**viz_config.get(viz_name, {}), 'config': self.config}
                    )
                    output_path = self._get_output_path(result['level'], result['seed']) / f"{viz_name}_steps.png"
                    plt.tight_layout()
                    viz_plugin.save(output_path, fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                elif viz_name == 'diffusion_3d':
                    # Multiple 3D subplots for diffusion steps
                    from mpl_toolkits.mplot3d import Axes3D
                    fig = plt.figure(figsize=(24, 8))
                    axes = []
                    for i in range(3):
                        ax = fig.add_subplot(1, 3, i + 1, projection='3d')
                        axes.append(ax)
                    viz_plugin.visualize(
                        fig, axes,
                        {
                            'result': result['result'],
                            'env': env,
                            'obstacles': obstacles,
                            'initial_state': result['result'].get('initial_state'),
                            'env_plugin': env_plugin,
                        },
                        {**viz_config.get(viz_name, {}), 'config': self.config}
                    )
                    output_path = self._get_output_path(result['level'], result['seed']) / f"{viz_name}_steps.png"
                    plt.tight_layout()
                    viz_plugin.save(output_path, fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                elif viz_name == 'energy_reward':
                    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                    viz_plugin.visualize(
                        fig, axes,
                        {
                            'result': result['result'],
                            'env': env,
                        },
                        {**viz_config.get(viz_name, {}), 'config': self.config}
                    )
                    output_path = self._get_output_path(result['level'], result['seed']) / f"{viz_name}.png"
                    viz_plugin.save(output_path, fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                elif viz_name == 'states':
                    state_dim = env_plugin.get_state_dim()
                    n_cols = 2
                    n_rows = (state_dim + 1) // 2
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
                    if state_dim == 1:
                        axes = [axes]
                    else:
                        axes = axes.ravel()
                    viz_plugin.visualize(
                        fig, axes[:state_dim],
                        {
                            'trajectory': result['trajectory'],
                            'env': env,
                        },
                        {**viz_config.get(viz_name, {}), 'config': self.config}
                    )
                    output_path = self._get_output_path(result['level'], result['seed']) / f"{viz_name}.png"
                    plt.tight_layout()
                    viz_plugin.save(output_path, fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
                elif viz_name == 'scheduler_params':
                    # Create 2x3 grid for scheduler parameters
                    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                    axes = axes.ravel()
                    viz_plugin.visualize(
                        fig, axes,
                        {
                            'result': result['result'],
                            'env': env,
                        },
                        {**viz_config.get(viz_name, {}), 'config': self.config}
                    )
                    output_path = self._get_output_path(result['level'], result['seed']) / f"{viz_name}.png"
                    plt.tight_layout()
                    viz_plugin.save(output_path, fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)
                
            except Exception as e:
                print(f"Warning: Failed to generate visualization '{viz_name}': {e}")
                import traceback
                traceback.print_exc()
    
    def _get_output_path(self, level: int, seed: int) -> Path:
        """
        Get output path for experiment result.
        
        Args:
            level: Obstacle level
            seed: Random seed
            
        Returns:
            Path to output directory
        """
        output_path = self.config.output_dir / f"level_{level}" / f"seed_{seed}"
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _save_result(self, result: Dict[str, Any]) -> None:
        """
        Save individual experiment result.
        
        Maintains backward compatibility with old experiment script format.
        
        Args:
            result: Experiment result dictionary
        """
        output_path = self._get_output_path(result['level'], result['seed'])
        
        # Prepare serializable result
        # Use flattened format compatible with old scripts
        serializable_result = {
            'level': result['level'],
            'seed': result['seed'],
            'planning_time': result['planning_time'],
            'total_time': result.get('total_time', result['planning_time']),
        }
        
        # Add obstacle statistics and CFS flag for backward compatibility
        serializable_result['num_obstacles'] = result.get('num_obstacles', 0)
        serializable_result['num_union_obstacles'] = result.get('num_union_obstacles', 0)
        serializable_result['num_primitives_total'] = result.get('num_primitives_total', 0)
        serializable_result['cfs_enabled'] = result.get('cfs_enabled', False)
        
        # Flatten metrics for backward compatibility
        # Extract SSR metrics to top level (as in old format)
        metrics = result.get('metrics', {})
        if 'ssr' in metrics and metrics['ssr'] is not None:
            ssr_data = metrics['ssr']
            if isinstance(ssr_data, dict):
                # Extract SSR fields to top level for backward compatibility
                serializable_result['ssr'] = float(ssr_data.get('ssr', 0.0))
                serializable_result['safe'] = bool(ssr_data.get('safe', False))
                serializable_result['task_success'] = bool(ssr_data.get('task_success', False))
                serializable_result['distance_to_target'] = float(ssr_data.get('distance_to_target', float('inf')))
                # For constraint feasibility, use 'feasible' if available, otherwise check for 'accel_feasible'
                if 'feasible' in ssr_data:
                    serializable_result['accel_feasible'] = bool(ssr_data.get('feasible', False))
                elif 'accel_feasible' in ssr_data:
                    serializable_result['accel_feasible'] = bool(ssr_data.get('accel_feasible', False))
        
        # Extract obstacle_density to top level
        if 'obstacle_density' in metrics and metrics['obstacle_density'] is not None:
            obs_density = metrics['obstacle_density']
            if isinstance(obs_density, dict):
                serializable_result['obstacle_density'] = float(obs_density.get('obstacle_density', 0.0))
            else:
                serializable_result['obstacle_density'] = float(obs_density)
        
        # Extract nonconvexity metrics
        if 'nonconvexity' in metrics and metrics['nonconvexity'] is not None:
            nonconv_data = metrics['nonconvexity']
            if isinstance(nonconv_data, dict):
                # Store nonconvexity dict (may contain nested 'sdf' key)
                serializable_result['nonconvexity'] = nonconv_data
                # Also extract SDF metrics if present
                if 'sdf' in nonconv_data and isinstance(nonconv_data['sdf'], dict):
                    serializable_result['nonconvexity_sdf'] = nonconv_data['sdf']
        
        # Keep metrics dict for new code compatibility
        serializable_result['metrics'] = metrics
        
        # Add trajectory
        trajectory = result.get('trajectory')
        if trajectory is not None:
            serializable_result['trajectory'] = {
                'states': [s.tolist() if isinstance(s, np.ndarray) else s 
                          for s in trajectory.states],
                'actions': [a.tolist() if isinstance(a, np.ndarray) else a 
                           for a in trajectory.actions],
            }
        
        # Add planning result data if available
        planning_result = result.get('result', {})
        # Standardized rollout fields (for MPC / execution-based methods)
        if isinstance(planning_result, dict):
            if "exec_states" in planning_result:
                serializable_result["exec_states"] = convert_to_json_serializable(planning_result["exec_states"])
            if "exec_actions" in planning_result:
                serializable_result["exec_actions"] = convert_to_json_serializable(planning_result["exec_actions"])
            if "success" in planning_result:
                serializable_result["success"] = convert_to_json_serializable(planning_result["success"])
            if "collision" in planning_result:
                serializable_result["collision"] = convert_to_json_serializable(planning_result["collision"])
            if "planning_time_per_step" in planning_result:
                serializable_result["planning_time_per_step"] = convert_to_json_serializable(planning_result["planning_time_per_step"])
                try:
                    pts = [float(x) for x in planning_result["planning_time_per_step"] if x is not None]
                    serializable_result["avg_planning_time_per_step"] = float(np.mean(pts)) if pts else 0.0
                except Exception:
                    pass

        if 'energies' in planning_result:
            energies = planning_result['energies']
            if hasattr(energies, 'tolist'):
                serializable_result['energies'] = energies.tolist()
            else:
                serializable_result['energies'] = list(energies)
        
        if 'rewards' in planning_result:
            rewards = planning_result['rewards']
            if hasattr(rewards, 'tolist'):
                serializable_result['rewards'] = rewards.tolist()
            else:
                serializable_result['rewards'] = list(rewards)
        
        # Save multi-mode candidate trajectories if available
        if 'candidate_states' in planning_result:
            candidate_states = planning_result['candidate_states']
            candidate_actions = planning_result.get('candidate_actions', [])
            candidate_costs = planning_result.get('candidate_costs', [])
            best_idx = planning_result.get('best_idx', 0)
            
            serializable_result['candidate_states'] = convert_to_json_serializable(candidate_states)
            if candidate_actions:
                serializable_result['candidate_actions'] = convert_to_json_serializable(candidate_actions)
            if len(candidate_costs) > 0:
                if hasattr(candidate_costs, 'tolist'):
                    serializable_result['candidate_costs'] = candidate_costs.tolist()
                else:
                    serializable_result['candidate_costs'] = list(candidate_costs)
            serializable_result['best_idx'] = int(best_idx)
        
        # Save JSON
        with open(output_path / "results.json", 'w') as f:
            json.dump(serializable_result, f, indent=2)
    
    def _save_summary(self, all_results: List[Dict[str, Any]]) -> None:
        """
        Save summary statistics across all experiments.
        
        Args:
            all_results: List of all experiment results
        """
        if not all_results:
            return
        
        # Compute level summaries
        level_summaries = {}
        for level in self.config.obstacle_levels:
            level_results = [r for r in all_results if r['level'] == level]
            if level_results:
                # Compute average metrics
                summary = {
                    'level': level,
                    'num_experiments': len(level_results),
                    'avg_planning_time': float(np.mean([r['planning_time'] for r in level_results])),
                }
                
                # Average metrics
                for metric_name in self.config.metrics:
                    metric_values = []
                    for r in level_results:
                        if metric_name in r['metrics'] and r['metrics'][metric_name] is not None:
                            metric_val = r['metrics'][metric_name]
                            # Handle nested metrics (e.g., SSR dict)
                            if isinstance(metric_val, dict) and 'ssr' in metric_val:
                                metric_values.append(metric_val['ssr'])
                            elif isinstance(metric_val, (int, float)):
                                metric_values.append(metric_val)
                    
                    if metric_values:
                        summary[f'avg_{metric_name}'] = float(np.mean(metric_values))
                
                level_summaries[f'level_{level}'] = summary
                
                # Save level summary
                level_dir = self.config.output_dir / f"level_{level}"
                level_dir.mkdir(parents=True, exist_ok=True)
                with open(level_dir / "summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
        
        # Overall summary
        overall_summary = {
            'total_experiments': len(all_results),
            'avg_planning_time': float(np.mean([r['planning_time'] for r in all_results])),
            'level_summaries': level_summaries,
        }
        
        # Save overall summary
        with open(self.config.output_dir / "overall_summary.json", 'w') as f:
            json.dump(overall_summary, f, indent=2)

