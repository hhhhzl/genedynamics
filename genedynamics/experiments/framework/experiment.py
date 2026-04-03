"""
Experiment execution framework.

This module provides the ExperimentRunner class that orchestrates
experiment execution using the plugin system.
"""

import inspect
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


def _accepts_env(fn: Any) -> bool:
    """True if fn accepts an 'env' argument (e.g. create_energy(self, env=None))."""
    try:
        sig = inspect.signature(fn)
        return "env" in sig.parameters
    except Exception:
        return False

from genedynamics.core.types import Trajectory
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.task_spec import get_default_task_spec

from .config import ExperimentConfig
from .registry import PluginRegistry
from ..common.constraints import create_constraint_pipeline


def convert_to_json_serializable(obj: Any) -> Any:
    """
    Convert numpy types and other non-JSON-serializable types to Python native types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (np.integer, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8, np.uint16,
                        np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        v = float(obj)
        if not np.isfinite(v):
            return None
        return v
    elif isinstance(obj, float):
        if not np.isfinite(obj):
            return None
        return obj
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


def _has_items(obj: Any) -> bool:
    """Robust non-empty check for list/tuple/ndarray-like containers."""
    if obj is None:
        return False
    try:
        return len(obj) > 0  # works for list/tuple/ndarray
    except Exception:
        return bool(obj)


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
        env_name = getattr(self.config, "env_name", "") or ""
        if "brax" in env_name.lower() and self.config.backend != "jax":
            raise ValueError(
                f"Brax env '{env_name}' requires JAX backend. "
                f"Set backend: jax in config (got: {self.config.backend})."
            )
        RuntimeBackendManager.set_backend(self.config.backend, device=self.config.device)
        backend = RuntimeBackendManager.get_backend()
        
        # 2. Setup environment
        env_plugin = self.registry.get_plugin('environment', self.config.env_name)
        
        # 3. Generate start/target positions (before obstacles, for obstacle generation)
        temp_env = env_plugin.create_env(self.config.env_params)
        target_pos = np.asarray(temp_env.target, dtype=np.float32)
        obstacle_gen_name = self.config.obstacle_config.get('generator', 'box2d')
        obstacle_gen = self.registry.get_plugin('obstacle_generator', obstacle_gen_name)
        obstacle_config_with_env = {
            **self.config.obstacle_config,
            'env_name': self.config.env_name,
        }
        # 保证 start/goal 为圆心、robot_radius 为半径的圆不在障碍上；失败则重试采样 start
        max_start_retries = 5
        obstacles = None
        for start_retry in range(max_start_retries):
            start_pos = self._generate_start_position(
                level, seed + 1000 * start_retry, temp_env, env_plugin
            )
            try:
                obstacles = obstacle_gen.generate(
                    level, seed, start_pos, target_pos, obstacle_config_with_env
                )
                break
            except RuntimeError as e:
                if "Start position" in str(e) or "Goal/target position" in str(e):
                    if start_retry == max_start_retries - 1:
                        raise
                    continue
                raise
        assert obstacles is not None, "obstacle_gen.generate did not return"
        
        # 5. Create environment with obstacles (for physics backends that need obstacles in model)
        env_params_with_obstacles = {**self.config.env_params}
        # Add obstacles to env_params if using physics backend that needs them
        physics_backend = env_params_with_obstacles.get('physics_backend', None)
        if physics_backend in ['mujoco', 'mjx', 'isaac', 'brax'] and len(obstacles) > 0:
            env_params_with_obstacles['obstacles'] = obstacles
        # For D3IL avoiding envs: pass obstacle info for MuJoCo scene sync.
        if self.config.env_name in ['d3il_avoiding_9d', 'd3il_avoiding']:
            env_params_with_obstacles['obstacle_level'] = level
            env_params_with_obstacles['obstacle_radius_by_level'] = (
                self.config.obstacle_config.get('obstacle_radius_by_level')
            )
            env_params_with_obstacles['obstacles'] = obstacles
            # EE-only collision settings are only used by the 9D env implementation.
            if self.config.env_name == 'd3il_avoiding_9d':
                env_params_with_obstacles['robot_radius'] = float(
                    self.config.obstacle_config.get('robot_radius', 0.05)
                )
                env_params_with_obstacles['collision_ee_only'] = True

        env = env_plugin.create_env(env_params_with_obstacles)
        energy = env_plugin.create_energy(env) if _accepts_env(env_plugin.create_energy) else env_plugin.create_energy()
        
        # Build SDF texture if needed (for CBF/MDOC/CFS filters that use sample_sdf_and_grad_2d)
        # Level 0 must also build texture so obstacles are seen by the filter
        if len(obstacles) > 0:
            physics_backend = self.config.env_params.get('physics_backend', None)
            is_3d_env = (
                self.config.env_name in ['drone_box_3d', 'drone', 'drone_full_3d', 'drone_full_3d_physics',
                                         'drone_full_3d_mujoco', 'drone_full_3d_isaac',
                                         'quadruped_flat_mjx', 'quadruped_go2_mjx',
                                         'quadruped_go2_brax', 'humanoid_run_brax',
                                         'humanoid_simplified_mjx', 'humanoid_g1_mjx'] or
                self.config.obstacle_config.get('generator', '') == 'box3d' or
                physics_backend in ['mujoco', 'mjx', 'isaac']
            )
            if not is_3d_env:
                map_bounds = self.config.obstacle_config.get('map_bounds', {})
                obstacles.build_sdf_texture_2d(
                    x_min=float(map_bounds.get('x_min', -2.0)),
                    x_max=float(map_bounds.get('x_max', 2.0)),
                    y_min=float(map_bounds.get('y_min', -2.0)),
                    y_max=float(map_bounds.get('y_max', 2.0)),
                    res=0.01,
                    force_rebuild=True,
                )
                # Pre-warm SDF texture to JAX when using JAX backend (avoids tracer leaks in JIT)
                if getattr(self.config, 'backend', None) == 'jax':
                    tex = obstacles.get_sdf_texture_2d()
                    if tex is not None and hasattr(tex, 'to_jax'):
                        tex.to_jax()
        
        # 6. Setup constraints
        constraint_config = self.config.constraint_config or {}
        
        # Use new pipeline architecture (preferred)
        constraint_pipeline = create_constraint_pipeline(
            obstacles, level, env, constraint_config, self.config.backend,
            obstacle_config=self.config.obstacle_config,
            method_params=self.config.method_params,
        )
        
        # Legacy constraint_manager removed; rely on pipeline only.
        
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
            'constraint_pipeline': constraint_pipeline,  # New architecture (preferred)
            'scheduler': scheduler,  # New scheduler system
            'obstacles': obstacles,  # Provide obstacles to methods that can use fast SDF (e.g., EB-MBD)
            'obstacle_config': self.config.obstacle_config,  # Provide robot_radius/map bounds, etc.
            'np_random_seed': seed,  # Pass seed for reproducibility
            'env_plugin': env_plugin,  # For TaskSpec / position_extractor
            'env_name': getattr(self.config, 'env_name', None),
        }
        # Merge first diffusion_scheduler's M_k / Ndiffuse / T_k / beta into method_config
        # so method plugins (MDOC, MBD, etc.) use YAML diffusion_schedulers values instead of defaults
        if scheduler is not None and getattr(scheduler, 'diffusion_schedulers', None):
            ds_list = scheduler.diffusion_schedulers
            if ds_list:
                from genedynamics.core.constraints.core.types import ScheduleState
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
            # Quadruped MJX has heavy JIT traces; always warmup to exclude compile from planning_time
            if getattr(self.config, "env_name", "") in (
                "quadruped_go2_mjx", "quadruped_flat_mjx",
                "quadruped_go2_brax", "humanoid_run_brax",
            ):
                needs_warmup = True

            # Constraint pipeline warmup is handled by planner/backends if needed.
        
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

        # 8b. Recompute best_idx: prefer (safe AND success) + lowest cost; else lowest cost.
        # D3IL: best_idx is set in 8b2 from execution results (among exec success & no collision, lowest cost).
        robot_radius = float(self.config.obstacle_config.get('robot_radius', 0.05))
        success_margin = float(self.config.obstacle_config.get('success_margin', 2 * robot_radius))
        cand_states = result.get('candidate_states', [])
        cand_actions = result.get('candidate_actions', [])
        cand_costs = result.get('candidate_costs', None)
        has_cand_states = _has_items(cand_states)
        has_cand_actions = _has_items(cand_actions)
        is_d3il_style = (
            getattr(self.config, 'method', None) == 'd3il_unified'
            or 'd3il' in str(getattr(self.config, 'env_name', ''))
        )
        if has_cand_states and has_cand_actions and cand_costs is not None:
            if not is_d3il_style:
                # Non-D3IL: plan-based best selection and overwrite now
                if result.get('states') is not None:
                    result['executed_states'] = [np.asarray(s, dtype=np.float32) for s in result['states']]
                if result.get('actions') is not None:
                    acts = result['actions']
                    if isinstance(acts, (list, tuple)):
                        result['executed_actions'] = [np.asarray(a, dtype=np.float32) for a in acts]
                    else:
                        act_arr = np.asarray(acts, dtype=np.float32)
                        result['executed_actions'] = [act_arr[i] for i in range(act_arr.shape[0])] if act_arr.ndim >= 2 else [act_arr]
                best_idx = self._compute_best_idx_from_candidates(
                    list(cand_states), np.asarray(cand_costs), env, obstacles,
                    env_plugin, robot_radius, success_margin,
                )
                result['best_idx'] = best_idx
                best_states = cand_states[best_idx]
                best_actions = cand_actions[best_idx]
                result['states'] = [np.asarray(s, dtype=np.float32) for s in best_states]
                act_arr = np.asarray(best_actions, dtype=np.float32)
                result['actions'] = [act_arr[t] for t in range(act_arr.shape[0])] if act_arr.ndim >= 2 else [act_arr]
                multirun_diff = result.get('multirun_diffusion_data', [])
                if multirun_diff and 0 <= best_idx < len(multirun_diff):
                    dd = multirun_diff[best_idx]
                    if dd.get('diffusion_actions_traj') is not None:
                        result['diffusion_actions_traj'] = dd['diffusion_actions_traj']
                    if dd.get('diffusion_sampled_actions') is not None:
                        result['diffusion_sampled_actions'] = dd['diffusion_sampled_actions']

        # 8b2. D3IL: run execution for every candidate; then set best_idx by execution (success & no collision, lowest cost)
        if is_d3il_style and has_cand_states and cand_costs is not None:
            from genedynamics.experiments.common.d3il_mpc import run_plan_once_episode
            method_params = getattr(self.config, 'method_params', None) or {}
            if not isinstance(method_params, dict):
                method_params = {}
            track_trajectory = bool(method_params.get('track_trajectory', False))
            tracker_k_joint = float(method_params.get('tracker_k_joint', 0.0))
            tracker_k_xy = float(method_params.get('tracker_k_xy', 1.0))
            tracker_ff_alpha = float(method_params.get('tracker_ff_alpha', 0.0))
            max_steps = int(method_params.get('max_episode_length', getattr(env, 'horizon', 150)))
            reset_rng = backend.create_rng(seed)
            exec_candidate_states = []
            exec_candidate_actions = []
            exec_candidate_states_9d = []
            exec_success = []
            exec_collision = []
            costs_arr = np.asarray(cand_costs, dtype=np.float64).ravel()
            for c in range(len(cand_states)):
                states_c = cand_states[c]
                actions_c = cand_actions[c] if (has_cand_actions and c < len(cand_actions)) else None
                if actions_c is None:
                    act_dim = getattr(env, 'act_dim', 7)
                    actions_c = [np.zeros(act_dim, dtype=np.float32) for _ in range(max(0, len(states_c) - 1))]
                else:
                    actions_c = np.asarray(actions_c, dtype=np.float32)
                    if actions_c.ndim == 2:
                        actions_c = [actions_c[i] for i in range(actions_c.shape[0])]
                    else:
                        actions_c = list(actions_c) if isinstance(actions_c, (list, tuple)) else [actions_c]

                def make_plan_fn(planned_states, planned_actions):
                    def plan_once_fn(_planner, _x0, _rng, horizon=None):
                        return {"actions": planned_actions, "states": planned_states}
                    return plan_once_fn

                plan_once_c = make_plan_fn(states_c, actions_c)
                try:
                    run_out = run_plan_once_episode(
                        exec_env=env,
                        planner=planner,
                        plan_once_fn=plan_once_c,
                        rng=reset_rng,
                        max_steps=max_steps,
                        track_trajectory=track_trajectory,
                        tracker_k_joint=tracker_k_joint,
                        tracker_k_xy=tracker_k_xy,
                        tracker_ff_alpha=tracker_ff_alpha,
                    )
                    exec_candidate_states.append(run_out.get('states', []))
                    exec_candidate_actions.append(run_out.get('actions', []))
                    exec_candidate_states_9d.append(run_out.get('states_9d'))
                    exec_success.append(bool(run_out.get('success', False)))
                    exec_collision.append(bool(run_out.get('collision', True)))
                except Exception:
                    exec_candidate_states.append(result.get('states', []))
                    exec_candidate_actions.append([])
                    exec_candidate_states_9d.append(None)
                    exec_success.append(False)
                    exec_collision.append(True)
            result['exec_candidate_states'] = exec_candidate_states
            result['exec_candidate_states_9d'] = exec_candidate_states_9d
            # Best = lowest cost among (execution success and no collision); else lowest cost overall
            valid = [c for c in range(len(cand_states)) if exec_success[c] and not exec_collision[c]]
            if valid:
                best_idx = int(valid[np.argmin(costs_arr[valid])])
            else:
                best_idx = int(np.argmin(costs_arr))
            result['best_idx'] = best_idx
            if _has_items(exec_candidate_states_9d) and best_idx < len(exec_candidate_states_9d):
                best_9d = exec_candidate_states_9d[best_idx]
                if best_9d is not None and len(best_9d) >= 2:
                    result['states_9d'] = [np.asarray(s, dtype=np.float32) for s in best_9d]
            result['states'] = [np.asarray(s, dtype=np.float32) for s in cand_states[best_idx]]
            act_arr = np.asarray(cand_actions[best_idx], dtype=np.float32)
            result['actions'] = [act_arr[t] for t in range(act_arr.shape[0])] if act_arr.ndim >= 2 else [act_arr]
            result['executed_states'] = [np.asarray(s, dtype=np.float32) for s in exec_candidate_states[best_idx]]
            n_exec = len(result['executed_states']) - 1
            exec_acts = exec_candidate_actions[best_idx] if best_idx < len(exec_candidate_actions) else []
            if isinstance(exec_acts, (list, tuple)) and len(exec_acts) >= n_exec:
                result['executed_actions'] = [np.asarray(exec_acts[i], dtype=np.float32) for i in range(n_exec)]
            elif isinstance(exec_acts, (list, tuple)) and len(exec_acts) > 0:
                result['executed_actions'] = [np.asarray(a, dtype=np.float32) for a in exec_acts]
                act_dim = len(result['executed_actions'][0].ravel()) if result['executed_actions'] else getattr(env, 'act_dim', 7)
                while len(result['executed_actions']) < n_exec:
                    result['executed_actions'].append(np.zeros(act_dim, dtype=np.float32))
            else:
                planned = result['actions']
                result['executed_actions'] = [np.asarray(planned[i], dtype=np.float32) for i in range(min(n_exec, len(planned)))]
                act_dim = len(result['executed_actions'][0].ravel()) if result['executed_actions'] else getattr(env, 'act_dim', 7)
                while len(result['executed_actions']) < n_exec:
                    result['executed_actions'].append(np.zeros(act_dim, dtype=np.float32))
            result['success'] = exec_success[best_idx]
            result['collision'] = exec_collision[best_idx]
            result['exec_success_per_mode'] = list(exec_success)
            result['exec_collision_per_mode'] = list(exec_collision)
            # Execution SSR = (executions that were success and no collision) / total (violation recomputed in _compute_metrics)
            exec_ssr_count = sum(1 for i in range(len(exec_success)) if exec_success[i] and not exec_collision[i])
            n_exec_total = len(exec_success)
            result['execution_ssr'] = {
                'execution_ssr': exec_ssr_count / max(1, n_exec_total),
                'execution_ssr_count': exec_ssr_count,
            }
            multirun_diff = result.get('multirun_diffusion_data', [])
            if multirun_diff and 0 <= best_idx < len(multirun_diff):
                dd = multirun_diff[best_idx]
                if dd.get('diffusion_actions_traj') is not None:
                    result['diffusion_actions_traj'] = dd['diffusion_actions_traj']
                if dd.get('diffusion_sampled_actions') is not None:
                    result['diffusion_sampled_actions'] = dd['diffusion_sampled_actions']

        # 8b1.5. D3IL: 3D GIF from *executed* states (after 8b2 so best is execution-based)
        _states_3d = result.get('executed_states') or result.get('states')
        info_dict = result.get("info", {}) if isinstance(result.get("info"), dict) else {}
        # For 4D DPCC runs, states are 4D; prefer lifted 9D states for robot 3D GIF.
        _has_states = _states_3d is not None and (len(_states_3d) if hasattr(_states_3d, '__len__') else 0) > 0
        if _has_states:
            try:
                s0 = np.asarray(_states_3d[0], dtype=np.float32).reshape(-1)
                if s0.size < 9:
                    lifted = result.get("states_9d", info_dict.get("states_9d"))
                    if lifted:
                        _states_3d = lifted
            except Exception:
                lifted = result.get("states_9d", info_dict.get("states_9d"))
                if lifted:
                    _states_3d = lifted

        if is_d3il_style and _has_states and len(_states_3d) >= 2:
            try:
                out_dir = self._get_output_path(level, seed)
                traj_dir = out_dir / "trajectory"
                traj_dir.mkdir(parents=True, exist_ok=True)
                self._render_d3il_exec_3d_gif(
                    env, {'result': {'states': _states_3d}, 'seed': seed}, traj_dir, seed,
                    use_current_scene=True, n_interp=2,
                )
            except Exception:
                pass

        # 8b1.6. D3IL: top-5 exec 3D GIFs (rank 1 = best already above; add rank2..rank5)
        exec_candidate_states = result.get('exec_candidate_states') or []
        exec_candidate_states_9d = result.get('exec_candidate_states_9d') or []
        cand_costs = result.get('candidate_costs')
        if isinstance(cand_costs, np.ndarray):
            costs_arr = np.asarray(cand_costs, dtype=np.float64).ravel()
        else:
            costs_arr = np.array(cand_costs or [], dtype=np.float64)
        exec_success = result.get('exec_success_per_mode') or []
        exec_collision = result.get('exec_collision_per_mode') or []
        n_modes = len(exec_candidate_states)
        if (
            is_d3il_style
            and n_modes >= 2
            and len(costs_arr) >= n_modes
            and len(exec_success) >= n_modes
            and len(exec_collision) >= n_modes
        ):
            valid = [c for c in range(n_modes) if exec_success[c] and not exec_collision[c]]
            invalid = [c for c in range(n_modes) if c not in valid]
            valid_sorted = [valid[i] for i in np.argsort(costs_arr[valid])] if valid else []
            invalid_sorted = [invalid[i] for i in np.argsort(costs_arr[invalid])] if invalid else []
            top5_indices = (valid_sorted + invalid_sorted)[:5]
            out_dir = self._get_output_path(level, seed)
            traj_dir = out_dir / "trajectory"
            traj_dir.mkdir(parents=True, exist_ok=True)
            for rank_one_based in range(2, 6):
                if rank_one_based - 1 >= len(top5_indices):
                    break
                idx = top5_indices[rank_one_based - 1]
                states_rank = list(exec_candidate_states[idx]) if idx < len(exec_candidate_states) else []
                if len(states_rank) < 2:
                    continue
                states_9d_rank = None
                if idx < len(exec_candidate_states_9d) and exec_candidate_states_9d[idx] is not None:
                    states_9d_rank = exec_candidate_states_9d[idx]
                if states_9d_rank and len(states_9d_rank) >= 2:
                    _states_3d_rank = [np.asarray(s, dtype=np.float32) for s in states_9d_rank]
                else:
                    try:
                        s0 = np.asarray(states_rank[0], dtype=np.float32).reshape(-1)
                        if s0.size >= 9:
                            _states_3d_rank = [np.asarray(s, dtype=np.float32) for s in states_rank]
                        else:
                            _states_3d_rank = None
                    except Exception:
                        _states_3d_rank = None
                if _states_3d_rank is None or len(_states_3d_rank) < 2:
                    continue
                try:
                    self._render_d3il_exec_3d_gif(
                        env,
                        {'result': {'states': _states_3d_rank}, 'seed': seed},
                        traj_dir,
                        seed,
                        use_current_scene=True,
                        n_interp=2,
                        filename_suffix=f"rank{rank_one_based}",
                    )
                except Exception:
                    pass

        # 8c. Multirun: re-run single plan with best key to get exact best trajectory planning_time
        if result.get('mode_strategy', '').lower() == 'multirun' and 'multirun_keys' in result:
            multirun_keys = result['multirun_keys']
            best_idx = int(result.get('best_idx', 0))
            if 0 <= best_idx < len(multirun_keys):
                best_key = multirun_keys[best_idx]
                # Ensure JAX PRNGKey so re-run reproduces the same trajectory (keys may be numpy/list after copy)
                try:
                    import jax.numpy as jnp
                    if hasattr(best_key, '__len__') and len(best_key) == 2:
                        best_key = jnp.asarray([int(best_key[0]), int(best_key[1])], dtype=jnp.uint32)
                    else:
                        best_key = jnp.asarray(best_key, dtype=jnp.uint32)
                except Exception:
                    pass
                print(f"[experiment] Re-running plan with best_idx={best_idx} for exact planning_time (key for selected trajectory)")
                # Force single plan (avoid plan_batch): temporarily set num_modes=1
                config = getattr(planner, 'config', None)
                old_num_modes = None
                if isinstance(config, dict) and 'num_modes' in config:
                    old_num_modes = config['num_modes']
                    config['num_modes'] = 1
                try:
                    t0 = time.time()
                    method_plugin.plan(planner, start_pos, best_key)
                    result['best_planning_time'] = float(time.time() - t0)
                except Exception:
                    pass
                finally:
                    if old_num_modes is not None and isinstance(config, dict):
                        config['num_modes'] = old_num_modes

        # 9. Extract trajectory (D3IL: use executed_states so trajectory_best_exec shows real execution)
        is_d3il_style = (
            getattr(self.config, 'method', None) == 'd3il_unified'
            or 'd3il' in str(getattr(self.config, 'env_name', ''))
        )
        if is_d3il_style and result.get('executed_states') is not None:
            trajectory = self._extract_trajectory_from_executed(result, env)
        else:
            trajectory = self._extract_trajectory(result, env)

        # 10. Compute metrics
        metrics = self._compute_metrics(
            trajectory, env, obstacles, constraint_pipeline, level,
            env_plugin=env_plugin, planning_result=result, planning_time=planning_time,
        )
        
        # 11. Prepare results
        # Add obstacle statistics for backward compatibility
        num_obstacles = len(obstacles) if obstacles else 0
        num_union_obstacles = 0
        num_primitives_total = 0
        if obstacles and num_obstacles > 0:
            from genedynamics.envs.obstacles.nonconvex import UnionObstacle
            for obs in list(obstacles):
                if isinstance(obs, UnionObstacle):
                    num_union_obstacles += 1
                    num_primitives_total += len(obs.obstacles)
                else:
                    num_primitives_total += 1
        
        # Check if CFS is enabled
        cfs_enabled = bool(constraint_pipeline is not None)
        
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
            print("Generating visualizations (folders/images)...")
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
                print(f"\n[experiment] Running level={level}, seed={seed}...")
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

        if getattr(self.config, "auto_report", True):
            self._generate_report()

        return all_results

    def _generate_report(self) -> None:
        """Generate unified report from results (industrial-grade pipeline)."""
        try:
            from genedynamics.reports import generate_report
            results_root = self.config.output_dir
            for _ in range(3):
                if (results_root / "deploy").exists():
                    break
                results_root = results_root.parent
                if results_root == results_root.parent:
                    break
            report_dir = results_root.parent / "reports"
            out = generate_report(results_root, report_dir, formats=["html"])
            if out:
                print(f"\n[report] Generated: {list(out.values())[0]}")
        except ImportError as e:
            print(f"\n[report] Skip (jinja2 required): {e}")
        except Exception as e:
            print(f"\n[report] Error: {e}")
    
    def _generate_start_position(self, level: int, seed: int, env: Any, env_plugin: Any) -> np.ndarray:
        """
        Generate start position based on level and seed.
        
        For 2D: start is sampled in the hardcoded box x in [-1, 0], y in [-1.5, -2],
        with an inner margin of robot_radius so the robot fits. For 1D/3D: use
        level/seed-based random direction and distance from target.
        
        Args:
            level: Obstacle level
            seed: Random seed
            env: Environment instance
            env_plugin: Environment plugin
            
        Returns:
            Start position array (may be full state vector with velocities)
        """
        # d3il_avoiding (4D): use the env reset observation directly.
        # The generic 2D sampler (x in [-1,0], y in [-2,-1.5]) is for toy 2D environments and
        # produces out-of-map starts for D3IL.
        if self.config.env_name == "d3il_avoiding":
            obs, _info = env.reset(seed=seed)
            return np.asarray(obs, dtype=np.float32)

        # d3il_avoiding_9d: fixed center start (0.5, -0.28) - center x, below first obstacle row
        if self.config.env_name == "d3il_avoiding_9d":
            state_dim = env_plugin.get_state_dim()
            start_xy = np.array([0.5, -0.28], dtype=np.float32)
            start_full = np.concatenate([start_xy, np.zeros(state_dim - 2, dtype=np.float32)])
            return start_full.astype(np.float32)

        # Stepping-stones: environment reset provides stance-consistent start feet.
        if self.config.env_name == "quadruped_stepping_stones_2d":
            obs, _info = env.reset(seed=seed)
            return np.asarray(obs, dtype=np.float32)

        # Humanoid corridor 2D: planning-level env with built-in start position.
        if self.config.env_name == "humanoid_corridor_2d":
            obs, _info = env.reset(seed=seed)
            return np.asarray(obs, dtype=np.float32)

        # Quadruped/Humanoid MJX: need valid [qpos; qvel] from reset, then override base xyz
        env_name_lower = (self.config.env_name or "").lower()
        if ("quadruped" in env_name_lower or "humanoid" in env_name_lower) and (
            "mjx" in env_name_lower or "brax" in env_name_lower
        ):
            state_full, _ = env.reset(seed=seed)
            state_full = np.asarray(state_full, dtype=np.float32).copy()
            # Sample 3D start position (same logic as 3D below)
            np.random.seed(seed)
            target_pos = np.asarray(env.target, dtype=np.float32).flatten()[:3]
            metadata = getattr(self.config, "metadata", {}) or {}
            map_bounds = (getattr(self.config, 'obstacle_config', None) or {}).get('map_bounds', {})
            x_min = float(map_bounds.get('x_min', -2.0))
            x_max = float(map_bounds.get('x_max', 2.0))
            y_min = float(map_bounds.get('y_min', -2.0))
            y_max = float(map_bounds.get('y_max', 2.0))
            z_min = float(map_bounds.get('z_min', 0.0))
            z_max = float(map_bounds.get('z_max', 2.0))
            # Allow task-specific start distance overrides without changing config schema.
            max_distance = float(np.linalg.norm(np.array([x_max - x_min, y_max - y_min, max(0.5, z_max - z_min)])))
            level_scale = float(metadata.get("start_dist_level_scale", 0.0))
            if "start_min_dist" in metadata:
                min_dist = float(metadata.get("start_min_dist", 0.5)) + (level / 10.0) * level_scale
            else:
                min_dist = 0.5 + (level / 10.0) * 1.0
            if "start_max_dist" in metadata:
                max_dist = float(metadata.get("start_max_dist", 1.0)) + (level / 10.0) * level_scale
            else:
                max_dist = 1.0 + (level / 10.0) * (max_distance - 1.0)
            max_dist = min(max_dist, max_distance * 0.9)
            min_dist = max(0.1, min(min_dist, max_dist - 1e-3))
            distance = np.random.uniform(min_dist, max_dist)
            theta = np.random.uniform(0, 2 * np.pi)
            # For legged locomotion on flat terrain, sample start in XY plane only.
            direction_xy = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
            start_strategy = str(metadata.get("start_strategy", "")).strip().lower()
            if start_strategy == "behind_target_x":
                # Force starts mostly behind target on x-axis for clear forward locomotion videos.
                direction_xy = np.array([-1.0, 0.0], dtype=np.float32)
            start_xyz = target_pos.copy()
            start_xyz[:2] = target_pos[:2] + distance * direction_xy
            start_xyz[0] = np.clip(start_xyz[0], x_min + 0.1, x_max - 0.1)
            start_xyz[1] = np.clip(start_xyz[1], y_min + 0.1, y_max - 0.1)
            # Keep nominal standing height from env.reset(), with optional override.
            z_nominal = float(state_full[2])
            z_nominal = float(metadata.get("start_nominal_z", z_nominal))
            z_lo = max(z_min + 0.05, z_nominal - 0.05)
            z_hi = min(z_max - 0.1, z_nominal + 0.05)
            if z_hi < z_lo:
                z_mid = float(np.clip(z_nominal, z_min + 0.02, z_max - 0.02))
                z_lo, z_hi = z_mid, z_mid
            start_xyz[2] = np.clip(z_nominal, z_lo, z_hi)
            state_full[:3] = start_xyz
            return state_full.astype(np.float32)

        np.random.seed(seed)
        target_pos = np.asarray(env.target, dtype=np.float32)
        p_max = getattr(env, 'p_max', 2.0)
        obstacle_config = getattr(self.config, 'obstacle_config', None) or {}
        robot_radius = float(obstacle_config.get('robot_radius', 0.05))
        
        # Extract target position (first 2 or 3 elements depending on environment)
        target_pos_flat = target_pos.flatten()
        pos_dim = min(len(target_pos_flat), 3)  # Support up to 3D positions
        target_pos_only = target_pos_flat[:pos_dim]
        
        # 2D: hardcoded start box x in [-1, 0], y in [-1.5, -2]; inner margin for robot
        # y 轴下界 -2：保证 robot 中心 y >= -2 + robot_radius，整机不超出 y=-2
        if pos_dim == 2:
            x_min, x_max = -1.0, 0.0
            y_low_bound = -2.0  # map 下界，robot 不得超出
            y_max = -1.5
            margin = robot_radius
            x_lo = x_min + margin
            x_hi = x_max - margin
            y_lo = y_low_bound + margin  # 中心至少 -2 + robot_radius
            y_hi = y_max - margin
            if x_lo < x_hi and y_lo < y_hi:
                start = np.array([
                    np.random.uniform(x_lo, x_hi),
                    np.random.uniform(y_lo, y_hi),
                ], dtype=np.float32)
            else:
                start = np.array([(x_lo + x_hi) / 2, (y_lo + y_hi) / 2], dtype=np.float32)
            # 显式裁剪，保证 robot 不超出 y=-2
            start[1] = max(float(start[1]), y_low_bound + robot_radius)
            state_dim = env_plugin.get_state_dim()
            if state_dim > pos_dim:
                start_full = np.concatenate([start, np.zeros(state_dim - pos_dim, dtype=np.float32)])
            else:
                start_full = start
            return start_full.astype(np.float32)
        
        # 1D/3D: distance from target increases with level
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

    def _extract_trajectory_from_executed(self, result: Dict[str, Any], env: Any) -> Trajectory:
        """
        Build trajectory from executed_states/executed_actions (D3IL: so trajectory_best_exec shows real execution).
        """
        if result is None:
            raise ValueError("Cannot extract trajectory from None result")
        states_list = [np.asarray(s, dtype=np.float32) for s in result.get('executed_states', [])]
        if len(states_list) == 0:
            return self._extract_trajectory(result, env)
        n_act = len(states_list) - 1
        exec_actions = result.get('executed_actions')
        if exec_actions is not None and len(exec_actions) >= n_act:
            actions_list = [np.asarray(exec_actions[i], dtype=np.float32) for i in range(n_act)]
        else:
            fallback = result.get('actions', [])
            act_dim = getattr(env, 'act_dim', 2)
            if _has_items(fallback) and len(fallback) >= n_act:
                actions_list = [np.asarray(fallback[i], dtype=np.float32) for i in range(n_act)]
            else:
                actions_list = []
                for i in range(n_act):
                    if _has_items(fallback) and i < len(fallback):
                        actions_list.append(np.asarray(fallback[i], dtype=np.float32))
                    else:
                        actions_list.append(np.zeros(act_dim, dtype=np.float32))
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

    def _extract_plan_trajectory(self, result: Dict[str, Any], env: Any) -> Optional[Trajectory]:
        """Extract best planned trajectory from candidate_states/candidate_actions (for D3IL plan viz)."""
        if result is None:
            return None
        cand_states = result.get('candidate_states', [])
        cand_actions = result.get('candidate_actions', [])
        best_idx = int(result.get('best_idx', 0))
        if (not _has_items(cand_states)) or best_idx < 0 or best_idx >= len(cand_states):
            return None
        states_list = [np.asarray(s, dtype=np.float32) for s in cand_states[best_idx]]
        if _has_items(cand_actions) and best_idx < len(cand_actions):
            act = cand_actions[best_idx]
            act = np.asarray(act, dtype=np.float32)
            actions_list = [act[i] for i in range(act.shape[0])] if act.ndim >= 2 else [act]
        else:
            act_dim = getattr(env, 'act_dim', 2)
            actions_list = [np.zeros(act_dim, dtype=np.float32) for _ in range(len(states_list) - 1)]
        if len(actions_list) < len(states_list) - 1:
            act_dim = getattr(env, 'act_dim', 2)
            while len(actions_list) < len(states_list) - 1:
                actions_list.append(np.zeros(act_dim, dtype=np.float32))
        return Trajectory(states=states_list, actions=actions_list, info=None)

    def _is_d3il_experiment(self, result: Dict[str, Any]) -> bool:
        """True if this experiment is a D3IL env (emit plan/exec split trajectory files)."""
        cfg = result.get('config_snapshot') or {}
        method = cfg.get('method') if isinstance(cfg, dict) else getattr(cfg, 'method', None)
        env_name = cfg.get('env_name', '') if isinstance(cfg, dict) else getattr(cfg, 'env_name', '')
        return (method == 'd3il_unified') or ("d3il" in str(env_name))

    def _render_d3il_exec_3d_gif(
        self,
        env: Any,
        result: Dict[str, Any],
        trajectory_dir: Any,
        seed: int,
        width: int = 640,
        height: int = 480,
        duration_ms: float = 80.0,
        use_current_scene: bool = False,
        n_interp: int = 1,
        filename_suffix: Optional[str] = None,
    ) -> None:
        """
        Render 3D GIF by setting the sim to each *recorded* state (pose from states), then capture.
        use_current_scene=True: do not reset; scene is from main run. Call before 8b2.
        n_interp: number of linear interpolated poses between consecutive states (1 = no interp).
        filename_suffix: if set (e.g. "rank2"), output is trajectory_best_exec_3d_rank2.gif; else trajectory_best_exec_3d.gif.
        """
        try:
            inner = getattr(getattr(env, "_task_env", None), "_env", None)
            if inner is None or not hasattr(inner, "bp_cam"):
                return
            cam = getattr(inner, "bp_cam", None)
            robot = getattr(inner, "robot", None)
            if cam is None or not hasattr(cam, "_get_img_data"):
                return
            if robot is None or not hasattr(robot, "set_q"):
                return
            plan_result = result.get("result", {})
            states = plan_result.get("states")
            # Fallback: methods that keep lifted 9D trajectory in info (e.g., SafeDiffuser 4D->9D lift)
            if (
                (not states or len(states) == 0)
                and isinstance(plan_result.get("info"), dict)
                and plan_result["info"].get("states_9d") is not None
            ):
                states = plan_result["info"].get("states_9d")
            if states and len(states) > 0:
                s0 = np.asarray(states[0], dtype=np.float64).reshape(-1)
                if s0.size < 9 and isinstance(plan_result.get("info"), dict):
                    states_9d = plan_result["info"].get("states_9d")
                    if states_9d is not None and len(states_9d) > 0:
                        states = states_9d
            if not states:
                return
            states = [np.asarray(s, dtype=np.float64).reshape(-1).copy() for s in states]
            if len(states) < 2:
                return
            if states[0].size < 9:
                return
            n_interp = max(1, int(n_interp))
            poses = []
            for i in range(len(states)):
                q = np.asarray(states[i][2:9], dtype=np.float64, order='C').reshape(-1)
                if q.size != 7:
                    continue
                poses.append(q)
                if n_interp > 1 and i < len(states) - 1:
                    q_next = np.asarray(states[i + 1][2:9], dtype=np.float64, order='C').reshape(-1)
                    if q_next.size == 7:
                        for k in range(1, n_interp):
                            alpha = k / n_interp
                            poses.append((1 - alpha) * q + alpha * q_next)
            if len(poses) < 2:
                return
            if not use_current_scene:
                backend = RuntimeBackendManager.get_backend()
                rng = backend.create_rng(seed)
                env.reset(rng=rng)
            frames_rgb = []
            last_ok_frame = None
            for q in poses:
                try:
                    q_ = np.asarray(q, dtype=np.float64, order='C').reshape(7)
                    robot.set_q(q_)
                    rgb = cam._get_img_data(width=width, height=height, depth=False)
                    if rgb is not None:
                        frame = np.asarray(rgb, dtype=np.uint8)
                        frames_rgb.append(frame)
                        last_ok_frame = frame
                    elif last_ok_frame is not None:
                        frames_rgb.append(np.asarray(last_ok_frame, dtype=np.uint8))
                except Exception:
                    # Do not break: keep rendering so we get full trajectory including goal.
                    # Use last successful frame for this pose (e.g. near-goal poses can fail in level 1/2).
                    if last_ok_frame is not None:
                        frames_rgb.append(np.asarray(last_ok_frame, dtype=np.uint8))
            if len(frames_rgb) == 1:
                frames_rgb.append(np.asarray(frames_rgb[0], dtype=np.uint8))
            if len(frames_rgb) >= 2:
                import imageio
                basename = "trajectory_best_exec_3d"
                if filename_suffix:
                    basename = f"{basename}_{filename_suffix}"
                out_path = trajectory_dir / f"{basename}.gif"
                # Play once to avoid visual jump from last frame back to first frame.
                imageio.v3.imwrite(out_path, frames_rgb, duration=duration_ms, loop=1)
        except Exception:
            pass

    def _compute_best_idx_from_candidates(
        self,
        candidate_states_list: list,
        candidate_costs: np.ndarray,
        env: Any,
        obstacles: Any,
        env_plugin: Any,
        robot_radius: float,
        success_margin: float,
    ) -> int:
        """
        Best = lowest cost among (safe AND success) modes; if none, lowest cost over all.
        """
        n_modes = len(candidate_states_list)
        if n_modes == 0:
            return 0
        costs = np.asarray(candidate_costs, dtype=np.float64).ravel()
        if len(costs) != n_modes:
            return int(np.argmin(costs)) if len(costs) > 0 else 0

        task_spec = get_default_task_spec(env_plugin, getattr(self.config, "env_name", None))

        target = np.asarray(env.target, dtype=np.float32)
        target_pos = np.asarray(task_spec.extract_position(target), dtype=np.float32).reshape(-1)
        safe_success_mask = []
        for states in candidate_states_list:
            states_arr = [np.asarray(s, dtype=np.float32) for s in states]
            if len(states_arr) == 0:
                safe_success_mask.append(False)
                continue
            safe = True
            if obstacles is not None and hasattr(obstacles, '__len__') and len(obstacles) > 0:
                for s in states_arr:
                    pos = np.asarray(task_spec.extract_position(s), dtype=np.float32).reshape(-1)
                    sdf = obstacles.sdf(pos)
                    sdf_val = float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf)
                    if sdf_val < robot_radius or obstacles.contains(pos):
                        safe = False
                        break
            final_pos = np.asarray(task_spec.extract_position(states_arr[-1]), dtype=np.float32).reshape(-1)
            task_success = task_spec.success_criterion(
                final_pos, target_pos, success_margin, env_name=getattr(self.config, "env_name", None)
            )
            safe_success_mask.append(safe and task_success)

        valid_indices = [i for i in range(n_modes) if safe_success_mask[i]]
        if valid_indices:
            return int(valid_indices[np.argmin(costs[valid_indices])])
        return int(np.argmin(costs))

    def _compute_trajectory_length_smoothness(self, trajectory: Trajectory, env_plugin: Any) -> tuple:
        """Compute length, smoothness, geometric_smoothness for a single trajectory."""
        task_spec = get_default_task_spec(env_plugin, getattr(self.config, "env_name", None))

        states_arr = [np.asarray(s, dtype=np.float32) for s in trajectory.states]
        if len(states_arr) < 2:
            return 0.0, 0.0, 0.0
        positions = np.array([np.asarray(task_spec.extract_position(s), dtype=np.float32).reshape(-1) for s in states_arr])
        if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
            return 0.0, 0.0, 0.0
        seg_len = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        length = float(np.sum(seg_len))
        if not (np.isfinite(length) and length >= 0):
            return 0.0, 0.0, 0.0
        smooth = 0.0
        if len(states_arr) >= 3:
            vel = np.diff(positions, axis=0)
            acc = np.diff(vel, axis=0)
            n_acc = max(1, acc.shape[0])
            smooth = float(np.sum(acc ** 2)) / n_acc
        geom_smooth = 0.0
        if len(positions) >= 3:
            seg = np.diff(positions, axis=0)
            seg_norm = np.linalg.norm(seg, axis=1, keepdims=True)
            seg_norm = np.where(seg_norm < 1e-8, 1.0, seg_norm)
            u = seg / seg_norm
            dp = np.sum(u[1:] * u[:-1], axis=1)
            dp = np.clip(dp, -1.0, 1.0)
            angles = np.arccos(dp)
            n_angles = max(1, len(angles))
            geom_smooth = float(np.sum(angles ** 2)) / n_angles
        return length, smooth, geom_smooth

    def _violation_rate_band(
        self,
        states_list: list,
        obstacles: Any,
        robot_radius: float,
        extract_pos_2d: Any,
        n_along_max: int = 150,
        n_perp: int = 8,
    ) -> float:
        """
        Compute fraction of trajectory band (width 2*robot_radius) that lies inside obstacles.
        Samples points along the polyline and in the perpendicular disk; returns
        (count of points inside obstacle) / (total points). Returns 0.0 if no obstacles.
        """
        if obstacles is None or not (hasattr(obstacles, '__len__') and len(obstacles) > 0):
            return 0.0
        states_arr = [np.asarray(s, dtype=np.float32) for s in states_list]
        if len(states_arr) < 2:
            return 0.0
        positions = np.array([extract_pos_2d(s) for s in states_arr])
        seg_len = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        total_len = float(np.sum(seg_len))
        if total_len < 1e-8:
            return 0.0
        n_along = min(n_along_max, max(10, int(total_len / 0.01)))
        # Linear interpolation along polyline by arc length
        cum = np.concatenate([[0], np.cumsum(seg_len)])
        s_vals = np.linspace(0, total_len, n_along, endpoint=False)
        path_pts = []
        for s in s_vals:
            idx = np.searchsorted(cum, s, side='right') - 1
            idx = max(0, min(idx, len(positions) - 2))
            local = (s - cum[idx]) / max(1e-10, seg_len[idx])
            local = float(np.clip(local, 0, 1))
            p = (1 - local) * positions[idx] + local * positions[idx + 1]
            path_pts.append(p)
        path_pts = np.array(path_pts, dtype=np.float32)
        # Sample band: at each path point, sample n_perp points on circle of radius robot_radius
        radius = float(robot_radius)
        angles = np.linspace(0, 2 * np.pi, n_perp, endpoint=False)
        violations = 0
        total = 0
        for i in range(path_pts.shape[0]):
            for a in angles:
                pt = path_pts[i] + radius * np.array([np.cos(a), np.sin(a)], dtype=np.float32)
                total += 1
                try:
                    sdf_val = obstacles.sdf(pt)
                    sdf_val = float(np.asarray(sdf_val).item() if hasattr(sdf_val, "item") else sdf_val)
                    if sdf_val < 0 or (hasattr(obstacles, 'contains') and obstacles.contains(pt)):
                        violations += 1
                except Exception:
                    violations += 1
        return violations / max(1, total)

    def _compute_modes_metrics(
        self,
        candidate_states_list: list,
        env: Any,
        obstacles: Any,
        constraints: Any,
        env_plugin: Any,
        robot_radius: float,
        success_margin: float,
        *,
        use_target_line: bool = False,
        num_targets: int = 4,
        num_modes: int = 0,
    ) -> Dict[str, Any]:
        """Compute modes-based metrics: ssr, length, smoothness, geometric_smoothness."""
        n_modes = len(candidate_states_list)
        if n_modes == 0:
            return {}
        if num_modes <= 0:
            num_modes = n_modes

        task_spec = get_default_task_spec(env_plugin, getattr(self.config, "env_name", None))

        ssr_count = 0
        lengths = []
        smoothnesses = []
        geom_smoothnesses = []
        violation_rates = []

        target = np.asarray(env.target, dtype=np.float32)
        target_pos = np.asarray(task_spec.extract_position(target), dtype=np.float32).reshape(-1)
        target_line = None
        if use_target_line and env_plugin is not None and getattr(env_plugin, "name", None) == "d3il_avoiding_9d":
            from genedynamics.experiments.plugins.obstacles.d3il_avoiding_fixed import get_d3il_target_line_positions
            target_line = get_d3il_target_line_positions(num_targets)

        for i, states in enumerate(candidate_states_list):
            states_arr = [np.asarray(s, dtype=np.float32) for s in states]
            if len(states_arr) == 0:
                continue

            # Safe: no collision
            safe = True
            if obstacles is not None and hasattr(obstacles, '__len__') and len(obstacles) > 0:
                for s in states_arr:
                    pos = np.asarray(task_spec.extract_position(s), dtype=np.float32).reshape(-1)
                    sdf = obstacles.sdf(pos)
                    sdf_val = float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf)
                    if sdf_val < robot_radius:
                        safe = False
                        break
                    if obstacles.contains(pos):
                        safe = False
                        break

            # Task success (TaskSpec handles 2D point target and d3il_avoiding line target)
            final_pos = np.asarray(task_spec.extract_position(states_arr[-1]), dtype=np.float32).reshape(-1)
            task_success = task_spec.success_criterion(
                final_pos, target_pos, success_margin, env_name=getattr(self.config, "env_name", None)
            )

            if safe and task_success:
                ssr_count += 1

            # Length: sum of segment lengths (skip if trajectory has NaN - e.g. Go2 rollout instability)
            positions = np.array([np.asarray(task_spec.extract_position(s), dtype=np.float32).reshape(-1) for s in states_arr])
            if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                continue
            seg_len = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            length = float(np.sum(seg_len))
            if not (np.isfinite(length) and length >= 0):
                continue
            lengths.append(length)

            # Smoothness (acceleration): mean squared ||acc|| per step (normalized by n_steps)
            if len(states_arr) >= 3:
                pos_arr = np.array([np.asarray(task_spec.extract_position(s), dtype=np.float32).reshape(-1) for s in states_arr])
                vel = np.diff(pos_arr, axis=0)
                acc = np.diff(vel, axis=0)
                n_acc = max(1, acc.shape[0])
                smooth = float(np.sum(acc ** 2)) / n_acc
            else:
                smooth = 0.0
            smoothnesses.append(smooth)

            # Geometric smoothness: mean squared angle (rad²) per turn (normalized by n_angles)
            if len(positions) >= 3:
                seg = np.diff(positions, axis=0)
                seg_norm = np.linalg.norm(seg, axis=1, keepdims=True)
                seg_norm = np.where(seg_norm < 1e-8, 1.0, seg_norm)
                u = seg / seg_norm
                dp = np.sum(u[1:] * u[:-1], axis=1)
                dp = np.clip(dp, -1.0, 1.0)
                angles = np.arccos(dp)
                n_angles = max(1, len(angles))
                geom_smooth = float(np.sum(angles ** 2)) / n_angles
            else:
                geom_smooth = 0.0
            geom_smoothnesses.append(geom_smooth)

            # Violation rate: fraction of trajectory band (width 2*robot_radius) inside obstacles
            def _extract_pos(s):
                return np.asarray(task_spec.extract_position(s), dtype=np.float32).reshape(-1)

            vr = self._violation_rate_band(states, obstacles, robot_radius, _extract_pos)
            violation_rates.append(vr)

        n_valid = len(lengths)
        if n_valid == 0:
            return {}

        out = {}
        out['ssr'] = {
            'ssr': float(ssr_count) / max(1, n_modes),
            'ssr_count': int(ssr_count),
            'violation_rate_mean': float(np.mean(violation_rates)) if violation_rates else 0.0,
            'violation_rate_std': float(np.std(violation_rates)) if len(violation_rates) > 1 else 0.0,
        }
        out['length'] = {'mean': float(np.mean(lengths)), 'std': float(np.std(lengths)) if n_valid > 1 else 0.0}
        out['smoothness'] = {'mean': float(np.mean(smoothnesses)), 'std': float(np.std(smoothnesses)) if n_valid > 1 else 0.0}
        out['geometric_smoothness'] = {'mean': float(np.mean(geom_smoothnesses)), 'std': float(np.std(geom_smoothnesses)) if n_valid > 1 else 0.0}
        out['_per_mode'] = {'lengths': lengths, 'smoothnesses': smoothnesses, 'geom_smoothnesses': geom_smoothnesses}
        return out

    def _compute_metrics(
        self,
        trajectory: Trajectory,
        env: Any,
        obstacles: Any,
        constraints: Any,
        level: int,
        env_plugin: Any = None,
        planning_result: Optional[Dict[str, Any]] = None,
        planning_time: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute all requested metrics.
        When planning_result has candidate_states/candidate_actions, modes-based ssr, length,
        smoothness, geometric_smoothness are computed and override/extend metrics.
        metrics["best"]: best mode's planning_time, length, smoothness, geometric_smoothness.
        For multirun: best.planning_time = planning_time / num_modes.
        """
        metrics_result = {}
        robot_radius = float(self.config.obstacle_config.get('robot_radius', 0.05))
        success_margin = float(self.config.obstacle_config.get('success_margin', 2 * robot_radius))

        # Modes-based metrics when candidate data exists
        candidate_states_list = []
        if planning_result is not None:
            cand_states = planning_result.get('candidate_states', [])
            cand_actions = planning_result.get('candidate_actions', [])
            initial_state = planning_result.get('initial_state', None)
            if _has_items(cand_states):
                candidate_states_list = list(cand_states)
            elif _has_items(cand_actions) and initial_state is not None and hasattr(env, 'rollout_actions'):
                initial = np.asarray(initial_state, dtype=np.float32)
                for acts in cand_actions:
                    acts_arr = np.asarray(acts, dtype=np.float32)
                    try:
                        states = env.rollout_actions(initial, acts_arr)
                        candidate_states_list.append([np.asarray(s, dtype=np.float32) for s in states])
                    except Exception:
                        pass

        if _has_items(candidate_states_list):
            method_params = getattr(self.config, "method_params", None) or {}
            use_target_line = bool(method_params.get("use_target_line", False))
            num_targets = int(method_params.get("num_targets", 4))
            modes_metrics = self._compute_modes_metrics(
                candidate_states_list, env, obstacles, constraints, env_plugin,
                robot_radius, success_margin,
                use_target_line=use_target_line,
                num_targets=num_targets,
                num_modes=len(candidate_states_list),
            )
            if 'ssr' in modes_metrics:
                metrics_result['ssr'] = modes_metrics['ssr']
            if 'length' in modes_metrics:
                metrics_result['length'] = modes_metrics['length']
            if 'smoothness' in modes_metrics:
                metrics_result['smoothness'] = modes_metrics['smoothness']
            if 'geometric_smoothness' in modes_metrics:
                metrics_result['geometric_smoothness'] = modes_metrics['geometric_smoothness']
            # metrics["best"]: best mode's planning_time, length, smoothness, geometric_smoothness
            best_idx = int(planning_result.get('best_idx', 0))
            per_mode = modes_metrics.get('_per_mode', {})
            lengths = per_mode.get('lengths', [])
            smoothnesses = per_mode.get('smoothnesses', [])
            geom_smoothnesses = per_mode.get('geom_smoothnesses', [])
            n_modes = len(candidate_states_list)
            mode_strategy = str(planning_result.get('mode_strategy', '')).lower()
            if 'best_planning_time' in planning_result and planning_result['best_planning_time'] is not None:
                best_planning_time = float(planning_result['best_planning_time'])
            else:
                best_planning_time = float(planning_time)
                if n_modes > 1 and mode_strategy == 'multirun':
                    best_planning_time = float(planning_time) / n_modes
            best_length = float(lengths[best_idx]) if best_idx < len(lengths) else 0.0
            best_smooth = float(smoothnesses[best_idx]) if best_idx < len(smoothnesses) else 0.0
            best_geom = float(geom_smoothnesses[best_idx]) if best_idx < len(geom_smoothnesses) else 0.0
            metrics_result['best'] = {
                'planning_time': best_planning_time,
                'length': best_length,
                'smoothness': best_smooth,
                'geometric_smoothness': best_geom,
            }
        else:
            # Single trajectory: best = trajectory metrics
            best_len, best_sm, best_gs = self._compute_trajectory_length_smoothness(
                trajectory, env_plugin,
            )
            metrics_result['best'] = {
                'planning_time': float(planning_time),
                'length': best_len,
                'smoothness': best_sm,
                'geometric_smoothness': best_gs,
            }

        # Standard plugin metrics (skip ssr if already set by modes)
        for metric_name in self.config.metrics:
            if metric_name == 'ssr' and 'ssr' in metrics_result:
                continue
            try:
                metric_plugin = self.registry.get_plugin('metric', metric_name)
                metric_value = metric_plugin.compute(
                    trajectory, env, obstacles, constraints,
                    robot_radius=robot_radius,
                    level=level,
                    obstacle_config=self.config.obstacle_config,
                    env_plugin=env_plugin,
                    env_name=getattr(self.config, "env_name", None),
                    planning_result=planning_result,
                    planning_time=float(planning_time),
                )
                val = convert_to_json_serializable(metric_value)
                if metric_name == 'ssr' and isinstance(val, dict):
                    # Keep only ssr field for modes consistency when we later add ssr from plugin
                    metrics_result[metric_name] = {'ssr': float(val.get('ssr', 0.0))}
                else:
                    metrics_result[metric_name] = val
            except Exception as e:
                print(f"Warning: Failed to compute metric '{metric_name}': {e}")
                if metric_name not in metrics_result:
                    metrics_result[metric_name] = None

        # Execution SSR (D3IL only): (success & no-collision executions) / total executions; or 0/1 for single run
        is_d3il_style = (
            getattr(self.config, 'method', None) == 'd3il_unified'
            or 'd3il' in str(getattr(self.config, 'env_name', ''))
        )
        if is_d3il_style and planning_result is not None:
            if isinstance(planning_result.get('execution_ssr'), dict):
                metrics_result['execution_ssr'] = dict(planning_result['execution_ssr'])
            else:
                succ = bool(planning_result.get('success', False))
                coll = bool(planning_result.get('collision', True))
                metrics_result['execution_ssr'] = {'execution_ssr': 1.0 if (succ and not coll) else 0.0}
            # Violation rate for executed trajectories (band 2*robot_radius, averaged over all executions)
            exec_states_list = planning_result.get('exec_candidate_states', [])
            if not exec_states_list:
                single_states = planning_result.get('executed_states') or planning_result.get('states')
                if single_states is not None:
                    exec_states_list = [single_states]
            if exec_states_list and obstacles is not None:
                task_spec = get_default_task_spec(env_plugin, getattr(self.config, "env_name", None))

                def extract_pos(s):
                    return np.asarray(task_spec.extract_position(s), dtype=np.float32).reshape(-1)

                exec_violation_rates = []
                for states in exec_states_list:
                    vr = self._violation_rate_band(states, obstacles, robot_radius, extract_pos)
                    exec_violation_rates.append(vr)
                metrics_result['execution_ssr']['violation_rate_mean'] = float(np.mean(exec_violation_rates)) if exec_violation_rates else 0.0
                metrics_result['execution_ssr']['violation_rate_std'] = float(np.std(exec_violation_rates)) if len(exec_violation_rates) > 1 else 0.0
                # Execution SSR only counts runs that are success & no collision & (effectively) zero violation
                exec_success_pm = planning_result.get('exec_success_per_mode')
                exec_collision_pm = planning_result.get('exec_collision_per_mode')
                if (
                    exec_success_pm is not None and exec_collision_pm is not None
                    and len(exec_success_pm) == len(exec_states_list)
                    and len(exec_violation_rates) == len(exec_states_list)
                ):
                    violation_threshold = 1e-9
                    exec_ssr_count = sum(
                        1 for i in range(len(exec_states_list))
                        if exec_success_pm[i] and not exec_collision_pm[i] and exec_violation_rates[i] < violation_threshold
                    )
                    n_exec_total = len(exec_states_list)
                    metrics_result['execution_ssr']['execution_ssr'] = float(exec_ssr_count) / max(1, n_exec_total)
                    metrics_result['execution_ssr']['execution_ssr_count'] = int(exec_ssr_count)
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
            # Shared robot motion replay (GIF + HTML)
            if viz_name == 'motion_replay':
                try:
                    self._generate_motion_replay_visualization(result, env)
                except Exception as e:
                    print(f"Warning: Failed to generate visualization 'motion_replay': {e}")
                    import traceback
                    traceback.print_exc()
                continue

            # Cost visualization is generated inline (no plugin)
            if viz_name == 'cost':
                try:
                    self._generate_cost_visualization(result, env, obstacles, env_plugin)
                except Exception as e:
                    print(f"Warning: Failed to generate visualization 'cost': {e}")
                    import traceback
                    traceback.print_exc()
                continue

            # Adaptive (CFS-MBD scheduler metrics) visualization: inline
            if viz_name == 'adaptive':
                try:
                    self._generate_adaptive_visualization(result, env_plugin)
                except Exception as e:
                    print(f"Warning: Failed to generate visualization 'adaptive': {e}")
                    import traceback
                    traceback.print_exc()
                continue

            try:
                viz_plugin = self.registry.get_plugin('visualization', viz_name)
            except KeyError:
                print(f"Warning: Visualization plugin '{viz_name}' not found in registry")
                continue

            try:
                # Create figure based on visualization type
                import matplotlib.pyplot as plt

                # Visualization-specific figure creation
                if viz_name in ('trajectory', 'stepping_trajectory', 'corridor_trajectory'):
                    out_dir = self._get_output_path(result['level'], result['seed'])
                    trajectory_dir = out_dir / "trajectory"
                    trajectory_dir.mkdir(parents=True, exist_ok=True)
                    viz_cfg = {**viz_config.get(viz_name, {}), 'config': self.config}
                    is_d3il = self._is_d3il_experiment(result)
                    if is_d3il:
                        # D3IL: save trajectory_best_plan and trajectory_best_exec (and their GIFs)
                        plan_traj = self._extract_plan_trajectory(result.get('result', {}), env)
                        exec_traj = result['trajectory']
                        for label, traj in [('plan', plan_traj), ('exec', exec_traj)]:
                            if traj is None or len(traj.states) == 0:
                                continue
                            data_base = {
                                'trajectory': traj,
                                'env': env,
                                'obstacles': obstacles,
                                'env_plugin': env_plugin,
                            }
                            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                            viz_plugin.visualize(fig, ax, data_base, viz_cfg)
                            viz_plugin.save(trajectory_dir / f"trajectory_best_{label}.png", fig, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            num_steps = max(0, len(traj.states) - 1)
                            if num_steps >= 0:
                                import imageio
                                temp_frames = []
                                try:
                                    for t in range(num_steps + 1):
                                        fig_g, ax_g = plt.subplots(1, 1, figsize=(8, 8))
                                        viz_plugin.visualize(
                                            fig_g, ax_g,
                                            {**data_base, 'partial_until_step': t, 'gif_style': True},
                                            viz_cfg
                                        )
                                        tmp_path = trajectory_dir / f"_gif_best_{label}_{t}.png"
                                        fig_g.savefig(tmp_path, dpi=100, bbox_inches='tight')
                                        plt.close(fig_g)
                                        temp_frames.append(tmp_path)
                                    if temp_frames:
                                        frames = [imageio.v3.imread(p) for p in temp_frames]
                                        imageio.v3.imwrite(
                                            trajectory_dir / f"trajectory_best_{label}.gif",
                                            frames,
                                            duration=80,
                                            loop=0,
                                        )
                                finally:
                                    for p in temp_frames:
                                        try:
                                            p.unlink()
                                        except OSError:
                                            pass
                        # 3D GIF already generated in run_single_experiment (8b1.5) before 8b2
                    else:
                        # single_2d etc.: original trajectory_best.png / trajectory_best.gif only
                        traj = result['trajectory']
                        data_base = {
                            'trajectory': traj,
                            'env': env,
                            'obstacles': obstacles,
                            'env_plugin': env_plugin,
                        }
                        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                        viz_plugin.visualize(fig, ax, data_base, viz_cfg)
                        viz_plugin.save(trajectory_dir / "trajectory_best.png", fig, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        num_steps = max(0, len(traj.states) - 1)
                        if num_steps >= 0:
                            import imageio
                            temp_frames = []
                            try:
                                for t in range(num_steps + 1):
                                    fig_g, ax_g = plt.subplots(1, 1, figsize=(8, 8))
                                    viz_plugin.visualize(
                                        fig_g, ax_g,
                                        {**data_base, 'partial_until_step': t, 'gif_style': True},
                                        viz_cfg
                                    )
                                    tmp_path = trajectory_dir / f"_gif_best_{t}.png"
                                    fig_g.savefig(tmp_path, dpi=100, bbox_inches='tight')
                                    plt.close(fig_g)
                                    temp_frames.append(tmp_path)
                                if temp_frames:
                                    frames = [imageio.v3.imread(p) for p in temp_frames]
                                    imageio.v3.imwrite(
                                        trajectory_dir / "trajectory_best.gif",
                                        frames,
                                        duration=80,
                                        loop=0,
                                    )
                            finally:
                                for p in temp_frames:
                                    try:
                                        p.unlink()
                                    except OSError:
                                        pass
                
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
                
                elif viz_name in ('trajectory_modes', 'stepping_modes'):
                    out_dir = self._get_output_path(result['level'], result['seed'])
                    trajectory_dir = out_dir / "trajectory"
                    trajectory_dir.mkdir(parents=True, exist_ok=True)
                    viz_cfg = {**viz_config.get(viz_name, {}), 'config': self.config}
                    is_d3il = self._is_d3il_experiment(result)
                    if is_d3il:
                        # D3IL: trajectory_modes_plan (candidate_states) and trajectory_modes_exec (exec states as single mode)
                        planning_result = result.get('result', {})
                        # Plan: use planning_result as-is (candidate_states from solver)
                        data_plan = {
                            'result': planning_result,
                            'env': env,
                            'obstacles': obstacles,
                            'env_plugin': env_plugin,
                        }
                        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                        viz_plugin.visualize(fig, ax, data_plan, viz_cfg)
                        viz_plugin.save(trajectory_dir / "trajectory_modes_plan.png", fig, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        cand_states_plan = planning_result.get('candidate_states')
                        cand_states_plan = [] if cand_states_plan is None else cand_states_plan
                        if not _has_items(cand_states_plan) and planning_result.get('states'):
                            cand_states_plan = [planning_result['states']]
                        max_steps_plan = max((max(0, len(c) - 1) for c in cand_states_plan), default=0)
                        if max_steps_plan >= 0 and _has_items(cand_states_plan):
                            import imageio
                            temp_frames = []
                            try:
                                for t in range(max_steps_plan + 1):
                                    fig_g, ax_g = plt.subplots(1, 1, figsize=(8, 8))
                                    viz_plugin.visualize(fig_g, ax_g, {**data_plan, 'partial_until_step': t}, viz_cfg)
                                    tmp_path = trajectory_dir / f"_gif_modes_plan_{t}.png"
                                    fig_g.savefig(tmp_path, dpi=100, bbox_inches='tight')
                                    plt.close(fig_g)
                                    temp_frames.append(tmp_path)
                                if temp_frames:
                                    frames = [imageio.v3.imread(p) for p in temp_frames]
                                    imageio.v3.imwrite(trajectory_dir / "trajectory_modes_plan.gif", frames, duration=80, loop=0)
                            finally:
                                for p in temp_frames:
                                    try:
                                        p.unlink()
                                    except OSError:
                                        pass
                        # Exec: all candidates' executions (exec_candidate_states) or fallback to single run
                        exec_candidate_states = planning_result.get('exec_candidate_states')
                        if _has_items(exec_candidate_states):
                            result_exec = {
                                'candidate_states': exec_candidate_states,
                                'candidate_actions': planning_result.get('candidate_actions'),
                                'best_idx': int(planning_result.get('best_idx', 0)),
                            }
                            if planning_result.get('candidate_costs') is not None:
                                result_exec['candidate_costs'] = planning_result['candidate_costs']
                        else:
                            exec_states = planning_result.get('states', [])
                            if exec_states:
                                result_exec = {
                                    'candidate_states': [exec_states],
                                    'candidate_actions': planning_result.get('actions'),
                                    'best_idx': 0,
                                }
                                if result_exec['candidate_actions'] is not None:
                                    act = np.asarray(result_exec['candidate_actions'], dtype=np.float32)
                                    result_exec['candidate_actions'] = [act] if act.ndim >= 2 else [act]
                            else:
                                result_exec = None
                        if result_exec:
                            data_exec = {
                                'result': result_exec,
                                'env': env,
                                'obstacles': obstacles,
                                'env_plugin': env_plugin,
                            }
                            fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                            viz_plugin.visualize(fig, ax, data_exec, viz_cfg)
                            viz_plugin.save(trajectory_dir / "trajectory_modes_exec.png", fig, dpi=150, bbox_inches='tight')
                            plt.close(fig)
                            cand_exec = result_exec['candidate_states']
                            num_steps_exec = max((max(0, len(s) - 1) for s in cand_exec), default=0)
                            if num_steps_exec >= 0 and cand_exec:
                                import imageio
                                temp_frames = []
                                try:
                                    for t in range(num_steps_exec + 1):
                                        fig_g, ax_g = plt.subplots(1, 1, figsize=(8, 8))
                                        viz_plugin.visualize(fig_g, ax_g, {**data_exec, 'partial_until_step': t}, viz_cfg)
                                        tmp_path = trajectory_dir / f"_gif_modes_exec_{t}.png"
                                        fig_g.savefig(tmp_path, dpi=100, bbox_inches='tight')
                                        plt.close(fig_g)
                                        temp_frames.append(tmp_path)
                                    if temp_frames:
                                        frames = [imageio.v3.imread(p) for p in temp_frames]
                                        imageio.v3.imwrite(trajectory_dir / "trajectory_modes_exec.gif", frames, duration=80, loop=0)
                                finally:
                                    for p in temp_frames:
                                        try:
                                            p.unlink()
                                        except OSError:
                                            pass
                    else:
                        # single_2d etc.: original trajectory_modes.png / trajectory_modes.gif only
                        data_base = {
                            'result': result['result'],
                            'env': env,
                            'obstacles': obstacles,
                            'env_plugin': env_plugin,
                        }
                        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
                        viz_plugin.visualize(fig, ax, data_base, viz_cfg)
                        viz_plugin.save(trajectory_dir / "trajectory_modes.png", fig, dpi=150, bbox_inches='tight')
                        plt.close(fig)
                        planning_result = result.get('result', {})
                        candidate_states = planning_result.get('candidate_states')
                        candidate_states = [] if candidate_states is None else candidate_states
                        if not _has_items(candidate_states):
                            states_list = planning_result.get('states', [])
                            if states_list:
                                candidate_states = [states_list]
                        max_steps = 0
                        for states_c in candidate_states:
                            max_steps = max(max_steps, max(0, len(states_c) - 1))
                        if max_steps >= 0 and _has_items(candidate_states):
                            import imageio
                            temp_frames = []
                            try:
                                for t in range(max_steps + 1):
                                    fig_g, ax_g = plt.subplots(1, 1, figsize=(8, 8))
                                    viz_plugin.visualize(
                                        fig_g, ax_g,
                                        {**data_base, 'partial_until_step': t},
                                        viz_cfg
                                    )
                                    tmp_path = trajectory_dir / f"_gif_modes_{t}.png"
                                    fig_g.savefig(tmp_path, dpi=100, bbox_inches='tight')
                                    plt.close(fig_g)
                                    temp_frames.append(tmp_path)
                                if temp_frames:
                                    frames = [imageio.v3.imread(p) for p in temp_frames]
                                    imageio.v3.imwrite(
                                        trajectory_dir / "trajectory_modes.gif",
                                        frames,
                                        duration=80,
                                        loop=0,
                                    )
                            finally:
                                for p in temp_frames:
                                    try:
                                        p.unlink()
                                    except OSError:
                                        pass
                
                elif viz_name == 'diffusion':
                    # Save individual PNGs (90%, 50%, 10%) and GIF to diffusion_steps/ subdirectory
                    output_path = self._get_output_path(result['level'], result['seed'])
                    diffusion_steps_dir = output_path / "diffusion_steps"
                    data = {
                        'result': result['result'],
                        'env': env,
                        'obstacles': obstacles,
                        'initial_state': result['result'].get('initial_state'),
                        'env_plugin': env_plugin,
                    }
                    viz_config_diffusion = {**viz_config.get(viz_name, {}), 'config': self.config}
                    viz_plugin.save_individual_and_gif(
                        diffusion_steps_dir,
                        data,
                        viz_config_diffusion,
                        dpi=150,
                    )
                
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

                elif viz_name == 'gate_dynamics':
                    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
                    viz_plugin.visualize(
                        fig, axes,
                        {
                            'result': result['result'],
                            'env': env,
                        },
                        {**viz_config.get(viz_name, {}), 'config': self.config}
                    )
                    out_dir = self._get_output_path(result['level'], result['seed'])
                    adaptive_dir = out_dir / "adaptive"
                    adaptive_dir.mkdir(parents=True, exist_ok=True)
                    output_path = adaptive_dir / "gate_dynamics.png"
                    plt.tight_layout()
                    viz_plugin.save(output_path, fig, dpi=150, bbox_inches='tight')
                    plt.close(fig)

            except Exception as e:
                print(f"Warning: Failed to generate visualization '{viz_name}': {e}")
                import traceback
                traceback.print_exc()

    def _generate_motion_replay_visualization(self, result: Dict[str, Any], env: Any) -> None:
        """
        Generate unified robot motion replay assets for experiments.

        Outputs:
        - motion_replay/motion_replay_*.gif
        - motion_replay/motion_replay_*.html
        """
        from genedynamics.viz.motion_episode import MotionEpisode
        from genedynamics.viz.motion_renderer import MotionRenderer

        trajectory = result.get("trajectory")
        if trajectory is None or not getattr(trajectory, "states", None):
            raise ValueError("No trajectory states for motion replay.")

        states = np.asarray([np.asarray(s, dtype=np.float64) for s in trajectory.states], dtype=np.float64)
        actions = None
        if getattr(trajectory, "actions", None):
            actions = np.asarray([np.asarray(a, dtype=np.float64) for a in trajectory.actions], dtype=np.float64)

        env_name = str(getattr(self.config, "env_name", "")).lower()
        if "humanoid" in env_name:
            robot_type = "humanoid"
        elif "quadruped" in env_name:
            robot_type = "quadruped"
        else:
            robot_type = "unknown"

        model_id = "auto"
        if hasattr(env, "model") and getattr(env, "model", None):
            model_id = str(getattr(env, "model"))

        output_path = self._get_output_path(result["level"], result["seed"])
        replay_dir = output_path / "motion_replay"
        replay_dir.mkdir(parents=True, exist_ok=True)

        episode = MotionEpisode(
            states=states,
            actions=actions,
            robot_type=robot_type,
            model_id=model_id,
            fps=20.0,
            metadata={
                "source": "experiments",
                "env_name": getattr(self.config, "env_name", ""),
                "level": int(result.get("level", 0)),
                "seed": int(result.get("seed", 0)),
            },
        )
        renderer = MotionRenderer(replay_dir)
        name = f"motion_replay_l{int(result.get('level', 0))}_s{int(result.get('seed', 0))}"
        renderer.render_html(episode, name=name)

    def _compute_cost_for_actions(
        self, env: Any, obstacles: Any, initial_state: np.ndarray, act_seq: np.ndarray,
        violation_weight: float=150, clearance: float=0.0, robot_radius: float=0.05,
        env_plugin: Any = None,
    ) -> float:
        """Compute task cost (stage+terminal) + violation_weight * sum_t [g]_+ for one action sequence.
        [g]_+ uses effective margin = max(clearance, robot_radius) so penetration (sdf < robot_radius) is penalized.
        Uses batch SDF for positions (one call per rollout) for speed."""
        task_cost = 0.0
        violation = 0.0
        if not hasattr(env, 'rollout_actions'):
            return 0.0
        try:
            states = env.rollout_actions(np.asarray(initial_state, dtype=np.float32), np.asarray(act_seq, dtype=np.float32))
            states = np.asarray(states)
            task_spec = get_default_task_spec(env_plugin, getattr(self.config, "env_name", None))
            positions = np.array([np.asarray(task_spec.extract_position(s), dtype=np.float32).reshape(-1) for s in states])

            for t in range(len(states)):
                if hasattr(env, 'cost'):
                    task_cost += float(env.cost(states[t]))

            # Batch SDF: one call for all positions
            sdf_box = np.full(len(positions), float('inf'), dtype=np.float64)
            if hasattr(env, 'jax_sdf'):
                try:
                    sdf_box = np.asarray(env.jax_sdf(positions), dtype=np.float64).ravel()
                except Exception:
                    pass
            sdf_obs = np.full(len(positions), float('inf'), dtype=np.float64)
            if obstacles is not None and hasattr(obstacles, 'sdf'):
                try:
                    sdf_obs = np.asarray(obstacles.sdf(positions), dtype=np.float64).ravel()
                except Exception:
                    pass
            min_sdf = np.minimum(sdf_box, sdf_obs)
            # Violation margin: at least robot_radius so penetration is penalized
            effective_margin = max(clearance, robot_radius)
            violation = float(np.sum(np.maximum(0.0, effective_margin - min_sdf)))

            if hasattr(env, 'target') and len(states) > 0:
                target = np.asarray(getattr(env, 'target', (0, 0)))
                final_pos = np.asarray(task_spec.extract_position(states[-1]), dtype=np.float64).reshape(-1)
                target_pos = np.asarray(task_spec.extract_position(target), dtype=np.float64).reshape(-1)
                task_cost += 100.0 * float(np.linalg.norm(final_pos - target_pos))
        except Exception:
            pass
        return max(0.0, task_cost) + violation_weight * violation

    def _generate_cost_visualization(self, result: Dict[str, Any], env: Any, obstacles: Any, env_plugin: Any = None) -> None:
        """
        Generate cost plots and cost.json under result_dir/cost/.
        Cost = task cost (-reward) + violation_weight * sum_t [g]_+ (violation cost).
        cost_modes = mean over M samples at each diffusion step, with std bound.
        cost.json: M x H array (cost_per_mode), best_idx.
        """
        import matplotlib.pyplot as plt
        from scipy.ndimage import uniform_filter1d

        out_dir = self._get_output_path(result['level'], result['seed'])
        cost_dir = out_dir / "cost"
        cost_dir.mkdir(parents=True, exist_ok=True)

        viz_config = self.config.visualization_config or {}
        cost_config = viz_config.get('cost', {})
        # Default 100 so unsafe trajectories get high cost (task + 100*sum_t [g]_+)
        violation_weight = float(cost_config.get('violation_weight', 150.0))
        clearance = float(cost_config.get('clearance', 0.0))
        robot_radius = float(self.config.obstacle_config.get('robot_radius', 0.05))

        planning_result = result.get('result', {})
        reward_history = planning_result.get('reward_history', None)
        if reward_history is None or (hasattr(reward_history, '__len__') and len(reward_history) == 0):
            with open(cost_dir / "cost.json", 'w') as f:
                json.dump({"cost_per_mode": [], "best_idx": 0}, f, indent=2)
            return

        reward_histories = [np.asarray(reward_history, dtype=np.float64).ravel()] if not (
            isinstance(reward_history, (list, tuple)) and len(reward_history) > 0 and
            (isinstance(reward_history[0], (list, tuple)) or (hasattr(reward_history[0], 'ndim') and reward_history[0].ndim >= 1))
        ) else [np.asarray(r, dtype=np.float64).ravel() for r in reward_history]

        n_steps = len(reward_histories[0])
        initial_state = planning_result.get('initial_state', None)

        # Build cost matrix: M samples x n_steps diffusion steps
        # Use diffusion_sampled_actions for per-sample cost (cost_modes = mean over samples)
        diffusion_sampled = planning_result.get('diffusion_sampled_actions', None)
        diffusion_actions_traj = planning_result.get('diffusion_actions_traj', None)

        cost_matrix = None  # (M, n_steps)
        if diffusion_sampled is not None and initial_state is not None:
            try:
                ds = np.asarray(diffusion_sampled)
                # ds: (n_steps, M, H, act_dim) - solver order index 0 = clean, so reverse for 100..1
                ds_rev = ds[::-1] if ds.ndim == 4 else ds
                n_s = ds_rev.shape[0]
                M = ds_rev.shape[1] if ds_rev.ndim >= 2 else 1
                cost_matrix = np.zeros((M, n_steps), dtype=np.float64)
                for i in range(min(n_steps, n_s)):
                    for m in range(M):
                        act_seq = np.asarray(ds_rev[i, m], dtype=np.float32)
                        c = self._compute_cost_for_actions(
                            env, obstacles, initial_state, act_seq, violation_weight, clearance, robot_radius,
                            env_plugin=env_plugin,
                        )
                        cost_matrix[m, i] = c
            except Exception:
                pass

        # Fallback: use reward_history + diffusion_actions_traj (one curve = mean trajectory)
        diffusion_actions_traj = planning_result.get('diffusion_actions_traj', None)
        if cost_matrix is None or cost_matrix.size == 0:
            task_cost_ordered = [np.maximum(-np.asarray(r), 0.0)[::-1] for r in reward_histories]
            if diffusion_actions_traj is not None and initial_state is not None:
                da = np.asarray(diffusion_actions_traj)
                if da.ndim >= 2 and len(da) >= n_steps:
                    actions_list = (da[::-1] if da.ndim == 3 else [da[i] for i in range(len(da))][::-1])
                    cost_list = []
                    for i in range(min(n_steps, len(actions_list))):
                        c = self._compute_cost_for_actions(
                            env, obstacles, initial_state, np.asarray(actions_list[i]),
                            violation_weight, clearance, robot_radius,
                            env_plugin=env_plugin,
                        )
                        cost_list.append(c)
                    cost_matrix = np.array([cost_list], dtype=np.float64)
                else:
                    cost_matrix = np.array([task_cost_ordered[0]], dtype=np.float64)
            else:
                cost_matrix = np.array([task_cost_ordered[0]], dtype=np.float64)

        # best_idx: prefer (safe AND success) + lowest final cost; else lowest final cost
        robot_radius = float(self.config.obstacle_config.get('robot_radius', 0.05))
        success_margin = float(self.config.obstacle_config.get('success_margin', 2 * robot_radius))
        M_rows = cost_matrix.shape[0]
        final_costs = cost_matrix[:, -1]
        candidate_states_for_cost = []
        if diffusion_sampled is not None and initial_state is not None and hasattr(env, 'rollout_actions') and M_rows > 0:
            try:
                ds = np.asarray(diffusion_sampled)
                ds_rev = ds[::-1] if ds.ndim == 4 else ds
                n_s = ds_rev.shape[0]
                for m in range(min(M_rows, ds_rev.shape[1] if ds_rev.ndim >= 2 else 1)):
                    act_seq = np.asarray(ds_rev[-1, m], dtype=np.float32)
                    states = env.rollout_actions(np.asarray(initial_state, dtype=np.float32), act_seq)
                    candidate_states_for_cost.append([np.asarray(s, dtype=np.float32) for s in states])
                if len(candidate_states_for_cost) == M_rows:
                    best_idx = self._compute_best_idx_from_candidates(
                        candidate_states_for_cost, final_costs, env, obstacles,
                        env_plugin, robot_radius, success_margin,
                    )
                else:
                    best_idx = int(np.argmin(final_costs))
            except Exception:
                best_idx = int(np.argmin(final_costs)) if M_rows > 0 else 0
        else:
            best_idx = int(np.argmin(final_costs)) if M_rows > 0 else 0
        best_idx = min(max(0, best_idx), cost_matrix.shape[0] - 1)
        cost_json = {
            "cost_per_mode": [row.tolist() for row in cost_matrix],
            "best_idx": best_idx,
        }
        with open(cost_dir / "cost.json", 'w') as f:
            json.dump(cost_json, f, indent=2)

        # cost_best = best trajectory (row best_idx)
        cost_best = cost_matrix[best_idx]

        # Smoothing window (odd)
        smooth_size = max(3, min(15, n_steps // 5)) | 1
        def smooth(y):
            return uniform_filter1d(np.asarray(y, dtype=np.float64), size=smooth_size, mode='nearest')

        x = np.arange(n_steps)
        # X-axis: 100, 80, 60, 40, 20, 1 (data goes to 1)
        def set_diffusion_axis(ax, n_steps):
            tick_labels = [100, 80, 60, 40, 20, 1]
            tick_positions = [n_steps - k for k in tick_labels if 1 <= k <= n_steps]
            tick_labels = [k for k in tick_labels if 1 <= k <= n_steps]
            if not tick_positions:
                tick_positions = [0, n_steps - 1]
                tick_labels = [str(n_steps), "1"]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(l) for l in tick_labels])
            ax.set_xlabel('Diffusion Step')

        # cost_best.png: best trajectory cost, smoothed, no grid
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        cost_smooth = smooth(cost_best)
        ax.plot(x, cost_smooth, color='#1f77b4', linewidth=2.0, label='Cost')
        set_diffusion_axis(ax, n_steps)
        ax.set_ylabel('Cost')
        ax.set_title('Cost Over Diffusion Steps (Best Trajectory)')
        ax.legend()
        ax.set_ylim(bottom=0)
        fig.savefig(cost_dir / "cost_best.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

        # cost_modes.png: mean ± std over M samples at each diffusion step, smoothed, with bound
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        mean_c = np.mean(cost_matrix, axis=0)
        std_c = np.std(cost_matrix, axis=0)
        mean_smooth = smooth(mean_c)
        std_smooth = smooth(std_c)
        ax.fill_between(x, mean_smooth - std_smooth, mean_smooth + std_smooth, alpha=0.3, color='#1f77b4')
        ax.plot(x, mean_smooth, color='#1f77b4', linewidth=2.0, label='Mean cost')
        set_diffusion_axis(ax, n_steps)
        ax.set_ylabel('Cost')
        ax.set_title('Cost Over Diffusion Steps (Modes Mean ± Std)')
        ax.legend()
        ax.set_ylim(bottom=0)
        fig.savefig(cost_dir / "cost_modes.png", dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _generate_adaptive_visualization(self, result: Dict[str, Any], env_plugin: Any = None) -> None:
        """
        Generate CFS-MBD adaptive/non-adaptive scheduler metrics under result_dir/adaptive/.
        Plots: r_k, v_rate, v_mean, (c_k, nu for adaptive only), p_k, rho, topK, I_QP, eps, lambda
        vs diffusion step; mean across modes with ±std band. Saves metrics.json (C, K, 11) + best_idx.
        """
        import matplotlib.pyplot as plt

        planning_result = result.get('result', {})
        if not isinstance(planning_result, dict):
            return
        r_hist = planning_result.get('r_hist')
        if r_hist is None or (hasattr(r_hist, '__len__') and len(r_hist) == 0):
            return

        out_dir = self._get_output_path(result['level'], result['seed'])
        adaptive_dir = out_dir / "adaptive"
        adaptive_dir.mkdir(parents=True, exist_ok=True)

        # Multirun: use all_* stacked (C, K) to build (C, K, 11); else single (1, K, 11)
        all_r = planning_result.get('all_r_hist')
        if all_r is not None and hasattr(all_r, 'ndim') and all_r.ndim == 2:
            C, K = all_r.shape
            def get_all(key: str):
                h = planning_result.get('all_' + key)
                if h is not None and hasattr(h, 'shape') and h.shape == (C, K):
                    return np.asarray(h, dtype=np.float64)
                return np.full((C, K), np.nan, dtype=np.float64)
            metrics_arr = np.stack([
                get_all('r_hist'),
                get_all('v_rate_hist'),
                get_all('v_mean_hist'),
                get_all('compute_cost_hist'),
                get_all('nu_hist'),
                get_all('p_hist'),
                get_all('rho_hist'),
                get_all('topK_hist'),
                get_all('I_QP_hist'),
                get_all('eps_hist'),
                get_all('lambda_hist'),
            ], axis=-1)
            best_idx = int(planning_result.get('best_idx', 0))
            is_adaptive = bool(np.any(np.isfinite(metrics_arr[:, :, 3])))
        else:
            r_hist_arr = np.asarray(r_hist, dtype=np.float64).ravel()
            K = len(r_hist_arr)
            def get_hist(key: str, default_nan: bool = False):
                h = planning_result.get(key)
                if h is None:
                    return np.full(K, np.nan, dtype=np.float64)
                a = np.asarray(h, dtype=np.float64).ravel()
                if len(a) < K:
                    a = np.resize(np.asarray(a), K)
                return a[:K].copy()

            r_k = np.asarray(planning_result.get('r_hist'), dtype=np.float64).ravel()[:K]
            v_rate = np.asarray(planning_result.get('v_rate_hist'), dtype=np.float64).ravel()[:K]
            v_mean = np.asarray(planning_result.get('v_mean_hist'), dtype=np.float64).ravel()[:K]
            c_k = get_hist('compute_cost_hist', default_nan=True)
            nu = get_hist('nu_hist', default_nan=True)
            p_k = get_hist('p_hist')
            rho = get_hist('rho_hist')
            topK = get_hist('topK_hist')
            I_QP = get_hist('I_QP_hist')
            eps = get_hist('eps_hist')
            lam = get_hist('lambda_hist')

            is_adaptive = bool(np.any(np.isfinite(np.asarray(c_k, dtype=np.float64))))
            best_idx = int(planning_result.get('best_idx', 0))
            metrics_arr = np.stack([
                r_k, v_rate, v_mean, c_k, nu, p_k, rho, topK, I_QP, eps, lam
            ], axis=-1)
            metrics_arr = np.expand_dims(metrics_arr, axis=0)

        metric_names = ['r_k', 'v_rate', 'v_mean', 'c_k', 'nu', 'p_k', 'rho', 'topK', 'I_QP', 'eps', 'lambda']
        integer_metrics = {'topK', 'I_QP'}

        def set_diffusion_axis(ax, n_steps):
            tick_labels = [100, 80, 60, 40, 20, 1]
            tick_positions = [n_steps - k for k in tick_labels if 1 <= k <= n_steps]
            tick_labels = [k for k in tick_labels if 1 <= k <= n_steps]
            if not tick_positions:
                tick_positions = [0, n_steps - 1]
                tick_labels = [str(n_steps), "1"]
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([str(l) for l in tick_labels], fontsize=18)
            ax.set_xlabel('Diffusion Step', fontsize=18)
            ax.tick_params(axis='y', labelsize=18)

        plot_indices = list(range(11)) if is_adaptive else [0, 1, 2, 5, 6, 7, 8, 9, 10]
        x = np.arange(K)

        for i in plot_indices:
            name = metric_names[i]
            # (C, K) -> mean (K,), std (K,); avoid RuntimeWarning when slice is all nan
            vals = np.asarray(metrics_arr[:, :, i], dtype=np.float64)
            has_finite = np.any(np.isfinite(vals))
            if has_finite:
                y_mean = np.nanmean(vals, axis=0)
                y_std = np.nanstd(vals, axis=0)
            else:
                y_mean = np.full(K, np.nan, dtype=np.float64)
                y_std = np.zeros(K, dtype=np.float64)
            if np.any(np.isnan(y_std)):
                y_std = np.where(np.isnan(y_std), 0.0, y_std)
            if name in integer_metrics:
                y_plot = np.round(y_mean).astype(np.float64)
            else:
                y_plot = y_mean
            # Ensure std band is visible when C=1 (std=0) or very small std
            y_finite = y_plot[np.isfinite(y_plot)]
            if len(y_finite) > 0:
                y_range = float(np.nanmax(y_plot) - np.nanmin(y_plot)) + 1e-9
                y_scale = float(np.nanmax(np.abs(y_plot))) + 1e-9
                min_half_width = max(0.02 * y_range, 1e-6 * y_scale)
                y_std = np.maximum(np.asarray(y_std, dtype=np.float64), min_half_width)
            fig, ax = plt.subplots(1, 1, figsize=(8, 5))
            ax.fill_between(x, y_plot - y_std, y_plot + y_std, alpha=0.3, color='#1f77b4')
            ax.plot(x, y_plot, color='#1f77b4', linewidth=2.0, label=name)
            set_diffusion_axis(ax, K)
            ax.set_ylabel(name, fontsize=18)
            ax.set_title(f'{name} vs Diffusion Step (modes mean ± std)', fontsize=22, fontweight='bold')
            ax.legend()
            # topK, I_QP: integer y-axis; eps: scientific; others: 2 decimal places
            from matplotlib.ticker import FormatStrFormatter, MaxNLocator
            if name in integer_metrics:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))
            elif name == 'eps':
                ax.ticklabel_format(axis='y', style='scientific', scilimits=(-2, 2))
            else:
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            fig.savefig(adaptive_dir / f"{name}.png", dpi=150, bbox_inches='tight')
            plt.close(fig)

        def _nan_to_none(obj):
            if isinstance(obj, (list, tuple)):
                return [_nan_to_none(x) for x in obj]
            if isinstance(obj, dict):
                return {k: _nan_to_none(v) for k, v in obj.items()}
            if isinstance(obj, float) and (obj != obj or abs(obj) == float('inf')):
                return None
            return obj

        metrics_serializable = convert_to_json_serializable(metrics_arr)
        metrics_json = {
            "metrics": _nan_to_none(metrics_serializable),
            "metric_names": metric_names,
            "best_idx": best_idx,
        }
        with open(adaptive_dir / "metrics.json", 'w') as f:
            json.dump(metrics_json, f, indent=2)

        # Optional runtime diagnostics for profiling TwoGO-style compute paths.
        diag_keys = [
            "qp_call_hist",
            "qp_call_minibatch_hist",
            "qp_call_geom_hist",
            "qp_call_retract_hist",
            "rollout_eval_calls_hist",
            "m_k_hist",
            "p_hist",
            "I_QP_hist",
            "v_rate_hist",
            "v_mean_hist",
            "cvar_hist",
        ]
        diag_payload: Dict[str, Any] = {"best_idx": best_idx}
        for key in diag_keys:
            all_key = "all_" + key
            if all_key in planning_result:
                diag_payload[all_key] = _nan_to_none(convert_to_json_serializable(planning_result[all_key]))
            elif key in planning_result:
                diag_payload[key] = _nan_to_none(convert_to_json_serializable(planning_result[key]))
        if len(diag_payload) > 1:
            with open(adaptive_dir / "diagnostics.json", 'w') as f:
                json.dump(diag_payload, f, indent=2)

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
        
        # Add obstacle statistics
        serializable_result['num_obstacles'] = result.get('num_obstacles', 0)
        serializable_result['num_union_obstacles'] = result.get('num_union_obstacles', 0)
        serializable_result['num_primitives_total'] = result.get('num_primitives_total', 0)

        planning_result = result.get('result', {})
        cand_states = planning_result.get('candidate_states', []) if isinstance(planning_result, dict) else []
        num_modes = len(cand_states) if _has_items(cand_states) else 1
        if num_modes == 1 and isinstance(result.get('config_snapshot'), dict):
            method_params = result['config_snapshot'].get('method_params', {})
            if isinstance(method_params, dict) and 'num_modes' in method_params:
                num_modes = int(method_params['num_modes'])
        serializable_result['num_modes'] = num_modes

        metrics = result.get('metrics', {})
        
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
        
        # Keep metrics dict (ssr = modes (success & safe)/total; no safe/feasible/task_success)
        metrics_to_save = dict(metrics)
        if 'ssr' in metrics_to_save and isinstance(metrics_to_save['ssr'], dict):
            m = metrics_to_save['ssr']
            metrics_to_save['ssr'] = {
                'ssr': float(m.get('ssr', 0.0)),
                'ssr_count': int(m.get('ssr_count', 0)),
                'violation_rate_mean': float(m.get('violation_rate_mean', 0.0)),
                'violation_rate_std': float(m.get('violation_rate_std', 0.0)),
            }
        serializable_result['metrics'] = metrics_to_save
        
        # Add planning result data if available (omit large arrays exec_states/exec_actions from results.json)
        if isinstance(planning_result, dict):
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
        
        # Save multi-mode trajectories to trajectory.json (not in results.json)
        if 'candidate_states' in planning_result:
            candidate_states = planning_result['candidate_states']
            candidate_actions = planning_result.get('candidate_actions', [])
            candidate_costs = planning_result.get('candidate_costs', [])
            best_idx = int(planning_result.get('best_idx', 0))
            trajectory_json = {
                'best_idx': best_idx,
                'candidate_states': convert_to_json_serializable(candidate_states),
            }
            # Optional per-mode goal assignment metadata (useful for multi-target debugging).
            for k in ("candidate_goals_xy", "candidate_goals_idx", "use_target_line", "num_targets"):
                if k in planning_result and planning_result.get(k) is not None:
                    trajectory_json[k] = convert_to_json_serializable(planning_result.get(k))
            if candidate_actions:
                trajectory_json['candidate_actions'] = convert_to_json_serializable(candidate_actions)
            if len(candidate_costs) > 0:
                trajectory_json['candidate_costs'] = (
                    candidate_costs.tolist() if hasattr(candidate_costs, 'tolist') else list(candidate_costs)
                )
            trajectory_dir = output_path / "trajectory"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            with open(trajectory_dir / "trajectory.json", 'w') as f:
                json.dump(trajectory_json, f, indent=2)

        # Save lifted 9D trajectory for 4D D3IL runs when available.
        if isinstance(planning_result, dict) and (
            "states_9d" in planning_result or ("info" in planning_result and isinstance(planning_result.get("info"), dict) and "states_9d" in planning_result.get("info", {}))
        ):
            info_dict = planning_result.get("info", {}) if isinstance(planning_result.get("info"), dict) else {}
            states_9d = planning_result.get("states_9d", info_dict.get("states_9d", []))
            actions_9d = planning_result.get("actions_9d", info_dict.get("actions_9d", []))
            if states_9d:
                trajectory_dir = output_path / "trajectory"
                trajectory_dir.mkdir(parents=True, exist_ok=True)
                lifted_json = {
                    "state_layout": "[x, y, q1..q7]",
                    "action_layout": "[qdot1..qdot7]",
                    "states": convert_to_json_serializable(states_9d),
                    "actions": convert_to_json_serializable(actions_9d),
                    # Backward-compatible aliases.
                    "state_layout_9d": "[x, y, q1..q7]",
                    "action_layout_9d": "[qdot1..qdot7]",
                    "states_9d": convert_to_json_serializable(states_9d),
                    "actions_9d": convert_to_json_serializable(actions_9d),
                }
                with open(trajectory_dir / "trajectory_9d.json", "w") as f:
                    json.dump(lifted_json, f, indent=2)

        # Multirun diagnostics (architecture performance visibility)
        # These keys are produced by the solver when using the minimal-batch multirun path.
        if isinstance(planning_result, dict):
            for k in [
                "multirun_impl",
                "multirun_C",
                "multirun_t_batch_minimal_s",
                "multirun_t_best_plan_s",
            ]:
                if k in planning_result:
                    serializable_result[k] = convert_to_json_serializable(planning_result[k])

            # Generic timing diagnostics (saved if present)
            for k, v in planning_result.items():
                if isinstance(k, str) and k.startswith("timing_"):
                    serializable_result[k] = convert_to_json_serializable(v)

            # MBD3D / 3DGS: total_log_prob, n_steps for metrics
            if "total_log_prob" in planning_result:
                serializable_result["total_log_prob"] = convert_to_json_serializable(planning_result["total_log_prob"])
            if "n_steps" in planning_result:
                serializable_result["n_steps"] = convert_to_json_serializable(planning_result["n_steps"])
        
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
                pt_list = [r['planning_time'] for r in level_results]
                n_pt = len(pt_list)
                # Compute average metrics
                summary = {
                    'level': level,
                    'num_experiments': n_pt,
                    'avg_planning_time': float(np.mean(pt_list)),
                    'std_planning_time': float(np.std(pt_list)) if n_pt > 1 else 0.0,
                }
                
                # Average metrics
                for metric_name in self.config.metrics:
                    metric_values = []
                    for r in level_results:
                        if metric_name in r['metrics'] and r['metrics'][metric_name] is not None:
                            metric_val = r['metrics'][metric_name]
                            # Handle nested metrics (e.g., SSR dict, execution_ssr dict)
                            if isinstance(metric_val, dict) and 'ssr' in metric_val:
                                metric_values.append(metric_val['ssr'])
                            elif isinstance(metric_val, dict) and 'execution_ssr' in metric_val:
                                metric_values.append(metric_val['execution_ssr'])
                            elif isinstance(metric_val, (int, float)):
                                metric_values.append(metric_val)
                    
                    if metric_values:
                        summary[f'avg_{metric_name}'] = float(np.mean(metric_values))

                # Execution SSR (not in config.metrics): average when present
                exec_ssr_list = []
                for r in level_results:
                    m = r.get('metrics') or {}
                    if 'execution_ssr' in m and isinstance(m['execution_ssr'], dict):
                        v = m['execution_ssr'].get('execution_ssr')
                        if v is not None:
                            exec_ssr_list.append(float(v))
                if exec_ssr_list:
                    summary['avg_execution_ssr'] = float(np.mean(exec_ssr_list))

                # Best (best mode) metrics: average across level's results
                best_list = [
                    r['metrics']['best'] for r in level_results
                    if isinstance(r.get('metrics'), dict) and isinstance(r['metrics'].get('best'), dict)
                ]
                if best_list:
                    n_b = len(best_list)
                    pt_b = [b['planning_time'] for b in best_list]
                    len_b = [b['length'] for b in best_list]
                    sm_b = [b['smoothness'] for b in best_list]
                    gs_b = [b['geometric_smoothness'] for b in best_list]
                    summary['best'] = {
                        'planning_time': float(np.mean(pt_b)),
                        'planning_time_std': float(np.std(pt_b)) if n_b > 1 else 0.0,
                        'length': float(np.mean(len_b)),
                        'length_std': float(np.std(len_b)) if n_b > 1 else 0.0,
                        'smoothness': float(np.mean(sm_b)),
                        'smoothness_std': float(np.std(sm_b)) if n_b > 1 else 0.0,
                        'geometric_smoothness': float(np.mean(gs_b)),
                        'geometric_smoothness_std': float(np.std(gs_b)) if n_b > 1 else 0.0,
                    }
                    # Per-level aggregate over all seeds (same as best when one best per seed; explicit for clarity)
                    summary['length'] = float(np.mean(len_b))
                    summary['length_std'] = float(np.std(len_b)) if n_b > 1 else 0.0
                    summary['smoothness'] = float(np.mean(sm_b))
                    summary['smoothness_std'] = float(np.std(sm_b)) if n_b > 1 else 0.0
                    summary['geometric_smoothness'] = float(np.mean(gs_b))
                    summary['geometric_smoothness_std'] = float(np.std(gs_b)) if n_b > 1 else 0.0
                
                level_summaries[f'level_{level}'] = summary
                
                # Save level summary
                level_dir = self.config.output_dir / f"level_{level}"
                level_dir.mkdir(parents=True, exist_ok=True)
                with open(level_dir / "summary.json", 'w') as f:
                    json.dump(summary, f, indent=2)
        
        # Overall summary (include best: average of best-mode metrics across all results)
        best_list_all = [
            r['metrics']['best'] for r in all_results
            if isinstance(r.get('metrics'), dict) and isinstance(r['metrics'].get('best'), dict)
        ]
        pt_all = [r['planning_time'] for r in all_results]
        n_all = len(pt_all)
        overall_summary = {
            'total_experiments': n_all,
            'avg_planning_time': float(np.mean(pt_all)),
            'std_planning_time': float(np.std(pt_all)) if n_all > 1 else 0.0,
            'level_summaries': level_summaries,
        }
        if best_list_all:
            n_b_all = len(best_list_all)
            pt_b = [b['planning_time'] for b in best_list_all]
            len_b = [b['length'] for b in best_list_all]
            sm_b = [b['smoothness'] for b in best_list_all]
            gs_b = [b['geometric_smoothness'] for b in best_list_all]
            overall_summary['best'] = {
                'planning_time': float(np.mean(pt_b)),
                'planning_time_std': float(np.std(pt_b)) if n_b_all > 1 else 0.0,
                'length': float(np.mean(len_b)),
                'length_std': float(np.std(len_b)) if n_b_all > 1 else 0.0,
                'smoothness': float(np.mean(sm_b)),
                'smoothness_std': float(np.std(sm_b)) if n_b_all > 1 else 0.0,
                'geometric_smoothness': float(np.mean(gs_b)),
                'geometric_smoothness_std': float(np.std(gs_b)) if n_b_all > 1 else 0.0,
            }
        
        # Save overall summary
        with open(self.config.output_dir / "overall_summary.json", 'w') as f:
            json.dump(overall_summary, f, indent=2)
