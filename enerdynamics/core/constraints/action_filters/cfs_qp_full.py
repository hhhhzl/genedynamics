"""
CFS-based full trajectory QP filter for action-space projection.

This filter uses CFS (Convex Feasible Set) to linearize obstacles,
then solves a single QP over the entire action sequence to project actions into feasible set.
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.core.constraints.action_filters.base import ConstraintFilter
from enerdynamics.core.types import Trajectory


class CFSQPFullFilter(ConstraintFilter):
    """
    CFS-based full trajectory QP filter.
    
    For each action sequence:
    1. Rollout states from actions
    2. Use CFS to generate per-step action-space constraints
    3. Solve a single QP over the entire action sequence to project actions
    """
    
    def __init__(
        self,
        max_constraints_per_point: int = 8,
        constraint_margin: float = 0.25,
        use_slack: bool = True,
    ):
        self.max_constraints_per_point = max_constraints_per_point
        self.constraint_margin = constraint_margin
        self.use_slack = use_slack
        self._cfs_action_convexifier = None  # Lazy init
        self._obstacles_list = None  # Cached obstacle list for JAX multi-constraint
        self._num_obstacles = None  # Cached number of obstacles
        # Step C: Cache obstacle branches to avoid repeated compilation
        self._obstacle_branches_cache = None
        self._obstacles_cache_key = None
    
    def _get_cfs_convexifier(self, obstacles, env):
        """Lazy initialization of CFS action convexifier."""
        if self._cfs_action_convexifier is None:
            from enerdynamics.core.constraints.convexify.cfs.action import CFSActionConvexifier
            self._cfs_action_convexifier = CFSActionConvexifier(
                obstacles=obstacles,
                env=env,
                action_mode="u_traj",
                max_constraints_per_point=self.max_constraints_per_point,
                constraint_margin=self.constraint_margin,
                backend="numpy",
            )
        return self._cfs_action_convexifier
    
    def _get_obstacles_list(self, obstacles):
        """Get flattened list of obstacles (cached, computed outside JAX traced context)."""
        if self._obstacles_list is None and obstacles is not None:
            # Flatten obstacles (similar to EBMBDBackendJax._flatten_obstacles)
            obstacles_list = []
            if hasattr(obstacles, "obstacles"):
                try:
                    for child in list(getattr(obstacles, "obstacles", [])):
                        if hasattr(child, "obstacles"):
                            # Recursive flattening
                            for grandchild in list(getattr(child, "obstacles", [])):
                                obstacles_list.append(grandchild)
                        else:
                            obstacles_list.append(child)
                except Exception:
                    pass
            else:
                obstacles_list = [obstacles]
            
            # Filter to only obstacles with jax_sdf method (for JAX compatibility)
            self._obstacles_list = [obs for obs in obstacles_list if hasattr(obs, 'jax_sdf')]
            self._num_obstacles = len(self._obstacles_list)
        return self._obstacles_list
    
    def _rollout_states(self, x0: np.ndarray, actions: np.ndarray, env) -> np.ndarray:
        """Rollout states from actions."""
        states = [np.asarray(x0, dtype=np.float32)]
        s = np.asarray(x0, dtype=np.float32)
        for a in actions:
            # Use env.step or env.jax_transition
            if hasattr(env, "step"):
                try:
                    s, _, _, _ = env.step(s, a, 0, {})
                except TypeError:
                    s = env.step(s, a)
            elif hasattr(env, "model_transition"):
                s = env.model_transition(s, a)
            else:
                s = env.jax_transition(s, a)
            states.append(np.asarray(s, dtype=np.float32))
        return np.stack(states, axis=0)
    
    def _solve_full_trajectory_qp_numpy(
        self,
        u_nom_flat: np.ndarray,  # (H*act_dim,)
        A_full: np.ndarray,  # (H*max_k, H*act_dim)
        b_full: np.ndarray,  # (H*max_k,)
        rho: float,
        control_limit: float = 1.0,
    ) -> np.ndarray:
        """
        Solve full trajectory QP in NumPy.
        
        min ||u - u_nom||^2 + rho * ||xi||^2
        s.t. A_full @ u >= b_full - xi, xi >= 0
             -control_limit <= u <= control_limit  (box constraint via clip)
        """
        H_act_dim = u_nom_flat.shape[0]
        m = A_full.shape[0]
        L = float(control_limit)

        # Filter out invalid constraints (b = -inf)
        valid_mask = np.isfinite(b_full)
        num_valid = np.sum(valid_mask)

        if num_valid == 0 or A_full.size == 0:
            return np.clip(u_nom_flat, -L, L)

        # Filter to valid constraints
        A_valid = A_full[valid_mask]
        b_valid = b_full[valid_mask]

        # Check violations
        violations = np.maximum(0.0, b_valid - A_valid @ u_nom_flat)
        max_violation = np.max(violations) if len(violations) > 0 else 0.0

        if max_violation < 1e-7:
            return np.clip(u_nom_flat, -L, L)

        # Use iterative projection (similar to per-step, but on full trajectory)
        u_star = np.clip(u_nom_flat.copy(), -L, L)
        max_iter = 20
        tol = 1e-7

        for i in range(max_iter):
            violations = np.maximum(0.0, b_valid - A_valid @ u_star)
            max_viol = np.max(violations) if len(violations) > 0 else 0.0

            if max_viol < tol:
                break

            # Project onto most violated constraint
            worst_idx = np.argmax(violations)
            A_row = A_valid[worst_idx]
            b_val = b_valid[worst_idx]

            den = np.dot(A_row, A_row) + 1e-9
            violation = violations[worst_idx]

            # Hard projection or slack-QP
            if not self.use_slack or rho <= 0:
                lam = violation / den  # Hard projection
            else:
                lam = violation / (den + 1.0 / rho)  # Slack-QP

            u_star = u_star + lam * A_row
            u_star = np.clip(u_star, -L, L)

        return u_star
    
    def _solve_full_trajectory_qp_jax(
        self,
        u_nom_flat: jnp.ndarray,  # (H*act_dim,)
        A_full: jnp.ndarray,  # (H*max_k, H*act_dim)
        b_full: jnp.ndarray,  # (H*max_k,)
        rho: jnp.ndarray,
        control_limit: float = 1.0,
    ) -> jnp.ndarray:
        """
        Solve full trajectory QP in JAX (JIT-safe version).

        min ||u - u_nom||^2 + rho * ||xi||^2
        s.t. A_full @ u >= b_full - xi, xi >= 0
             -control_limit <= u <= control_limit  (box constraint via clip)
        """
        H_act_dim = u_nom_flat.shape[0]
        m = A_full.shape[0]
        L = float(control_limit)

        # Filter out invalid constraints (b = -inf for invalid, finite for valid)
        valid_mask = jnp.isfinite(b_full)
        num_valid = jnp.sum(valid_mask.astype(jnp.int32))

        # Use lax.cond for empty/invalid cases (JIT-safe)
        def return_original(_):
            return jnp.clip(u_nom_flat, -L, L)
        
        def solve_qp(_):
            b_masked = b_full  # Already has -inf for invalid constraints
            
            # Check violations
            violations = jnp.maximum(0.0, b_masked - A_full @ u_nom_flat)
            violations_clean = jnp.where(jnp.isfinite(violations), violations, -jnp.inf)
            max_violation = jnp.max(violations_clean)
            
            # Use lax.cond for no violation case
            def return_original_no_viol(_):
                return jnp.clip(u_nom_flat, -L, L)

            def solve_with_violations(_):
                # Use iterative projection with early stopping
                max_iter = 20
                tol = 1e-7
                u_init = jnp.clip(u_nom_flat, -L, L)

                def body_fn(i, carry):
                    u_iter, converged = carry
                    def continue_iter(_):
                        violations = jnp.maximum(0.0, b_masked - A_full @ u_iter)
                        violations_clean = jnp.where(jnp.isfinite(violations), violations, -jnp.inf)
                        max_viol = jnp.max(violations_clean)
                        
                        # Early stop
                        converged_now = max_viol < tol
                        
                        # Project onto most violated constraint
                        worst_idx = jnp.argmax(violations_clean)
                        A_row = A_full[worst_idx]
                        b_val = b_masked[worst_idx]
                        
                        is_valid = jnp.isfinite(b_val)
                        den = jnp.dot(A_row, A_row) + 1e-9
                        violation = violations_clean[worst_idx]
                        violation = jnp.where(is_valid, violation, 0.0)

                        # Hard projection or slack-QP
                        use_slack_flag = jnp.logical_and(rho > 0, jnp.asarray(self.use_slack, dtype=jnp.bool_))
                        lam = jax.lax.cond(
                            use_slack_flag,
                            lambda _: violation / (den + 1.0 / rho),  # Slack-QP
                            lambda _: violation / den,  # Hard projection
                            operand=None
                        )
                        u_next = u_iter + jnp.where(is_valid, lam * A_row, 0.0)
                        u_next = jnp.clip(u_next, -L, L)
                        return u_next, converged_now
                    
                    def skip_iter(_):
                        return u_iter, converged
                    
                    u_next, converged_new = jax.lax.cond(converged, skip_iter, continue_iter, operand=None)
                    return u_next, converged_new
                
                u_result, _ = jax.lax.fori_loop(0, max_iter, body_fn, (u_init, jnp.array(False)))
                return u_result
            
            return jax.lax.cond(
                max_violation < 1e-7,
                return_original_no_viol,
                solve_with_violations,
                operand=None
            )
        
        has_constraints = jnp.logical_and(A_full.size > 0, num_valid > 0)
        return jax.lax.cond(
            has_constraints,
            solve_qp,
            return_original,
            operand=None
        )
    
    def apply_actions(
        self,
        x0: Any,
        actions: Any,
        *,
        env: Any,
        obstacles: Any = None,
        schedule_state: Optional[Any] = None,
        schedule_params: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Filter actions using CFS + full trajectory QP."""
        if obstacles is None:
            return actions
        
        # Check if input is JAX array/tracer (in traced context)
        is_jax = False
        if JAX_AVAILABLE:
            try:
                from jax.core import Tracer
                is_jax = isinstance(actions, (jax.Array, Tracer))
            except Exception:
                try:
                    module_name = type(actions).__module__.lower()
                    is_jax = 'jax' in module_name
                except Exception:
                    pass
        
        if is_jax:
            try:
                return self._apply_actions_jax(
                    x0, actions, env=env, obstacles=obstacles,
                    schedule_state=schedule_state, schedule_params=schedule_params, **kwargs
                )
            except Exception as e:
                # Return unchanged (no-op)
                if 'tracer' in str(e).lower() or 'jax' in str(e).lower():
                    return actions
                raise
        
        # NumPy path: full CFS + QP filtering
        actions_np = np.asarray(actions, dtype=np.float32)
        is_batch = actions_np.ndim == 3
        if not is_batch:
            actions_np = actions_np[None, :, :]  # (1, H, act_dim)
        
        # Get schedule params
        params_dict = schedule_params or {}
        margin = float(params_dict.get("margin", 0.0))
        rho = float(params_dict.get("rho", 10.0))
        qp_gate = params_dict.get("qp_gate", True)
        qp_prob = params_dict.get("qp_prob", 1.0)
        cfs_outer_iters = max(1, int(params_dict.get("cfs_outer_iters", 1)))
        
        # Check qp_gate and qp_prob
        if isinstance(qp_gate, (bool, np.bool_)) and not qp_gate:
            return actions if not is_batch else actions_np[0]
        if isinstance(qp_prob, (float, np.floating)) and np.random.random() > qp_prob:
            return actions if not is_batch else actions_np[0]
        
        # Create ScheduleParams for CFS
        from enerdynamics.core.constraints.core.types import ScheduleParams, ScheduleState
        if schedule_state is None:
            sched_state = ScheduleState(k=0, K=1)
        elif isinstance(schedule_state, dict):
            sched_state = ScheduleState(
                k=schedule_state.get("k", 0),
                K=schedule_state.get("K", 1)
            )
        else:
            sched_state = schedule_state
        sched_params = ScheduleParams(margin=margin, rho=rho)
        
        # Get CFS convexifier
        cfs_convexifier = self._get_cfs_convexifier(obstacles, env)
        control_limit = float(getattr(env, "control_limit", 1.0))

        filtered_actions_list = []
        for batch_idx in range(actions_np.shape[0]):
            u_seq = np.asarray(actions_np[batch_idx], dtype=np.float32)  # (H, act_dim)
            H, act_dim = u_seq.shape
            L = float(control_limit)

            for outer_iter in range(cfs_outer_iters):
                # Rollout states from current nominal u
                states = self._rollout_states(x0, u_seq, env)  # (H+1, state_dim)

                ref_traj = Trajectory(
                    states=[states[i] for i in range(len(states))],
                    actions=[u_seq[i] for i in range(len(u_seq))],
                    info={},
                )

                constraints = cfs_convexifier.build_constraints(ref_traj, sched_params, sched_state)
                A_ps = np.asarray(constraints.A)
                b_ps = np.asarray(constraints.b)

                if A_ps.ndim == 3:
                    H_ps, max_k_ps, act_dim_ps = A_ps.shape
                    A_full = np.zeros((H_ps * max_k_ps, H_ps * act_dim_ps), dtype=np.float32)
                    b_full = np.zeros(H_ps * max_k_ps, dtype=np.float32)
                    for t in range(H_ps):
                        valid_mask_t = b_ps[t] > -1e8
                        if valid_mask_t.any():
                            A_t = A_ps[t][valid_mask_t]
                            b_t = b_ps[t][valid_mask_t]
                            row_start = t * max_k_ps
                            col_start = t * act_dim_ps
                            num_valid = int(np.sum(valid_mask_t))
                            A_full[row_start : row_start + num_valid, col_start : col_start + act_dim_ps] = A_t
                            b_full[row_start : row_start + num_valid] = b_t
                        else:
                            row_start = t * max_k_ps
                            b_full[row_start : row_start + max_k_ps] = -np.inf
                elif A_ps.ndim == 2 and A_ps.shape[1] == H * act_dim:
                    valid = np.isfinite(b_ps)
                    if np.any(valid):
                        A_full = np.asarray(A_ps[valid], dtype=np.float32)
                        b_full = np.asarray(b_ps[valid], dtype=np.float32)
                    else:
                        A_full = np.zeros((0, H * act_dim), dtype=np.float32)
                        b_full = np.zeros(0, dtype=np.float32)
                else:
                    A_full = np.zeros((H, H * act_dim), dtype=np.float32)
                    b_full = np.full(H, -np.inf, dtype=np.float32)

                if outer_iter == 0:
                    n_valid = 0
                    n_infeasible = 0
                    if A_ps.ndim == 3:
                        for t in range(A_ps.shape[0]):
                            for k in range(A_ps.shape[1]):
                                if b_ps[t, k] <= -1e8:
                                    continue
                                n_valid += 1
                                A_row = np.asarray(A_ps[t, k, :], dtype=np.float64)
                                b_val = float(b_ps[t, k])
                                max_lhs = L * float(np.sum(np.abs(A_row)))
                                if b_val > max_lhs:
                                    n_infeasible += 1
                        label = "u_perstep"
                    elif A_ps.ndim == 2 and A_ps.shape[1] == H * act_dim:
                        for i in range(A_ps.shape[0]):
                            if not np.isfinite(b_ps[i]):
                                continue
                            n_valid += 1
                            A_row = np.asarray(A_ps[i, :], dtype=np.float64)
                            b_val = float(b_ps[i])
                            max_lhs = L * float(np.sum(np.abs(A_row)))
                            if b_val > max_lhs:
                                n_infeasible += 1
                        label = "u_traj"
                    else:
                        label = None
                    if label and n_valid > 0:
                        ratio = n_infeasible / n_valid
                        print(
                            f"[CFS {label}] valid={n_valid} infeasible(b>L*||A||_1)={n_infeasible} "
                            f"ratio={ratio:.2%} (L={L}) outer_iters={cfs_outer_iters}"
                        )

                u_nom_flat = u_seq.flatten()
                u_safe_flat = self._solve_full_trajectory_qp_numpy(
                    u_nom_flat, A_full, b_full, rho, control_limit=control_limit
                )
                u_safe = np.asarray(u_safe_flat.reshape(H, act_dim), dtype=np.float32)
                u_seq = u_safe

            filtered_actions_list.append(u_seq)
        
        result = np.stack(filtered_actions_list, axis=0) if is_batch else filtered_actions_list[0]
        return result
    
    def _apply_actions_jax(
        self,
        x0: Any,
        actions: Any,
        *,
        env: Any,
        obstacles: Any = None,
        schedule_state: Optional[Any] = None,
        schedule_params: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """
        JAX-compatible CFS filter implementation with full trajectory QP.
        
        Similar to per-step version, but solves a single QP over the entire action sequence.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available for JAX filter implementation")
        
        params_dict = schedule_params or {}
        margin_raw = params_dict.get("margin", 0.0)
        rho_raw = params_dict.get("rho", 10.0)
        qp_gate = params_dict.get("qp_gate", True)
        qp_prob = params_dict.get("qp_prob", 1.0)
        
        # Convert to JAX arrays
        try:
            if isinstance(margin_raw, (jax.Array, jnp.ndarray)):
                margin = margin_raw
            else:
                margin = jnp.asarray(float(margin_raw), dtype=jnp.float32)
        except (TypeError, ValueError):
            margin = margin_raw if isinstance(margin_raw, (jax.Array, jnp.ndarray)) else jnp.asarray(0.0, dtype=jnp.float32)
        
        try:
            if isinstance(rho_raw, (jax.Array, jnp.ndarray)):
                rho = rho_raw
            else:
                rho = jnp.asarray(float(rho_raw), dtype=jnp.float32)
        except (TypeError, ValueError):
            rho = rho_raw if isinstance(rho_raw, (jax.Array, jnp.ndarray)) else jnp.asarray(10.0, dtype=jnp.float32)
        
        # Check qp_gate
        if isinstance(qp_gate, (bool, np.bool_)) and not qp_gate:
            return actions
        
        # Get environment parameters
        dt = float(getattr(env, "dt", 0.05))
        robot_radius = float(getattr(env, "robot_radius", 0.05))
        control_limit = float(getattr(env, "control_limit", 1.0))
        constraint_margin_val = float(self.constraint_margin)
        
        margin_jax = margin if isinstance(margin, (jnp.ndarray, jax.Array)) else jnp.asarray(float(margin), dtype=jnp.float32)
        robot_radius_jax = jnp.asarray(robot_radius, dtype=jnp.float32)
        clearance = margin_jax + robot_radius_jax
        constraint_margin_jax = jnp.asarray(constraint_margin_val, dtype=jnp.float32)
        threshold = clearance + constraint_margin_jax
        
        # Get obstacles list
        obstacles_list = self._get_obstacles_list(obstacles)
        num_obstacles = len(obstacles_list) if obstacles_list else 0
        max_k = min(self.max_constraints_per_point, num_obstacles) if num_obstacles > 0 else 0
        
        # Cache obstacle branches (same as per-step version)
        obstacles_cache_key = (
            id(obstacles) if obstacles is not None else None,
            max_k,
            dt,
            constraint_margin_val
        )
        
        if (self._obstacle_branches_cache is not None and 
            self._obstacles_cache_key == obstacles_cache_key and
            len(self._obstacle_branches_cache[0]) >= max_k):
            obstacle_branches_sdf_tuple = self._obstacle_branches_cache[0][:max_k]
            obstacle_branches_grad_tuple = self._obstacle_branches_cache[1][:max_k]
            num_obs_fns = min(len(self._obstacle_branches_cache[0]), max_k)
            can_use_multi = num_obs_fns > 0
        else:
            obstacle_branches_sdf = []
            obstacle_branches_grad = []
            
            if obstacles_list and len(obstacles_list) > 0:
                for obs in obstacles_list[:max_k]:
                    if hasattr(obs, 'jax_sdf'):
                        def make_branch_fns(obstacle=obs):
                            def sdf_fn(pos):
                                return obstacle.jax_sdf(pos)
                            
                            if hasattr(obstacle, 'jax_gradient'):
                                def sdf_only_fn(pos):
                                    return obstacle.jax_sdf(pos)
                                
                                def sdf_grad_fn(pos):
                                    sdf_val = obstacle.jax_sdf(pos)
                                    grad_val = obstacle.jax_gradient(pos)
                                    return sdf_val, grad_val
                            else:
                                grad_fn = jax.grad(sdf_fn)
                                
                                def sdf_only_fn(pos):
                                    return obstacle.jax_sdf(pos)
                                
                                def sdf_grad_fn(pos):
                                    sdf_val = sdf_fn(pos)
                                    grad_val = grad_fn(pos)
                                    return sdf_val, grad_val
                            
                            return sdf_only_fn, sdf_grad_fn
                        
                        sdf_fn, grad_fn = make_branch_fns()
                        obstacle_branches_sdf.append(sdf_fn)
                        obstacle_branches_grad.append(grad_fn)
            
            num_obs_fns = len(obstacle_branches_sdf)
            can_use_multi = num_obs_fns > 0
            
            while len(obstacle_branches_sdf) < max_k:
                def dummy_sdf_fn(pos):
                    return jnp.inf
                def dummy_grad_fn(pos):
                    return jnp.inf, jnp.zeros(2)
                obstacle_branches_sdf.append(dummy_sdf_fn)
                obstacle_branches_grad.append(dummy_grad_fn)
            
            obstacle_branches_sdf_tuple = tuple(obstacle_branches_sdf[:max_k])
            obstacle_branches_grad_tuple = tuple(obstacle_branches_grad[:max_k])
            self._obstacle_branches_cache = (obstacle_branches_sdf_tuple, obstacle_branches_grad_tuple)
            self._obstacles_cache_key = obstacles_cache_key
        
        def filter_single(u_seq):
            """Filter single action sequence using CFS + full trajectory QP."""
            H, act_dim = u_seq.shape
            
            def body_fn(carry, u):
                x = carry
                pos_t = x[0:2]
                x_next = env.jax_transition(x, u)
                return x_next, u
            
            # Rollout states to get positions for constraint building
            # Use scan to rollout states (JAX-compatible)
            def rollout_step(carry, u):
                x = carry
                x_next = env.jax_transition(x, u)
                return x_next, x_next
            
            _, states_rollout = jax.lax.scan(rollout_step, x0, u_seq)
            # Prepend initial state
            states_rollout = jnp.concatenate([x0[None, :], states_rollout], axis=0)  # (H+1, state_dim)
            
            # Build constraints for all timesteps
            if num_obstacles == 0 or max_k == 0:
                # No obstacles - return original
                return u_seq
            
            if can_use_multi and num_obs_fns > 0:
                # Build constraints for each timestep
                def build_constraints_for_timestep(t):
                    pos_t = states_rollout[t][0:2]
                    
                    # Stage 1: Compute SDF for all obstacles
                    def compute_obstacle_sdf_only(obs_idx, pos):
                        obs_idx_clipped = jnp.clip(obs_idx, 0, max_k - 1)
                        sdf_val = jax.lax.switch(obs_idx_clipped, obstacle_branches_sdf_tuple, pos)
                        valid = obs_idx < num_obs_fns
                        return jnp.where(valid, sdf_val, jnp.inf)
                    
                    obs_indices = jnp.arange(max_k)
                    sdf_array = jax.vmap(lambda idx: compute_obstacle_sdf_only(idx, pos_t))(obs_indices)
                    
                    # Gate: check if constraints needed
                    min_sdf = jnp.min(sdf_array)
                    needs_constraints = min_sdf < threshold
                    
                    def compute_constraints(_):
                        cand_mask = sdf_array < threshold
                        sdf_for_sort = jnp.where(cand_mask, sdf_array, jnp.inf)
                        
                        neg_sdf = -sdf_for_sort
                        topk_neg_values, topk_indices = jax.lax.top_k(neg_sdf, max_k)
                        selected_indices = jnp.flip(topk_indices, axis=0)
                        
                        num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
                        k = jnp.minimum(max_k, jnp.where(num_candidates == 0, max_k, num_candidates))
                        
                        def compute_obstacle_grad_for_candidate(rank_idx, pos):
                            obs_idx = selected_indices[rank_idx]
                            obs_idx_clipped = jnp.clip(obs_idx, 0, max_k - 1)
                            sdf_val, grad_val = jax.lax.switch(obs_idx_clipped, obstacle_branches_grad_tuple, pos)
                            valid_rank = rank_idx < k
                            valid_obs = obs_idx < num_obs_fns
                            valid = jnp.logical_and(valid_rank, valid_obs)
                            sdf_val = jnp.where(valid, sdf_val, jnp.inf)
                            grad_val = jnp.where(valid, grad_val, jnp.zeros(2))
                            return sdf_val, grad_val
                        
                        grad_results = jax.vmap(lambda rank_idx: compute_obstacle_grad_for_candidate(rank_idx, pos_t))(jnp.arange(max_k))
                        grad_array = grad_results[1]
                        
                        def build_constraint_for_obstacle(idx):
                            valid_idx = idx < k
                            obs_idx = selected_indices[idx]
                            sdf_obs = jnp.where(valid_idx, sdf_array[obs_idx], jnp.inf)
                            grad_obs = jnp.where(valid_idx, grad_array[idx], jnp.zeros(2))
                            
                            grad_norm = jnp.linalg.norm(grad_obs)
                            valid_grad = jnp.logical_and(valid_idx, grad_norm > 1e-8)
                            
                            g = jnp.where(valid_grad, grad_obs / (grad_norm + 1e-9), jnp.zeros(2))
                            
                            b_state = jnp.where(
                                valid_grad,
                                (clearance - sdf_obs) / (grad_norm + 1e-9) + jnp.dot(g, pos_t),
                                -jnp.inf
                            )
                            
                            A_row = dt * g
                            b_val = b_state - jnp.dot(g, pos_t)
                            
                            return A_row, b_val, valid_grad
                        
                        constraint_results = jax.vmap(build_constraint_for_obstacle)(jnp.arange(max_k))
                        A_all = constraint_results[0]
                        b_all = constraint_results[1]
                        valid_mask = constraint_results[2]
                        
                        A_constraints = A_all
                        b_constraints = jnp.where(valid_mask, b_all, -jnp.inf)
                        
                        return A_constraints, b_constraints
                    
                    def skip_constraints(_):
                        A_empty = jnp.zeros((max_k, 2))
                        b_empty = jnp.full((max_k,), -jnp.inf)
                        return A_empty, b_empty
                    
                    A_t, b_t = jax.lax.cond(needs_constraints, compute_constraints, skip_constraints, operand=None)
                    return A_t, b_t
                
                # Build constraints for all timesteps
                constraint_results = jax.vmap(build_constraints_for_timestep)(jnp.arange(H))
                A_per_step = constraint_results[0]  # (H, max_k, act_dim)
                b_per_step = constraint_results[1]  # (H, max_k)
                
                # Stack into full trajectory constraints
                # A_full: (H*max_k, H*act_dim), b_full: (H*max_k,)
                H_ps, max_k_ps, act_dim_ps = A_per_step.shape
                
                # Build block diagonal matrix using JAX operations
                def build_block_diagonal(t):
                    row_start = t * max_k_ps
                    col_start = t * act_dim_ps
                    # Create a sparse matrix for this timestep
                    A_block = jnp.zeros((H_ps * max_k_ps, H_ps * act_dim_ps), dtype=jnp.float32)
                    A_block = A_block.at[row_start:row_start+max_k_ps, col_start:col_start+act_dim_ps].set(A_per_step[t])
                    b_block = jnp.full(H_ps * max_k_ps, -jnp.inf, dtype=jnp.float32)
                    b_block = b_block.at[row_start:row_start+max_k_ps].set(b_per_step[t])
                    return A_block, b_block
                
                # Sum all blocks (they don't overlap, so sum works)
                A_blocks, b_blocks = jax.vmap(build_block_diagonal)(jnp.arange(H_ps))
                A_full = jnp.sum(A_blocks, axis=0)  # (H*max_k, H*act_dim)
                # For b, we need to take max (since invalid are -inf)
                b_full = jnp.max(b_blocks, axis=0)  # (H*max_k,)
                
                # Flatten actions
                u_nom_flat = u_seq.flatten()  # (H*act_dim,)
                
                # Solve full trajectory QP (with box constraint -control_limit <= u <= control_limit)
                u_safe_flat = self._solve_full_trajectory_qp_jax(
                    u_nom_flat, A_full, b_full, rho, control_limit=control_limit
                )

                # Reshape back
                u_safe = u_safe_flat.reshape(H, act_dim)
                return u_safe
            else:
                return u_seq
        
        # Handle batching
        if actions.ndim == 3:
            return jax.vmap(filter_single)(actions)
        return filter_single(actions)
    
    def apply_actions_batch(
        self,
        x0: Any,
        actions_batch: Any,
        *,
        env: Any,
        obstacles: Any = None,
        schedule_state: Optional[Any] = None,
        schedule_params: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Batch version - already handled in apply_actions."""
        return self.apply_actions(
            x0, actions_batch, env=env, obstacles=obstacles,
            schedule_state=schedule_state, schedule_params=schedule_params, **kwargs
        )
