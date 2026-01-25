"""
CFS-based per-step QP filter for action-space projection.

This filter uses CFS (Convex Feasible Set) to linearize obstacles,
then solves per-step QP to project actions into feasible set.
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
from enerdynamics.core.constraints.core.types import ScheduleState, ScheduleParams
from enerdynamics.core.types import Trajectory


class CFSQPPerStepFilter(ConstraintFilter):
    """
    CFS-based per-step QP filter.
    
    For each action sequence:
    1. Rollout states from actions
    2. Use CFS to generate per-step action-space constraints
    3. Solve per-step QP to project actions
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
    
    def _get_cfs_convexifier(self, obstacles, env):
        """Lazy initialization of CFS action convexifier."""
        if self._cfs_action_convexifier is None:
            from enerdynamics.core.constraints.convexify.cfs.action import CFSActionConvexifier
            self._cfs_action_convexifier = CFSActionConvexifier(
                obstacles=obstacles,
                env=env,
                action_mode="u_perstep",
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
    
    def _solve_per_step_qp_simple(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        rho: float,
    ) -> np.ndarray:
        """
        Simple per-step QP solver (slack-QP).
        
        min ||u - u_nom||^2 + rho * ||xi||^2
        s.t. A u >= b - xi, xi >= 0
        """
        if A.size == 0:
            return u_nom
        
        # Check violations
        violations = np.maximum(0, b - A @ u_nom)  # (m,)
        if violations.max() < 1e-7:
            return u_nom
        
        # For single constraint or small number, use analytical solution
        if A.shape[0] == 1:
            # Single constraint: analytical solution
            A_row = A[0]
            b_val = b[0]
            lhs = np.dot(A_row, u_nom)
            den = np.dot(A_row, A_row) + 1e-9
            violation = max(0.0, b_val - lhs)
            xi = violation / (1.0 + den / rho)
            alpha = xi / rho
            return u_nom + alpha * A_row
        else:
            # Multiple constraints: iterative projection
            u_safe = u_nom.copy()
            for _ in range(10):
                violations = np.maximum(0, b - A @ u_safe)
                if violations.max() < 1e-7:
                    break
                # Project onto most violated constraint
                worst_idx = np.argmax(violations)
                A_row = A[worst_idx]
                b_val = b[worst_idx]
                lhs = np.dot(A_row, u_safe)
                den = np.dot(A_row, A_row) + 1e-9
                violation = violations[worst_idx]
                xi = violation / (1.0 + den / rho)
                alpha = xi / rho
                u_safe = u_safe + alpha * A_row
            return u_safe
    
    def _solve_multi_constraint_qp_jax(
        self,
        u_nom: jnp.ndarray,
        A: jnp.ndarray,  # (m, act_dim)
        b: jnp.ndarray,  # (m,)
        rho: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Solve multi-constraint slack-QP in JAX.
        
        min ||u - u_nom||^2 + rho * ||xi||^2
        s.t. A @ u >= b - xi, xi >= 0
        
        Note: Constraints with b = inf are treated as inactive (filtered out).
        """
        if A.size == 0:
            return u_nom
        
        m = A.shape[0]
        if m == 0:
            return u_nom
        
        # Filter out invalid constraints (b = inf)
        valid_mask = jnp.isfinite(b)
        num_valid = jnp.sum(valid_mask.astype(jnp.int32))
        
        if num_valid == 0:
            # No valid constraints
            return u_nom
        
        # Extract valid constraints
        A_valid = A[valid_mask]  # (num_valid, act_dim)
        b_valid = b[valid_mask]  # (num_valid,)
        
        # Check violations
        violations = jnp.maximum(0.0, b_valid - A_valid @ u_nom)  # (num_valid,)
        max_violation = jnp.max(violations)
        
        # If no violation, return original
        if max_violation < 1e-7:
            return u_nom
        
        # For single constraint, use analytical solution
        if num_valid == 1:
            A_row = A_valid[0]
            b_val = b_valid[0]
            lhs = jnp.dot(A_row, u_nom)
            den = jnp.dot(A_row, A_row) + 1e-9
            violation = jnp.maximum(0.0, b_val - lhs)
            xi = violation / (1.0 + den / rho)
            alpha = xi / rho
            return u_nom + alpha * A_row
        
        # For multiple constraints, use iterative projection
        u_iter = u_nom
        for _ in range(10):
            violations = jnp.maximum(0.0, b_valid - A_valid @ u_iter)
            max_viol = jnp.max(violations)
            if max_viol < 1e-7:
                break
            
            # Project onto most violated constraint
            worst_idx = jnp.argmax(violations)
            A_row = A_valid[worst_idx]
            b_val = b_valid[worst_idx]
            lhs = jnp.dot(A_row, u_iter)
            den = jnp.dot(A_row, A_row) + 1e-9
            violation = violations[worst_idx]
            
            # Slack-QP solution
            xi = violation / (1.0 + den / rho)
            alpha = xi / rho
            u_iter = u_iter + alpha * A_row
        
        return u_iter
    
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
        """Filter actions using CFS + per-step QP."""
        if obstacles is None:
            return actions
        
        # Check if input is JAX array (in traced context)
        # We need to detect this BEFORE calling np.asarray, which will fail in traced context
        # Use safe detection that doesn't trigger array conversion
        is_jax = False
        if JAX_AVAILABLE:
            try:
                # Method 1: Check for JAX-specific attributes (safest)
                if hasattr(actions, 'block_until_ready'):
                    is_jax = True
                # Method 2: Check type name (doesn't trigger conversion)
                elif hasattr(actions, '__class__'):
                    class_name = type(actions).__name__
                    if 'Array' in class_name or 'DeviceArray' in class_name:
                        is_jax = True
                # Method 3: Check module name
                elif hasattr(actions, '__class__') and hasattr(actions.__class__, '__module__'):
                    module_name = actions.__class__.__module__
                    if 'jax' in module_name.lower():
                        is_jax = True
            except Exception:
                # If detection fails, assume it's not JAX (will try NumPy path)
                pass
        
        # For JAX arrays, implement JAX-compatible CFS filter
        # Similar to CBF filter: use jax.lax.scan to rollout and apply constraints per-step
        if is_jax:
            return self._apply_actions_jax(
                x0, actions, env=env, obstacles=obstacles,
                schedule_state=schedule_state, schedule_params=schedule_params,
            )
        
        # Try to convert to NumPy (will fail if it's a JAX traced array)
        try:
            actions_np = np.asarray(actions, dtype=np.float32)
        except (TypeError, ValueError) as e:
            # If conversion fails, it's likely a JAX traced array
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
        qp_prob = float(params_dict.get("qp_prob", 1.0))
        
        # Check qp_gate and qp_prob (only for NumPy path)
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
            sched_state = schedule_state  # Already a ScheduleState
        sched_params = ScheduleParams(margin=margin, rho=rho)
        
        # Get CFS convexifier
        cfs_convexifier = self._get_cfs_convexifier(obstacles, env)
        
        filtered_actions_list = []
        for batch_idx in range(actions_np.shape[0]):
            u_seq = actions_np[batch_idx]  # (H, act_dim)
            
            # Rollout states
            states = self._rollout_states(x0, u_seq, env)  # (H+1, state_dim)
            
            # Create reference trajectory for CFS
            ref_traj = Trajectory(
                states=[states[i] for i in range(len(states))],
                actions=[u_seq[i] for i in range(len(u_seq))],
                info={}
            )
            
            # Build CFS constraints (per-step action constraints)
            constraints = cfs_convexifier.build_constraints(ref_traj, sched_params, sched_state)
            A_ps = np.asarray(constraints.A)  # (H, max_k, act_dim) or (H, m, act_dim)
            b_ps = np.asarray(constraints.b)  # (H, max_k) or (H, m)
            
            # Solve per-step QP
            H = u_seq.shape[0]
            u_filtered = []
            for t in range(H):
                # Get constraints for this timestep
                if A_ps.ndim == 3:
                    A_t = A_ps[t]  # (max_k, act_dim)
                    b_t = b_ps[t]  # (max_k,)
                    # Filter out invalid constraints (marked with -1e9)
                    valid_mask = b_t > -1e8
                    if valid_mask.any():
                        A_t = A_t[valid_mask]
                        b_t = b_t[valid_mask]
                else:
                    # Single constraint matrix for all steps
                    A_t = A_ps
                    b_t = b_ps
                
                u_nom_t = u_seq[t]
                u_safe_t = self._solve_per_step_qp_simple(u_nom_t, A_t, b_t, rho)
                u_filtered.append(u_safe_t)
            
            filtered_actions_list.append(np.stack(u_filtered, axis=0))
        
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
        JAX-compatible CFS filter implementation with multi-constraint support.
        
        Similar to CBF filter: uses jax.lax.scan to rollout and apply constraints per-step.
        Supports multiple constraints per step for better handling of sharp corners.
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available for JAX filter implementation")
        
        params_dict = schedule_params or {}
        margin = float(params_dict.get("margin", 0.0)) if isinstance(params_dict.get("margin"), (float, int)) else params_dict.get("margin", jnp.asarray(0.0))
        rho = float(params_dict.get("rho", 10.0)) if isinstance(params_dict.get("rho"), (float, int)) else params_dict.get("rho", jnp.asarray(10.0))
        qp_gate = params_dict.get("qp_gate", True)
        qp_prob = params_dict.get("qp_prob", 1.0)
        
        # Convert to JAX arrays if needed
        if isinstance(margin, (float, int)):
            margin = jnp.asarray(float(margin), dtype=jnp.float32)
        if isinstance(rho, (float, int)):
            rho = jnp.asarray(float(rho), dtype=jnp.float32)
        
        # Check qp_gate (JAX-friendly)
        if isinstance(qp_gate, (bool, np.bool_)) and not qp_gate:
            return actions
        # Note: qp_prob random check can't be done in traced context, so we skip it
        
        dt = float(getattr(env, "dt", 0.05))
        robot_radius = float(getattr(env, "robot_radius", 0.05))
        clearance = margin + robot_radius if isinstance(margin, (float, int)) else margin + robot_radius
        
        # Get obstacles list (outside JAX traced context) - cache it
        obstacles_list = self._get_obstacles_list(obstacles)
        num_obstacles = len(obstacles_list) if obstacles_list else 0
        max_k = min(self.max_constraints_per_point, num_obstacles) if num_obstacles > 0 else 0
        
        # Compute threshold for obstacle selection
        threshold_val = float(clearance) + self.constraint_margin if isinstance(clearance, (float, int)) else float(clearance) + self.constraint_margin
        threshold = jnp.asarray(threshold_val, dtype=jnp.float32)
        
        # Build JAX-compatible obstacle SDF functions for multi-constraint
        # Strategy: Create a single function that computes SDF for all obstacles
        # This function will be called in traced context, so it must be JAX-compatible
        
        # Build obstacle SDF/gradient functions for multi-constraint
        # Strategy: Pre-compute functions for each obstacle (outside traced context)
        # Then use them in traced context via jax.lax.switch or direct calls
        
        # Create functions for each obstacle (similar to CFS JAX convexifier)
        # Pre-build branches for jax.lax.switch (must be done outside traced context)
        obstacle_branches = []
        if obstacles_list and len(obstacles_list) > 0:
            for obs in obstacles_list[:max_k]:
                if hasattr(obs, 'jax_sdf'):
                    # Create closure that captures this specific obstacle
                    def make_branch_fn(obstacle=obs):  # Default arg to avoid late binding
                        def sdf_fn(pos):
                            return obstacle.jax_sdf(pos)
                        def branch_fn(pos):
                            sdf_val = sdf_fn(pos)
                            grad_val = jax.grad(sdf_fn)(pos)
                            return sdf_val, grad_val
                        return branch_fn
                    obstacle_branches.append(make_branch_fn())
        
        num_obs_fns = len(obstacle_branches)
        can_use_multi = num_obs_fns > 0
        
        # Pad branches to max_k for fixed size (required for JAX)
        while len(obstacle_branches) < max_k:
            def dummy_branch(pos):
                return jnp.inf, jnp.zeros(2)
            obstacle_branches.append(dummy_branch)
        
        # Convert to tuple for JAX switch
        obstacle_branches_tuple = tuple(obstacle_branches[:max_k])
        
        def filter_single(u_seq):
            """Filter single action sequence using CFS + per-step QP with multiple constraints."""
            def body_fn(carry, u):
                x = carry  # Current state (position for single_2d)
                pos_t = x[0:2]  # Current position
                
                # Multi-constraint CFS: compute SDF and gradient for each obstacle
                if num_obstacles == 0 or max_k == 0:
                    # No obstacles - no constraint
                    x_next = env.jax_transition(x, u)
                    return x_next, u
                
                # Multi-constraint CFS: compute SDF and gradient for each obstacle
                # Use pre-computed obstacle functions (captured in closure)
                
                # Multi-constraint CFS: try to use individual obstacles if available
                if can_use_multi and num_obs_fns > 0:
                    # Compute SDF and gradient for each obstacle using pre-computed functions
                    # We'll use a fixed-size approach: compute for up to max_k obstacles
                    
                    def compute_obstacle_sdf_grad(obs_idx, pos):
                        """Compute SDF and gradient for obstacle at index obs_idx using jax.lax.switch."""
                        # Clip index to valid range for switch
                        obs_idx_clipped = jnp.clip(obs_idx, 0, max_k - 1)
                        
                        # Use switch to select the appropriate function (branches pre-computed in closure)
                        sdf_val, grad_val = jax.lax.switch(obs_idx_clipped, obstacle_branches_tuple, pos)
                        
                        # If obs_idx is out of range (>= num_obs_fns), return inf
                        valid = obs_idx < num_obs_fns
                        sdf_val = jnp.where(valid, sdf_val, jnp.inf)
                        grad_val = jnp.where(valid, grad_val, jnp.zeros(2))
                        
                        return sdf_val, grad_val
                    
                    # Compute SDF for all obstacles (up to max_k)
                    num_obs_to_check = min(max_k, num_obs_fns)
                    obs_indices = jnp.arange(max_k)  # Fixed size for JAX
                    
                    # Vectorize over obstacles
                    sdf_grad_results = jax.vmap(lambda idx: compute_obstacle_sdf_grad(idx, pos_t))(obs_indices)
                    sdf_array = sdf_grad_results[0]  # (max_k,)
                    grad_array = sdf_grad_results[1]  # (max_k, 2)
                    
                    # Select obstacles within threshold
                    cand_mask = sdf_array < threshold
                    num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
                    
                    # Select up to max_k closest obstacles
                    k = jnp.minimum(max_k, jnp.where(num_candidates == 0, num_obs_to_check, num_candidates))
                    
                    # Sort by SDF (ascending = closest first)
                    sorted_indices = jnp.argsort(sdf_array)
                    selected_indices = sorted_indices[:max_k]  # Fixed size for JAX
                    
                    # Build constraints for selected obstacles
                    def build_constraint_for_obstacle(idx):
                        """Build CFS constraint for obstacle at index idx."""
                        valid_idx = idx < k
                        obs_idx = selected_indices[idx]
                        
                        sdf_obs = jnp.where(valid_idx, sdf_array[obs_idx], jnp.inf)
                        grad_obs = jnp.where(valid_idx, grad_array[obs_idx], jnp.zeros(2))
                        
                        grad_norm = jnp.linalg.norm(grad_obs)
                        valid_grad = jnp.logical_and(valid_idx, grad_norm > 1e-8)
                        
                        # Normalize gradient
                        g = jnp.where(valid_grad, grad_obs / (grad_norm + 1e-9), jnp.zeros(2))
                        
                        # CFS constraint: b_state = (clearance - sdf) / ||grad|| + g^T pos_t
                        b_state = jnp.where(
                            valid_grad,
                            (clearance - sdf_obs) / (grad_norm + 1e-9) + jnp.dot(g, pos_t),
                            jnp.inf
                        )
                        
                        # Convert to action-space: (dt*g)^T u_t >= b_state - g^T p_t
                        A_row = dt * g  # (2,)
                        b_val = b_state - jnp.dot(g, pos_t)
                        
                        return A_row, b_val, valid_grad
                    
                    # Build constraints for all selected obstacles
                    constraint_results = jax.vmap(build_constraint_for_obstacle)(jnp.arange(max_k))
                    A_all = constraint_results[0]  # (max_k, 2)
                    b_all = constraint_results[1]  # (max_k,)
                    valid_mask = constraint_results[2]  # (max_k,)
                    
                    # Filter out invalid constraints
                    # Count valid constraints
                    num_valid = jnp.sum(valid_mask.astype(jnp.int32))
                    
                    # Use valid_mask to select constraints
                    # For JAX, we need to use a fixed-size approach
                    # We'll keep all max_k constraints, but mark invalid ones with inf in b
                    A_constraints = A_all  # (max_k, 2) - keep all for fixed size
                    b_constraints = jnp.where(valid_mask, b_all, jnp.inf)  # (max_k,) - mark invalid with inf
                else:
                    # Fallback to union SDF with enhanced safety factor
                    try:
                        if hasattr(obstacles, "sample_sdf_and_grad_2d"):
                            sdf_result, grad_result = obstacles.sample_sdf_and_grad_2d(
                                pos_t[None, :], backend="jax"
                            )
                            if isinstance(sdf_result, (list, tuple)):
                                sdf = jnp.asarray(sdf_result[0], dtype=jnp.float32)
                                grad = jnp.asarray(grad_result[0], dtype=jnp.float32)
                            else:
                                sdf = jnp.asarray(sdf_result, dtype=jnp.float32).flatten()
                                if sdf.shape[0] == 1:
                                    sdf = sdf[0]
                                grad = jnp.asarray(grad_result, dtype=jnp.float32)
                                if grad.ndim > 1:
                                    grad = grad[0]
                        elif hasattr(obstacles, "jax_sdf"):
                            sdf = obstacles.jax_sdf(pos_t)
                            def sdf_fn(pos):
                                return obstacles.jax_sdf(pos)
                            grad = jax.grad(sdf_fn)(pos_t)
                        else:
                            x_next = env.jax_transition(x, u)
                            return x_next, u
                        
                        grad = jnp.asarray(grad, dtype=jnp.float32).flatten()
                        if grad.shape[0] != 2:
                            if grad.shape[0] < 2:
                                grad = jnp.pad(grad, (0, 2 - grad.shape[0]))
                            else:
                                grad = grad[:2]
                        
                        grad_norm = jnp.linalg.norm(grad)
                        if grad_norm < 1e-8:
                            x_next = env.jax_transition(x, u)
                            return x_next, u
                        
                        g = grad / (grad_norm + 1e-9)
                        
                        # Enhanced safety factor for sharp corners (more aggressive)
                        # When very close, increase clearance significantly
                        safety_factor = jnp.where(
                            sdf < clearance * 1.2,  # Very close
                            1.5,  # Increase by 50%
                            jnp.where(
                                sdf < clearance * 2.0,  # Close
                                1.2,  # Increase by 20%
                                1.0   # Normal
                            )
                        )
                        effective_clearance = clearance * safety_factor
                        
                        b_state = (effective_clearance - sdf) / (grad_norm + 1e-9) + jnp.dot(g, pos_t)
                        A_row = dt * g
                        b_val = b_state - jnp.dot(g, pos_t)
                        
                        # Single constraint (union SDF)
                        A_constraints = A_row[None, :]  # (1, 2)
                        b_constraints = b_val[None]  # (1,)
                    except Exception:
                        # If SDF computation fails, return original action
                        x_next = env.jax_transition(x, u)
                        return x_next, u
                
                # Solve multi-constraint QP
                if A_constraints.shape[0] > 0:
                    u_safe = self._solve_multi_constraint_qp_jax(u, A_constraints, b_constraints, rho)
                else:
                    u_safe = u
                
                x_next = env.jax_transition(x, u_safe)
                return x_next, u_safe
            
            _, u_seq_safe = jax.lax.scan(body_fn, x0, u_seq)
            return u_seq_safe
        
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
