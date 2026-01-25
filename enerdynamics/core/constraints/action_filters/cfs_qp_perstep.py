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
            # Corrected slack-QP formula: lambda = violation / (den + 1/rho)
            # This makes rho larger -> harder constraint, rho smaller -> softer (allows slack)
            lam = violation / (den + 1.0 / rho)
            return u_nom + lam * A_row
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
                # Corrected slack-QP formula: lambda = violation / (den + 1/rho)
                lam = violation / (den + 1.0 / rho)
                u_safe = u_safe + lam * A_row
            return u_safe
    
    def _solve_multi_constraint_qp_jax(
        self,
        u_nom: jnp.ndarray,
        A: jnp.ndarray,  # (m, act_dim)
        b: jnp.ndarray,  # (m,)
        rho: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Solve multi-constraint slack-QP in JAX (JIT-safe version).
        
        min ||u - u_nom||^2 + rho * ||xi||^2
        s.t. A @ u >= b - xi, xi >= 0
        
        Note: Constraints with b = inf are treated as inactive (filtered out).
        Uses lax.cond and lax.fori_loop for JIT compatibility.
        """
        # Filter out invalid constraints (b = inf)
        valid_mask = jnp.isfinite(b)
        num_valid = jnp.sum(valid_mask.astype(jnp.int32))
        
        # Use lax.cond for empty/invalid cases (JIT-safe)
        def return_original(_):
            return u_nom
        
        def solve_qp(_):
            # JIT-safe: Don't use boolean indexing (requires concrete values)
            # Instead, mark invalid constraints with inf in b, and filter them in computation
            # Keep fixed-size arrays for JIT compatibility
            
            # Mark invalid constraints: set b to inf for invalid ones
            b_masked = jnp.where(valid_mask, b, jnp.inf)
            
            # Check violations (inf constraints will have inf violations, which we ignore)
            violations = jnp.maximum(0.0, b_masked - A @ u_nom)  # (m,)
            # Filter out inf violations (from invalid constraints)
            violations_clean = jnp.where(jnp.isfinite(violations), violations, -jnp.inf)
            max_violation = jnp.max(violations_clean)
            
            # Use lax.cond for no violation case
            def return_original_no_viol(_):
                return u_nom
            
            def solve_with_violations(_):
                # Use lax.cond for single vs multiple constraints
                def solve_single(_):
                    # Find first valid constraint (where b is finite)
                    # Since we can't use boolean indexing, we'll use argmax on violations_clean
                    # The first valid constraint will have the highest finite violation
                    # But for single constraint, we just need to find any valid one
                    # Use a simpler approach: find the constraint with the highest finite violation
                    # (which should be the only valid one if num_valid == 1)
                    worst_idx = jnp.argmax(violations_clean)
                    A_row = A[worst_idx]
                    b_val = b_masked[worst_idx]
                    
                    # Only apply if constraint is valid
                    is_valid = jnp.isfinite(b_val)
                    lhs = jnp.dot(A_row, u_nom)
                    den = jnp.dot(A_row, A_row) + 1e-9
                    violation = jnp.maximum(0.0, b_val - lhs)
                    violation = jnp.where(is_valid, violation, 0.0)
                    # Corrected slack-QP formula: lambda = violation / (den + 1/rho)
                    lam = violation / (den + 1.0 / rho)
                    return u_nom + jnp.where(is_valid, lam * A_row, 0.0)
                
                def solve_multiple(_):
                    # Use lax.fori_loop for iterative projection (JIT-safe)
                    # Fixed 30 iterations (increased from 10 for better convergence)
                    max_iter = 30
                    
                    def body_fn(i, u_iter):
                        violations = jnp.maximum(0.0, b_masked - A @ u_iter)
                        # Filter out inf violations
                        violations_clean = jnp.where(jnp.isfinite(violations), violations, -jnp.inf)
                        max_viol = jnp.max(violations_clean)
                        
                        # Project onto most violated constraint
                        worst_idx = jnp.argmax(violations_clean)
                        A_row = A[worst_idx]
                        b_val = b_masked[worst_idx]
                        
                        # Only project if constraint is valid
                        is_valid = jnp.isfinite(b_val)
                        lhs = jnp.dot(A_row, u_iter)
                        den = jnp.dot(A_row, A_row) + 1e-9
                        violation = violations_clean[worst_idx]
                        
                        # Corrected slack-QP formula: lambda = violation / (den + 1/rho)
                        violation = jnp.where(is_valid, violation, 0.0)
                        lam = violation / (den + 1.0 / rho)
                        u_next = u_iter + jnp.where(is_valid, lam * A_row, 0.0)
                        
                        return u_next
                    
                    u_result = jax.lax.fori_loop(0, max_iter, body_fn, u_nom)
                    return u_result
                
                # Branch on num_valid == 1
                return jax.lax.cond(
                    num_valid == 1,
                    solve_single,
                    solve_multiple,
                    operand=None
                )
            
            # Branch on max_violation < 1e-7
            return jax.lax.cond(
                max_violation < 1e-7,
                return_original_no_viol,
                solve_with_violations,
                operand=None
            )
        
        # Branch on num_valid == 0 or A.size == 0
        has_constraints = jnp.logical_and(A.size > 0, num_valid > 0)
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
        """Filter actions using CFS + per-step QP."""
        if obstacles is None:
            return actions
        
        # Check if input is JAX array/tracer (in traced context)
        # Use most reliable detection: check for jax.core.Tracer and jax.Array
        # This is critical: wrong detection leads to no-op in JIT/scan/vmap
        is_jax = False
        if JAX_AVAILABLE:
            try:
                from jax.core import Tracer
                # Check if actions is a JAX Array or Tracer (most reliable)
                is_jax = isinstance(actions, (jax.Array, Tracer))
            except Exception:
                # Fallback: check module name (less reliable but better than nothing)
                try:
                    module_name = type(actions).__module__.lower()
                    is_jax = 'jax' in module_name
                except Exception:
                    pass
        
        # Also check x0 (it might also be a JAX array/tracer)
        is_jax_x0 = False
        if JAX_AVAILABLE and not is_jax:
            try:
                from jax.core import Tracer
                is_jax_x0 = isinstance(x0, (jax.Array, Tracer))
            except Exception:
                try:
                    module_name = type(x0).__module__.lower()
                    is_jax_x0 = 'jax' in module_name
                except Exception:
                    pass
        
        # If either is JAX, use JAX path
        is_jax = is_jax or is_jax_x0
        
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
        # Get margin and rho - handle both concrete values and JAX arrays
        # In JIT context, these might be traced arrays, so we need to be careful
        margin_raw = params_dict.get("margin", 0.0)
        rho_raw = params_dict.get("rho", 10.0)
        qp_gate = params_dict.get("qp_gate", True)
        qp_prob = params_dict.get("qp_prob", 1.0)
        
        # Convert to JAX arrays (JIT-safe: use jnp.asarray which handles both concrete and traced values)
        # Check if already JAX array/tracer, otherwise convert
        try:
            # Try to check if it's already a JAX array/tracer
            if isinstance(margin_raw, (jax.Array, jnp.ndarray)):
                margin = margin_raw
            else:
                # It's a concrete value, convert to JAX array
                margin = jnp.asarray(float(margin_raw), dtype=jnp.float32)
        except (TypeError, ValueError):
            # Fallback: assume it's already a JAX array
            margin = margin_raw if isinstance(margin_raw, (jax.Array, jnp.ndarray)) else jnp.asarray(0.0, dtype=jnp.float32)
        
        try:
            if isinstance(rho_raw, (jax.Array, jnp.ndarray)):
                rho = rho_raw
            else:
                rho = jnp.asarray(float(rho_raw), dtype=jnp.float32)
        except (TypeError, ValueError):
            rho = rho_raw if isinstance(rho_raw, (jax.Array, jnp.ndarray)) else jnp.asarray(10.0, dtype=jnp.float32)
        
        # Check qp_gate (JAX-friendly)
        if isinstance(qp_gate, (bool, np.bool_)) and not qp_gate:
            return actions
        # Note: qp_prob random check can't be done in traced context, so we skip it
        
        # Get environment parameters (must be concrete values, computed outside traced context)
        # These are used to build closures, so they must be concrete
        dt = float(getattr(env, "dt", 0.05))
        robot_radius = float(getattr(env, "robot_radius", 0.05))
        constraint_margin_val = float(self.constraint_margin)
        
        # Convert margin to JAX array if needed (for JAX operations)
        margin_jax = margin if isinstance(margin, (jnp.ndarray, jax.Array)) else jnp.asarray(float(margin), dtype=jnp.float32)
        
        # Compute clearance using JAX operations (JIT-safe)
        robot_radius_jax = jnp.asarray(robot_radius, dtype=jnp.float32)
        clearance = margin_jax + robot_radius_jax
        
        # Get obstacles list (outside JAX traced context) - cache it
        obstacles_list = self._get_obstacles_list(obstacles)
        num_obstacles = len(obstacles_list) if obstacles_list else 0
        max_k = min(self.max_constraints_per_point, num_obstacles) if num_obstacles > 0 else 0
        
        # Compute threshold for obstacle selection (JIT-safe: use JAX operations)
        constraint_margin_jax = jnp.asarray(constraint_margin_val, dtype=jnp.float32)
        threshold = clearance + constraint_margin_jax
        
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
                    # CRITICAL FIX: Use cand_mask to filter BEFORE sorting
                    # Set out-of-threshold SDFs to +inf so they won't be selected
                    cand_mask = sdf_array < threshold
                    sdf_for_sort = jnp.where(cand_mask, sdf_array, jnp.inf)
                    
                    # Sort by SDF (ascending = closest first)
                    # Now only thresholded obstacles will be at the front
                    sorted_indices = jnp.argsort(sdf_for_sort)
                    selected_indices = sorted_indices[:max_k]  # Fixed size for JAX
                    
                    # Count valid candidates (for constraint building)
                    num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
                    k = jnp.minimum(max_k, jnp.where(num_candidates == 0, num_obs_to_check, num_candidates))
                    
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
                    # Fallback: No multi-constraint support (no individual obstacle jax_sdf)
                    # In JIT context, we cannot call Python object methods safely
                    # Return original action (no constraint applied)
                    # NOTE: This is a limitation - for proper JIT support, obstacles must expose jax_sdf
                    x_next = env.jax_transition(x, u)
                    return x_next, u
                
                # Solve multi-constraint QP (JIT-safe: use lax.cond)
                # Step 1: Add safety gate - only call QP if max_violation > tol
                # This avoids running QP when there's no violation, improving performance
                tol = 1e-7
                max_viol_before = jnp.max(jnp.maximum(0.0, b_constraints - A_constraints @ u))
                # Filter out inf violations
                max_viol_before_clean = jnp.where(jnp.isfinite(max_viol_before), max_viol_before, -jnp.inf)
                max_viol_before_clean = jnp.max(max_viol_before_clean)
                
                def solve_qp_wrapper(_):
                    return self._solve_multi_constraint_qp_jax(u, A_constraints, b_constraints, rho)
                
                def return_original(_):
                    return u
                
                # Gate: only solve QP if there are constraints AND violation > tol
                has_constraints = A_constraints.shape[0] > 0
                has_violation = max_viol_before_clean > tol
                should_solve = jnp.logical_and(has_constraints, has_violation)
                
                u_safe = jax.lax.cond(
                    should_solve,
                    solve_qp_wrapper,
                    return_original,
                    operand=None
                )
                
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
