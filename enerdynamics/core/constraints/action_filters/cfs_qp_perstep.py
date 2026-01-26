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
            s = self._step_once(s, a, env)
            states.append(np.asarray(s, dtype=np.float32))
        return np.stack(states, axis=0)

    def _step_once(self, x: np.ndarray, u: np.ndarray, env) -> np.ndarray:
        """Single transition x_next = f(x, u). Uses same logic as rollout."""
        if hasattr(env, "step"):
            try:
                s, _, _, _ = env.step(x, u, 0, {})
            except TypeError:
                s = env.step(x, u)
            return np.asarray(s, dtype=np.float32)
        if hasattr(env, "model_transition"):
            return np.asarray(env.model_transition(x, u), dtype=np.float32)
        return np.asarray(env.jax_transition(x, u), dtype=np.float32)
    
    def _solve_per_step_qp_simple(
        self,
        u_nom: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        rho: float,
        control_limit: float = 1.0,
    ) -> np.ndarray:
        """
        Simple per-step QP solver (slack-QP).

        min ||u - u_nom||^2 + rho * ||xi||^2
        s.t. A u >= b - xi, xi >= 0
             -control_limit <= u <= control_limit  (box via clip)
        """
        L = float(control_limit)
        if A.size == 0:
            return np.clip(u_nom, -L, L)

        # Check violations
        violations = np.maximum(0, b - A @ u_nom)  # (m,)
        if violations.max() < 1e-7:
            return np.clip(u_nom, -L, L)

        # For single constraint or small number, use analytical solution
        if A.shape[0] == 1:
            A_row = A[0]
            b_val = b[0]
            lhs = np.dot(A_row, u_nom)
            den = np.dot(A_row, A_row) + 1e-9
            violation = max(0.0, b_val - lhs)
            lam = violation / (den + 1.0 / rho)
            return np.clip(u_nom + lam * A_row, -L, L)
        else:
            # Multiple constraints: iterative projection
            u_safe = np.clip(u_nom.copy(), -L, L)
            for _ in range(10):
                violations = np.maximum(0, b - A @ u_safe)
                if violations.max() < 1e-7:
                    break
                worst_idx = np.argmax(violations)
                A_row = A[worst_idx]
                b_val = b[worst_idx]
                lhs = np.dot(A_row, u_safe)
                den = np.dot(A_row, A_row) + 1e-9
                violation = violations[worst_idx]
                lam = violation / (den + 1.0 / rho)
                u_safe = np.clip(u_safe + lam * A_row, -L, L)
            return u_safe
    
    def _solve_multi_constraint_qp_jax(
        self,
        u_nom: jnp.ndarray,
        A: jnp.ndarray,  # (m, act_dim)
        b: jnp.ndarray,  # (m,)
        rho: jnp.ndarray,
        control_limit: float = 1.0,
    ) -> jnp.ndarray:
        """
        Solve multi-constraint slack-QP in JAX (JIT-safe version).

        min ||u - u_nom||^2 + rho * ||xi||^2
        s.t. A @ u >= b - xi, xi >= 0
             -control_limit <= u <= control_limit  (box via clip)

        Note: Constraints with b = -inf are treated as inactive (filtered out).
        Uses lax.cond and lax.fori_loop for JIT compatibility.
        """
        L = float(control_limit)
        valid_mask = jnp.isfinite(b)
        num_valid = jnp.sum(valid_mask.astype(jnp.int32))

        def return_original(_):
            return jnp.clip(u_nom, -L, L)
        
        def solve_qp(_):
            # JIT-safe: Don't use boolean indexing (requires concrete values)
            # Fix 1: b already uses -inf for invalid constraints, so we can use it directly
            # Keep fixed-size arrays for JIT compatibility
            b_masked = b  # Already has -inf for invalid constraints
            
            # Check violations (inf constraints will have inf violations, which we ignore)
            violations = jnp.maximum(0.0, b_masked - A @ u_nom)  # (m,)
            # Filter out inf violations (from invalid constraints)
            violations_clean = jnp.where(jnp.isfinite(violations), violations, -jnp.inf)
            max_violation = jnp.max(violations_clean)
            
            def return_original_no_viol(_):
                return jnp.clip(u_nom, -L, L)

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
                    # Fix 2: Support hard projection when use_slack=False or rho <= 0
                    use_slack_flag = jnp.logical_and(rho > 0, jnp.asarray(self.use_slack, dtype=jnp.bool_))
                    # Hard projection: lambda = violation / den
                    # Slack-QP: lambda = violation / (den + 1/rho)
                    lam = jax.lax.cond(
                        use_slack_flag,
                        lambda _: violation / (den + 1.0 / rho),  # Slack-QP
                        lambda _: violation / den,  # Hard projection
                        operand=None
                    )
                    u_out = u_nom + jnp.where(is_valid, lam * A_row, 0.0)
                    return jnp.clip(u_out, -L, L)

                def solve_multiple(_):
                    # Fix 3: Use early stopping with reduced max_iter (5-10 is usually enough)
                    max_iter = 10  # Reduced from 30
                    tol = 1e-7
                    
                    def body_fn(i, carry):
                        u_iter, converged = carry
                        def continue_iter(_):
                            violations = jnp.maximum(0.0, b_masked - A @ u_iter)
                            # Filter out inf violations (invalid constraints have -inf, so violation = 0)
                            violations_clean = jnp.where(jnp.isfinite(violations), violations, -jnp.inf)
                            max_viol = jnp.max(violations_clean)
                            
                            # Early stop: if max_viol < tol, mark as converged
                            converged_now = max_viol < tol
                            
                            # Project onto most violated constraint
                            worst_idx = jnp.argmax(violations_clean)
                            A_row = A[worst_idx]
                            b_val = b_masked[worst_idx]
                            
                            # Only project if constraint is valid
                            is_valid = jnp.isfinite(b_val)
                            lhs = jnp.dot(A_row, u_iter)
                            den = jnp.dot(A_row, A_row) + 1e-9
                            violation = violations_clean[worst_idx]
                            violation = jnp.where(is_valid, violation, 0.0)
                            
                            # Fix 2: Support hard projection when use_slack=False or rho <= 0
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

                    u_init = jnp.clip(u_nom, -L, L)
                    u_result, _ = jax.lax.fori_loop(0, max_iter, body_fn, (u_init, jnp.array(False)))
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
        cfs_outer_iters = max(1, int(params_dict.get("cfs_outer_iters", 1)))

        # Check qp_gate and qp_prob (only for NumPy path)
        if isinstance(qp_gate, (bool, np.bool_)) and not qp_gate:
            return actions if not is_batch else actions_np[0]
        if isinstance(qp_prob, (float, np.floating)) and np.random.random() > qp_prob:
            return actions if not is_batch else actions_np[0]

        control_limit = float(getattr(env, "control_limit", 1.0))

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
        x0_use = np.asarray(x0, dtype=np.float32).reshape(-1)
        for batch_idx in range(actions_np.shape[0]):
            u_seq = np.asarray(actions_np[batch_idx], dtype=np.float32)  # (H, act_dim)

            for _ in range(cfs_outer_iters):
                H = u_seq.shape[0]
                u_filtered = []
                x = np.asarray(x0_use, dtype=np.float32).copy()

                for t in range(H):
                    u_nom_t = u_seq[t]
                    x_next_nom = self._step_once(x, u_nom_t, env)
                    ref_traj = Trajectory(
                        states=[x, x_next_nom],
                        actions=[u_nom_t],
                        info={},
                    )
                    constraints = cfs_convexifier.build_constraints(
                        ref_traj, sched_params, sched_state
                    )
                    A_ps = np.asarray(constraints.A)  # (1, max_k, act_dim)
                    b_ps = np.asarray(constraints.b)  # (1, max_k)

                    if A_ps.ndim == 3:
                        A_t = A_ps[0]
                        b_t = b_ps[0]
                    else:
                        A_t = np.asarray(A_ps, dtype=np.float32)
                        b_t = np.asarray(b_ps, dtype=np.float32).reshape(-1)
                    valid_mask = np.isfinite(b_t) & (b_t > -1e8)
                    if valid_mask.any():
                        A_t = np.asarray(A_t, dtype=np.float32)
                        if A_t.ndim == 1:
                            A_t = A_t.reshape(1, -1)
                        A_t = A_t[valid_mask]
                        b_t = b_t[valid_mask]
                    else:
                        A_t = np.empty((0, int(u_nom_t.size)), dtype=np.float32)
                        b_t = np.empty(0, dtype=np.float32)

                    u_safe_t = self._solve_per_step_qp_simple(
                        u_nom_t, A_t, b_t, rho, control_limit=control_limit
                    )
                    u_filtered.append(u_safe_t)
                    x = self._step_once(x, u_safe_t, env)

                u_seq = np.stack(u_filtered, axis=0)

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
        dt = float(getattr(env, "dt", 0.05))
        robot_radius = float(getattr(env, "robot_radius", 0.05))
        control_limit = float(getattr(env, "control_limit", 1.0))
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
        
        # Step C: Cache obstacle branches to avoid repeated compilation
        # Optimization 3: Include max_k, dt in cache key to avoid recompilation when these change
        obstacles_cache_key = (
            id(obstacles) if obstacles is not None else None,
            max_k,
            dt,
            constraint_margin_val
        )
        
        # Check if we can reuse cached branches
        if (self._obstacle_branches_cache is not None and 
            self._obstacles_cache_key == obstacles_cache_key and
            len(self._obstacle_branches_cache[0]) >= max_k):
            # Reuse cached branches
            obstacle_branches_sdf_tuple = self._obstacle_branches_cache[0][:max_k]
            obstacle_branches_grad_tuple = self._obstacle_branches_cache[1][:max_k]
            num_obs_fns = min(len(self._obstacle_branches_cache[0]), max_k)
            can_use_multi = num_obs_fns > 0
        else:
            # Build obstacle SDF/gradient functions for multi-constraint
            # Optimization 2: Use jax_gradient if available, otherwise use jax.grad
            # Optimization 1: Separate SDF-only and SDF+grad functions for two-stage computation
            obstacle_branches_sdf = []  # Only compute SDF (fast)
            obstacle_branches_grad = []  # Compute SDF + grad (slower, only for candidates)
            
            if obstacles_list and len(obstacles_list) > 0:
                for obs in obstacles_list[:max_k]:
                    if hasattr(obs, 'jax_sdf'):
                        # Create closure that captures this specific obstacle
                        def make_branch_fns(obstacle=obs):  # Default arg to avoid late binding
                            def sdf_fn(pos):
                                return obstacle.jax_sdf(pos)
                            
                            # Optimization 2: Check if obstacle has jax_gradient method (analytical)
                            if hasattr(obstacle, 'jax_gradient'):
                                # Use analytical gradient (much faster than jax.grad)
                                def sdf_only_fn(pos):
                                    return obstacle.jax_sdf(pos)
                                
                                def sdf_grad_fn(pos):
                                    sdf_val = obstacle.jax_sdf(pos)
                                    grad_val = obstacle.jax_gradient(pos)
                                    return sdf_val, grad_val
                            else:
                                # Fallback: Use jax.grad (slower, but more general)
                                # Pre-compute grad function (outside traced context)
                                grad_fn = jax.grad(sdf_fn)
                                
                                def sdf_only_fn(pos):
                                    return obstacle.jax_sdf(pos)
                                
                                def sdf_grad_fn(pos):
                                    sdf_val = sdf_fn(pos)
                                    grad_val = grad_fn(pos)  # Use pre-computed grad_fn
                                    return sdf_val, grad_val
                            
                            return sdf_only_fn, sdf_grad_fn
                        
                        sdf_fn, grad_fn = make_branch_fns()
                        obstacle_branches_sdf.append(sdf_fn)
                        obstacle_branches_grad.append(grad_fn)
            
            num_obs_fns = len(obstacle_branches_sdf)
            can_use_multi = num_obs_fns > 0
            
            # Pad branches to max_k for fixed size (required for JAX)
            while len(obstacle_branches_sdf) < max_k:
                def dummy_sdf_fn(pos):
                    return jnp.inf
                def dummy_grad_fn(pos):
                    return jnp.inf, jnp.zeros(2)
                obstacle_branches_sdf.append(dummy_sdf_fn)
                obstacle_branches_grad.append(dummy_grad_fn)
            
            # Convert to tuple for JAX switch and cache it
            obstacle_branches_sdf_tuple = tuple(obstacle_branches_sdf[:max_k])
            obstacle_branches_grad_tuple = tuple(obstacle_branches_grad[:max_k])
            self._obstacle_branches_cache = (obstacle_branches_sdf_tuple, obstacle_branches_grad_tuple)
            self._obstacles_cache_key = obstacles_cache_key
        
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
                    # Optimization 1: Two-stage computation - first compute SDF only (fast), then grad for candidates
                    # Stage 1: Compute SDF for all obstacles (fast, no gradient)
                    def compute_obstacle_sdf_only(obs_idx, pos):
                        """Compute SDF only for obstacle at index obs_idx (fast, no gradient)."""
                        obs_idx_clipped = jnp.clip(obs_idx, 0, max_k - 1)
                        sdf_val = jax.lax.switch(obs_idx_clipped, obstacle_branches_sdf_tuple, pos)
                        valid = obs_idx < num_obs_fns
                        return jnp.where(valid, sdf_val, jnp.inf)
                    
                    # Stage 1: Compute SDF for all obstacles
                    obs_indices = jnp.arange(max_k)  # Fixed size for JAX
                    sdf_array = jax.vmap(lambda idx: compute_obstacle_sdf_only(idx, pos_t))(obs_indices)  # (max_k,)
                    
                    # Fix 4: Gate前移 - 如果min_sdf > threshold，直接跳过约束计算
                    min_sdf = jnp.min(sdf_array)
                    needs_constraints = min_sdf < threshold
                    
                    # Use lax.cond to skip constraint computation if not needed
                    def compute_constraints(_):
                        # Stage 2: Select candidates (sdf < threshold)
                        cand_mask = sdf_array < threshold
                        sdf_for_sort = jnp.where(cand_mask, sdf_array, jnp.inf)
                        
                        # Stage 3: Use top_k to select closest obstacles (optimization: avoid full argsort)
                        neg_sdf = -sdf_for_sort
                        topk_neg_values, topk_indices = jax.lax.top_k(neg_sdf, max_k)
                        selected_indices = jnp.flip(topk_indices, axis=0)  # (max_k,) - reverse for ascending order
                        
                        # Count valid candidates
                        num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
                        k = jnp.minimum(max_k, jnp.where(num_candidates == 0, max_k, num_candidates))
                        
                        # Stage 4: Compute gradient ONLY for selected candidates (much faster!)
                        def compute_obstacle_grad_for_candidate(rank_idx, pos):
                            """Compute SDF and gradient for obstacle at rank rank_idx (0=closest, 1=second closest, etc.)."""
                            # Get the original obstacle index from selected_indices
                            obs_idx = selected_indices[rank_idx]
                            obs_idx_clipped = jnp.clip(obs_idx, 0, max_k - 1)
                            sdf_val, grad_val = jax.lax.switch(obs_idx_clipped, obstacle_branches_grad_tuple, pos)
                            # Check if this rank is valid (within k closest)
                            valid_rank = rank_idx < k
                            valid_obs = obs_idx < num_obs_fns
                            valid = jnp.logical_and(valid_rank, valid_obs)
                            sdf_val = jnp.where(valid, sdf_val, jnp.inf)
                            grad_val = jnp.where(valid, grad_val, jnp.zeros(2))
                            return sdf_val, grad_val
                        
                        # Compute gradients only for selected obstacles (k closest, not all max_k)
                        # This is the key optimization: we only compute grad for candidates, not all obstacles
                        grad_results = jax.vmap(lambda rank_idx: compute_obstacle_grad_for_candidate(rank_idx, pos_t))(jnp.arange(max_k))
                        grad_array = grad_results[1]  # (max_k, 2) - gradients for selected obstacles
                        # Use sdf_array from stage 1 (already computed), but re-index using selected_indices
                        # For constraint building, we'll use sdf_array[selected_indices[idx]] when needed
                        
                        # Build constraints for selected obstacles
                        def build_constraint_for_obstacle(idx):
                            """Build CFS constraint for obstacle at rank idx (0=closest, 1=second closest, etc.)."""
                            valid_idx = idx < k
                            # grad_array[idx] already contains the gradient for the obstacle at rank idx
                            # sdf_array needs to be re-indexed using selected_indices
                            obs_idx = selected_indices[idx]
                            sdf_obs = jnp.where(valid_idx, sdf_array[obs_idx], jnp.inf)
                            grad_obs = jnp.where(valid_idx, grad_array[idx], jnp.zeros(2))  # Use idx, not obs_idx, for grad_array
                            
                            grad_norm = jnp.linalg.norm(grad_obs)
                            valid_grad = jnp.logical_and(valid_idx, grad_norm > 1e-8)
                            
                            # Normalize gradient
                            g = jnp.where(valid_grad, grad_obs / (grad_norm + 1e-9), jnp.zeros(2))
                            
                            # CFS constraint: b_state = (clearance - sdf) / ||grad|| + g^T pos_t
                            b_state = jnp.where(
                                valid_grad,
                                (clearance - sdf_obs) / (grad_norm + 1e-9) + jnp.dot(g, pos_t),
                                -jnp.inf  # Fix 1: Use -inf for invalid (not +inf)
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
                        # Fix 1: Use -inf for invalid constraints (not +inf) so that A@u >= b is always satisfied for invalid
                        # This prevents gate logic from being corrupted by inf values
                        A_constraints = A_all  # (max_k, 2) - keep all for fixed size
                        b_constraints = jnp.where(valid_mask, b_all, -jnp.inf)  # (max_k,) - mark invalid with -inf
                        
                        return A_constraints, b_constraints
                    
                    def skip_constraints(_):
                        # Return empty constraints (all -inf)
                        A_empty = jnp.zeros((max_k, 2))
                        b_empty = jnp.full((max_k,), -jnp.inf)
                        return A_empty, b_empty
                    
                    # Conditionally compute constraints based on min_sdf
                    A_constraints, b_constraints = jax.lax.cond(
                        needs_constraints,
                        compute_constraints,
                        skip_constraints,
                        operand=None
                    )
                else:
                    # Fallback: No multi-constraint support (no individual obstacle jax_sdf)
                    # In JIT context, we cannot call Python object methods safely
                    # Return original action (no constraint applied)
                    # NOTE: This is a limitation - for proper JIT support, obstacles must expose jax_sdf
                    A_constraints = jnp.zeros((max_k, 2))
                    b_constraints = jnp.full((max_k,), -jnp.inf)
                
                # Solve multi-constraint QP (JIT-safe: use lax.cond)
                # Fix 1: Correct gate logic - compute violations first, then clean, then max
                # This ensures inf values don't corrupt the gate check
                tol = 1e-7
                # Compute violations for all constraints (invalid ones with -inf will have violation = 0)
                violations = jnp.maximum(0.0, b_constraints - A_constraints @ u)  # (max_k,)
                # Clean violations: filter out any inf/nan (shouldn't happen with -inf, but be safe)
                violations_clean = jnp.where(jnp.isfinite(violations), violations, -jnp.inf)
                max_viol_before_clean = jnp.max(violations_clean)
                
                def solve_qp_wrapper(_):
                    return self._solve_multi_constraint_qp_jax(
                        u, A_constraints, b_constraints, rho, control_limit=control_limit
                    )
                
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
