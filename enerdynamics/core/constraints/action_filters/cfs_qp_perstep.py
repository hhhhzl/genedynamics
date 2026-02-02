"""
CFS-based per-step QP filter for action-space projection.

This filter uses CFS (Convex Feasible Set) to linearize obstacles,
then solves per-step QP to project actions into feasible set.
"""

from __future__ import annotations
from typing import Any, Optional
import numpy as np
import os

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
        # Step D: Cache a JAX-safe spatial grid of obstacle candidates
        self._spatial_grid_cache = None
        self._spatial_grid_cache_key = None
    
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

        # Prefer jaxopt.OSQP backend implementation for consistency with JAX path.
        try:
            from enerdynamics.core.constraints.solvers.jaxopt_osqp_solver import JAXOPTOsqpSolver
            if not hasattr(self, "_jaxopt_solver") or self._jaxopt_solver is None:
                self._jaxopt_solver = JAXOPTOsqpSolver(use_jit=False)

            # Always use solve_slack_qp(...) as requested; emulate hard constraints with huge rho.
            rho_eff = float(rho) if (self.use_slack and rho > 0) else 1e9
            u_star, _viol = self._jaxopt_solver.solve_slack_qp(u_nom, A, b, rho_eff)

            u_star = np.clip(np.asarray(u_star, dtype=np.float32), -L, L)

            # Safety post-pass: a few hard projections to eliminate residual violation.
            for _ in range(5):
                v = np.maximum(0.0, b - A @ u_star)
                if v.max(initial=0.0) < 1e-7:
                    break
                i = int(np.argmax(v))
                A_row = A[i]
                den = float(np.dot(A_row, A_row) + 1e-9)
                lam = float(v[i]) / den
                u_star = np.clip(u_star + lam * A_row, -L, L)
            return u_star
        except Exception:
            # Fallback: original lightweight iterative projection
            violations = np.maximum(0, b - A @ u_nom)  # (m,)
            if violations.max() < 1e-7:
                return np.clip(u_nom, -L, L)
            u_safe = np.clip(u_nom.copy(), -L, L)
            for _ in range(10):
                violations = np.maximum(0, b - A @ u_safe)
                if violations.max() < 1e-7:
                    break
                worst_idx = np.argmax(violations)
                A_row = A[worst_idx]
                den = np.dot(A_row, A_row) + 1e-9
                lam = violations[worst_idx] / (den + 1.0 / max(1e-9, rho))
                u_safe = np.clip(u_safe + lam * A_row, -L, L)
            return u_safe
    
    def _solve_multi_constraint_qp_jax(
        self,
        u_nom: jnp.ndarray,
        A: jnp.ndarray,  # (m, act_dim)
        b: jnp.ndarray,  # (m,)
        rho: jnp.ndarray,
        proj_iters: jnp.ndarray,
        control_limit: float = 1.0,
    ) -> jnp.ndarray:
        """
        Fast JAX per-step "solver": fixed-iteration halfspace projection (POCS).

        This keeps CFS exactly the same (still produces linear halfspaces A u >= b),
        but replaces per-step OSQP with a small, fixed number of closed-form
        halfspace projections + box clipping. This is intentionally similar in
        spirit to the single-constraint closed-form projection used by CBF-QP,
        but extended to multiple constraints by iterating on the most violated
        halfspace.

        Notes:
        - `rho` is kept for API compatibility with previous slack-QP signature,
          but is not used here (hard projection).
        - `proj_iters` is masked against a static MAX_PROJ_ITERS for JIT stability.
        """
        L = float(control_limit)
        MAX_PROJ_ITERS = 16  # keep static for JIT (mask with `proj_iters`)
        tol = jnp.asarray(1e-7, dtype=jnp.float32)
        eps = jnp.asarray(1e-9, dtype=jnp.float32)

        # Clip first (box constraints).
        u0 = jnp.clip(jnp.asarray(u_nom, dtype=jnp.float32), -L, L)
        A = jnp.asarray(A, dtype=jnp.float32)
        b = jnp.asarray(b, dtype=jnp.float32)

        # Treat non-finite b as inactive constraints (common representation: -inf).
        valid = jnp.isfinite(b)

        # Precompute row squared norms once (saves a dot inside each iteration).
        row_den = jnp.sum(A * A, axis=1) + eps  # (m,)

        # JIT-stable iteration count: fixed upper bound with mask.
        try:
            proj_iters_i32 = jnp.asarray(proj_iters, dtype=jnp.int32)
        except Exception:
            proj_iters_i32 = jnp.asarray(0, dtype=jnp.int32)
        proj_iters_i32 = jnp.clip(proj_iters_i32, 0, MAX_PROJ_ITERS)

        def body_fn(i, u):
            def project_once(u_in):
                Au = A @ u_in  # (m,)
                # Keep invalid constraints at -inf so argmax won't select them unless all are inactive.
                viol_raw = jnp.where(valid, b - Au, -jnp.inf)
                viol_pos = jnp.maximum(0.0, viol_raw)
                viol = jnp.where(valid, viol_pos, -jnp.inf)

                v_max = jnp.max(viol)
                idx = jnp.argmax(viol)
                a = A[idx]  # (act_dim,)
                den = row_den[idx]
                lam = jnp.where(v_max > tol, viol[idx] / den, 0.0)
                return jnp.clip(u_in + lam * a, -L, L)

            return jax.lax.cond(i < proj_iters_i32, project_once, lambda uu: uu, u)

        u_fast = jax.lax.fori_loop(0, MAX_PROJ_ITERS, body_fn, u0)

        # Safety fallback: if still violating after the requested fast iterations,
        # run a few extra hard projections. This is only triggered on hard corner cases
        # and keeps the common case fast.
        def max_violation(u_in):
            Au = A @ u_in
            viol = jnp.maximum(0.0, jnp.where(valid, b - Au, 0.0))
            return jnp.max(viol)

        v_fast = max_violation(u_fast)

        EXTRA_ITERS = 16

        def extra_project(u_in):
            def extra_body(_i, uu):
                Au = A @ uu
                viol_raw = jnp.where(valid, b - Au, -jnp.inf)
                viol_pos = jnp.maximum(0.0, viol_raw)
                viol = jnp.where(valid, viol_pos, -jnp.inf)

                idx = jnp.argmax(viol)
                a = A[idx]
                den = row_den[idx]
                lam = jnp.where(jnp.max(viol) > tol, viol[idx] / den, 0.0)
                return jnp.clip(uu + lam * a, -L, L)

            return jax.lax.fori_loop(0, EXTRA_ITERS, extra_body, u_in)

        return jax.lax.cond(v_fast <= tol, lambda _: u_fast, lambda _: extra_project(u_fast), operand=None)
    
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

        # Optional runtime assertion to confirm we're not silently taking NumPy fallback.
        # Enable via: ENERDYNAMICS_ASSERT_JAX_FILTER=1
        if os.getenv("ENERDYNAMICS_ASSERT_JAX_FILTER", "").strip() in ("1", "true", "True"):
            if not is_jax:
                raise RuntimeError(
                    "CFSQPPerStepFilter.apply_actions took NumPy fallback, but JAX was expected. "
                    "Check caller is passing jax.Array/Tracer actions and x0."
                )
        
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
        # Unify: treat I_QP as CFS outer iterations (paper-style "iteration").
        # Backward compatible: still accept cfs_outer_iters if provided.
        cfs_outer_iters = int(params_dict.get("I_QP", params_dict.get("cfs_outer_iters", 1)))
        cfs_outer_iters = max(1, cfs_outer_iters)

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
        # Unify: treat I_QP as CFS outer iterations (paper-style "iteration").
        # Backward compatible: still accept cfs_outer_iters if provided.
        I_QP_raw = params_dict.get("I_QP", params_dict.get("cfs_outer_iters", 1))
        # JAX-safe int conversion (supports Python int or traced scalar).
        try:
            I_QP = jnp.asarray(I_QP_raw, dtype=jnp.int32)
        except Exception:
            I_QP = jnp.asarray(1, dtype=jnp.int32)
        # Use a constant outer-iteration count at every diffusion step (paper-style CFS iteration).
        MAX_OUTER_ITERS = 64
        I_QP = jnp.clip(I_QP, 1, MAX_OUTER_ITERS)

        # Fast projection solver iterations (POCS). Keep a static upper bound for JIT.
        proj_iters_raw = params_dict.get("projection_iters", params_dict.get("proj_iters", 5))
        MAX_PROJ_ITERS = 16
        try:
            proj_iters = jnp.asarray(proj_iters_raw, dtype=jnp.int32)
        except Exception:
            proj_iters = jnp.asarray(5, dtype=jnp.int32)
        proj_iters = jnp.clip(proj_iters, 0, MAX_PROJ_ITERS)
        
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
        # IMPORTANT: evaluate SDF against *all* obstacles, then select top-K closest per step.
        # If we only build branches for the first K obstacles, we can miss collisions.
        max_k = int(num_obstacles) if num_obstacles > 0 else 0  # number of obstacle branches
        k_select = int(min(self.max_constraints_per_point, max_k)) if max_k > 0 else 0  # top-K per step
        
        # Compute threshold for obstacle selection (JIT-safe: use JAX operations)
        constraint_margin_jax = jnp.asarray(constraint_margin_val, dtype=jnp.float32)
        threshold = clearance + constraint_margin_jax
        
        # Step C: Cache obstacle branches to avoid repeated compilation
        # Optimization 3: Include max_k, dt in cache key to avoid recompilation when these change
        obstacles_cache_key = (
            id(obstacles) if obstacles is not None else None,
            max_k,
            k_select,
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
                # Build branches for ALL obstacles (max_k = num_obstacles).
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
                                
                                # Grad-only: stage-2 already has sdf from stage-1.
                                def grad_only_fn(pos):
                                    return obstacle.jax_gradient(pos)
                            else:
                                # Fallback: Use jax.grad (slower, but more general)
                                # Pre-compute grad function (outside traced context)
                                grad_fn = jax.grad(sdf_fn)
                                
                                def sdf_only_fn(pos):
                                    return obstacle.jax_sdf(pos)
                                
                                # Grad-only: avoid extra sdf_fn(pos) primal evaluation.
                                def grad_only_fn(pos):
                                    return grad_fn(pos)  # Use pre-computed grad_fn
                            
                            return sdf_only_fn, grad_only_fn
                        
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
                    return jnp.zeros(2)
                obstacle_branches_sdf.append(dummy_sdf_fn)
                obstacle_branches_grad.append(dummy_grad_fn)
            
            # Convert to tuple for JAX switch and cache it
            obstacle_branches_sdf_tuple = tuple(obstacle_branches_sdf[:max_k])
            obstacle_branches_grad_tuple = tuple(obstacle_branches_grad[:max_k])
            self._obstacle_branches_cache = (obstacle_branches_sdf_tuple, obstacle_branches_grad_tuple)
            self._obstacles_cache_key = obstacles_cache_key

        # ------------------------------------------------------------------
        # Step D: Spatial grid candidate cache (B) for per-step filter.
        # Same idea as full-QP: avoid scanning all obstacles for stage-1 SDF.
        # ------------------------------------------------------------------
        use_spatial_grid = False
        cell_obs_idx_np = None
        grid_x_min = grid_y_min = cell_size = None
        grid_W = grid_H = cell_max = 0
        try:
            # Heuristic: for small obstacle counts, a spatial grid often costs more than it saves.
            # Enable only when obstacle count is large enough to benefit.
            if max_k < 64:
                raise RuntimeError("Spatial grid disabled for small obstacle count")

            p_max = float(getattr(env, "p_max", 2.0))
            grid_x_min = -p_max
            grid_y_min = -p_max
            grid_x_max = p_max
            grid_y_max = p_max

            # Conservative but not overly loose: most configs use small margins (<= 0.05).
            threshold_upper = float(robot_radius + constraint_margin_val + 0.1)
            threshold_upper = max(0.0, threshold_upper)
            cell_size = float(max(0.25, min(0.75, threshold_upper)))

            spatial_key = (
                id(obstacles) if obstacles is not None else None,
                max_k,
                float(grid_x_min),
                float(grid_y_min),
                float(grid_x_max),
                float(grid_y_max),
                float(cell_size),
                float(threshold_upper),
            )

            if self._spatial_grid_cache is not None and self._spatial_grid_cache_key == spatial_key:
                cell_obs_idx_np, grid_x_min, grid_y_min, cell_size, grid_W, grid_H, cell_max = self._spatial_grid_cache
                use_spatial_grid = True
            else:
                centers = []
                radii = []
                for obs in (obstacles_list or [])[:max_k]:
                    c = None
                    r = None
                    if hasattr(obs, "center"):
                        try:
                            c = np.asarray(getattr(obs, "center"), dtype=np.float32).reshape(-1)[:2]
                        except Exception:
                            c = None
                    if c is None and hasattr(obs, "bounds"):
                        try:
                            b = getattr(obs, "bounds")
                            if b is not None and len(b) == 2:
                                b0 = np.asarray(b[0], dtype=np.float32).reshape(-1)[:2]
                                b1 = np.asarray(b[1], dtype=np.float32).reshape(-1)[:2]
                                c = 0.5 * (b0 + b1)
                        except Exception:
                            c = None
                    if hasattr(obs, "radius"):
                        try:
                            r = float(getattr(obs, "radius"))
                        except Exception:
                            r = None
                    if r is None and hasattr(obs, "half_extents"):
                        try:
                            he = np.asarray(getattr(obs, "half_extents"), dtype=np.float32).reshape(-1)[:2]
                            r = float(np.linalg.norm(he))
                        except Exception:
                            r = None
                    if r is None and hasattr(obs, "bounds"):
                        try:
                            b = getattr(obs, "bounds")
                            if b is not None and len(b) == 2:
                                b0 = np.asarray(b[0], dtype=np.float32).reshape(-1)[:2]
                                b1 = np.asarray(b[1], dtype=np.float32).reshape(-1)[:2]
                                r = float(0.5 * np.linalg.norm(b1 - b0))
                        except Exception:
                            r = None
                    if c is None or r is None:
                        centers = []
                        radii = []
                        break
                    centers.append(c)
                    radii.append(r)

                if len(centers) == max_k and max_k > 0:
                    centers = np.asarray(centers, dtype=np.float32)
                    radii = np.asarray(radii, dtype=np.float32)

                    grid_W = int(np.ceil((grid_x_max - grid_x_min) / cell_size))
                    grid_H = int(np.ceil((grid_y_max - grid_y_min) / cell_size))
                    grid_W = max(1, grid_W)
                    grid_H = max(1, grid_H)

                    xs0 = grid_x_min + np.arange(grid_W, dtype=np.float32) * cell_size
                    ys0 = grid_y_min + np.arange(grid_H, dtype=np.float32) * cell_size
                    xs1 = xs0 + cell_size
                    ys1 = ys0 + cell_size

                    cell_lists = [[[] for _ in range(grid_W)] for _ in range(grid_H)]
                    max_count = 0
                    for iy in range(grid_H):
                        y0 = ys0[iy]
                        y1 = ys1[iy]
                        for ix in range(grid_W):
                            x0 = xs0[ix]
                            x1 = xs1[ix]
                            dx0 = np.maximum(x0 - centers[:, 0], 0.0)
                            dx1 = np.maximum(centers[:, 0] - x1, 0.0)
                            dy0 = np.maximum(y0 - centers[:, 1], 0.0)
                            dy1 = np.maximum(centers[:, 1] - y1, 0.0)
                            dx = np.maximum(dx0, dx1)
                            dy = np.maximum(dy0, dy1)
                            dist = np.sqrt(dx * dx + dy * dy)
                            include = dist <= (radii + threshold_upper)
                            idxs = np.nonzero(include)[0].tolist()
                            cell_lists[iy][ix] = idxs
                            if len(idxs) > max_count:
                                max_count = len(idxs)

                    # Ensure candidate list length >= k_select so top_k is always valid.
                    cell_max = int(max(k_select, min(max_k, max_count)))
                    if cell_max >= max_k:
                        raise RuntimeError("Spatial grid provides no pruning (cell_max >= max_k)")
                    cell_obs_idx_np = -np.ones((grid_H, grid_W, cell_max), dtype=np.int32)
                    for iy in range(grid_H):
                        for ix in range(grid_W):
                            lst = cell_lists[iy][ix]
                            if not lst:
                                continue
                            lst = lst[:cell_max]
                            cell_obs_idx_np[iy, ix, : len(lst)] = np.asarray(lst, dtype=np.int32)

                    self._spatial_grid_cache = (
                        cell_obs_idx_np,
                        float(grid_x_min),
                        float(grid_y_min),
                        float(cell_size),
                        int(grid_W),
                        int(grid_H),
                        int(cell_max),
                    )
                    self._spatial_grid_cache_key = spatial_key
                    use_spatial_grid = True
        except Exception:
            use_spatial_grid = False

        if use_spatial_grid and cell_obs_idx_np is not None:
            cell_obs_idx = jnp.asarray(cell_obs_idx_np, dtype=jnp.int32)
        else:
            cell_obs_idx = None
        
        def filter_single_once(u_seq):
            """One CFS pass: rollout + linearize + per-step QP."""
            def body_fn(carry, u):
                x = carry  # Current state (position for single_2d)
                pos_t = x[0:2]  # Current position
                
                # Multi-constraint CFS: compute SDF and gradient for each obstacle
                if num_obstacles == 0 or max_k == 0 or k_select == 0:
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
                        valid = jnp.logical_and(obs_idx >= 0, obs_idx < num_obs_fns)
                        obs_idx_clipped = jnp.clip(obs_idx, 0, max_k - 1)
                        sdf_val = jax.lax.switch(obs_idx_clipped, obstacle_branches_sdf_tuple, pos)
                        return jnp.where(valid, sdf_val, jnp.inf)
                    
                    # Stage 1: Compute SDF for all obstacles
                    if cell_obs_idx is not None:
                        ix = jnp.floor((pos_t[0] - jnp.asarray(grid_x_min, dtype=jnp.float32)) / jnp.asarray(cell_size, dtype=jnp.float32)).astype(jnp.int32)
                        iy = jnp.floor((pos_t[1] - jnp.asarray(grid_y_min, dtype=jnp.float32)) / jnp.asarray(cell_size, dtype=jnp.float32)).astype(jnp.int32)
                        ix = jnp.clip(ix, 0, jnp.asarray(grid_W - 1, dtype=jnp.int32))
                        iy = jnp.clip(iy, 0, jnp.asarray(grid_H - 1, dtype=jnp.int32))
                        cand_idx = cell_obs_idx[iy, ix]  # (M,)
                    else:
                        cand_idx = jnp.arange(max_k, dtype=jnp.int32)
                    sdf_array = jax.vmap(lambda idx: compute_obstacle_sdf_only(idx, pos_t))(cand_idx)  # (M,)
                    
                    # Fix 4: Gate前移 - 如果min_sdf > threshold，直接跳过约束计算
                    min_sdf = jnp.min(sdf_array)
                    needs_constraints = min_sdf < threshold
                    
                    # Use lax.cond to skip constraint computation if not needed
                    def compute_constraints(_):
                        # Stage 2: Select candidates (sdf < threshold)
                        cand_mask = sdf_array < threshold
                        sdf_for_sort = jnp.where(cand_mask, sdf_array, jnp.inf)
                        
                        # Stage 3: Use top_k to select closest obstacles (optimization: avoid full argsort)
                        # Select closest obstacles (smallest sdf). Using neg_sdf makes larger = closer.
                        # jax.lax.top_k returns indices sorted by descending score, so this is already
                        # closest-first. Do NOT flip, otherwise we'd pick farthest-first.
                        neg_sdf = -sdf_for_sort
                        _, selected_rank = jax.lax.top_k(neg_sdf, k_select)
                        
                        # Count valid candidates
                        num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
                        k = jnp.minimum(
                            jnp.asarray(k_select, dtype=jnp.int32),
                            jnp.where(num_candidates == 0, k_select, num_candidates),
                        )
                        selected_indices = jnp.take(cand_idx, selected_rank, axis=0)
                        sdf_sel = jnp.take(sdf_array, selected_rank, axis=0)
                        
                        # Stage 4: Compute gradient ONLY for selected candidates (much faster!)
                        def compute_obstacle_grad_for_candidate(rank_idx, pos):
                            """Compute SDF and gradient for obstacle at rank rank_idx (0=closest, 1=second closest, etc.)."""
                            # Get the original obstacle index from selected_indices
                            obs_idx = selected_indices[rank_idx]
                            obs_idx_clipped = jnp.clip(obs_idx, 0, max_k - 1)
                            grad_val = jax.lax.switch(obs_idx_clipped, obstacle_branches_grad_tuple, pos)
                            # Check if this rank is valid (within k closest)
                            valid_rank = rank_idx < k
                            valid_obs = jnp.logical_and(obs_idx >= 0, obs_idx < num_obs_fns)
                            valid = jnp.logical_and(valid_rank, valid_obs)
                            grad_val = jnp.where(valid, grad_val, jnp.zeros(2))
                            return grad_val
                        
                        # Compute gradients only for selected obstacles (k closest, not all max_k)
                        # This is the key optimization: we only compute grad for candidates, not all obstacles
                        grad_array = jax.vmap(lambda rank_idx: compute_obstacle_grad_for_candidate(rank_idx, pos_t))(jnp.arange(k_select))
                        # Use sdf_array from stage 1 (already computed), but re-index using selected_indices
                        # For constraint building, we'll use sdf_array[selected_indices[idx]] when needed
                        
                        # Build constraints for selected obstacles
                        def build_constraint_for_obstacle(idx):
                            """Build CFS constraint for obstacle at rank idx (0=closest, 1=second closest, etc.)."""
                            valid_idx = idx < k
                            # grad_array[idx] already contains the gradient for the obstacle at rank idx
                            # sdf_array needs to be re-indexed using selected_indices
                            sdf_obs = jnp.where(valid_idx, sdf_sel[idx], jnp.inf)
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
                        constraint_results = jax.vmap(build_constraint_for_obstacle)(jnp.arange(k_select))
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
                        A_constraints = A_all  # (k_select, 2) - keep all for fixed size
                        b_constraints = jnp.where(valid_mask, b_all, -jnp.inf)  # (k_select,) - mark invalid with -inf
                        
                        return A_constraints, b_constraints
                    
                    def skip_constraints(_):
                        # Return empty constraints (all -inf)
                        A_empty = jnp.zeros((k_select, 2))
                        b_empty = jnp.full((k_select,), -jnp.inf)
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
                        u, A_constraints, b_constraints, rho, proj_iters, control_limit=control_limit
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

        def filter_single(u_seq):
            """
            Paper-style CFS iteration: repeat (linearize + solve QP) I_QP times.
            We use a fixed upper bound and mask extra iterations for JIT safety.
            """
            def outer_body(i, u_curr):
                return jax.lax.cond(
                    i < I_QP,
                    lambda uu: filter_single_once(uu),
                    lambda uu: uu,
                    u_curr,
                )
            return jax.lax.fori_loop(0, MAX_OUTER_ITERS, outer_body, u_seq)
        
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
