"""
CFS-based full trajectory QP filter for action-space projection.

This filter uses CFS (Convex Feasible Set) to linearize obstacles,
then solves a single QP over the entire action sequence to project actions into feasible set.
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
        # Step D: Cache a JAX-safe spatial grid of obstacle candidates
        # (cell -> fixed-size list of obstacle indices). This accelerates stage-1
        # SDF scanning by avoiding "all obstacles per query point".
        self._spatial_grid_cache = None
        self._spatial_grid_cache_key = None
    
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

            # Fallback: many obstacle containers (e.g., ObstacleManager) are iterable but don't expose
            # a stable `.obstacles` attribute. If the above logic produced nothing, try iteration.
            if len(obstacles_list) == 0:
                try:
                    obstacles_list = list(obstacles)
                except Exception:
                    obstacles_list = []
            
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

        # Prefer jaxopt.OSQP backend implementation for consistency with JAX path.
        try:
            from enerdynamics.core.constraints.solvers.jaxopt_osqp_solver import JAXOPTOsqpSolver
            if not hasattr(self, "_jaxopt_solver") or self._jaxopt_solver is None:
                self._jaxopt_solver = JAXOPTOsqpSolver(use_jit=False)

            # Always use solve_slack_qp(...) as requested; emulate hard constraints with huge rho.
            rho_eff = float(rho) if (self.use_slack and rho > 0) else 1e9
            u_star, _viol = self._jaxopt_solver.solve_slack_qp(u_nom_flat, A_valid, b_valid, rho_eff)

            u_star = np.clip(np.asarray(u_star, dtype=np.float32), -L, L)

            # Safety post-pass: a few hard projections to eliminate residual violation.
            for _ in range(8):
                v = np.maximum(0.0, b_valid - A_valid @ u_star)
                if v.max(initial=0.0) < 1e-7:
                    break
                i = int(np.argmax(v))
                A_row = A_valid[i]
                den = float(np.dot(A_row, A_row) + 1e-9)
                lam = float(v[i]) / den
                u_star = np.clip(u_star + lam * A_row, -L, L)
            return u_star
        except Exception:
            # Fallback: original iterative projection
            u_star = np.clip(u_nom_flat.copy(), -L, L)
            for _ in range(20):
                violations = np.maximum(0.0, b_valid - A_valid @ u_star)
                if violations.max(initial=0.0) < 1e-7:
                    break
                worst_idx = int(np.argmax(violations))
                A_row = A_valid[worst_idx]
                den = np.dot(A_row, A_row) + 1e-9
                if not self.use_slack or rho <= 0:
                    lam = violations[worst_idx] / den
                else:
                    lam = violations[worst_idx] / (den + 1.0 / max(1e-9, rho))
                u_star = np.clip(u_star + lam * A_row, -L, L)
            return u_star
    
    def _solve_full_trajectory_qp_jax(
        self,
        u_nom_flat: jnp.ndarray,  # (H*act_dim,)
        A_full: jnp.ndarray,  # (H*max_k, H*act_dim)
        b_full: jnp.ndarray,  # (H*max_k,)
        rho: jnp.ndarray,
        max_iter: jnp.ndarray | int = 20,
        control_limit: float = 1.0,
    ) -> jnp.ndarray:
        """
        Solve full trajectory QP in JAX via jaxopt.OSQP wrappers.

        Uses slack-QP when self.use_slack and rho > 0, otherwise hard-QP.
        Includes box bounds in the QP and does a small hard-feasibility post-pass
        to guarantee safety (eliminate residual solver tolerance violations).
        """
        L = float(control_limit)
        from enerdynamics.core.constraints.solvers.jaxopt_osqp_solver import solve_slack_qp_jax

        # Always use solve_slack_qp_jax(...); emulate hard constraints with huge rho.
        rho_eff = jax.lax.cond(
            jnp.logical_and(jnp.asarray(self.use_slack, dtype=jnp.bool_), rho > 0),
            lambda __: jnp.asarray(rho, dtype=jnp.float32),
            lambda __: jnp.asarray(1e9, dtype=jnp.float32),
            operand=None,
        )
        u0, v0 = solve_slack_qp_jax(u_nom_flat, A_full, b_full, rho_eff, control_limit=L)

        # Safety post-pass: eliminate any residual violation after solver tolerances.
        b_masked = b_full
        valid = jnp.isfinite(b_masked)

        tol = 1e-7

        def max_violation(u):
            viol = jnp.maximum(0.0, jnp.where(valid, b_masked - A_full @ u, -jnp.inf))
            return jnp.max(viol)

        def cond_fn(carry):
            i, _u, v = carry
            return jnp.logical_and(i < 8, v > tol)

        def body_fn(carry):
            i, u, _v = carry
            viol = jnp.maximum(0.0, jnp.where(valid, b_masked - A_full @ u, -jnp.inf))
            idx = jnp.argmax(viol)
            A_row = A_full[idx]
            den = jnp.dot(A_row, A_row) + 1e-9
            lam = viol[idx] / den
            u_next = jnp.clip(u + lam * A_row, -L, L)
            v_next = max_violation(u_next)
            return (i + 1, u_next, v_next)

        def do_post(_):
            _, u_out, _ = jax.lax.while_loop(
                cond_fn,
                body_fn,
                (jnp.asarray(0, dtype=jnp.int32), u0, v0),
            )
            return u_out

        return jax.lax.cond(v0 <= tol, lambda _: u0, do_post, operand=None)
    
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
                print("CFSQPFullFilter JAX path failed, returning unfiltered actions:", str(e)[:200])
                if 'tracer' in str(e).lower() or 'jax' in str(e).lower():
                    return actions
                raise

        # Optional runtime assertion to confirm we're not silently taking NumPy fallback.
        # Enable via: ENERDYNAMICS_ASSERT_JAX_FILTER=1
        if os.getenv("ENERDYNAMICS_ASSERT_JAX_FILTER", "").strip() in ("1", "true", "True"):
            raise RuntimeError(
                "CFSQPFullFilter.apply_actions took NumPy fallback, but JAX was expected. "
                "Check caller is passing jax.Array/Tracer actions."
            )
        
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
        # Unify: treat I_QP as CFS outer iterations (paper-style "iteration").
        # Backward compatible: still accept cfs_outer_iters if provided.
        cfs_outer_iters = int(params_dict.get("I_QP", params_dict.get("cfs_outer_iters", 1)))
        cfs_outer_iters = max(1, cfs_outer_iters)
        
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
        # Unify: treat I_QP as CFS outer iterations (paper-style "iteration"):
        # each outer iteration re-linearizes constraints and solves a (convex) QP.
        # Backward compatible: still accept cfs_outer_iters if provided.
        I_QP_raw = params_dict.get("I_QP", params_dict.get("cfs_outer_iters", 1))
        try:
            I_QP = jnp.asarray(I_QP_raw, dtype=jnp.int32)
        except Exception:
            I_QP = jnp.asarray(1, dtype=jnp.int32)
        # Use a constant outer-iteration count at every diffusion step (paper-style CFS iteration).
        # We keep a hard upper bound only as a safety guard against runaway.
        MAX_OUTER_ITERS = 64
        I_QP = jnp.clip(I_QP, 1, MAX_OUTER_ITERS)
        
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
        
        # JAX-friendly qp_gate + qp_prob: use jax.random.uniform for probabilistic QP skip.
        # Get rng_key from schedule_params or kwargs; fallback to deterministic key if not provided.
        rng_key = params_dict.get("rng_key", kwargs.get("rng_key", None))
        if rng_key is None:
            rng_key = jax.random.PRNGKey(42)  # deterministic fallback (qp_prob should be 1.0 if not using random)
        
        qp_gate_pred = jnp.asarray(qp_gate, dtype=jnp.bool_)
        qp_prob_jax = jnp.asarray(qp_prob, dtype=jnp.float32)
        rand_val = jax.random.uniform(rng_key)
        prob_pass = rand_val < qp_prob_jax
        should_do_qp = jnp.logical_and(qp_gate_pred, prob_pass)
        
        operand = (x0, actions)
        def skip_qp(operand):
            return operand[1]  # actions
        def do_qp(operand):
            x0, actions = operand
            # Get environment parameters
            dt = float(getattr(env, "dt", 0.05))
            robot_radius = float(getattr(env, "robot_radius", 0.05))
            control_limit = float(getattr(env, "control_limit", 1.0))
            constraint_margin_val = float(self.constraint_margin)
            
            # NOTE: `margin` / `rho` / `I_QP` may be per-sample vectors when called from
            # batched multirun paths (e.g. adaptive scheduler). We therefore compute
            # (clearance, threshold) per-trajectory inside the vmapped filter, instead
            # of closing over them as a single array here.
            margin_jax = margin if isinstance(margin, (jnp.ndarray, jax.Array)) else jnp.asarray(float(margin), dtype=jnp.float32)
            robot_radius_jax = jnp.asarray(robot_radius, dtype=jnp.float32)
            constraint_margin_jax = jnp.asarray(constraint_margin_val, dtype=jnp.float32)
            
            # Get obstacles list
            obstacles_list = self._get_obstacles_list(obstacles)
            num_obstacles = len(obstacles_list) if obstacles_list else 0
            # IMPORTANT: evaluate SDF against *all* obstacles, then select top-K closest per timestep.
            # If we only build branches for the first K obstacles, we can miss collisions.
            max_k = int(num_obstacles) if num_obstacles > 0 else 0  # number of obstacle branches
            k_select = int(min(self.max_constraints_per_point, max_k)) if max_k > 0 else 0  # top-K per timestep
        
            # Fast path (extreme performance): vectorized analytic SDF+grad for convex primitives.
            # This avoids large lax.switch branch trees over Python objects.
            use_fast_analytic = False
            circle_centers_np = circle_radii_np = box_centers_np = box_half_np = None
            try:
                from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle
                if obstacles_list is not None and len(obstacles_list) > 0:
                    circles = [o for o in obstacles_list if isinstance(o, SphereObstacle)]
                    boxes = [o for o in obstacles_list if isinstance(o, BoxObstacle)]
                    others = [o for o in obstacles_list if not isinstance(o, (SphereObstacle, BoxObstacle))]
                    if len(others) == 0:
                        use_fast_analytic = True
                        circle_centers_np = np.asarray([np.asarray(o.center, dtype=np.float32)[:2] for o in circles], dtype=np.float32) if circles else np.zeros((0, 2), dtype=np.float32)
                        circle_radii_np = np.asarray([float(o.radius) for o in circles], dtype=np.float32) if circles else np.zeros((0,), dtype=np.float32)
                        box_centers_np = np.asarray([np.asarray(o.center, dtype=np.float32)[:2] for o in boxes], dtype=np.float32) if boxes else np.zeros((0, 2), dtype=np.float32)
                        box_half_np = np.asarray([np.asarray(o.half_extents, dtype=np.float32)[:2] for o in boxes], dtype=np.float32) if boxes else np.zeros((0, 2), dtype=np.float32)
            except Exception:
                use_fast_analytic = False

            if use_fast_analytic:
                circle_centers = jnp.asarray(circle_centers_np, dtype=jnp.float32)
                circle_radii = jnp.asarray(circle_radii_np, dtype=jnp.float32)
                box_centers = jnp.asarray(box_centers_np, dtype=jnp.float32)
                box_half = jnp.asarray(box_half_np, dtype=jnp.float32)

            # Cache obstacle branches (fallback, slower)
            obstacles_cache_key = (
                id(obstacles) if obstacles is not None else None,
                max_k,
                k_select,
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
                    # Build branches for ALL obstacles (max_k = num_obstacles).
                    for obs in obstacles_list[:max_k]:
                        if hasattr(obs, 'jax_sdf'):
                            def make_branch_fns(obstacle=obs):
                                def sdf_fn(pos):
                                    return obstacle.jax_sdf(pos)
                            
                                if hasattr(obstacle, 'jax_gradient'):
                                    def sdf_only_fn(pos):
                                        return obstacle.jax_sdf(pos)
                                
                                    # Grad-only: stage-2 already has sdf from stage-1.
                                    # Avoid recomputing sdf here.
                                    def grad_only_fn(pos):
                                        return obstacle.jax_gradient(pos)
                                else:
                                    grad_fn = jax.grad(sdf_fn)
                                
                                    def sdf_only_fn(pos):
                                        return obstacle.jax_sdf(pos)
                                
                                    # Grad-only: avoid an extra primal sdf_fn(pos).
                                    # Note: grad_fn(pos) still evaluates the primal once internally (needed for AD),
                                    # but we avoid the *additional* sdf_fn(pos) call.
                                    def grad_only_fn(pos):
                                        return grad_fn(pos)
                            
                                return sdf_only_fn, grad_only_fn
                        
                            sdf_fn, grad_fn = make_branch_fns()
                            obstacle_branches_sdf.append(sdf_fn)
                            obstacle_branches_grad.append(grad_fn)
            
                num_obs_fns = len(obstacle_branches_sdf)
                can_use_multi = num_obs_fns > 0
            
                while len(obstacle_branches_sdf) < max_k:
                    def dummy_sdf_fn(pos):
                        return jnp.inf
                    def dummy_grad_fn(pos):
                        return jnp.zeros(2)
                    obstacle_branches_sdf.append(dummy_sdf_fn)
                    obstacle_branches_grad.append(dummy_grad_fn)
            
                obstacle_branches_sdf_tuple = tuple(obstacle_branches_sdf[:max_k])
                obstacle_branches_grad_tuple = tuple(obstacle_branches_grad[:max_k])
                self._obstacle_branches_cache = (obstacle_branches_sdf_tuple, obstacle_branches_grad_tuple)
                self._obstacles_cache_key = obstacles_cache_key
            
            use_spatial_grid = False
            cell_obs_idx_np = None
            grid_x_min = grid_y_min = cell_size = None
            grid_W = grid_H = cell_max = 0
            try:
                # Heuristic: for small obstacle counts, a spatial grid often costs more than it saves.
                if max_k < 128:
                    raise RuntimeError("Spatial grid disabled for small obstacle count")

                # Only attempt if obstacles have centers and sizes.
                # This covers the common single2d case (Box/Sphere).
                p_max = float(getattr(env, "p_max", 2.0))
                grid_x_min = -p_max
                grid_y_min = -p_max
                grid_x_max = p_max
                grid_y_max = p_max

                # Conservative upper bound for threshold = robot_radius + margin + constraint_margin.
                # We do NOT have access to margin as a python float under JIT (it's a tracer),
                # so we pick a safe bound that still enables pruning in practice.
                # Conservative but not overly loose: most configs use small margins (<= 0.05).
                # If you increase margin above this, raise this buffer accordingly.
                threshold_upper = float(robot_radius + constraint_margin_val + 0.1)
                threshold_upper = max(0.0, threshold_upper)

                # Pick a moderate cell size to bound per-cell candidate counts.
                # Smaller => fewer candidates per cell, more cells. Keep it stable.
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
                    # Build (center, R) arrays for obstacles.
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
                            # Cannot safely index: fall back to no spatial grid.
                            centers = []
                            radii = []
                            break
                        centers.append(c)
                        radii.append(r)

                    if len(centers) == max_k and max_k > 0:
                        centers = np.asarray(centers, dtype=np.float32)  # (N,2)
                        radii = np.asarray(radii, dtype=np.float32)      # (N,)

                        grid_W = int(np.ceil((grid_x_max - grid_x_min) / cell_size))
                        grid_H = int(np.ceil((grid_y_max - grid_y_min) / cell_size))
                        grid_W = max(1, grid_W)
                        grid_H = max(1, grid_H)

                        # Precompute cell AABB bounds.
                        xs0 = grid_x_min + np.arange(grid_W, dtype=np.float32) * cell_size
                        ys0 = grid_y_min + np.arange(grid_H, dtype=np.float32) * cell_size
                        xs1 = xs0 + cell_size
                        ys1 = ys0 + cell_size

                        # Build candidate lists.
                        cell_lists = [[[] for _ in range(grid_W)] for _ in range(grid_H)]
                        max_count = 0

                        for iy in range(grid_H):
                            y0 = ys0[iy]
                            y1 = ys1[iy]
                            for ix in range(grid_W):
                                x0 = xs0[ix]
                                x1 = xs1[ix]
                                # Distance from obstacle center to cell AABB (0 if inside)
                                dx0 = np.maximum(x0 - centers[:, 0], 0.0)
                                dx1 = np.maximum(centers[:, 0] - x1, 0.0)
                                dy0 = np.maximum(y0 - centers[:, 1], 0.0)
                                dy1 = np.maximum(centers[:, 1] - y1, 0.0)
                                dx = np.maximum(dx0, dx1)
                                dy = np.maximum(dy0, dy1)
                                dist = np.sqrt(dx * dx + dy * dy)
                                # If a point in this cell could be within threshold of obstacle boundary,
                                # include obstacle. (Bounding circle conservative)
                                include = dist <= (radii + threshold_upper)
                                idxs = np.nonzero(include)[0].tolist()
                                cell_lists[iy][ix] = idxs
                                if len(idxs) > max_count:
                                    max_count = len(idxs)

                        # Ensure candidate list length >= k_select so top_k is always valid.
                        cell_max = int(max(k_select, min(max_k, max_count)))
                        # If we can't prune (cell contains almost all obstacles), disable to avoid overhead.
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
                cell_obs_idx = jnp.asarray(cell_obs_idx_np, dtype=jnp.int32)  # (Hc,Wc,M)
            else:
                cell_obs_idx = None
        
            def filter_single_once(u_seq, *, clearance_s, threshold_s, rho_s, I_QP_s):
                """One CFS pass: rollout + linearize + full-trajectory QP."""
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
                if num_obstacles == 0 or max_k == 0 or k_select == 0:
                    # No obstacles - return original
                    return u_seq

                # ------------------------ FAST ANALYTIC PATH ------------------------ #
                if use_fast_analytic:
                    # Positions to constrain: p_{t+1}, t=0..H-1
                    p0 = x0[0:2]  # (2,)
                    pos = states_rollout[1:, 0:2]  # (H, 2)
                    eps = jnp.asarray(1e-8, dtype=jnp.float32)

                    # Circles: sdf = ||p-c|| - r ; grad = (p-c)/||p-c||
                    def circle_sdf_grad(p):
                        # p: (H,2)
                        if circle_centers.shape[0] == 0:
                            sdf = jnp.full((0, H), jnp.inf, dtype=jnp.float32)
                            grad = jnp.zeros((0, H, 2), dtype=jnp.float32)
                            return sdf, grad
                        diff = p[None, :, :] - circle_centers[:, None, :]  # (Nc,H,2)
                        dist = jnp.linalg.norm(diff, axis=-1)  # (Nc,H)
                        sdf = dist - circle_radii[:, None]
                        grad = diff / (dist[:, :, None] + eps)
                        return sdf, grad

                    # Boxes (axis-aligned): standard SDF + piecewise grad
                    def box_sdf_grad(p):
                        if box_centers.shape[0] == 0:
                            sdf = jnp.full((0, H), jnp.inf, dtype=jnp.float32)
                            grad = jnp.zeros((0, H, 2), dtype=jnp.float32)
                            return sdf, grad
                        rel = p[None, :, :] - box_centers[:, None, :]  # (Nb,H,2)
                        q = jnp.abs(rel) - box_half[:, None, :]        # (Nb,H,2)
                        outside = jnp.maximum(q, 0.0)
                        outside_dist = jnp.linalg.norm(outside, axis=-1)  # (Nb,H)
                        inside_dist = jnp.max(q, axis=-1)  # negative/zero inside
                        outside_mask = outside_dist > eps
                        sdf = jnp.where(outside_mask, outside_dist, inside_dist)

                        # outside gradient via closest point
                        closest = box_centers[:, None, :] + jnp.clip(rel, -box_half[:, None, :], box_half[:, None, :])
                        d = p[None, :, :] - closest
                        grad_out = d / (outside_dist[:, :, None] + eps)

                        # inside gradient: axis of max(q), direction = sign(rel)
                        axis = jnp.argmax(q, axis=-1)  # (Nb,H)
                        onehot = jax.nn.one_hot(axis, 2, dtype=jnp.float32)  # (Nb,H,2)
                        # IMPORTANT: inside-box SDF is nonsmooth; we need a nonzero subgradient.
                        # jnp.sign(0)=0 would produce a zero gradient and stall projection.
                        rel_sign = jnp.where(rel >= 0.0, 1.0, -1.0)
                        grad_in = onehot * rel_sign
                        grad = jnp.where(outside_mask[:, :, None], grad_out, grad_in)
                        return sdf, grad

                    sdf_c, grad_c = circle_sdf_grad(pos)
                    sdf_b, grad_b = box_sdf_grad(pos)
                    sdf_all = jnp.concatenate([sdf_c, sdf_b], axis=0)  # (Nobs,H)
                    grad_all = jnp.concatenate([grad_c, grad_b], axis=0)  # (Nobs,H,2)
                    n_obs = sdf_all.shape[0]

                    # Select top-K closest obstacles per timestep (within threshold).
                    sdf_t = sdf_all.T  # (H, Nobs)
                    grad_t = jnp.transpose(grad_all, (1, 0, 2))  # (H, Nobs, 2)
                    cand_mask = sdf_t < threshold_s
                    sdf_for_sort = jnp.where(cand_mask, sdf_t, jnp.inf)
                    neg = -sdf_for_sort
                    def topk_idx(v):
                        return jax.lax.top_k(v, k_select)[1]
                    idx_sel = jax.vmap(topk_idx)(neg)  # (H, k_select)
                    sdf_sel = jnp.take_along_axis(sdf_t, idx_sel, axis=1)  # (H,k)
                    grad_sel = jnp.take_along_axis(grad_t, idx_sel[:, :, None], axis=1)  # (H,k,2)
                    valid = jnp.isfinite(sdf_sel) & (sdf_sel < threshold_s)

                    # b_{t,k} for prefix inequality: dt * sum_{i<=t} g^T u_i >= b
                    rhs = clearance_s - sdf_sel + jnp.einsum("hkd,hd->hk", grad_sel, pos)
                    b = rhs - jnp.einsum("hkd,d->hk", grad_sel, p0)
                    b = jnp.where(valid, b, -jnp.inf)
                    grad_sel = jnp.where(valid[:, :, None], grad_sel, 0.0)

                    # Structured hard/slack projection in action space (prefix constraints).
                    tol = jnp.asarray(1e-7, dtype=jnp.float32)
                    # This is the main inner loop cost; keep it large enough to actually converge
                    # on multi-constraint scenes (still cheap vs building A_full).
                    max_iter = jnp.maximum(jnp.asarray(200, dtype=jnp.int32), jnp.asarray(I_QP_s, dtype=jnp.int32) * 20)
                    time_idx = jnp.arange(H, dtype=jnp.int32)  # static shape (H,)

                    def compute_max_violation(u_curr):
                        u_clip = jnp.clip(u_curr, -control_limit, control_limit)
                        s = jnp.cumsum(u_clip, axis=0)  # (H,2)
                        lhs = jnp.asarray(dt, dtype=jnp.float32) * jnp.einsum("hkd,hd->hk", grad_sel, s)  # (H,k)
                        viol = b - lhs
                        viol = jnp.where(jnp.isfinite(b), jnp.maximum(0.0, viol), -jnp.inf)
                        maxv = jnp.max(viol)
                        arg = jnp.argmax(viol.reshape(-1))
                        t_idx = arg // jnp.asarray(k_select, dtype=jnp.int32)
                        k_idx = arg - t_idx * jnp.asarray(k_select, dtype=jnp.int32)
                        return maxv, t_idx, k_idx

                    def cond_fn(carry):
                        i, u_curr, maxv, *_ = carry
                        return jnp.logical_and(i < max_iter, maxv > tol)

                    def body_loop(carry):
                        i, u_curr, _maxv, _t, _k = carry
                        maxv, t_idx, k_idx = compute_max_violation(u_curr)
                        g = grad_sel[t_idx, k_idx]  # (2,)
                        g2 = jnp.dot(g, g) + 1e-9
                        denom = (jnp.asarray(t_idx, dtype=jnp.float32) + 1.0) * (jnp.asarray(dt, dtype=jnp.float32) ** 2) * g2
                        # Slack support if enabled
                        use_slack_flag = jnp.logical_and(rho_s > 0, jnp.asarray(self.use_slack, dtype=jnp.bool_))
                        denom = jax.lax.cond(
                            use_slack_flag,
                            lambda d: d + (jnp.asarray(dt, dtype=jnp.float32) ** 2) * (1.0 / (rho_s + 1e-9)),
                            lambda d: d,
                            denom,
                        )
                        lam = maxv / (denom + 1e-9)
                        du = (jnp.asarray(dt, dtype=jnp.float32) * lam) * g  # (2,)
                        # Update prefix actions 0..t_idx (JIT-safe: no dynamic slicing)
                        prefix_mask = (time_idx <= t_idx).astype(jnp.float32)[:, None]  # (H,1)
                        u_next = u_curr + prefix_mask * du[None, :]
                        u_next = jnp.clip(u_next, -control_limit, control_limit)
                        return (i + 1, u_next, maxv, t_idx, k_idx)

                    maxv0, t0, k0 = compute_max_violation(u_seq)
                    init = (jnp.asarray(0, dtype=jnp.int32), u_seq, maxv0, t0, k0)
                    _, u_proj, *_ = jax.lax.while_loop(cond_fn, body_loop, init)
                    return u_proj
            
                if can_use_multi and num_obs_fns > 0:
                    # Build constraints for each timestep
                    # For single-integrator dynamics, state x_{t+1} depends on prefix actions u_0..u_t.
                    # We build constraints on positions p_{t+1} using the linearized obstacle SDF at the
                    # nominal p_{t+1}. This yields action-space constraints with *dynamics coupling*
                    # (non-block-diagonal A_full).
                    p0 = x0[0:2]

                    def build_constraints_for_timestep(t):
                        # Constrain the *next* state position p_{t+1} (more consistent than constraining p_t).
                        pos_t = states_rollout[t + 1][0:2]
                    
                        # Stage 1: Compute SDF for all obstacles
                        def compute_obstacle_sdf_only(obs_idx, pos):
                            valid = jnp.logical_and(obs_idx >= 0, obs_idx < num_obs_fns)
                            obs_idx_clipped = jnp.clip(obs_idx, 0, max_k - 1)
                            sdf_val = jax.lax.switch(obs_idx_clipped, obstacle_branches_sdf_tuple, pos)
                            return jnp.where(valid, sdf_val, jnp.inf)

                        # Candidate set (spatial grid): fixed-size list of obstacle indices for this cell.
                        # IMPORTANT: `cell_obs_idx` is either a constant JAX array (available) or None (disabled).
                        # Do NOT use jax.lax.cond on a python-level None, because both branches are traced.
                        if cell_obs_idx is not None:
                            ix = jnp.floor((pos_t[0] - jnp.asarray(grid_x_min, dtype=jnp.float32)) / jnp.asarray(cell_size, dtype=jnp.float32)).astype(jnp.int32)
                            iy = jnp.floor((pos_t[1] - jnp.asarray(grid_y_min, dtype=jnp.float32)) / jnp.asarray(cell_size, dtype=jnp.float32)).astype(jnp.int32)
                            ix = jnp.clip(ix, 0, jnp.asarray(grid_W - 1, dtype=jnp.int32))
                            iy = jnp.clip(iy, 0, jnp.asarray(grid_H - 1, dtype=jnp.int32))
                            cand_idx = cell_obs_idx[iy, ix]  # (M,)
                        else:
                            cand_idx = jnp.arange(max_k, dtype=jnp.int32)

                        sdf_array = jax.vmap(lambda idx: compute_obstacle_sdf_only(idx, pos_t))(cand_idx)
                    
                        # Gate: check if constraints needed
                        min_sdf = jnp.min(sdf_array)
                        needs_constraints = min_sdf < threshold
                    
                        def compute_constraints(_):
                            cand_mask = sdf_array < threshold
                            sdf_for_sort = jnp.where(cand_mask, sdf_array, jnp.inf)
                        
                            # Select closest obstacles (smallest sdf). Using neg_sdf makes larger = closer.
                            # jax.lax.top_k returns indices sorted by descending score, so this is already
                            # closest-first. Do NOT flip, otherwise we'd pick farthest-first.
                            neg_sdf = -sdf_for_sort
                            # Pick top-K closest obstacles across ALL obstacles.
                            _, selected_rank = jax.lax.top_k(neg_sdf, k_select)
                        
                            num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
                            k = jnp.minimum(
                                jnp.asarray(k_select, dtype=jnp.int32),
                                jnp.where(num_candidates == 0, k_select, num_candidates),
                            )
                            selected_indices = jnp.take(cand_idx, selected_rank, axis=0)  # original obstacle indices
                            sdf_sel = jnp.take(sdf_array, selected_rank, axis=0)  # (k_select,)
                        
                            def compute_obstacle_grad_for_candidate(rank_idx, pos):
                                obs_idx = selected_indices[rank_idx]
                                obs_idx_clipped = jnp.clip(obs_idx, 0, max_k - 1)
                                grad_val = jax.lax.switch(obs_idx_clipped, obstacle_branches_grad_tuple, pos)
                                valid_rank = rank_idx < k
                                valid_obs = jnp.logical_and(obs_idx >= 0, obs_idx < num_obs_fns)
                                valid = jnp.logical_and(valid_rank, valid_obs)
                                grad_val = jnp.where(valid, grad_val, jnp.zeros(2))
                                return grad_val
                        
                            grad_array = jax.vmap(lambda rank_idx: compute_obstacle_grad_for_candidate(rank_idx, pos_t))(jnp.arange(k_select))
                        
                            def build_constraint_for_obstacle(idx):
                                valid_idx = idx < k
                                sdf_obs = jnp.where(valid_idx, sdf_sel[idx], jnp.inf)
                                grad_obs = jnp.where(valid_idx, grad_array[idx], jnp.zeros(2))
                            
                                grad_norm = jnp.linalg.norm(grad_obs)
                                valid_grad = jnp.logical_and(valid_idx, grad_norm > 1e-8)

                                # Linearized position-space constraint (standard):
                                #   d(p) >= clearance
                                #   d(p_ref) + grad^T (p - p_ref) >= clearance
                                # => grad^T p >= clearance - d(p_ref) + grad^T p_ref
                                #
                                # With single-integrator dynamics:
                                #   p_{t+1} = p0 + dt * sum_{i=0}^t u_i
                                # => dt * sum_{i=0}^t grad^T u_i >= (clearance - d_ref + grad^T p_ref) - grad^T p0
                                #
                                # We return (grad, b) for this timestep; A_full is assembled with prefix coupling.
                                grad_use = jnp.where(valid_grad, grad_obs, jnp.zeros(2))
                                rhs_state = clearance - sdf_obs + jnp.dot(grad_use, pos_t)
                                b_val = rhs_state - jnp.dot(grad_use, p0)
                                b_val = jnp.where(valid_grad, b_val, -jnp.inf)
                                return grad_use, b_val, valid_grad
                        
                            constraint_results = jax.vmap(build_constraint_for_obstacle)(jnp.arange(k_select))
                            A_all = constraint_results[0]
                            b_all = constraint_results[1]
                            valid_mask = constraint_results[2]
                        
                            A_constraints = A_all
                            b_constraints = jnp.where(valid_mask, b_all, -jnp.inf)
                        
                            return A_constraints, b_constraints
                    
                        def skip_constraints(_):
                            A_empty = jnp.zeros((k_select, 2))
                            b_empty = jnp.full((k_select,), -jnp.inf)
                            return A_empty, b_empty
                    
                        A_t, b_t = jax.lax.cond(needs_constraints, compute_constraints, skip_constraints, operand=None)
                        return A_t, b_t
                
                    # Build constraints for all timesteps (t = 0..H-1 constrains p_{t+1})
                    constraint_results = jax.vmap(build_constraints_for_timestep)(jnp.arange(H))
                    A_per_step = constraint_results[0]  # (H, max_k, act_dim)
                    b_per_step = constraint_results[1]  # (H, max_k)
                
                    # Avoid materializing dense A_full: solve in structured (prefix-sum) form.
                    # Include dt scaling inside A (matches previous A_full assembly).
                    A_eff = (jnp.asarray(dt, dtype=jnp.float32) * A_per_step).astype(jnp.float32)  # (H, K, act_dim)
                    b_eff = b_per_step.astype(jnp.float32)  # (H, K)

                    # Inner solver effort: same knob as before (kept for compatibility).
                    solver_iters = jnp.maximum(jnp.asarray(10, dtype=jnp.int32), I_QP * 2)

                    from enerdynamics.core.constraints.solvers.jaxopt_osqp_solver import solve_slack_qp_prefixsum_jax

                    # Always use slack-QP; emulate hard constraints with huge rho.
                    rho_eff = jax.lax.cond(
                        jnp.logical_and(jnp.asarray(self.use_slack, dtype=jnp.bool_), rho > 0),
                        lambda __: jnp.asarray(rho, dtype=jnp.float32),
                        lambda __: jnp.asarray(1e9, dtype=jnp.float32),
                        operand=None,
                    )

                    u_safe, _v = solve_slack_qp_prefixsum_jax(
                        u_seq,
                        A_eff,
                        b_eff,
                        rho_eff,
                        control_limit=float(control_limit),
                        tol=1e-7,
                        maxiter=solver_iters,
                    )
                    return u_safe
                else:
                    return u_seq

            def filter_single(u_seq, margin_s, rho_s, I_QP_s):
                """Paper-style CFS iteration with while_loop (supports per-sample params)."""
                margin_s = jnp.asarray(margin_s, dtype=jnp.float32)
                rho_s = jnp.asarray(rho_s, dtype=jnp.float32)
                I_QP_i32 = jnp.asarray(I_QP_s, dtype=jnp.int32)
                clearance_s = margin_s + robot_radius_jax
                threshold_s = clearance_s + constraint_margin_jax

                def cond_fn(carry):
                    i, _u = carry
                    return i < I_QP_i32

                def body_loop(carry):
                    i, u_curr = carry
                    return (i + 1, filter_single_once(u_curr, clearance_s=clearance_s, threshold_s=threshold_s, rho_s=rho_s, I_QP_s=I_QP_i32))

                _, u_out = jax.lax.while_loop(cond_fn, body_loop, (jnp.asarray(0, dtype=jnp.int32), u_seq))
                return u_out
        
            # Handle batching
            if actions.ndim == 3:
                # Support per-sample schedule parameters (vectors) as well as scalars.
                margin_b = margin_jax
                rho_b = rho
                I_b = I_QP
                if hasattr(margin_b, "ndim") and getattr(margin_b, "ndim", 0) > 0:
                    return jax.vmap(filter_single, in_axes=(0, 0, 0, 0))(actions, margin_b, rho_b, I_b)
                return jax.vmap(lambda u: filter_single(u, margin_b, rho_b, I_b))(actions)
            return filter_single(actions, margin_jax, rho, I_QP)
        return jax.lax.cond(should_do_qp, do_qp, skip_qp, operand)
    
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
