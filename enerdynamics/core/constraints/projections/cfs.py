"""
Convex Feasible Set (CFS) projection implementation.

This module provides CFSProjection: A feasibility operator that implements
CFS-QP (Convex Feasible Set via Quadratic Programming) for non-convex obstacle
constraints using linearization.

Optimized with batch tensor operations for efficiency.
"""

import numpy as np
from typing import Callable, Optional, Tuple, List
from enerdynamics.core.constraints.base import FeasibilityOperator
from enerdynamics.core.types import Trajectory, State
from enerdynamics.envs.obstacles.base import ObstacleManager
from typing import TYPE_CHECKING
from itertools import combinations

if TYPE_CHECKING:
    from enerdynamics.core.constraints.schedule import ConstraintScheduleManager

try:
    # Optional: used for true trajectory-level CFS-QP.
    import cvxopt  # type: ignore
    from cvxopt import matrix as cvx_matrix, spmatrix as cvx_spmatrix, solvers as cvx_solvers  # type: ignore
except Exception:  # pragma: no cover
    cvxopt = None
    cvx_matrix = None
    cvx_spmatrix = None
    cvx_solvers = None

try:
    import scipy.sparse as sp
except Exception:  # pragma: no cover
    sp = None

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jax = None
    jnp = None
    JAX_AVAILABLE = False

try:
    # Optional: JAX-based QP solver
    import jaxopt
    JAXOPT_AVAILABLE = True
except ImportError:
    jaxopt = None
    JAXOPT_AVAILABLE = False


class CFSProjection(FeasibilityOperator):
    """
    Convex Feasible Set (CFS) projection via linearization.
    
    For non-convex obstacles, linearizes constraints around reference trajectory
    to create convex inner approximation. Implements QP-based projection.
    
    This implements the CFS-QP method from the theory for efficient projection
    onto non-convex obstacle-free space.
    
    Optimized with batch tensor operations to avoid nested loops.
    """
    
    def __init__(
        self,
        obstacles: ObstacleManager,
        clearance_schedule: Optional[Callable[[Optional[int], Optional[int]], float]] = None,
        position_extractor: Optional[Callable[[State], np.ndarray]] = None,
        use_late_stage_only: bool = False,
        late_stage_ratio: float = 0.3,  # Only apply in last 30% of steps
        use_trajectory_qp: bool = True,
        smoothness_weight: float = 0.0,
        reconstruct_velocity: bool = False,
        velocity_dt: Optional[float] = None,
        max_iterations: int = 5,
        convergence_tol: float = 1e-6,
        # Performance knobs: limit number of constraints per point
        max_constraints_per_point: int = 8,
        constraint_margin: float = 0.25,
        schedule_manager: Optional["ConstraintScheduleManager"] = None,
    ):
        """
        Initialize CFS projection operator.
        
        Args:
            obstacles: ObstacleManager containing obstacles
            clearance_schedule: Function (step, total_steps) -> clearance
                               (if None, uses fixed clearance from hard constraints or schedule_manager)
            position_extractor: Function to extract position from state
            use_late_stage_only: If True, only apply projection in late stages
            late_stage_ratio: Ratio of steps for late-stage application
            use_trajectory_qp: If True, solve a single trajectory-level QP (CFS-QP) when cvxopt is available.
            smoothness_weight: Optional smoothness regularization weight for trajectory-level QP.
            reconstruct_velocity: If True and state dim==4, rebuild velocities from projected positions.
            velocity_dt: Time step used for velocity reconstruction (required if reconstruct_velocity=True).
            max_iterations: Maximum iterations for iterative projection
            convergence_tol: Convergence tolerance for iterative projection
            schedule_manager: Optional ConstraintScheduleManager (overrides clearance_schedule if provided)
        """
        self.obstacles = obstacles
        self.clearance_schedule = clearance_schedule  # Legacy support
        self.position_extractor = position_extractor or self._default_extract_position
        self.use_late_stage_only = use_late_stage_only
        self.late_stage_ratio = late_stage_ratio
        self.use_trajectory_qp = bool(use_trajectory_qp)
        self.smoothness_weight = float(max(0.0, smoothness_weight))
        self.reconstruct_velocity = bool(reconstruct_velocity)
        self.velocity_dt = None if velocity_dt is None else float(velocity_dt)
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.max_constraints_per_point = int(max(1, max_constraints_per_point))
        self.constraint_margin = float(max(0.0, constraint_margin))
        self.schedule_manager = schedule_manager
    
    def should_apply(self, step: Optional[int] = None, total_steps: Optional[int] = None) -> bool:
        """Check if projection should be applied at current step."""
        if not self.use_late_stage_only:
            return True
        
        if step is None or total_steps is None:
            return True
        
        # Only apply in late stages
        progress = step / total_steps if total_steps > 0 else 1.0
        return progress >= (1.0 - self.late_stage_ratio)
    
    def project(
        self,
        trajectory: Trajectory,
        step: Optional[int] = None,
        total_steps: Optional[int] = None
    ) -> Trajectory:
        """
        Project using CFS: linearize obstacles around trajectory and solve QP.
        
        Optimized batch version that processes all states in parallel.
        
        Args:
            trajectory: Trajectory to project
            step: Current diffusion step
            total_steps: Total diffusion steps
            
        Returns:
            Projected trajectory
        """
        # Late-stage gating (optional): if disabled for this step, return trajectory unchanged.
        if not self.should_apply(step=step, total_steps=total_steps):
            return trajectory

        # Get clearance (scheduled if provided)
        if self.schedule_manager is not None:
            clearance = self.schedule_manager.get_hard_clearance(
                default=0.0, step=step, total_steps=total_steps
            )
        elif self.clearance_schedule is not None:
            clearance = self.clearance_schedule(step, total_steps)
        else:
            clearance = 0.0  # Default
        
        # Extract all positions at once (batch processing)
        # Use list comprehension then stack for better performance
        position_list = [self.position_extractor(state) for state in trajectory.states]
        if len(position_list) > 0:
            # Stack into array: more efficient than creating from list
            positions = np.stack(position_list, axis=0).astype(np.float32)
        else:
            positions = np.zeros((0, 2), dtype=np.float32)  # Empty trajectory
        
        # Batch project all positions
        projected_positions = self._project_cfs_batch(
            positions, 
            clearance, 
            step
        )
        
        # Optionally reconstruct velocities (for 4D [px,py,vx,vy] states) from the projected positions.
        vel_from_pos = None
        if (
            self.reconstruct_velocity
            and self.velocity_dt is not None
            and len(trajectory.states) >= 2
            and np.asarray(trajectory.states[0]).shape[0] == 4
        ):
            dt = float(self.velocity_dt)
            if dt > 0:
                vel = np.zeros_like(projected_positions, dtype=np.float32)
                vel[1:] = (projected_positions[1:] - projected_positions[:-1]) / dt
                vel[0] = vel[1]
                vel_from_pos = vel

        # Reconstruct states with projected positions (and optional velocity update)
        projected_states = []
        for i, state in enumerate(trajectory.states):
            state_np = np.asarray(state, dtype=np.float32)
            new_state = state_np.copy()
            new_pos = np.asarray(projected_positions[i], dtype=np.float32).flatten()
            if state_np.shape[0] == 4:
                new_state[:2] = new_pos[:2]
                if vel_from_pos is not None:
                    new_state[2:4] = vel_from_pos[i][:2]
            elif state_np.shape[0] >= 2:
                new_state[:2] = new_pos[:2]
            elif state_np.shape[0] == 1:
                new_state[:1] = new_pos[:1]
            projected_state = new_state
            projected_states.append(projected_state)
        
        return Trajectory(states=projected_states, actions=trajectory.actions)
    
    def _project_cfs_batch(
        self,
        positions: np.ndarray,
        clearance: float,
        step: Optional[int]
    ) -> np.ndarray:
        """
        Batch project multiple points onto CFS using linearized constraints and a QP.

        Implements a CFS-QP style projection step:
        For each point x (reference), linearize each obstacle SDF constraint

            d_m(x) >= clearance

        as a halfspace constraint:

            d_m(x_ref) + ∇d_m(x_ref)^T (x - x_ref) >= clearance
            => ∇d_m(x_ref)^T x >= clearance - d_m(x_ref) + ∇d_m(x_ref)^T x_ref

        Then solve the (small) QP (Euclidean projection onto the intersection of halfspaces):

            minimize    0.5 ||x - x_ref||^2
            subject to  A x >= b

        With an identity Hessian, the solution is the projection of x_ref onto the polyhedron.
        
        Args:
            positions: Points to project, shape (N, dim)
            clearance: Minimum clearance required
            step: Current step (for logging/debugging)
            
        Returns:
            Projected positions, shape (N, dim)
        """
        positions = np.asarray(positions, dtype=np.float32)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        N, dim = positions.shape
        current = positions.copy()

        if N == 0:
            return current

        obstacles_list = list(self.obstacles)
        if len(obstacles_list) == 0:
            return current
        
        can_solve_trajectory_qp = (
            bool(self.use_trajectory_qp)
            and cvx_solvers is not None
            and cvx_matrix is not None
            and cvx_spmatrix is not None
            and sp is not None
        )

        # Iteratively linearize + solve QP (CFS outer loop)
        for _ in range(self.max_iterations):
            # Compute per-obstacle SDF for all points ONCE (batch),
            # so we can reuse the same values for:
            # - deciding violations (union SDF)
            # - selecting a small set of "active" constraints per point
            # This avoids re-evaluating sdf(point) in a second pass.
            sdf_rows = []
            for obs in obstacles_list:
                vals = obs.sdf(current)
                vals = np.asarray(vals, dtype=np.float32)
                if vals.ndim == 0:
                    vals = np.full((N,), float(vals), dtype=np.float32)
                sdf_rows.append(vals)
            sdf_matrix = np.stack(sdf_rows, axis=0)  # (M, N)

            # Union SDF: closest obstacle
            union_sdf = np.min(sdf_matrix, axis=0)
            violating_mask = (union_sdf < clearance)

            if not np.any(violating_mask):
                break  # All feasible

            prev = current.copy()

            if can_solve_trajectory_qp and N > 1:
                # True CFS-QP: solve one trajectory-level QP for all points at once.
                # We include constraints for points that are within (clearance + margin) of any obstacle.
                active_mask = union_sdf < float(clearance + self.constraint_margin)
                active_indices = np.where(active_mask)[0]
                if active_indices.size == 0:
                    # Nothing near obstacles; no need to solve a QP.
                    break

                A_data: List[float] = []
                A_row: List[int] = []
                A_col: List[int] = []
                b_rows: List[float] = []
                row = 0

                for idx in active_indices:
                    x_ref = current[idx]
                    d0_all = sdf_matrix[:, idx]  # (M,)
                    threshold = float(clearance + self.constraint_margin)
                    cand_mask = d0_all < threshold
                    cand_indices = np.where(cand_mask)[0]
                    if cand_indices.size == 0:
                        # Fallback: take the closest K obstacles (by SDF)
                        k = min(self.max_constraints_per_point, d0_all.shape[0])
                        cand_indices = np.argsort(d0_all)[:k]
                    else:
                        # Keep only K closest among candidates
                        k = min(self.max_constraints_per_point, cand_indices.size)
                        cand_indices = cand_indices[np.argsort(d0_all[cand_indices])[:k]]

                    for j in cand_indices:
                        obstacle = obstacles_list[int(j)]
                        d0 = float(d0_all[int(j)])

                        grad = None
                        if hasattr(obstacle, "gradient"):
                            try:
                                grad = obstacle.gradient(x_ref)
                            except Exception:
                                grad = None
                        if grad is None:
                            grad = self._finite_difference_gradient(obstacle, x_ref)
                        grad = np.asarray(grad, dtype=np.float32).flatten()

                        gnorm = float(np.linalg.norm(grad))
                        if not np.isfinite(gnorm) or gnorm < 1e-8:
                            continue

                        g = grad / gnorm
                        b = float((clearance - d0) / gnorm + float(np.dot(g, x_ref)))

                        for kdim in range(dim):
                            A_data.append(float(g[kdim]))
                            A_row.append(row)
                            A_col.append(int(idx * dim + kdim))
                        b_rows.append(b)
                        row += 1

                x_flat = prev.reshape(-1).astype(np.float64)
                b_vec = np.asarray(b_rows, dtype=np.float64)
                try:
                    x_new_flat = self._solve_trajectory_qp_identity_cvxopt(
                        x0=x_flat,
                        A_data=A_data,
                        A_row=A_row,
                        A_col=A_col,
                        b=b_vec,
                        n_vars=int(N * dim),
                        n_cons=int(len(b_rows)),
                        dim=int(dim),
                        T=int(N),
                        smoothness_weight=float(self.smoothness_weight),
                    )
                    current = x_new_flat.reshape(N, dim).astype(np.float32)
                except Exception:
                    # If the global QP fails (e.g., infeasible), fall back to pointwise projection.
                    current = prev
                    can_solve_trajectory_qp = False

            if not can_solve_trajectory_qp:
                # Fallback: project each violating point by solving a small QP (projection onto halfspaces)
                for idx in np.where(violating_mask)[0]:
                    x_ref = current[idx]
                    # Select a small set of candidate obstacles for constraints at this point.
                    # We include obstacles inside (clearance + margin), then keep the closest K.
                    d0_all = sdf_matrix[:, idx]  # (M,)
                    threshold = float(clearance + self.constraint_margin)
                    cand_mask = d0_all < threshold
                    cand_indices = np.where(cand_mask)[0]
                    if cand_indices.size == 0:
                        # Fallback: take the closest K obstacles (by SDF)
                        k = min(self.max_constraints_per_point, d0_all.shape[0])
                        cand_indices = np.argsort(d0_all)[:k]
                    else:
                        # Keep only K closest among candidates
                        k = min(self.max_constraints_per_point, cand_indices.size)
                        cand_indices = cand_indices[np.argsort(d0_all[cand_indices])[:k]]

                    A, b = self._build_linearized_halfspaces_from_candidates(
                        x_ref,
                        clearance,
                        obstacles_list,
                        cand_indices,
                        d0_all,
                    )
                    if A.size == 0:
                        continue
                    current[idx] = self._solve_projection_qp_identity(x_ref, A, b)

            max_step = float(np.max(np.linalg.norm(current - prev, axis=1)))

            # Avoid treating "no movement" as convergence if still violating.
            # If we can't move (e.g., zero/invalid gradients), stop iterating to avoid wasting time.
            if max_step < self.convergence_tol:
                # One last check: if still violating, we're stagnating.
                union_sdf2 = np.asarray(self.obstacles.sdf(current), dtype=np.float32)
                if union_sdf2.ndim == 0:
                    union_sdf2 = np.full((N,), float(union_sdf2), dtype=np.float32)
                if np.any(union_sdf2 < clearance):
                    break
        
        return current

    def _build_linearized_halfspaces_from_candidates(
        self,
        x_ref: np.ndarray,
        clearance: float,
        obstacles_list: List,
        cand_indices: np.ndarray,
        d0_all: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build halfspace constraints A x >= b from linearized obstacle SDF constraints
        at the reference point x_ref, using only a subset of candidate obstacles.

        This is the main speedup path: limit constraint count per point.
        """
        x_ref = np.asarray(x_ref, dtype=np.float32).flatten()
        A_rows: List[np.ndarray] = []
        b_rows: List[float] = []

        for j in cand_indices:
            obstacle = obstacles_list[int(j)]
            d0 = float(d0_all[int(j)])
            # Note: we intentionally do NOT skip obstacles with d0 >= clearance here.
            # Candidate selection already limits us to near obstacles; keeping these
            # constraints helps prevent the QP from moving into obstacles.

            grad = None
            if hasattr(obstacle, "gradient"):
                try:
                    grad = obstacle.gradient(x_ref)
                except Exception:
                    grad = None

            if grad is None:
                grad = self._finite_difference_gradient(obstacle, x_ref)
            grad = np.asarray(grad, dtype=np.float32).flatten()

            gnorm = float(np.linalg.norm(grad))
            if not np.isfinite(gnorm) or gnorm < 1e-8:
                # If gradient is unusable, skip this obstacle; projection may stagnate.
                continue

            # Linearized constraint:
            #   grad^T x >= clearance - d0 + grad^T x_ref
            # We normalize for numerical stability, but must scale b consistently.
            g = grad / gnorm
            b = float((clearance - d0) / gnorm + float(np.dot(g, x_ref)))
            A_rows.append(g)
            b_rows.append(b)

        if not A_rows:
            return np.zeros((0, x_ref.shape[0]), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        A = np.stack(A_rows, axis=0).astype(np.float32)
        b = np.asarray(b_rows, dtype=np.float32)
        return A, b

    @staticmethod
    def _finite_difference_gradient(obstacle, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """Numerically approximate ∇sdf(x) with central differences."""
        x = np.asarray(x, dtype=np.float32).flatten()
        dim = x.shape[0]
        grad = np.zeros((dim,), dtype=np.float32)
        for k in range(dim):
            xp = x.copy()
            xm = x.copy()
            xp[k] += eps
            xm[k] -= eps
            dp = obstacle.sdf(xp)
            dm = obstacle.sdf(xm)
            dp = float(np.asarray(dp).item() if hasattr(dp, "item") else dp)
            dm = float(np.asarray(dm).item() if hasattr(dm, "item") else dm)
            grad[k] = (dp - dm) / (2.0 * eps)
        return grad

    @staticmethod
    def _is_feasible(A: np.ndarray, b: np.ndarray, x: np.ndarray, tol: float = 1e-7) -> bool:
        """Check A x >= b within a tolerance."""
        if A.size == 0:
            return True
        lhs = A @ x
        return bool(np.all(lhs + tol >= b))

    @staticmethod
    def _coo_to_cvx_spmatrix(
        data: np.ndarray, row: np.ndarray, col: np.ndarray, shape: Tuple[int, int]
    ):
        if cvx_spmatrix is None:
            raise RuntimeError("cvxopt is not available")
        return cvx_spmatrix(
            data.astype(np.float64).tolist(),
            row.astype(int).tolist(),
            col.astype(int).tolist(),
            size=shape,
        )

    def _solve_trajectory_qp_identity_cvxopt(
        self,
        *,
        x0: np.ndarray,
        A_data: List[float],
        A_row: List[int],
        A_col: List[int],
        b: np.ndarray,
        n_vars: int,
        n_cons: int,
        dim: int,
        T: int,
        smoothness_weight: float,
    ) -> np.ndarray:
        """
        Solve trajectory-level CFS-QP:

            minimize   0.5 ||x - x0||^2 + 0.5 * w * ||D2 x||^2
            subject to A x >= b

        where x is the flattened trajectory [p0, p1, ..., p_{T-1}] and D2 is a
        second-difference operator along time (applied per dimension).

        Notes:
        - Uses cvxopt.solvers.qp. Inequalities are provided as Gx <= h.
        - Falls back to caller on exceptions (e.g., infeasible / solver unavailable).
        """
        if cvx_solvers is None or cvx_matrix is None or sp is None:
            raise RuntimeError("Trajectory QP solver dependencies not available (cvxopt/scipy).")
        if n_cons <= 0:
            return np.asarray(x0, dtype=np.float64).reshape(-1)

        x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
        if x0.shape[0] != n_vars:
            raise ValueError(f"x0 has shape {x0.shape}, expected ({n_vars},)")

        # Hessian: I + w * (D2^T D2) kron I_dim
        P = sp.eye(n_vars, format="csc", dtype=np.float64)
        w = float(max(0.0, smoothness_weight))
        if w > 0.0 and T >= 3:
            r = []
            c = []
            d = []
            for t in range(T - 2):
                # second difference: x_t - 2 x_{t+1} + x_{t+2}
                r.extend([t, t, t])
                c.extend([t, t + 1, t + 2])
                d.extend([1.0, -2.0, 1.0])
            D2 = sp.coo_matrix((d, (r, c)), shape=(T - 2, T), dtype=np.float64).tocsr()
            K = (D2.T @ D2).tocsc()
            P_time = (sp.eye(T, format="csc", dtype=np.float64) + w * K).tocsc()
            P = sp.kron(P_time, sp.eye(dim, format="csc", dtype=np.float64), format="csc")

        P_coo = P.tocoo()
        P_cvx = self._coo_to_cvx_spmatrix(P_coo.data, P_coo.row, P_coo.col, P_coo.shape)

        # Linear term: 0.5||x-x0||^2 -> q = -x0
        q_cvx = cvx_matrix((-x0).astype(np.float64))

        # Constraints: A x >= b  <=>  (-A) x <= (-b)
        A_sp = sp.coo_matrix(
            (np.asarray(A_data, dtype=np.float64), (np.asarray(A_row), np.asarray(A_col))),
            shape=(n_cons, n_vars),
            dtype=np.float64,
        )
        G_sp = (-A_sp).tocoo()
        G_cvx = self._coo_to_cvx_spmatrix(G_sp.data, G_sp.row, G_sp.col, G_sp.shape)
        h_cvx = cvx_matrix((-np.asarray(b, dtype=np.float64)).reshape(-1))

        # Solve QP
        old_show = cvx_solvers.options.get("show_progress", True)
        cvx_solvers.options["show_progress"] = False
        try:
            sol = cvx_solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        finally:
            cvx_solvers.options["show_progress"] = old_show

        if sol is None or sol.get("status", "") not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"cvxopt qp failed: status={None if sol is None else sol.get('status')}")
        x = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
        return x

    @staticmethod
    def _solve_projection_qp_identity(x0: np.ndarray, A: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Solve: min 0.5||x - x0||^2 s.t. A x >= b.

        With identity Hessian, the optimum lies on an intersection of up to 'dim' active constraints.
        We enumerate candidate active sets up to dimension 2/3 (works well for 2D planning).
        """
        print(f"Solving projection QP with x0 shape {x0.shape}, A shape {A.shape}, b shape {b.shape}")
        x0 = np.asarray(x0, dtype=np.float32).flatten()
        A = np.asarray(A, dtype=np.float32)
        b = np.asarray(b, dtype=np.float32).flatten()
        m, dim = A.shape

        # If already feasible, nothing to do.
        if CFSProjection._is_feasible(A, b, x0):
            return x0

        best_x = None
        best_obj = float("inf")

        # Active set size 1: projection onto a single hyperplane a^T x = b.
        for i in range(m):
            a = A[i]
            denom = float(np.dot(a, a))
            if denom < 1e-12:
                continue
            alpha = float((b[i] - np.dot(a, x0)) / denom)
            x = x0 + alpha * a
            if CFSProjection._is_feasible(A, b, x):
                obj = float(np.sum((x - x0) ** 2))
                if obj < best_obj:
                    best_obj = obj
                    best_x = x

        # Active sets of size 2..dim: project onto equalities A_I x = b_I.
        # For identity Hessian, the constrained minimizer is:
        #   x = x0 + A_I^T λ,  where (A_I A_I^T) λ = b_I - A_I x0.
        max_k = min(dim, m)
        for k in range(2, max_k + 1):
            for idxs in combinations(range(m), k):
                AI = A[list(idxs)]
                bI = b[list(idxs)]
                rhs = bI - (AI @ x0)
                M = AI @ AI.T  # (k, k)
                # Skip near-singular active sets
                try:
                    rank = int(np.linalg.matrix_rank(M))
                except Exception:
                    rank = 0
                if rank < k:
                    continue
                try:
                    lam = np.linalg.solve(M, rhs)
                except Exception:
                    lam, *_ = np.linalg.lstsq(M, rhs, rcond=None)
                x = x0 + (AI.T @ lam)
                if CFSProjection._is_feasible(A, b, x):
                    obj = float(np.sum((x - x0) ** 2))
                    if obj < best_obj:
                        best_obj = obj
                        best_x = x.astype(np.float32)

        # Fallback: if enumeration didn't find a feasible point, do a conservative step along the most violated constraint.
        if best_x is None:
            violations = b - (A @ x0)
            i = int(np.argmax(violations))
            a = A[i]
            denom = float(np.dot(a, a))
            if denom > 1e-12:
                alpha = float((b[i] - np.dot(a, x0)) / denom)
                best_x = (x0 + alpha * a).astype(np.float32)
            else:
                best_x = x0

        return best_x.astype(np.float32)
    
    @staticmethod
    def _default_extract_position(state: State) -> np.ndarray:
        """Default position extractor."""
        state_np = np.asarray(state, dtype=np.float32)
        if len(state_np) == 4:
            return state_np[:2]
        elif len(state_np) == 2:
            # 2D position-only state
            return state_np[:2]
        elif len(state_np) == 1:
            return state_np[:1]
        return state_np[: min(2, len(state_np))]
    
    @staticmethod
    def _reconstruct_state(original_state: State, new_pos: np.ndarray) -> np.ndarray:
        """Reconstruct state with new position."""
        state_np = np.asarray(original_state, dtype=np.float32)
        new_state = state_np.copy()
        new_pos = np.asarray(new_pos, dtype=np.float32).flatten()
        if len(state_np) == 4:
            new_state[:2] = new_pos
        elif len(state_np) == 2:
            new_state[:2] = new_pos[:2]
        return new_state

    def make_jax_projector(self):
        """
        Create a JAX-compatible projection function for GPU acceleration.
        
        Returns a JAX-callable function that projects positions onto the CFS.
        Falls back to None if JAX is not available or obstacles are not JAX-compatible.
        
        Returns:
            Optional[Callable]: JAX function (positions_jax, clearance) -> projected_positions_jax,
                              or None if JAX projection is not available
        """
        if not JAX_AVAILABLE or jnp is None:
            return None
        
        # Check if obstacles can be converted to JAX-compatible functions
        obstacles_list = list(self.obstacles)
        if len(obstacles_list) == 0:
            # No obstacles: return identity function
            @jax.jit
            def identity_projector(positions, clearance):
                return positions
            return identity_projector
        
        # Check if ObstacleManager has SDF texture (preferred method for JAX)
        # If ObstacleManager has sample_sdf_and_grad_2d, we can use it directly
        # Otherwise, we need to check individual obstacles
        use_manager_sdf = hasattr(self.obstacles, "sample_sdf_and_grad_2d")
        if use_manager_sdf:
            # Check if SDF texture is built
            if not hasattr(self.obstacles, "_sdf_texture_2d") or self.obstacles._sdf_texture_2d is None:
                import warnings
                warnings.warn(
                    "ObstacleManager has sample_sdf_and_grad_2d but SDF texture not built. "
                    "Call obstacles.build_sdf_texture_2d(...) before creating CFSProjection to enable JAX acceleration.",
                    UserWarning
                )
                use_manager_sdf = False
        
        # Try to build JAX-compatible SDF and gradient functions
        try:
            sdf_fn, grad_fn = self._build_jax_obstacle_functions(obstacles_list, use_manager_sdf=use_manager_sdf)
        except RuntimeError as e:
            # Obstacles don't support JAX - return None to fall back to NumPy version
            # The error message is informative, but we don't want to fail here
            # Instead, EDOC will use the Python loop with NumPy CFS projection
            import warnings
            warnings.warn(
                f"JAX CFS projection not available: {str(e)}. "
                "Falling back to NumPy version (Python loop). "
                "To use JAX acceleration, ensure all obstacles support JAX "
                "(e.g., use SDF texture with build_sdf_texture_2d()).",
                UserWarning
            )
            return None
        except Exception as e:
            # Other unexpected errors - also fall back gracefully
            import warnings
            warnings.warn(
                f"Failed to create JAX CFS projector: {str(e)}. "
                "Falling back to NumPy version.",
                UserWarning
            )
            return None
        
        # Create JAX projection function
        max_iter = self.max_iterations
        conv_tol = self.convergence_tol
        max_constraints = self.max_constraints_per_point
        constraint_margin = self.constraint_margin
        
        @jax.jit
        def jax_project(positions: jnp.ndarray, clearance: jnp.ndarray) -> jnp.ndarray:
            """
            JAX-compatible CFS projection.
            
            Args:
                positions: Points to project, shape (N, dim) as JAX array
                clearance: Minimum clearance required (scalar or array)
                
            Returns:
                Projected positions, shape (N, dim) as JAX array
            """
            return _project_cfs_jax(
                positions,
                sdf_fn,
                grad_fn,
                clearance,
                max_iterations=max_iter,
                convergence_tol=conv_tol,
                max_constraints_per_point=max_constraints,
                constraint_margin=constraint_margin,
            )
        
        return jax_project
    
    def _build_jax_obstacle_functions(
        self, obstacles_list: List, use_manager_sdf: bool = False
    ) -> Tuple[Callable, Callable]:
        """
        Build JAX-compatible SDF and gradient functions from obstacles.
        
        Args:
            obstacles_list: List of obstacles
            use_manager_sdf: If True, use ObstacleManager's sample_sdf_and_grad_2d method
                           (requires SDF texture to be built)
            
        Returns:
            Tuple of (sdf_fn, grad_fn) where:
            - sdf_fn: (points) -> sdf_values, shape (M, N) for M obstacles and N points
            - grad_fn: (point) -> gradient, shape (M, dim) for M obstacles
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX is not available")
        
        # If using ObstacleManager's SDF texture, we can use it directly
        if use_manager_sdf:
            # Use ObstacleManager's sample_sdf_and_grad_2d (which uses SDF texture)
            def sdf_batch(points: jnp.ndarray) -> jnp.ndarray:
                """
                Compute SDF for all obstacles using ObstacleManager's SDF texture.
                
                Args:
                    points: Shape (N, dim)
                    
                Returns:
                    SDF matrix, shape (M, N) where M is number of obstacles
                """
                # Get union SDF from ObstacleManager (closest obstacle)
                sdf_vals, _ = self.obstacles.sample_sdf_and_grad_2d(points, backend="jax")
                sdf_vals = jnp.asarray(sdf_vals)
                
                # Ensure shape is (N,)
                if sdf_vals.ndim == 0:
                    sdf_vals = jnp.full((points.shape[0],), float(sdf_vals), dtype=jnp.float32)
                
                # For union SDF, we return a single row (M=1) representing the closest obstacle
                # This is sufficient for CFS projection which only needs the minimum SDF
                return sdf_vals[None, :]  # (1, N)
            
            def grad_multiple(point: jnp.ndarray, obs_indices: jnp.ndarray) -> jnp.ndarray:
                """
                Compute gradients using ObstacleManager's SDF texture.
                
                Args:
                    point: Shape (dim,)
                    obs_indices: Not used when using manager SDF (union gradient)
                    
                Returns:
                    Gradient, shape (1, dim) for union gradient
                """
                _, grad = self.obstacles.sample_sdf_and_grad_2d(point[None, :], backend="jax")
                grad = jnp.asarray(grad[0])  # Extract first (and only) point
                return grad[None, :]  # (1, dim)
            
            return sdf_batch, grad_multiple
        
        # Otherwise, check individual obstacles
        # Check if all obstacles support JAX (required for JIT compilation)
        # We cannot convert JAX arrays to NumPy inside JIT functions
        for obs in obstacles_list:
            has_jax_support = hasattr(obs, "jax_sdf")
            if not has_jax_support:
                # This obstacle doesn't support JAX - cannot use in JIT function
                raise RuntimeError(
                    f"Obstacle {type(obs).__name__} does not support JAX. "
                    "All obstacles must have either 'jax_sdf' method or use ObstacleManager's "
                    "'sample_sdf_and_grad_2d' method (requires build_sdf_texture_2d()) "
                    "to use JAX CFS projection."
                )
        
        # Build SDF function for all obstacles (all are JAX-compatible now)
        def sdf_batch(points: jnp.ndarray) -> jnp.ndarray:
            """
            Compute SDF for all obstacles at given points.
            
            Args:
                points: Shape (N, dim)
                
            Returns:
                SDF matrix, shape (M, N) where M is number of obstacles
            """
            sdf_rows = []
            for obs in obstacles_list:
                # All obstacles are guaranteed to have jax_sdf at this point
                if hasattr(obs, "jax_sdf"):
                    sdf_vals = obs.jax_sdf(points)
                else:
                    # This should not happen due to check above, but add safety
                    raise RuntimeError(f"Obstacle {type(obs).__name__} does not support JAX")
                
                # Ensure shape is (N,)
                if sdf_vals.ndim == 0:
                    sdf_vals = jnp.full((points.shape[0],), float(sdf_vals), dtype=jnp.float32)
                elif sdf_vals.ndim > 1:
                    sdf_vals = sdf_vals.flatten()[:points.shape[0]]
                
                sdf_rows.append(sdf_vals)
            
            if not sdf_rows:
                # No obstacles: return large positive values
                return jnp.full((1, points.shape[0]), 1e6, dtype=jnp.float32)
            
            return jnp.stack(sdf_rows, axis=0)  # (M, N)
        
            # Build gradient function for individual obstacles
            def grad_single(point: jnp.ndarray, obs_idx: int) -> jnp.ndarray:
                """
                Compute gradient for a single obstacle at a single point.
                
                Args:
                    point: Shape (dim,)
                    obs_idx: Index of obstacle
                    
                Returns:
                    Gradient, shape (dim,)
                """
                obs = obstacles_list[obs_idx]
                
                # All obstacles are guaranteed to have jax_sdf at this point
                if hasattr(obs, "jax_gradient"):
                    grad = obs.jax_gradient(point)
                else:
                    # Use finite differences with JAX (pure JAX implementation)
                    grad = _finite_difference_gradient_jax(obs, point)
                
                return jnp.asarray(grad, dtype=jnp.float32).flatten()
            
            # Create a function that computes gradients for multiple obstacles
            def grad_multiple(point: jnp.ndarray, obs_indices: jnp.ndarray) -> jnp.ndarray:
                """
                Compute gradients for multiple obstacles at a single point.
                
                Args:
                    point: Shape (dim,)
                    obs_indices: Indices of obstacles, shape (k,)
                    
                Returns:
                    Gradients, shape (k, dim)
                """
                # Use JAX-compatible indexing (cannot use NumPy in JIT function)
                num_indices = obs_indices.shape[0]
                grads = []
                for i in range(num_indices):
                    idx = int(obs_indices[i])  # Convert JAX array element to Python int
                    grad = grad_single(point, idx)
                    grads.append(grad)
                return jnp.stack(grads, axis=0) if grads else jnp.zeros((0, point.shape[0]), dtype=jnp.float32)
        
        return sdf_batch, grad_multiple


# ============================================================================
# JAX-compatible CFS projection functions
# ============================================================================

if JAX_AVAILABLE:

    def _finite_difference_gradient_jax(obstacle, point: jnp.ndarray, eps: float = 1e-4) -> jnp.ndarray:
        """
        Numerically approximate ∇sdf(x) with central differences using JAX.
        
        This function requires the obstacle to have JAX-compatible SDF method.
        It cannot use NumPy conversion inside JIT functions.
        
        Args:
            obstacle: Obstacle with jax_sdf or sample_sdf_and_grad_2d method
            point: Point to evaluate, shape (dim,) as JAX array
            eps: Step size for finite differences
            
        Returns:
            Gradient, shape (dim,)
        """
        point = jnp.asarray(point, dtype=jnp.float32).flatten()
        dim = point.shape[0]
        
        # Use JAX-compatible SDF computation (cannot convert to NumPy in JIT)
        grad = jnp.zeros((dim,), dtype=jnp.float32)
        
        for k in range(dim):
            # Create shifted points using JAX operations
            xp = point.at[k].add(eps)
            xm = point.at[k].add(-eps)
            
            # Compute SDF using JAX-compatible method
            if hasattr(obstacle, "jax_sdf"):
                dp = obstacle.jax_sdf(xp[None, :])  # Add batch dimension
                dm = obstacle.jax_sdf(xm[None, :])
                dp_val = dp[0] if dp.ndim > 0 else dp
                dm_val = dm[0] if dm.ndim > 0 else dm
            elif hasattr(obstacle, "sample_sdf_and_grad_2d"):
                dp, _ = obstacle.sample_sdf_and_grad_2d(xp[None, :], backend="jax")
                dm, _ = obstacle.sample_sdf_and_grad_2d(xm[None, :], backend="jax")
                dp_val = dp[0] if dp.ndim > 0 else dp
                dm_val = dm[0] if dm.ndim > 0 else dm
            else:
                # This should not happen if check above passed
                raise RuntimeError(f"Obstacle {type(obstacle).__name__} does not support JAX for finite differences")
            
            # Compute gradient using JAX operations
            grad = grad.at[k].set((dp_val - dm_val) / (2.0 * eps))
        
        return grad

    def _is_feasible_jax(A: jnp.ndarray, b: jnp.ndarray, x: jnp.ndarray, tol: float = 1e-7) -> bool:
        """
        Check A x >= b within a tolerance (JAX version).
        
        Args:
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            x: Point to check, shape (dim,)
            tol: Tolerance
            
        Returns:
            Boolean indicating if all constraints are satisfied
        """
        if A.size == 0:
            return True
        lhs = A @ x
        return bool(jnp.all(lhs + tol >= b).item())

    def _solve_projection_qp_identity_jax(
        x0: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Solve: min 0.5||x - x0||^2 s.t. A x >= b (JAX version).
        
        Uses enumeration of active sets for small problems (works well for 2D planning).
        For larger problems, consider using JAXOpt.
        
        Args:
            x0: Reference point, shape (dim,)
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            
        Returns:
            Projected point, shape (dim,)
        """
        x0 = jnp.asarray(x0, dtype=jnp.float32).flatten()
        A = jnp.asarray(A, dtype=jnp.float32)
        b = jnp.asarray(b, dtype=jnp.float32).flatten()
        m, dim = A.shape
        
        # Filter out zero rows (invalid constraints) before solving
        # Zero rows have zero norm - these are masked out constraints
        row_norms = jnp.linalg.norm(A, axis=1)
        valid_mask = row_norms > 1e-8
        
        # Mask invalid constraints by setting b to very negative (always satisfied)
        b_masked = jnp.where(valid_mask, b, jnp.full((m,), -1e6, dtype=jnp.float32))
        
        # Check if already feasible (use JAX-compatible check)
        lhs = A @ x0
        feasible = jnp.all(lhs + 1e-7 >= b_masked)
        
        def solve_qp():
            # Try using JAXOpt if available
            if JAXOPT_AVAILABLE and jaxopt is not None:
                try:
                    from jaxopt import BoxOSQP
                    
                    # Convert to standard QP form: min 0.5 x^T P x + q^T x s.t. G x <= h
                    P = jnp.eye(dim, dtype=jnp.float32)
                    q = -x0
                    G = -A  # A x >= b  =>  -A x <= -b
                    h = -b_masked
                    
                    # BoxOSQP requires box constraints, so we use large bounds
                    lower = jnp.full((dim,), -1e6, dtype=jnp.float32)
                    upper = jnp.full((dim,), 1e6, dtype=jnp.float32)
                    
                    qp = BoxOSQP()
                    sol = qp.run(P=P, q=q, G=G, h=h, lower=lower, upper=upper).params
                    return sol.primal.astype(jnp.float32)
                except Exception:
                    # Fallback to enumeration if JAXOpt fails
                    pass
            
            # Fallback: enumeration method (similar to NumPy version)
            # For JAX compatibility, we'll use a simplified approach
            # Project onto the most violated constraint as a simple solution
            violations = b_masked - (A @ x0)
            # Mask out invalid constraints
            violations_masked = jnp.where(valid_mask, violations, jnp.full((m,), -1e6, dtype=jnp.float32))
            i = jnp.argmax(violations_masked)
            a = A[i]
            denom = jnp.dot(a, a)
            denom = jnp.where(denom < 1e-12, 1e-12, denom)
            alpha = (b[i] - jnp.dot(a, x0)) / denom
            best_x = x0 + alpha * a
            
            # Verify feasibility
            lhs_check = A @ best_x
            is_feasible = jnp.all(lhs_check + 1e-7 >= b_masked)
            
            def return_projected():
                return best_x
            
            def return_original():
                return x0
            
            return jax.lax.cond(is_feasible, return_projected, return_original)
        
        def return_original():
            return x0
        
        # Use cond to handle feasibility check
        return jax.lax.cond(feasible, return_original, solve_qp)
        
        # Try using JAXOpt if available
        if JAXOPT_AVAILABLE and jaxopt is not None:
            try:
                from jaxopt import BoxOSQP
                
                # Convert to standard QP form: min 0.5 x^T P x + q^T x s.t. G x <= h
                P = jnp.eye(dim, dtype=jnp.float32)
                q = -x0
                G = -A  # A x >= b  =>  -A x <= -b
                h = -b
                
                # BoxOSQP requires box constraints, so we use large bounds
                # For unconstrained variables, use very large bounds
                lower = jnp.full((dim,), -1e6, dtype=jnp.float32)
                upper = jnp.full((dim,), 1e6, dtype=jnp.float32)
                
                qp = BoxOSQP()
                sol = qp.run(P=P, q=q, G=G, h=h, lower=lower, upper=upper).params
                return sol.primal.astype(jnp.float32)
            except Exception:
                # Fallback to enumeration if JAXOpt fails
                pass
        
        # Fallback: enumeration method (similar to NumPy version)
        best_x = None
        best_obj = jnp.inf
        
        # Active set size 1: projection onto a single hyperplane
        for i in range(m):
            a = A[i]
            denom = jnp.dot(a, a)
            denom = jnp.where(denom < 1e-12, 1e-12, denom)
            alpha = (b[i] - jnp.dot(a, x0)) / denom
            x = x0 + alpha * a
            if _is_feasible_jax(A, b, x):
                obj = jnp.sum((x - x0) ** 2)
                if best_x is None or obj < best_obj:
                    best_obj = obj
                    best_x = x
        
        # Active sets of size 2..dim
        max_k = min(dim, m)
        for k in range(2, max_k + 1):
            # Generate combinations (using JAX-compatible approach)
            # For small k, we can enumerate
            if k > 3:  # Limit enumeration to avoid combinatorial explosion
                break
            
            # Use itertools.combinations converted to JAX
            from itertools import combinations
            for idxs in combinations(range(m), k):
                AI = A[list(idxs)]
                bI = b[list(idxs)]
                rhs = bI - (AI @ x0)
                M = AI @ AI.T
                
                # Check rank (simplified: check determinant)
                det = jnp.linalg.det(M)
                if jnp.abs(det) < 1e-10:
                    continue
                
                try:
                    lam = jnp.linalg.solve(M, rhs)
                    x = x0 + (AI.T @ lam)
                    if _is_feasible_jax(A, b, x):
                        obj = jnp.sum((x - x0) ** 2)
                        if best_x is None or obj < best_obj:
                            best_obj = obj
                            best_x = x
                except Exception:
                    continue
        
        # Fallback: step along most violated constraint
        if best_x is None:
            violations = b - (A @ x0)
            i = jnp.argmax(violations)
            a = A[i]
            denom = jnp.dot(a, a)
            denom = jnp.where(denom < 1e-12, 1e-12, denom)
            alpha = (b[i] - jnp.dot(a, x0)) / denom
            best_x = x0 + alpha * a
        
        return best_x.astype(jnp.float32)

    def _project_cfs_jax(
        positions: jnp.ndarray,
        sdf_fn: Callable,
        grad_fn: Callable,
        clearance: jnp.ndarray,
        max_iterations: int = 5,
        convergence_tol: float = 1e-6,
        max_constraints_per_point: int = 8,
        constraint_margin: float = 0.25,
    ) -> jnp.ndarray:
        """
        JAX-compatible batch CFS projection.
        
        Projects multiple points onto CFS using linearized constraints and QP.
        This is the JAX version of _project_cfs_batch, designed for GPU acceleration.
        
        Note: This is a simplified implementation that processes points sequentially.
        For better performance with many points, consider using vmap or batching.
        
        Args:
            positions: Points to project, shape (N, dim) as JAX array
            sdf_fn: Function that computes SDF for all obstacles, (points) -> (M, N)
            grad_fn: Function that computes gradients, (point, obs_indices) -> (k, dim)
            clearance: Minimum clearance required (scalar or broadcastable)
            max_iterations: Maximum iterations for iterative projection
            convergence_tol: Convergence tolerance
            max_constraints_per_point: Maximum constraints per point
            constraint_margin: Margin for constraint selection
            
        Returns:
            Projected positions, shape (N, dim) as JAX array
        """
        positions = jnp.asarray(positions, dtype=jnp.float32)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)
        
        N, dim = positions.shape
        
        if N == 0:
            return positions
        
        clearance = jnp.asarray(clearance, dtype=jnp.float32)
        if clearance.ndim > 0:
            clearance = clearance[0]  # Use first element if array
        
        current = positions
        
        # Project a single point (used in loop)
        def project_single_point(x_ref: jnp.ndarray, sdf_vals: jnp.ndarray, obs_list_len: int) -> jnp.ndarray:
            """
            Project a single point using CFS.
            
            Args:
                x_ref: Reference point, shape (dim,)
                sdf_vals: SDF values for all obstacles at this point, shape (M,)
                obs_list_len: Number of obstacles (M)
                
            Returns:
                Projected point, shape (dim,)
            """
            threshold = clearance + constraint_margin
            cand_mask = sdf_vals < threshold
            
            # Select candidate obstacles (closest k)
            num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
            k = jnp.minimum(max_constraints_per_point, jnp.maximum(1, num_candidates))
            
            # Get k closest obstacles (use fixed size for JIT compatibility)
            sorted_indices = jnp.argsort(sdf_vals)
            # Take up to max_constraints_per_point indices
            cand_indices = sorted_indices[:max_constraints_per_point]
            
            # Compute gradients for candidate obstacles
            grads = grad_fn(x_ref, cand_indices)  # (max_constraints_per_point, dim)
            
            # Build linearized constraints A x >= b
            # Build constraints for all candidate indices, then select first k
            max_k = max_constraints_per_point
            
            def build_constraint(i):
                """Build constraint for obstacle at index i."""
                j = cand_indices[i]
                grad = grads[i]
                d0 = sdf_vals[j]
                
                gnorm = jnp.linalg.norm(grad)
                gnorm = jnp.where(gnorm < 1e-8, 1e-8, gnorm)
                g = grad / gnorm
                b_val = (clearance - d0) / gnorm + jnp.dot(g, x_ref)
                return g, b_val
            
            # Build all constraints using vmap
            gs, bs = jax.vmap(build_constraint)(jnp.arange(max_k))
            # gs shape: (max_k, dim), bs shape: (max_k,)
            
            # Select only the first k constraints using a mask
            # Create mask: first k are valid
            valid_mask = jnp.arange(max_k) < k
            # Mask out invalid constraints (set to very small values to make them inactive)
            # For invalid rows, set A to zero (no constraint) and b to very negative (satisfied constraint)
            A_masked = jnp.where(valid_mask[:, None], gs, jnp.zeros((max_k, dim), dtype=jnp.float32))
            b_masked = jnp.where(valid_mask, bs, jnp.full((max_k,), -1e6, dtype=jnp.float32))
            
            # Check if we have any valid constraints
            has_constraints = k > 0
            
            def solve_with_constraints():
                # Use all constraints (invalid ones are automatically satisfied due to large negative b)
                return _solve_projection_qp_identity_jax(x_ref, A_masked, b_masked)
            
            def return_original():
                return x_ref
            
            # Use cond to handle empty constraints case
            return jax.lax.cond(
                has_constraints,
                solve_with_constraints,
                return_original
            )
        
        # Iteratively linearize + solve QP (CFS outer loop)
        # Use fori_loop instead of Python for loop to be JIT-compatible
        def iteration_body(i, current_pos):
            """Single iteration of CFS projection."""
            # Compute SDF for all obstacles at all points
            sdf_matrix = sdf_fn(current_pos)  # (M, N)
            
            # Union SDF: closest obstacle
            union_sdf = jnp.min(sdf_matrix, axis=0)  # (N,)
            violating_mask = union_sdf < clearance
            
            # Check if all feasible (use jnp.all result as condition)
            all_feasible = jnp.all(~violating_mask)
            
            prev = current_pos
            
            # Project each violating point (use jnp.where to handle condition)
            def project_point(idx):
                """Project a single point if it's violating."""
                is_violating = violating_mask[idx]
                sdf_vals = sdf_matrix[:, idx]  # (M,)
                projected = project_single_point(current_pos[idx], sdf_vals, sdf_matrix.shape[0])
                # Only update if violating, otherwise keep original
                return jnp.where(is_violating, projected, current_pos[idx])
            
            # Project all points using vmap
            new_pos = jax.vmap(project_point)(jnp.arange(N))
            
            # Check convergence
            max_step = jnp.max(jnp.linalg.norm(new_pos - prev, axis=1))
            converged = max_step < convergence_tol
            
            # Final feasibility check if converged
            def check_final_feasibility(pos):
                sdf_matrix_final = sdf_fn(pos)
                union_sdf_final = jnp.min(sdf_matrix_final, axis=0)
                return jnp.all(union_sdf_final >= clearance)
            
            # If converged, check final feasibility; otherwise assume not converged
            final_feasible = jax.lax.cond(
                converged,
                check_final_feasibility,
                lambda _: jnp.array(False),
                new_pos
            )
            
            # Continue if not all feasible and not converged
            # Use jnp.logical_and for boolean operations
            should_continue = jnp.logical_and(~all_feasible, ~jnp.logical_and(converged, final_feasible))
            
            # Return new position (always update, fori_loop will handle stopping)
            return new_pos
        
        # Run iterations using fori_loop
        # Note: fori_loop always runs all iterations, but that's okay for JIT compatibility
        current = jax.lax.fori_loop(0, max_iterations, iteration_body, current)
        
        return current

else:
    # JAX not available: provide dummy functions
    def _project_cfs_jax(*args, **kwargs):
        raise RuntimeError("JAX is not available. Cannot use JAX CFS projection.")
    
    def _finite_difference_gradient_jax(*args, **kwargs):
        raise RuntimeError("JAX is not available. Cannot use JAX gradient computation.")
