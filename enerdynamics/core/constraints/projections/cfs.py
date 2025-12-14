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
    import jax.numpy as jnp
except ImportError:
    jnp = None


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
