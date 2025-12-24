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

# Import RuntimeBackendManager to detect backend
try:
    from enerdynamics.core.backends.runtime import RuntimeBackendManager
    BACKEND_MANAGER_AVAILABLE = True
except ImportError:
    BACKEND_MANAGER_AVAILABLE = False
    RuntimeBackendManager = None

# Import registry system
try:
    from enerdynamics.core.registry.projections import get_projection_registry
    REGISTRY_AVAILABLE = True
except ImportError:
    REGISTRY_AVAILABLE = False
    get_projection_registry = None

# Import backend implementations to trigger registration
try:
    from enerdynamics.core.constraints.projections import backends  # noqa: F401
except ImportError:
    pass  # Backends may not be available

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
    # Optional: JAX-based QP solver (supports general inequalities)
    import qpax

    QPAX_AVAILABLE = True
except ImportError:
    qpax = None
    QPAX_AVAILABLE = False


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
            # Backend detection: if None, auto-detect from RuntimeBackendManager
            force_python_backend: Optional[bool] = None,
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
        
        # Determine backend name from RuntimeBackendManager or force_python_backend
        backend_name = None
        if force_python_backend is None:
            if BACKEND_MANAGER_AVAILABLE and RuntimeBackendManager is not None:
                try:
                    backend = RuntimeBackendManager.get_backend()
                    backend_name = backend.name
                except Exception:
                    # If backend detection fails, default to "numpy"
                    backend_name = "numpy"
            else:
                backend_name = "numpy"  # Default to numpy if manager not available
        else:
            # force_python_backend is legacy parameter for backward compatibility
            backend_name = "numpy" if force_python_backend else "jax"
            if force_python_backend:
                print(f"[CFS] Force Python backend enabled, using NumPy implementation")
        
        # Get backend implementation from registry
        self._backend_impl = None
        if REGISTRY_AVAILABLE and get_projection_registry is not None:
            registry = get_projection_registry()
            config = {
                'max_iterations': self.max_iterations,
                'convergence_tol': self.convergence_tol,
                'max_constraints_per_point': self.max_constraints_per_point,
                'constraint_margin': self.constraint_margin,
                'use_trajectory_qp': self.use_trajectory_qp,
                'smoothness_weight': self.smoothness_weight,
            }
            
            # Try to get the requested backend
            impl_class = registry.get("cfs", backend_name)
            if impl_class is not None:
                try:
                    self._backend_impl = impl_class(obstacles, **config)
                    print(f"[CFS] Using {backend_name} backend implementation from registry")
                except Exception as e:
                    print(f"[CFS] WARNING: Failed to create {backend_name} backend implementation: {e}")
                    self._backend_impl = None
                    
                    # If JAX backend failed, try NumPy backend as fallback
                    if backend_name == "jax":
                        numpy_class = registry.get("cfs", "numpy")
                        if numpy_class is not None:
                            try:
                                self._backend_impl = numpy_class(obstacles, **config)
                                print(f"[CFS] Falling back to numpy backend implementation")
                                backend_name = "numpy"  # Update backend_name for consistency
                            except Exception as e2:
                                print(f"[CFS] WARNING: Failed to create numpy backend implementation: {e2}")
                                self._backend_impl = None
            else:
                available_backends = registry.list_backends("cfs")
                print(f"[CFS] WARNING: {backend_name} backend not found in registry. "
                      f"Available backends: {available_backends}. Falling back to legacy implementation.")
        
        # Fallback: use legacy _use_python_backend flag for backward compatibility
        if self._backend_impl is None:
            self._use_python_backend = (backend_name == "numpy")
            if self._use_python_backend:
                print(f"[CFS] Using legacy NumPy/cvxopt path (not JAX)")
                if cvx_solvers is None or cvx_matrix is None:
                    print(f"[CFS] WARNING: Python backend requested but cvxopt is not available. "
                          f"CFS will fall back to pointwise projection.")

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

        # Batch project all positions using backend implementation or legacy method
        if self._backend_impl is not None:
            # Use registry-based backend implementation
            projected_positions = self._backend_impl.project_batch(
                positions=positions,
                clearance=clearance,
                step=step,
                max_iterations=self.max_iterations,
                convergence_tol=self.convergence_tol,
                max_constraints_per_point=self.max_constraints_per_point,
                constraint_margin=self.constraint_margin,
                use_trajectory_qp=self.use_trajectory_qp,
                smoothness_weight=self.smoothness_weight,
            )
        else:
            # Fallback to legacy implementation
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

        # Check if we should use trajectory QP
        # If using Python backend (NumPy), only use cvxopt (not JAX qpax)
        # If using JAX backend, can use either cvxopt or JAX qpax (but prefer cvxopt for consistency)
        use_python_backend = getattr(self, '_use_python_backend', False)
        
        # When using Python backend, only allow cvxopt (never JAX qpax)
        can_solve_trajectory_qp = (
                bool(self.use_trajectory_qp)
                and cvx_solvers is not None
                and cvx_matrix is not None
                and cvx_spmatrix is not None
                and sp is not None
        )
        
        # If using Python backend, ensure we only use cvxopt (not JAX)
        if use_python_backend:
            # Force cvxopt path only - never use JAX qpax
            # can_solve_trajectory_qp already checks for cvxopt availability
            if can_solve_trajectory_qp:
                # Debug: confirm we're using cvxopt
                if not hasattr(self, '_python_backend_confirmed'):
                    print(f"[CFS] Using Python backend with cvxopt for trajectory QP")
                    self._python_backend_confirmed = True
        # If using JAX backend, we can use cvxopt or JAX qpax, but prefer cvxopt for consistency
        
        # Debug: Check trajectory QP availability
        if self.use_trajectory_qp:
            debug_info = {
                'use_trajectory_qp': self.use_trajectory_qp,
                'cvx_solvers': cvx_solvers is not None,
                'cvx_matrix': cvx_matrix is not None,
                'cvx_spmatrix': cvx_spmatrix is not None,
                'sp': sp is not None,
                'can_solve_trajectory_qp': can_solve_trajectory_qp,
                'N': N,
            }
            if hasattr(self, '_debug_trajectory_qp'):
                self._debug_trajectory_qp.append(debug_info)
            else:
                self._debug_trajectory_qp = [debug_info]
            if not can_solve_trajectory_qp:
                missing = [k for k, v in debug_info.items() if k != 'use_trajectory_qp' and k != 'can_solve_trajectory_qp' and k != 'N' and not v]
                print(f"[DEBUG] Trajectory QP disabled. Missing: {missing}, N={N}")

        # Iteratively linearize + solve QP (CFS outer loop)
        iteration = 0
        for _ in range(self.max_iterations):
            iteration += 1
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
                    if hasattr(self, '_debug_trajectory_qp_attempts'):
                        self._debug_trajectory_qp_attempts.append({
                            'iteration': iteration,
                            'status': 'no_active_points',
                            'active_indices_size': 0
                        })
                    break
                
                # Debug: Log trajectory QP attempt
                if hasattr(self, '_debug_trajectory_qp_attempts'):
                    self._debug_trajectory_qp_attempts.append({
                        'iteration': iteration,
                        'status': 'attempting',
                        'active_indices_size': active_indices.size,
                        'N': N
                    })
                else:
                    self._debug_trajectory_qp_attempts = [{
                        'iteration': iteration,
                        'status': 'attempting',
                        'active_indices_size': active_indices.size,
                        'N': N
                    }]

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
                    # Debug: Log success
                    if hasattr(self, '_debug_trajectory_qp_attempts'):
                        self._debug_trajectory_qp_attempts[-1]['status'] = 'success'
                        self._debug_trajectory_qp_attempts[-1]['n_cons'] = len(b_rows)
                except RuntimeError as e:
                    # Handle different types of failures differently
                    error_str = str(e).lower()
                    current = prev
                    
                    # Debug: Log failure
                    if hasattr(self, '_debug_trajectory_qp_attempts'):
                        self._debug_trajectory_qp_attempts[-1]['status'] = 'failed'
                        self._debug_trajectory_qp_attempts[-1]['error'] = str(e)
                        self._debug_trajectory_qp_attempts[-1]['error_type'] = type(e).__name__
                    
                    # Only permanently disable if it's a fundamental issue (infeasible, not numerical)
                    # For numerical issues (unknown status), allow retry in next iteration
                    if "infeasible" in error_str:
                        # Problem is fundamentally infeasible, disable trajectory QP
                        can_solve_trajectory_qp = False
                        print(f"[DEBUG] Trajectory QP infeasible at iteration {iteration}, disabling trajectory QP")
                    elif "unknown" in error_str or "numerical" in error_str:
                        # Numerical issue, don't permanently disable - allow retry next iteration
                        # Just fall back to pointwise for this iteration
                        print(f"[DEBUG] Trajectory QP numerical issue at iteration {iteration}: {e}")
                        print(f"[DEBUG]   Falling back to pointwise for this iteration, will retry trajectory QP next iteration")
                        print(f"[DEBUG]   n_vars={N * dim}, n_cons={len(b_rows)}, active_indices={active_indices.size}")
                    else:
                        # Other errors: be conservative and disable
                        can_solve_trajectory_qp = False
                        print(f"[DEBUG] Trajectory QP failed at iteration {iteration}: {type(e).__name__}: {e}")
                        print(f"[DEBUG]   Disabling trajectory QP")
                except Exception as e:
                    # Other unexpected exceptions: disable trajectory QP
                    current = prev
                    can_solve_trajectory_qp = False
                    # Debug: Log failure
                    if hasattr(self, '_debug_trajectory_qp_attempts'):
                        self._debug_trajectory_qp_attempts[-1]['status'] = 'failed'
                        self._debug_trajectory_qp_attempts[-1]['error'] = str(e)
                        self._debug_trajectory_qp_attempts[-1]['error_type'] = type(e).__name__
                    print(f"[DEBUG] Trajectory QP exception at iteration {iteration}: {type(e).__name__}: {e}")
                    print(f"[DEBUG]   Disabling trajectory QP")

            if not can_solve_trajectory_qp:
                # Debug: Log pointwise fallback
                if self.use_trajectory_qp and iteration == 1:
                    print(f"[DEBUG] Using pointwise projection (iteration {iteration})")
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
        
        # Check for numerical issues in constraints
        if n_cons > 0 and n_vars > 0:
            A_dense = A_sp.toarray()
            b_arr = np.asarray(b, dtype=np.float64)
            # Check for non-finite values
            if np.any(~np.isfinite(A_dense)) or np.any(~np.isfinite(b_arr)):
                raise RuntimeError("Non-finite values in QP constraints")
        
        G_sp = (-A_sp).tocoo()
        G_cvx = self._coo_to_cvx_spmatrix(G_sp.data, G_sp.row, G_sp.col, G_sp.shape)
        h_cvx = cvx_matrix((-np.asarray(b, dtype=np.float64)).reshape(-1))

        # Solve QP with improved numerical stability options
        old_show = cvx_solvers.options.get("show_progress", True)
        old_maxiters = cvx_solvers.options.get("maxiters", 100)
        old_abstol = cvx_solvers.options.get("abstol", 1e-7)
        old_reltol = cvx_solvers.options.get("reltol", 1e-6)
        old_feastol = cvx_solvers.options.get("feastol", 1e-7)
        old_refinement = cvx_solvers.options.get("refinement", 1)
        
        # Set improved options for better numerical stability
        cvx_solvers.options["show_progress"] = False
        cvx_solvers.options["maxiters"] = 200  # Increase max iterations
        cvx_solvers.options["abstol"] = 1e-6  # Slightly relax absolute tolerance
        cvx_solvers.options["reltol"] = 1e-5  # Slightly relax relative tolerance
        cvx_solvers.options["feastol"] = 1e-6  # Slightly relax feasibility tolerance
        cvx_solvers.options["refinement"] = 2  # Increase refinement iterations
        
        try:
            sol = cvx_solvers.qp(P_cvx, q_cvx, G_cvx, h_cvx)
        finally:
            # Restore original options
            cvx_solvers.options["show_progress"] = old_show
            cvx_solvers.options["maxiters"] = old_maxiters
            cvx_solvers.options["abstol"] = old_abstol
            cvx_solvers.options["reltol"] = old_reltol
            cvx_solvers.options["feastol"] = old_feastol
            cvx_solvers.options["refinement"] = old_refinement

        if sol is None:
            raise RuntimeError("cvxopt qp failed: solver returned None")
        
        status = sol.get("status", "")
        
        # Handle "unknown" status: try to use the solution if it exists and is finite
        if status == "unknown":
            if "x" in sol and sol["x"] is not None:
                x_candidate = np.asarray(sol["x"], dtype=np.float64).reshape(-1)
                # If solution is finite, use it even if status is unknown
                if np.all(np.isfinite(x_candidate)):
                    return x_candidate
            # If no valid solution, raise error with informative message
            raise RuntimeError(
                f"cvxopt qp failed: status=unknown. "
                f"This may indicate numerical issues. "
                f"Consider using pointwise projection or checking constraint conditioning."
            )
        
        if status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError(f"cvxopt qp failed: status={status}")
        
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

        # Try per-obstacle jax_sdf/jax_gradient first (most accurate, aligns with NumPy version)
        # If that fails (e.g., UnionObstacle doesn't support JAX), fall back to SDF texture
        use_manager_sdf = False  # First try per-obstacle method

        # Try to build JAX-compatible SDF and gradient functions
        try:
            sdf_fn, grad_fn = self._build_jax_obstacle_functions(obstacles_list, use_manager_sdf=use_manager_sdf)
        except RuntimeError as e:
            # If per-obstacle method fails, try using SDF texture (if available)
            # This handles cases like UnionObstacle that don't have jax_sdf but can use texture
            if (hasattr(self.obstacles, 'sample_sdf_and_grad_2d') and
                    hasattr(self.obstacles, 'get_sdf_texture_2d') and
                    self.obstacles.get_sdf_texture_2d() is not None):
                try:
                    use_manager_sdf = True
                    sdf_fn, grad_fn = self._build_jax_obstacle_functions(obstacles_list,
                                                                         use_manager_sdf=use_manager_sdf)
                except RuntimeError as e2:
                    # Both methods failed - fall back to NumPy version
                    import warnings
                    warnings.warn(
                        f"JAX CFS projection not available: {str(e2)}. "
                        "Falling back to NumPy version (Python loop). "
                        "To use JAX acceleration, ensure all obstacles support JAX "
                        "(e.g., use SDF texture with build_sdf_texture_2d()).",
                        UserWarning
                    )
                    return None
            else:
                # No SDF texture available - fall back to NumPy version
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
        use_traj_qp = self.use_trajectory_qp
        smooth_weight = self.smoothness_weight

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
                use_trajectory_qp=use_traj_qp,
                smoothness_weight=smooth_weight,
            )

        # Warmup: Pre-compile the JAX function to avoid first-call slowdown
        # Use dummy inputs with typical shapes (typical trajectory: 21 points for horizon=20, 2D)
        try:
            dummy_positions = jnp.zeros((21, 2), dtype=jnp.float32)
            dummy_clearance = jnp.asarray(0.1, dtype=jnp.float32)
            # Trigger compilation (block_until_ready ensures compilation completes)
            _ = jax_project(dummy_positions, dummy_clearance).block_until_ready()
        except Exception:
            # Warmup failed, but that's okay - will compile on first real use
            pass

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
        # BUT: For alignment with NumPy version, we need per-obstacle SDF values for candidate selection
        # So we compute per-obstacle SDF even when using texture for gradients
        if use_manager_sdf:
            # Check if obstacles support JAX (needed for per-obstacle SDF)
            all_support_jax = all(hasattr(obs, "jax_sdf") for obs in obstacles_list)

            if all_support_jax:
                # Use per-obstacle SDF (for candidate selection) + direct gradients (for alignment)
                def sdf_batch(points: jnp.ndarray) -> jnp.ndarray:
                    """
                    Compute SDF for all obstacles (matching NumPy version for candidate selection).
                    
                    Args:
                        points: Shape (N, dim)
                        
                    Returns:
                        SDF matrix, shape (M, N) where M is number of obstacles
                    """
                    sdf_rows = []
                    for obs in obstacles_list:
                        if hasattr(obs, "jax_sdf"):
                            sdf_vals = obs.jax_sdf(points)
                        else:
                            raise RuntimeError(f"Obstacle {type(obs).__name__} does not support JAX")

                        # Ensure shape is (N,)
                        sdf_vals = jnp.asarray(sdf_vals, dtype=jnp.float32)
                        if sdf_vals.ndim == 0:
                            # Scalar: broadcast to (N,)
                            sdf_vals = jnp.broadcast_to(sdf_vals, (points.shape[0],))
                        elif sdf_vals.ndim > 1:
                            sdf_vals = sdf_vals.flatten()[:points.shape[0]]
                        elif sdf_vals.shape[0] != points.shape[0]:
                            # Shape mismatch: pad or truncate
                            if sdf_vals.shape[0] < points.shape[0]:
                                # Pad with last value
                                last_val = sdf_vals[-1] if sdf_vals.shape[0] > 0 else jnp.array(1e6, dtype=jnp.float32)
                                padding = jnp.full((points.shape[0] - sdf_vals.shape[0],), last_val, dtype=jnp.float32)
                                sdf_vals = jnp.concatenate([sdf_vals, padding])
                            else:
                                # Truncate
                                sdf_vals = sdf_vals[:points.shape[0]]

                        sdf_rows.append(sdf_vals)

                    if not sdf_rows:
                        return jnp.full((1, points.shape[0]), 1e6, dtype=jnp.float32)

                    return jnp.stack(sdf_rows, axis=0)  # (M, N)

                def grad_multiple(point: jnp.ndarray, obs_indices: jnp.ndarray) -> jnp.ndarray:
                    """
                    Compute gradients for specific obstacles.
                    Since obstacles support JAX, we use JAX operations directly for optimal performance.
                    
                    Args:
                        point: Shape (dim,)
                        obs_indices: Indices of obstacles, shape (k,)
                        
                    Returns:
                        Gradients, shape (k, dim)
                    """
                    num_obstacles = len(obstacles_list)

                    # Pre-build gradient functions for each obstacle (capture obstacles in closure)
                    # This allows us to use JAX directly without pure_callback
                    def make_grad_fn_for_obs(obstacle):
                        """Create a gradient function for a specific obstacle."""
                        if hasattr(obstacle, "jax_gradient"):
                            # Use JAX-native gradient method
                            def grad_fn(p):
                                return obstacle.jax_gradient(p)

                            return grad_fn
                        else:
                            # Use finite differences with JAX
                            def grad_fn(p):
                                return _finite_difference_gradient_jax(obstacle, p)

                            return grad_fn

                    grad_fns = [make_grad_fn_for_obs(obs) for obs in obstacles_list]

                    # Create a function that selects the appropriate gradient function based on index
                    def compute_grad_for_idx(idx: jnp.ndarray) -> jnp.ndarray:
                        """Compute gradient for obstacle at index idx using JAX switch."""
                        # Use jax.lax.switch to select the gradient function
                        # Note: idx must be within [0, num_obstacles)
                        idx_int = jnp.clip(jnp.asarray(idx, dtype=jnp.int32), 0, num_obstacles - 1)

                        # Build branches for switch (one per obstacle)
                        branches = tuple(grad_fns)

                        # Use switch to select the appropriate gradient function
                        grad = jax.lax.switch(idx_int, branches, point)
                        return jnp.asarray(grad, dtype=jnp.float32).flatten()

                    # Compute gradients for all indices using vmap
                    grads = jax.vmap(compute_grad_for_idx)(obs_indices)
                    return grads

                return sdf_batch, grad_multiple
            else:
                # Fallback: compute per-obstacle SDF using pure_callback (for candidate selection)
                # This ensures candidate selection matches NumPy version even when obstacles don't support JAX
                def sdf_batch(points: jnp.ndarray) -> jnp.ndarray:
                    """
                    Compute SDF for all obstacles (matching NumPy version for candidate selection).
                    Uses pure_callback to call NumPy sdf methods.
                    """
                    num_obstacles = len(obstacles_list)
                    num_points = points.shape[0]

                    # Compute SDF for each obstacle at each point using pure_callback
                    sdf_rows = []
                    for obs_idx in range(num_obstacles):
                        obs = obstacles_list[obs_idx]

                        def numpy_sdf_batch(pts):
                            """NumPy SDF computation for batch of points."""
                            return np.asarray([obs.sdf(p) for p in pts], dtype=np.float32)

                        # Use pure_callback to call NumPy sdf from JAX (vectorized)
                        sdf_vals = jax.pure_callback(
                            numpy_sdf_batch,
                            jax.ShapeDtypeStruct((num_points,), jnp.float32),
                            points,
                            vectorized=True
                        )
                        sdf_rows.append(sdf_vals)

                    if not sdf_rows:
                        return jnp.full((1, num_points), 1e6, dtype=jnp.float32)

                    return jnp.stack(sdf_rows, axis=0)  # (M, N)

                def grad_multiple(point: jnp.ndarray, obs_indices: jnp.ndarray) -> jnp.ndarray:
                    """
                    Compute gradients for specific obstacles using finite differences with pure_callback.
                    This matches NumPy version's gradient computation.
                    """
                    num_indices = obs_indices.shape[0]
                    dim = point.shape[0]

                    # Create a function that computes all gradients at once (avoids tracer issues)
                    def compute_all_grads(args):
                        """Compute gradients for all obstacles (runs in Python, not JIT)."""
                        point_np, indices_np = args
                        point_np = np.asarray(point_np, dtype=np.float32)
                        indices_np = np.asarray(indices_np, dtype=np.int32)

                        grads_list = []
                        for idx in indices_np:
                            obs = obstacles_list[int(idx)]
                            grad_np = np.zeros((dim,), dtype=np.float32)

                            for k in range(dim):
                                xp = point_np.copy()
                                xm = point_np.copy()
                                xp[k] += 1e-4
                                xm[k] -= 1e-4
                                dp = float(obs.sdf(xp))
                                dm = float(obs.sdf(xm))
                                grad_np[k] = (dp - dm) / (2.0 * 1e-4)

                            grads_list.append(grad_np)

                        return np.stack(grads_list, axis=0) if grads_list else np.zeros((0, dim), dtype=np.float32)

                    # Use pure_callback to compute all gradients at once
                    grads = jax.pure_callback(
                        compute_all_grads,
                        jax.ShapeDtypeStruct((num_indices, dim), jnp.float32),
                        (point, obs_indices),
                        vectorized=False
                    )

                    return grads

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
                sdf_vals = jnp.asarray(sdf_vals, dtype=jnp.float32)
                if sdf_vals.ndim == 0:
                    # Scalar: broadcast to (N,)
                    sdf_vals = jnp.broadcast_to(sdf_vals, (points.shape[0],))
                elif sdf_vals.ndim > 1:
                    sdf_vals = sdf_vals.flatten()[:points.shape[0]]
                elif sdf_vals.shape[0] != points.shape[0]:
                    # Shape mismatch: pad or truncate
                    if sdf_vals.shape[0] < points.shape[0]:
                        # Pad with last value
                        last_val = sdf_vals[-1] if sdf_vals.shape[0] > 0 else jnp.array(1e6, dtype=jnp.float32)
                        padding = jnp.full((points.shape[0] - sdf_vals.shape[0],), last_val, dtype=jnp.float32)
                        sdf_vals = jnp.concatenate([sdf_vals, padding])
                    else:
                        # Truncate
                        sdf_vals = sdf_vals[:points.shape[0]]

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
            Since obstacles support JAX, we use JAX operations directly for optimal performance.
            
            Args:
                point: Shape (dim,)
                obs_indices: Indices of obstacles, shape (k,)
                
            Returns:
                Gradients, shape (k, dim)
            """
            num_obstacles = len(obstacles_list)

            # Pre-build gradient functions for each obstacle (capture obstacles in closure)
            # This allows us to use JAX directly without pure_callback
            def make_grad_fn_for_obs(obstacle):
                """Create a gradient function for a specific obstacle."""
                if hasattr(obstacle, "jax_gradient"):
                    # Use JAX-native gradient method
                    def grad_fn(p):
                        return obstacle.jax_gradient(p)

                    return grad_fn
                else:
                    # Use finite differences with JAX
                    def grad_fn(p):
                        return _finite_difference_gradient_jax(obstacle, p)

                    return grad_fn

            grad_fns = [make_grad_fn_for_obs(obs) for obs in obstacles_list]

            # Create a function that selects the appropriate gradient function based on index
            def compute_grad_for_idx(idx: jnp.ndarray) -> jnp.ndarray:
                """Compute gradient for obstacle at index idx using JAX switch."""
                # Use jax.lax.switch to select the gradient function
                # Note: idx must be within [0, num_obstacles)
                idx_int = jnp.clip(jnp.asarray(idx, dtype=jnp.int32), 0, num_obstacles - 1)

                # Build branches for switch (one per obstacle)
                branches = tuple(grad_fns)

                # Use switch to select the appropriate gradient function
                grad = jax.lax.switch(idx_int, branches, point)
                return jnp.asarray(grad, dtype=jnp.float32).flatten()

            # Compute gradients for all indices using vmap
            grads = jax.vmap(compute_grad_for_idx)(obs_indices)
            return grads

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


    def _solve_trajectory_qp_identity_jax(
            x0: jnp.ndarray,
            A_data: jnp.ndarray,
            A_row: jnp.ndarray,
            A_col: jnp.ndarray,
            b: jnp.ndarray,
            n_vars: int,
            n_cons: int,
            dim: int,
            T: int,
            smoothness_weight: float,
    ) -> jnp.ndarray:
        """
        Solve trajectory-level CFS-QP using JAX-compatible QP solver (qpax):
        
            minimize   0.5 ||x - x0||^2 + 0.5 * w * ||D2 x||^2
            subject to A x >= b
        
        where x is the flattened trajectory [p0, p1, ..., p_{T-1}] and D2 is a
        second-difference operator along time (applied per dimension).
        
        Args:
            x0: Flattened reference trajectory, shape (n_vars,)
            A_data, A_row, A_col: Sparse constraint matrix in COO format
            b: Constraint vector, shape (n_cons,)
            n_vars: Number of variables (T * dim)
            n_cons: Number of constraints
            dim: Dimension of each point
            T: Number of time steps
            smoothness_weight: Weight for smoothness regularization
            
        Returns:
            Flattened projected trajectory, shape (n_vars,)
        """
        if not QPAX_AVAILABLE or qpax is None:
            raise RuntimeError("qpax is required for JAX trajectory QP but not available")
        
        x0 = jnp.asarray(x0, dtype=jnp.float32).flatten()
        b = jnp.asarray(b, dtype=jnp.float32).flatten()
        
        if n_cons <= 0:
            return x0
        
        # Build Hessian: I + w * (D2^T D2) kron I_dim
        # For identity case (w=0), P = I
        w = jnp.asarray(max(0.0, smoothness_weight), dtype=jnp.float32)
        
        if w > 0.0 and T >= 3:
            # Build second-difference operator D2: (T-2) x T
            # D2[i, t] = 1 if t==i, -2 if t==i+1, 1 if t==i+2, else 0
            def build_D2_row(i):
                """Build row i of D2 matrix."""
                row = jnp.zeros(T, dtype=jnp.float32)
                row = row.at[i].set(1.0)
                row = row.at[i+1].set(-2.0)
                row = row.at[i+2].set(1.0)
                return row
            
            D2 = jax.vmap(build_D2_row)(jnp.arange(T - 2))  # (T-2, T)
            K = D2.T @ D2  # (T, T)
            P_time = jnp.eye(T, dtype=jnp.float32) + w * K
            
            # Kronecker product: P_time kron I_dim
            P = jnp.kron(P_time, jnp.eye(dim, dtype=jnp.float32))  # (n_vars, n_vars)
        else:
            # Identity Hessian
            P = jnp.eye(n_vars, dtype=jnp.float32)
        
        # Build constraint matrix A from sparse format (COO)
        # JAX doesn't have sparse matrices, so we build dense A
        # Use scatter-add pattern to accumulate values
        A = jnp.zeros((n_cons, n_vars), dtype=jnp.float32)
        
        # Scatter-add: for each (row, col, value) triple, add value to A[row, col]
        # Use a loop to accumulate (JAX-compatible)
        def scatter_entry(carry, idx):
            """Scatter entry at index idx into A matrix."""
            A_mat = carry
            row = A_row[idx]
            col = A_col[idx]
            val = A_data[idx]
            # Use index_add to accumulate (handles duplicate entries)
            A_mat = A_mat.at[row, col].add(val)
            return A_mat
        
        # Process all entries using scan
        A = jax.lax.scan(
            scatter_entry,
            A,
            jnp.arange(len(A_data))
        )[0]
        
        # Convert to qpax format: min 0.5 x^T P x + q^T x s.t. G x <= h
        # Original: min 0.5||x-x0||^2 + 0.5*w*||D2 x||^2 = 0.5 x^T P x - x0^T x + const
        Q = P
        q = -x0  # Linear term
        G = -A  # A x >= b  =>  -A x <= -b
        h = -b
        
        # No equality constraints
        A_eq = jnp.zeros((0, n_vars), dtype=jnp.float32)
        b_eq = jnp.zeros((0,), dtype=jnp.float32)
        
        # Solve with qpax
        try:
            x, s, z, y, converged, iters = qpax.solve_qp(
                Q, q, A_eq, b_eq, G, h,
                solver_tol=1e-7,  # Tight tolerance for accuracy
                max_iter=200,  # Increased iterations
            )
            
            if not converged:
                # If not converged, try with looser tolerance
                x, s, z, y, converged, iters = qpax.solve_qp(
                    Q, q, A_eq, b_eq, G, h,
                    solver_tol=1e-6,
                    max_iter=200,
                )
            
            if not converged:
                raise RuntimeError(f"qpax trajectory QP did not converge after {iters} iterations")
            
            return x.astype(jnp.float32)
        except Exception as e:
            raise RuntimeError(f"qpax trajectory QP failed: {e}")

    def _solve_projection_qp_identity_jax(
            x0: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Solve: min 0.5||x - x0||^2 s.t. A x >= b (JAX version).
        
        Priority:
        1. Try qpax (JAX-compatible QP solver, supports general inequalities)
        2. Fallback to optimized JAX enumeration method (fast for small problems)
        
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
        # Align with NumPy: completely mask invalid rows (A=0, b=-1e6) to avoid qpax seeing ill-conditioned constraints
        row_norms = jnp.linalg.norm(A, axis=1)
        valid_mask = row_norms > 1e-8
        # Mask A: set invalid rows to zero
        A_masked = jnp.where(valid_mask[:, None], A, jnp.zeros((m, dim), dtype=jnp.float32))
        # Mask b: set invalid rows to very negative (satisfied constraint)
        b_masked = jnp.where(valid_mask, b, jnp.full((m,), -1e6, dtype=jnp.float32))

        # Check if already feasible
        lhs = A_masked @ x0
        feasible = jnp.all(lhs + 1e-7 >= b_masked)

        def solve_qp():
            # Try qpax first (supports general inequalities, fully JAX-compatible)
            num_valid = jnp.sum(valid_mask.astype(jnp.int32))

            # Only try qpax if we have valid constraints
            def try_qpax():
                if QPAX_AVAILABLE and qpax is not None:
                    try:
                        # Convert to qpax format: min 0.5 x^T Q x + q^T x s.t. G x <= h
                        # Use masked A to avoid ill-conditioned constraints
                        Q = jnp.eye(dim, dtype=jnp.float32)
                        q = -x0
                        G = -A_masked  # A x >= b  =>  -A x <= -b, use masked A
                        h = -b_masked

                        # No equality constraints - use empty arrays instead of None
                        A_eq = jnp.zeros((0, dim), dtype=jnp.float32)  # Empty matrix
                        b_eq = jnp.zeros((0,), dtype=jnp.float32)  # Empty vector

                        # Use tighter tolerance to match NumPy enumeration accuracy
                        # Increased max iterations for better convergence
                        x, s, z, y, converged, iters = qpax.solve_qp(
                            Q, q, A_eq, b_eq, G, h,
                            solver_tol=1e-7,  # Tighter tolerance to match NumPy precision
                            max_iter=200,  # Increase max iterations for better convergence
                        )
                        # Return result and convergence flag (ensure converged is JAX bool)
                        converged_bool = jnp.asarray(converged, dtype=jnp.bool_)
                        return x.astype(jnp.float32), converged_bool
                    except Exception:
                        # If qpax fails, return failure with JAX bool
                        return x0, jnp.array(False, dtype=jnp.bool_)
                # If qpax not available, return failure with JAX bool
                return x0, jnp.array(False, dtype=jnp.bool_)

            def skip_qpax():
                return x0, jnp.array(False, dtype=jnp.bool_)

            # Try qpax only if we have valid constraints
            qpax_result, qpax_converged = jax.lax.cond(
                num_valid > 0,
                try_qpax,
                skip_qpax
            )

            # Use qpax result if converged, otherwise use enumeration
            def use_qpax():
                return qpax_result

            def use_enumeration():
                return _solve_projection_qp_enumeration_jax(x0, A_masked, b_masked, valid_mask)

            result = jax.lax.cond(qpax_converged, use_qpax, use_enumeration)
            return result

        def return_original():
            return x0

        return jax.lax.cond(feasible, return_original, solve_qp)


    def _solve_projection_qp_enumeration_jax(
            x0: jnp.ndarray, A: jnp.ndarray, b: jnp.ndarray, valid_mask: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Optimized JAX enumeration method for QP projection.
        
        This is a JAX-compatible version of the NumPy enumeration method,
        optimized for small problems (2D, few constraints).
        
        Args:
            x0: Reference point, shape (dim,)
            A: Constraint matrix, shape (m, dim)
            b: Constraint vector, shape (m,)
            valid_mask: Boolean mask for valid constraints, shape (m,)
            
        Returns:
            Projected point, shape (dim,)
        """
        m, dim = A.shape

        # Active set size 1: projection onto single hyperplane
        def project_single(i):
            """Project onto constraint i."""
            a = A[i]
            denom = jnp.dot(a, a)
            denom = jnp.where(denom < 1e-12, 1e-12, denom)
            alpha = (b[i] - jnp.dot(a, x0)) / denom
            x = x0 + alpha * a

            # Check feasibility
            lhs_check = A @ x
            is_feas = jnp.all(lhs_check + 1e-7 >= b)
            obj = jnp.sum((x - x0) ** 2)
            # Return large objective if infeasible
            obj = jnp.where(is_feas, obj, jnp.inf)
            return obj, x

        # Vectorize over all constraints
        objs, xs = jax.vmap(project_single)(jnp.arange(m))
        best_idx = jnp.argmin(objs)
        best_x = xs[best_idx]
        best_obj = objs[best_idx]

        # For 2D/3D problems, enumerate all pairs (and triples for 3D) of constraints
        # This matches NumPy version's full enumeration (combinations(range(m), k))
        # Since max_constraints_per_point is typically small (<=8), this is affordable
        max_k = min(dim, m)

        # Enumerate pairs (k=2) - full enumeration matching NumPy version
        # Since m is typically small (<=8 from max_constraints_per_point), we use a fixed-size approach
        # Generate all pairs (i, j) where i < j using JAX-compatible method
        if max_k >= 2 and m >= 2:
            # Use a fixed maximum number of pairs (for m=8, we have 28 pairs; use 50 as safe upper bound)
            max_pairs = 50
            max_m = 10  # Maximum expected m

            def check_pair(i, j):
                """Check a pair of constraints (i, j)."""
                # Skip if not a valid pair (i < j) or out of bounds
                valid = jnp.logical_and(i < m, jnp.logical_and(j < m, i < j))
                valid = jnp.logical_and(valid, valid_mask[i] & valid_mask[j])

                AI = A[jnp.array([i, j])]
                bI = b[jnp.array([i, j])]
                rhs = bI - (AI @ x0)
                M = AI @ AI.T
                det = jnp.linalg.det(M)

                def solve_pair():
                    lam = jnp.linalg.solve(M, rhs)
                    x = x0 + (AI.T @ lam)
                    lhs_check = A @ x
                    is_feas = jnp.all(lhs_check + 1e-7 >= b)
                    obj = jnp.sum((x - x0) ** 2)
                    return jnp.where(is_feas, obj, jnp.inf), x

                def return_inf():
                    return jnp.inf, x0

                obj, x = jax.lax.cond(
                    jnp.logical_and(valid, jnp.abs(det) > 1e-10),
                    solve_pair,
                    return_inf
                )
                return obj, x

            # Generate all pairs using nested vmap with fixed shape
            # For each i, check all j (j > i will be valid, others will be masked)
            def check_pairs_for_i(i):
                """Check all pairs starting with i."""
                # Use fixed-size array: check j from 0 to max_m-1, but only j > i is valid
                js = jnp.arange(max_m)
                objs, xs = jax.vmap(lambda j: check_pair(i, j))(js)
                return objs, xs

            # Check all pairs for all i
            i_indices = jnp.arange(max_m)
            all_pair_objs, all_pair_xs = jax.vmap(check_pairs_for_i)(i_indices)
            # Flatten results: shape (max_m, max_m) -> (max_m*max_m,)
            all_pair_objs_flat = all_pair_objs.flatten()
            all_pair_xs_flat = all_pair_xs.reshape(-1, dim)

            # Update best solution (invalid pairs will have obj=inf, so they won't be selected)
            best_pair_idx = jnp.argmin(all_pair_objs_flat)
            best_pair_obj = all_pair_objs_flat[best_pair_idx]
            best_pair_x = all_pair_xs_flat[best_pair_idx]

            best_x = jnp.where(best_pair_obj < best_obj, best_pair_x, best_x)
            best_obj = jnp.minimum(best_obj, best_pair_obj)

        # For 3D problems, also enumerate triples (k=3) - matching NumPy version
        # For 2D (dim=2), pairs are sufficient, but we can still enumerate triples for completeness
        # Since m is typically small (<=8 from max_constraints_per_point), we can enumerate triples
        if max_k >= 3 and m >= 3 and dim >= 3:
            # Enumerate triples (k=3) for 3D problems
            # Use fixed-size approach for JAX compatibility
            max_triples = 120  # C(10,3) = 120, safe upper bound
            max_m_triples = 10
            
            def check_triple(i, j, k_idx):
                """Check a triple of constraints (i, j, k) where i < j < k."""
                # Generate k from k_idx (k_idx represents position in sequence)
                # For each (i,j) pair, k can be j+1, j+2, ..., m-1
                k = j + 1 + k_idx
                valid = jnp.logical_and(
                    i < m,
                    jnp.logical_and(
                        j < m,
                        jnp.logical_and(
                            k < m,
                            jnp.logical_and(
                                i < j,
                                j < k
                            )
                        )
                    )
                )
                valid = jnp.logical_and(valid, valid_mask[i] & valid_mask[j])
                valid = jnp.logical_and(valid, valid_mask[k] if k < m else False)
                
                AI = A[jnp.array([i, j, k])]
                bI = b[jnp.array([i, j, k])]
                rhs = bI - (AI @ x0)
                M = AI @ AI.T
                det = jnp.linalg.det(M)
                
                def solve_triple():
                    lam = jnp.linalg.solve(M, rhs)
                    x = x0 + (AI.T @ lam)
                    lhs_check = A @ x
                    is_feas = jnp.all(lhs_check + 1e-7 >= b)
                    obj = jnp.sum((x - x0) ** 2)
                    return jnp.where(is_feas, obj, jnp.inf), x
                
                def return_inf():
                    return jnp.inf, x0
                
                obj, x = jax.lax.cond(
                    jnp.logical_and(valid, jnp.abs(det) > 1e-10),
                    solve_triple,
                    return_inf
                )
                return obj, x
            
            # Enumerate all triples: for each (i,j) pair, check all k > j
            def check_triples_for_pair(i, j):
                """Check all triples starting with (i, j)."""
                max_k_for_pair = max_m_triples - j - 1  # k can be j+1 to max_m-1
                k_indices = jnp.arange(max_m_triples)
                objs, xs = jax.vmap(lambda k_idx: check_triple(i, j, k_idx))(k_indices)
                return objs, xs
            
            # For each pair (i, j), check triples
            def check_triples_for_i(i):
                """Check all triples starting with i."""
                js = jnp.arange(max_m_triples)
                all_objs, all_xs = jax.vmap(lambda j: check_triples_for_pair(i, j))(js)
                # Flatten: shape (max_m, max_m) -> (max_m*max_m,)
                return all_objs.flatten(), all_xs.reshape(-1, dim)
            
            i_indices = jnp.arange(max_m_triples)
            all_triple_objs, all_triple_xs = jax.vmap(check_triples_for_i)(i_indices)
            # Flatten further: shape (max_m, max_m*max_m) -> (max_m*max_m*max_m,)
            all_triple_objs_flat = all_triple_objs.flatten()
            all_triple_xs_flat = all_triple_xs.reshape(-1, dim)
            
            # Update best solution
            best_triple_idx = jnp.argmin(all_triple_objs_flat)
            best_triple_obj = all_triple_objs_flat[best_triple_idx]
            best_triple_x = all_triple_xs_flat[best_triple_idx]
            
            best_x = jnp.where(best_triple_obj < best_obj, best_triple_x, best_x)
            best_obj = jnp.minimum(best_obj, best_triple_obj)

        # Final fallback: step along most violated constraint
        violations = b - (A @ x0)
        violations_masked = jnp.where(valid_mask, violations, jnp.full((m,), -1e6, dtype=jnp.float32))
        i_max = jnp.argmax(violations_masked)
        a = A[i_max]
        denom = jnp.dot(a, a)
        denom = jnp.where(denom < 1e-12, 1e-12, denom)
        alpha = (b[i_max] - jnp.dot(a, x0)) / denom
        fallback_x = x0 + alpha * a

        # Use best solution found
        return jnp.where(best_obj < jnp.inf, best_x, fallback_x).astype(jnp.float32)


    def _project_cfs_jax(
            positions: jnp.ndarray,
            sdf_fn: Callable,
            grad_fn: Callable,
            clearance: jnp.ndarray,
            max_iterations: int = 5,
            convergence_tol: float = 1e-6,
            max_constraints_per_point: int = 8,
            constraint_margin: float = 0.25,
            use_trajectory_qp: bool = False,
            smoothness_weight: float = 0.0,
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
            
            This matches the NumPy version logic:
            1. First check for candidates within (clearance + margin)
            2. If candidates exist, select k closest from candidates
            3. If no candidates, fallback to k closest from all obstacles
            
            Args:
                x_ref: Reference point, shape (dim,)
                sdf_vals: SDF values for all obstacles at this point, shape (M,)
                obs_list_len: Number of obstacles (M)
                
            Returns:
                Projected point, shape (dim,)
            """
            threshold = clearance + constraint_margin
            cand_mask = sdf_vals < threshold
            num_candidates = jnp.sum(cand_mask.astype(jnp.int32))

            # Align with NumPy version: if candidates exist, use them; otherwise fallback to all
            sorted_indices = jnp.argsort(sdf_vals)

            def use_candidates():
                """Select k closest from candidates (matching NumPy line 382-391).
                
                NumPy logic:
                1. cand_indices = np.where(cand_mask)[0]  # Get all candidates
                2. cand_indices = cand_indices[np.argsort(d0_all[cand_indices])[:k]]  # Sort candidates by SDF, take k closest
                """
                # Step 1: Collect all candidate indices (those within threshold)
                # Use a fixed-size array to store candidates
                cand_all = jnp.zeros(obs_list_len, dtype=jnp.int32)  # Max possible candidates
                cand_count = 0

                def collect_cand(carry, i):
                    """Collect candidate index i if it's in cand_mask."""
                    cand_all, cand_count = carry
                    is_candidate = cand_mask[i]
                    # Add to list if candidate
                    cand_all = jnp.where(
                        jnp.arange(obs_list_len) == cand_count,
                        jnp.where(is_candidate, i, cand_all[cand_count]),
                        cand_all
                    )
                    cand_count = jnp.where(is_candidate, cand_count + 1, cand_count)
                    return (cand_all, cand_count), None

                # Collect all candidates
                initial_state = (cand_all, 0)
                (cand_all, total_candidates), _ = jax.lax.scan(
                    collect_cand,
                    initial_state,
                    jnp.arange(obs_list_len)
                )

                # Step 2: Sort candidates by their SDF values and take k closest
                # Get SDF values for all candidates (matching NumPy: d0_all[cand_indices])
                def get_cand_sdf(i):
                    """Get SDF value for candidate at position i."""
                    cand_idx = cand_all[i]
                    # Only valid candidates (i < total_candidates) have real SDF values
                    return jnp.where(i < total_candidates, sdf_vals[cand_idx], jnp.inf)

                cand_sdf_vals = jax.vmap(get_cand_sdf)(jnp.arange(obs_list_len))

                # Sort candidates by SDF (matching NumPy: np.argsort(d0_all[cand_indices]))
                # argsort will put inf values last, so first total_candidates indices are valid
                cand_sorted_by_sdf = jnp.argsort(cand_sdf_vals)

                # Take k closest (matching NumPy: [:k])
                k_val = jnp.minimum(max_constraints_per_point, total_candidates)

                # Extract k closest candidate indices (matching NumPy: cand_indices[sorted_indices[:k]])
                # Since argsort puts inf values last, first total_candidates indices in cand_sorted_by_sdf
                # correspond to valid candidates, so we can safely use them
                def get_sorted_cand(i):
                    """Get sorted candidate index at position i."""
                    # sorted_pos is the position in cand_all after sorting by SDF
                    sorted_pos = cand_sorted_by_sdf[i]
                    cand_idx = cand_all[sorted_pos]
                    # Only use if i < k_val (we want k closest)
                    return jnp.where(i < k_val, cand_idx, 0)

                cand_fixed = jax.vmap(get_sorted_cand)(jnp.arange(max_constraints_per_point))

                # Pad with last valid index if needed
                last_val = jnp.where(k_val > 0, cand_fixed[k_val - 1], 0)
                cand_fixed = jnp.where(
                    jnp.arange(max_constraints_per_point) >= k_val,
                    last_val,
                    cand_fixed
                )
                return cand_fixed, k_val

            def use_all():
                """Fallback: take k closest from all obstacles (matching NumPy line 386-387)."""
                k_val = jnp.minimum(max_constraints_per_point, obs_list_len)
                # Create fixed-size array for JIT compatibility
                cand_fixed = jnp.zeros(max_constraints_per_point, dtype=jnp.int32)
                n_fill = jnp.minimum(k_val, max_constraints_per_point)

                # Fill array using conditional assignment to avoid dynamic slicing
                # For each position, if it's within n_fill and within obs_list_len, take from sorted_indices
                def fill_pos(i):
                    """Fill position i with sorted_indices[i] if i < n_fill and i < obs_list_len, else 0."""
                    idx = jnp.where(i < obs_list_len, sorted_indices[i], 0)
                    return jnp.where(i < n_fill, idx, 0)

                cand_fixed = jax.vmap(fill_pos)(jnp.arange(max_constraints_per_point))
                # Pad with last valid index (or 0 if empty)
                # Ensure last_idx is non-negative and within bounds
                last_idx = jnp.maximum(0, jnp.minimum(n_fill - 1, obs_list_len - 1))
                last_val = jnp.where(n_fill > 0, sorted_indices[last_idx], 0)
                cand_fixed = jnp.where(
                    jnp.arange(max_constraints_per_point) >= n_fill,
                    last_val,
                    cand_fixed
                )
                return cand_fixed, k_val

            # Use candidates if available, otherwise use all (matching NumPy line 383-391)
            cand_indices, k = jax.lax.cond(
                num_candidates > 0,
                use_candidates,
                use_all
            )

            # Compute gradients for candidate obstacles
            grads = grad_fn(x_ref, cand_indices)  # (max_constraints_per_point, dim)

            # Build linearized constraints A x >= b
            # Build constraints for all candidate indices, then select first k
            max_k = max_constraints_per_point

            def build_constraint(i):
                """Build constraint for obstacle at index i (matching NumPy line 436-465)."""
                j = cand_indices[i]
                grad = grads[i]
                d0 = sdf_vals[j]

                # Check if gradient is valid (matching NumPy line 454-457)
                gnorm = jnp.linalg.norm(grad)
                is_valid = jnp.logical_and(jnp.isfinite(gnorm), gnorm >= 1e-8)

                # Normalize gradient (matching NumPy line 462)
                gnorm_safe = jnp.where(gnorm < 1e-8, 1e-8, gnorm)
                g = grad / gnorm_safe
                b_val = (clearance - d0) / gnorm_safe + jnp.dot(g, x_ref)

                # Return constraint and validity flag
                return g, b_val, is_valid

            # Build all constraints using vmap
            gs, bs, constraint_valid = jax.vmap(build_constraint)(jnp.arange(max_k))
            # gs shape: (max_k, dim), bs shape: (max_k,), constraint_valid shape: (max_k,)

            # Select only the first k constraints that are valid (matching NumPy line 400)
            # Create mask: first k are candidate indices, AND constraint is valid
            candidate_mask = jnp.arange(max_k) < k
            valid_mask = jnp.logical_and(candidate_mask, constraint_valid)

            # Count actual valid constraints
            num_valid_constraints = jnp.sum(valid_mask.astype(jnp.int32))

            # Mask out invalid constraints (set to very small values to make them inactive)
            # For invalid rows, set A to zero (no constraint) and b to very negative (satisfied constraint)
            A_masked = jnp.where(valid_mask[:, None], gs, jnp.zeros((max_k, dim), dtype=jnp.float32))
            b_masked = jnp.where(valid_mask, bs, jnp.full((max_k,), -1e6, dtype=jnp.float32))

            # Check if we have any valid constraints (matching NumPy line 400: if A.size == 0: continue)
            has_constraints = num_valid_constraints > 0

            def solve_with_constraints():
                # Use all constraints (invalid ones are automatically satisfied due to large negative b)
                x_proj = _solve_projection_qp_identity_jax(x_ref, A_masked, b_masked)
                
                # Post-process: ensure feasibility by projecting onto violated constraints
                # This ensures the result satisfies all constraints (matching NumPy behavior)
                lhs = A_masked @ x_proj
                violations = b_masked - lhs  # Positive means violated
                max_violation = jnp.max(violations)
                
                # If there are violations, project onto the most violated constraint
                def fix_violation():
                    # Find most violated constraint
                    i_max = jnp.argmax(violations)
                    a = A_masked[i_max]
                    b_val = b_masked[i_max]
                    
                    # Project onto this constraint: a^T x = b
                    denom = jnp.dot(a, a)
                    denom = jnp.where(denom < 1e-12, 1e-12, denom)
                    alpha = (b_val - jnp.dot(a, x_proj)) / denom
                    x_fixed = x_proj + alpha * a
                    
                    # Verify feasibility after fix
                    lhs_fixed = A_masked @ x_fixed
                    still_violated = jnp.any(lhs_fixed + 1e-6 < b_masked)
                    
                    # If still violated, try iterating a few times
                    def iterate_fix(x, _):
                        lhs_check = A_masked @ x
                        violations_check = b_masked - lhs_check
                        i_max_check = jnp.argmax(violations_check)
                        a_check = A_masked[i_max_check]
                        b_check = b_masked[i_max_check]
                        denom_check = jnp.dot(a_check, a_check)
                        denom_check = jnp.where(denom_check < 1e-12, 1e-12, denom_check)
                        alpha_check = (b_check - jnp.dot(a_check, x)) / denom_check
                        return x + alpha_check * a_check, None
                    
                    # Iterate up to 3 times to fix violations
                    x_final, _ = jax.lax.scan(iterate_fix, x_fixed, None, length=3)
                    
                    return x_final
                
                def return_proj():
                    return x_proj
                
                # Only fix if there are significant violations
                return jax.lax.cond(
                    max_violation > 1e-6,
                    fix_violation,
                    return_proj
                )

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

            # Try trajectory QP if enabled and we have multiple points
            def try_trajectory_qp():
                """Try trajectory-level QP for all points."""
                # Find active points (within clearance + margin)
                active_mask = union_sdf < (clearance + constraint_margin)
                num_active = jnp.sum(active_mask.astype(jnp.int32))
                
                # If no active points, return original
                def solve_traj_qp():
                    """Solve trajectory QP."""
                    # Build constraints for each point (same logic as pointwise)
                    # We'll build a dense constraint matrix A: (max_cons, n_vars)
                    max_cons = N * max_constraints_per_point
                    
                    def build_point_constraints(point_idx):
                        """Build constraints for point point_idx, returns (max_constraints_per_point, dim+1)."""
                        x_ref = current_pos[point_idx]
                        sdf_vals = sdf_matrix[:, point_idx]
                        
                        # Select candidate obstacles (same logic as project_single_point)
                        threshold = clearance + constraint_margin
                        cand_mask = sdf_vals < threshold
                        num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
                        sorted_indices = jnp.argsort(sdf_vals)
                        
                        # Get candidate indices (simplified version)
                        k = jnp.minimum(max_constraints_per_point, sdf_vals.shape[0])
                        cand_indices = jnp.where(
                            jnp.arange(max_constraints_per_point) < k,
                            sorted_indices[jnp.arange(max_constraints_per_point)],
                            0
                        )
                        
                        # Get gradients
                        grads = grad_fn(x_ref, cand_indices)  # (max_constraints_per_point, dim)
                        
                        # Build constraints: each row is [g, b] where constraint is g^T x >= b
                        def build_constraint(j):
                            """Build constraint j."""
                            obs_idx = cand_indices[j]
                            grad = grads[j]
                            d0 = sdf_vals[obs_idx]
                            
                            gnorm = jnp.linalg.norm(grad)
                            is_valid = jnp.logical_and(jnp.isfinite(gnorm), gnorm >= 1e-8)
                            
                            g = grad / jnp.maximum(gnorm, 1e-12)
                            b_val = (clearance - d0) / jnp.maximum(gnorm, 1e-12) + jnp.dot(g, x_ref)
                            
                            # Return [g, b] padded to (dim+1,)
                            constraint = jnp.concatenate([g, jnp.array([b_val])])
                            return jnp.where(is_valid, constraint, jnp.zeros(dim + 1, dtype=jnp.float32))
                        
                        constraints = jax.vmap(build_constraint)(jnp.arange(max_constraints_per_point))
                        return constraints  # (max_constraints_per_point, dim+1)
                    
                    # Build constraints for all points
                    all_point_constraints = jax.vmap(build_point_constraints)(jnp.arange(N))
                    # Shape: (N, max_constraints_per_point, dim+1)
                    
                    # Flatten and extract A and b
                    # Reshape to (N * max_constraints_per_point, dim+1)
                    constraints_flat = all_point_constraints.reshape(-1, dim + 1)
                    A_dense = constraints_flat[:, :dim]  # (N * max_constraints_per_point, dim)
                    b_dense = constraints_flat[:, dim]    # (N * max_constraints_per_point,)
                    
                    # Filter out zero rows (invalid constraints)
                    row_norms = jnp.linalg.norm(A_dense, axis=1)
                    valid_rows = row_norms > 1e-8
                    num_valid = jnp.sum(valid_rows.astype(jnp.int32))
                    
                    # Build sparse format for trajectory QP
                    # For simplicity, use all rows (invalid ones will have very negative b)
                    A_valid = jnp.where(valid_rows[:, None], A_dense, jnp.zeros((max_cons, dim), dtype=jnp.float32))
                    b_valid = jnp.where(valid_rows, b_dense, jnp.full((max_cons,), -1e6, dtype=jnp.float32))
                    
                    # Convert to sparse COO format
                    # For each valid constraint, we need to map it to the correct variable indices
                    def build_sparse_entry(cons_idx):
                        """Build sparse matrix entry for constraint cons_idx."""
                        point_idx = cons_idx // max_constraints_per_point
                        constraint_in_point = cons_idx % max_constraints_per_point
                        
                        # Get constraint data
                        a_row = A_valid[cons_idx]  # (dim,)
                        b_val = b_valid[cons_idx]
                        
                        # Build sparse entries: for each dimension, we have one entry
                        # Return as arrays for vmap compatibility
                        values = a_row  # (dim,)
                        rows = jnp.full(dim, cons_idx, dtype=jnp.int32)  # (dim,)
                        cols = point_idx * dim + jnp.arange(dim)  # (dim,)
                        
                        # Stack into (dim, 3) array: [value, row, col]
                        entries = jnp.stack([values, rows.astype(jnp.float32), cols.astype(jnp.float32)], axis=1)
                        return entries, b_val
                    
                    # Build all sparse entries
                    all_entries, all_b = jax.vmap(build_sparse_entry)(jnp.arange(max_cons))
                    # all_entries: (max_cons, dim, 3) - [value, row, col]
                    # all_b: (max_cons,)
                    
                    # Flatten to get A_data, A_row, A_col
                    A_data = all_entries[:, :, 0].flatten()  # (max_cons * dim,)
                    A_row = all_entries[:, :, 1].flatten().astype(jnp.int32)  # (max_cons * dim,)
                    A_col = all_entries[:, :, 2].flatten().astype(jnp.int32)  # (max_cons * dim,)
                    
                    # Filter out zero entries (from invalid constraints)
                    # Keep only entries where value is non-zero
                    nonzero_mask = jnp.abs(A_data) > 1e-10
                    A_data = jnp.where(nonzero_mask, A_data, 0.0)
                    A_row = jnp.where(nonzero_mask, A_row, 0)
                    A_col = jnp.where(nonzero_mask, A_col, 0)
                    
                    # Solve trajectory QP
                    x_flat = prev.reshape(-1)
                    try:
                        x_new_flat = _solve_trajectory_qp_identity_jax(
                            x0=x_flat,
                            A_data=A_data,
                            A_row=A_row,
                            A_col=A_col,
                            b=all_b,
                            n_vars=N * dim,
                            n_cons=max_cons,
                            dim=dim,
                            T=N,
                            smoothness_weight=smoothness_weight,
                        )
                        return x_new_flat.reshape(N, dim)
                    except Exception:
                        # If trajectory QP fails, fall back to pointwise
                        return prev
                
                def use_pointwise():
                    """Use pointwise projection."""
                    def project_point(idx):
                        """Project a single point if it's violating."""
                        is_violating = violating_mask[idx]
                        sdf_vals = sdf_matrix[:, idx]  # (M,)
                        projected = project_single_point(current_pos[idx], sdf_vals, sdf_matrix.shape[0])
                        return jnp.where(is_violating, projected, current_pos[idx])
                    
                    return jax.vmap(project_point)(jnp.arange(N))
                
                # Use trajectory QP if enabled, we have multiple points, and qpax is available
                can_use_traj_qp = jnp.logical_and(
                    use_trajectory_qp,
                    jnp.logical_and(N > 1, QPAX_AVAILABLE)
                )
                
                return jax.lax.cond(
                    can_use_traj_qp,
                    solve_traj_qp,
                    use_pointwise
                )
            
            # Use trajectory QP or pointwise
            new_pos = try_trajectory_qp()

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
