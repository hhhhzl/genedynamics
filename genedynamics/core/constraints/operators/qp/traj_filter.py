"""
Full-horizon trajectory QP filter.

This operator solves a single QP over the entire trajectory horizon,
enforcing trajectory-level constraints (e.g., CFS constraints).
"""

import time
from typing import Tuple, Optional, Dict, Any
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from genedynamics.core.constraints.operators.base import Operator
from genedynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
    OperatorInfo,
)
from genedynamics.core.constraints.core.registry import get_registry, register
from genedynamics.core.types import Trajectory


@register("operator", "traj_qp", "numpy")
class TrajQPFilter(Operator):
    """
    Full-horizon trajectory QP filter.
    
    Solves a single QP over the entire trajectory:
        min ||u - u_nom||^2 + ρ||ξ||^2
        s.t. A u >= b - ξ, ξ >= 0
    
    This is used for trajectory-level constraints (e.g., CFS) where
    constraints are defined over the entire trajectory, not per-step.
    """
    
    def __init__(
        self,
        use_slack: bool = True,
        solver_backend: str = "numpy",
        smoothness_weight: float = 1.0,  # Smoothness regularization weight 
        max_iterations: int = 30,  # Maximum iterations for iterative linearization 
        convergence_tol: float = 1e-6,  # Convergence tolerance 
        **kwargs
    ):
        """
        Initialize trajectory QP filter.
        
        Args:
            use_slack: If True, use slack-QP; if False, use hard-QP
            solver_backend: Solver backend ("numpy", "jax", "osqp")
            **kwargs: Additional arguments
        """
        self.use_slack = use_slack
        self.solver_backend = solver_backend
        self.smoothness_weight = smoothness_weight
        self.max_iterations = max_iterations
        self.convergence_tol = convergence_tol
        self.kwargs = kwargs
        
        # Cache for Hessian matrix (to avoid rebuilding in each iteration)
        self._hessian_cache = {}  # Key: (T, dim, w), Value: P_dense
        
        # Get solver from registry
        registry = get_registry()
        solver_class = None
        
        if solver_backend == "jax":
            # Try "jaxopt_osqp" first (JAXOPTOsqpSolver is registered as "jaxopt_osqp")
            solver_class = registry.get("solver", "jaxopt_osqp", "jax")
        elif solver_backend == "osqp":
            solver_class = registry.get("solver", "osqp", "numpy")
        else:
            # For numpy backend, try different solvers in order of preference
            # Try cvxopt first (if available) - this matches legacy implementation
            solver_class = registry.get("solver", "cvxopt", "numpy")
            # If not available, try osqp
            if solver_class is None:
                solver_class = registry.get("solver", "osqp", "numpy")
        
        if solver_class is not None:
            try:
                self.solver = solver_class(**kwargs)
            except Exception as e:
                # If initialization fails, try fallback
                print(f"Warning: Failed to initialize solver {solver_class}: {e}, using fallback")
                self.solver = None
        else:
            self.solver = None
        
        # Fallback: try to import JAXOPTOsqpSolver or OSQPSolver directly if no solver was created
        if self.solver is None:
            if solver_backend == "jax":
                try:
                    from genedynamics.core.constraints.solvers.jaxopt_osqp_solver import JAXOPTOsqpSolver
                    self.solver = JAXOPTOsqpSolver(**kwargs)
                except (ImportError, TypeError, RuntimeError) as e:
                    print(f"Warning: Failed to import JAXOPTOsqpSolver: {e}")
                    self.solver = None
            elif solver_backend == "osqp":
                try:
                    from genedynamics.core.constraints.solvers.osqp_solver import OSQPSolver
                    self.solver = OSQPSolver(**kwargs)
                except (ImportError, TypeError, RuntimeError) as e:
                    print(f"Warning: Failed to import OSQPSolver: {e}")
                    self.solver = None
            else:
                # For numpy backend, try cvxopt first (matches legacy implementation)
                try:
                    from genedynamics.core.constraints.solvers.cvxopt_solver import CVXOPTSolver
                    self.solver = CVXOPTSolver(**kwargs)
                except (ImportError, TypeError, RuntimeError) as e:
                    print(f"Warning: CVXOPTSolver not available. Please install cvxopt.")
                    self.solver = None
        
        # Final check: raise error if no solver is available
        if self.solver is None:
            raise RuntimeError(
                f"No QP solver available for backend '{solver_backend}'. "
                f"Please install qpax/jaxopt (for JAX) or cvxopt/osqp (for NumPy)."
            )
    
    def apply(
        self,
        nominal: Trajectory,
        constraints: ConvexConstraint,
        params: ScheduleParams,
        state: ScheduleState
    ) -> Tuple[Trajectory, OperatorInfo]:
        """
        Apply full-horizon QP filter to trajectory.
        
        Args:
            nominal: Nominal trajectory to repair
            constraints: Convex constraints (A, b) defined over trajectory
            params: Schedule parameters (rho, etc.)
            state: Schedule state
            
        Returns:
            Tuple of (repaired trajectory, operator info)
        """
        start_time = time.time()
        
        # Extract trajectory actions and states
        H = len(nominal.actions)  # Number of actions
        num_states = len(nominal.states)  # Number of states (usually H+1)
        states = np.stack([np.asarray(s, dtype=np.float32) for s in nominal.states])  # (num_states, state_dim)
        state_dim = states.shape[1] if states.size > 0 else 2
        action_dim = len(nominal.actions[0]) if nominal.actions else state_dim
        
        # Extract positions from states (CFS constraints are on positions)
        pos_dim = min(2, state_dim)  # Usually 2 for 2D, first pos_dim elements of state
        positions = states[:, :pos_dim]  # (num_states, pos_dim)
        
        u_nom = np.stack([np.asarray(a, dtype=np.float32) for a in nominal.actions])  # (H, action_dim)
        u_nom_flat = u_nom.flatten()  # (H*action_dim,)
        
        # Get constraints - handle BackendArray wrapper
        from genedynamics.core.constraints.core.array_interface import BackendArray
        
        # Use BackendArray.to_numpy() method which handles all conversions properly
        if isinstance(constraints.A, BackendArray):
            A = constraints.A.to_numpy()
        else:
            # Not a BackendArray, convert directly
            if JAX_AVAILABLE:
                try:
                    if hasattr(constraints.A, 'block_until_ready'):
                        constraints.A.block_until_ready()
                    A = np.asarray(jax.device_get(constraints.A), dtype=np.float32)
                except (TypeError, AttributeError, ValueError):
                    A = np.asarray(constraints.A, dtype=np.float32)
            else:
                A = np.asarray(constraints.A, dtype=np.float32)
        
        if isinstance(constraints.b, BackendArray):
            b = constraints.b.to_numpy()
        else:
            if JAX_AVAILABLE:
                try:
                    if hasattr(constraints.b, 'block_until_ready'):
                        constraints.b.block_until_ready()
                    b = np.asarray(jax.device_get(constraints.b), dtype=np.float32)
                except (TypeError, AttributeError, ValueError):
                    b = np.asarray(constraints.b, dtype=np.float32)
            else:
                b = np.asarray(constraints.b, dtype=np.float32)
        
        # Handle different constraint shapes
        # Check if constraints are on states (positions) or actions
        # For CFS constraints, they are on states (positions), so we should optimize positions directly
        constraints_on_positions = False
        optimize_positions = False
        
        if A.ndim == 2:
            if A.shape[1] == u_nom_flat.shape[0]:
                # A is (m, H*action_dim) - full trajectory action constraints
                A_full = A
                constraints_on_positions = False
                optimize_positions = False
            elif A.shape[1] == num_states * pos_dim:
                # A is (m, num_states*pos_dim) - constraints on positions only (CFS case)
                # This happens when CFSConvexifier only constrains positions
                constraints_on_positions = True
                optimize_positions = True
                A_full = A  # Already in position space
            elif A.shape[1] == H * pos_dim:
                # A is (m, H*pos_dim) - constraints on H positions (not including initial state)
                # This is another CFS case
                constraints_on_positions = True
                optimize_positions = True
                A_full = A  # Already in position space
            elif A.shape[1] == num_states * state_dim:
                # A is (m, num_states*state_dim) - full trajectory state constraints
                # For CFS, constraints are on positions (first pos_dim elements of each state)
                # We should optimize positions directly, not actions (like legacy implementation)
                constraints_on_positions = True
                optimize_positions = True
                
                # Extract position constraints from state constraints
                # A_state is (m, num_states * state_dim), we need (m, num_states * pos_dim)
                m = A.shape[0]
                A_pos = np.zeros((m, num_states * pos_dim), dtype=np.float32)
                
                for t in range(num_states):
                    # Extract position part from state constraint at state t
                    A_pos[:, t*pos_dim:(t+1)*pos_dim] = A[:, t*state_dim:t*state_dim+pos_dim]
                
                A_full = A_pos  # Use position constraints directly
            elif A.shape[1] == H * state_dim:
                # A is (m, H*state_dim) - state constraints (one per state, not including initial)
                # This is the case for CFS constraints
                # We should optimize positions directly, not actions (like legacy implementation)
                constraints_on_positions = True
                optimize_positions = True
                
                # Extract position constraints from state constraints
                # A_state is (m, H * state_dim), we need (m, H * pos_dim)
                m = A.shape[0]
                A_pos = np.zeros((m, H * pos_dim), dtype=np.float32)
                
                for t in range(H):
                    # Extract position part from state constraint at time t
                    A_pos[:, t*pos_dim:(t+1)*pos_dim] = A[:, t*state_dim:t*state_dim+pos_dim]
                
                A_full = A_pos  # Use position constraints directly
            elif A.shape[1] == u_nom.shape[1]:
                # A is (m, action_dim) - per-step action constraints, apply to all steps
                # Expand to full trajectory: [A, 0, ...; 0, A, ...; ...]
                m, dim = A.shape
                A_full = np.zeros((m, H * dim), dtype=np.float32)
                for t in range(H):
                    A_full[:, t*dim:(t+1)*dim] = A
            else:
                raise ValueError(
                    f"Constraint matrix A shape {A.shape} incompatible with trajectory. "
                    f"Expected one of: (m, {H * action_dim}), (m, {action_dim}), "
                    f"(m, {num_states * state_dim}), (m, {H * state_dim}), "
                    f"(m, {num_states * pos_dim}), or (m, {H * pos_dim}). "
                    f"Note: State-space constraints should use ProjectionOperator, not TrajQPFilter."
                )
        else:
            raise ValueError(f"Constraint matrix A must be 2D, got {A.ndim}D")
        
        # Solve QP
        # For CFS constraints (on positions), optimize positions directly 
        # For action constraints, optimize actions
        pos_star_flat = None  # Initialize for violation computation
        u_star_flat = None  # Initialize for violation computation
        
        if optimize_positions:
            # Optimize positions directly (like legacy implementation)
            # x0 is flattened positions from states
            if A.shape[1] == num_states * pos_dim:
                # Constraints on all states' positions (num_states * pos_dim)
                pos_nom_flat = positions.flatten()  # (num_states * pos_dim,)
            elif A.shape[1] == H * pos_dim:
                # Constraints on H states' positions (not including initial)
                pos_nom_flat = positions[1:].flatten()  # (H * pos_dim,) - skip initial state
            elif A.shape[1] == num_states * state_dim:
                # Constraints on all states including initial (extracted to positions)
                pos_nom_flat = positions.flatten()  # (num_states * pos_dim,)
            else:
                # Constraints on H states (not including initial, extracted to positions)
                pos_nom_flat = positions[1:].flatten()  # (H * pos_dim,) - skip initial state
                # Adjust A_full if needed (should already be H * pos_dim)
                if A_full.shape[1] != pos_nom_flat.shape[0]:
                    # Need to adjust - extract positions for H states
                    pos_nom_flat = positions[:H].flatten()  # Use first H states
            
            # Solve QP on positions with smoothness term using solver's native method
            # Determine T (number of time steps) based on A shape
            if A.shape[1] == num_states * pos_dim:
                T = num_states
            elif A.shape[1] == H * pos_dim:
                T = H
            elif A.shape[1] == num_states * state_dim:
                T = num_states
            else:
                T = H
            
            # Extract initial state for equality constraint (align with legacy)
            # Get initial position from nominal trajectory (not from current positions)
            # This ensures we fix the initial state to the original starting point
            initial_pos = None
            if len(positions) > 0:
                initial_pos = np.asarray(positions[0], dtype=np.float32).copy()
            elif len(nominal.states) > 0:
                # Fallback: extract from nominal trajectory
                initial_state = np.asarray(nominal.states[0], dtype=np.float32)
                initial_pos = initial_state[:pos_dim].copy()
            
            # Use solver's native implementation (JAX or NumPy)
            pos_star_flat, info = self.solver.solve_traj_qp_with_smoothness(
                pos_nom_flat, A_full, b, params.rho if self.use_slack else None,
                T, pos_dim, self.smoothness_weight, self.use_slack,
                fix_initial_state=True,  # Align with legacy: fix_initial_state=True
                initial_state=initial_pos,  # Pass initial state position
                **self.kwargs
            )
            
            # Reshape optimized positions back to trajectory
            if A.shape[1] == num_states * pos_dim:
                # Optimized all states' positions (num_states * pos_dim)
                pos_star = pos_star_flat.reshape(num_states, pos_dim)
                repaired_states = []
                for t in range(num_states):
                    state = np.asarray(nominal.states[t], dtype=np.float32).copy()
                    state[:pos_dim] = pos_star[t]
                    repaired_states.append(state)
            elif A.shape[1] == H * pos_dim:
                # Optimized H states' positions (not including initial)
                pos_star = pos_star_flat.reshape(H, pos_dim)
                repaired_states = [np.asarray(nominal.states[0], dtype=np.float32).copy()]  # Keep initial state
                for t in range(H):
                    state = np.asarray(nominal.states[t + 1], dtype=np.float32).copy()
                    state[:pos_dim] = pos_star[t]
                    repaired_states.append(state)
            elif A.shape[1] == num_states * state_dim:
                # Optimized all states including initial (extracted to positions)
                pos_star = pos_star_flat.reshape(num_states, pos_dim)
                repaired_states = []
                for t in range(num_states):
                    state = np.asarray(nominal.states[t], dtype=np.float32).copy()
                    state[:pos_dim] = pos_star[t]
                    repaired_states.append(state)
            else:
                # Optimized H states (not including initial, extracted to positions)
                pos_star = pos_star_flat.reshape(H, pos_dim)
                repaired_states = [np.asarray(nominal.states[0], dtype=np.float32).copy()]  # Keep initial state
                for t in range(H):
                    state = np.asarray(nominal.states[t + 1], dtype=np.float32).copy()
                    state[:pos_dim] = pos_star[t]
                    repaired_states.append(state)
            
            # Actions remain unchanged (or could be computed from state differences)
            repaired_actions = nominal.actions
        else:
            # Optimize actions: build QP and solve directly
            n = len(u_nom_flat)
            
            # Standard QP: min ||u - u_nom||^2 = min 0.5 u^T I u - u_nom^T u
            P = np.eye(n, dtype=np.float32)
            q = -u_nom_flat.astype(np.float32)
            
            if self.use_slack:
                # Slack-QP: min ||u - u_nom||^2 + ρ||ξ||^2 s.t. A u >= b - ξ, ξ >= 0
                m = len(b)
                n_total = n + m
                
                # Extended P matrix
                P_ext = np.zeros((n_total, n_total), dtype=np.float32)
                P_ext[:n, :n] = P
                P_ext[n:, n:] = params.rho * np.eye(m, dtype=np.float32)
                
                # Extended q vector
                q_ext = np.zeros(n_total, dtype=np.float32)
                q_ext[:n] = q
                
                # Extended constraints: A u + ξ >= b, ξ >= 0
                # => -A u - ξ <= -b, -ξ <= 0
                G_ext = np.zeros((2 * m, n_total), dtype=np.float32)
                h_ext = np.zeros(2 * m, dtype=np.float32)
                
                # -A u - ξ <= -b
                G_ext[:m, :n] = -A_full.astype(np.float32)
                G_ext[:m, n:] = -np.eye(m, dtype=np.float32)
                h_ext[:m] = -b.astype(np.float32)
                
                # -ξ <= 0 (i.e., ξ >= 0)
                G_ext[m:, n:] = -np.eye(m, dtype=np.float32)
                h_ext[m:] = 0.0
                
                solution_ext, info = self.solver.solve_qp(P_ext, q_ext, G_ext, h_ext)
                u_star_flat = solution_ext[:n]
                info['slack'] = solution_ext[n:]
            else:
                # Hard-QP: min ||u - u_nom||^2 s.t. A u >= b
                # Convert A u >= b to -A u <= -b
                G = -A_full.astype(np.float32) if A_full.size > 0 else None
                h = -b.astype(np.float32) if b.size > 0 else None
                
                u_star_flat, info = self.solver.solve_qp(P, q, G, h)
            
            # Reshape back to trajectory
            u_star = u_star_flat.reshape(H, -1)
            repaired_actions = [u_star[t] for t in range(H)]
            repaired_states = nominal.states
        
        # Create repaired trajectory
        repaired = Trajectory(
            states=repaired_states,
            actions=repaired_actions,
            info=nominal.info
        )
        
        elapsed_time = time.time() - start_time
        
        # Compute violations
        if optimize_positions:
            # Violations computed on positions
            # Use the same pos_nom_flat that was used for optimization
            # This ensures dimensions match A_full
            violation_before = np.maximum(0, b - A_full @ pos_nom_flat).max() if A_full.size > 0 else 0.0
            violation_after = np.maximum(0, b - A_full @ pos_star_flat).max() if A_full.size > 0 else 0.0
        else:
            # Violations computed on actions
            violation_before = np.maximum(0, b - A_full @ u_nom_flat).max() if A_full.size > 0 else 0.0
            violation_after = np.maximum(0, b - A_full @ u_star_flat).max() if A_full.size > 0 else 0.0
        
        operator_info = OperatorInfo(
            success=info.get('status', 'optimal') == 'optimal',
            violation_before=violation_before,
            violation_after=violation_after,
            iterations=info.get('iterations', 1),
            time=elapsed_time,
            extra={
                'qp_count': 1,
                'solver_method': info.get('method', 'traj_qp'),
                **info
            }
        )
        
        return repaired, operator_info


