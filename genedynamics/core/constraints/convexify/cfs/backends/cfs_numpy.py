"""
NumPy backend for CFS convexifier.

This is the reference implementation using NumPy.
"""

from typing import Optional, Callable
import numpy as np

from genedynamics.core.constraints.convexify.cfs.cfs import CFSConvexifier
from genedynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)
from genedynamics.core.constraints.core.registry import register
from genedynamics.core.types import Trajectory, State


@register("convexifier", "cfs", "numpy")
class CFSNumpyConvexifier(CFSConvexifier):
    """
    NumPy backend for CFS convexifier.
    
    This is the reference implementation that other backends should match.
    """
    
    def build_constraints(
        self,
        ref: Trajectory,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build CFS constraints using NumPy.
        
        This is the complete NumPy implementation of CFS constraint generation.
        """
        return self._build_constraints_numpy(ref, params, state)
    
    def _build_constraints_numpy(
        self,
        ref: Trajectory,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """NumPy implementation of CFS constraint generation."""
        # Extract positions
        positions = [self.position_extractor(s) for s in ref.states]
        positions = np.stack(positions, axis=0)  # (H, dim)
        
        # Compute SDF and gradients for all positions
        clearance = params.margin  # clearance (minimum distance)
        threshold = clearance + self.constraint_margin  # Threshold for selecting active obstacles
        
        constraints_list = []  # List of (position_idx, A_row, b_val)
        
        # Get all obstacles
        obstacles_list = self.obstacles.obstacles if hasattr(self.obstacles, 'obstacles') else [self.obstacles]
        
        # Process all points
        
        # Debug: Track constraint generation per point
        debug_info = {
            'points_processed': 0,
            'constraints_per_point': [],
            'total_constraints': 0,
            'points_with_no_constraints': 0
        }
        
        for t, pos in enumerate(positions):
            # Compute SDF for all obstacles at this position
            sdf_values = []
            for obs in obstacles_list:
                sdf_val = obs.sdf(pos) if hasattr(obs, 'sdf') else self.obstacles.sdf(pos)
                sdf_values.append(sdf_val)
            sdf_values = np.asarray(sdf_values, dtype=np.float32)
            
            # Find obstacles that are close enough
            cand_mask = sdf_values < threshold
            cand_indices = np.where(cand_mask)[0]
            
            # Select candidate obstacles
            # This ensures we get the closest obstacles even if they're not in candidate list
            if cand_indices.size == 0:
                # No obstacles close enough, but still add constraint for closest obstacle
                # Select k closest from all obstacles
                k = min(self.max_constraints_per_point, len(sdf_values))
                cand_indices = np.argsort(sdf_values)[:k]
            else:
                # Select from candidates
                k = min(self.max_constraints_per_point, cand_indices.size)
                # Sort all obstacles by SDF (ascending = most violating first)
                sorted_all = np.argsort(sdf_values)
                cand_indices = sorted_all[:k]
            
            # Build constraints for each selected obstacle
            valid_constraints = 0
            constraints_before = len(constraints_list)
            for j_idx in range(k):
                if valid_constraints >= self.max_constraints_per_point:
                    break
                
                j = cand_indices[j_idx]
                obstacle = obstacles_list[int(j)]
                d0 = float(sdf_values[int(j)])
                
                # Get gradient
                grad = None
                if hasattr(obstacle, "gradient"):
                    try:
                        grad = obstacle.gradient(pos)
                    except Exception:
                        grad = None
                if grad is None:
                    # Use finite differences for THIS obstacle (not union SDF)
                    grad = self._finite_difference_gradient(obstacle, pos)
                
                grad = np.asarray(grad, dtype=np.float32).flatten()
                gnorm = float(np.linalg.norm(grad))
                
                if not np.isfinite(gnorm) or gnorm < 1e-8:
                    continue  # Skip this obstacle if gradient is invalid
                
                # Normalize gradient
                g = grad / gnorm
                
                # Build constraint: b = (clearance - d0) / gnorm + g^T x_ref
                # Constraint: g^T x >= (clearance - d0) / gnorm + g^T pos
                A_row = g[None, :]  # (1, dim)
                b_val = float((clearance - d0) / gnorm + float(np.dot(g, pos)))
                constraints_list.append((t, A_row, b_val))
                valid_constraints += 1
            
            # Debug: Track constraints generated for this point
            constraints_for_this_point = len(constraints_list) - constraints_before
            debug_info['constraints_per_point'].append(constraints_for_this_point)
            debug_info['points_processed'] += 1
            if constraints_for_this_point == 0:
                debug_info['points_with_no_constraints'] += 1
        
        # Debug: Print constraint generation summary
        debug_info['total_constraints'] = len(constraints_list)
        verbose = getattr(self, 'config', None) and getattr(self.config, 'verbose', False)
        if verbose:
            print(f"[CFS Debug] Points processed: {debug_info['points_processed']}")
            print(f"[CFS Debug] Total constraints: {debug_info['total_constraints']}")
            print(f"[CFS Debug] Points with no constraints: {debug_info['points_with_no_constraints']}")
            if debug_info['constraints_per_point']:
                print(f"[CFS Debug] Constraints per point - min: {min(debug_info['constraints_per_point'])}, "
                      f"max: {max(debug_info['constraints_per_point'])}, "
                      f"mean: {np.mean(debug_info['constraints_per_point']):.2f}")
        
        # Stack constraints
        # CFS constraints are per-position, so we need to expand them to trajectory-level
        # for use with ProjectionOperator
        H = len(ref.states)
        pos_dim = positions.shape[1]  # Usually 2 for 2D
        state_dim = len(ref.states[0]) if ref.states else pos_dim
        
        if constraints_list:
            # Build trajectory-level constraint matrix
            # Each constraint corresponds to one position at time t
            # We need to expand to (total_constraints, H * state_dim)
            # But we only constrain the position part of each state
            total_constraints = len(constraints_list)
            
            A_traj = np.zeros((total_constraints, H * state_dim), dtype=np.float32)
            b_traj = np.zeros(total_constraints, dtype=np.float32)
            
            for i, (t, A_row, b_val) in enumerate(constraints_list):
                # Place constraint at the correct position in trajectory
                # Only constrain the position part (first pos_dim elements) of state t
                A_traj[i, t*state_dim:t*state_dim+pos_dim] = A_row[0]
                b_traj[i] = b_val
            
            A = A_traj
            b = b_traj
            
            # Debug: Print constraint matrix info
            verbose = getattr(self, 'config', None) and getattr(self.config, 'verbose', False)
            if verbose:
                print(f"[CFS Debug] Constraint matrix shape: A={A.shape}, b={b.shape}")
                print(f"[CFS Debug] Trajectory shape: H={H}, state_dim={state_dim}, pos_dim={pos_dim}")
                # Check for non-zero elements
                non_zero_rows = np.sum(np.any(A != 0, axis=1))
                print(f"[CFS Debug] Non-zero constraint rows: {non_zero_rows}/{A.shape[0]}")
                # Check constraint distribution across trajectory
                constraints_per_timestep = {}
                for i, (t, _, _) in enumerate(constraints_list):
                    constraints_per_timestep[t] = constraints_per_timestep.get(t, 0) + 1
                if constraints_per_timestep:
                    print(f"[CFS Debug] Constraints per timestep - min: {min(constraints_per_timestep.values())}, "
                          f"max: {max(constraints_per_timestep.values())}, "
                          f"mean: {np.mean(list(constraints_per_timestep.values())):.2f}")
        else:
            A = np.zeros((0, H * state_dim), dtype=np.float32)
            b = np.zeros((0,), dtype=np.float32)
            verbose = getattr(self, 'config', None) and getattr(self.config, 'verbose', False)
            if verbose:
                print(f"[CFS Debug] No constraints generated! A={A.shape}, b={b.shape}")
        
        return ConvexConstraint(
            A=A,
            b=b,
            meta={"per_step": False, "type": "cfs", "constrains": "states"}
        )
    
    def _finite_difference_gradient(self, obstacle, point: np.ndarray, eps: float = 1e-4) -> np.ndarray:
        """
        Compute gradient using finite differences for a specific obstacle.
    
        Args:
            obstacle: The specific obstacle to compute gradient for
            point: Point at which to compute gradient
            eps: Step size for finite differences
            
        Returns:
            Gradient vector, shape (dim,)
        """
        point = np.asarray(point, dtype=np.float32)
        dim = point.shape[0]
        grad = np.zeros(dim, dtype=np.float32)
        
        for i in range(dim):
            xp = point.copy()
            xm = point.copy()
            xp[i] += eps
            xm[i] -= eps
            
            # Use THIS obstacle's SDF
            if hasattr(obstacle, 'sdf'):
                sdf_p = obstacle.sdf(xp)
                sdf_m = obstacle.sdf(xm)
            else:
                # Fallback: use obstacles manager (should not happen in normal flow)
                sdf_p = self.obstacles.sdf(xp)
                sdf_m = self.obstacles.sdf(xm)
            
            grad[i] = (sdf_p - sdf_m) / (2 * eps)
        
        return grad
