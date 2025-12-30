"""
JAX backend for CFS convexifier.

This implementation uses JAX for JIT compilation and GPU acceleration.
"""

from typing import Optional, Callable
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False
    jax = None
    jnp = None

from enerdynamics.core.constraints.convexify.cfs.cfs import CFSConvexifier
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)
from enerdynamics.core.constraints.core.registry import register
from enerdynamics.core.types import Trajectory


@register("convexifier", "cfs", "jax")
class CFSJAXConvexifier(CFSConvexifier):
    """
    JAX backend for CFS convexifier.
    
    This implementation uses JAX for:
    - JIT compilation
    - GPU acceleration
    - Automatic differentiation
    """
    
    def __init__(
        self,
        obstacles,
        position_extractor: Optional[Callable] = None,
        max_constraints_per_point: int = 8,
        constraint_margin: float = 0.25,
        backend: str = "jax",
        use_jit: bool = True,
        **kwargs
    ):
        """
        Initialize JAX backend CFS convexifier.
        
        Args:
            obstacles: ObstacleManager
            position_extractor: Function to extract position from state
            max_constraints_per_point: Maximum constraints per point
            constraint_margin: Margin for constraint tightening
            backend: Backend ("jax")
            use_jit: Whether to JIT compile
            **kwargs: Additional arguments
        """
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX not available. Install JAX to use CFSJAXConvexifier.")
        
        super().__init__(
            obstacles, position_extractor, max_constraints_per_point,
            constraint_margin, backend, _skip_backend_lookup=True, **kwargs
        )
        
        self.use_jit = use_jit
        
        # Pre-compile SDF texture sampling function if texture is available
        # This avoids repeated compilation even without JIT on the full function
        self._texture_sample_fn = None
        if hasattr(self.obstacles, 'get_sdf_texture_2d') and self.obstacles.get_sdf_texture_2d() is not None:
            # Pre-convert texture to JAX (must be done outside JIT)
            try:
                texture = self.obstacles.get_sdf_texture_2d()
                texture.to_jax()  # Pre-convert to avoid tracer leaks
            except Exception:
                pass  # Will fall back to autodiff
        
        if use_jit:
            # Mark params and state as static since they are dataclasses, not JAX arrays
            # static_argnums: 0=ref_states_jax (JAX array), 1=params (static), 2=state (static)
            self._jit_build = jax.jit(self._build_constraints_jax, static_argnums=(1, 2))
    
    def build_constraints(
        self,
        ref: Trajectory,
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build CFS constraints using JAX.
        
        Args:
            ref: Reference trajectory
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            ConvexConstraint
        """
        if self.use_jit:
            # Convert to JAX arrays
            ref_states_jax = jnp.stack([jnp.asarray(s) for s in ref.states])
            return self._jit_build(ref_states_jax, params, state)
        else:
            return self._build_constraints_jax(
                jnp.stack([jnp.asarray(s) for s in ref.states]),
                params,
                state
            )
    
    def _build_constraints_jax(
        self,
        ref_states: jnp.ndarray,  # (H+1, state_dim)
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        JAX implementation of constraint building using JAX autodiff.
        
        This is the function that gets JIT compiled.
        Uses SDF texture for acceleration if available, otherwise falls back to autodiff.
        """
        # Extract positions
        positions = ref_states[:, :2]  # Assume first 2 dims are position
        
        # OPTIMIZATION: Use SDF texture if available (much faster than autodiff)
        has_sdf_texture = (
            hasattr(self.obstacles, 'get_sdf_texture_2d') and
            hasattr(self.obstacles, 'sample_sdf_and_grad_2d') and
            self.obstacles.get_sdf_texture_2d() is not None
        )
        
        if has_sdf_texture:
            # Use SDF texture for fast batch sampling (no autodiff needed!)
            # This is much faster than vmap(jax.grad(...)) and avoids compilation overhead
            try:
                # Sample SDF and gradients from texture (batch operation)
                # Returns (sdf, grad) tuple where sdf is (H+1,) and grad is (H+1, 2)
                sdf_result, grad_result = self.obstacles.sample_sdf_and_grad_2d(
                    positions, backend="jax", device=None
                )
                # Convert to JAX arrays (already JAX arrays if backend="jax")
                sdfs_jax = jnp.asarray(sdf_result, dtype=jnp.float32)  # (H+1,)
                grads_jax = jnp.asarray(grad_result, dtype=jnp.float32)  # (H+1, 2)
            except Exception as e:
                # Fallback if texture sampling fails
                if self.use_jit:  # Only warn in JIT mode
                    import warnings
                    warnings.warn(f"SDF texture sampling failed: {e}, falling back to autodiff")
                has_sdf_texture = False
        
        if not has_sdf_texture:
            # Fallback: Use JAX autodiff (slower but more general)
            # Try to use JAX-compatible obstacle methods
            use_jax_obstacles = False
            if hasattr(self.obstacles, 'jax_sdf'):
                use_jax_obstacles = True
            elif hasattr(self.obstacles, 'obstacles') and len(self.obstacles.obstacles) > 0:
                use_jax_obstacles = hasattr(self.obstacles.obstacles[0], 'jax_sdf')
            
            if use_jax_obstacles:
                # Use JAX-compatible obstacle methods
                if hasattr(self.obstacles, 'jax_sdf'):
                    # ObstacleManager has jax_sdf method
                    sdfs_jax = self.obstacles.jax_sdf(positions)  # (H+1,)
                    
                    # Compute gradients using JAX autodiff
                    def sdf_fn(pos):
                        return self.obstacles.jax_sdf(pos)
                    
                    # Compute gradients using jax.grad
                    grad_fn = jax.vmap(jax.grad(sdf_fn))
                    grads_jax = grad_fn(positions)  # (H+1, 2)
                else:
                    # Individual obstacles have jax_sdf
                    def sdf_fn(pos):
                        min_sdf = jnp.inf
                        for obs in self.obstacles.obstacles:
                            if hasattr(obs, 'jax_sdf'):
                                sdf_val = obs.jax_sdf(pos)
                                min_sdf = jnp.minimum(min_sdf, sdf_val)
                        return min_sdf
                    
                    sdfs_jax = jax.vmap(sdf_fn)(positions)  # (H+1,)
                    grad_fn = jax.vmap(jax.grad(sdf_fn))
                    grads_jax = grad_fn(positions)  # (H+1, 2)
            else:
                # Fallback: Use JAX autodiff on NumPy SDF function
                def sdf_fn_np(pos_jax):
                    """Wrapper that converts to numpy, calls sdf, converts back."""
                    pos_np = np.asarray(pos_jax)
                    sdf_val = self.obstacles.sdf(pos_np)
                    return jnp.asarray(sdf_val)
                
                sdfs_jax = jax.vmap(sdf_fn_np)(positions)  # (H+1,)
                grad_fn = jax.vmap(jax.grad(sdf_fn_np))
                grads_jax = grad_fn(positions)  # (H+1, 2)
        
        # Build constraints like legacy: for each position, find all obstacles
        # and build constraints for obstacles within threshold
        clearance = params.margin  # This is the clearance (minimum distance)
        threshold = clearance + self.constraint_margin  # Threshold for selecting active obstacles
        
        # Get all obstacles
        obstacles_list = self.obstacles.obstacles if hasattr(self.obstacles, 'obstacles') else [self.obstacles]
        num_obstacles = len(obstacles_list)
        H = positions.shape[0]
        pos_dim = positions.shape[1]
        
        # OPTIMIZATION: Batch compute SDF and gradients for all position-obstacle pairs
        # This replaces the nested Python loops with JAX vectorized operations
        
        # Check if we can batch compute (all obstacles have jax_sdf)
        can_batch = not has_sdf_texture and all(hasattr(obs, 'jax_sdf') for obs in obstacles_list)
        
        if can_batch:
            # Batch compute: (H, num_obstacles) for SDF and (H, num_obstacles, pos_dim) for gradients
            def compute_obstacle_sdf_grad(pos, obstacle):
                """Compute SDF and gradient for a single position-obstacle pair."""
                sdf_val = obstacle.jax_sdf(pos)
                grad_fn = jax.grad(obstacle.jax_sdf)
                grad_val = grad_fn(pos)
                return sdf_val, grad_val
            
            # Vectorize over positions and obstacles
            # Note: We can't directly vmap over obstacles_list (Python list), so we use a different approach
            # Instead, we'll process each obstacle separately and stack results
            
            # Compute for each obstacle across all positions
            sdf_matrix = []  # Will be (num_obstacles, H)
            grad_matrix = []  # Will be (num_obstacles, H, pos_dim)
            
            for obstacle in obstacles_list:
                # Vectorize over positions for this obstacle
                sdf_fn = lambda pos: obstacle.jax_sdf(pos)
                grad_fn = jax.vmap(jax.grad(sdf_fn))
                sdf_vals = jax.vmap(sdf_fn)(positions)  # (H,)
                grad_vals = grad_fn(positions)  # (H, pos_dim)
                sdf_matrix.append(sdf_vals)
                grad_matrix.append(grad_vals)
            
            # Stack: (num_obstacles, H) and (num_obstacles, H, pos_dim)
            sdf_all = jnp.stack(sdf_matrix, axis=0)  # (num_obstacles, H)
            grad_all = jnp.stack(grad_matrix, axis=0)  # (num_obstacles, H, pos_dim)
            
            # Transpose to (H, num_obstacles) and (H, num_obstacles, pos_dim) for easier indexing
            sdf_all = sdf_all.T  # (H, num_obstacles)
            grad_all = jnp.transpose(grad_all, (1, 0, 2))  # (H, num_obstacles, pos_dim)
        else:
            # Fallback: use original loop-based approach (for obstacles without jax_sdf or with texture)
            sdf_all = None
            grad_all = None
        
        # Build constraints using vectorized operations where possible
        # Pre-allocate fixed-size arrays for all constraints
        # Maximum possible constraints: H * max_constraints_per_point
        max_k = self.max_constraints_per_point
        max_total_constraints = H * max_k
        
        # Initialize pre-allocated arrays
        A_rows_prealloc = jnp.zeros((max_total_constraints, pos_dim), dtype=jnp.float32)
        b_vals_prealloc = jnp.full((max_total_constraints,), jnp.inf, dtype=jnp.float32)
        position_indices_prealloc = jnp.zeros((max_total_constraints,), dtype=jnp.int32)
        
        # Use jax.lax.scan to process all positions efficiently
        def process_position(carry, t):
            """Process a single position and return constraints."""
            A_rows, b_vals, position_indices, current_idx = carry
            pos = positions[t]
            
            # Get SDF and gradients for this position
            if can_batch:
                # Use pre-computed batch results
                sdf_array = sdf_all[t]  # (num_obstacles,)
                grads_array = grad_all[t]  # (num_obstacles, pos_dim)
            else:
                # Compute on-the-fly (original logic)
                sdf_values = []
                grads_list = []
                
                for obstacle in obstacles_list:
                    if has_sdf_texture:
                        sdf_val = sdfs_jax[t]
                        grad_val = grads_jax[t]
                    else:
                        if hasattr(obstacle, 'jax_sdf'):
                            sdf_val = obstacle.jax_sdf(pos)
                            grad_fn = jax.grad(obstacle.jax_sdf)
                            grad_val = grad_fn(pos)
                        else:
                            sdf_val = sdfs_jax[t]
                            grad_val = grads_jax[t]
                    
                    sdf_values.append(sdf_val)
                    grads_list.append(grad_val)
                
                sdf_array = jnp.stack(sdf_values)  # (num_obstacles,)
                grads_array = jnp.stack(grads_list)  # (num_obstacles, pos_dim)
            
            # Find obstacles within threshold
            cand_mask = sdf_array < threshold
            
            # Select up to max_constraints_per_point closest obstacles
            num_candidates = jnp.sum(cand_mask.astype(jnp.int32))
            k = jnp.minimum(
                max_k,
                jnp.where(num_candidates == 0, num_obstacles, num_candidates)
            )
            
            # Sort all obstacles by SDF (ascending = most violating first)
            sorted_indices = jnp.argsort(sdf_array)
            # Use fixed max_k for static shape (required for JIT)
            cand_indices = sorted_indices[:max_k]  # Static slice, shape (max_k,)
            
            # Build constraints for selected obstacles (vectorized)
            def build_constraint(j_idx):
                """Build a single constraint."""
                # Check if j_idx is within valid range (k)
                valid_idx = j_idx < k
                # Use cand_indices directly (already padded to max_k)
                j = cand_indices[j_idx]
                d0 = jnp.where(valid_idx, sdf_array[j], 0.0)
                grad = jnp.where(valid_idx, grads_array[j], jnp.zeros(pos_dim))  # (pos_dim,)
                
                grad_flat = jnp.asarray(grad, dtype=jnp.float32).flatten()
                gnorm = jnp.linalg.norm(grad_flat)
                
                # Check if gradient is valid (both valid_idx and gnorm check)
                valid = jnp.logical_and(valid_idx, gnorm >= 1e-8)
                
                # Normalize gradient
                g = jnp.where(valid, grad_flat / gnorm, jnp.zeros_like(grad_flat))
                
                # Build constraint: b = (clearance - d0) / gnorm + g^T pos
                b_val = jnp.where(
                    valid,
                    (clearance - d0) / gnorm + jnp.dot(g, pos),
                    jnp.inf  # Invalid constraint (will be filtered)
                )
                
                return g, b_val, valid
            
            # Vectorize constraint building (use max_k for static shape)
            constraint_results = jax.vmap(build_constraint)(jnp.arange(max_k))
            A_rows_new = constraint_results[0]  # (max_k, pos_dim)
            b_vals_new = constraint_results[1]  # (max_k,)
            valid_mask = constraint_results[2]  # (max_k,)
            
            # Write valid constraints to pre-allocated arrays at current_idx
            # Count valid constraints
            num_valid = jnp.sum(valid_mask.astype(jnp.int32))
            
            # Get valid indices (padded to max_k for static shape)
            valid_indices = jnp.where(valid_mask, size=max_k)[0]  # (max_k,)
            
            # Extract valid constraints
            A_rows_valid = A_rows_new[valid_indices]  # (max_k, pos_dim)
            b_vals_valid = b_vals_new[valid_indices]  # (max_k,)
            
            # Mark invalid entries with inf
            b_vals_masked = jnp.where(valid_mask, b_vals_valid, jnp.inf)
            
            # Write to pre-allocated arrays at current_idx
            # Use dynamic_update_slice to write max_k entries
            # All entries are written (invalid ones marked with inf)
            A_rows = jax.lax.dynamic_update_slice(
                A_rows, A_rows_valid, (current_idx, 0)
            )
            b_vals = jax.lax.dynamic_update_slice(
                b_vals, b_vals_masked, (current_idx,)
            )
            position_indices = jax.lax.dynamic_update_slice(
                position_indices, jnp.full(max_k, t, dtype=jnp.int32), (current_idx,)
            )
            
            # Update current index
            new_idx = current_idx + max_k
            
            return (A_rows, b_vals, position_indices, new_idx), None
        
        # Initialize carry with pre-allocated arrays and index
        carry_init = (A_rows_prealloc, b_vals_prealloc, position_indices_prealloc, 0)
        
        # Process all positions using scan
        (A_rows, b_vals, position_indices, final_idx), _ = jax.lax.scan(
            process_position, carry_init, jnp.arange(H)
        )
        
        # Filter out invalid constraints (marked with inf in b_vals)
        # This is done after scan to avoid dynamic_slice issues with traced values
        valid_final_mask = jnp.isfinite(b_vals)
        valid_final_indices = jnp.where(valid_final_mask)[0]
        
        # Extract only valid constraints
        if valid_final_indices.shape[0] > 0:
            A_rows_final = A_rows[valid_final_indices]
            b_vals_final = b_vals[valid_final_indices]
            position_indices_final = position_indices[valid_final_indices]
        else:
            A_rows_final = jnp.zeros((0, pos_dim), dtype=jnp.float32)
            b_vals_final = jnp.zeros((0,), dtype=jnp.float32)
            position_indices_final = jnp.zeros((0,), dtype=jnp.int32)
        
        # Build trajectory-level constraint matrix
        total_constraints = A_rows_final.shape[0]
        state_dim = pos_dim
        
        if total_constraints > 0:
            # Expand to trajectory-level: (total_constraints, H * state_dim)
            # Vectorized assignment using vmap
            A_traj = jnp.zeros((total_constraints, H * state_dim), dtype=jnp.float32)
            
            # Vectorized assignment: for each constraint, set the appropriate slice
            # Use dynamic_update_slice instead of dynamic slicing (vmap doesn't support dynamic slices)
            def set_constraint_row(i):
                """Set constraint row i in trajectory matrix."""
                t = position_indices_final[i]
                A_traj_row = A_traj[i]
                # Use dynamic_update_slice to update the slice at t*state_dim
                # update shape must match the slice shape: (pos_dim,) matches (state_dim,) if pos_dim == state_dim
                A_traj_row = jax.lax.dynamic_update_slice(
                    A_traj_row,
                    A_rows_final[i],  # (pos_dim,)
                    (t * state_dim,)  # Start index
                )
                return A_traj_row
            
            # Vectorize the assignment
            A_traj = jax.vmap(set_constraint_row)(jnp.arange(total_constraints))
            
            A = A_traj
            b = b_vals_final
        else:
            # No constraints
            A = jnp.zeros((0, H * state_dim), dtype=jnp.float32)
            b = jnp.zeros((0,), dtype=jnp.float32)
        
        # Keep as JAX arrays (ConvexConstraint should handle BackendArray)
        from enerdynamics.core.constraints.core.array_interface import BackendArray
        return ConvexConstraint(
            A=BackendArray(A, "jax"),
            b=BackendArray(b, "jax"),
            meta={"per_step": False, "type": "cfs", "backend": "jax"}
        )

