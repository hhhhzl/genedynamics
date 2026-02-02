"""
Closed-form CBF projection filter for single-integrator dynamics.
"""

from __future__ import annotations

from typing import Any, Optional, Dict
import jax
import jax.numpy as jnp
from enerdynamics.core.constraints.action_filters.base import ConstraintFilter

class ClosedFormCBFFilter(ConstraintFilter):
    """
    Closed-form projection filter based on Control Barrier Functions.
    
    Condition: A^T u >= b, where b = - (eta/dt) * h + beta.
    Projection: u' = u + [ (b - A^T u) / ||A||^2 ]_+ * A
    """
    
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
        # Extract params from schedule_params or use defaults
        # JAX-friendly: we assume these are already jnp arrays if they come from MDOCBackendJax
        params = schedule_params or {}
        tau = params.get("cbf_tau", 0.005)
        eta = params.get("cbf_eta", 1.5)
        margin = params.get("cbf_margin", 0.1)
        base_beta = params.get("base_beta", 0.05)
        
        # robot_radius from env or obstacles if available
        robot_radius = getattr(env, "robot_radius", 0.05)
        dt = getattr(env, "dt", 0.05)
        control_limit = float(getattr(env, "control_limit", 1.0))

        def filter_single(u_seq):
            # Rollout states to get h and A at each step
            # For single integrator: x_{t+1} = x_t + dt * u_t
            # We filter u_t based on h(x_t) and grad_h(x_t).
            
            def body_fn(carry, u):
                x = carry
                
                # Get SDF and grad at current state x
                # Note: obstacles is expected to be an ObstacleManager with sample_sdf_and_grad_2d
                sdf, grad = obstacles.sample_sdf_and_grad_2d(x, backend="jax")
                
                h = sdf - (robot_radius + margin)
                A = grad # Grad of SDF is A
                
                # CBF condition: A^T u >= b
                # b = -(eta/dt) * h + base_beta
                b = -(eta / dt) * h + base_beta
                
                # Projection logic (Two-pass as in mdoc.py)
                lhs = jnp.dot(A, u)
                den = jnp.dot(A, A) + 1e-9
                
                # Filter by tau: only apply if h < tau
                # Also check if grad is valid
                need = (h < tau) & (jnp.linalg.norm(A) > 1e-6)
                
                # Pass 1
                alpha = jnp.maximum(0.0, (b - lhs) / den) * need
                u_safe = u + alpha * A
                
                # Pass 2 (re-check with updated u)
                lhs2 = jnp.dot(A, u_safe)
                alpha2 = jnp.maximum(0.0, (b - lhs2) / den) * (lhs2 < b) * need
                u_safe = u_safe + alpha2 * A

                # Clip to control limit
                u_safe = jnp.clip(u_safe, -control_limit, control_limit)

                # Next state
                x_next = env.jax_transition(x, u_safe)
                return x_next, u_safe

            _, u_seq_safe = jax.lax.scan(body_fn, x0, u_seq)
            return u_seq_safe

        # Handle batching
        if actions.ndim == 3: # (B, H, D)
            return jax.vmap(filter_single)(actions)
        return filter_single(actions)
