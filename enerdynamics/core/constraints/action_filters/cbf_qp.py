"""
QP-based CBF projection filter for single-integrator dynamics.
"""

from __future__ import annotations

from typing import Any, Optional, Dict
import jax
import jax.numpy as jnp
from enerdynamics.core.constraints.action_filters.base import ConstraintFilter

class QPBasedCBFFilter(ConstraintFilter):
    """
    QP-based projection filter based on Control Barrier Functions.
    
    Problem: min ||u - u_nom||^2
             s.t. A_i^T u >= b_i
    where b_i = - (eta/dt) * h_i + beta.
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
        params = schedule_params or {}
        eta = params.get("cbf_eta", 1.5)
        margin = params.get("cbf_margin", 0.1)
        base_beta = params.get("base_beta", 0.05)
        rho = params.get("rho", 10.0) # Slack penalty if needed
        
        robot_radius = getattr(env, "robot_radius", 0.05)
        dt = getattr(env, "dt", 0.05)
        
        def filter_single(u_seq):
            def body_fn(carry, u):
                x = carry
                sdf, grad = obstacles.sample_sdf_and_grad_2d(x, backend="jax")
                
                h = sdf - (robot_radius + margin)
                A = grad
                b = -(eta / dt) * h + base_beta
                
                # Single-constraint QP solution (for single integrator):
                # min ||u_safe - u||^2  s.t. A^T u_safe >= b
                # If violation (A^T u < b), project:
                # u_safe = u + max(0, (b - A^T u) / ||A||^2) * A
                
                # Note: Even if we use a "QP" name, for single-integrator with 
                # global SDF (one constraint), the closed-form IS the QP solution.
                # However, we can add a slack variable if requested by rho.
                
                lhs = jnp.dot(A, u)
                den = jnp.dot(A, A) + 1e-9
                
                # Slack-QP solution for single constraint:
                # min 0.5||u_s - u||^2 + 0.5 * rho * xi^2
                # s.t. A^T u_s >= b - xi, xi >= 0
                # Solution (analytical for 1 constraint):
                # xi = max(0, (b - A^T u) / (1 + ||A||^2 / rho))
                # u_s = u + (xi / rho) * A
                
                violation = jnp.maximum(0.0, b - lhs)
                xi = violation / (1.0 + den / rho)
                alpha = (xi / rho)
                
                u_safe = u + alpha * A
                
                x_next = env.jax_transition(x, u_safe)
                return x_next, u_safe

            _, u_seq_safe = jax.lax.scan(body_fn, x0, u_seq)
            return u_seq_safe

        if actions.ndim == 3:
            return jax.vmap(filter_single)(actions)
        return filter_single(actions)
