"""
ORCA (Optimal Reciprocal Collision Avoidance) convexifier.

ORCA converts velocity obstacles into halfspace constraints for multi-agent
collision avoidance. This is useful for multi-agent scenarios where agents
need to avoid each other.
"""

from typing import Optional, List, Dict, Any
import numpy as np

from enerdynamics.core.constraints.convexify.base import Convexifier
from enerdynamics.core.constraints.core.types import (
    ScheduleState,
    ScheduleParams,
    ConvexConstraint,
)
from enerdynamics.core.constraints.core.registry import register


@register("convexifier", "orca", "numpy")
class ORCAConvexifier(Convexifier):
    """
    ORCA convexifier for multi-agent collision avoidance.
    
    ORCA (Optimal Reciprocal Collision Avoidance) converts velocity obstacles
    into halfspace constraints. For each pair of agents, ORCA computes a
    halfspace constraint that ensures collision avoidance.
    
    The constraint is: n^T v >= b
    where n is the normal vector and b is the RHS.
    """
    
    def __init__(
        self,
        agent_radius: float = 0.1,
        time_horizon: float = 1.0,
        backend: str = "numpy",
        **kwargs
    ):
        """
        Initialize ORCA convexifier.
        
        Args:
            agent_radius: Radius of agents (for collision detection)
            time_horizon: Time horizon for collision avoidance
            backend: Backend to use
            **kwargs: Additional arguments
        """
        self.agent_radius = agent_radius
        self.time_horizon = time_horizon
        self.backend = backend
        self.config = kwargs
    
    def build_constraints(
        self,
        ref: Any,  # Reference: (agent_states, agent_velocities) or trajectory
        params: ScheduleParams,
        state: ScheduleState
    ) -> ConvexConstraint:
        """
        Build ORCA constraints from reference.
        
        Args:
            ref: Reference agent states and velocities
                 Can be tuple of (states, velocities) or trajectory
            params: Schedule parameters
            state: Schedule state
            
        Returns:
            ConvexConstraint with ORCA halfspace constraints
        """
        # Parse reference
        if isinstance(ref, tuple) and len(ref) == 2:
            agent_states, agent_velocities = ref
        else:
            # Assume trajectory - extract states and velocities
            agent_states = [s for s in ref.states] if hasattr(ref, 'states') else [ref]
            agent_velocities = [np.zeros_like(s) for s in agent_states]  # Default velocities
        
        # Convert to arrays
        agent_states = [np.asarray(s, dtype=np.float32) for s in agent_states]
        agent_velocities = [np.asarray(v, dtype=np.float32) for v in agent_velocities]
        
        # Build ORCA constraints for all agent pairs
        constraints_list = []
        
        for i in range(len(agent_states)):
            for j in range(i + 1, len(agent_states)):
                # Compute ORCA constraint for pair (i, j)
                constraint = self._compute_orca_constraint(
                    agent_states[i], agent_velocities[i],
                    agent_states[j], agent_velocities[j],
                    params
                )
                if constraint is not None:
                    constraints_list.append(constraint)
        
        # Stack constraints
        if constraints_list:
            A = np.vstack([c[0] for c in constraints_list])  # (m, dim)
            b = np.array([c[1] for c in constraints_list])  # (m,)
        else:
            # No constraints
            A = np.zeros((0, 2), dtype=np.float32)  # Assume 2D velocity space
            b = np.zeros((0,), dtype=np.float32)
        
        return ConvexConstraint(
            A=A,
            b=b,
            meta={"per_step": True, "type": "orca"}
        )
    
    def _compute_orca_constraint(
        self,
        state_i: np.ndarray,
        vel_i: np.ndarray,
        state_j: np.ndarray,
        vel_j: np.ndarray,
        params: ScheduleParams
    ) -> Optional[tuple]:
        """
        Compute ORCA constraint for agent pair (i, j).
        
        Args:
            state_i: State of agent i (position)
            vel_i: Velocity of agent i
            state_j: State of agent j (position)
            vel_j: Velocity of agent j
            params: Schedule parameters
            
        Returns:
            Tuple of (A_row, b_val) or None if no constraint needed
        """
        # Extract positions (assume first 2 dims are position)
        pos_i = state_i[:2] if len(state_i) >= 2 else state_i
        pos_j = state_j[:2] if len(state_j) >= 2 else state_j
        
        # Relative position and velocity
        rel_pos = pos_j - pos_i
        rel_vel = vel_j - vel_i
        
        # Distance between agents
        dist = np.linalg.norm(rel_pos)
        if dist < 1e-8:
            # Agents are at same position - skip
            return None
        
        # Combined radius
        combined_radius = 2 * self.agent_radius
        
        # Check if collision is imminent
        if dist > combined_radius + self.time_horizon * np.linalg.norm(rel_vel):
            # No collision risk - no constraint needed
            return None
        
        # Compute ORCA halfspace
        # Normal vector (from i to j)
        n = rel_pos / dist
        
        # ORCA constraint: n^T (v_i - v_j) >= u
        # where u is the minimum relative velocity to avoid collision
        # u = (dist - combined_radius) / time_horizon
        
        u = (dist - combined_radius) / self.time_horizon
        
        # Add margin
        margin = params.margin
        u = u + margin
        
        # Constraint for agent i: n^T v_i >= n^T v_j + u
        # This is: n^T v_i >= b
        b_val = np.dot(n, vel_j) + u
        
        # Return constraint (for agent i's velocity)
        A_row = n[None, :]  # (1, 2)
        
        return (A_row, b_val)

