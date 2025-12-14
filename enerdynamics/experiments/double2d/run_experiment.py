"""
EDOC experiment on Double Integrator 2D with obstacles.

This script:
1. Creates 10 levels of obstacle configurations (convex → non-convex, low → high density)
2. Adds acceleration constraints
3. Runs EDOC with soft→hard constraint scheduling
4. Visualizes diffusion steps, trajectories, energy, rewards, states
5. Computes SSR (Safety Success Rate)
6. Saves all results to /results/double2d/
"""

import os
import sys
import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
import json
from dataclasses import asdict
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import time
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from enerdynamics.envs.double_integrator_box_2d import DoubleIntegratorBox2DEnv
from enerdynamics.envs.obstacles.base import ObstacleManager
from enerdynamics.envs.obstacles.convex import SphereObstacle, BoxObstacle

# Alias for 2D: SphereObstacle is CircleObstacle in 2D
CircleObstacle = SphereObstacle
from enerdynamics.envs.obstacles.nonconvex import UnionObstacle
from enerdynamics.core.constraints import (
    SoftConstraint,
    HardConstraint,
    ConstraintManager,
    ObstacleSoftConstraint,
    ObstacleHardConstraint,
    CFSProjection,
    CBFDoubleIntegrator2DActionFilter,
    ConstraintScheduleManager,
)
from enerdynamics.core.types import Trajectory, State, Action
from enerdynamics.solvers.edoc import EDOCPlanner
from enerdynamics.envs.factories import make_energy
from enerdynamics.core.energy import LegacyEnergyFunctional, EnergyTerm

# ============================================================================
# Constants (matching compare_di_planners.py)
# ============================================================================

EDOC_COLOR = "#1f77b4"  # Blue
OBSTACLE_COLOR = "gray"
OBSTACLE_ALPHA = 0.5
DIFFUSION_FRACTIONS = (0.1, 0.5, 0.90)  # 90%, 50%, 10%
MAX_SAMPLE_TRAJ_PLOT = 80
ROBOT_RADIUS = 0.05  # Robot radius for collision checking and visualization
MIN_OBSTACLE_MARGIN = 4 * ROBOT_RADIUS  # Minimum margin between obstacles
OBSTACLE_RADIUS_SCALE = 1  # Obstacle size relative to robot radius (1.0 = same size)


# ============================================================================
# Acceleration Constraint
# ============================================================================

class AccelerationConstraint(HardConstraint):
    """
    Hard constraint on acceleration (control input).
    
    Ensures |u| ≤ u_max for all actions in trajectory.
    """

    def __init__(self, u_max: float = 1.0):
        """
        Initialize acceleration constraint.
        
        Args:
            u_max: Maximum acceleration magnitude
        """
        self.u_max = float(u_max)

    def is_feasible(self, trajectory: Trajectory) -> bool:
        """Check if all actions satisfy |u| ≤ u_max."""
        for action in trajectory.actions:
            action_np = np.asarray(action, dtype=np.float32)
            if np.linalg.norm(action_np) > self.u_max:
                return False
        return True

    def violations(self, trajectory: Trajectory) -> np.ndarray:
        """Return violations: max(0, |u| - u_max) for each action."""
        violations = []
        for action in trajectory.actions:
            action_np = np.asarray(action, dtype=np.float32)
            violation = max(0.0, np.linalg.norm(action_np) - self.u_max)
            violations.append(violation)
        return np.array(violations, dtype=np.float32)

    def project(
            self,
            trajectory: Trajectory,
            step: Optional[int] = None,
            total_steps: Optional[int] = None
    ) -> Trajectory:
        """Project actions to satisfy |u| ≤ u_max."""
        projected_actions = []
        for action in trajectory.actions:
            action_np = np.asarray(action, dtype=np.float32)
            norm = np.linalg.norm(action_np)
            if norm > self.u_max:
                # Clip to u_max
                action_np = action_np / norm * self.u_max
            projected_actions.append(action_np)

        return Trajectory(states=trajectory.states, actions=projected_actions)


# ============================================================================
# Obstacle Configuration Generator
# ============================================================================

def _check_obstacle_spacing(new_center: np.ndarray, new_radius: float,
                            existing_obstacles: List[Tuple[np.ndarray, float]],
                            margin: float) -> bool:
    """Check if new obstacle has enough margin from existing obstacles."""
    for existing_center, existing_radius in existing_obstacles:
        distance = np.linalg.norm(new_center - existing_center)
        min_distance = new_radius + existing_radius + margin
        if distance < min_distance:
            return False
    return True


def _get_obstacle_radius(obstacle) -> float:
    """Get obstacle radius (for circles) or max half-extent (for boxes)."""
    if isinstance(obstacle, (SphereObstacle, CircleObstacle)):
        return float(obstacle.radius)
    elif isinstance(obstacle, BoxObstacle):
        return float(np.max(obstacle.half_extents))
    elif isinstance(obstacle, UnionObstacle):
        # For union, get max radius of all primitives
        max_radius = 0.0
        for prim in obstacle.obstacles:
            prim_radius = _get_obstacle_radius(prim)
            max_radius = max(max_radius, prim_radius)
        return max_radius
    return 0.0


def _check_start_target_clearance(
        obstacle,
        start_pos: np.ndarray,
        target_pos: np.ndarray,
        buffer: float = ROBOT_RADIUS,
) -> bool:
    """
    Ensure start/target are not inside (or too close to) the obstacle.

    We use obstacle SDF semantics:
      - sdf < 0  : inside
      - sdf == 0 : on boundary
      - sdf > 0  : outside (distance)

    For safety evaluation in this experiment we treat the robot as a disc with
    radius ROBOT_RADIUS, so require sdf >= buffer.
    """
    buf = float(buffer)
    for p in (start_pos, target_pos):
        p = np.asarray(p, dtype=np.float32)
        sdf = obstacle.sdf(p)
        sdf = float(np.asarray(sdf).item() if hasattr(sdf, "item") else sdf)
        if sdf < buf:
            return False
        # Extra guard if obstacle implements an exact contains() check
        try:
            if hasattr(obstacle, "contains") and obstacle.contains(p):
                return False
        except Exception:
            # If contains() fails for any reason, rely on sdf threshold above
            pass
    return True


def generate_obstacle_config(level: int, seed: int, start_pos: np.ndarray,
                             target_pos: np.ndarray) -> ObstacleManager:
    """
    Generate obstacle configuration for given level with proper spacing.
    
    Levels progress in both density (number of obstacles) and non-convexity:
    0: No obstacles (0 obstacles)
    1-3: Convex only (3, 6, 10 obstacles) - simple circles/boxes
    4-6: Mixed convex (14, 18, 22 obstacles) - diverse sizes/types
    7-10: Non-convex unions (4/5/6/8 unions with 3/4/5/6 primitives each)
          Level 7: 4 unions × 3 primitives = 4 union obstacles
          Level 8: 5 unions × 4 primitives = 5 union obstacles
          Level 9: 6 unions × 5 primitives = 6 union obstacles
          Level 10: 8 unions × 6 primitives = 8 union obstacles
    
    Both obstacle count and non-convexity increase from level 0 to 10.
    
    Args:
        level: Obstacle level (0-10)
        seed: Random seed for reproducibility
        start_pos: Start position (to place obstacles between start and target)
        target_pos: Target position
        
    Returns:
        ObstacleManager with obstacles
    """
    np.random.seed(seed)
    manager = ObstacleManager()

    if level == 0:
        # No obstacles
        return manager

    p_max = 2.0  # Environment bounds
    margin = MIN_OBSTACLE_MARGIN

    # Place obstacles between start and target
    # Compute region between start and target
    direction = target_pos - start_pos
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1e-6:
        # Start and target are same, place obstacles randomly
        center_region = np.array([0.0, 0.0], dtype=np.float32)
        region_size = p_max * 0.8
    else:
        # Place obstacles along the path from start to target
        center_region = (start_pos + target_pos) / 2.0
        region_size = direction_norm * 0.8 + 0.5  # Extend region

    # Existing obstacles for spacing check
    existing = []

    if level <= 3:
        # Convex only (simple primitives)
        # Level 1-3: increasing density
        num_obstacles = {1: 3, 2: 6, 3: 10}[level]

        max_attempts = 100
        for i in range(num_obstacles):
            # Random size (around robot radius)
            base_radius = ROBOT_RADIUS * OBSTACLE_RADIUS_SCALE
            radius = np.random.uniform(base_radius * 0.8, base_radius * 1.5)

            # Try to place obstacle with proper spacing
            placed = False
            for attempt in range(max_attempts):
                # Place in region between start and target
                offset = np.array([
                    np.random.uniform(-region_size / 2, region_size / 2),
                    np.random.uniform(-region_size / 2, region_size / 2)
                ], dtype=np.float32)
                center = center_region + offset

                # Keep within bounds
                center = np.clip(center, -p_max + radius + margin, p_max - radius - margin)

                # Check spacing
                if _check_obstacle_spacing(center, radius, existing, margin):
                    # Random type: circle or box
                    if np.random.rand() < 0.5:
                        obstacle = SphereObstacle(center=center, radius=radius, name=f"circle_{i}")
                    else:
                        size = np.array([radius, radius], dtype=np.float32)
                        obstacle = BoxObstacle(center=center, half_extents=size, name=f"box_{i}")

                    # Hard guarantee: start/target cannot be in (or too close to) obstacles
                    if not _check_start_target_clearance(obstacle, start_pos, target_pos, buffer=ROBOT_RADIUS):
                        continue

                    manager.add(obstacle)
                    existing.append((center, radius))
                    placed = True
                    break

            if not placed:
                print(f"Warning: Could not place obstacle {i} with proper spacing")

    elif level <= 6:
        # Mixed convex (more diverse types and sizes)
        # Level 4-6: increasing density with more variety
        num_obstacles = {4: 14, 5: 18, 6: 22}[level]

        max_attempts = 100
        for i in range(num_obstacles):
            obstacle_type = np.random.choice(['circle', 'box', 'small_circle'])

            base_radius = ROBOT_RADIUS * OBSTACLE_RADIUS_SCALE
            if obstacle_type == 'circle':
                radius = np.random.uniform(base_radius * 0.8, base_radius * 1.5)
            elif obstacle_type == 'box':
                size = np.random.uniform(base_radius * 0.8, base_radius * 1.5)
                radius = size  # Use size as radius for spacing check
            else:  # small_circle
                radius = np.random.uniform(base_radius * 0.5, base_radius * 1.0)

            # Try to place obstacle
            placed = False
            for attempt in range(max_attempts):
                offset = np.array([
                    np.random.uniform(-region_size / 2, region_size / 2),
                    np.random.uniform(-region_size / 2, region_size / 2)
                ], dtype=np.float32)
                center = center_region + offset
                center = np.clip(center, -p_max + radius + margin, p_max - radius - margin)

                if _check_obstacle_spacing(center, radius, existing, margin):
                    if obstacle_type == 'circle':
                        obstacle = SphereObstacle(center=center, radius=radius, name=f"circle_{i}")
                    elif obstacle_type == 'box':
                        half_extents = np.array([radius, radius], dtype=np.float32)
                        obstacle = BoxObstacle(center=center, half_extents=half_extents, name=f"box_{i}")
                    else:  # small_circle
                        obstacle = SphereObstacle(center=center, radius=radius, name=f"small_circle_{i}")

                    # Hard guarantee: start/target cannot be in (or too close to) obstacles
                    if not _check_start_target_clearance(obstacle, start_pos, target_pos, buffer=ROBOT_RADIUS):
                        continue

                    manager.add(obstacle)
                    existing.append((center, radius))
                    placed = True
                    break

            if not placed:
                print(f"Warning: Could not place obstacle {i} with proper spacing")

    else:
        # Non-convex unions (level 7-10)
        # Increasing both number of unions and primitives per union
        # Level 7: 4 unions × 3 primitives = 12 primitives, 4 union obstacles
        # Level 8: 5 unions × 4 primitives = 20 primitives, 5 union obstacles
        # Level 9: 6 unions × 5 primitives = 30 primitives, 6 union obstacles
        # Level 10: 8 unions × 6 primitives = 48 primitives, 8 union obstacles
        num_unions = {7: 4, 8: 5, 9: 6, 10: 8}[level]
        primitives_per_union = {7: 3, 8: 4, 9: 5, 10: 6}[level]

        max_attempts = 100
        for union_idx in range(num_unions):
            # Create union of primitives
            primitives = []

            # Try to place union base
            placed = False
            union_radius = 0.0  # Will be computed

            for attempt in range(max_attempts):
                # Base position for union
                offset = np.array([
                    np.random.uniform(-region_size / 2, region_size / 2),
                    np.random.uniform(-region_size / 2, region_size / 2)
                ], dtype=np.float32)
                base_center = center_region + offset

                # Create primitives for this union
                primitives = []
                for prim_idx in range(primitives_per_union):
                    # Relative position within union
                    offset_prim = np.array([
                        np.random.uniform(-0.3, 0.3),
                        np.random.uniform(-0.3, 0.3)
                    ], dtype=np.float32)
                    center = base_center + offset_prim

                    # Random primitive (around robot radius)
                    base_radius = ROBOT_RADIUS * OBSTACLE_RADIUS_SCALE
                    if np.random.rand() < 0.6:
                        radius_prim = np.random.uniform(base_radius * 0.5, base_radius * 1.2)
                        prim = SphereObstacle(center=center, radius=radius_prim,
                                              name=f"union_{union_idx}_circle_{prim_idx}")
                    else:
                        size_prim = np.random.uniform(base_radius * 0.5, base_radius * 1.2)
                        half_extents = np.array([size_prim, size_prim], dtype=np.float32)
                        prim = BoxObstacle(center=center, half_extents=half_extents,
                                           name=f"union_{union_idx}_box_{prim_idx}")

                    primitives.append(prim)

                # Compute union radius (max distance from base to any primitive edge)
                union_radius = 0.0
                for prim in primitives:
                    prim_center = prim.center if hasattr(prim, 'center') else np.array([0, 0])
                    prim_radius = _get_obstacle_radius(prim)
                    dist_to_base = np.linalg.norm(prim_center - base_center)
                    union_radius = max(union_radius, dist_to_base + prim_radius)

                base_center = np.clip(base_center, -p_max + union_radius + margin, p_max - union_radius - margin)

                # Check spacing (use base_center and union_radius)
                if _check_obstacle_spacing(base_center, union_radius, existing, margin):
                    # Create union
                    union = UnionObstacle(obstacles=primitives, name=f"union_{union_idx}")

                    # Hard guarantee: start/target cannot be in (or too close to) obstacles
                    if not _check_start_target_clearance(union, start_pos, target_pos, buffer=ROBOT_RADIUS):
                        continue

                    manager.add(union)
                    existing.append((base_center, union_radius))
                    placed = True
                    break

            if not placed:
                print(f"Warning: Could not place union {union_idx} with proper spacing")

    # Final sanity check: start position must be safe w.r.t. all obstacles
    if len(manager) > 0:
        sdf_start = manager.sdf(np.asarray(start_pos, dtype=np.float32))
        sdf_start = float(np.asarray(sdf_start).item() if hasattr(sdf_start, "item") else sdf_start)
        if sdf_start < ROBOT_RADIUS or manager.contains(np.asarray(start_pos, dtype=np.float32)):
            raise RuntimeError(
                f"Start position is in/too close to an obstacle (sdf={sdf_start:.6f}). "
                "Obstacle generation should prevent this."
            )

    return manager


# ============================================================================
# Visualization
# ============================================================================

def visualize_diffusion_step(
        ax: plt.Axes,
        env: DoubleIntegratorBox2DEnv,
        obstacles: ObstacleManager,
        initial_state: np.ndarray,
        action_sequence: np.ndarray,
        sample_actions: np.ndarray = None,
        feasible_set_trajectory: Optional[Trajectory] = None,
        clearance: float = 0.0,
        robot_radius: float = ROBOT_RADIUS,
        title: str = None
):
    """
    Visualize diffusion rollouts and feasible sets at a specific diffusion step.
    
    Args:
        ax: Matplotlib axis
        env: Environment
        obstacles: Obstacle manager
        initial_state: Initial state
        action_sequence: Action sequence for main trajectory
        sample_actions: Sample action sequences for rollouts (num_samples, horizon, act_dim)
        robot_radius: Robot radius
        title: Optional title
    """
    ax.set_aspect('equal')
    ax.set_xlim(-env.p_max, env.p_max)
    ax.set_ylim(-env.p_max, env.p_max)

    # Draw obstacles
    draw_obstacles(ax, obstacles)

    # Draw convex feasible set (CFS) if provided
    cfs_label_added = False
    if feasible_set_trajectory is not None and len(feasible_set_trajectory.states) > 0 and clearance > 0:
        # Draw clearance circles around trajectory states (sample every few states for clarity)
        step = max(1, len(feasible_set_trajectory.states) // 10)  # Sample ~10 circles
        for idx, state in enumerate(feasible_set_trajectory.states[::step]):
            pos = np.asarray(state)[:2]
            # Draw clearance circle (feasible set approximation)
            cfs_circle = plt.Circle(
                tuple(pos),
                clearance,
                facecolor='lightgreen',
                alpha=0.12,
                edgecolor='green',
                linewidth=0.6,
                linestyle='--',
                label='CFS' if not cfs_label_added else ''
            )
            ax.add_patch(cfs_circle)
            if not cfs_label_added:
                cfs_label_added = True

    # Draw sample rollouts (if provided)
    if sample_actions is not None and len(sample_actions) > 0:
        num_samples = min(len(sample_actions), MAX_SAMPLE_TRAJ_PLOT)
        if num_samples > MAX_SAMPLE_TRAJ_PLOT:
            # Sample evenly
            indices = np.linspace(0, len(sample_actions) - 1, MAX_SAMPLE_TRAJ_PLOT, dtype=int)
            sample_actions = sample_actions[indices]

        import matplotlib.colors as mcolors
        light_rgba = mcolors.to_rgba(EDOC_COLOR, alpha=0.15)
        for acts in sample_actions:
            states = env.rollout_actions(initial_state, acts)
            if len(states) > 1:
                ax.plot(states[:, 0], states[:, 1], color=light_rgba, linewidth=0.8)

    # Draw main trajectory
    if action_sequence is not None and len(action_sequence) > 0:
        states = env.rollout_actions(initial_state, action_sequence)
        if len(states) > 1:
            ax.plot(states[:, 0], states[:, 1], color=EDOC_COLOR, linewidth=2.5)
            # Draw start
            ax.scatter(
                states[0, 0],
                states[0, 1],
                marker='o',
                s=30,
                facecolors='white',
                edgecolors=EDOC_COLOR,
                linewidths=1.0,
            )
            # Draw end
            ax.scatter(
                states[-1, 0],
                states[-1, 1],
                marker='*',
                s=55,
                facecolors=EDOC_COLOR,
                edgecolors='black',
                linewidths=0.5,
            )

    # Draw target
    target = np.asarray(env.target)
    ax.plot(target[0], target[1], 'r*', markersize=15, label='Target', zorder=10)

    if title:
        ax.set_title(title, fontsize=12)
    if cfs_label_added or len(sample_actions) > 0 if sample_actions is not None else False:
        ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)


def draw_obstacles(ax: plt.Axes, obstacles: ObstacleManager):
    """Draw obstacles on axis."""
    for obstacle in obstacles:
        if isinstance(obstacle, (SphereObstacle, CircleObstacle)):
            # Handle sphere/circle obstacle
            center = np.asarray(obstacle.center, dtype=np.float32)
            if center.ndim == 0:
                # Scalar, convert to array
                center = np.array([center, 0.0], dtype=np.float32)
            elif len(center) > 2:
                center = center[:2]
            elif len(center) == 1:
                # 1D case, pad with 0
                center = np.concatenate([center, [0.0]])

            radius = float(obstacle.radius)
            circle = plt.Circle(
                tuple(center),
                radius,
                facecolor=OBSTACLE_COLOR,
                alpha=OBSTACLE_ALPHA,
                edgecolor='darkgray',
                linewidth=1.5
            )
            ax.add_patch(circle)

        elif isinstance(obstacle, BoxObstacle):
            # Handle box obstacle
            center = np.asarray(obstacle.center, dtype=np.float32)
            half_extents = np.asarray(obstacle.half_extents, dtype=np.float32)

            # Ensure 2D
            if center.ndim == 0:
                center = np.array([center, 0.0], dtype=np.float32)
            elif len(center) > 2:
                center = center[:2]
            elif len(center) == 1:
                center = np.concatenate([center, [0.0]])

            if half_extents.ndim == 0:
                half_extents = np.array([half_extents, half_extents], dtype=np.float32)
            elif len(half_extents) > 2:
                half_extents = half_extents[:2]
            elif len(half_extents) == 1:
                half_extents = np.array([half_extents[0], half_extents[0]], dtype=np.float32)

            rect = mpatches.Rectangle(
                tuple(center - half_extents),
                2 * float(half_extents[0]),
                2 * float(half_extents[1]),
                facecolor=OBSTACLE_COLOR,
                alpha=OBSTACLE_ALPHA,
                edgecolor='darkgray',
                linewidth=1.5
            )
            ax.add_patch(rect)

        elif isinstance(obstacle, UnionObstacle):
            # Draw all primitives in union
            for prim in obstacle.obstacles:
                if isinstance(prim, (SphereObstacle, CircleObstacle)):
                    center = np.asarray(prim.center, dtype=np.float32)
                    if center.ndim == 0:
                        center = np.array([center, 0.0], dtype=np.float32)
                    elif len(center) > 2:
                        center = center[:2]
                    elif len(center) == 1:
                        center = np.concatenate([center, [0.0]])

                    radius = float(prim.radius)
                    circle = plt.Circle(
                        tuple(center),
                        radius,
                        facecolor=OBSTACLE_COLOR,
                        alpha=OBSTACLE_ALPHA * 0.8,
                        edgecolor='darkgray',
                        linewidth=1.2
                    )
                    ax.add_patch(circle)

                elif isinstance(prim, BoxObstacle):
                    center = np.asarray(prim.center, dtype=np.float32)
                    half_extents = np.asarray(prim.half_extents, dtype=np.float32)

                    # Ensure 2D
                    if center.ndim == 0:
                        center = np.array([center, 0.0], dtype=np.float32)
                    elif len(center) > 2:
                        center = center[:2]
                    elif len(center) == 1:
                        center = np.concatenate([center, [0.0]])

                    if half_extents.ndim == 0:
                        half_extents = np.array([half_extents, half_extents], dtype=np.float32)
                    elif len(half_extents) > 2:
                        half_extents = half_extents[:2]
                    elif len(half_extents) == 1:
                        half_extents = np.array([half_extents[0], half_extents[0]], dtype=np.float32)

                    rect = mpatches.Rectangle(
                        tuple(center - half_extents),
                        2 * float(half_extents[0]),
                        2 * float(half_extents[1]),
                        facecolor=OBSTACLE_COLOR,
                        alpha=OBSTACLE_ALPHA * 0.8,
                        edgecolor='darkgray',
                        linewidth=1.2
                    )
                    ax.add_patch(rect)


def generate_optimized_start_position(level: int, seed: int, target: np.ndarray,
                                      p_max: float) -> np.ndarray:
    """
    Generate optimized start position based on level.
    
    For higher levels (more obstacles), start farther from target.
    Maximum distance: 2 * sqrt(2) ≈ 2.83
    
    Args:
        level: Obstacle level (0-10)
        seed: Random seed
        target: Target position
        p_max: Environment bounds
        
    Returns:
        Start position
    """
    np.random.seed(seed)
    target = np.asarray(target)

    # Distance from target increases with level
    # Level 0: close (0.5-1.0)
    # Level 10: far (1.5 - 2*sqrt(2) ≈ 2.83)
    max_distance = 2.0 * np.sqrt(2.0)  # Maximum distance
    min_dist = 0.5 + (level / 10.0) * 1.0
    max_dist = 1.0 + (level / 10.0) * (max_distance - 1.0)
    distance = np.random.uniform(min_dist, max_dist)

    # Random angle
    angle = np.random.uniform(0, 2 * np.pi)
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float32)

    start = target + distance * direction

    # Clip to bounds (with some margin)
    margin = 0.2
    start = np.clip(start, -p_max + margin, p_max - margin)

    return start


def visualize_trajectory(
        ax: plt.Axes,
        env: DoubleIntegratorBox2DEnv,
        obstacles: ObstacleManager,
        trajectory: Trajectory,
        robot_radius: float = ROBOT_RADIUS,
        title: str = "Final Trajectory"
):
    """Visualize final trajectory."""
    ax.set_aspect('equal')
    ax.set_xlim(-env.p_max, env.p_max)
    ax.set_ylim(-env.p_max, env.p_max)

    # Draw obstacles
    draw_obstacles(ax, obstacles)

    # Draw trajectory
    if len(trajectory.states) > 1:
        positions = np.array([np.asarray(s)[:2] for s in trajectory.states])
        ax.plot(positions[:, 0], positions[:, 1], color=EDOC_COLOR, linewidth=2.0, label='Trajectory')

        # Draw start with robot radius
        start_circle = plt.Circle(
            tuple(positions[0]),
            robot_radius,
            facecolor='white',
            edgecolor=EDOC_COLOR,
            linewidth=1.5,
            label='Start'
        )
        ax.add_patch(start_circle)

        # Draw end with robot radius
        end_circle = plt.Circle(
            tuple(positions[-1]),
            robot_radius,
            facecolor=EDOC_COLOR,
            edgecolor='black',
            linewidth=0.5
        )
        ax.add_patch(end_circle)

    # Draw target with margin (robot diameter)
    target = np.asarray(env.target)
    target_circle = plt.Circle(
        tuple(target),
        robot_radius * 2,  # Robot diameter
        facecolor='none',
        edgecolor='red',
        linewidth=1.5,
        linestyle='--',
        label='Target (margin)'
    )
    ax.add_patch(target_circle)
    ax.plot(target[0], target[1], 'r*', markersize=15, label='Target')

    ax.set_title(title, fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_energy_reward(
        ax_energy: plt.Axes,
        ax_reward: plt.Axes,
        energies: np.ndarray,
        rewards: np.ndarray,
        diffusion_history: Dict[str, Any] = None,
        dt: float = 0.1
):
    """Plot energy and reward over time."""
    time = np.arange(len(energies)) * dt

    ax_energy.plot(time, energies, color=EDOC_COLOR, linewidth=2.0, label='Energy')
    ax_energy.set_xlabel('Time (s)')
    ax_energy.set_ylabel('Energy')
    ax_energy.set_title('Energy Over Time')
    ax_energy.legend()
    ax_energy.grid(True, alpha=0.3)

    time_rewards = np.arange(len(rewards)) * dt
    ax_reward.plot(time_rewards, rewards, color=EDOC_COLOR, linewidth=2.0, label='Reward')
    if diffusion_history is not None and 'reward_history' in diffusion_history:
        reward_hist = diffusion_history['reward_history']
        if len(reward_hist) > 0:
            diffusion_steps = np.linspace(0, time_rewards[-1] if len(time_rewards) > 0 else 0, len(reward_hist))
            ax_reward.plot(diffusion_steps, reward_hist, '--', color=EDOC_COLOR, linewidth=1, alpha=0.5,
                           label='Diffusion Reward')
    ax_reward.set_xlabel('Time (s)')
    ax_reward.set_ylabel('Reward')
    ax_reward.set_title('Reward Over Time')
    ax_reward.legend()
    ax_reward.grid(True, alpha=0.3)


def plot_states(
        axes: List[plt.Axes],
        trajectory: Trajectory,
        dt: float = 0.1
):
    """Plot state components over time in 4 subplots."""
    if len(trajectory.states) == 0:
        return

    states_array = np.array([np.asarray(s) for s in trajectory.states])
    time = np.arange(len(trajectory.states)) * dt

    labels = [
        ("x position", 0),
        ("y position", 1),
        ("x velocity", 2),
        ("y velocity", 3),
    ]

    for ax, (title, idx) in zip(axes, labels):
        ax.plot(time, states_array[:, idx], color=EDOC_COLOR, linewidth=2.0)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)


# ============================================================================
# JSON Serialization Helper
# ============================================================================

def convert_to_json_serializable(obj):
    """
    Convert numpy types and other non-JSON-serializable types to Python native types.
    
    Args:
        obj: Object to convert
        
    Returns:
        JSON-serializable object
    """
    if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8,
                        np.int16, np.int32, np.int64, np.uint8, np.uint16,
                        np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    else:
        return obj


# ============================================================================
# SSR Computation
# ============================================================================

def compute_ssr(
        trajectory: Trajectory,
        env: DoubleIntegratorBox2DEnv,
        obstacles: ObstacleManager,
        accel_constraint: AccelerationConstraint,
        robot_radius: float = ROBOT_RADIUS,
        success_margin: float = None  # If None, uses 2 * robot_radius (robot diameter)
) -> Dict[str, Any]:
    """
    Compute Safety Success Rate metrics.
    
    SSR = 1 if trajectory is safe AND reaches target, else 0.
    
    Args:
        trajectory: Final trajectory
        env: Environment
        obstacles: Obstacle manager
        accel_constraint: Acceleration constraint
        robot_radius: Robot radius for collision checking
        success_margin: Distance threshold for success (robot diameter if None)
        
    Returns:
        Dictionary with SSR and metrics
    """
    if success_margin is None:
        success_margin = 2 * robot_radius  # Robot diameter

    # Check safety: no collisions (considering robot radius)
    safe = True
    if len(obstacles) > 0:  # Only check if there are obstacles
        for state in trajectory.states:
            pos = np.asarray(state)[:2]
            # Check if robot (with radius) collides with obstacles
            sdf = obstacles.sdf(pos)
            if sdf < robot_radius:  # Robot overlaps with obstacle
                safe = False
                break
            if obstacles.contains(pos):
                safe = False
                break

    # Check acceleration constraint
    accel_feasible = bool(accel_constraint.is_feasible(trajectory))

    # Check task success: reached target within margin (robot diameter)
    if len(trajectory.states) > 0:
        final_pos = np.asarray(trajectory.states[-1])[:2]
        target = np.asarray(env.target)
        distance_to_target = float(np.linalg.norm(final_pos - target))
        task_success = bool(distance_to_target < success_margin)
    else:
        task_success = False
        distance_to_target = float('inf')

    # SSR: safe AND task success
    ssr = 1.0 if (safe and accel_feasible and task_success) else 0.0

    return {
        'ssr': float(ssr),
        'safe': bool(safe),
        'accel_feasible': bool(accel_feasible),
        'task_success': bool(task_success),
        'distance_to_target': float(distance_to_target),
    }


# ============================================================================
# Main Experiment
# ============================================================================

def run_single_experiment(
        level: int,
        seed: int,
        output_dir: Path,
        edoc_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Run single experiment with given level and seed.
    
    Returns:
        Dictionary with results
    """
    print(f"\n{'=' * 60}")
    print(f"Starting Experiment: Level {level}, Seed {seed}")
    print(f"{'=' * 60}")
    experiment_start_time = time.time()

    # Create environment
    env = DoubleIntegratorBox2DEnv(
        dt=edoc_config.get('dt', 0.1),
        horizon=edoc_config.get('horizon', 80),
        p_max=2.0,
        v_max=2.0,
    )

    # Generate optimized start position and target
    target = np.asarray(env.target, dtype=np.float32)
    start_pos = generate_optimized_start_position(level, seed, target, env.p_max)
    print(f"  Start position: [{start_pos[0]:.3f}, {start_pos[1]:.3f}]")
    print(f"  Target position: [{target[0]:.3f}, {target[1]:.3f}]")
    print(f"  Distance: {np.linalg.norm(start_pos - target):.3f}")

    # Generate obstacles (between start and target)
    print(f"  Generating obstacles...")
    obstacles = generate_obstacle_config(level, seed, start_pos, target)
    print(f"  Generated {len(obstacles)} obstacle(s)")

    # Build MDOC-style SDF texture for fast (sdf, grad) queries (used by CBF-style hard layer)
    if level > 0 and len(obstacles) > 0:
        # Match MDOC structure: precompute sdf + grad on a dense grid and sample by bilinear interpolation.
        # NOTE: This is an approximation (trade accuracy for speed in inner loop).
        obstacles.build_sdf_texture_2d(
            x_min=-env.p_max,
            x_max=env.p_max,
            y_min=-env.p_max,
            y_max=env.p_max,
            res=0.01,
            force_rebuild=True,
        )

        # Check if obstacles fit in bounds (especially for level 10)
    if level > 0:
        all_obstacles = list(obstacles)
        max_extent = 0.0
        min_clearance_to_boundary = float('inf')

        for obs in all_obstacles:
            if hasattr(obs, 'center'):
                center = np.asarray(obs.center)[:2]
                obs_radius = _get_obstacle_radius(obs)
                # Distance to boundary
                dist_to_boundary = env.p_max - np.max(np.abs(center)) - obs_radius
                min_clearance_to_boundary = min(min_clearance_to_boundary, dist_to_boundary)

                extent = np.max(np.abs(center)) + obs_radius
                max_extent = max(max_extent, extent)

        if min_clearance_to_boundary < 0.1:
            print(f"  WARNING: Some obstacles too close to boundaries (min clearance: {min_clearance_to_boundary:.3f})")
        else:
            print(
                f"  Obstacles fit within bounds (max extent: {max_extent:.3f}, min clearance: {min_clearance_to_boundary:.3f})")

    # Create energy functional
    energy = make_energy("double_integrator_box_2d")

    # Create constraints (skip obstacle constraints for level 0)
    soft_constraint = None
    hard_constraint = None
    if level > 0:
        soft_constraint = ObstacleSoftConstraint(
            obstacles=obstacles,
            alpha=1.0,
            beta=10.0
        )

        hard_constraint = ObstacleHardConstraint(
            obstacles=obstacles,
            clearance=0.1
        )

    accel_constraint = AccelerationConstraint(u_max=1.0)

    # Create schedule manager (soft → hard, only for level > 0)
    schedule_manager = None
    if level > 0:
        schedule_manager = ConstraintScheduleManager.create_soft_to_hard(
            soft_alpha_start=1.0,
            soft_alpha_end=0.0,
            hard_clearance_start=0.5,
            hard_clearance_end=0.1,
            schedule_type="linear",
            reverse_mode=True
        )

        # Link schedule to constraints (if they exist)
        if soft_constraint is not None:
            soft_constraint.schedule_manager = schedule_manager
        if hard_constraint is not None:
            hard_constraint.schedule_manager = schedule_manager

    # Create action-space hard filter (CBF-style) (only for level > 0)
    action_filter_op = None
    if level > 0:
        action_filter_op = CBFDoubleIntegrator2DActionFilter(
            obstacles=obstacles,
            robot_radius=ROBOT_RADIUS,
            dt=env.dt,
            tau=0.05,
            k0=1.0,
            k1=4.0,
            schedule_manager=schedule_manager,
        )
    action_filter_enabled = action_filter_op is not None

    # Create constraint manager
    soft_constraints_list = [soft_constraint] if soft_constraint is not None else []
    hard_constraints_list = []
    # Obstacle avoidance is enforced by the action-space filter; keep hard obstacle constraint optional.
    hard_constraints_list.append(accel_constraint)

    constraint_manager = ConstraintManager(
        soft_constraints=soft_constraints_list,
        hard_constraints=hard_constraints_list,
        feasibility_operator=None,
        action_filter_operator=action_filter_op if level > 0 else None,
        schedule_manager=schedule_manager if level > 0 else None
    )

    # Create EDOC planner with constraint support
    planner = EDOCPlanner(
        env=env,
        energy=energy,
        horizon=env.horizon,
        dt=env.dt,
        action_space=True,
        diffusion_mode="reverse",
        action_diffuse_steps=edoc_config.get('action_diffuse_steps', 100),
        action_nsample=edoc_config.get('action_nsample', 256),
        use_antithetic=edoc_config.get('use_antithetic', True),
        action_score_mode="energy",
        constraint_manager=constraint_manager,
        use_constraint_in_scoring=True,
        lambda_energy=1.0,
    )

    # Set initial state (use optimized start position)
    initial_state_full = np.concatenate([start_pos, np.zeros(2, dtype=np.float32)])  # [px, py, vx, vy]

    # Run planning with custom initial state
    print(f"  Starting EDOC planning...")
    rng = jax.random.PRNGKey(seed)
    # Temporarily override env reset to use our start position
    original_reset = env.reset

    def custom_reset(rng=None):
        return initial_state_full.copy(), {}

    env.reset = custom_reset

    start_time = time.time()
    result = planner.plan(rng)
    planning_time = time.time() - start_time

    # Restore original reset
    env.reset = original_reset
    print(f"  Planning completed in {planning_time:.2f} seconds")

    # Extract trajectory
    states = result.get('states', [])
    actions = result.get('actions', None)

    if actions is None or len(actions) == 0:
        # Generate dummy actions
        actions = [np.zeros(env.act_dim, dtype=np.float32) for _ in range(len(states) - 1)]

    # Convert to Trajectory
    states_list = [np.asarray(s) for s in states]
    actions_list = [np.asarray(a) for a in actions]
    trajectory = Trajectory(states=states_list, actions=actions_list)

    # Apply constraints post-hoc (for visualization)
    # In full implementation, constraints would be integrated into EDOC

    # Compute SSR (with robot radius)
    ssr_metrics = compute_ssr(trajectory, env, obstacles, accel_constraint, robot_radius=ROBOT_RADIUS)

    # Print important results
    print(f"  Results:")
    print(f"    SSR: {ssr_metrics['ssr']:.3f}")
    print(f"    Safe: {ssr_metrics['safe']}")
    print(f"    Task Success: {ssr_metrics['task_success']}")
    print(f"    Distance to Target: {ssr_metrics['distance_to_target']:.4f}")
    print(f"    Accel Feasible: {ssr_metrics['accel_feasible']}")

    # Prepare results
    results = {
        'level': level,
        'seed': seed,
        'ssr': ssr_metrics['ssr'],
        'safe': ssr_metrics['safe'],
        'accel_feasible': ssr_metrics['accel_feasible'],
        'task_success': ssr_metrics['task_success'],
        'distance_to_target': float(ssr_metrics['distance_to_target']),
        'planning_time': float(planning_time),
        'action_filter_enabled': bool(action_filter_enabled),
        'trajectory': {
            'states': [s.tolist() for s in states_list],
            'actions': [a.tolist() for a in actions_list],
        },
        'energies': result.get('energies', []).tolist() if hasattr(result.get('energies', []), 'tolist') else [],
        'rewards': result.get('rewards', []).tolist() if hasattr(result.get('rewards', []), 'tolist') else [],
    }

    # Save visualizations
    level_dir = output_dir / f"level_{level}" / f"seed_{seed}"
    level_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Generating visualizations...")

    # Create visualizations
    fig_size = (15, 12)

    # Diffusion steps visualization (10%, 50%, 90%)
    fig_diff, axes_diff = plt.subplots(1, 3, figsize=(18, 6))
    diffusion_actions = result.get('diffusion_actions_traj', None)
    diffusion_samples = result.get('diffusion_sampled_actions', None)

    if diffusion_actions is not None and len(diffusion_actions) > 0:
        diffusion_actions = np.asarray(diffusion_actions, dtype=np.float32)
        Ndiffuse = diffusion_actions.shape[0]

        for ax, frac in zip(axes_diff, DIFFUSION_FRACTIONS):
            step_idx = int((1.0 - frac) * (Ndiffuse - 1))  # Convert to reverse diffusion index
            step_idx = max(0, min(step_idx, Ndiffuse - 1))

            action_seq = diffusion_actions[step_idx]
            sample_acts = None
            if diffusion_samples is not None and len(diffusion_samples) > 0:
                diffusion_samples_arr = np.asarray(diffusion_samples, dtype=np.float32)
                if diffusion_samples_arr.ndim == 4 and diffusion_samples_arr.shape[0] > step_idx:
                    sample_acts = diffusion_samples_arr[step_idx]

            visualize_diffusion_step(
                ax, env, obstacles, initial_state_full, action_seq, sample_acts,
                feasible_set_trajectory=None,
                clearance=0.0,
                robot_radius=ROBOT_RADIUS,
                title=f"Diffusion {step_idx}%"
            )
    else:
        # Fallback: show final trajectory 3 times
        for ax, frac in zip(axes_diff, DIFFUSION_FRACTIONS):
            visualize_diffusion_step(
                ax, env, obstacles, initial_state_full,
                np.array(actions_list, dtype=np.float32) if actions_list else None,
                None,
                feasible_set_trajectory=None,
                clearance=0.0,
                robot_radius=ROBOT_RADIUS,
                title=f"Diffusion {int(frac * 100)}% (Final)"
            )

    plt.tight_layout()
    plt.savefig(level_dir / "diffusion_steps.png", dpi=150, bbox_inches='tight')
    plt.close(fig_diff)

    # Trajectory
    fig_traj, ax_traj = plt.subplots(1, 1, figsize=(8, 8))
    visualize_trajectory(ax_traj, env, obstacles, trajectory)
    plt.savefig(level_dir / "trajectory.png", dpi=150, bbox_inches='tight')
    plt.close(fig_traj)

    # Energy and reward
    fig_er, (ax_energy, ax_reward) = plt.subplots(1, 2, figsize=(12, 5))
    energies_array = np.array(results['energies']) if len(results['energies']) > 0 else np.array([0.0])
    rewards_array = np.array(results['rewards']) if len(results['rewards']) > 0 else np.array([0.0])
    plot_energy_reward(ax_energy, ax_reward, energies_array, rewards_array, dt=env.dt)
    plt.savefig(level_dir / "energy_reward.png", dpi=150, bbox_inches='tight')
    plt.close(fig_er)

    # States (4 subplots: px, py, vx, vy)
    fig_states, axes_states = plt.subplots(2, 2, figsize=(12, 8))
    axes_states = axes_states.ravel()
    plot_states(axes_states, trajectory, dt=env.dt)
    plt.tight_layout()
    plt.savefig(level_dir / "states.png", dpi=150, bbox_inches='tight')
    plt.close(fig_states)

    # Save results JSON (convert numpy types to Python native types)
    results_serializable = convert_to_json_serializable(results)
    with open(level_dir / "results.json", 'w') as f:
        json.dump(results_serializable, f, indent=2)

    total_experiment_time = time.time() - experiment_start_time
    print(f"  Results saved to {level_dir}")
    print(f"  Total experiment time: {total_experiment_time:.2f}s")
    print(f"{'=' * 60}\n")

    return results


def run_all_experiments(
        output_dir: Path = Path("results/double2d"),
        num_seeds: int = 10,
        edoc_config: Dict[str, Any] = None
):
    """
    Run all experiments across all levels and seeds.
    
    Args:
        output_dir: Output directory for results
        num_seeds: Number of random seeds per level
        edoc_config: EDOC configuration
    """
    if edoc_config is None:
        edoc_config = {
            'dt': 0.05,
            'horizon': 80,
            'action_diffuse_steps': 100,
            'action_nsample': 256,
            'use_antithetic': True,
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"Starting All Experiments")
    print(f"  Output directory: {output_dir}")
    print(f"  Levels: 0-10 ({11} levels)")
    print(f"  Seeds per level: {num_seeds}")
    print(f"  Total experiments: {11 * num_seeds}")
    print(f"  EDOC config: {edoc_config}")
    print(f"{'#' * 70}\n")

    # Run experiments
    all_results = []

    for level in range(0, 11):  # Levels 0-10 (0 = no obstacles)
        print(f"\n>>> Processing Level {level} <<<")
        level_results = []
        for seed in range(num_seeds):
            try:
                print(f"  Seed {seed}/{num_seeds - 1}...")
                result = run_single_experiment(level, seed, output_dir, edoc_config)
                level_results.append(result)
                all_results.append(result)
            except Exception as e:
                print(f"  ERROR in Seed {seed}: {e}")
                import traceback
                traceback.print_exc()

        # Compute SSR for level
        if level_results:
            ssr_values = [r['ssr'] for r in level_results]
            level_ssr = np.mean(ssr_values)
            planning_times = [r.get('planning_time', 0) for r in level_results]
            avg_time = np.mean(planning_times)

            print(f"\n  Level {level} Summary:")
            print(f"    SSR: {level_ssr:.3f} ({np.sum(ssr_values)}/{len(ssr_values)} successes)")
            print(f"    Avg Planning Time: {avg_time:.2f}s")

            # Save level summary
            level_summary = {
                'level': level,
                'ssr': float(level_ssr),
                'num_success': int(np.sum(ssr_values)),
                'num_total': len(level_results),
                'results': level_results,
            }

            level_summary_serializable = convert_to_json_serializable(level_summary)
            with open(output_dir / f"level_{level}" / "summary.json", 'w') as f:
                json.dump(level_summary_serializable, f, indent=2)

        # Compute overall SSR
    if all_results:
        overall_ssr = np.mean([r['ssr'] for r in all_results])
        total_planning_time = sum([r.get('planning_time', 0) for r in all_results])
        print(f"\n{'#' * 70}")
        print(f"Overall Results:")
        print(f"  Overall SSR: {overall_ssr:.3f}")
        print(f"  Total Planning Time: {total_planning_time:.2f}s")
        print(f"  Average Planning Time: {total_planning_time / len(all_results):.2f}s per experiment")
        print(f"{'#' * 70}\n")

        # Save overall summary
        overall_summary = {
            'overall_ssr': float(overall_ssr),
            'total_experiments': len(all_results),
            'level_ssrs': {}
        }

        for level in range(0, 11):  # Levels 0-10
            level_results = [r for r in all_results if r['level'] == level]
            if level_results:
                level_ssr = np.mean([r['ssr'] for r in level_results])
                overall_summary['level_ssrs'][f'level_{level}'] = float(level_ssr)

        overall_summary_serializable = convert_to_json_serializable(overall_summary)
        with open(output_dir / "overall_summary.json", 'w') as f:
            json.dump(overall_summary_serializable, f, indent=2)

        # Plot SSR by level
        fig_ssr, ax_ssr = plt.subplots(1, 1, figsize=(10, 6))
        level_numbers = sorted([int(k.split('_')[1]) for k in overall_summary['level_ssrs'].keys()])
        ssrs = [overall_summary['level_ssrs'][f'level_{l}'] for l in level_numbers]
        ax_ssr.bar(level_numbers, ssrs, alpha=0.7, color=EDOC_COLOR)
        ax_ssr.set_xlabel('Level')
        ax_ssr.set_ylabel('SSR')
        ax_ssr.set_title('Safety Success Rate by Level')
        ax_ssr.set_xticks(level_numbers)
        ax_ssr.grid(True, alpha=0.3, axis='y')
        plt.savefig(output_dir / "ssr_by_level.png", dpi=150, bbox_inches='tight')
        plt.close(fig_ssr)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run EDOC experiments on Double Integrator 2D")
    parser.add_argument("--output_dir", type=str, default="results/double2d", help="Output directory")
    parser.add_argument("--num_seeds", type=int, default=10, help="Number of seeds per level")
    parser.add_argument("--level", type=int, default=None, help="Run single level only")
    parser.add_argument("--seed", type=int, default=None, help="Run single seed only")

    args = parser.parse_args()

    if args.level is not None and args.seed is not None:
        # Single experiment
        output_dir = Path(args.output_dir)
        edoc_config = {
            'dt': 0.07,
            'horizon': 64,
            'action_diffuse_steps': 100,
            'action_nsample': 64,
            'use_antithetic': True,
        }
        run_single_experiment(args.level, args.seed, output_dir, edoc_config)
    else:
        # All experiments
        run_all_experiments(
            output_dir=Path(args.output_dir),
            num_seeds=args.num_seeds
        )
