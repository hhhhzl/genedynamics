"""
MPPI (Model Predictive Path Integral) solver implementation.

MPPI is a sampling-based control algorithm that uses importance sampling
to find optimal trajectories. It's particularly effective for systems
with complex dynamics and cost landscapes.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import jax
import jax.numpy as jnp

from enerdynamics.core.solvers import SamplingSolver
from enerdynamics.core.dynamics import DynamicsModel, DynamicsToEnvAdapter
from enerdynamics.core.energy import EnergyFunctional, LegacyEnergyFunctional
from enerdynamics.core.backends import Backend, JaxBackend
from enerdynamics.core.types import State, Action, Trajectory
from enerdynamics.solvers.single.edoc import EnergyToLegacyAdapter


class MPPISolver(SamplingSolver):
    """
    MPPI solver using path integral control.
    
    MPPI samples trajectories from a proposal distribution and uses
    importance weighting to select optimal actions.
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        horizon: int = 80,
        dt: float = 0.1,
        num_samples: int = 512,
        num_iterations: int = 6,
        noise_sigma: float = 0.3,
        lambda_: float = 1.0,
        action_limit: float = 1.0,
        seed: int = 0,
        **kwargs
    ):
        """
        Initialize MPPI solver.
        
        Args:
            dynamics: Dynamics model
            energy: Energy functional
            backend: Computational backend (must be JAX for MPPI)
            horizon: Planning horizon
            dt: Time step
            num_samples: Number of trajectory samples per iteration
            num_iterations: Number of optimization iterations
            noise_sigma: Standard deviation of noise for sampling
            lambda_: Temperature parameter for importance weighting
            action_limit: Action limit (clipping)
            seed: Random seed
            **kwargs: Additional MPPI configuration
        """
        super().__init__(dynamics, energy, backend, **kwargs)
        
        # MPPI currently requires JAX backend
        if backend.name != "jax":
            raise ValueError(
                f"MPPI solver requires JAX backend, got {backend.name}. "
                "Use backend=get_backend('jax') when creating the solver."
            )
        
        # Store configuration
        self.horizon = horizon
        self.dt = dt
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.noise_sigma = noise_sigma
        self.lambda_ = lambda_
        self.action_limit = action_limit
        self.seed = seed
        
        # Create adapters
        self._env_adapter = DynamicsToEnvAdapter(dynamics, dt)
        
        # Convert energy to legacy format if needed
        if isinstance(energy, LegacyEnergyFunctional):
            self._legacy_energy = energy
        else:
            self._legacy_energy = EnergyToLegacyAdapter(energy, dynamics).legacy_energy
        
        # Infer action dimension
        self.act_dim = self._env_adapter.act_dim
        
        # Build JAX functions
        self._build_jax_functions()
    
    def _build_jax_functions(self):
        """Build JAX-compiled functions for rollout and scoring."""
        env = self._env_adapter
        
        # Transition function
        def transition_fn(state, action):
            return env.jax_transition(state, action)
        
        self._transition_fn = jax.jit(transition_fn)
        
        # Cost function (from energy)
        def cost_fn(state, action, ctx):
            return self._legacy_energy.compute(state, action, ctx)
        
        self._cost_fn = jax.jit(cost_fn)
        
        # Rollout rewards
        def rollout_rewards(state_init, actions):
            def step_fn(carry, action):
                next_state = self._transition_fn(carry, action)
                ctx = {"t": 0}  # Simplified context
                reward = -self._cost_fn(next_state, action, ctx)
                return next_state, reward
            
            _, rewards = jax.lax.scan(step_fn, state_init, actions)
            return rewards
        
        self._rollout_rewards_fn = jax.jit(rollout_rewards)
        self._rollout_rewards_batch_fn = jax.jit(jax.vmap(self._rollout_rewards_fn, in_axes=(None, 0)))
        
        # Rollout states
        def rollout_states(state_init, actions):
            def step_fn(carry, action):
                next_state = self._transition_fn(carry, action)
                return next_state, next_state
            
            _, states = jax.lax.scan(step_fn, state_init, actions)
            return jnp.concatenate([state_init[None, :], states], axis=0)
        
        self._rollout_states_fn = jax.jit(rollout_states)
    
    def sample_trajectories(
        self,
        x0: State,
        horizon: int,
        n_samples: int,
        **kwargs
    ) -> List[Trajectory]:
        """
        Sample trajectories using MPPI importance sampling.
        
        Args:
            x0: Initial state
            horizon: Planning horizon
            n_samples: Number of samples
            **kwargs: Additional parameters
            
        Returns:
            List of candidate trajectories
        """
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        x0_jnp = jnp.asarray(x0_data, dtype=jnp.float32)
        
        rng = jax.random.PRNGKey(self.seed)
        trajectories = []
        
        # Use current mean (zero for first iteration)
        mean_actions = jnp.zeros((horizon, self.act_dim), dtype=jnp.float32)
        
        for i in range(n_samples):
            rng, noise_key = jax.random.split(rng)
            
            # Sample noise
            noise = jax.random.normal(
                noise_key, (horizon, self.act_dim), dtype=jnp.float32
            ) * self.noise_sigma
            
            # Generate candidate
            candidate = mean_actions + noise
            if self.action_limit is not None:
                candidate = jnp.clip(candidate, -self.action_limit, self.action_limit)
            
            # Rollout
            states = self._rollout_states_fn(x0_jnp, candidate)
            
            # Convert to Trajectory
            states_list = [np.asarray(states[j], dtype=np.float32) for j in range(len(states))]
            actions_list = [np.asarray(candidate[j], dtype=np.float32) for j in range(len(candidate))]
            
            traj = Trajectory(states=states_list, actions=actions_list)
            trajectories.append(traj)
        
        return trajectories
    
    def solve(
        self,
        x0: State,
        horizon: int,
        **kwargs
    ) -> Trajectory:
        """
        Solve for optimal trajectory using MPPI.
        
        Args:
            x0: Initial state
            horizon: Planning horizon
            **kwargs: Additional solver parameters
            
        Returns:
            Optimized trajectory
        """
        # Use provided horizon or default
        if horizon != self.horizon:
            self.horizon = horizon
        
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        x0_jnp = jnp.asarray(x0_data, dtype=jnp.float32)
        
        rng = jax.random.PRNGKey(self.seed)
        mean_actions = jnp.zeros((horizon, self.act_dim), dtype=jnp.float32)
        best_actions = mean_actions
        best_return = -jnp.inf
        
        for iteration in range(self.num_iterations):
            rng, noise_key = jax.random.split(rng)
            
            # Sample noise
            noise = jax.random.normal(
                noise_key, (self.num_samples, horizon, self.act_dim), dtype=jnp.float32
            ) * self.noise_sigma
            
            # Generate candidates
            candidates = mean_actions[None, :, :] + noise
            if self.action_limit is not None:
                candidates = jnp.clip(candidates, -self.action_limit, self.action_limit)
            
            # Evaluate candidates
            rewards_seq = self._rollout_rewards_batch_fn(x0_jnp, candidates)
            total_returns = jnp.sum(rewards_seq, axis=1)
            
            total_returns_np = np.asarray(total_returns)
            candidates_np = np.asarray(candidates)
            
            # Track best
            best_idx = int(np.argmax(total_returns_np))
            if total_returns_np[best_idx] > float(best_return):
                best_return = total_returns[best_idx]
                best_actions = candidates[best_idx]
            
            # Update mean using importance weighting
            costs_np = -total_returns_np
            beta = np.min(costs_np)
            weights = np.exp(-(costs_np - beta) / max(self.lambda_, 1e-6))
            weights_sum = np.sum(weights) + 1e-8
            mean_actions = jnp.asarray(
                np.einsum("i,ijk->jk", weights, candidates_np) / weights_sum, dtype=jnp.float32
            )
        
        # Final clip
        if self.action_limit is not None:
            best_actions = jnp.clip(best_actions, -self.action_limit, self.action_limit)
        
        # Rollout final trajectory
        states = self._rollout_states_fn(x0_jnp, best_actions)
        
        # Convert to Trajectory
        states_list = [np.asarray(states[j], dtype=np.float32) for j in range(len(states))]
        actions_list = [np.asarray(best_actions[j], dtype=np.float32) for j in range(len(best_actions))]
        
        traj = Trajectory(states=states_list, actions=actions_list)
        return traj


# ============================================================================
# Main Entry Point (for backward compatibility)
# ============================================================================

def run_mppi(args, initial_state: Optional[np.ndarray] = None):
    """
    Run MPPI planner (main entry point for backward compatibility).
    
    Args:
        args: MPPI configuration arguments (MPPIArgs from configs)
        initial_state: Optional initial state
        
    Returns:
        Dictionary with planning results
    """
    """
    Run MPPI planner (main entry point for backward compatibility).
    
    Args:
        args: MPPI configuration arguments (includes env_name)
        initial_state: Optional initial state
        
    Returns:
        Dictionary with planning results
    """
    from enerdynamics.envs.factories import make_env, make_energy
    from enerdynamics.core import get_backend
    
    # Create environment and energy
    env = make_env(args.env_name)
    energy = make_energy(args.env_name)
    
    # Set environment parameters
    if hasattr(env, "dt"):
        env.dt = args.dt
    if hasattr(env, "horizon"):
        env.horizon = args.horizon
    if hasattr(env, "control_limit"):
        env.control_limit = args.action_limit
    
    # Create dynamics adapter
    dynamics = DynamicsToEnvAdapter(env, dt=args.dt)
    
    # Create solver
    backend = get_backend("jax")
    solver = MPPISolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=args.horizon,
        dt=args.dt,
        num_samples=args.num_samples,
        num_iterations=args.num_iterations,
        noise_sigma=args.noise_sigma,
        lambda_=args.lambda_,
        action_limit=args.action_limit,
        seed=args.seed,
    )
    
    # Get initial state
    if initial_state is None:
        rng = jax.random.PRNGKey(args.seed)
        try:
            x0, _ = env.reset(rng)
        except TypeError:
            x0, _ = env.reset()
    else:
        x0 = np.asarray(initial_state, dtype=np.float32)
    
    # Solve
    trajectory = solver.solve(x0, horizon=args.horizon)
    
    # Compute rewards and energies
    states_np = np.stack([np.asarray(s, dtype=np.float32) for s in trajectory.states], axis=0)
    actions_np = np.stack([np.asarray(a, dtype=np.float32) for a in trajectory.actions], axis=0)
    
    # Compute rewards
    x0_jnp = jnp.asarray(x0, dtype=jnp.float32)
    actions_jnp = jnp.asarray(actions_np, dtype=jnp.float32)
    rewards = solver._rollout_rewards_fn(x0_jnp, actions_jnp)
    
    # Compute energies
    energy_list = []
    for t in range(actions_np.shape[0]):
        state_t = states_np[t]
        action_t = actions_np[t]
        ctx = {"t": int(t)}
        E_val = energy.compute(state_t, action_t, ctx)
        energy_list.append(float(E_val))
    energies = np.asarray(energy_list, dtype=np.float32)
    
    result = {
        "actions": actions_np,
        "states": states_np,
        "rewards": np.asarray(rewards, dtype=np.float32),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)) if rewards.size > 0 else 0.0,
        "initial_state": x0,
        "energies": energies,
    }
    
    if args.verbose:
        print(f"MPPI total reward: {result['total_reward']:.3f}")
    
    return result
