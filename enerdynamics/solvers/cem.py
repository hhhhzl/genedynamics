"""
CEM (Cross-Entropy Method) solver implementation.

CEM is a sampling-based optimization algorithm that iteratively refines
a distribution over actions by selecting elite samples and updating
the distribution parameters.
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


class CEMSolver(SamplingSolver):
    """
    CEM solver using cross-entropy method.
    
    CEM iteratively samples from a distribution, selects elite samples,
    and updates the distribution parameters to converge to optimal actions.
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
        elite_frac: float = 0.1,
        init_std: float = 0.5,
        min_std: float = 0.05,
        action_limit: float = 1.0,
        seed: int = 0,
        **kwargs
    ):
        """
        Initialize CEM solver.
        
        Args:
            dynamics: Dynamics model
            energy: Energy functional
            backend: Computational backend (must be JAX for CEM)
            horizon: Planning horizon
            dt: Time step
            num_samples: Number of samples per iteration
            num_iterations: Number of optimization iterations
            elite_frac: Fraction of samples to use as elites
            init_std: Initial standard deviation for sampling
            min_std: Minimum standard deviation (prevents collapse)
            action_limit: Action limit (clipping)
            seed: Random seed
            **kwargs: Additional CEM configuration
        """
        super().__init__(dynamics, energy, backend, **kwargs)
        
        # CEM currently requires JAX backend
        if backend.name != "jax":
            raise ValueError(
                f"CEM solver requires JAX backend, got {backend.name}. "
                "Use backend=get_backend('jax') when creating the solver."
            )
        
        # Store configuration
        self.horizon = horizon
        self.dt = dt
        self.num_samples = num_samples
        self.num_iterations = num_iterations
        self.elite_frac = elite_frac
        self.init_std = init_std
        self.min_std = min_std
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
        Sample trajectories using CEM distribution.
        
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
        
        # Use current distribution (mean=0, std=init_std)
        mean = jnp.zeros((horizon, self.act_dim), dtype=jnp.float32)
        std = jnp.full((horizon, self.act_dim), float(self.init_std), dtype=jnp.float32)
        
        for i in range(n_samples):
            rng, sample_key = jax.random.split(rng)
            
            # Sample from distribution
            sample = jax.random.normal(
                sample_key, (horizon, self.act_dim), dtype=jnp.float32
            ) * std + mean
            
            if self.action_limit is not None:
                sample = jnp.clip(sample, -self.action_limit, self.action_limit)
            
            # Rollout
            states = self._rollout_states_fn(x0_jnp, sample)
            
            # Convert to Trajectory
            states_list = [np.asarray(states[j], dtype=np.float32) for j in range(len(states))]
            actions_list = [np.asarray(sample[j], dtype=np.float32) for j in range(len(sample))]
            
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
        Solve for optimal trajectory using CEM.
        
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
        mean = jnp.zeros((horizon, self.act_dim), dtype=jnp.float32)
        std = jnp.full((horizon, self.act_dim), float(self.init_std), dtype=jnp.float32)
        min_std_val = float(self.min_std)
        
        elite_count = max(1, int(self.num_samples * self.elite_frac))
        best_actions = mean
        best_return = -jnp.inf
        
        for iteration in range(self.num_iterations):
            rng, sample_key = jax.random.split(rng)
            
            # Sample from current distribution
            samples = jax.random.normal(
                sample_key, (self.num_samples, horizon, self.act_dim), dtype=jnp.float32
            ) * std[None, :, :] + mean[None, :, :]
            
            if self.action_limit is not None:
                samples = jnp.clip(samples, -self.action_limit, self.action_limit)
            
            # Evaluate samples
            rewards_seq = self._rollout_rewards_batch_fn(x0_jnp, samples)
            total_returns = jnp.sum(rewards_seq, axis=1)
            
            total_returns_np = np.asarray(total_returns)
            samples_np = np.asarray(samples)
            
            # Track best
            best_idx = int(np.argmax(total_returns_np))
            if total_returns_np[best_idx] > float(best_return):
                best_return = total_returns[best_idx]
                best_actions = samples[best_idx]
            
            # Select elites and update distribution
            elite_idx = np.argsort(total_returns_np)[-elite_count:]
            elites = samples_np[elite_idx]
            mean = jnp.asarray(np.mean(elites, axis=0), dtype=jnp.float32)
            std = jnp.asarray(np.clip(np.std(elites, axis=0), min_std_val, None), dtype=jnp.float32)
        
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

def run_cem(args, initial_state: Optional[np.ndarray] = None):
    """
    Run CEM planner (main entry point for backward compatibility).
    
    Args:
        args: CEM configuration arguments (CEMArgs from configs)
        initial_state: Optional initial state
        
    Returns:
        Dictionary with planning results
    """
    """
    Run CEM planner (main entry point for backward compatibility).
    
    Args:
        args: CEM configuration arguments (includes env_name)
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
    solver = CEMSolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=args.horizon,
        dt=args.dt,
        num_samples=args.num_samples,
        num_iterations=args.num_iterations,
        elite_frac=args.elite_frac,
        init_std=args.init_std,
        min_std=args.min_std,
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
        print(f"CEM total reward: {result['total_reward']:.3f}")
    
    return result

