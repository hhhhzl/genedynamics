"""
MBD (Multi-scale Barrier Diffusion) solver implementation.

MBD uses barrier functions with multi-scale diffusion to handle constraints
while optimizing trajectories. This is a variant of EDOC that focuses on
constraint satisfaction through barrier methods.
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
from enerdynamics.solvers.edoc import EnergyToLegacyAdapter


class MBDSolver(SamplingSolver):
    """
    MBD solver using multi-scale barrier diffusion.
    
    This solver uses reverse diffusion in action space to find optimal
    trajectories, similar to EDOC but with a simpler implementation.
    """
    
    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        horizon: int = 80,
        dt: float = 0.1,
        Nsample: int = 2048,
        Ndiffuse: int = 100,
        temp_sample: float = 0.1,
        beta0: float = 1e-4,
        betaT: float = 1e-2,
        action_limit: float = 1.0,
        seed: int = 0,
        **kwargs
    ):
        """
        Initialize MBD solver.
        
        Args:
            dynamics: Dynamics model
            energy: Energy functional
            backend: Computational backend (must be JAX for MBD)
            horizon: Planning horizon
            dt: Time step
            Nsample: Number of samples per diffusion step
            Ndiffuse: Number of diffusion steps
            temp_sample: Temperature for sampling
            beta0: Initial noise level for diffusion
            betaT: Final noise level for diffusion
            action_limit: Action limit (clipping)
            seed: Random seed
            **kwargs: Additional MBD-specific configuration
        """
        super().__init__(dynamics, energy, backend, **kwargs)
        
        # MBD currently requires JAX backend
        if backend.name != "jax":
            raise ValueError(
                f"MBD solver requires JAX backend, got {backend.name}. "
                "Use backend=get_backend('jax') when creating the solver."
            )
        
        # Store configuration
        self.horizon = horizon
        self.dt = dt
        self.Nsample = Nsample
        self.Ndiffuse = Ndiffuse
        self.temp_sample = temp_sample
        self.beta0 = beta0
        self.betaT = betaT
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
        Sample trajectories using MBD diffusion process.
        
        Args:
            x0: Initial state
            horizon: Planning horizon
            n_samples: Number of samples
            **kwargs: Additional parameters
            
        Returns:
            List of candidate trajectories
        """
        # For MBD, we use the diffusion process to generate samples
        # This is a simplified version - full implementation would use the diffusion
        
        x0_data = np.asarray(x0, dtype=np.float32) if not isinstance(x0, jnp.ndarray) else x0
        x0_jnp = jnp.asarray(x0_data, dtype=jnp.float32)
        
        rng = jax.random.PRNGKey(self.seed)
        trajectories = []
        
        for i in range(n_samples):
            rng, sample_key = jax.random.split(rng)
            
            # Sample random actions
            actions = jax.random.normal(
                sample_key, (horizon, self.act_dim), dtype=jnp.float32
            ) * 0.1  # Small noise
            
            if self.action_limit is not None:
                actions = jnp.clip(actions, -self.action_limit, self.action_limit)
            
            # Rollout
            states = self._rollout_states_fn(x0_jnp, actions)
            
            # Convert to Trajectory
            states_list = [np.asarray(states[j], dtype=np.float32) for j in range(len(states))]
            actions_list = [np.asarray(actions[j], dtype=np.float32) for j in range(len(actions))]
            
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
        Solve for optimal trajectory using MBD diffusion.
        
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
        rng, diffuse_rng = jax.random.split(rng)
        
        # Diffusion parameters
        betas = jnp.linspace(self.beta0, self.betaT, self.Ndiffuse, dtype=jnp.float32)
        alphas = 1.0 - betas
        alphas_bar = jnp.cumprod(alphas)
        sigmas = jnp.sqrt(1.0 - alphas_bar)
        diffusion_indices = jnp.arange(self.Ndiffuse - 1, 0, -1, dtype=jnp.int32)
        
        def reverse_diffuse(rng_in, Ybar_init):
            def body(carry, idx):
                rng_curr, Ybar_curr = carry
                rng_curr, noise_key = jax.random.split(rng_curr)
                
                Yi = Ybar_curr * jnp.sqrt(alphas_bar[idx])
                eps = jax.random.normal(
                    noise_key, (self.Nsample, horizon, self.act_dim), dtype=jnp.float32
                )
                Y0s = eps * sigmas[idx] + Ybar_curr
                Y0s = jnp.clip(Y0s, -self.action_limit, self.action_limit)
                
                # Score using rewards
                rews = self._rollout_rewards_batch_fn(x0_jnp, Y0s)
                rews_mean = jnp.mean(rews, axis=-1)
                
                rew_mean = jnp.mean(rews_mean)
                rew_std = jnp.std(rews_mean)
                rew_std = jnp.where(rew_std < 1e-4, 1.0, rew_std)
                
                logp0 = (rews_mean - rew_mean) / (rew_std * self.temp_sample)
                weights = jax.nn.softmax(logp0)
                Ybar_weighted = jnp.einsum("n,nij->ij", weights, Y0s)
                
                score = (-Yi + jnp.sqrt(alphas_bar[idx]) * Ybar_weighted) / (
                    1.0 - alphas_bar[idx]
                )
                Yim1 = (Yi + (1.0 - alphas_bar[idx]) * score) / jnp.sqrt(alphas[idx])
                Ybar_next = Yim1 / jnp.sqrt(alphas_bar[idx - 1])
                
                return (rng_curr, Ybar_next), (jnp.mean(rews_mean), Ybar_next, Y0s)
            
            (rng_out, Ybar_final), (reward_hist, Ybar_hist, Ysamples_hist) = jax.lax.scan(
                body, (rng_in, Ybar_init), diffusion_indices
            )
            reward_hist = reward_hist[::-1]
            Ybar_hist = Ybar_hist[::-1]
            Ysamples_hist = Ysamples_hist[::-1]
            return rng_out, Ybar_final, reward_hist, Ybar_hist, Ysamples_hist
        
        reverse_diffuse_jit = jax.jit(reverse_diffuse)
        
        Ybar_init = jnp.zeros((horizon, self.act_dim), dtype=jnp.float32)
        _, Ybar_final, _, _, _ = reverse_diffuse_jit(diffuse_rng, Ybar_init)
        
        final_actions = jnp.clip(Ybar_final, -self.action_limit, self.action_limit)
        states = self._rollout_states_fn(x0_jnp, final_actions)
        
        # Convert to Trajectory
        states_list = [np.asarray(states[j], dtype=np.float32) for j in range(len(states))]
        actions_list = [np.asarray(final_actions[j], dtype=np.float32) for j in range(len(final_actions))]
        
        traj = Trajectory(states=states_list, actions=actions_list)
        return traj


# ============================================================================
# Main Entry Point (for backward compatibility)
# ============================================================================

def run_mbd(args):
    """
    Run MBD diffusion planner (main entry point for backward compatibility).
    
    Args:
        args: Diffusion configuration arguments (DiffusionArgs from configs)
        
    Returns:
        Dictionary with planning results
    """
    """
    Run MBD diffusion planner (main entry point for backward compatibility).
    
    Args:
        args: Diffusion configuration arguments (includes env_name)
        
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
    solver = MBDSolver(
        dynamics=dynamics,
        energy=energy,
        backend=backend,
        horizon=args.horizon,
        dt=args.dt,
        Nsample=args.Nsample,
        Ndiffuse=args.Ndiffuse,
        temp_sample=args.temp_sample,
        beta0=args.beta0,
        betaT=args.betaT,
        action_limit=args.action_limit,
        seed=args.seed,
    )
    
    # Get initial state
    rng = jax.random.PRNGKey(args.seed)
    try:
        x0, _ = env.reset(rng)
    except TypeError:
        x0, _ = env.reset()
    
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
        # For backward compatibility with visualization code
        "diffusion_actions_traj": np.array([], dtype=np.float32),  # Empty for now
        "diffusion_sampled_actions": np.array([], dtype=np.float32),  # Empty for now
    }
    
    if args.verbose:
        print("initial state:", x0)
        print("final state:", states_np[-1])
        print("total reward:", result["total_reward"])
        print("mean per-step reward:", result["mean_reward"])
        print("total energy:", float(np.sum(energies)))
        print("final energy:", float(energies[-1]) if energies.size > 0 else 0.0)
    
    return result
