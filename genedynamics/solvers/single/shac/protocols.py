"""SHACConfig + result/info types — writeup §8 controller-only baseline.

This is the OUR-side configuration object; the actual JAX implementation
lives in `backends/shac_jax.py`. Keeping the dataclass here means tests can
import config types without pulling in JAX at module load.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple


@dataclass
class SHACConfig:
    """Hyperparameters for Short-Horizon Actor-Critic (Xu et al., ICLR 2022).

    Defaults follow the paper's published values for Ant / Humanoid (Tables 4-5)
    rescaled for the soft-robot's smaller state and shorter env horizon.
    """

    # Topology --------------------------------------------------------------
    h: int = 32                          # truncated rollout window (paper: 32)
    n_envs: int = 32                     # parallel trajectories sampled per episode
    n_episodes: int = 500                # outer learning episodes

    # Rollout / reward ------------------------------------------------------
    discount: float = 0.99
    td_lambda: float = 0.95              # td-λ coefficient for value targets
    env_horizon: int = 200               # writeup crawling: 200 frames per training rollout
    reset_every: int = 1                 # episodes between hard resets to _init_carry

    # Policy (phi) parameters ----------------------------------------------
    phi_dim: int = 80
    explore_sigma: float = 0.1           # SHAC paper §3.3: stochastic policy
                                         # — additive Gaussian noise on the
                                         # mean phi for exploration. With
                                         # sigma=0 the algo collapses to BPTT.
    deterministic: bool = False          # only meaningful with explore_sigma>0
    phi_lo: float = -0.5
    phi_hi: float = 0.5

    # Critic ----------------------------------------------------------------
    critic_hidden: Tuple[int, ...] = (64, 64)
    target_alpha: float = 0.995          # Polyak averaging
    n_critic_iters: int = 16             # SGD steps on critic per episode
    n_critic_minibatches: int = 4

    # Optimizers ------------------------------------------------------------
    actor_lr: float = 2.0e-3
    critic_lr: float = 5.0e-4
    actor_grad_clip: float = 1.0         # gradient norm clip — stability
    critic_grad_clip: float = 1.0

    # Soft-robot integration ------------------------------------------------
    n_actuators: int = 10
    morphology: Any = None               # optional fixed (n_voxels,) occupancy.
                                         # When None the eval body uses all-1 mass.
    friction: float = 0.5
    seed: int = 0

    # Diagnostics -----------------------------------------------------------
    show_tqdm: bool = False
    log_every: int = 10                  # episodes between logger info lines
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SHACInfo:
    """Per-episode diagnostic record (one entry per training step)."""

    episode: int
    mean_episode_return: float
    actor_loss: float
    critic_loss: float
    actor_grad_norm: float
    ess: float
    wall_time: float
