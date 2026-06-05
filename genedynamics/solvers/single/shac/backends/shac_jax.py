"""JAX-MPM backend for SHAC.

Maintains N parallel simulator carries (the (x, v, C, F) state of N
independent envs), runs h-step differentiable rollouts via
``rollout_h_from_state``, and dispatches the actor/critic update to
``policy_loss.episode_step``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from genedynamics.envs.external.jax_mpm.scene import (
    MPMConfig,
    SceneData,
    _OBS_DIM,
    _init_carry,
    rollout_h_from_state,
)

from ..critic import init_mlp, mlp_apply, adam_init
from ..policy_loss import episode_step, make_actor_loss_fn
from ..protocols import SHACConfig, SHACInfo


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Carry-batch helpers (vmap convention: leading axis = env index)
# ---------------------------------------------------------------------------


def _tile_carry(single_carry, n_envs: int):
    """Tile a single carry tuple along a fresh leading axis of size n_envs."""
    return tuple(
        jnp.broadcast_to(arr[None, ...], (n_envs,) + arr.shape)
        for arr in single_carry
    )


# ---------------------------------------------------------------------------
# SHAC backend
# ---------------------------------------------------------------------------


class SHACBackendJax:
    """JAX implementation of SHAC over a fixed soft-robot morphology.

    Construction is cheap; the heavy initialization (critic + JIT trace)
    fires inside ``train()``.

    Parameters
    ----------
    scene, mpm_cfg
        From a JaxMpmRolloutEvaluator. Captured by closure into JIT.
    cfg
        SHACConfig dataclass.
    morphology
        Optional (n_voxels,) occupancy. None → all-ones (full body mass).
    friction
        Scalar ground friction.
    terrain_height
        Optional (n_grid, n_grid) — Phase 2 terrain field. None → flat.
    """

    def __init__(
        self,
        *,
        scene: SceneData,
        mpm_cfg: MPMConfig,
        cfg: SHACConfig,
        morphology: Optional[np.ndarray] = None,
        friction: float = 0.5,
        terrain_height: Optional[np.ndarray] = None,
    ):
        self.scene = scene
        self.mpm_cfg = mpm_cfg
        self.cfg = cfg
        self.friction = float(friction)
        if morphology is None:
            self.morphology = np.ones((scene.n_voxels,), dtype=np.float32)
        else:
            self.morphology = np.asarray(morphology, dtype=np.float32).reshape(-1)
            if self.morphology.shape[0] != scene.n_voxels:
                raise ValueError(
                    f"morphology shape {self.morphology.shape} does not match "
                    f"scene.n_voxels={scene.n_voxels}"
                )
        self.terrain_height = (
            None if terrain_height is None
            else jnp.asarray(np.asarray(terrain_height, dtype=np.float32))
        )

    # ------------------------------------------------------------------

    def _make_rollout_fn(self):
        """Closure: bind scene/cfg/friction/morphology/terrain into a 3-arg fn."""
        scene = self.scene
        mpm_cfg = self.mpm_cfg
        h = int(self.cfg.h)
        friction = jnp.asarray(self.friction, dtype=jnp.float32)
        x_morph = jnp.asarray(self.morphology, dtype=jnp.float32)
        terrain_height = self.terrain_height

        def rollout_fn(carry_init, t0, phi):
            return rollout_h_from_state(
                carry_init, t0, phi, h, friction, scene, mpm_cfg,
                x_morph=x_morph, terrain_height=terrain_height,
                horizon_for_time=int(self.cfg.env_horizon),
            )

        return rollout_fn

    # ------------------------------------------------------------------

    def train(self) -> Dict[str, Any]:
        cfg = self.cfg
        rng = jax.random.PRNGKey(int(cfg.seed))

        # --- Init phi (mean) ---------------------------------------------
        rng, phi_key = jax.random.split(rng)
        phi_lo, phi_hi = float(cfg.phi_lo), float(cfg.phi_hi)
        phi_mean = jax.random.uniform(
            phi_key, (cfg.phi_dim,), dtype=jnp.float32,
            minval=phi_lo, maxval=phi_hi,
        )

        # --- Init critic + target ---------------------------------------
        rng, crit_key = jax.random.split(rng)
        critic_params = init_mlp(
            crit_key, in_dim=_OBS_DIM, hidden=tuple(cfg.critic_hidden), out_dim=1,
        )
        target_params = jax.tree_util.tree_map(jnp.array, critic_params)

        # --- Init optimizer states --------------------------------------
        actor_state = adam_init(phi_mean)
        critic_state = adam_init(critic_params)

        # --- Init N parallel carries from a single _init_carry ---------
        single_carry = _init_carry(self.scene)
        carries = _tile_carry(single_carry, int(cfg.n_envs))

        # --- Build jitted episode_step ----------------------------------
        rollout_fn = self._make_rollout_fn()
        actor_loss_fn = make_actor_loss_fn(
            rollout_fn, h=int(cfg.h), discount=float(cfg.discount),
        )

        def _ep_step(state, t0):
            return episode_step(
                phi_mean=state["phi_mean"],
                actor_state=state["actor_state"],
                critic_params=state["critic_params"],
                critic_state=state["critic_state"],
                target_params=state["target_params"],
                carries_init=state["carries"],
                t0=t0,
                rng=state["rng"],
                actor_loss_fn=actor_loss_fn,
                actor_lr=float(cfg.actor_lr),
                actor_grad_clip=float(cfg.actor_grad_clip),
                critic_lr=float(cfg.critic_lr),
                critic_grad_clip=float(cfg.critic_grad_clip),
                n_critic_iters=int(cfg.n_critic_iters),
                n_critic_minibatches=int(cfg.n_critic_minibatches),
                target_alpha=float(cfg.target_alpha),
                sigma=float(cfg.explore_sigma) if not cfg.deterministic else 0.0,
                discount=float(cfg.discount),
                td_lambda_coef=float(cfg.td_lambda),
            )

        ep_step_jit = jax.jit(_ep_step)

        # --- Episode loop -----------------------------------------------
        history: List[SHACInfo] = []
        state = {
            "phi_mean": phi_mean,
            "actor_state": actor_state,
            "critic_params": critic_params,
            "critic_state": critic_state,
            "target_params": target_params,
            "carries": carries,
            "rng": rng,
        }
        t_in_episode = 0
        wall_total_start = time.perf_counter()

        n_consecutive_bad = 0
        for ep in range(int(cfg.n_episodes)):
            t0 = jnp.asarray(t_in_episode, dtype=jnp.int32)
            t_step_start = time.perf_counter()
            out = ep_step_jit(state, t0)
            jax.block_until_ready(out["phi_mean"])
            wall = time.perf_counter() - t_step_start

            # Phase 4.3: finite-grad guard. Non-finite actor grad / loss /
            # phi_mean signals BPTT instability (writeup §3.2 — typical when h
            # is too long or the simulator has hit a numerically-bad state).
            # We DROP the offending update and reset carries to _init_carry.
            actor_g = float(out["actor_grad_norm"])
            actor_l = float(out["actor_loss"])
            phi_finite = bool(np.all(np.isfinite(np.asarray(out["phi_mean"]))))
            if not (np.isfinite(actor_g) and np.isfinite(actor_l) and phi_finite):
                n_consecutive_bad += 1
                log.warning(
                    "[SHAC] non-finite gradient at ep=%d (|g|=%s loss=%s); "
                    "dropping update + resetting carries",
                    ep, actor_g, actor_l,
                )
                if n_consecutive_bad >= 3:
                    log.error(
                        "[SHAC] %d consecutive bad episodes — aborting. "
                        "Try smaller h, smaller actor_lr, or stronger grad_clip.",
                        n_consecutive_bad,
                    )
                    break
                state["carries"] = _tile_carry(single_carry, int(cfg.n_envs))
                t_in_episode = 0
                history.append(SHACInfo(
                    episode=ep, mean_episode_return=float("nan"),
                    actor_loss=actor_l, critic_loss=float(out["critic_loss"]),
                    actor_grad_norm=actor_g, ess=0.0, wall_time=float(wall),
                ))
                continue
            n_consecutive_bad = 0

            state["phi_mean"] = out["phi_mean"]
            state["actor_state"] = out["actor_state"]
            state["critic_params"] = out["critic_params"]
            state["critic_state"] = out["critic_state"]
            state["target_params"] = out["target_params"]
            state["carries"] = out["final_carries"]
            state["rng"] = out["rng"]

            t_in_episode += int(cfg.h)
            # Hard reset every reset_every episodes (SHAC paper §3.3) to
            # restart from _init_carry — keeps the state distribution from
            # drifting outside the support the critic was trained on.
            if (ep + 1) % max(int(cfg.reset_every), 1) == 0:
                state["carries"] = _tile_carry(single_carry, int(cfg.n_envs))
                t_in_episode = 0

            info = SHACInfo(
                episode=ep,
                mean_episode_return=float(out["mean_episode_return"]),
                actor_loss=float(out["actor_loss"]),
                critic_loss=float(out["critic_loss"]),
                actor_grad_norm=float(out["actor_grad_norm"]),
                ess=0.0,    # placeholder — only meaningful in MBD branch
                wall_time=float(wall),
            )
            history.append(info)
            if cfg.show_tqdm and (ep % max(int(cfg.log_every), 1) == 0):
                log.info(
                    "[SHAC] ep=%d return=%.4f L_a=%.4f L_c=%.4f |g|=%.3e t=%.2fs",
                    ep, info.mean_episode_return, info.actor_loss,
                    info.critic_loss, info.actor_grad_norm, info.wall_time,
                )

        wall_total = time.perf_counter() - wall_total_start

        # Final eval: deterministic rollout from _init_carry over the full
        # env horizon to report a clean return that's comparable with MRMFMBD.
        rollout_fn_eval = self._make_rollout_fn()
        single_carry = _init_carry(self.scene)
        carry_batched = _tile_carry(single_carry, 1)

        def _eval_full_horizon(phi):
            n_blocks = max(int(cfg.env_horizon) // int(cfg.h), 1)
            carry = jax.tree_util.tree_map(lambda a: a[0], carry_batched)
            total_r = jnp.float32(0.0)
            t0 = jnp.int32(0)
            for _ in range(n_blocks):
                carry, rs, _ = rollout_fn_eval(carry, t0, phi)
                total_r = total_r + jnp.sum(rs)
                t0 = t0 + int(cfg.h)
            return total_r

        final_return = float(_eval_full_horizon(state["phi_mean"]))

        return {
            "phi": np.asarray(state["phi_mean"]),
            "final_return": final_return,
            "history": history,
            "wall_clock": wall_total,
        }
