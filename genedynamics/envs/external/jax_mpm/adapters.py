"""Adapters: JAX-MPM rollout → RolloutBatchResult protocol.

Wraps the JAX scene module behind the standard
`evaluate_batch(RolloutBatchRequest)` interface so the MRMFMBD backend
(and CMA-ES baseline) can plug in by name.

Mode → friction mapping: configurable via runtime_config["mode_friction"]
(list of floats per mode_id), defaulting to [0.3, 0.4, 0.5, 0.6].
Fidelity → env-step count: level-0 → 30, level-1 → 100, level-2 → 200.
"""

from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from .scene import (
    MPMConfig, SceneData, build_scene,
    rollout_return_batch, rollout_return_push_batch,
)


# Precompile on first use and cache so subsequent `evaluate_batch` calls are
# pure kernel dispatch. The cache is keyed by (scene.n_particles, num_env_steps).
_JIT_CACHE: dict = {}


def _get_batched_fn(scene: SceneData, cfg: MPMConfig, num_env_steps: int):
    key = (scene.n_particles, int(num_env_steps), cfg.n_grid, cfg.dt)
    if key in _JIT_CACHE:
        return _JIT_CACHE[key]

    @jax.jit
    def _jitted(x_morph_batch, phi_batch, friction_batch):
        return rollout_return_batch(
            x_morph_batch, phi_batch, friction_batch, scene, cfg, num_env_steps
        )

    _JIT_CACHE[key] = _jitted
    return _jitted


# Fidelity level → env step count (coarse=30, medium=100, fine=200).
FIDELITY_STEPS = {0: 30, 1: 100, 2: 200}


def evaluate_batch_request(
    request: Any,               # genedynamics.envs.evaluators.RolloutBatchRequest
    scene: SceneData,
    cfg: MPMConfig,
    mode_friction_table: List[float],
    regime_bank: List[Any] = None,    # Phase 2.2: optional list[RegimeSpec]
    task: str = "crawling_ground",   # "crawling_ground" | "locomotion" | "push"
    push_goal_x: float = 0.85,
    push_weights: Tuple[float, float, float, float] = (1.0, 5.0, 0.01, 50.0),
) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate all requests in one JIT'd vmap'd forward.

    Args:
        request: object with `.requests` list; each item has morphology_params,
            controller_params, mode_id, fidelity_level, seed.
        scene: built once upfront.
        cfg: MPMConfig.
        mode_friction_table: friction coeff per mode_id (legacy fallback used
            when regime_bank is None).
        regime_bank: Phase 2.2 optional list of RegimeSpec. When provided,
            mode_id indexes this list; per-mode terrain (and per-mode
            manipuland for push) is dispatched to the appropriate batched
            rollout. Returns the SAME (returns, disp) shape regardless.
        task: which rollout to use when regime_bank is set. "locomotion" uses
            rollout_return_batch (with terrain); "push" uses
            rollout_return_push_batch (terrain + manipuland). Ignored when
            regime_bank is None.

    Returns:
        (returns_float32[B], disp_float32[B])
        For push, the second array carries dist_to_goal_T (not COM displacement).
    """
    req_list = request.requests
    if not req_list:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    # Phase 2.2 dispatch: per-mode terrain/manip means we group by (fid, mode),
    # not just fid. Without a regime_bank we keep the legacy fid-only grouping
    # (terrain is implicitly flat, friction comes from mode_friction_table).
    if regime_bank is None:
        by_key: dict = {}
        for i, r in enumerate(req_list):
            by_key.setdefault(int(r.fidelity_level), []).append(i)
        keyed = [((fid, None), idxs) for fid, idxs in by_key.items()]
    else:
        from .terrain import to_grid as _to_grid
        by_fid_mode: dict = {}
        for i, r in enumerate(req_list):
            by_fid_mode.setdefault((int(r.fidelity_level), int(r.mode_id)), []).append(i)
        keyed = list(by_fid_mode.items())

    returns_out = np.zeros(len(req_list), dtype=np.float32)
    disp_out = np.zeros(len(req_list), dtype=np.float32)

    for key, idxs in keyed:
        if regime_bank is None:
            fid = key
            num_env_steps = int(FIDELITY_STEPS.get(fid, FIDELITY_STEPS[2]))
            fr_b = np.asarray(
                [mode_friction_table[int(req_list[i].mode_id) % len(mode_friction_table)] for i in idxs],
                dtype=np.float32,
            )
            terrain_h = None
            push = False
            manip_cfg_local = None
        else:
            fid, mode_id = key
            num_env_steps = int(FIDELITY_STEPS.get(fid, FIDELITY_STEPS[2]))
            regime = regime_bank[int(mode_id) % len(regime_bank)]
            fr_b = np.full(len(idxs), float(regime.friction), dtype=np.float32)
            from .terrain import to_grid as _to_grid
            terrain_h = jnp.asarray(_to_grid(regime.terrain, cfg.n_grid))
            push = (task == "push") and regime.has_manipuland
            manip_cfg_local = regime.manipuland if push else None

        x_b = np.stack([np.asarray(req_list[i].morphology_params, dtype=np.float32) for i in idxs])
        phi_b = np.stack([np.asarray(req_list[i].controller_params, dtype=np.float32) for i in idxs])
        x_b = jnp.asarray(x_b); phi_b = jnp.asarray(phi_b); fr_b = jnp.asarray(fr_b)

        if push:
            rs, ds = rollout_return_push_batch(
                x_b, phi_b, fr_b, scene, cfg, num_env_steps,
                manip_cfg=manip_cfg_local, goal_x=push_goal_x,
                terrain_height=terrain_h, weights=push_weights,
            )
        elif terrain_h is not None:
            # Per-regime terrain — bypass the fidelity-keyed JIT cache (which
            # keys on env-step count alone) and call rollout_return_batch
            # directly. JAX will trace once per (n_env_steps, terrain.shape).
            rs, ds = rollout_return_batch(
                x_b, phi_b, fr_b, scene, cfg, num_env_steps,
                terrain_height=terrain_h,
            )
        else:
            fn = _get_batched_fn(scene, cfg, num_env_steps)
            rs, ds = fn(x_b, phi_b, fr_b)

        rs = np.asarray(rs, dtype=np.float32)
        ds = np.asarray(ds, dtype=np.float32)
        for j, i in enumerate(idxs):
            returns_out[i] = rs[j]
            disp_out[i] = ds[j]
    return returns_out, disp_out


def evaluate_rollout_request(
    morphology_params: np.ndarray,
    controller_params: np.ndarray,
    mode_id: int,
    fidelity_level: int,
    friction: float,
    scene: SceneData,
    cfg: MPMConfig,
) -> Tuple[float, float]:
    """Single-rollout convenience wrapper (for gif / capture calls)."""
    num_env_steps = int(FIDELITY_STEPS.get(int(fidelity_level), FIDELITY_STEPS[2]))
    fn = _get_batched_fn(scene, cfg, num_env_steps)
    x_b = jnp.asarray(morphology_params, dtype=jnp.float32)[None, :]
    phi_b = jnp.asarray(controller_params, dtype=jnp.float32)[None, :]
    fr_b = jnp.asarray([float(friction)], dtype=jnp.float32)
    rs, ds = fn(x_b, phi_b, fr_b)
    return float(np.asarray(rs)[0]), float(np.asarray(ds)[0])
