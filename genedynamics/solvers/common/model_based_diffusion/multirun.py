"""
Shared multirun helpers for model-based diffusion solvers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np
import jax
import jax.numpy as jnp


def split_multirun_keys(rng_key: Any, num_modes: int, backend_name: str, seed: int) -> Any:
    """Split a root RNG into per-run keys for JAX or integer seeds for NumPy/Torch-like backends."""
    if backend_name == "jax":
        key = rng_key
        if not isinstance(key, jnp.ndarray) or key.shape != (2,):
            if isinstance(key, (int, np.integer)):
                key = jax.random.PRNGKey(int(key))
            else:
                key = jax.random.PRNGKey(int(seed))
        return jax.random.split(key, int(num_modes))

    # Non-JAX backend: use reproducible integer seeds.
    if isinstance(rng_key, jnp.ndarray) and rng_key.shape == (2,):
        base_seed = int(rng_key[0]) ^ int(rng_key[1])
    elif isinstance(rng_key, (int, np.integer)):
        base_seed = int(rng_key)
    else:
        base_seed = int(seed)
    return [base_seed + i for i in range(int(num_modes))]


def run_multirun(
    *,
    planner: Any,
    x0_data: Any,
    keys: Sequence[Any],
    plan_once_fn,
    plan_batch_fn=None,
) -> List[Dict[str, Any]]:
    """Execute C independent planning runs with optional batch fast path."""
    planner_num_modes_orig = int(getattr(planner, "num_modes", 1))
    planner.num_modes = 1
    try:
        if callable(plan_batch_fn):
            batch_results = plan_batch_fn(planner, x0_data, keys)
            if batch_results is not None:
                return list(batch_results)
        return [plan_once_fn(planner, x0_data, k) for k in keys]
    finally:
        planner.num_modes = planner_num_modes_orig


def aggregate_multirun_results(results: Sequence[Dict[str, Any]], keys: Sequence[Any]) -> Dict[str, Any]:
    """Aggregate multirun outputs into a single best result with candidate bundles."""
    candidate_states_list = [np.asarray(r["states"], dtype=np.float32) for r in results]
    candidate_actions_list = [np.asarray(r["actions"], dtype=np.float32) for r in results]
    candidate_costs = np.asarray(
        [
            float(np.asarray(r.get("candidate_costs", [np.nan]))[int(r.get("best_idx", 0))])
            for r in results
        ],
        dtype=np.float32,
    )
    best_idx = int(np.nanargmin(candidate_costs))
    best_result = dict(results[best_idx])
    best_result["candidate_states"] = candidate_states_list
    best_result["candidate_actions"] = candidate_actions_list
    best_result["candidate_costs"] = candidate_costs
    best_result["best_idx"] = best_idx
    best_result["mode_strategy"] = "multirun"
    best_result["multirun_keys"] = list(keys)
    best_result["multirun_diffusion_data"] = [
        {
            "diffusion_actions_traj": r.get("diffusion_actions_traj"),
            "diffusion_sampled_actions": r.get("diffusion_sampled_actions"),
        }
        for r in results
    ]
    return best_result

