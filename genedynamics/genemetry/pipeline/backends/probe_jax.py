"""
JAX probe mini-batch pipeline.

Extracts the ``run_probe_pack`` logic from the 2GO scan body into a
reusable genemetry component.  The pipeline:

1. Selects a mini-batch of M trajectories from a mix of tail (high-cost)
   and random samples.
2. Runs single-iteration retraction on each via ``vmap``.
3. Computes the mean residual (retracted - original) as probe geometry.
4. Scatters retracted trajectories back into the full sample set.

Registered as ``("pipeline", "probe", "jax")``.
"""

from typing import Any, Optional

import jax
import jax.numpy as jnp

from genedynamics.genemetry.base import ProbeSampler, RetractionOperator
from genedynamics.genemetry.registry import register_genemetry
from genedynamics.genemetry.types import ProbeResult


@register_genemetry("pipeline", "probe", "jax")
class ProbePipelineJax(ProbeSampler):
    """JAX implementation of the probe mini-batch pipeline.

    Parameters
    ----------
    tail_mix : float
        Fraction of mini-batch drawn from high-cost tail pool.
    pool_ratio : float
        Fraction of samples forming the tail candidate pool.
    max_probes : int or None
        Maximum mini-batch size (caps actual probe count).
    """

    def __init__(
        self,
        tail_mix: float = 0.5,
        pool_ratio: float = 0.25,
        max_probes: Optional[int] = None,
    ) -> None:
        self._tail_mix = float(tail_mix)
        self._pool_ratio = max(1e-6, float(pool_ratio))
        self._max_probes = max_probes

    def sample_and_retract(
        self,
        trajectories: Any,
        costs: Any,
        retraction_op: RetractionOperator,
        state: Any,
        retract_params: Any,
        rng_keys: Any,
    ) -> ProbeResult:
        """Run probe mini-batch and extract residual geometry.

        Parameters
        ----------
        trajectories : (N, H, D) jnp.ndarray
        costs : (N,) jnp.ndarray — per-trajectory violation / cost.
        retraction_op : RetractionOperator
        state : initial state (x0), jnp.ndarray
        retract_params : dict with ``sched_state``, ``sched_params``,
            and ``M_eff`` (effective mini-batch size as jnp scalar).
        rng_keys : dict with ``rng_tail``, ``rng_rand``, ``rng_vmap``.

        Returns
        -------
        ProbeResult with ``fixed_trajectories`` and ``residual_geometry``.
        """
        Y0s = trajectories
        v_batch = costs
        Nsample = Y0s.shape[0]
        horizon = Y0s.shape[1]
        act_dim = Y0s.shape[2]

        M_eff = retract_params["M_eff"]
        rng_tail = rng_keys["rng_tail"]
        rng_rand = rng_keys["rng_rand"]
        rng_vmap_parent = rng_keys["rng_vmap"]

        k_pool_i = int(min(Nsample, max(1, int(self._pool_ratio * Nsample))))
        M_max = self._max_probes if self._max_probes is not None else Nsample
        M_max = int(max(1, min(Nsample, M_max)))
        tail_mix_f = self._tail_mix

        # Compute how many slots come from the tail vs random pool.
        tail_n = jnp.clip(
            jnp.round(
                M_eff.astype(jnp.float32)
                * jnp.asarray(tail_mix_f, dtype=jnp.float32)
            ).astype(jnp.int32),
            0,
            M_eff,
        )

        # Select tail candidates: top-k by cost, then shuffle.
        _, idx_hi = jax.lax.top_k(v_batch, k_pool_i)
        perm_pool = jax.random.permutation(rng_tail, k_pool_i)
        shuffled_hi = idx_hi[perm_pool]

        # Select random candidates.
        perm_full = jax.random.permutation(rng_rand, Nsample)

        # Build index array: first tail_n from tail, rest from random.
        i_arr = jnp.arange(M_max, dtype=jnp.int32)
        valid = i_arr < M_eff
        tail_n_mx = jnp.minimum(tail_n, M_max)
        cand_tail = shuffled_hi[jnp.minimum(i_arr, k_pool_i - 1)]
        j_rand = i_arr - tail_n_mx
        cand_rand = perm_full[
            jnp.minimum(jnp.maximum(j_rand, 0), Nsample - 1)
        ]
        idx_m = jnp.where(i_arr < tail_n_mx, cand_tail, cand_rand)
        Y_g = Y0s[idx_m]

        # Run retraction on each probed trajectory.
        probe_keys = jax.random.split(rng_vmap_parent, M_max)

        # Build per-probe retract params (single QP iteration).
        sched_params = dict(retract_params.get("sched_params", {}))
        sched_params["I_QP"] = jnp.asarray(1, dtype=jnp.int32)
        sched_state = retract_params.get("sched_state", {})

        def _retract_one(y, keyp):
            sp = dict(sched_params)
            sp["rng_key"] = keyp
            params = {"sched_state": sched_state, "sched_params": sp}
            return retraction_op.retract(state, y, params).trajectory

        y_f = jax.vmap(_retract_one)(Y_g, probe_keys)

        # Mask out invalid slots.
        valid3 = valid[:, None, None]
        y_out = jnp.where(valid3, y_f, Y_g)

        # Residual geometry: mean(retracted - original) over valid probes.
        resid = y_out - Y_g
        wsum = jnp.maximum(
            jnp.sum(valid.astype(jnp.float32)),
            jnp.asarray(1.0, dtype=jnp.float32),
        )
        a_probe = jnp.sum(resid * valid3, axis=0) / wsum

        # Scatter retracted trajectories back into the full sample set.
        def _scatter_step(ii, carry):
            Y_c, bm_c = carry
            Y_n = jnp.where(
                valid[ii],
                Y_c.at[idx_m[ii]].set(y_out[ii]),
                Y_c,
            )
            bm_n = jnp.where(
                valid[ii],
                bm_c.at[idx_m[ii]].set(jnp.asarray(1.0, dtype=jnp.float32)),
                bm_c,
            )
            return (Y_n, bm_n)

        Y_fix, bmask = jax.lax.fori_loop(
            0, M_max, _scatter_step,
            (Y0s, jnp.zeros((Nsample,), dtype=jnp.float32)),
        )

        return ProbeResult(
            fixed_trajectories=Y_fix,
            residual_geometry=a_probe,
            meta={"bmask": bmask, "n_probed": M_eff, "idx_m": idx_m},
        )
