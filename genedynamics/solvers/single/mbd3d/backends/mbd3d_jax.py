"""
JAX backend for MBD3D (3DGS robust mapping).

Implements annealed bridge π_k(θ) ∝ p0(θ) * p(y|θ)^β_k with MCSA score ascent.
Uses core.inference (BridgeSchedule, MCSA, log_pi) and core.prob (LowRankCovariance).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jax
import jax.numpy as jnp

from genedynamics.core.types import Trajectory


@jax.jit
def _normalize_quat(q: jnp.ndarray) -> jnp.ndarray:
    return q / (jnp.linalg.norm(q, axis=-1, keepdims=True) + 1e-8)


class MBD3DBackendJax:
    """
    JAX implementation of MBD3D: annealed bridge + MCSA for 3DGS robust mapping.

    State θ = (scene_params_flat, camera_trajectory_flat).
    Bridge: π_k(θ) ∝ p0(θ) * p(y|θ)^β_k.
    MCSA: score ascent with importance-weighted proposals.
    """

    def __init__(
        self,
        *,
        scene_repr: Any,
        renderer: Any,
        likelihood: Any,
        bridge_schedule: Any,
        camera_prior: Optional[Any] = None,
        horizon: int = 10,
        n_gaussians: int = 32,
        M: int = 16,
        sigma_mcsa: float = 0.05,
        ess_min: float = 1.0,
        seed: int = 0,
        show_tqdm: bool = False,
        fidelity_ladder: Optional[Any] = None,
        **kwargs: Any,
    ):
        self.scene_repr = scene_repr
        self.renderer = renderer
        self.likelihood = likelihood
        self.bridge_schedule = bridge_schedule
        self.camera_prior = camera_prior
        self.horizon = horizon
        self.n_gaussians = n_gaussians
        self.M = M
        self.sigma_mcsa = sigma_mcsa
        self.ess_min = ess_min
        self.seed = seed
        self.show_tqdm = show_tqdm
        self.fidelity_ladder = fidelity_ladder
        self.config = kwargs

        self._scene_dim = scene_repr.dim()
        self._camera_pose_dim = 7  # position + quat
        self._camera_dim = (horizon + 1) * self._camera_pose_dim
        self._theta_dim = self._scene_dim + self._camera_dim

        self._build_jax_functions()

    def _build_jax_functions(self) -> None:
        from genedynamics.core.inference.annealed_bridge import log_pi
        from genedynamics.core.inference.mcsa import ImportanceWeighter, MCSAScoreEstimator
        from genedynamics.core.inference.diagnostics import effective_sample_size

        weighter = ImportanceWeighter(temperature=1.0, backend="jax")
        score_estimator = MCSAScoreEstimator(
            sigma=self.sigma_mcsa,
            weighter=weighter,
            ess_min=self.ess_min,
            backend="jax",
        )

        def log_prior(theta: jnp.ndarray) -> jnp.ndarray:
            scene_flat = theta[: self._scene_dim]
            cam_flat = theta[self._scene_dim :]
            scene = self.scene_repr.unflatten(scene_flat)
            lp_scene = self.scene_repr.prior_log_prob(scene)
            if self.camera_prior is not None:
                cam_poses = jnp.reshape(cam_flat, (self.horizon + 1, self._camera_pose_dim))
                lp_cam = self.camera_prior.log_prob(cam_poses)
                return jnp.asarray(lp_scene, dtype=jnp.float32) + jnp.asarray(lp_cam, dtype=jnp.float32)
            else:
                # Simple Gaussian prior on camera
                return jnp.asarray(lp_scene, dtype=jnp.float32) - 0.5 * jnp.sum(jnp.square(cam_flat)) / 10.0

        def log_likelihood(theta: jnp.ndarray, obs: Any) -> jnp.ndarray:
            scene_flat = theta[: self._scene_dim]
            cam_flat = theta[self._scene_dim :]
            scene = self.scene_repr.unflatten(scene_flat)
            cam_poses = jnp.reshape(cam_flat, (self.horizon + 1, self._camera_pose_dim))
            return self.likelihood.log_likelihood(scene, obs, cam_poses)

        def log_pi_k(theta: jnp.ndarray, obs: Any, beta: float) -> jnp.ndarray:
            lp = log_prior(theta)
            ll = log_likelihood(theta, obs)
            return log_pi(lp, ll, beta)

        def mcsa_step(theta: jnp.ndarray, obs: Any, beta: float, sigma: float, rng: Any) -> Tuple[jnp.ndarray, Dict]:
            rng, key_proposals = jax.random.split(rng)
            deltas = jax.random.normal(key_proposals, (self.M, self._theta_dim))
            proposals = theta + sigma * deltas

            # Clamp scene params for stability (optional)
            proposals = jax.lax.cond(
                True,
                lambda p: p,
                lambda p: p,
                proposals,
            )

            log_probs = jax.vmap(log_pi_k, in_axes=(0, None, None))(proposals, obs, beta)
            score, diag = score_estimator.estimate(proposals, deltas, log_probs, axis=0)
            theta_new = theta + self.bridge_schedule.eta(0) * score  # Use eta from schedule
            return theta_new, {
                "ess": float(np.asarray(diag.ess)),
                "score_norm": float(np.asarray(jnp.linalg.norm(score))),
            }

        self._log_prior_fn = jax.jit(log_prior)
        self._log_likelihood_fn = jax.jit(log_likelihood)
        self._log_pi_k_fn = log_pi_k  # Not JIT: obs (ObservationBundle) is not JAX-traceable
        self._mcsa_step_fn = jax.jit(mcsa_step, static_argnums=())

        # Full bridge loop (not jitted to allow variable K)
        def bridge_loop(rng: Any, theta_init: jnp.ndarray, obs: Any) -> Tuple[jnp.ndarray, Dict]:
            K = self.bridge_schedule.config.K
            theta = theta_init
            history = {"ess": [], "score_norm": [], "log_pi": []}

            for k in range(K):
                beta = self.bridge_schedule.beta(k)
                sigma = self.bridge_schedule.sigma(k)
                eta = self.bridge_schedule.eta(k)
                rng, key = jax.random.split(rng)

                # MCSA proposals (Python loop over proposals to avoid tracing ObservationBundle)
                key_proposals, key_next = jax.random.split(key)
                deltas = jax.random.normal(key_proposals, (self.M, self._theta_dim))
                proposals = theta + sigma * deltas
                log_probs = jnp.array([self._log_pi_k_fn(proposals[i], obs, beta) for i in range(self.M)])

                weights = jax.nn.softmax(log_probs)
                scores = jnp.sum(weights[:, None] * deltas, axis=0) / max(sigma, 1e-8)
                theta = theta + eta * scores

                lp = self._log_pi_k_fn(theta, obs, beta)
                ess = 1.0 / (jnp.sum(jnp.square(weights)) + 1e-12)
                history["ess"].append(float(np.asarray(ess)))
                history["score_norm"].append(float(np.asarray(jnp.linalg.norm(scores))))
                history["log_pi"].append(float(np.asarray(lp)))

            return theta, history

        self._bridge_loop_fn = bridge_loop

    def plan(
        self,
        x0: Any,
        rng_key: Optional[Any] = None,
        observations: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """
        Run MBD3D bridge inference.

        Args:
            x0: initial state (camera pose or scene init hint)
            rng_key: JAX PRNG key
            observations: ObservationBundle (required for likelihood)

        Returns:
            Dict with states, actions, scene_params, diagnostics, etc.
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(self.seed)
        if observations is None:
            raise ValueError("MBD3D requires observations for likelihood.")

        rng_key, key_scene, key_cam = jax.random.split(rng_key, 3)
        scene_init = self.scene_repr.sample_prior(key_scene, self.n_gaussians)
        scene_flat = self.scene_repr.flatten(scene_init)

        # Camera init: from x0 or random
        if x0 is not None and hasattr(x0, "shape"):
            x0_arr = jnp.asarray(x0, dtype=jnp.float32)
            if x0_arr.size >= self._camera_dim:
                cam_init = jnp.reshape(x0_arr[: self._camera_dim], (self.horizon + 1, self._camera_pose_dim))
            else:
                cam_init = jnp.tile(jnp.reshape(x0_arr, (-1,))[: self._camera_pose_dim], (self.horizon + 1, 1))
                if cam_init.shape[1] < self._camera_pose_dim:
                    cam_init = jnp.concatenate(
                        [cam_init, jnp.zeros((self.horizon + 1, self._camera_pose_dim - cam_init.shape[1]))],
                        axis=-1,
                    )
        else:
            cam_init = 0.1 * jax.random.normal(key_cam, (self.horizon + 1, self._camera_pose_dim))
            cam_init = cam_init.at[:, 3:7].set(_normalize_quat(cam_init[:, 3:7]))

        theta_init = jnp.concatenate([scene_flat, jnp.ravel(cam_init)], axis=0)

        theta_final, history = self._bridge_loop_fn(rng_key, theta_init, observations)

        scene_final = self.scene_repr.unflatten(theta_final[: self._scene_dim])
        cam_final = jnp.reshape(theta_final[self._scene_dim :], (self.horizon + 1, self._camera_pose_dim))

        # Build states/actions for Trajectory compatibility
        states_list = [np.asarray(cam_final[i]) for i in range(self.horizon + 1)]
        actions_list = [
            np.asarray(cam_final[i + 1] - cam_final[i]) for i in range(self.horizon)
        ]

        if self.show_tqdm:
            import tqdm
            tqdm.tqdm.write(f"MBD3D: final log_pi={history['log_pi'][-1]:.4f}")

        from ..types import MBD3DResult
        result = MBD3DResult(
            scene_params=scene_final,
            camera_trajectory=np.asarray(cam_final),
            states=states_list,
            actions=actions_list,
            diagnostics={"bridge_history": history},
            bridge_history=history,
            total_log_prob=history["log_pi"][-1],
            n_steps=len(history["log_pi"]),
        )

        return {
            "states": states_list,
            "actions": actions_list,
            "scene_params": scene_final,
            "camera_trajectory": np.asarray(cam_final),
            "diagnostics": result.diagnostics,
            "bridge_history": history,
            "total_log_prob": result.total_log_prob,
            "initial_state": states_list[0],
            "candidate_states": [states_list],
            "candidate_actions": [actions_list],
            "candidate_costs": np.asarray([-float(result.total_log_prob)], dtype=np.float32),
            "best_idx": 0,
            **result.to_trajectory_info(),
        }

    def sample_trajectories(
        self,
        x0: Any,
        n_samples: int,
        rng_key: Optional[Any] = None,
        observations: Optional[Any] = None,
    ) -> List[Trajectory]:
        result = self.plan(x0, rng_key=rng_key, observations=observations)
        traj = Trajectory(
            states=result["states"],
            actions=result["actions"],
            info=result,
        )
        return [traj for _ in range(max(1, n_samples))]

    def plan_batch(
        self,
        x0: Any,
        keys: Any,
        observations: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        results = []
        for i in range(keys.shape[0]):
            r = self.plan(x0, rng_key=keys[i], observations=observations)
            results.append(r)
        return results
