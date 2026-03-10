"""
MBD3D method plugin for 3DGS robust mapping.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from genedynamics.solvers.single.mbd3d import MBD3DSolver
from genedynamics.solvers.single.mbd3d.types import ObservationBundle
from genedynamics.solvers.single.mbd3d.implementations import (
    GaussianSplatScene,
    GaussianObservationLikelihood,
    JaxSplatRenderer,
    JAX_SPLAT_AVAILABLE,
    GsplatRenderer,
    GSPLAT_AVAILABLE,
)
from genedynamics.core.dynamics.adapters import EnvDynamicsAdapter
from genedynamics.core.backends.runtime import RuntimeBackendManager
from genedynamics.core.inference.annealed_bridge import create_linear_bridge_schedule
from ...framework.base import MethodPlugin
from ._result_utils import normalize_result_from_trajectory


def _make_observation_bundle(env: Any, config: Dict[str, Any]) -> ObservationBundle:
    """Build ObservationBundle from env or config or synthetic."""
    if hasattr(env, "get_observations") and callable(getattr(env, "get_observations")):
        obs = env.get_observations()
        if config.get("initial_scene") is not None:
            obs.initial_scene = config["initial_scene"]
        if config.get("initial_scene_center") is not None:
            obs.initial_scene = config["initial_scene_center"]
        return obs
    obs = config.get("observations")
    if obs is not None and isinstance(obs, ObservationBundle):
        return obs
    # Synthetic observations for testing
    n_views = config.get("n_views", config.get("horizon", 10) + 1)
    H, W, C = config.get("image_height", 64), config.get("image_width", 64), config.get("image_channels", 3)
    images = np.random.randn(n_views, H, W, C).astype(np.float32) * 0.1
    camera_poses = np.random.randn(n_views, 7).astype(np.float32) * 0.1
    camera_poses[:, 3:7] = camera_poses[:, 3:7] / (np.linalg.norm(camera_poses[:, 3:7], axis=1, keepdims=True) + 1e-8)
    return ObservationBundle(images=images, camera_poses=camera_poses)


class MBD3DMethodPlugin(MethodPlugin):
    """
    Plugin for MBD3D (3DGS robust mapping) solver.
    """

    @property
    def name(self) -> str:
        return "mbd3d"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> MBD3DSolver:
        backend = RuntimeBackendManager.get_backend()
        dynamics = EnvDynamicsAdapter(env)
        horizon = config.get("horizon", 10)
        if hasattr(env, "get_observations") and callable(getattr(env, "get_observations")):
            obs = env.get_observations()
            n_views = obs.num_views if hasattr(obs, "num_views") else getattr(obs.images, "shape", [0])[0]
            if n_views > 0:
                horizon = max(0, n_views - 1)
        dt = config.get("dt", 0.1)

        n_gaussians = config.get("n_gaussians", 32)
        prior_center = config.get("initial_scene_center", None)
        scene_repr = GaussianSplatScene(
            n_gaussians=n_gaussians,
            prior_scale_means=config.get("prior_scale_means", 1.0),
            prior_scale_scales=config.get("prior_scale_scales", 0.1),
        )
        if prior_center is not None and hasattr(scene_repr, "set_prior_center"):
            scene_repr.set_prior_center(prior_center)

        img_h = config.get("image_height", 64)
        img_w = config.get("image_width", 64)
        img_c = config.get("image_channels", 3)
        use_real_splat = config.get("use_real_splat", True)
        enable_subspace = bool(config.get("enable_subspace", True))
        prefer_jax_renderer = bool(config.get("prefer_jax_renderer", enable_subspace))

        if prefer_jax_renderer and JAX_SPLAT_AVAILABLE and JaxSplatRenderer is not None:
            renderer = JaxSplatRenderer(
                image_height=img_h,
                image_width=img_w,
                image_channels=img_c,
            )
        elif use_real_splat and GSPLAT_AVAILABLE and GsplatRenderer is not None:
            renderer = GsplatRenderer(
                image_height=img_h,
                image_width=img_w,
                image_channels=img_c,
            )
        elif use_real_splat and JAX_SPLAT_AVAILABLE and JaxSplatRenderer is not None:
            renderer = JaxSplatRenderer(
                image_height=img_h,
                image_width=img_w,
                image_channels=img_c,
            )
        else:
            raise RuntimeError(
                "No production renderer available. Install gsplat or enable JAX renderer."
            )

        likelihood = GaussianObservationLikelihood(
            renderer=renderer,
            sigma2=config.get("observation_sigma2", 0.01),
            use_lowrank=config.get("use_lowrank_noise", False),
            backend="jax",
            lowrank_view_rank=config.get("lowrank_view_rank", 4),
            lowrank_spatial_rank=config.get("lowrank_spatial_rank", 2),
            lowrank_color_rank=config.get("lowrank_color_rank", 3),
            lowrank_max_rank=config.get("lowrank_max_rank", 16),
        )

        bridge_K = config.get("bridge_K", 50)
        eta_start = config.get("bridge_eta", 0.02)
        eta_end = config.get("bridge_eta_end", eta_start * 0.1)
        eta_schedule = np.linspace(float(eta_start), float(eta_end), bridge_K, dtype=np.float32).tolist()
        bridge_schedule = create_linear_bridge_schedule(
            K=bridge_K,
            beta0=config.get("bridge_beta0", 0.0),
            betaK=config.get("bridge_betaK", 1.0),
            eta_schedule=eta_schedule,
        )

        solver = MBD3DSolver(
            dynamics=dynamics,
            energy=energy,
            backend=backend,
            scene_repr=scene_repr,
            renderer=renderer,
            likelihood=likelihood,
            bridge_schedule=bridge_schedule,
            horizon=horizon,
            n_gaussians=n_gaussians,
            M=config.get("M", 16),
            sigma_mcsa=config.get("sigma_mcsa", 0.05),
            ess_min=config.get("ess_min", 1.0),
            seed=config.get("np_random_seed", 0),
            show_tqdm=config.get("show_tqdm", False),
            fix_cameras=config.get("fix_cameras", True),
            initialization_mode=config.get("initialization_mode", "prior_center"),
            init_jitter_scale=config.get("init_jitter_scale", 1.0),
            enable_subspace=enable_subspace,
            subspace_rank=config.get("subspace_rank", 64),
            subspace_rank_start=config.get("subspace_rank_start", config.get("subspace_rank", 64)),
            subspace_rank_end=config.get("subspace_rank_end", config.get("subspace_rank", 64)),
            subspace_power_iters=config.get("subspace_power_iters", 2),
            subspace_refresh_every=config.get("subspace_refresh_every", 1),
            subspace_refresh_every_start=config.get(
                "subspace_refresh_every_start", config.get("subspace_refresh_every", 1)
            ),
            subspace_refresh_every_end=config.get(
                "subspace_refresh_every_end", config.get("subspace_refresh_every", 1)
            ),
            proposal_count_start=config.get("proposal_count_start", config.get("M", 16)),
            proposal_count_end=config.get("proposal_count_end", config.get("M", 16)),
            profiling=config.get("profiling", True),
            subspace_oversample=config.get("subspace_oversample", 2),
            compile_stable_shapes=config.get("compile_stable_shapes", True),
            fidelity_ladder=config.get("fidelity_ladder"),
        )
        return solver

    def plan(self, planner: MBD3DSolver, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        env = getattr(planner.dynamics, "env", None) if hasattr(planner, "dynamics") else None
        config = getattr(planner, "config", {})
        observations = _make_observation_bundle(env or {}, config)
        traj = planner.solve(
            initial_state,
            horizon=planner.horizon,
            observations=observations,
            rng_key=rng,
        )
        return normalize_result_from_trajectory(traj, initial_state=initial_state, preserve_info=True)
