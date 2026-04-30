"""
MBD3D method plugin with active view selection (exp2B / exp3).

This plugin reuses the regular MBD3D solver (one chain per initial-view pool) but
exposes a `run_active(...)` convenience that orchestrates the
`ActiveSelectionLoop`:

    for round in 1..K:
        score candidates via posterior variance
        pick top-n, add to active bundle
        re-run MBD3D on the enlarged bundle
        record metrics

Implementation note: this plugin only wraps planner construction; the loop
itself lives in `genedynamics.solvers.single.mbd3d.active.selection_loop`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from genedynamics.data import ObservationBundle

from ...framework.base import MethodPlugin
from genedynamics.solvers.single.mbd3d import MBD3DSolver
from genedynamics.solvers.single.mbd3d.active import (
    ActiveSelectionLoop,
    PosteriorVarianceScorer,
    RandomScorer,
    MaxDistanceScorer,
    SelectionRecord,
    ViewScorer,
)
from genedynamics.solvers.single.mbd3d.types import SceneParams

from .mbd3d import MBD3DMethodPlugin


def _make_scorer(name: str, seed: int = 0) -> ViewScorer:
    name = name.lower()
    if name in ("posterior_variance", "variance", "ours"):
        return PosteriorVarianceScorer()
    if name == "random":
        return RandomScorer(seed=seed)
    if name in ("max_distance", "maxdist", "max_dist"):
        return MaxDistanceScorer(seed=seed)
    raise ValueError(f"Unknown active scorer: {name!r}")


class MBD3DActiveMethodPlugin(MethodPlugin):
    """Wraps MBD3D with an active-selection loop around `reconstruct_fn`."""

    @property
    def name(self) -> str:
        return "mbd3d_active"

    def create_planner(self, env: Any, energy: Any, config: Dict[str, Any]) -> MBD3DSolver:
        # Reuse base MBD3D planner construction verbatim.
        return MBD3DMethodPlugin().create_planner(env, energy, config)

    def plan(self, planner: MBD3DSolver, initial_state: np.ndarray, rng: Any) -> Dict[str, Any]:
        # Backwards-compatible: if someone calls plan(), act like regular MBD3D.
        return MBD3DMethodPlugin().plan(planner, initial_state, rng)

    def run_active(
        self,
        env: Any,
        energy: Any,
        method_config: Dict[str, Any],
        *,
        scorer_name: str = "posterior_variance",
        init_indices: List[int],
        n_rounds: int = 5,
        n_select_per_round: int = 1,
        rng_seed: int = 0,
    ) -> Tuple[List[SelectionRecord], List[SceneParams]]:
        """Run the active-view-selection loop over candidates from env.

        The env must expose `get_observations()` returning the *full* candidate
        pool (all frames), from which `init_indices` seeds are drawn.
        """
        full_bundle: ObservationBundle = env.get_observations()
        images = np.asarray(full_bundle.images, dtype=np.float32)
        poses = np.asarray(full_bundle.camera_poses, dtype=np.float32)
        intrinsics = (
            np.asarray(full_bundle.intrinsics, dtype=np.float32)
            if full_bundle.intrinsics is not None
            else None
        )

        scorer = _make_scorer(scorer_name, seed=rng_seed)

        # Reconstruction closure: build planner per round from the active bundle.
        def reconstruct_fn(
            bundle: ObservationBundle, ctx: Dict[str, Any]
        ) -> Tuple[List[SceneParams], Dict[str, float]]:
            # Force the env to expose the active bundle during plan()
            env._observations = bundle  # type: ignore[attr-defined]
            planner = MBD3DMethodPlugin().create_planner(env, energy, method_config)
            out = MBD3DMethodPlugin().plan(
                planner, np.asarray(bundle.camera_poses[0], dtype=np.float32),
                rng=np.random.default_rng(rng_seed + ctx.get("round", 0)),
            )
            scene = out.get("scene_params")
            metrics = dict(out.get("metrics_3dgs", {}))
            metrics["n_active_views"] = int(bundle.num_views)
            return [scene] if scene is not None else [], metrics

        # Renderer — grab it from a throwaway planner.
        probe_planner = MBD3DMethodPlugin().create_planner(env, energy, method_config)
        renderer = getattr(probe_planner, "renderer", None) or \
            (getattr(probe_planner, "config", {}) or {}).get("renderer")

        loop = ActiveSelectionLoop(
            scorer=scorer,
            reconstruct_fn=reconstruct_fn,
            candidate_images=images,
            candidate_poses=poses,
            intrinsics=intrinsics,
            init_indices=list(init_indices),
            n_rounds=n_rounds,
            n_select_per_round=n_select_per_round,
            renderer=renderer,
        )
        records = loop.run()

        # Final posterior = most recent round's reconstruction.
        last_bundle_ids = records[-1].active_indices if records else init_indices
        env._observations = ObservationBundle(
            images=images[last_bundle_ids],
            camera_poses=poses[last_bundle_ids],
            intrinsics=intrinsics,
        )
        final_planner = MBD3DMethodPlugin().create_planner(env, energy, method_config)
        final = MBD3DMethodPlugin().plan(
            final_planner,
            np.asarray(poses[last_bundle_ids[0]], dtype=np.float32),
            rng=np.random.default_rng(rng_seed + 999),
        )
        final_scene = final.get("scene_params")
        return records, ([final_scene] if final_scene is not None else [])
