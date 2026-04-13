"""Phase 9 deploy runner — load a preset, build components, run a loop.

This is the new entry point for the registry-driven config system. It's
intentionally separate from the legacy ``deploy/cli.py`` so the migration
can land without disturbing the existing flat-config CLI; once every
caller has flipped, the legacy module can be deleted.

Usage::

    python -m genedynamics.deploy.runner \\
        --preset genedynamics.deploy.presets:G1CorridorMujocoSportModePreset

The runner does five things and nothing more:

1. Import the preset class given by ``module:ClassName``.
2. Resolve every component via :func:`build_components`.
3. Run an episode through ``deploy/pipeline.py``'s loop helper if one is
   available, otherwise a tiny in-line loop. The plan was always for the
   "single pipeline" loop to be < 200 lines; this runner is simply where
   it gets called from in the registry-driven world.
4. Forward observer hooks at episode start / step / end.
5. Return the assembled :class:`BuiltComponents` for testing or REPL use.
"""

from __future__ import annotations

import argparse
import importlib
import logging
from typing import Any, Optional

from genedynamics.deploy.config_schema import (
    BuiltComponents,
    DeployConfig,
    build_components,
)

logger = logging.getLogger(__name__)

__all__ = ["load_preset", "build_from_preset", "run_preset", "main"]


def load_preset(spec: str) -> type[DeployConfig]:
    """Import a preset class from a ``module:ClassName`` string.

    Raises:
        ValueError: if ``spec`` is malformed.
        ImportError: if the module fails to import.
        AttributeError: if the class is missing.
    """
    if ":" not in spec:
        raise ValueError(
            f"preset spec must be 'module.path:ClassName', got {spec!r}"
        )
    module_path, class_name = spec.split(":", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


def build_from_preset(preset: type[DeployConfig] | DeployConfig) -> BuiltComponents:
    """Resolve a preset to its instantiated components."""
    return build_components(preset)


def run_preset(
    preset: type[DeployConfig] | DeployConfig,
    *,
    episode_id: str = "ep_0001",
    max_steps: Optional[int] = None,
) -> BuiltComponents:
    """Build a preset and run a single episode through it.

    The loop is intentionally minimal — it's the same shape as the one
    sketched in §6 of the deploy refactor plan, with the explicit
    follower/safety/observer hookups. Components that the preset omits
    are handled gracefully (no follower → empty intent; no safety →
    pass-through; no observers → no telemetry).
    """
    built = build_from_preset(preset)
    if built.io is None:
        raise RuntimeError("Preset is missing the 'io' section; nothing to run")
    if built.controller is None:
        raise RuntimeError("Preset is missing the 'controller' section; nothing to run")

    cfg = preset
    sim_dt = float(getattr(cfg, "sim_dt", 1.0 / 200.0))
    steps = int(max_steps if max_steps is not None else getattr(cfg, "max_steps", 1000))

    state = built.io.reset()
    if hasattr(built.controller, "reset"):
        try:
            built.controller.reset(built.io)
        except TypeError:
            built.controller.reset()
    if built.safety is not None and hasattr(built.safety, "reset"):
        built.safety.reset()
    if built.follower is not None and hasattr(built.follower, "reset"):
        try:
            built.follower.reset()
        except TypeError:
            logger.warning(
                "Follower %s.reset() signature mismatch; called without args",
                type(built.follower).__name__,
            )
    if built.task is not None and hasattr(built.task, "reset"):
        try:
            built.task.reset(built.io)
        except TypeError:
            try:
                built.task.reset()
            except Exception as exc:
                raise RuntimeError(
                    f"Task {type(built.task).__name__}.reset() failed: {exc}"
                ) from exc

    for obs in built.observers:
        try:
            obs.on_episode_start(episode_id, {"preset": type(preset).__name__})
        except Exception as exc:
            logger.error(
                "Observer %s.on_episode_start failed: %s",
                type(obs).__name__, exc, exc_info=True,
            )

    from genedynamics.deploy.interfaces.messages import Intent, StepInfo
    import numpy as np

    empty_intent = Intent(
        t=0.0,
        base_yaw=0.0,
        base_height=float(getattr(cfg, "base_height", 0.75)),
        base_lin_vel=np.zeros(2, dtype=np.float64),
        base_yaw_rate=0.0,
    )

    actual_steps = 0
    for step in range(steps):
        t = state.t
        intent = (
            built.follower.step(t, state)
            if built.follower is not None and hasattr(built.follower, "step")
            else empty_intent
        )
        cmd = built.controller.act(state, intent)
        if built.safety is not None:
            cmd = built.safety.filter(state, cmd).command
        built.io.send_control(cmd)
        state = built.io.step(sim_dt)
        info = (
            built.task.step(state, intent, cmd)
            if built.task is not None and hasattr(built.task, "step")
            else StepInfo(done=False)
        )
        for obs in built.observers:
            try:
                obs.on_step(state.t, state, intent, cmd, info)
            except Exception as exc:
                logger.error(
                    "Observer %s.on_step failed at t=%.3f: %s",
                    type(obs).__name__, state.t, exc, exc_info=True,
                )
        actual_steps += 1
        if info.done:
            break

    summary: dict[str, Any] = {
        "steps": actual_steps,
        "max_steps_reached": actual_steps >= steps,
        "preset": type(preset).__name__,
    }
    if built.task is not None and hasattr(built.task, "summary"):
        try:
            summary.update(dict(built.task.summary()))
        except Exception as exc:
            logger.warning("Task summary failed: %s", exc)
    for obs in built.observers:
        try:
            obs.on_episode_end(summary)
        except Exception as exc:
            logger.error(
                "Observer %s.on_episode_end failed: %s",
                type(obs).__name__, exc, exc_info=True,
            )

    built.io.close()
    return built


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 9 deploy runner (registry-driven, nested-class configs)"
    )
    parser.add_argument(
        "--preset",
        required=True,
        help="Preset spec as 'module.path:ClassName'",
    )
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--episode-id", type=str, default="ep_0001")
    args = parser.parse_args()

    preset_cls = load_preset(args.preset)
    run_preset(preset_cls, episode_id=args.episode_id, max_steps=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
