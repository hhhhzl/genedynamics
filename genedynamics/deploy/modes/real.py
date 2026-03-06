"""
Real mode: deploy on physical robot.

Uses RealStateProvider and RealControlPublisher. Supports:
- state_backend / control_backend from config.extra
- localization_plugin for base pose (ROS2, Vicon, mock)
- Stub backends when no hardware connected.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from genedynamics.deploy.config import DeployConfig
from genedynamics.deploy.profiles.base import RobotProfile
from genedynamics.execution.core.contracts import ExecutionMode as ExMode, SessionConfig
from genedynamics.execution.core.executor import Executor
from genedynamics.execution.core.safety import SafetyGuard
from genedynamics.execution.bridges.planner_bridge import PlannerBridge
from genedynamics.execution.providers.real_state_provider import RealStateProvider
from genedynamics.execution.publishers.real_control_publisher import RealControlPublisher
from genedynamics.execution.logging.telemetry import TelemetryLogger
from genedynamics.execution.logging.episode_writer import EpisodeWriter

from genedynamics.deploy.backends.stub import StubStateBackend as _StubStateBackend
from genedynamics.deploy.backends.stub import StubControlBackend as _StubControlBackend


class RealMode:
    """Real robot execution mode."""

    def run(
        self,
        config: DeployConfig,
        profile: RobotProfile,
        env: Any,
        planner: Any,
    ) -> Dict[str, Any]:
        """Run real deployment (or stub if no backends)."""
        nq, nv, act_dim = profile.get_nq_nv_act_dim()
        ctrl_lim = float(config.env_params.get("control_limit", 1.0))

        state_backend = config.extra.get("state_backend")
        control_backend = config.extra.get("control_backend")

        if state_backend is None:
            real_cfg = config.real
            loc_plugin_name = real_cfg.localization_plugin if real_cfg else "mock"
            if loc_plugin_name and loc_plugin_name != "none":
                try:
                    from genedynamics.deploy.localization import load_plugin
                    plugin_cls = load_plugin(loc_plugin_name)
                    if plugin_cls:
                        plugin_config = {"timeout_sec": getattr(real_cfg, "localization_timeout_sec", 2.0) if real_cfg else 2.0}
                        plugin = plugin_cls(plugin_config)
                        from genedynamics.deploy.backends.localization_adapter import LocalizationStateBackend
                        state_backend = LocalizationStateBackend(plugin, nq=nq, nv=nv)
                    else:
                        state_backend = _StubStateBackend(nq=nq, nv=nv)
                        print("Warning: Localization plugin not found. Using stub.")
                except Exception as e:
                    print(f"Warning: Localization failed ({e}). Using stub.")
                    state_backend = _StubStateBackend(nq=nq, nv=nv)
            else:
                print("Warning: No state backend. Using stub (mock state).")
                state_backend = _StubStateBackend(nq=nq, nv=nv)
        if control_backend is None:
            print("Warning: No control backend. Using stub (no-op control).")
            control_backend = _StubControlBackend()

        state_provider = RealStateProvider(backend=state_backend, nq=nq, nv=nv)
        control_publisher = RealControlPublisher(backend=control_backend, act_dim=act_dim, env=None)

        stand_posture = config.extra.get("stand_posture")
        fallback = np.asarray(stand_posture, dtype=np.float32) if stand_posture else np.zeros(act_dim, dtype=np.float32)

        planner_bridge = PlannerBridge(planner, horizon=config.horizon, plan_mode=config.plan_mode)
        safety_guard = SafetyGuard(
            action_clip_min=np.full(act_dim, -ctrl_lim, dtype=np.float32),
            action_clip_max=np.full(act_dim, ctrl_lim, dtype=np.float32),
            state_bounds=config.extra.get("state_bounds"),
            fallback_action=fallback,
            fallback_reason="safety_trigger",
        )

        session_config = SessionConfig(
            execution_mode=ExMode.REAL_ONLY,
            control_rate_hz=config.control_rate_hz,
            plan_rate_hz=config.plan_rate_hz,
            real_time_factor=1.0,
            sync_mode=True,
            record=config.record,
            episode_dir=config.get_episode_dir(),
            action_clip_min=np.full(act_dim, -ctrl_lim, dtype=np.float32),
            action_clip_max=np.full(act_dim, ctrl_lim, dtype=np.float32),
            state_bounds=config.extra.get("state_bounds"),
            tags=dict(config.tags),
            extra=config.get_session_config_extra(),
        )

        episode_writer = EpisodeWriter(
            config.get_episode_dir(),
            tags=session_config.tags,
        ) if config.get_episode_dir() else None
        telemetry = TelemetryLogger(enabled=config.record)

        if stand_posture:
            control_publisher.set_safe_posture(np.asarray(stand_posture, dtype=np.float32))

        executor = Executor(
            state_provider=state_provider,
            control_publisher=control_publisher,
            planner_bridge=planner_bridge,
            safety_guard=safety_guard,
            config=session_config,
            telemetry=telemetry,
            episode_writer=episode_writer,
        )

        from genedynamics.core.backends.runtime import RuntimeBackendManager
        backend = RuntimeBackendManager.get_backend()

        results: List[Dict[str, Any]] = []
        episode_paths: List[str] = []

        for ep in range(config.episodes):
            seed = config.seed + ep * 1000
            rng = backend.create_rng(seed)
            executor.initialize(session_cfg={"rng": rng})
            try:
                result = executor.run_episode(
                    plan_mode=config.plan_mode,
                    max_steps=config.max_steps,
                    rng=rng,
                )
                results.append(result)
                if result.get("episode_path"):
                    episode_paths.append(result["episode_path"])
                print(f"Episode {ep + 1}/{config.episodes}: steps={result['steps']}")
            except ValueError as e:
                if "initial_state required" in str(e):
                    print("Stub mode: state_provider returns None. Connect real backend for hardware.")
                raise
            finally:
                executor.shutdown()

        return {
            "mode": "real",
            "results": results,
            "episode_paths": episode_paths,
            "steps": [r["steps"] for r in results],
        }
