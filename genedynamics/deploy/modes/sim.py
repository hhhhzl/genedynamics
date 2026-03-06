"""
Sim mode: run simulation with full control loop.
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
from genedynamics.execution.providers.sim_state_provider import SimStateProvider
from genedynamics.execution.publishers.sim_control_publisher import SimControlPublisher
from genedynamics.execution.logging.telemetry import TelemetryLogger
from genedynamics.execution.logging.episode_writer import EpisodeWriter


class SimMode:
    """Simulation execution mode."""

    def run(
        self,
        config: DeployConfig,
        profile: RobotProfile,
        env: Any,
        planner: Any,
    ) -> Dict[str, Any]:
        """Run sim episodes."""
        nq, nv, act_dim = profile.infer_spec(env)
        ctrl_lim = float(getattr(env, "control_limit", 1.0))

        state_provider = SimStateProvider(env, nq=nq, nv=nv)
        control_publisher = SimControlPublisher(env)
        planner_bridge = PlannerBridge(
            planner,
            horizon=config.horizon,
            plan_mode=config.plan_mode,
        )
        safety_guard = SafetyGuard(
            action_clip_min=np.full(act_dim, -ctrl_lim, dtype=np.float32),
            action_clip_max=np.full(act_dim, ctrl_lim, dtype=np.float32),
            fallback_action=np.zeros(act_dim, dtype=np.float32),
        )

        session_config = SessionConfig(
            execution_mode=ExMode.SIM_ONLY,
            control_rate_hz=config.control_rate_hz,
            plan_rate_hz=config.plan_rate_hz,
            real_time_factor=config.sim.real_time_factor if config.sim else 1.0,
            sync_mode=config.sim.sync_mode if config.sim else True,
            record=config.record,
            episode_dir=config.get_episode_dir(),
            action_clip_min=np.full(act_dim, -ctrl_lim, dtype=np.float32),
            action_clip_max=np.full(act_dim, ctrl_lim, dtype=np.float32),
            tags=dict(config.tags),
            extra=config.get_session_config_extra(),
        )

        episode_writer = EpisodeWriter(
            config.get_episode_dir(),
            tags=session_config.tags,
        ) if config.get_episode_dir() else None
        telemetry = TelemetryLogger(enabled=config.record)

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
            np.random.seed(seed)
            rng = backend.create_rng(seed)
            executor.initialize(session_cfg={"rng": rng})

            initial_state = self._get_initial_state(config, env, rng)
            result = executor.run_episode(
                plan_mode=config.plan_mode,
                max_steps=config.max_steps,
                initial_state=initial_state,
                rng=rng,
            )
            results.append(result)
            if result.get("episode_path"):
                episode_paths.append(result["episode_path"])
            print(f"Episode {ep + 1}/{config.episodes}: steps={result['steps']}")

        if hasattr(env, "close"):
            env.close()

        return {
            "mode": "sim",
            "results": results,
            "episode_paths": episode_paths,
            "steps": [r["steps"] for r in results],
        }

    def _get_initial_state(
        self,
        config: DeployConfig,
        env: Any,
        rng: Any,
    ) -> Optional[np.ndarray]:
        """Compute initial state from start config."""
        start = config.start
        if start is None or start.mode == "random":
            return None

        if start.mode == "near_target":
            target = np.asarray(
                config.env_params.get("target", (2.0, 0.0, 0.5)),
                dtype=np.float32,
            )[:3]
            min_d = start.near_target_min_dist
            max_d = start.near_target_max_dist
            distance = np.random.uniform(min_d, max_d)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            direction = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ], dtype=np.float32)
            start_xyz = target + distance * direction
            clip_xy = max(5.0, max_d)
            start_xyz[:2] = np.clip(start_xyz[:2], -clip_xy, clip_xy)
            start_xyz[2] = np.clip(start_xyz[2], start.start_min_z, clip_xy)

            x0, _ = env.reset(rng=rng)
            x0 = np.asarray(x0, dtype=np.float32).ravel()
            nq = getattr(env, "nq", 19)
            x0[:3] = start_xyz
            return x0

        if start.mode == "fixed" and start.fixed_state:
            return np.asarray(start.fixed_state, dtype=np.float32)

        return None
