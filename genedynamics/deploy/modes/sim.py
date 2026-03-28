"""
Sim mode: run simulation with full control loop.
"""

from __future__ import annotations

import json
from pathlib import Path
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
        if config.robot_type == "quadruped" and str(config.planner).lower() in {
            "walk_follow",
            "stepping_walk",
            "stepping_walk_follow",
        }:
            return self._run_quadruped_stepping_walk_follow(config)

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

    def _run_quadruped_stepping_walk_follow(self, config: DeployConfig) -> Dict[str, Any]:
        """
        Offline stepping walk follower:
        plan seed dir -> walk IK/PD rollout -> deploy episode + HTML.
        """
        from genedynamics.deploy.sim_plan.stepping_walk_follower import (
            SteppingWalkFollower,
            WalkFollowerConfig,
            load_stepping_plan_from_seed_dir,
        )
        from genedynamics.execution.logging.episode_writer import EpisodeWriter
        from genedynamics.tasks.stepping_stones import sample_stepping_stones_scene, stepping_scene_to_dict
        from genedynamics.viz.motion_episode import MotionEpisode
        from genedynamics.viz.motion_renderer import MotionRenderer

        env_params = dict(config.env_params or {})
        method_params = dict(config.method_params or {})
        plan_seed_dir = Path(
            env_params.get(
                "plan_seed_dir",
                "results/quadruped/stepping_stones_2d/smoke/mbd/level_1/seed_0",
            )
        )
        if not plan_seed_dir.is_absolute():
            plan_seed_dir = Path.cwd() / plan_seed_dir
        if not plan_seed_dir.exists():
            raise FileNotFoundError(f"plan_seed_dir not found: {plan_seed_dir}")

        step_width = float(
            method_params.get(
                "step_width",
                env_params.get("step_width", env_params.get("stance_width", 0.30)),
            )
        )
        follower_cfg = WalkFollowerConfig(
            gait=str(method_params.get("gait", "walk")),
            sim_dt=float(method_params.get("sim_dt", 0.01)),
            phase_steps=int(method_params.get("phase_steps", 20)),
            swing_height=float(method_params.get("swing_height", 0.08)),
            leg_half_length=float(method_params.get("leg_half_length", 0.18)),
            step_width=step_width,
            kp=float(method_params.get("kp", 35.0)),
            kd=float(method_params.get("kd", 1.8)),
            ik_iters=int(method_params.get("ik_iters", 8)),
            ik_damping=float(method_params.get("ik_damping", 1e-3)),
            ik_tol=float(method_params.get("ik_tol", 2e-3)),
            control_clip=float(method_params.get("control_clip", 3.0)),
            settle_steps=int(method_params.get("settle_steps", 0)),
            phase_extend_factor=float(method_params.get("phase_extend_factor", 2.5)),
            stance_contact_min=int(method_params.get("stance_contact_min", 2)),
            release_contact_steps=int(method_params.get("release_contact_steps", 2)),
            swing_release_height=float(method_params.get("swing_release_height", 0.08)),
            touchdown_contact_steps=int(method_params.get("touchdown_contact_steps", 2)),
            touchdown_xy_tol=float(method_params.get("touchdown_xy_tol", 0.08)),
            support_base_blend=float(method_params.get("support_base_blend", 0.85)),
            lift_phase_ratio=float(method_params.get("lift_phase_ratio", 0.25)),
            advance_phase_ratio=float(method_params.get("advance_phase_ratio", 0.5)),
            swing_task_kp_xy=float(method_params.get("swing_task_kp_xy", 220.0)),
            swing_task_kp_z=float(method_params.get("swing_task_kp_z", 260.0)),
            swing_task_kd_xy=float(method_params.get("swing_task_kd_xy", 18.0)),
            swing_task_kd_z=float(method_params.get("swing_task_kd_z", 22.0)),
            stance_task_kp_xy=float(method_params.get("stance_task_kp_xy", 140.0)),
            stance_task_kp_z=float(method_params.get("stance_task_kp_z", 180.0)),
            stance_task_kd_xy=float(method_params.get("stance_task_kd_xy", 20.0)),
            stance_task_kd_z=float(method_params.get("stance_task_kd_z", 24.0)),
            task_torque_clip=float(method_params.get("task_torque_clip", 8.0)),
            base_height_offset=float(method_params.get("base_height_offset", 0.02)),
            min_foot_z=float(method_params.get("min_foot_z", 0.015)),
            lock_base_pose=bool(method_params.get("lock_base_pose", False)),
            base_z_min=float(method_params.get("base_z_min", 0.22)),
            use_base_pd=bool(method_params.get("use_base_pd", True)),
            touchdown_alpha_min=float(method_params.get("touchdown_alpha_min", 0.55)),
            base_kp_xy=float(method_params.get("base_kp_xy", 40.0)),
            base_kd_xy=float(method_params.get("base_kd_xy", 10.0)),
            base_kp_z=float(method_params.get("base_kp_z", 220.0)),
            base_kd_z=float(method_params.get("base_kd_z", 30.0)),
            base_kp_rp=float(method_params.get("base_kp_rp", 80.0)),
            base_kd_rp=float(method_params.get("base_kd_rp", 10.0)),
            base_kp_yaw=float(method_params.get("base_kp_yaw", 25.0)),
            base_kd_yaw=float(method_params.get("base_kd_yaw", 4.0)),
            base_weight_comp=float(method_params.get("base_weight_comp", 1.0)),
            base_support_contact_min=int(method_params.get("base_support_contact_min", 2)),
            base_force_xy_clip=float(method_params.get("base_force_xy_clip", 80.0)),
            base_force_z_clip=float(method_params.get("base_force_z_clip", 200.0)),
            base_torque_clip=float(method_params.get("base_torque_clip", 40.0)),
            contact_base_kp_scale_x=float(method_params.get("contact_base_kp_scale_x", 2.2)),
            contact_base_kd_scale_x=float(method_params.get("contact_base_kd_scale_x", 1.4)),
            contact_base_kp_scale_y=float(method_params.get("contact_base_kp_scale_y", 0.30)),
            contact_base_kd_scale_y=float(method_params.get("contact_base_kd_scale_y", 0.60)),
            contact_phase_steps_scale=float(method_params.get("contact_phase_steps_scale", 1.35)),
            contact_swing_height_scale=float(method_params.get("contact_swing_height_scale", 0.70)),
            contact_touchdown_immediate_relock=bool(
                method_params.get("contact_touchdown_immediate_relock", True)
            ),
            contact_base_y_support_blend=float(
                method_params.get("contact_base_y_support_blend", 0.75)
            ),
            contact_stance_slip_kp_xy=float(method_params.get("contact_stance_slip_kp_xy", 120.0)),
            contact_stance_slip_kd_xy=float(method_params.get("contact_stance_slip_kd_xy", 35.0)),
            contact_stance_slip_kp_z=float(method_params.get("contact_stance_slip_kp_z", 40.0)),
            contact_stance_slip_kd_z=float(method_params.get("contact_stance_slip_kd_z", 12.0)),
            contact_stance_anchor_err_clip=float(
                method_params.get("contact_stance_anchor_err_clip", 0.03)
            ),
            contact_progress_min_support_ratio=float(
                method_params.get("contact_progress_min_support_ratio", 0.60)
            ),
            contact_early_switch_on_touchdown=bool(
                method_params.get("contact_early_switch_on_touchdown", True)
            ),
            contact_early_switch_min_phase_ratio=float(
                method_params.get("contact_early_switch_min_phase_ratio", 0.35)
            ),
            contact_post_touchdown_stabilize_steps=int(
                method_params.get("contact_post_touchdown_stabilize_steps", 4)
            ),
            contact_base_vel_kd_scale_x=float(
                method_params.get("contact_base_vel_kd_scale_x", 1.0)
            ),
            contact_base_vel_kd_scale_y=float(
                method_params.get("contact_base_vel_kd_scale_y", 0.4)
            ),
            contact_base_yaw_rate_kd_scale=float(
                method_params.get("contact_base_yaw_rate_kd_scale", 1.0)
            ),
        )

        def _resolve_scene_dict(seed_dir: Path) -> Optional[Dict[str, Any]]:
            scene_dict = None
            ex_path = seed_dir / "execution_results.json"
            if ex_path.exists():
                try:
                    with open(ex_path, "r", encoding="utf-8") as f:
                        ex_blob = json.load(f) or {}
                    if isinstance(ex_blob.get("stepping_scene"), dict):
                        scene_dict = ex_blob.get("stepping_scene")
                except Exception:
                    scene_dict = None
            if scene_dict is None:
                try:
                    res_path = seed_dir / "results.json"
                    level = 1
                    seed = 0
                    if res_path.exists():
                        with open(res_path, "r", encoding="utf-8") as f:
                            res_blob = json.load(f) or {}
                        level = int(res_blob.get("level", 1))
                        seed = int(res_blob.get("seed", 0))
                    scene = sample_stepping_stones_scene(
                        level=level,
                        seed=seed,
                        l_max=float(env_params.get("l_max", 0.35)),
                        stance_width=step_width,
                        start_mid=tuple(env_params.get("start_mid", (-1.25, 0.0))),
                        goal_mid=tuple(env_params.get("goal_mid", (1.25, 0.0))),
                    )
                    scene_dict = stepping_scene_to_dict(scene)
                except Exception:
                    scene_dict = None
            return scene_dict

        writer = EpisodeWriter(config.get_episode_dir(), tags=dict(config.tags))
        results: List[Dict[str, Any]] = []
        episode_paths: List[str] = []

        for ep in range(config.episodes):
            seed_dir = plan_seed_dir
            if config.episodes > 1:
                cand = plan_seed_dir.parent / f"seed_{config.seed + ep}"
                if cand.exists():
                    seed_dir = cand

            scene_dict = _resolve_scene_dict(seed_dir)
            follower = SteppingWalkFollower(cfg=follower_cfg, stepping_scene=scene_dict)
            plan = load_stepping_plan_from_seed_dir(seed_dir, step_width=step_width)
            qpos, qvel, ctrl = follower.rollout(
                mid=plan["mid"],
                yaw=plan["yaw"],
                step_width=step_width,
                leg_half_length=float(follower_cfg.leg_half_length),
                feet_ref=plan["feet"],
                mode_ref=plan["mode"],
                tau_ref=plan["tau"],
            )
            states = np.concatenate([qpos, qvel], axis=1)
            ep_path = writer.write_episode(
                states,
                ctrl,
                meta={
                    "env_name": "quadruped_stepping_stones_2d_walk_follow",
                    "planner": "walk_follow",
                    "gait": str(follower_cfg.gait),
                    "robot_type": "quadruped",
                    "model_id": "go2",
                    "source_plan_seed_dir": str(seed_dir),
                },
            )

            # Keep Path-B-style arrays for convenience.
            np.save(ep_path / "qpos.npy", qpos.astype(np.float32))
            np.save(ep_path / "qvel.npy", qvel.astype(np.float32))
            np.save(ep_path / "ctrl.npy", ctrl.astype(np.float32))

            if scene_dict:
                with open(ep_path / "stepping_scene.json", "w", encoding="utf-8") as f:
                    json.dump(scene_dict, f, indent=2)

            # Render preview html/gif.
            max_frames = int(method_params.get("max_render_frames", 140))
            stride = max(1, int(np.ceil(len(states) / max_frames)))
            idx = np.arange(0, len(states), stride, dtype=np.int32)
            ctrl_idx = np.clip(idx[:-1], 0, max(0, len(ctrl) - 1))
            ep_actions = ctrl[ctrl_idx] if len(ctrl) > 0 and ctrl_idx.size > 0 else None
            ep_viz = MotionEpisode(
                states=states[idx],
                actions=ep_actions,
                robot_type="quadruped",
                model_id="go2",
                fps=1.0 / (float(follower_cfg.sim_dt) * float(stride)),
            )
            renderer = MotionRenderer(ep_path)
            html_path = renderer.render_html(
                ep_viz,
                name=f"go2_walk_follow_ep{ep + 1:04d}",
                width=int(method_params.get("render_width", 960)),
                height=int(method_params.get("render_height", 720)),
                fps=ep_viz.fps,
            )

            results.append(
                {
                    "steps": int(ctrl.shape[0]),
                    "frames": int(qpos.shape[0]),
                    "episode_path": str(ep_path),
                    "html_path": str(html_path),
                }
            )
            episode_paths.append(str(ep_path))
            print(
                f"Episode {ep + 1}/{config.episodes}: steps={ctrl.shape[0]} "
                f"(walk_follow/{follower_cfg.gait}, source={seed_dir.name})"
            )

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
