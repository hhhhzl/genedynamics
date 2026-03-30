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
        from genedynamics.deploy.sim_plan.stepping_walk_follower_v2 import (
            SteppingWalkFollowerMinimal,
            MinimalFollowerConfig,
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
        centerline_y = float(
            method_params.get(
                "centerline_y",
                env_params.get(
                    "centerline_y",
                    0.5 * float(env_params.get("start_mid", (-1.25, 0.0))[1])
                    + 0.5 * float(env_params.get("goal_mid", (1.25, 0.0))[1]),
                ),
            )
        )
        follower_cfg = MinimalFollowerConfig(
            gait=str(method_params.get("gait", "walk")),
            sim_dt=float(method_params.get("sim_dt", 0.01)),
            phase_steps=int(method_params.get("phase_steps", 36)),
            settle_steps=int(method_params.get("settle_steps", 20)),
            step_width=step_width,
            centerline_y=centerline_y,
            leg_half_length=float(method_params.get("leg_half_length", 0.18)),
            x_f_nominal=float(method_params.get("x_f_nominal", 0.18)),
            x_r_nominal=float(method_params.get("x_r_nominal", -0.18)),
            y_L_nominal=float(method_params.get("y_L_nominal", 0.15)),
            y_R_nominal=float(method_params.get("y_R_nominal", -0.15)),
            base_height_offset=float(method_params.get("base_height_offset", 0.01)),
            base_z_min=float(method_params.get("base_z_min", 0.23)),
            min_foot_z=float(method_params.get("min_foot_z", 0.015)),
            swing_height=float(method_params.get("swing_height", 0.045)),
            swing_stance_phase=float(method_params.get("swing_stance_phase", 0.5)),
            swing_use_stance_gate=bool(method_params.get("swing_use_stance_gate", True)),
            min_swing_clearance=float(method_params.get("min_swing_clearance", 0.0)),
            min_swing_clearance_first_n_intervals=int(
                method_params.get("min_swing_clearance_first_n_intervals", 0)
            ),
            use_base_pd=bool(method_params.get("use_base_pd", True)),
            base_kp_xy=float(method_params.get("base_kp_xy", 44.0)),
            base_kd_xy=float(method_params.get("base_kd_xy", 11.0)),
            base_kp_z=float(method_params.get("base_kp_z", 260.0)),
            base_kd_z=float(method_params.get("base_kd_z", 34.0)),
            base_kp_rp=float(method_params.get("base_kp_rp", 95.0)),
            base_kd_rp=float(method_params.get("base_kd_rp", 12.0)),
            base_kp_yaw=float(method_params.get("base_kp_yaw", 35.0)),
            base_kd_yaw=float(method_params.get("base_kd_yaw", 6.0)),
            base_weight_comp=float(method_params.get("base_weight_comp", 1.0)),
            base_force_xy_clip=float(method_params.get("base_force_xy_clip", 90.0)),
            base_force_z_clip=float(method_params.get("base_force_z_clip", 260.0)),
            base_torque_clip=float(method_params.get("base_torque_clip", 48.0)),
            base_support_contact_min=int(method_params.get("base_support_contact_min", 2)),
            base_support_scale_min=float(method_params.get("base_support_scale_min", 0.0)),
            use_dynamic_base_z_ref=bool(method_params.get("use_dynamic_base_z_ref", True)),
            base_target_clearance_from_feet=float(
                method_params.get("base_target_clearance_from_feet", 0.31)
            ),
            use_stance_force_distribution=bool(
                method_params.get("use_stance_force_distribution", True)
            ),
            use_stance_qp=bool(method_params.get("use_stance_qp", True)),
            stance_qp_solver_order=tuple(
                method_params.get("stance_qp_solver_order", ["clarabel", "osqp", "cvxopt"])
            ),
            stance_fd_lambda=float(method_params.get("stance_fd_lambda", 1e-3)),
            stance_fd_mu=float(method_params.get("stance_fd_mu", 0.6)),
            stance_fd_fz_min=float(method_params.get("stance_fd_fz_min", 5.0)),
            stance_fd_fz_max=float(method_params.get("stance_fd_fz_max", 220.0)),
            stance_fd_torque_blend=float(method_params.get("stance_fd_torque_blend", 0.65)),
            swing_kp_xy=float(method_params.get("swing_kp_xy", 170.0)),
            swing_kp_z=float(method_params.get("swing_kp_z", 230.0)),
            swing_kd_xy=float(method_params.get("swing_kd_xy", 14.0)),
            swing_kd_z=float(method_params.get("swing_kd_z", 18.0)),
            use_swing_jointspace_tracking=bool(
                method_params.get("use_swing_jointspace_tracking", True)
            ),
            swing_ik_kp_pos=float(method_params.get("swing_ik_kp_pos", 18.0)),
            swing_ik_damping=float(method_params.get("swing_ik_damping", 0.010)),
            swing_joint_kp=float(method_params.get("swing_joint_kp", 80.0)),
            swing_joint_kd=float(method_params.get("swing_joint_kd", 5.0)),
            swing_q_step_clip=float(method_params.get("swing_q_step_clip", 0.22)),
            swing_qd_ref_clip=float(method_params.get("swing_qd_ref_clip", 8.0)),
            swing_jointspace_tau_blend=float(
                method_params.get("swing_jointspace_tau_blend", 0.35)
            ),
            stance_kp_xy=float(method_params.get("stance_kp_xy", 110.0)),
            stance_kp_z=float(method_params.get("stance_kp_z", 135.0)),
            stance_kd_xy=float(method_params.get("stance_kd_xy", 24.0)),
            stance_kd_z=float(method_params.get("stance_kd_z", 16.0)),
            flat_only_foot_lock=bool(method_params.get("flat_only_foot_lock", False)),
            flat_lock_stance_kp_xy_scale=float(
                method_params.get("flat_lock_stance_kp_xy_scale", 1.8)
            ),
            flat_lock_stance_kp_z_scale=float(
                method_params.get("flat_lock_stance_kp_z_scale", 1.25)
            ),
            flat_lock_stance_kd_xy_scale=float(
                method_params.get("flat_lock_stance_kd_xy_scale", 1.35)
            ),
            flat_lock_stance_kd_z_scale=float(
                method_params.get("flat_lock_stance_kd_z_scale", 1.20)
            ),
            flat_lock_max_anchor_error_xy=float(
                method_params.get("flat_lock_max_anchor_error_xy", 0.06)
            ),
            joint_kp=float(method_params.get("joint_kp", 26.0)),
            joint_kd=float(method_params.get("joint_kd", 1.4)),
            joint_torque_clip=float(method_params.get("joint_torque_clip", 3.0)),
            task_torque_clip=float(method_params.get("task_torque_clip", 12.0)),
            touchdown_force_thresh=float(method_params.get("touchdown_force_thresh", 12.0)),
            touchdown_force_thresh_stone=float(
                method_params.get("touchdown_force_thresh_stone", 12.0)
            ),
            touchdown_force_thresh_bank=float(
                method_params.get("touchdown_force_thresh_bank", 10.0)
            ),
            touchdown_force_thresh_river=float(
                method_params.get("touchdown_force_thresh_river", 8.0)
            ),
            touchdown_tangent_speed_thresh=float(
                method_params.get("touchdown_tangent_speed_thresh", 0.20)
            ),
            touchdown_xy_tol=float(method_params.get("touchdown_xy_tol", 0.06)),
            touchdown_stable_steps=int(method_params.get("touchdown_stable_steps", 2)),
            post_touchdown_hold_steps=int(method_params.get("post_touchdown_hold_steps", 8)),
            strong_touchdown_force_thresh=float(
                method_params.get("strong_touchdown_force_thresh", 28.0)
            ),
            strong_touchdown_force_thresh_stone=float(
                method_params.get("strong_touchdown_force_thresh_stone", 28.0)
            ),
            strong_touchdown_force_thresh_bank=float(
                method_params.get("strong_touchdown_force_thresh_bank", 22.0)
            ),
            strong_touchdown_force_thresh_river=float(
                method_params.get("strong_touchdown_force_thresh_river", 18.0)
            ),
            touchdown_counter_decay_on_contact=int(
                method_params.get("touchdown_counter_decay_on_contact", 1)
            ),
            allow_early_switch=bool(method_params.get("allow_early_switch", False)),
            allow_early_switch_on_contact=bool(
                method_params.get("allow_early_switch_on_contact", False)
            ),
            early_switch_contact_alpha_min=float(
                method_params.get("early_switch_contact_alpha_min", 0.55)
            ),
            roll_pitch_abort_rad=float(method_params.get("roll_pitch_abort_rad", 0.45)),
            max_anchor_error_xy=float(method_params.get("max_anchor_error_xy", 0.04)),
            max_swing_tracking_error_xy=float(method_params.get("max_swing_tracking_error_xy", 0.08)),
            stance_press_z_offset=float(method_params.get("stance_press_z_offset", -0.004)),
            stance_plan_blend=float(method_params.get("stance_plan_blend", 0.65)),
            uniform_base_speed=bool(method_params.get("uniform_base_speed", True)),
            base_speed_mps=float(method_params.get("base_speed_mps", 0.16)),
            support_contact_abort_steps=int(method_params.get("support_contact_abort_steps", 80)),
            touchdown_timeout_alpha=float(method_params.get("touchdown_timeout_alpha", 0.90)),
            touchdown_timeout_steps=int(method_params.get("touchdown_timeout_steps", 16)),
            force_interval_end_on_touchdown_timeout=bool(
                method_params.get("force_interval_end_on_touchdown_timeout", True)
            ),
            abort_after_consecutive_timeouts=int(
                method_params.get("abort_after_consecutive_timeouts", 2)
            ),
        )

        def _resolve_scene_dict(seed_dir: Path) -> Optional[Dict[str, Any]]:
            if bool(method_params.get("straight_stones_scene", False)):
                start_mid_v = np.asarray(env_params.get("start_mid", (-1.25, 0.0)), dtype=np.float64)
                goal_mid_v = np.asarray(env_params.get("goal_mid", (1.25, 0.0)), dtype=np.float64)
                step_w = float(step_width)
                half_w = 0.5 * step_w
                stone_r_req = float(method_params.get("straight_stones_radius", 0.085))
                n_pairs = int(max(0, method_params.get("straight_stones_pairs", 8)))
                side_pad = float(method_params.get("straight_stones_side_pad", 0.02))
                deck_half_len = float(method_params.get("straight_platform_half_length", 0.24))
                deck_half_w = float(method_params.get("straight_platform_half_width", half_w + stone_r_req + 0.02))
                lane_gap_to_deck = float(method_params.get("straight_lane_gap_to_platform", 0.10))

                x0 = float(start_mid_v[0])
                x1 = float(goal_mid_v[0])
                y0 = float(start_mid_v[1])
                sign = 1.0 if x1 >= x0 else -1.0
                lane_x0 = x0 + sign * (deck_half_len + lane_gap_to_deck)
                lane_x1 = x1 - sign * (deck_half_len + lane_gap_to_deck)
                if n_pairs > 0:
                    if sign > 0:
                        xs = np.linspace(min(lane_x0, lane_x1), max(lane_x0, lane_x1), n_pairs, dtype=np.float64)
                    else:
                        xs = np.linspace(max(lane_x0, lane_x1), min(lane_x0, lane_x1), n_pairs, dtype=np.float64)
                    if n_pairs >= 2:
                        dx = float(abs(xs[1] - xs[0]))
                        # Keep visible separation so stones form a clean lane, not overlapped "clusters".
                        stone_r = float(min(stone_r_req, max(0.015, 0.48 * dx)))
                    else:
                        stone_r = float(stone_r_req)
                    yL = y0 + half_w
                    yR = y0 - half_w
                    cL = np.stack([xs, np.full_like(xs, yL)], axis=1)
                    cR = np.stack([xs, np.full_like(xs, yR)], axis=1)
                    stones = np.concatenate([cL, cR], axis=0).astype(np.float32)
                    radii = np.full((2 * n_pairs,), stone_r, dtype=np.float32)
                else:
                    stones = np.zeros((0, 2), dtype=np.float32)
                    radii = np.zeros((0,), dtype=np.float32)

                start_plat = [x0 - deck_half_len, x0 + deck_half_len, y0 - deck_half_w, y0 + deck_half_w]
                goal_plat = [x1 - deck_half_len, x1 + deck_half_len, y0 - deck_half_w, y0 + deck_half_w]
                map_x = tuple(method_params.get("straight_map_x", (-2.0, 2.0)))
                map_y = tuple(method_params.get("straight_map_y", (-1.0, 1.0)))
                return {
                    "map_x": [float(map_x[0]), float(map_x[1])],
                    "map_y": [float(map_y[0]), float(map_y[1])],
                    "river_x": [0.0, 0.0],
                    "has_river": False,
                    "stones_centers": stones.tolist(),
                    "stones_radii": radii.tolist(),
                    "support_platforms": [start_plat, goal_plat],
                }
            if bool(method_params.get("flat_ground_scene", False)):
                map_x = tuple(method_params.get("flat_ground_map_x", (-2.0, 2.0)))
                map_y = tuple(method_params.get("flat_ground_map_y", (-1.0, 1.0)))
                return {
                    "map_x": [float(map_x[0]), float(map_x[1])],
                    "map_y": [float(map_y[0]), float(map_y[1])],
                    "river_x": [0.0, 0.0],
                    "has_river": False,
                    "stones_centers": [],
                    "stones_radii": [],
                    "support_platforms": [],
                }
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
            follower = SteppingWalkFollowerMinimal(cfg=follower_cfg, stepping_scene=scene_dict)
            plan = load_stepping_plan_from_seed_dir(
                seed_dir,
                step_width=step_width,
                centerline_y=float(follower_cfg.centerline_y),
                x_f_nominal=float(follower_cfg.x_f_nominal),
                x_r_nominal=float(follower_cfg.x_r_nominal),
                y_L_nominal=float(follower_cfg.y_L_nominal),
                y_R_nominal=float(follower_cfg.y_R_nominal),
            )
            roll = follower.rollout(plan_states=plan["states"])
            qpos = np.asarray(roll.get("qpos", np.zeros((0, int(follower.model.nq)))), dtype=np.float32)
            qvel = np.asarray(roll.get("qvel", np.zeros((0, int(follower.model.nv)))), dtype=np.float32)
            ctrl = np.asarray(roll.get("ctrl", np.zeros((0, int(follower.model.nu)))), dtype=np.float32)
            summary = dict(roll.get("summary", {}))
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
            swing_ref = roll.get("swing_ref", {})
            if isinstance(swing_ref, dict) and len(swing_ref) > 0:
                np.savez(
                    ep_path / "swing_ref_targets.npz",
                    **{str(k): np.asarray(v, dtype=np.float32) for k, v in swing_ref.items()},
                )
            with open(ep_path / "rollout_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            with open(ep_path / "rollout_interval_stats.json", "w", encoding="utf-8") as f:
                json.dump(list(roll.get("interval_stats", [])), f, indent=2)

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
                    "goal_error_xy": float(summary.get("goal_error_xy", float("nan"))),
                    "pitch_abs_max": float(summary.get("pitch_abs_max", 0.0)),
                    "interval_timeout_ratio": float(summary.get("interval_timeout_ratio", 0.0)),
                    "touchdown_any_ratio": float(summary.get("touchdown_any_ratio", 0.0)),
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
