"""
SafeDiffuser solver wrapper for enerdynamics.

This integrates a SafeDiffuser-style safety correction into the same denoising
architecture used by DPCC (vendored `third_party/diffuser`).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from enerdynamics.core.backends import Backend
from enerdynamics.core.backends.runtime import RuntimeBackendManager
from enerdynamics.core.dynamics import DynamicsModel
from enerdynamics.core.energy import EnergyFunctional
from enerdynamics.core.solvers import SamplingSolver
from enerdynamics.core.types import State, Trajectory
from enerdynamics.solvers.single.safediffuser.backends.safediffuser_plan_torch import (
    SafeDiffuserBackendTorch,
)
from enerdynamics.solvers.single.safediffuser.patch.avoiding_9d_adapter import (
    Avoiding9DAdapter,
)

try:
    from enerdynamics.core.registry.solvers import register_solver
except Exception:
    register_solver = None


class SafeDiffuserSolver(SamplingSolver):
    """
    Solver wrapper: delegates SafeDiffuser planning to a torch backend.
    """

    def __init__(
        self,
        dynamics: DynamicsModel,
        energy: EnergyFunctional,
        backend: Backend,
        *,
        env: Any,
        diffusion: Any,
        normalizer: Any,
        plan_config: Dict[str, Any],
        goal_xy: Optional[np.ndarray] = None,
        device: str = "cuda",
        seed: int = 0,
        constraint_manager: Any = None,
        constraint_pipeline: Any = None,
        **kwargs: Any,
    ):
        super().__init__(dynamics, energy, backend, **kwargs)
        self.env = env
        self.diffusion = diffusion
        self.normalizer = normalizer
        self.plan_config = plan_config
        self.goal_xy = goal_xy
        self.device = device
        self.seed = int(seed)
        self.constraint_manager = constraint_manager
        self.constraint_pipeline = constraint_pipeline

        self._backend_impl: SafeDiffuserBackendTorch | None = None

    def _get_backend_impl(self) -> SafeDiffuserBackendTorch:
        if self._backend_impl is None:
            runtime_backend = RuntimeBackendManager.get_backend()
            if runtime_backend.name != "torch":
                raise ValueError(
                    f"SafeDiffuser solver requires torch backend, got '{runtime_backend.name}'."
                )
            self._backend_impl = SafeDiffuserBackendTorch(
                env=self.env,
                diffusion=self.diffusion,
                normalizer=self.normalizer,
                plan_config=self.plan_config,
                device=str(self.device),
                seed=self.seed,
                goal_xy=self.goal_xy,
                constraint_manager=self.constraint_manager,
                constraint_pipeline=self.constraint_pipeline,
            )
        return self._backend_impl

    def sample_trajectories(self, x0: State, horizon: int, n_samples: int, **kwargs: Any) -> List[Trajectory]:
        # SafeDiffuser policy does its own batching; for now, just return n_samples repeats of solve().
        _ = (horizon, n_samples)
        traj = self.solve(x0, horizon=horizon, **kwargs)
        return [traj]

    def _step_env(
        self, action: np.ndarray, obs: np.ndarray, *, t: int | None = None
    ) -> tuple[np.ndarray, bool, bool, dict]:
        """Best-effort env.step wrapper.

        Returns: (next_obs, success, done, info)
        """
        info: dict = {}
        # Try a few common step signatures.
        # Many enerdynamics env wrappers (e.g., D3ILAvoidingEnv) use: step(state, action, t=?, info=?).
        # Gym-style envs use: step(action).
        try:
            # Most specific: include time + info if supported.
            out = self.env.step(obs, action, t=t, info={})
        except TypeError:
            try:
                # Common: step(state, action)
                out = self.env.step(obs, action)
            except TypeError:
                try:
                    # Gym: step(action)
                    out = self.env.step(action)
                except TypeError:
                    # Legacy signature used by DPCC adapter-like wrappers: step(action, obs, fixed_z)
                    try:
                        out = self.env.step(action, obs)
                    except TypeError:
                        fixed_z = getattr(self.env, "fixed_z", None)
                        if fixed_z is None:
                            raise
                        out = self.env.step(action, obs, fixed_z)

        # Parse outputs.
        success = False
        done = False

        if isinstance(out, tuple) or isinstance(out, list):
            if len(out) == 4:
                next_obs, r2, r3, info = out
                info = info or {}
                if isinstance(r2, (bool, np.bool_)) and isinstance(r3, (bool, np.bool_)):
                    success = bool(r2)
                    done = bool(r3)
                else:
                    # (obs, reward, done, info)
                    done = bool(r3)
                    success = bool(info.get("success", info.get("is_success", False)))
            elif len(out) == 5:
                # Gymnasium-style: (obs, reward, terminated, truncated, info)
                next_obs, _rew, terminated, truncated, info = out
                info = info or {}
                done = bool(terminated) or bool(truncated)
                success = bool(info.get("success", info.get("is_success", False)))
            elif len(out) == 2:
                next_obs, done = out
                done = bool(done)
                info = {}
            else:
                next_obs = out[0]
                info = out[-1] if isinstance(out[-1], dict) else {}
                success = bool(info.get("success", info.get("is_success", False)))
                done = bool(info.get("done", False))
        else:
            next_obs = out

        return np.asarray(next_obs, dtype=np.float32), success, done, (info or {})

    def solve(self, x0: State, horizon: int, **kwargs: Any) -> Trajectory:
        # NOTE: `Solver.solve(x0, horizon, ...)` requires `horizon` by interface.
        # We use `plan_config.horizon` (or model horizon) as the authoritative one.
        _ = horizon # type: ignore
        planner = self._get_backend_impl()
        rng_key = kwargs.get("rng_key")

        use_mpc = bool(self.plan_config.get("receding_horizon", True))
        x0_arr = np.asarray(x0, dtype=np.float32).reshape(-1)
        if int(getattr(self.env, "state_dim", x0_arr.size)) >= 9:
            return self._solve_9d(planner, x0_arr, rng_key=rng_key, use_mpc=use_mpc)

        if not use_mpc:
            result = planner.plan(x0_arr, rng_key=rng_key)
            states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]
            actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
            info = dict(result.get("info") or {})
            info = self._attach_4d_lift_to_info(states_list, info)
            return Trajectory(states=states_list, actions=actions_list[:-1], info=info)

        # --- Receding-horizon rollout (SafeDiffuser / MPC style) ---
        max_steps = int(self.plan_config.get("max_episode_length", 200))
        obs = np.asarray(x0, dtype=np.float32)

        states_exec: List[np.ndarray] = [obs.copy()]
        actions_exec: List[np.ndarray] = []

        last_info: Dict[str, Any] = {}
        success = False
        done = False

        for _t in range(max_steps):
            result = planner.plan(obs, rng_key=rng_key)
            last_info = dict(result.get("info") or {})
            a0 = np.asarray(result["actions"][0], dtype=np.float32)

            next_obs, step_success, step_done, step_info = self._step_env(a0, obs, t=_t)
            actions_exec.append(a0.copy())
            obs = next_obs
            states_exec.append(obs.copy())

            success = bool(step_success or success or step_info.get("success", False))
            done = bool(step_done or step_info.get("terminated", False) or step_info.get("done", False))

            if success or done:
                last_info.update({"terminal_info": step_info})
                break

        last_info.update({"mpc": True, "success": success, "steps": len(actions_exec)})
        last_info = self._attach_4d_lift_to_info(states_exec, last_info)

        return Trajectory(states=states_exec, actions=actions_exec, info=last_info)

    def _solve_9d(
        self,
        planner: SafeDiffuserBackendTorch,
        x0_9d: np.ndarray,
        *,
        rng_key: Any | None,
        use_mpc: bool,
    ) -> Trajectory:
        dt = float(getattr(self.env, "dt", self.plan_config.get("dt", 0.035)))
        qdot_limit = float(getattr(self.env, "control_limit", 1.5))
        adapter = Avoiding9DAdapter(
            env=self.env,
            dt=dt,
            qdot_limit=qdot_limit,
            target_xy=self.goal_xy,
        )

        max_steps = int(self.plan_config.get("max_episode_length", 200))
        obs9 = np.asarray(x0_9d, dtype=np.float32).reshape(-1)
        if obs9.size < 9:
            try:
                reset_out = self.env.reset()
                obs9 = np.asarray(reset_out[0] if isinstance(reset_out, (tuple, list)) else reset_out, dtype=np.float32).reshape(-1)
            except Exception as exc:
                raise ValueError(f"9D solve requires 9D initial state; got shape {x0_9d.shape}") from exc
        if obs9.size < 9:
            raise ValueError(f"9D solve requires 9D initial state; got shape {obs9.shape}")

        q_cur = adapter.get_current_q(obs9)
        if q_cur is None:
            q_cur = obs9[2:9].copy()

        states_exec_9d: List[np.ndarray] = [obs9.copy()]
        actions_exec_9d: List[np.ndarray] = []
        states_query_4d: List[np.ndarray] = []
        actions_plan_2d: List[np.ndarray] = []

        last_info: Dict[str, Any] = {}
        success = False
        done = False

        if not use_mpc:
            plan_in = adapter.obs9d_to_obs4d(obs9)
            result = planner.plan(plan_in, rng_key=rng_key)
            last_info = dict(result.get("info") or {})
            planned_actions = [np.asarray(a, dtype=np.float32).reshape(-1)[:2] for a in result.get("actions", [])]

            for t, a2 in enumerate(planned_actions):
                qdot = adapter.delta_xy_to_qdot7(q_cur, a2, dt=dt)
                next_obs9, step_success, step_done, step_info = adapter.step_9d(qdot, obs9, t=t)
                actions_exec_9d.append(qdot.copy())
                obs9 = next_obs9
                states_exec_9d.append(obs9.copy())
                states_query_4d.append(plan_in.copy())
                actions_plan_2d.append(a2.copy())
                q_next = adapter.get_current_q(obs9)
                if q_next is not None:
                    q_cur = q_next

                success = bool(step_success or success or step_info.get("success", False))
                done = bool(step_done or step_info.get("terminated", False) or step_info.get("done", False))
                if success or done:
                    last_info.update({"terminal_info": step_info})
                    break
        else:
            for t in range(max_steps):
                plan_in = adapter.obs9d_to_obs4d(obs9)
                result = planner.plan(plan_in, rng_key=rng_key)
                last_info = dict(result.get("info") or {})
                a2 = np.asarray(result["actions"][0], dtype=np.float32).reshape(-1)[:2]
                qdot = adapter.delta_xy_to_qdot7(q_cur, a2, dt=dt)

                next_obs9, step_success, step_done, step_info = adapter.step_9d(qdot, obs9, t=t)
                actions_exec_9d.append(qdot.copy())
                obs9 = next_obs9
                states_exec_9d.append(obs9.copy())
                states_query_4d.append(plan_in.copy())
                actions_plan_2d.append(a2.copy())
                q_next = adapter.get_current_q(obs9)
                if q_next is not None:
                    q_cur = q_next

                success = bool(step_success or success or step_info.get("success", False))
                done = bool(step_done or step_info.get("terminated", False) or step_info.get("done", False))
                if success or done:
                    last_info.update({"terminal_info": step_info})
                    break

        last_info.update(
            {
                "mpc": bool(use_mpc),
                "success": bool(success),
                "steps": int(len(actions_exec_9d)),
                "state_layout_9d": "[x, y, q1..q7]",
                "action_layout_9d": "[qdot1..qdot7]",
                "states_9d": [np.asarray(s, dtype=np.float32) for s in states_exec_9d],
                "actions_9d": [np.asarray(a, dtype=np.float32) for a in actions_exec_9d],
                "states_query_4d": [np.asarray(s, dtype=np.float32) for s in states_query_4d],
                "actions_plan_2d": [np.asarray(a, dtype=np.float32) for a in actions_plan_2d],
            }
        )

        return Trajectory(
            states=[np.asarray(s, dtype=np.float32) for s in states_exec_9d],
            actions=[np.asarray(a, dtype=np.float32) for a in actions_exec_9d],
            info=last_info,
        )

    def _attach_4d_lift_to_info(self, states_4d: List[np.ndarray], info_in: Dict[str, Any]) -> Dict[str, Any]:
        """
        For 4D runs, generate a DPCC-style post-processed 9D trajectory for visualization.
        """
        info = dict(info_in or {})
        if not bool(self.plan_config.get("export_lifted_9d", True)):
            return info

        if len(states_4d) < 2:
            return info
        s0 = np.asarray(states_4d[0], dtype=np.float32).reshape(-1)
        if s0.size < 4:
            return info

        dt = float(getattr(self.env, "dt", self.plan_config.get("dt", 0.035)))
        qdot_limit = float(self.plan_config.get("lift_qdot_limit", 1.5))
        adapter = Avoiding9DAdapter(
            env=self.env,
            dt=dt,
            qdot_limit=qdot_limit,
            target_xy=self.goal_xy,
        )
        q0 = adapter.get_current_q()
        states_9d, actions_9d = adapter.lift_4d_to_9d_trajectory(
            [np.asarray(s, dtype=np.float32) for s in states_4d],
            initial_q=q0,
            dt=dt,
            xy_indices=(
                int(self.plan_config.get("lift_xy_idx0", 2)),
                int(self.plan_config.get("lift_xy_idx1", 3)),
            ),
        )
        if states_9d is None or actions_9d is None:
            return info

        info["states_9d"] = [np.asarray(s, dtype=np.float32) for s in states_9d]
        info["actions_9d"] = [np.asarray(a, dtype=np.float32) for a in actions_9d]
        info["state_layout_9d"] = "[x, y, q1..q7]"
        info["action_layout_9d"] = "[qdot1..qdot7]"
        return info


if register_solver is not None:
    try:
        register_solver("safediffuser", SafeDiffuserSolver)
    except Exception:
        pass
