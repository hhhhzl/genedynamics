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
        if not use_mpc:
            result = planner.plan(np.asarray(x0, dtype=np.float32), rng_key=rng_key)
            states_list = [np.asarray(s, dtype=np.float32) for s in result["states"]]
            actions_list = [np.asarray(a, dtype=np.float32) for a in result["actions"]]
            return Trajectory(states=states_list, actions=actions_list[:-1], info=result.get("info"))

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

        return Trajectory(states=states_exec, actions=actions_exec, info=last_info)


if register_solver is not None:
    try:
        register_solver("safediffuser", SafeDiffuserSolver)
    except Exception:
        pass
