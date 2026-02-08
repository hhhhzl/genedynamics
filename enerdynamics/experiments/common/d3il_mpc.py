from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import time


@dataclass(frozen=True)
class MPCResult:
    states: List[np.ndarray]
    actions: List[np.ndarray]
    costs: List[float]
    infos: List[Dict[str, Any]]
    planned_first_actions: List[np.ndarray]


def _is_jax_key(rng: Any) -> bool:
    try:
        import jax
        import jax.numpy as jnp

        return isinstance(rng, jax.Array) or isinstance(rng, jnp.ndarray)
    except Exception:
        return False


def _split_rng(rng: Any, n: int) -> Tuple[Any, List[Any]]:
    """
    Split rng into n subkeys if it is a JAX PRNGKey; otherwise return rng as-is.
    """
    if not _is_jax_key(rng):
        return rng, [rng for _ in range(n)]
    import jax

    keys = jax.random.split(rng, n + 1)
    return keys[0], [keys[i + 1] for i in range(n)]


def _tracking_action(
    exec_env: Any,
    current_state: np.ndarray,
    planned_state_next: np.ndarray,
    planned_action_t: np.ndarray,
    dt: float,
    control_limit: float,
    tracker_k_joint: float = 0.0,
    tracker_k_xy: float = 1.0,
    tracker_ff_alpha: float = 0.0,
) -> np.ndarray:
    """
    Compute action that tracks the planned trajectory (task-space xy and optionally joints).

    Used for plan_once + track_trajectory: instead of u = planned_action_t, we compute
    u such that tcp_xy moves toward planned_state_next[:2] and (if tracker_k_joint > 0)
    joints q move toward planned_state_next[2:9]. tracker_k_xy scales task-space velocity;
    tracker_ff_alpha blends in planned_action_t (feed-forward). Requires exec_env with
    get_jacobian_xy and 9D state; otherwise returns planned_action_t (open-loop).
    """
    current_state = np.asarray(current_state, dtype=np.float32).reshape(-1)
    planned_state_next = np.asarray(planned_state_next, dtype=np.float32).reshape(-1)
    planned_action_t = np.asarray(planned_action_t, dtype=np.float32).reshape(-1)
    if current_state.size != 9 or planned_state_next.size < 2:
        return planned_action_t
    if not hasattr(exec_env, "get_jacobian_xy"):
        return planned_action_t
    J_xy = exec_env.get_jacobian_xy(current_state)
    if J_xy is None:
        return planned_action_t
    J_xy = np.asarray(J_xy, dtype=np.float64)
    if J_xy.shape != (2, 7):
        return planned_action_t
    k_xy = max(0.0, float(tracker_k_xy))
    v_xy_des = k_xy * (planned_state_next[:2].astype(np.float64) - current_state[:2].astype(np.float64)) / max(dt, 1e-8)
    # u = J_xy^+ @ v_xy_des (min-norm solution to J_xy @ u ≈ v_xy_des)
    u_track, _res, _rank, _s = np.linalg.lstsq(J_xy, v_xy_des, rcond=None)
    u_track = np.asarray(u_track, dtype=np.float32)
    if u_track.size != 7:
        return planned_action_t
    # Joint-space tracking: pull q toward planned q (reduces drift, keeps J(q) aligned)
    if tracker_k_joint > 0 and planned_state_next.size >= 9:
        q_err = np.asarray(planned_state_next[2:9], dtype=np.float32) - np.asarray(current_state[2:9], dtype=np.float32)
        u_track = u_track + tracker_k_joint * q_err
    u = np.clip(u_track, -float(control_limit), float(control_limit))
    if tracker_ff_alpha > 0 and planned_action_t.size >= 7:
        alpha = np.clip(float(tracker_ff_alpha), 0.0, 1.0)
        u = (1.0 - alpha) * u + alpha * np.asarray(planned_action_t[:7], dtype=np.float32)
        u = np.clip(u, -float(control_limit), float(control_limit))
    return u


def run_mpc_episode(
    *,
    exec_env: Any,
    planner: Any,
    plan_step_fn: Any,
    rng: Any,
    max_steps: int,
) -> Dict[str, Any]:
    """
    Run receding-horizon MPC on a stateful execution env.

    Args:
        exec_env: D3ILAvoidingEnv (stateful MuJoCo env wrapper)
        planner: EDOCPlanner or MBDSolver (operates on a *planning* env)
        plan_step_fn: callable(planner, x0, rng_key) -> (u0, plan_info_dict)
        rng: RNG object passed from ExperimentRunner (typically a JAX PRNGKey)
        max_steps: max rollout length

    Returns:
        result dict compatible with ExperimentRunner trajectory extraction:
        - states: list[np.ndarray] (executed states, length T+1)
        - actions: list[np.ndarray] (executed actions, length T)
        - costs: list[float] (executed costs, length T)
        - infos: list[dict] (per-step info)
        - planned_first_actions: list[np.ndarray] (u0 from planner each step)
    """
    x, info0 = exec_env.reset()

    states: List[np.ndarray] = [np.asarray(x, dtype=np.float32)]
    actions: List[np.ndarray] = []
    costs: List[float] = []
    infos: List[Dict[str, Any]] = [info0]
    planned_u0s: List[np.ndarray] = []
    planning_time_per_step: List[float] = []

    rng, subkeys = _split_rng(rng, max_steps)

    done = False
    for t in range(max_steps):
        if done:
            break

        x0 = np.asarray(states[-1], dtype=np.float32)
        t0 = time.time()
        u0, plan_info = plan_step_fn(planner, x0, subkeys[t])
        planning_time_per_step.append(float(time.time() - t0))
        u0 = np.asarray(u0, dtype=np.float32).reshape(-1)
        planned_u0s.append(u0.copy())

        x_next, cost, done, step_info = exec_env.step(None, u0, t=t, info={})

        states.append(np.asarray(x_next, dtype=np.float32))
        actions.append(u0.copy())
        costs.append(float(cost) if cost is not None else 0.0)
        infos.append({**step_info, "plan_info": plan_info})

    # Episode-level summary signals
    success = any(bool(i.get("success", False)) for i in infos if isinstance(i, dict))
    collision = any(bool(i.get("collision", False)) for i in infos if isinstance(i, dict))

    return {
        # Keep "states/actions" for ExperimentRunner trajectory extraction
        "states": states,
        "actions": actions,
        # Also store explicit exec_* fields for unified downstream consumption
        "exec_states": states,
        "exec_actions": actions,
        "costs": costs,
        "infos": infos,
        "planned_first_actions": planned_u0s,
        "planning_time_per_step": planning_time_per_step,
        "success": bool(success),
        "collision": bool(collision),
        "initial_state": states[0],
        "done": bool(done),
        "steps": int(len(actions)),
    }


def run_open_loop_episode(
    *,
    exec_env: Any,
    actions: List[np.ndarray],
    rng: Any = None,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Execute a precomputed action sequence on the execution env (plan-once, open-loop).

    All solvers can be used: plan once to get a full trajectory, then pass
    result["actions"] here to roll out on the real env.

    Args:
        exec_env: Execution env (e.g. D3ILAvoidingEnv or D3ILAvoiding7dVelEnv).
        actions: List of action arrays to apply (length T).
        rng: Optional RNG for exec_env.reset(rng).
        max_steps: Max steps to run. If None, uses len(actions).

    Returns:
        Same dict shape as run_mpc_episode: states, actions, costs, infos, etc.
    """
    x, info0 = exec_env.reset(rng=rng)
    T = len(actions) if max_steps is None else min(len(actions), max_steps)

    states: List[np.ndarray] = [np.asarray(x, dtype=np.float32)]
    executed_actions: List[np.ndarray] = []
    costs: List[float] = []
    infos: List[Dict[str, Any]] = [info0]

    done = False
    for t in range(T):
        if done:
            break
        u = np.asarray(actions[t], dtype=np.float32).reshape(-1)
        x_next, cost, done, step_info = exec_env.step(None, u, t=t, info={})
        states.append(np.asarray(x_next, dtype=np.float32))
        executed_actions.append(u.copy())
        costs.append(float(cost) if cost is not None else 0.0)
        infos.append(step_info)

    success = any(bool(i.get("success", False)) for i in infos if isinstance(i, dict))
    collision = any(bool(i.get("collision", False)) for i in infos if isinstance(i, dict))

    return {
        "states": states,
        "actions": executed_actions,
        "exec_states": states,
        "exec_actions": executed_actions,
        "costs": costs,
        "infos": infos,
        "planned_first_actions": executed_actions,
        "success": bool(success),
        "collision": bool(collision),
        "initial_state": states[0],
        "done": bool(done),
        "steps": len(executed_actions),
    }


def run_plan_once_episode(
    *,
    exec_env: Any,
    planner: Any,
    plan_once_fn: Any,
    rng: Any,
    max_steps: Optional[int] = None,
    continue_after_done: bool = False,
    replan_every: Optional[int] = None,
    track_trajectory: bool = False,
    tracker_k_joint: float = 0.0,
    tracker_k_xy: float = 1.0,
    tracker_ff_alpha: float = 0.0,
) -> Dict[str, Any]:
    """
    Plan once (or every replan_every steps) then execute on the execution env.

    If replan_every is None: one plan for full horizon, execute open-loop (or track).
    If replan_every = N > 0: every N steps re-plan from current state with horizon N (chunked execution).

    plan_once_fn(planner, x0, rng, horizon=None) should return dict with "actions" (list)
    and optionally "states" (list, length horizon+1) for track_trajectory.

    When track_trajectory is True and plan returns "states", each step uses a tracking
    action (J_xy^+ @ v_xy_des toward next planned state) instead of open-loop planned
    action, when exec_env has get_jacobian_xy (9D). Falls back to open-loop otherwise.
    Works with any model-based solver (MBD, EBMBD, MDOC, CFS-MBD).
    """
    x0, info0 = exec_env.reset(rng=rng)
    states: List[np.ndarray] = [np.asarray(x0, dtype=np.float32)]
    executed_actions: List[np.ndarray] = []
    costs: List[float] = []
    infos: List[Dict[str, Any]] = [info0]
    done = False
    rng_cur = rng
    total_steps = int(max_steps) if max_steps is not None else 999999
    dt = getattr(exec_env, "dt", 0.035)
    control_limit = getattr(exec_env, "control_limit", 1.5)
    k_joint = float(tracker_k_joint)
    k_xy = float(tracker_k_xy)
    ff_alpha = float(tracker_ff_alpha)

    if replan_every is None or replan_every <= 0:
        # True plan-once: one plan, execute all
        result = plan_once_fn(planner, np.asarray(x0, dtype=np.float32), rng_cur)
        actions = result.get("actions") or result.get("action") or []
        if not actions:
            raise RuntimeError("plan_once_fn returned no actions")
        _ps = result.get("states")
        planned_states = [] if _ps is None else (_ps if isinstance(_ps, list) else list(_ps))
        T = min(len(actions), total_steps)
        for t in range(T):
            if done and not continue_after_done:
                break
            if track_trajectory and len(planned_states) > 0:
                # planned_states[0]=s0, ..., planned_states[T]=sT; at step t we want to go to planned_states[t+1]
                idx_next = min(t + 1, len(planned_states) - 1)
                planned_next = planned_states[idx_next] if idx_next >= 0 else planned_states[0]
                u = _tracking_action(
                    exec_env,
                    states[-1],
                    planned_next,
                    np.asarray(actions[t], dtype=np.float32),
                    dt,
                    control_limit,
                    tracker_k_joint=k_joint,
                    tracker_k_xy=k_xy,
                    tracker_ff_alpha=ff_alpha,
                )
            else:
                u = np.asarray(actions[t], dtype=np.float32).reshape(-1)
            x_next, cost, done, step_info = exec_env.step(None, u, t=t, info={})
            states.append(np.asarray(x_next, dtype=np.float32))
            executed_actions.append(u.copy())
            costs.append(float(cost) if cost is not None else 0.0)
            infos.append(step_info)
    else:
        # Chunked: replan every replan_every steps with horizon = chunk size
        step_count = 0
        while step_count < total_steps:
            chunk = min(replan_every, total_steps - step_count)
            if chunk <= 0:
                break
            current_x = states[-1]
            result = plan_once_fn(planner, np.asarray(current_x, dtype=np.float32), rng_cur, horizon=chunk)
            actions = result.get("actions") or result.get("action") or []
            if not actions:
                break
            _psc = result.get("states")
            planned_states_chunk = [] if _psc is None else (_psc if isinstance(_psc, list) else list(_psc))
            take = min(chunk, len(actions))
            for i in range(take):
                if done and not continue_after_done:
                    break
                if track_trajectory and len(planned_states_chunk) > 0:
                    idx_next = min(i + 1, len(planned_states_chunk) - 1)
                    planned_next = planned_states_chunk[idx_next] if idx_next >= 0 else planned_states_chunk[0]
                    u = _tracking_action(
                        exec_env,
                        states[-1],
                        planned_next,
                        np.asarray(actions[i], dtype=np.float32),
                        dt,
                        control_limit,
                        tracker_k_joint=k_joint,
                        tracker_k_xy=k_xy,
                        tracker_ff_alpha=ff_alpha,
                    )
                else:
                    u = np.asarray(actions[i], dtype=np.float32).reshape(-1)
                x_next, cost, done, step_info = exec_env.step(None, u, t=step_count, info={})
                states.append(np.asarray(x_next, dtype=np.float32))
                executed_actions.append(u.copy())
                costs.append(float(cost) if cost is not None else 0.0)
                infos.append(step_info)
                step_count += 1
            if take < chunk or (done and not continue_after_done):
                break
            # Advance rng for next chunk (simple split if has attr)
            if hasattr(rng_cur, "split"):
                rng_cur, _ = rng_cur.split(2)

    success = any(bool(i.get("success", False)) for i in infos if isinstance(i, dict))
    collision = any(bool(i.get("collision", False)) for i in infos if isinstance(i, dict))
    out = {
        "states": states,
        "actions": executed_actions,
        "exec_states": states,
        "exec_actions": executed_actions,
        "costs": costs,
        "infos": infos,
        "planned_first_actions": executed_actions,
        "success": bool(success),
        "collision": bool(collision),
        "initial_state": states[0],
        "done": bool(done),
        "steps": len(executed_actions),
    }
    for k in ("reward_history", "diffusion_actions_traj", "diffusion_sampled_actions",
              "candidate_states", "candidate_actions", "candidate_costs", "best_idx"):
        if k in result and result[k] is not None:
            out[k] = result[k]
    if "initial_state" in result and result.get("initial_state") is not None and result.get("initial_state") is not states[0]:
        out["plan_initial_state"] = result["initial_state"]
    return out


def make_plan_once_fn(
    plan_env: Any,
    solver_key: str = "solver",
    horizon_attr: str = "horizon",
    rng_key_name: str = "rng_key",
) -> Any:
    """
    Build a generic plan-once function for use with run_plan_once_episode.

    Returns callable(planner, x0, rng) -> {"actions": [...], "states": [...]}.
    """

    def plan_once_fn(planner: Any, x0: np.ndarray, rng_key: Any, horizon: Optional[int] = None) -> Dict[str, Any]:
        solver = planner[solver_key] if isinstance(planner, dict) else planner
        H = horizon if horizon is not None else getattr(solver, horizon_attr, getattr(plan_env, "horizon", 20))
        if hasattr(plan_env, "set_initial_state"):
            plan_env.set_initial_state(x0)
        if hasattr(plan_env, "set_linearization") and isinstance(planner, dict):
            exec_env = planner.get("exec_env")
            if exec_env is not None and hasattr(exec_env, "get_jacobian_xy"):
                J_xy = exec_env.get_jacobian_xy(np.asarray(x0, dtype=np.float32))
                if J_xy is not None:
                    plan_env.set_linearization(J_xy)
        kwargs: Dict[str, Any] = {"horizon": H}
        if rng_key is not None:
            kwargs[rng_key_name] = rng_key
        traj = solver.solve(x0, **kwargs)
        actions = getattr(traj, "actions", None) or []
        states = getattr(traj, "states", None) or []
        info = getattr(traj, "info", None) or {}
        out = {"actions": list(actions), "states": list(states)}
        for k in ("reward_history", "diffusion_actions_traj", "diffusion_sampled_actions", "initial_state",
                  "candidate_states", "candidate_actions", "candidate_costs", "best_idx"):
            if k in info and info[k] is not None:
                out[k] = info[k]
        return out

    return plan_once_fn


def make_mpc_plan_step_fn(
    plan_env: Any,
    solver_key: str = "solver",
    horizon_attr: str = "horizon",
    rng_key_name: str = "rng_key",
) -> Any:
    """
    Build a generic MPC plan_step_fn so any solver can plug into run_mpc_episode.

    The planner is expected to be a dict (e.g. from method plugin create_planner)
    with at least planner[solver_key] = solver. The solver must implement
    solve(x0, horizon=..., rng_key=...) and return an object with .actions (list).

    Usage:
        plan_step_fn = make_mpc_plan_step_fn(plan_env)
        result = run_mpc_episode(exec_env=env, planner=planner, plan_step_fn=plan_step_fn, rng=rng, max_steps=150)
    """

    def plan_step_fn(planner: Any, x0: np.ndarray, rng_key: Any) -> Tuple[np.ndarray, Dict[str, Any]]:
        solver = planner[solver_key] if isinstance(planner, dict) else planner
        horizon = getattr(solver, horizon_attr, getattr(plan_env, "horizon", 20))
        if hasattr(plan_env, "set_initial_state"):
            plan_env.set_initial_state(x0)
        if hasattr(plan_env, "set_linearization") and isinstance(planner, dict):
            exec_env = planner.get("exec_env")
            if exec_env is not None and hasattr(exec_env, "get_jacobian_xy"):
                J_xy = exec_env.get_jacobian_xy(np.asarray(x0, dtype=np.float32))
                if J_xy is not None:
                    plan_env.set_linearization(J_xy)
        kwargs: Dict[str, Any] = {"horizon": horizon}
        if rng_key is not None:
            kwargs[rng_key_name] = rng_key
        traj = solver.solve(x0, **kwargs)
        actions = getattr(traj, "actions", None) or (traj if isinstance(traj, (list, tuple)) else [])
        if not actions:
            raise RuntimeError("Solver returned no actions")
        u0 = np.asarray(actions[0], dtype=np.float32)
        return u0, {"traj_len": len(actions)}

    return plan_step_fn


def run_d3il_unified(
    planner: Any,
    start_pos: np.ndarray,
    rng: Any,
) -> Dict[str, Any]:
    """
    Unified D3IL execution: MPC or plan-once based on planner["config"]["execution"].

    planner must be a dict with keys: exec_env, plan_env, solver, config.
    config should have execution: "mpc" | "plan_once" and max_episode_length (optional).
    All solvers (MBD, EBMBD, MDOC, CFS-MBD) use this same path.
    """
    exec_env = planner["exec_env"]
    plan_env = planner["plan_env"]
    config = planner["config"]
    mode = config.get("execution", "mpc")
    max_steps = int(config.get("max_episode_length", getattr(exec_env, "horizon", 150)))

    if mode == "plan_once":
        plan_once_fn = make_plan_once_fn(plan_env)
        replan_every = config.get("replan_every")
        if isinstance(replan_every, (int, float)):
            replan_every = int(replan_every)
        else:
            replan_every = None
        track_trajectory = bool(config.get("track_trajectory", False))
        tracker_k_joint = float(config.get("tracker_k_joint", 0.0))
        tracker_k_xy = float(config.get("tracker_k_xy", 1.0))
        tracker_ff_alpha = float(config.get("tracker_ff_alpha", 0.0))
        return run_plan_once_episode(
            exec_env=exec_env,
            planner=planner,
            plan_once_fn=plan_once_fn,
            rng=rng,
            max_steps=max_steps,
            continue_after_done=bool(config.get("continue_after_done", False)),
            replan_every=replan_every,
            track_trajectory=track_trajectory,
            tracker_k_joint=tracker_k_joint,
            tracker_k_xy=tracker_k_xy,
            tracker_ff_alpha=tracker_ff_alpha,
        )
    plan_step_fn = make_mpc_plan_step_fn(plan_env)
    return run_mpc_episode(
        exec_env=exec_env,
        planner=planner,
        plan_step_fn=plan_step_fn,
        rng=rng,
        max_steps=max_steps,
    )


