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
) -> Dict[str, Any]:
    """
    Plan once (full trajectory) then execute open-loop on the execution env.

    All solvers can be used: plan_once_fn(planner, x0, rng) should return
    a dict with "actions" (list of arrays). Optional "states" is ignored for execution.

    Args:
        exec_env: Execution env.
        planner: Planner dict or solver (passed to plan_once_fn).
        plan_once_fn: callable(planner, x0, rng) -> dict with "actions" (list).
        rng: RNG for reset and for plan_once_fn.
        max_steps: Cap execution steps. If None, use len(actions).

    Returns:
        Same dict shape as run_mpc_episode.
    """
    x0, info0 = exec_env.reset(rng=rng)
    result = plan_once_fn(planner, np.asarray(x0, dtype=np.float32), rng)
    actions = result.get("actions") or result.get("action") or []
    if not actions:
        raise RuntimeError("plan_once_fn returned no actions")
    T = len(actions) if max_steps is None else min(len(actions), max_steps)

    states: List[np.ndarray] = [np.asarray(x0, dtype=np.float32)]
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

    def plan_once_fn(planner: Any, x0: np.ndarray, rng_key: Any) -> Dict[str, Any]:
        solver = planner[solver_key] if isinstance(planner, dict) else planner
        horizon = getattr(solver, horizon_attr, getattr(plan_env, "horizon", 20))
        if hasattr(plan_env, "set_initial_state"):
            plan_env.set_initial_state(x0)
        kwargs: Dict[str, Any] = {"horizon": horizon}
        if rng_key is not None:
            kwargs[rng_key_name] = rng_key
        traj = solver.solve(x0, **kwargs)
        actions = getattr(traj, "actions", None) or []
        states = getattr(traj, "states", None) or []
        return {"actions": list(actions), "states": list(states)}

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
        return run_plan_once_episode(
            exec_env=exec_env,
            planner=planner,
            plan_once_fn=plan_once_fn,
            rng=rng,
            max_steps=max_steps,
        )
    plan_step_fn = make_mpc_plan_step_fn(plan_env)
    return run_mpc_episode(
        exec_env=exec_env,
        planner=planner,
        plan_step_fn=plan_step_fn,
        rng=rng,
        max_steps=max_steps,
    )


