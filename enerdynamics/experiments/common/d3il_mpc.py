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
        costs.append(float(cost))
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


