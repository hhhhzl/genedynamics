"""
Plan process: reads state from shared memory, runs planner, writes actions.

Runs as standalone process. Connects to sim process via shared memory.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def main() -> int:
    parser = argparse.ArgumentParser(description="Plan process (shared memory)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--nx", type=int, default=37)
    parser.add_argument("--nu", type=int, default=12)
    parser.add_argument("--n-acts", type=int, default=65)
    parser.add_argument("--horizon", type=int, default=64)
    parser.add_argument("--ctrl-dt", type=float, default=0.05)
    args = parser.parse_args()

    import numpy as np
    import yaml
    from genedynamics.deploy.sim_plan.shm_protocol import attach_plan_shm
    from genedynamics.deploy.pipeline import run_pipeline
    from genedynamics.deploy.config import DeployConfig

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    cfg = DeployConfig.from_dict(config) if isinstance(config, dict) else DeployConfig.from_yaml(args.config)
    cfg = cfg if hasattr(cfg, "robot_type") else DeployConfig.from_dict(config)

    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.deploy.profiles import get_profile_registry

    RuntimeBackendManager.set_backend("jax", device="cpu")
    profile = get_profile_registry().require(cfg.robot_type, cfg.model_id)
    use_mjx = profile.requires_mjx(cfg.planner)
    config_dict = {
        "robot_type": cfg.robot_type,
        "model_id": cfg.model_id,
        "planner": cfg.planner,
        "horizon": args.horizon,
        "env_params": dict(cfg.env_params),
        "method_params": dict(cfg.method_params),
    }
    config_dict["env_params"]["model"] = cfg.model_id
    config_dict["env_params"]["use_mjx"] = use_mjx

    env = profile.make_env(config_dict)
    planner = profile.make_planner(env, config_dict)

    nx = env.state_dim if hasattr(env, "state_dim") else (getattr(env, "nq", 19) + getattr(env, "nv", 18))
    nu = getattr(env, "act_dim", 12)
    n_acts = args.n_acts

    shm = attach_plan_shm(nx, nu, n_acts)
    default_u = np.zeros(nu, dtype=np.float32)
    try:
        if hasattr(env, "sys") and hasattr(env.sys, "mj_model"):
            default_u = np.array(env.sys.mj_model.keyframe("home").ctrl, dtype=np.float32)
        elif hasattr(env, "_model") and hasattr(env._model, "keyframe"):
            default_u = np.array(env._model.keyframe("home").ctrl, dtype=np.float32)
    except Exception:
        pass
    shm["acts_shared"][:] = default_u

    from genedynamics.execution.bridges.planner_bridge import PlannerBridge
    from genedynamics.execution.core.contracts import RobotState

    bridge = PlannerBridge(planner, horizon=args.horizon, plan_mode="plan_once")
    ctrl_dt = args.ctrl_dt
    rng = RuntimeBackendManager.get_backend().create_rng(cfg.seed)

    last_plan_t = shm["time_shared"][0]
    step = 0
    try:
        while True:
            plan_t = shm["time_shared"][0]
            state_flat = np.array(shm["state_shared"], dtype=np.float32)
            nq = getattr(env, "nq", nx // 2)
            nv = nx - nq
            state = RobotState.from_flat(state_flat, nq, timestamp=plan_t, source="shm")

            if plan_t - last_plan_t >= ctrl_dt - 1e-4:
                packet = bridge.plan_once(state, {"rng": rng, "horizon": args.horizon})
                for i, act in enumerate(packet.actions[: n_acts]):
                    if i < len(packet.actions):
                        shm["acts_shared"][i] = np.asarray(packet.actions[i], dtype=np.float32)
                shm["plan_time_shared"][0] = plan_t
                last_plan_t = plan_t
                step += 1

            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        for k, v in shm.items():
            if k.endswith("_shm") and hasattr(v, "close"):
                try:
                    v.close()
                except Exception:
                    pass
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
