"""
Sim process: MuJoCo physics loop with shared memory I/O.

Runs as standalone process. Publishes state, consumes actions from plan process.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

_root = Path(__file__).resolve().parents[3]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

if "MUJOCO_GL" not in os.environ:
    import platform
    os.environ.setdefault("MUJOCO_GL", "egl" if platform.system() == "Linux" else "glfw")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sim process (shared memory)")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--ctrl-dt", type=float, default=0.05)
    parser.add_argument("--sim-dt", type=float, default=0.002)
    parser.add_argument("--n-acts", type=int, default=65)
    parser.add_argument("--record", action="store_true")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--sync", action="store_true", default=True)
    args = parser.parse_args()

    import numpy as np
    import mujoco
    import yaml
    from genedynamics.deploy.sim_plan.shm_protocol import create_sim_shm, ShmNames

    with open(args.config) as f:
        config = yaml.safe_load(f) or {}

    model_path = args.model_path or config.get("model_path")
    if not model_path or not Path(model_path).exists():
        from genedynamics.robots.registry import _get_go2_path
        model_path = _get_go2_path() or _get_go2_path(mjx=False)
    if not model_path:
        print("Error: No MuJoCo model path")
        return 1

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    mj_model.opt.timestep = args.sim_dt
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetDataKeyframe(mj_model, mj_data, 0)
    mujoco.mj_forward(mj_model, mj_data)

    nx = mj_model.nq + mj_model.nv
    nu = mj_model.nu
    n_acts = args.n_acts
    ctrl_dt = args.ctrl_dt
    n_frame = max(1, int(ctrl_dt / args.sim_dt))

    shm = create_sim_shm(nx, nu, n_acts)
    default_u = mj_model.keyframe("home").ctrl if hasattr(mj_model, "keyframe") else np.zeros(nu)
    shm["acts_shared"][:] = default_u

    if args.viewer:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(mj_model, mj_data, show_left_ui=False, show_right_ui=True)
    else:
        viewer = None

    t = 0.0
    data_log = []
    try:
        while True:
            if args.sync:
                plan_t = shm["plan_time_shared"][0]
                while t <= plan_t + ctrl_dt:
                    mj_data.ctrl[:] = shm["acts_shared"][0]
                    if args.record:
                        data_log.append(np.concatenate([[t], mj_data.qpos, mj_data.qvel, mj_data.ctrl]))
                    mujoco.mj_step(mj_model, mj_data)
                    t += args.sim_dt
            else:
                mj_data.ctrl[:] = shm["acts_shared"][0]
                if args.record:
                    data_log.append(np.concatenate([[t], mj_data.qpos, mj_data.qvel, mj_data.ctrl]))
                for _ in range(n_frame):
                    mujoco.mj_step(mj_model, mj_data)
                    t += args.sim_dt

            state = np.concatenate([mj_data.qpos, mj_data.qvel]).astype(np.float32)
            shm["time_shared"][0] = t
            shm["state_shared"][:] = state
            if viewer:
                viewer.sync()
            time.sleep(max(0, ctrl_dt - (n_frame * args.sim_dt)) if not args.sync else 0)
    except KeyboardInterrupt:
        pass
    finally:
        if args.record and data_log:
            out_dir = Path(config.get("output_dir", "results/deploy/sim_plan")) / "episodes"
            out_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d-%H%M%S")
            ep_dir = out_dir / f"ep_sim_{ts}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            np.save(ep_dir / "states.npy", np.array([r[1:1+nx] for r in data_log]))
            np.save(ep_dir / "actions.npy", np.array([r[1+nx:1+nx+nu] for r in data_log]))
            print(f"Saved to {ep_dir}")
        for k, v in shm.items():
            if k.endswith("_shm") and hasattr(v, "close"):
                try:
                    v.close()
                    if hasattr(v, "unlink"):
                        v.unlink()
                except Exception:
                    pass
        if viewer:
            viewer.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
