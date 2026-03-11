#!/usr/bin/env python3
"""
Debug SoftZoo reward: trace reward flow and check per-step values.
"""

from __future__ import annotations

import sys
from pathlib import Path

# macOS: ossaudiodev is Linux-only; stub for SoftZoo deps
if "ossaudiodev" not in sys.modules:
    import types
    _stub = types.ModuleType("ossaudiodev")
    _stub.SNDCTL_COPR_SENDMSG = 0
    sys.modules["ossaudiodev"] = _stub

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    from genedynamics.envs.external.softzoo.bootstrap import ensure_softzoo_on_path
    from genedynamics.envs.external.softzoo.adapters import (
        make_softzoo_env,
        encode_morphology,
        encode_controller,
    )
    from genedynamics.envs.external.softzoo.task_registry import get_task_spec
    import numpy as np

    ensure_softzoo_on_path(str(ROOT))
    task_spec = get_task_spec("crawling_ground")

    # Same params as verify_softzoo
    x = np.array([1.0, 1.0, 1.0])
    phi = np.ones(20) * 10.0
    mode_id = 0
    fidelity_level = 0

    from genedynamics.envs.external.softzoo.config import SoftZooRuntimeConfig

    runtime = SoftZooRuntimeConfig(project_root=str(ROOT), use_renderer=False)
    env = make_softzoo_env(
        task_spec=task_spec,
        mode_spec=task_spec.modes[mode_id] if task_spec.modes else None,
        fidelity_spec=task_spec.fidelity_levels[fidelity_level],
        runtime_config=runtime,
    )
    controller = encode_controller(phi, task_spec, env)
    design = encode_morphology(x, task_spec, env=env)

    print("=== Debug SoftZoo Reward ===\n")
    print(f"design_space: {env.cfg.ENVIRONMENT.design_space}")
    print(f"design_space_config.base_shape: {getattr(env.cfg.ENVIRONMENT.design_space_config, 'base_shape', 'N/A')}")
    print(f"PCD path: {getattr(env, '_softzoo_pcd_path', 'N/A')}")
    print(f"design: {design}")
    print(f"n_actuators: {env.design_space.n_actuators}")
    print(f"max_steps: {task_spec.max_steps}")
    print()

    np.random.seed(0)
    obs = env.reset(design)
    controller.reset()

    rewards = []
    for step in range(min(20, task_spec.max_steps)):
        act = controller(env.sim.solver.current_s, obs)
        obs, reward, done, info = env.step(act)
        r = float(reward) if hasattr(reward, "item") else float(reward)
        rewards.append(r)
        if step < 5 or r != 0:
            print(f"  step {step}: reward={r:.6f}, done={done}")
        if done:
            break

    total = sum(rewards)
    print(f"\nTotal reward (first 20 steps): {total:.6f}")
    print(f"Non-zero steps: {sum(1 for r in rewards if r != 0)}")
    print(f"Reward range: [{min(rewards):.6f}, {max(rewards):.6f}]")

    env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
