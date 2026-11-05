"""T1 public-observation identifiability for task-adapted active system ID."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from genedynamics.core.artifacts import atomic_json
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.solvers.single.asid import ASIDSolver
from genedynamics.tasks.transfer.controllers.obstacle_feedback import OBSTACLE_BOUNDS
from genedynamics.tasks.transfer.controllers.obstacle_path import PublicObstaclePathPolicy
from genedynamics.tasks.transfer.runtime import MechanismSpec


VERSION = "t1-public-system-identifiability-v1"
ROOT = Path(__file__).resolve().parents[3]


def plan(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("version") != VERSION or config.get("task") != "T1" or config.get("robot") != "xarm7":
        raise ValueError("T1 xArm7 public identifiability config required")
    if (config.get("seed") != 3817250 or config.get("modes") != [0, 1]
            or config.get("speed_scales") != [.8, 1., 1.15]
            or config.get("horizons") != [20, 60]
            or config.get("delta") != .1
            or config.get("source_mechanism") != {"servo_gain": .85, "payload_mass": .65, "delay_steps": 0}
            or config.get("target_mechanism") != {"servo_gain": .7, "payload_mass": .5, "delay_steps": 0}
            or config.get("gain_grid") != [.6, .7, .8, .9, 1.]
            or config.get("payload_grid") != [.3, .4, .5, .6, .7]
            or config.get("maximum_simulator_episodes") != 80):
        raise ValueError("public identifiability sweep changed")
    path = np.asarray(config.get("public_path"), dtype=np.float64)
    if path.shape != (4,) or not np.isfinite(path).all():
        raise ValueError("finite four-component T1 public path required")
    if (config.get("source_commit") != "cf2cb670fb983f67cf2af156d4e17b4a690787e2"
            or not (ROOT / "baselines/ASID/README.md").is_file()):
        raise ValueError("pinned ASID source checkout is missing or changed")
    output = ROOT / config["output"]
    if output.exists():
        raise FileExistsError(f"identifiability output exists: {output}")
    return {"version": VERSION, "status": "planned_unexecuted", "scope":
        "CPU-only simulator identifiability, not an ASID task policy or target performance result",
        "task": "T1", "robot": "xarm7", "source_commit": config["source_commit"],
        "seed": config["seed"], "modes": config["modes"],
        "speed_scales": config["speed_scales"], "horizons": config["horizons"],
        "maximum_simulator_episodes": 80, "output": config["output"]}


def fit_grid(target_observation: np.ndarray, candidates: dict[tuple[float, float], np.ndarray]) -> list[dict[str, Any]]:
    """Rank simulated hypotheses using only the observed public TCP state."""
    target = np.asarray(target_observation, dtype=np.float64)
    if target.shape != (2, 6) or not np.isfinite(target).all():
        raise ValueError("two finite public TCP observations required")
    ranked = []
    for theta, observation in candidates.items():
        value = np.asarray(observation, dtype=np.float64)
        if value.shape != target.shape or not np.isfinite(value).all():
            raise ValueError("candidate public observation differs")
        position_mm = np.linalg.norm(value[:, :3] - target[:, :3], axis=1) * 1000
        velocity_mmps = np.linalg.norm(value[:, 3:] - target[:, 3:], axis=1) * 1000
        ranked.append({"servo_gain": theta[0], "payload_mass": theta[1],
            "position_error_mm": position_mm.tolist(),
            "velocity_error_mmps": velocity_mmps.tolist(),
            "ranking_score": float(np.linalg.norm(position_mm) + .02 * np.linalg.norm(velocity_mmps))})
    return sorted(ranked, key=lambda row: (row["ranking_score"], row["servo_gain"], row["payload_mass"]))


def run(config: dict[str, Any]) -> dict[str, Any]:
    frozen = plan(config)
    output = ROOT / config["output"]
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "recipe.json", config)
    atomic_json(output / "plan.json", frozen)
    seed = int(config["seed"])
    path = np.asarray(config["public_path"], dtype=np.float64)
    calls = 0
    controls = 0
    started = time.monotonic()

    def simulator(theta: tuple[float, float], mode: int, actions: np.ndarray,
                  steps: int) -> np.ndarray:
        nonlocal calls, controls
        if calls >= frozen["maximum_simulator_episodes"]:
            raise RuntimeError("frozen ASID diagnostic simulator budget exceeded")
        task = instantiate_task("T1", robot="xarm7", mechanism_spec=MechanismSpec(
            servo_gain=float(theta[0]), payload_mass=float(theta[1]), delay_steps=0), seed=seed)
        public = task.reset((0,), seed=seed, modes=np.asarray([mode]))
        observations = []
        calls += 1
        for tick, action in enumerate(actions[:steps]):
            public = task.step(action[None]).observation
            controls += 1
            if tick in (19, 59):
                observations.append(np.r_[public["tcp_position"][0], public["tcp_twist"][0, :3]])
        return np.asarray(observations)

    def source_actions(mode: int, speed: float) -> np.ndarray:
        nonlocal calls, controls
        if calls >= frozen["maximum_simulator_episodes"]:
            raise RuntimeError("frozen ASID diagnostic simulator budget exceeded")
        mechanism = MechanismSpec(**config["source_mechanism"])
        task = instantiate_task("T1", robot="xarm7", mechanism_spec=mechanism, seed=seed)
        public = task.reset((0,), seed=seed, modes=np.asarray([mode]))
        actor = PublicObstaclePathPolicy(task.spec, modes=[mode], path=path)
        actions = []
        calls += 1
        for _ in range(60):
            action = actor.act(public)
            action[:, :3] = np.clip(action[:, :3] * speed,
                                   -OBSTACLE_BOUNDS[:3], OBSTACLE_BOUNDS[:3])
            actions.append(action[0].copy())
            public = task.step(action).observation
            controls += 1
        return np.asarray(actions)

    report: dict[str, Any] = {"version": VERSION, "status": "running", "plan": frozen}
    try:
        source = tuple(config["source_mechanism"][key] for key in ("servo_gain", "payload_mass"))
        bank = {(mode, speed): source_actions(mode, speed)
                for mode in config["modes"] for speed in config["speed_scales"]}

        def transition(state, action, theta):
            mode, horizon = state
            return simulator(tuple(theta), mode, action, horizon)[-1]

        solver = ASIDSolver(transition, delta=config["delta"], normalization=1.)
        fisher_rows = []
        for mode in config["modes"]:
            for speed in config["speed_scales"]:
                for horizon in config["horizons"]:
                    jacobian = solver.observation_jacobian(
                        (mode, horizon), bank[(mode, speed)], np.asarray(source))
                    singular = np.linalg.svd(jacobian, compute_uv=False)
                    fisher_rows.append({"mode": mode, "speed_scale": speed,
                        "horizon_controls": horizon, "jacobian": jacobian.tolist(),
                        "singular_values": singular.tolist(),
                        "column_norms": np.linalg.norm(jacobian, axis=0).tolist(),
                        "fisher_trace": float(np.sum(jacobian * jacobian))})
        target_theta = tuple(config["target_mechanism"][key] for key in ("servo_gain", "payload_mass"))
        action = bank[(0, 1.)]
        target = simulator(target_theta, 0, action, 60)
        candidates = {(gain, mass): simulator((gain, mass), 0, action, 60)
                      for gain in config["gain_grid"] for mass in config["payload_grid"]}
        ranked = fit_grid(target, candidates)
        if calls != 80:
            raise RuntimeError(f"ASID diagnostic executed {calls}, expected 80 simulator episodes")
        report.update({"status": "complete_development_diagnostic",
            "fisher_rows": fisher_rows, "grid_fit": ranked,
            "target_public_observation": target.tolist(),
            "source_action_sha256": {f"m{mode}_s{speed}": hashlib.sha256(actions.tobytes()).hexdigest()
                for (mode, speed), actions in bank.items()},
            "simulator_episodes": calls, "simulator_control_steps": controls,
            "wall_seconds": time.monotonic() - started,
            "formal_performance_evidence": False})
    except BaseException as exc:
        report.update({"status": "failed_retained", "error": str(exc),
            "simulator_episodes": calls, "simulator_control_steps": controls,
            "wall_seconds": time.monotonic() - started})
        raise
    finally:
        atomic_json(output / "report.json", report)
    return report
