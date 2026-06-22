#!/usr/bin/env python
"""Re-plan a 2GO corridor trajectory for an AR / measured obstacle scene (M2).

Takes a scene (from the M1 ``SceneSource`` / its wire contract, a JSON spec, or
a built-in preset), registers it as a runtime scene-preset, and drives the
**existing** experiment pipeline (same 2GO solver, multirun, ALM scheduler,
best_idx logic that produced the validated zone plans) to emit a
``trajectory.json`` the deploy follower / governor can consume.

The only thing that changes vs. a normal zone run is the obstacle geometry:
``register_corridor_scene_preset`` makes the AR scene resolve like a built-in
preset for both the env and the corridor obstacle generator, so nothing in the
framework is special-cased.

Run (needs jax + the experiment deps — use Docker)::

    python scripts/tasks/robot/humanoid/replan_from_scene.py --preset zone_d \
        --out results/ar/replan_zone_d --collision-margin 0.18

    # from a saved contract / scene spec:
    python scripts/tasks/robot/humanoid/replan_from_scene.py --scene-file my_scene.json --out results/ar/run1

    # quick wiring check (reduced modes / diffusion steps):
    python scripts/tasks/robot/humanoid/replan_from_scene.py --preset zone_d --out results/ar/fast --fast
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from genedynamics.deploy.ar.scene_source import SceneSource  # noqa: E402
from genedynamics.envs.domains.humanoid.corridor import register_corridor_scene_preset  # noqa: E402

# 2GO template config whose env/method/scheduler settings produced the
# validated zone plans. We reuse it verbatim and only swap the scene + output.
DEFAULT_TEMPLATE = _ROOT / "configs/humanoid/corridor_2d/main/twogo_zone_d.yaml"

# Reduced planning knobs for a fast wiring / acceptance check (NOT production).
FAST_PLANNING = {"num_modes": 4, "M_k": 256, "Ndiffuse": 40}


def _resolve_source(source: Any) -> SceneSource:
    if isinstance(source, SceneSource):
        return source
    if isinstance(source, dict):
        return SceneSource.from_contract(source)
    s = str(source)
    if Path(s).exists():
        return SceneSource.from_file(s)
    return SceneSource.from_preset(s)


def replan_from_scene(
    source: Any,
    *,
    output_dir: str | Path,
    base_config: str | Path = DEFAULT_TEMPLATE,
    dt: Optional[float] = None,
    horizon: Optional[int] = None,
    collision_margin: Optional[float] = None,
    seed: int = 0,
    level: int = 1,
    preset_name: str = "ar_runtime",
    planning_overrides: Optional[Dict[str, Any]] = None,
    calibrate: bool = True,
    visualizations: bool = False,
    quiet: bool = False,
) -> Path:
    """Plan a 2GO trajectory for *source* and return the trajectory.json path.

    Args:
        source: ``SceneSource`` | contract dict | scene-spec JSON path | preset name.
        output_dir: results root (``level_/seed_/trajectory/trajectory.json`` below).
        collision_margin: 2GO planning clearance = the ALM constraint-scheduler
            ``margin`` (the CFS ``params.margin`` applied to the raw obstacle
            SDF). ``None`` (default) keeps the template's value (zone_d: 0.07).
            Raise it (e.g. 0.15–0.25) to fatten the planned clearance for real-HW
            / Vicon error — re-validate certification, since a larger margin
            makes tight scenes harder to plan. (robot_radius / the env
            collision_margin field do NOT affect the CFS clearance.)
        planning_overrides: optional ``{num_modes, M_k, Ndiffuse}`` (e.g. FAST_PLANNING).
        visualizations: when False (default) skip figure/GIF generation (faster;
            deploy needs only trajectory.json).
        calibrate: when True (default) run the reference-governor Step-1/3/4
            calibration after planning and bake the executed-safe ``best_idx`` +
            ``m_track`` into the plan, making it deploy-ready (this is what makes
            ``diagnose`` certify, mirroring the validated zone plans). Needs the
            MuJoCo + Spark policy, so it runs in Docker alongside planning.
    """
    from genedynamics.experiments.framework import ExperimentConfig, ExperimentRunner
    from genedynamics.experiments.runner import register_all_plugins

    src = _resolve_source(source)
    register_corridor_scene_preset(preset_name, src.scene)
    if not quiet:
        print(f"[replan] scene '{preset_name}': {len(src.scene.obstacles)} obstacles "
              f"(from {src.scene_preset or 'custom'})", flush=True)

    config = ExperimentConfig.from_yaml(Path(base_config))
    config.output_dir = Path(output_dir)
    config.obstacle_levels = [int(level)]
    config.seeds = [int(seed)]
    if not visualizations and hasattr(config, "visualizations"):
        config.visualizations = []  # deploy needs only trajectory.json; skip figures/GIFs

    # Swap scene → AR runtime preset in BOTH the env and the obstacle generator.
    config.env_params = {**config.env_params, "scene_preset": preset_name}
    config.obstacle_config = {**config.obstacle_config, "scene_preset": preset_name}
    if dt is not None:
        config.env_params["dt"] = float(dt)
    if horizon is not None:
        config.env_params["horizon"] = int(horizon)
    if collision_margin is not None:
        # 2GO uses the CFS convexifier (cfs_action). Its enforced clearance is the
        # ALM constraint-scheduler ``margin`` (== params.margin in
        # core/constraints/convexify/cfs/backends/cfs_jax.py, applied to the RAW
        # obstacle SDF). It is NOT obstacle_config.robot_radius (that's the CBF
        # path) and NOT method_params.constraint_margin (which only widens
        # active-obstacle selection). The env ``collision_margin`` field is dead.
        # So the single correct knob to fatten the planned clearance is the ALM
        # scheduler margin — raise every constraint scheduler's margin to it.
        sched = getattr(config, "scheduler_config", None)
        cs_list = sched.get("constraint_schedulers") if isinstance(sched, dict) else None
        bumped = [cs for cs in (cs_list or []) if "margin" in cs]
        if not bumped:
            raise ValueError("collision_margin set but no constraint-scheduler 'margin' to raise")
        for cs in bumped:
            cs["margin"] = float(collision_margin)

    if planning_overrides:
        if "num_modes" in planning_overrides:
            config.method_params = {**config.method_params,
                                    "num_modes": int(planning_overrides["num_modes"])}
        sched = getattr(config, "scheduler_config", None)
        diff = (sched or {}).get("diffusion_schedulers") if isinstance(sched, dict) else None
        if diff:
            for key in ("M_k", "Ndiffuse"):
                if key in planning_overrides:
                    diff[0][key] = planning_overrides[key]

    errors = config.validate()
    if errors:
        raise ValueError(f"replan config invalid: {errors}")

    runner = ExperimentRunner(config)
    register_all_plugins(runner)
    if not quiet:
        _alm = None
        _sched = getattr(config, "scheduler_config", None)
        if isinstance(_sched, dict):
            _cs = _sched.get("constraint_schedulers") or []
            _alm = next((cs.get("margin") for cs in _cs if "margin" in cs), None)
        print(f"[replan] planning 2GO (modes={config.method_params.get('num_modes')}, "
              f"dt={config.env_params.get('dt')}, horizon={config.env_params.get('horizon')}, "
              f"alm_margin={_alm}) → {config.output_dir}", flush=True)
    runner.run_all()

    traj = config.output_dir / f"level_{level}" / f"seed_{seed}" / "trajectory" / "trajectory.json"
    if not traj.exists():
        hits = list(config.output_dir.rglob("trajectory.json"))
        if not hits:
            raise FileNotFoundError(f"replan produced no trajectory.json under {config.output_dir}")
        traj = hits[0]
    if not quiet:
        n_modes = len(json.loads(traj.read_text()).get("candidate_states", []))
        print(f"[replan] wrote {traj}  ({n_modes} candidate modes)", flush=True)

    # Governor calibration: pick the executed-safe best_idx + derive m_track and
    # bake them in, so the plan certifies under the runtime governor (Step 1/3/4).
    if calibrate:
        from genedynamics.envs.domains.humanoid.corridor import corridor_scene_to_dict
        from genedynamics.deploy.followers.governor.reference_selector import PolicyEnvelope
        from scripts.tasks.robot.humanoid.governor_calibrate import calibrate as _gov_calibrate

        scene_blob = corridor_scene_to_dict(src.scene, scene_preset=preset_name)
        if not quiet:
            print("[replan] calibrating governor (Step-1 executed-safe best_idx + m_track) ...", flush=True)
        rep = _gov_calibrate(
            zone=preset_name,
            plan_path=traj,
            scene=scene_blob,
            bake=True,
            envelope=PolicyEnvelope(v_max_lat=0.55, yaw_rate_max=0.65),
            dt=float(config.env_params.get("dt", 0.25)),
            quiet_diagnose=True,
        )
        if not quiet:
            print(f"[replan] calibrated: best_idx={rep['selected_best_idx']} "
                  f"m_track={rep['m_track_derived']} "
                  f"exec_clear={rep['selected_mode']['executed_clear']:+.3f} "
                  f"({'CERT' if rep['certified_safe'] else 'VIOL'})", flush=True)
    return traj


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--preset", default="zone_d", help="Built-in scene preset (default zone_d).")
    g.add_argument("--scene-file", default=None, help="JSON scene/contract spec.")
    p.add_argument("--out", required=True, help="Output results root.")
    p.add_argument("--base-config", default=str(DEFAULT_TEMPLATE))
    p.add_argument("--dt", type=float, default=None)
    p.add_argument("--horizon", type=int, default=None)
    p.add_argument("--collision-margin", type=float, default=0.18)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--level", type=int, default=1)
    p.add_argument("--fast", action="store_true", help="Reduced modes/diffusion for a quick check.")
    args = p.parse_args(argv)

    source = args.scene_file if args.scene_file else args.preset
    traj = replan_from_scene(
        source,
        output_dir=args.out,
        base_config=args.base_config,
        dt=args.dt,
        horizon=args.horizon,
        collision_margin=args.collision_margin,
        seed=args.seed,
        level=args.level,
        planning_overrides=FAST_PLANNING if args.fast else None,
    )
    print(f"trajectory: {traj}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
