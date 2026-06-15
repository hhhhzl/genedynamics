#!/usr/bin/env python
"""Audit the deploy's body-SDF certificate against the ACTUAL executed G1 geometry.

The certificate (sport_mode.json executed_min_body_sdf / certified_safe) evaluates a
REDUCED body model (elliptical torso + tucked arm tips). This audit re-evaluates the
SAME per-obstacle SDF used by that certificate (genedynamics...governor.admissible_bodysdf
._obs_point_sdf — box/sphere/qc) but at the FK'd world positions of EVERY actual robot
link, so an arm that swings wider than the tucked model is caught.

Per (zone, seed) it reports: certificate min SDF vs the true min SDF over actual links,
the worst body part, and a collision verdict (true_min < 0). Aggregates a certificate-
based vs true exec-SSR per zone.

Run in docker (needs mujoco + the G1 model)::

  docker run --rm -e PYTHONPATH=/workspace -e MUJOCO_GL=osmesa \\
    -e MUJOCO_MENAGERIE_PATH=/workspace/third_party/mujoco_menagerie \\
    -v "$PWD:/workspace" -w /workspace genedynamics/dev-cpu:torch \\
    python scripts/tasks/robot/humanoid/audit_exec_collision.py \\
      --deploy-root results/humanoid/corridor_2d/deploy/governed
"""
import argparse
import glob
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[4]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _audit_seed(seed_dir: Path):
    import mujoco
    from genedynamics.deploy.followers.governor.admissible_bodysdf import _parse_obstacles, _obs_point_sdf
    from genedynamics.envs.utils.mujoco_model_generator import create_g1_render_xml_with_trajectory
    from scripts.visualizations.render_deploy_humanoid import _resolve_g1_xml_path

    npz = seed_dir / "sport_mode.npz"
    sj = seed_dir / "sport_mode.json"
    scene_p = seed_dir / "corridor_scene.json"
    if not (npz.exists() and sj.exists() and scene_p.exists()):
        return None
    z = np.load(npz, allow_pickle=True)
    qp = np.asarray(z["qpos"])
    cert_min = float(np.min(z["executed_body_sdf"])) if "executed_body_sdf" in z.files else float("nan")
    info = json.load(open(sj))
    fell = bool(info.get("fell_over", True))
    cert_flag = bool(info.get("certified_safe", False))
    scene = json.load(open(scene_p))
    obstacles = _parse_obstacles(scene.get("obstacles", []))
    if not obstacles:
        return None

    g1 = _resolve_g1_xml_path()
    tmp = Path(g1).parent / "_audit_temp.xml"
    create_g1_render_xml_with_trajectory(str(tmp), trajectory_positions=[], corridor_scene=scene)
    try:
        m = mujoco.MjModel.from_xml_path(str(tmp))
        d = mujoco.MjData(m)
        ng = m.ngeom
        gname = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or f"g{i}" for i in range(ng)]
        robot = [i for i in range(ng)
                 if not gname[i].startswith(("corridor_", "traj_", "floor"))
                 and int(m.geom_bodyid[i]) != 0]

        def bname(i):
            return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[i])) or "?"

        best = (1e9, None, -1)
        for fi in range(qp.shape[0]):
            d.qpos[:] = qp[fi, :m.nq]; d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            for i in robot:
                px, py = float(d.geom_xpos[i, 0]), float(d.geom_xpos[i, 1])
                s = min(_obs_point_sdf(px, py, o) for o in obstacles)
                if s < best[0]:
                    best = (s, bname(i), fi)
        endpoint = float(info.get("endpoint_distance_m", float("inf")))
        return {
            "true_min_sdf": round(best[0], 4),
            "worst_body": best[1],
            "worst_frame": best[2],
            "cert_min_sdf": round(cert_min, 4),
            "certified_safe": cert_flag,
            "fell_over": fell,
            "true_collision": bool(best[0] < 0.0),
            "endpoint_m": round(endpoint, 4),
        }
    finally:
        tmp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-root", default="results/humanoid/corridor_2d/deploy/governed")
    ap.add_argument("--zones", nargs="+", default=None,
                    help="Limit to these run names (e.g. twogo_zone_a). Default: all found.")
    ap.add_argument("--reach-margin", type=float, default=0.20,
                    help="endpoint_distance_m must be <= this to count as reaching the goal (default 0.20)")
    args = ap.parse_args()
    root = Path(args.deploy_root)
    mode = root.name
    rm = float(args.reach_margin)

    combos = defaultdict(list)
    for f in glob.glob(str(root / "*/level_1/seed_*/sport_mode.npz")):
        mobj = re.search(r"/([^/]+)/level_1/seed_(\d+)/sport_mode.npz$", f)
        if mobj and (args.zones is None or mobj.group(1) in args.zones):
            combos[mobj.group(1)].append(int(mobj.group(2)))

    rows = []
    print(f"# Exec audit ({mode}) — exec-SSR = no-fall AND collision-free(true geom) AND reach (endpoint<={rm})\n")
    hdr = (f'{"run":>14} {"seed":>4} | {"true_sdf":>8} {"safe":>5} {"endpt":>6} {"reach":>5} '
           f'{"worst_body":>22} | {"eSSR":>4}')
    print(hdr)
    summary = {}
    for run in sorted(combos):
        true_ok = n = 0
        for s in sorted(combos[run]):
            r = _audit_seed(root / run / "level_1" / f"seed_{s}")
            if r is None:
                continue
            n += 1
            safe = (not r["fell_over"]) and (not r["true_collision"])
            reached = r["endpoint_m"] <= rm
            ok = safe and reached
            true_ok += int(ok)
            r["reached"] = bool(reached); r["exec_ok"] = bool(ok)
            print(f'{run:>14} {s:>4} | {r["true_min_sdf"]:>+8.3f} {"Y" if not r["true_collision"] and not r["fell_over"] else "n":>5} '
                  f'{r["endpoint_m"]:>6.2f} {"Y" if reached else "n":>5} {r["worst_body"]:>22} | {"Y" if ok else "n":>4}')
            rows.append({"run": run, "seed": s, **r})
        if n:
            summary[run] = {"n": n, "exec_ssr": round(true_ok / n, 3)}
            print(f'{run:>14}  ->  exec-SSR = {true_ok}/{n} = {true_ok/n:.2f}\n')

    out = root.parent / f"exec_audit_{mode}.json"
    json.dump({"reach_margin": rm, "criterion": "no-fall AND collision-free(true) AND endpoint<=reach_margin",
               "summary": summary, "seeds": rows}, open(out, "w"), indent=2)
    print("=" * 52)
    print(f'{"run":>14} | {"exec-SSR (safe & reach)":>24}')
    tot_ok = tot_n = 0
    for run, sv in sorted(summary.items()):
        print(f'{run:>14} | {sv["exec_ssr"]:>24.2f}')
        tot_ok += sv["exec_ssr"] * sv["n"]; tot_n += sv["n"]
    if tot_n:
        print(f'{"OVERALL":>14} | {tot_ok/tot_n:>24.2f}   (n={tot_n})')
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
