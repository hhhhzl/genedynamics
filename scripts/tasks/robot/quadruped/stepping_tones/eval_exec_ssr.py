#!/usr/bin/env python
"""Reproducible plan-consistent exec-SSR for the stepping-stones governed deploy.

Judges EXECUTION by the SAME criterion the planner SSR uses, but on the ACTUAL landed feet
(not the governor's planned/reference feet):

    exec success  <=>  exec_goal_error_xy <= plan.success_goal_margin
                       AND max_t foot_stepping_violation(landed_xy_t) <= plan.success_foothold_margin

The margins are read per-scene from the plan's own metrics (main/.../results.json) so the two
SSRs are apples-to-apples. Writes one ``exec_ssr.json`` under each <algo>/level_<L>/ dir
(aggregated over its seeds) plus a top-level ``exec_ssr_consistent.{json,md}`` summary.

Run after a deploy batch:
    python scripts/tasks/robot/quadruped/stepping_tones/eval_exec_ssr.py \
        --deploy-root results/quadruped/stepping_stones_2d/deploy/governed \
        --main-root   results/quadruped/stepping_stones_2d/main
"""
import argparse
import glob
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from genedynamics.envs.obstacles.stepping_stones import foot_stepping_violation_np


def _foot_max(res, scene):
    C = np.asarray(scene.get("stones_centers", []), np.float32).reshape(-1, 2)
    R = np.asarray(scene.get("stones_radii", []), np.float32).reshape(-1)
    P = np.asarray(scene.get("support_platforms", []), np.float32).reshape(-1, 4)
    ef = (res.get("governed", {}) or {}).get("executed_footholds", []) or []
    v = [float(foot_stepping_violation_np(np.asarray(e["landed_xy"], np.float32), C, R, P, 0.0)) for e in ef]
    return max(v) if v else 0.0


def _seed_record(deploy_root, main_root, algo, level, seed):
    gov = Path(deploy_root) / algo / f"level_{level}" / f"seed_{seed}" / "governed"
    rj = gov / "result.json"
    if not rj.exists():
        return None
    j = json.load(open(rj))
    res = pickle.load(open(gov / "res.pkl", "rb"))
    scene = json.load(open(gov / "stepping_scene.json"))
    pm = json.load(open(Path(main_root) / algo / f"level_{level}" / f"seed_{seed}" / "results.json"))["metrics"]
    gm = float(pm["stepping_metrics"]["success_goal_margin"])
    fm = float(pm["stepping_metrics"]["success_foothold_margin"])
    ge = float(j["summary"]["goal_error_xy"])
    fmax = _foot_max(res, scene)
    return {
        "seed": int(seed),
        "plan_ssr": float(pm["ssr"]["ssr"]),
        "exec_goal_error": round(ge, 4),
        "exec_foot_max": round(fmax, 4),
        "goal_margin": gm,
        "foothold_margin": fm,
        "exec_consistent": bool(ge <= gm and fmax <= fm),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy-root", default="results/quadruped/stepping_stones_2d/deploy/governed")
    ap.add_argument("--main-root", default="results/quadruped/stepping_stones_2d/main")
    args = ap.parse_args()

    # discover (algo, level, seed) from the deploy tree
    combos = defaultdict(list)  # (algo, level) -> [seed,...]
    for f in glob.glob(str(Path(args.deploy_root) / "*/level_*/seed_*/governed/result.json")):
        m = re.search(r"/([^/]+)/level_(\d+)/seed_(\d+)/governed/result.json$", f)
        if m:
            combos[(m.group(1), int(m.group(2)))].append(int(m.group(3)))

    summary_rows, lines = [], []
    lines.append("# Plan-consistent exec-SSR  (exec judged by the plan's own criterion, on ACTUAL landed feet)")
    lines.append(f'{"algo":>10} {"lvl":>3} | {"plan-SSR":>8} {"exec-SSR":>8} | {"mean_foot_max":>13}')
    tp = te = tn = 0
    for (algo, level) in sorted(combos):
        recs = [r for s in sorted(combos[(algo, level)]) if (r := _seed_record(args.deploy_root, args.main_root, algo, level, s))]
        if not recs:
            continue
        plan_ssr = float(np.mean([r["plan_ssr"] for r in recs]))
        exec_ssr = float(np.mean([r["exec_consistent"] for r in recs]))
        mean_fmax = float(np.mean([r["exec_foot_max"] for r in recs]))
        out = {
            "algo": algo, "level": level, "n_seeds": len(recs),
            "plan_ssr": round(plan_ssr, 3),
            "exec_ssr_consistent": round(exec_ssr, 3),
            "mean_exec_foot_max": round(mean_fmax, 4),
            "criterion": "exec_goal_error <= success_goal_margin AND actual foot_max <= foothold_margin",
            "seeds": recs,
        }
        dst = Path(args.deploy_root) / algo / f"level_{level}" / "exec_ssr.json"
        json.dump(out, open(dst, "w"), indent=2)
        print(f"wrote {dst}")
        summary_rows.append(out)
        lines.append(f'{algo:>10} {level:>3} | {plan_ssr:>8.2f} {exec_ssr:>8.2f} | {mean_fmax:>13.3f}')
        tp += plan_ssr * len(recs); te += exec_ssr * len(recs); tn += len(recs)
    if tn:
        lines.append(f'{"OVERALL":>14} | {tp/tn:>8.2f} {te/tn:>8.2f}   (n={tn})')
    txt = "\n".join(lines)
    print("\n" + txt)
    top = Path(args.deploy_root).parent
    open(top / "exec_ssr_consistent.md", "w").write(txt + "\n")
    json.dump(summary_rows, open(top / "exec_ssr_consistent.json", "w"), indent=2)
    print(f"\nsaved summary -> {top}/exec_ssr_consistent.md + .json")


if __name__ == "__main__":
    main()
