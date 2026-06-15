#!/usr/bin/env python
"""Humanoid corridor deploy visualizations — the G1 analog of the stepping-stones
deploy figures (run_stepping_execution.py make_tracking_plot / make_motion_strip /
make_motion_gif).

Reads a diagnose ``sport_mode.npz`` (+ sibling ``corridor_scene.json``) and writes,
into the same directory, the stepping-styled set:

  tracking.png          forward position s + forward speed v, executed vs reference
                        (pure matplotlib — runs anywhere)
  sport_mode.png        the 2x2 plan-vs-executed panel, restyled to the stepping
                        _paper_style typography (pure matplotlib)
  motion_strip.png      ghosted multi-pose strip, earlier poses faded -> final solid
                        (MuJoCo G1 render + background-subtraction compositing)
  trajectory_mujoco.gif following-camera replay (MuJoCo G1 render)

Typography matches stepping's _paper_style (font 18 / labels 26 / ticks 22 / legend 22).

Run the MuJoCo parts inside the dev-cpu:torch image::

  docker run --rm -v "$PWD:/workspace" -w /workspace genedynamics/dev-cpu:torch \\
    python scripts/visualizations/render_deploy_humanoid.py \\
      --npz results/humanoid/corridor_2d/deploy/governed/twogo_zone_a/level_1/seed_0/sport_mode.npz \\
      --which all
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CONTROL_HZ = 50.0


def _paper_style() -> None:
    """Match the stepping-stones deploy figures' typography (large labels/ticks)."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 18, "axes.labelsize": 26, "axes.titlesize": 24,
        "xtick.labelsize": 22, "ytick.labelsize": 22, "legend.fontsize": 22,
        "axes.linewidth": 1.1,
    })


def _movavg(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win <= 1 or x.size < win:
        return x
    k = np.ones(win) / win
    return np.convolve(x, k, mode="same")


def _load(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    return {k: np.asarray(d[k]) for k in d.files}


# --------------------------------------------------------------------------- tracking.png
def make_tracking_plot(npz_path: Path, *, out: Optional[Path] = None) -> Optional[Path]:
    """Execution-tracks-reference: forward position s (executed pelvis vs plan) and
    forward speed v (executed vs commanded), the corridor analog of the stepping
    tracking plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _paper_style()
    z = _load(npz_path)
    pelvis_xy = z["pelvis_xy"]; plan_xy = z["plan_xy"]
    intent = z["intent_lin_vel"]
    n = pelvis_xy.shape[0]
    dt = 1.0 / CONTROL_HZ
    t = np.arange(n) * dt
    s_act = pelvis_xy[:, 0]
    s_ref = plan_xy[:, 0]
    win = max(10, int(0.4 * CONTROL_HZ))  # ~0.4 s smoothing for the speed traces
    v_act = _movavg(np.gradient(s_act, dt), win)
    v_cmd = intent[:, 0]  # commanded forward (body-frame ~ world-x here)

    fig, ax = plt.subplots(2, 1, figsize=(8.6, 6.0), sharex=True, constrained_layout=True)
    ax[0].plot(t, s_ref, color="0.35", ls="--", lw=2.2, label="reference")
    ax[0].plot(t, s_act, color="#1f4fd8", lw=2.4, label="executed")
    ax[0].set_ylabel("s (m)"); ax[0].grid(alpha=0.25)
    ax[0].legend(loc="upper left", framealpha=0.9)
    ax[1].plot(t, v_act, color="#1f4fd8", lw=2.4, label="actual")
    ax[1].plot(t, v_cmd, color="#e23b3b", ls="--", lw=2.0, label="commanded")
    ax[1].axhline(0.0, color="0.8", lw=0.8)
    ax[1].set_ylabel("v (m/s)"); ax[1].set_xlabel("time (s)"); ax[1].grid(alpha=0.25)
    ax[1].legend(loc="upper left", framealpha=0.9)
    out = out or (npz_path.parent / "tracking.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    print("  [tracking] wrote", out)
    return out


# --------------------------------------------------------------------------- sport_mode.png (restyled)
def make_sport_mode_plot(npz_path: Path, *, out: Optional[Path] = None) -> Optional[Path]:
    """Regenerate the 2x2 plan-vs-executed panel from the npz with the stepping
    _paper_style typography (replaces the small-font default sport_mode.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _paper_style()
    z = _load(npz_path)
    n = z["pelvis_xy"].shape[0]
    t = np.arange(n) / CONTROL_HZ
    names = list(z["actuated_joint_names"].tolist())
    cmd = z["cmd_joint_pos"]; arm = z["plan_arm_torso"]
    pelvis = z["pelvis_xy"]; plan = z["plan_xy"]

    def ci(name):
        return names.index(name) if name in names else None

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    LEG = 14  # legend fontsize (the rcParams 22 is too big for a dense 2x2)

    ax = axes[0, 0]
    ax.plot(plan[:, 0], plan[:, 1], "k--", lw=2.4, label="plan")
    ax.plot(pelvis[:, 0], pelvis[:, 1], color="#1f4fd8", lw=2.0, label="pelvis")
    ax.scatter(plan[-1, 0], plan[-1, 1], s=120, c="k", marker="x", label="plan end")
    ax.scatter(pelvis[-1, 0], pelvis[-1, 1], s=120, c="#1f4fd8", marker="o", label="pelvis end")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.set_title("XY trace")
    ax.legend(loc="best", fontsize=LEG); ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, arm[:n, 0], "C0--", lw=1.8, label="plan a_left")
    ax.plot(t, arm[:n, 1], "C3--", lw=1.8, label="plan a_right")
    for nm, c, lab in (("left_elbow_joint", "C0", "cmd L elbow"), ("right_elbow_joint", "C3", "cmd R elbow")):
        i = ci(nm)
        if i is not None:
            ax.plot(t, cmd[:n, i], c + "-", lw=1.8, label=lab)
    ax.set_xlabel("t (s)"); ax.set_title("Arm tuck"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=LEG)

    ax = axes[1, 0]
    ax.plot(t, arm[:n, 2], "C0--", lw=1.8, label="plan p_left")
    ax.plot(t, arm[:n, 3], "C3--", lw=1.8, label="plan p_right")
    for nm, c, lab in (("left_shoulder_pitch_joint", "C0", "cmd L sh.pitch"),
                       ("right_shoulder_pitch_joint", "C3", "cmd R sh.pitch")):
        i = ci(nm)
        if i is not None:
            ax.plot(t, cmd[:n, i], c + "-", lw=1.8, label=lab)
    ax.set_xlabel("t (s)"); ax.set_title("Arm posture"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=LEG)

    ax = axes[1, 1]
    ax.plot(t, arm[:n, 4], "k--", lw=2.0, label="plan psi_torso")
    i = ci("waist_yaw_joint")
    if i is not None:
        ax.plot(t, cmd[:n, i], "C2-", lw=2.0, label="cmd waist_yaw")
    ax.set_xlabel("t (s)"); ax.set_ylabel("rad"); ax.set_title("Torso yaw"); ax.grid(True, alpha=0.3); ax.legend(loc="best", fontsize=LEG)

    fig.tight_layout()
    out = out or (npz_path.parent / "sport_mode.png")
    fig.savefig(out, dpi=160); plt.close(fig)
    print("  [sport_mode] wrote", out)
    return out


# --------------------------------------------------------------------------- MuJoCo helpers
def _resolve_g1_xml_path():
    """Resolve g1.xml so the temp render XML can sit beside it (mesh <include>s resolve)."""
    import os
    try:
        from genedynamics.robots.registry import _get_g1_path
        p = _get_g1_path()
        if p:
            return str(p)
    except Exception:
        pass
    menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
    if menagerie:
        cand = Path(menagerie) / "unitree_g1" / "g1.xml"
        if cand.exists():
            return str(cand)
    proj = Path(__file__).resolve().parents[2]
    for d in (proj / "third_party" / "mujoco_menagerie", proj / "mujoco_menagerie"):
        cand = d / "unitree_g1" / "g1.xml"
        if cand.exists():
            return str(cand)
    return None


def _build_g1_scene(corridor_scene, trajectory_positions, *,
                    line_radius: float = 0.008, line_rgba: str = "0.2 0.6 1.0 0.7"):
    """Create the G1 + corridor XML (temp must sit beside g1.xml so meshes resolve)."""
    import mujoco
    import tempfile, os
    from genedynamics.envs.utils.mujoco_model_generator import create_g1_render_xml_with_trajectory

    g1_xml = _resolve_g1_xml_path()
    if g1_xml is not None:
        tmp = Path(g1_xml).parent / "_deploy_humanoid_temp.xml"
    else:
        fd, p = tempfile.mkstemp(suffix=".xml"); os.close(fd); tmp = Path(p)
    create_g1_render_xml_with_trajectory(str(tmp), trajectory_positions=trajectory_positions,
                                         line_radius=line_radius, line_rgba=line_rgba,
                                         corridor_scene=corridor_scene)
    m = mujoco.MjModel.from_xml_path(str(tmp))
    return m, tmp


def _corridor_scene(npz_path: Path):
    p = npz_path.parent / "corridor_scene.json"
    if not p.exists():
        return None
    scene = json.load(open(p))
    scene = dict(scene); scene["hide_floor_patch"] = True
    return scene


# --------------------------------------------------------------------------- motion_strip.png
def make_motion_strip(npz_path: Path, *, n_poses: int = 6, width: int = 1280, height: int = 360,
                      azimuth: float = 90.0, elevation: float = -24.0, distance: float = 2.5,
                      out: Optional[Path] = None) -> Optional[Path]:
    """Ghosted multi-pose strip of the G1 traverse (earlier poses faded -> final solid)
    via per-pixel background subtraction, front side-elevation camera. Mirrors the
    stepping make_motion_strip. ``elevation`` more negative = camera higher / looking
    down more; ``distance`` = camera distance (larger = farther)."""
    import mujoco, imageio
    z = _load(npz_path)
    qp = z["qpos"]
    scene = _corridor_scene(npz_path)
    # Thin green pelvis line stringing the ghost poses together. It's static across
    # frames, so the median-background compositing keeps it. Subsample for fewer geoms.
    pelvis_line = qp[::3, :3].tolist()
    m, tmp = _build_g1_scene(scene, trajectory_positions=pelvis_line,
                             line_radius=0.005, line_rgba="0.20 0.80 0.30 0.95")
    try:
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, height=height, width=width)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        x0, x1 = float(qp[:, 0].min()), float(qp[:, 0].max())
        span = max(1.0, x1 - x0)
        cam.lookat[:] = [0.5 * (x0 + x1), 0.0, 0.55]
        cam.azimuth = float(azimuth); cam.elevation = float(elevation)
        cam.distance = float(distance)
        idx = np.linspace(0, qp.shape[0] - 1, int(n_poses)).astype(int)
        frames = []
        for i in idx:
            d.qpos[:] = qp[i, : m.nq]; d.qvel[:] = 0.0
            mujoco.mj_forward(m, d); r.update_scene(d, camera=cam)
            frames.append(r.render().astype(np.float32))
        frames = np.stack(frames, 0)
        bg = np.median(frames, axis=0)
        result = bg.copy()
        ramp = np.linspace(0.40, 1.0, int(n_poses))
        for i in range(int(n_poses)):
            mask = np.abs(frames[i] - bg).max(axis=2) > 16.0
            w = float(ramp[i])
            result[mask] = (1.0 - w) * result[mask] + w * frames[i][mask]
        out = out or (npz_path.parent / "motion_strip.png")
        imageio.imwrite(str(out), np.clip(result, 0, 255).astype(np.uint8))
        print("  [strip] wrote", out, f"({n_poses} poses, az={azimuth} el={elevation} dist={cam.distance:.2f})")
        return out
    finally:
        tmp.unlink(missing_ok=True)


# --------------------------------------------------------------------------- trajectory_mujoco.gif
def make_motion_gif(npz_path: Path, *, width: int = 720, height: int = 540, every_n: int = 2,
                    speed: float = 0.3, fps: float = 50.0, cam_distance: float = 3.5,
                    cam_azimuth: float = 150.0, cam_elevation: float = -14.0,
                    out: Optional[Path] = None) -> Optional[Path]:
    """Following-camera replay GIF (G1): the robot walks the corridor with a blue pelvis
    trace, the camera tracks the pelvis. Self-contained (no external renderer dep)."""
    import mujoco, imageio
    z = _load(npz_path)
    qp = z["qpos"]
    if every_n > 1:
        qp = qp[::int(every_n)]
    scene = _corridor_scene(npz_path)
    # blue pelvis trace (default line color) for the follow-cam replay.
    m, tmp = _build_g1_scene(scene, trajectory_positions=qp[:, :3].tolist())
    try:
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), width)
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), height)
        d = mujoco.MjData(m)
        r = mujoco.Renderer(m, height=height, width=width)
        cam = mujoco.MjvCamera(); cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.distance = float(cam_distance); cam.azimuth = float(cam_azimuth); cam.elevation = float(cam_elevation)
        nq = m.nq
        frames = []
        for i in range(qp.shape[0]):
            d.qpos[:] = qp[i, :nq]; d.qvel[:] = 0.0
            mujoco.mj_forward(m, d)
            cam.lookat[0] = float(d.qpos[0]); cam.lookat[1] = float(d.qpos[1]); cam.lookat[2] = 0.9
            r.update_scene(d, camera=cam)
            frames.append(r.render())
        out = out or (npz_path.parent / "trajectory_mujoco.gif")
        imageio.mimsave(str(out), frames, fps=max(1.0, float(fps) * float(speed)), loop=0)
        print("  [gif] wrote", out, f"({len(frames)} frames)")
        return out
    finally:
        tmp.unlink(missing_ok=True)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", type=Path, required=True, help="Path to a deploy sport_mode.npz")
    p.add_argument("--which", choices=("all", "tracking", "sport", "strip", "gif"), default="all")
    p.add_argument("--poses", type=int, default=6)
    p.add_argument("--every-n", type=int, default=2)
    p.add_argument("--strip-elev", type=float, default=-24.0, help="strip camera elevation (more negative = higher/looking down)")
    p.add_argument("--strip-dist", type=float, default=2.5, help="strip camera distance (larger = farther)")
    args = p.parse_args(argv)
    npz = args.npz
    if not npz.exists():
        print(f"ERROR: npz not found: {npz}", file=sys.stderr); return 2
    w = args.which
    if w in ("all", "tracking"):
        make_tracking_plot(npz)
    if w in ("all", "sport"):
        make_sport_mode_plot(npz)
    if w in ("all", "strip"):
        try:
            make_motion_strip(npz, n_poses=args.poses, elevation=args.strip_elev, distance=args.strip_dist)
        except Exception as e:
            print("  [strip] skipped (needs native MuJoCo):", e)
    if w in ("all", "gif"):
        try:
            make_motion_gif(npz, every_n=args.every_n)
        except Exception as e:
            print("  [gif] skipped (needs native MuJoCo):", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())
