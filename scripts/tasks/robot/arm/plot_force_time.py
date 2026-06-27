"""Force-vs-time plots with f_min/f_max/f_target bands, from the saved per-run series.

Reads results/arm/impedence/rigid/<fam>/<role>/<method>/metrics.json (each record carries a "series" dict with the
per-step real contact force + the limits) and plots, per surface family, the mdac vs
dial executed contact force over the scan, against the [f_min, f_max] band and the
f_target line. Pure-numpy/matplotlib (no brax) -> runs in fedguide.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMS = ["plane", "cylinder", "convex", "bumpy", "unseen"]
# run from repo root (matches the other arm scripts' cwd-relative IO)
RES = "results/arm/impedence/rigid"


def _role(method):
    if method == "mdac":
        return "main"
    return "baseline" if method in ("dial", "mppi") else "ablation"


def _rec(fam, method, seed=0):
    # per-(environment, role, method) metrics.json — aligned with configs/output_dir
    p = os.path.join(RES, fam, _role(method), method, "metrics.json")
    if not os.path.exists(p):
        return None
    for r in json.load(open(p)):
        if r.get("method") == method and r.get("seed") == seed:
            return r
    return None


fig, axes = plt.subplots(1, len(FAMS), figsize=(4 * len(FAMS), 3.6), sharey=True)
for ax, fam in zip(axes, FAMS):
    rm, rd = _rec(fam, "mdac"), _rec(fam, "dial")
    s = rm["series"]
    fmin, fmax, ftgt = float(s["f_min"]), float(s["f_max"]), float(s["f_target"])
    fm = np.asarray(s["force"]); fd = np.asarray(rd["series"]["force"])
    t = np.arange(len(fm))
    ax.axhspan(fmin, fmax, color="0.85", label="[f_min, f_max]")
    ax.axhline(ftgt, color="green", ls="--", lw=1.2, label=f"f_target={ftgt:.0f}")
    ax.axhline(fmin, color="0.5", lw=0.8); ax.axhline(fmax, color="0.5", lw=0.8)
    ax.plot(t, fm, "-o", color="C0", ms=3, lw=1.8, label="mdac")
    ax.plot(t, fd, "--s", color="C3", ms=3, lw=1.4, label="dial")
    ftrk_m = rm.get("force_tracking_error_tracked", float("nan"))
    ftrk_d = rd.get("force_tracking_error_tracked", float("nan"))
    ax.set_title(f"{fam}\nforce|trk  mdac {ftrk_m:.1f} / dial {ftrk_d:.1f}", fontsize=10)
    ax.set_xlabel("scan step"); ax.set_ylim(-5, max(fmax, fm.max(), fd.max()) + 8)
axes[0].set_ylabel("contact force (N)")
axes[0].legend(fontsize=7, loc="upper right")
fig.suptitle("Real contact force over the scan: mdac vs dial (f_target=45, band=[0,60])", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = os.path.join(RES, "force_vs_time.png")
fig.savefig(out, dpi=130)
print("wrote", out)
