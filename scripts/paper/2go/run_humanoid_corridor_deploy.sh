#!/usr/bin/env bash
# All-in-one humanoid corridor deploy: execution + figures + eSSR audit, in one command.
#
# Per (algo, zone, seed):
#   1. [CALIB=1] governor_calibrate -> bake executed-safe best_idx + m_track into the plan
#   2. run_sport_mode_zones (reference governor + plan_speed 0.5 + goal-hold + light arm tuck)
#        -> sport_mode.{npz,json} + corridor_scene.json
#   3. [RENDER=1] render_deploy_humanoid --which all
#        -> tracking.png / sport_mode.png / motion_strip.png (green pelvis line) / trajectory_mujoco.gif
# Then [AUDIT=1] audit_exec_collision -> exec_audit_<mode>.json + the eSSR table
#        (eSSR = no-fall AND true-collision-free AND reach<=0.20).
#
# Plan source : results/humanoid/corridor_2d/main/<algo>_zone_<z>/level_1/seed_<s>
# Output       : results/humanoid/corridor_2d/deploy/<mode>/<algo>_zone_<z>/level_1/seed_<s>/
#
# REQUIRES MuJoCo + PyTorch -> run inside the arm64 docker image genedynamics/dev-cpu:torch
# (NOT the x86_64/Rosetta fedguide env):
#   docker run --rm --platform linux/arm64 -e MUJOCO_GL=osmesa -e JAX_PLATFORMS=cpu \
#     -e PYTHONPATH=/workspace -e MUJOCO_MENAGERIE_PATH=/workspace/third_party/mujoco_menagerie \
#     -v "$PWD":/workspace -w /workspace genedynamics/dev-cpu:torch \
#     bash scripts/paper/2go/run_humanoid_corridor_deploy.sh
#
# Env flags: MODE(governed|raw|both) ALGOS ZONES SEEDS MAXSTEPS PLOT
#   RENDER(default 1)  per-seed figures (tracking/sport_mode/motion_strip/gif)
#   AUDIT (default 1)  eSSR + collision table per mode
#   CALIB (default 0)  calibrate plans first (bakes best_idx+m_track into main/ plans;
#                      EXPENSIVE — rolls out every candidate mode; mutates the paper plans).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
RUNNER="scripts/tasks/robot/humanoid/run_sport_mode_zones.py"
CAL="scripts/tasks/robot/humanoid/governor_calibrate.py"
VIZ="scripts/visualizations/render_deploy_humanoid.py"
AUDITOR="scripts/tasks/robot/humanoid/audit_exec_collision.py"
PLAN_ROOT="results/humanoid/corridor_2d/main"
OUT_ROOT="results/humanoid/corridor_2d/deploy"

MODE="${MODE:-governed}"          # governed | raw | both
PLOT="${PLOT:-1}"                 # 1 -> pass --plot (per-seed sport_mode.png from the runner)
RENDER="${RENDER:-1}"             # 1 -> render the full figure set per seed (overrides sport_mode.png)
AUDIT="${AUDIT:-1}"               # 1 -> run the eSSR / collision audit at the end
CALIB="${CALIB:-0}"              # 1 -> governor_calibrate each plan before deploy (expensive, mutates main/)
ALGOS="${ALGOS:-mbd mppi ebmbd mdoc mdcoas twogo}"
ZONES="${ZONES:-a b c d}"
SEEDS="${SEEDS:-0 1 2 3 4}"
MAXSTEPS="${MAXSTEPS:-}"          # optional cap on rollout length per plan

plot_flag=""; [ "$PLOT" = "1" ] && plot_flag="--plot"
maxsteps_flag=""; [ -n "$MAXSTEPS" ] && maxsteps_flag="--max-steps $MAXSTEPS"

case "$MODE" in
  governed) MODES="governed" ;;
  raw)      MODES="raw" ;;
  both)     MODES="governed raw" ;;
  *) echo "!! bad MODE=$MODE (governed|raw|both)"; exit 2 ;;
esac

total=0; ok=0; skip=0; fail=0
for m in $MODES; do
  gov_flag=""; [ "$m" = "raw" ] && gov_flag="--no-governor"
  for algo in $ALGOS; do
    for z in $ZONES; do
      run="${algo}_zone_${z}"
      # Execution-only fixes (the planner is fixed) are mapper/diagnose DEFAULTS:
      # arm tuck (shoulder_roll 0.15 / elbow_base 0.60), plan_speed 0.5, goal-hold 5s.
      for s in $SEEDS; do
        total=$((total+1))
        plan="$PLAN_ROOT/$run/level_1/seed_${s}/trajectory/trajectory.json"
        out_root="$OUT_ROOT/$m/$run/level_1"   # run_sport_mode_zones names the leaf seed_<s>
        seed_dir="$out_root/seed_${s}"
        if [ ! -f "$plan" ]; then
          echo "-- skip [$m] $run s$s : no plan at $plan"; skip=$((skip+1)); continue
        fi

        # 1. calibrate (governed only): bake executed-safe best_idx + m_track into the plan.
        if [ "$CALIB" = "1" ] && [ "$m" = "governed" ]; then
          echo ".. calibrate $run s$s"
          "$PY" "$CAL" --zones "$plan" >/dev/null 2>&1 || echo "   !! calib warn $run s$s"
        fi

        # 2. deploy (execution rollout).
        echo "== [$m] $run s$s -> $seed_dir"
        "$PY" "$RUNNER" \
          --zones "$plan" \
          --out-root "$out_root" \
          $gov_flag $plot_flag $maxsteps_flag --continue-on-fall
        if [ $? -eq 0 ]; then ok=$((ok+1)); else echo "!! FAILED [$m] $run s$s"; fail=$((fail+1)); fi

        # 3. render the full figure set (tracking / sport_mode / motion_strip / gif).
        if [ "$RENDER" = "1" ] && [ -f "$seed_dir/sport_mode.npz" ]; then
          "$PY" "$VIZ" --npz "$seed_dir/sport_mode.npz" --which all >/dev/null 2>&1 \
            || echo "   !! render warn $run s$s"
        fi
      done
    done
  done
done

# eSSR / collision audit per mode (over the runs just deployed).
if [ "$AUDIT" = "1" ]; then
  zlist=""; for algo in $ALGOS; do for z in $ZONES; do zlist="$zlist ${algo}_zone_${z}"; done; done
  for m in $MODES; do
    echo
    echo "#### eSSR audit [$m] ####"
    "$PY" "$AUDITOR" --deploy-root "$OUT_ROOT/$m" --zones $zlist
  done
fi

echo
echo "done: $ok ok / $skip skipped / $fail failed  (of $total)"
