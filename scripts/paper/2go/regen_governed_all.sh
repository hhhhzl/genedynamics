#!/usr/bin/env bash
# Regenerate deploy/governed outputs for EVERY (algo, level, seed).
#
# Plan source : results/quadruped/stepping_stones_2d/main/<algo>/level_<L>/seed_<s>
# Output       : results/quadruped/stepping_stones_2d/deploy/governed/<algo>/level_<L>/seed_<s>/{governed[,raw]}
#
# Each run reuses the per-algo tuned walker/governor config (via --config) but
# overrides the plan seed-dir + output-dir so the scene/plan come from `main`.
#
# REQUIRES MuJoCo (sim + offscreen render). It does NOT run in the x86_64/Rosetta
# `fedguide` env — run this on the native-arm64 / GPU box. Point PY at that env:
#
#   PY=/path/to/arm64/python bash scripts/tasks/robot/quadruped/stepping_tones/regen_governed_all.sh
#
# Overridable via env: PY, MODE (governed|raw|both), LEVELS, SEEDS, ALGOS, FIGURES(0|1).
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/../../../../.." && pwd)"
cd "$ROOT"

PY="${PY:-python}"
RUNNER="scripts/tasks/robot/quadruped/stepping_tones/run_stepping_execution.py"
PLAN_ROOT="results/quadruped/stepping_stones_2d/main"
OUT_ROOT="results/quadruped/stepping_stones_2d/deploy/governed"
CFG_DIR="configs/quadruped/stepping_stones_2d/deploy"

MODE="${MODE:-governed}"          # governed | raw | both
FIGURES="${FIGURES:-1}"           # 1 -> pass --figures (tracking/gait/motion-strip pngs)
ALGOS="${ALGOS:-ebmbd mbd mdcoas mdoc twogo}"
LEVELS="${LEVELS:-1 5}"
SEEDS="${SEEDS:-0 1 2 3 4}"

cfg_for() {  # algo -> deploy yaml (twogo uses the generic governed_exec.yaml)
  case "$1" in
    twogo) echo "$CFG_DIR/governed_exec.yaml" ;;
    *)     echo "$CFG_DIR/governed_$1.yaml" ;;
  esac
}

fig_flag=""; [ "$FIGURES" = "1" ] && fig_flag="--figures"

total=0; ok=0; skip=0; fail=0
for algo in $ALGOS; do
  cfg="$(cfg_for "$algo")"
  if [ ! -f "$cfg" ]; then echo "!! missing config $cfg (skip $algo)"; continue; fi
  for L in $LEVELS; do
    for s in $SEEDS; do
      total=$((total+1))
      plan="$PLAN_ROOT/$algo/level_${L}/seed_${s}"
      out="$OUT_ROOT/$algo/level_${L}/seed_${s}"
      if [ ! -f "$plan/trajectory/trajectory.json" ]; then
        echo "-- skip $algo L$L s$s : no plan at $plan"; skip=$((skip+1)); continue
      fi
      echo "== $algo L$L s$s -> $out"
      "$PY" "$RUNNER" exec \
        --config "$cfg" \
        --seed-dir "$plan" \
        --output-dir "$out" \
        --mode "$MODE" $fig_flag
      if [ $? -eq 0 ]; then ok=$((ok+1)); else echo "!! FAILED $algo L$L s$s"; fail=$((fail+1)); fi
    done
  done
done

echo
echo "done: $ok ok / $skip skipped / $fail failed  (of $total)"
