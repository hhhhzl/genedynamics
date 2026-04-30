#!/bin/bash
# Sequential execution of all 7 co-design experiments (GPU single-tenant).
# Each run logs to results/soft_robot/run_all_<name>.log; progress is visible
# via `tail -f results/soft_robot/run_all.log`.
set -u

cd /workspace/genedynamics

RUNS=(
  "main:configs/soft_robot/main/crawling_ground.yaml"
  "no_mode_marg:configs/soft_robot/ablation/no_mode_marginalization.yaml"
  "no_fid_env30:configs/soft_robot/ablation/no_fidelity_ladder_env30.yaml"
  "no_fid_env100:configs/soft_robot/ablation/no_fidelity_ladder_env100.yaml"
  "main_smooth:configs/soft_robot/main/crawling_ground_smooth.yaml"
  "cmaes:configs/soft_robot/baselines/cmaes_crawling.yaml"
  "cem:configs/soft_robot/baselines/cem_crawling.yaml"
)

mkdir -p results/soft_robot
MASTER_LOG=results/soft_robot/run_all.log
echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue: ${#RUNS[@]} runs" | tee "$MASTER_LOG"

for entry in "${RUNS[@]}"; do
  NAME="${entry%%:*}"
  CONFIG="${entry#*:}"
  LOG="results/soft_robot/run_all_${NAME}.log"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START ${NAME}  config=${CONFIG}" | tee -a "$MASTER_LOG"
  t0=$(date +%s)
  python scripts/tasks/soft_robot/co_design/main/run_co_design_with_gif.py "$CONFIG" > "$LOG" 2>&1
  rc=$?
  t1=$(date +%s)
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] END   ${NAME}  rc=${rc}  elapsed=$((t1-t0))s" | tee -a "$MASTER_LOG"
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] ALL DONE" | tee -a "$MASTER_LOG"
