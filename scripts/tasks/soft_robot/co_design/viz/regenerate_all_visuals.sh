#!/bin/bash
# Re-render all visual outputs for the 7 completed runs:
#   - crawling_best.gif / .summary.png    (3D, via --skip-train)
#   - morphology.png, controller_heatmap.png, cross_mode_robustness.{png,csv}
#   - For diffusion runs: reward_vs_k.png, fidelity_budget.png,
#     diffusion_evolution.png, diffusion_evolution.gif
#   - For baselines: reward_vs_generation.png
#
# Does NOT retrain — operates purely on saved results.json.
set -u
cd /workspace/genedynamics

DIFF_CONFIGS=(
  "configs/soft_robot/main/crawling_ground.yaml"
  "configs/soft_robot/main/crawling_ground_smooth.yaml"
  "configs/soft_robot/ablation/no_mode_marginalization.yaml"
  "configs/soft_robot/ablation/no_fidelity_ladder_env30.yaml"
  "configs/soft_robot/ablation/no_fidelity_ladder_env100.yaml"
)
BASELINE_CONFIGS=(
  "configs/soft_robot/baselines/cmaes_crawling.yaml"
  "configs/soft_robot/baselines/cem_crawling.yaml"
)

echo "=== Regenerating 3D crawling_best.gif for ALL runs ==="
for cfg in "${DIFF_CONFIGS[@]}" "${BASELINE_CONFIGS[@]}"; do
  echo "[gif] $cfg"
  python scripts/tasks/soft_robot/co_design/main/run_co_design_with_gif.py "$cfg" --skip-train \
    > /tmp/regen_gif.log 2>&1 || { echo "  FAILED — tail log:"; tail -5 /tmp/regen_gif.log; }
done

echo ""
echo "=== Diffusion runs: re-render all analysis figures ==="
python scripts/tasks/soft_robot/co_design/analysis/analyze_diffusion_run.py "${DIFF_CONFIGS[@]}" 2>&1 | \
  grep -E "^\s*\[|cross-mode|diffusion_evolution|^===" | head -60

echo ""
echo "=== Baseline runs: morphology + controller + reward curves ==="
cd /workspace/genedynamics/scripts/tasks/soft_robot/co_design/analysis
PYTHONPATH="/workspace/genedynamics/scripts/tasks/soft_robot/co_design/analysis:$PYTHONPATH" \
  python analyze_baseline_run.py \
    /workspace/genedynamics/configs/soft_robot/baselines/cmaes_crawling.yaml \
    /workspace/genedynamics/configs/soft_robot/baselines/cem_crawling.yaml \
  2>&1 | grep -E "^\s*\[|cross-mode|reward_vs|^===" | head -40

echo ""
echo "ALL VISUAL REGENERATION COMPLETE"
