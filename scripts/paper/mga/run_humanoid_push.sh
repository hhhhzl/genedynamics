#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
configs=(
  configs/humanoid/push_to_line/main/mga.yaml
  configs/humanoid/push_to_line/baseline/model_based_only.yaml
  configs/humanoid/push_to_line/baseline/standalone_rl.yaml
  configs/humanoid/push_to_line/baseline/dial.yaml
  configs/humanoid/push_to_line/baseline/mppi.yaml
  configs/humanoid/push_to_line/baseline/pegasusflow.yaml
  configs/humanoid/push_to_line/baseline/issa.yaml
  configs/humanoid/push_to_line/baseline/atacom.yaml
)
for config in "${configs[@]}"; do
  python -m genedynamics.experiments.runner "$config" --resume
done

python -m genedynamics.experiments.runner \
  configs/humanoid/push_to_line/ablation/no_rl_prior.yaml --resume \
  --suites p2_push_ood p3_unjam p4_walk_push
python -m genedynamics.experiments.runner \
  configs/humanoid/push_to_line/ablation/no_learned_reliability.yaml --resume \
  --suites p2_push_ood p3_unjam p4_walk_push
python -m genedynamics.experiments.runner \
  configs/humanoid/push_to_line/ablation/no_tangent.yaml --resume \
  --suite p3_unjam
python -m genedynamics.experiments.runner \
  configs/humanoid/push_to_line/ablation/no_retraction.yaml --resume \
  --suite p3_unjam
python -m genedynamics.experiments.runner \
  configs/humanoid/push_to_line/ablation/no_stiffness.yaml --resume \
  --suites p1_force_15n p1_force_30n
