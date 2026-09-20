#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

# macOS installations commonly expose only ``python3`` while the CPU image
# exposes ``python``.  Keep one paper entry point portable without changing
# the runner or introducing a second execution path.
if [[ -n "${MGA_PYTHON:-}" ]]; then
  python_bin="$MGA_PYTHON"
elif command -v python >/dev/null 2>&1; then
  python_bin="python"
elif command -v python3 >/dev/null 2>&1; then
  python_bin="python3"
else
  echo "No Python interpreter found (set MGA_PYTHON)" >&2
  exit 127
fi

# The default invocation remains the complete formal matrix.  The isolated
# P1--P3 gate uses the same canonical YAMLs and runner; only the requested
# suite/seed/output selectors differ, so no development algorithm fork is
# hidden in this paper entry point.
scope="${MGA_HUMANOID_SCOPE:-all}"
if [[ "$scope" != "all" && "$scope" != "p123" ]]; then
  echo "MGA_HUMANOID_SCOPE must be 'all' or 'p123'" >&2
  exit 2
fi
common_args=(--resume)
if [[ -n "${MGA_DEVELOPMENT_ROOT:-}" ]]; then
  common_args+=(--development-root "$MGA_DEVELOPMENT_ROOT")
fi
if [[ -n "${MGA_SEEDS:-}" ]]; then
  read -r -a requested_seeds <<< "$MGA_SEEDS"
  common_args+=(--seeds "${requested_seeds[@]}")
fi
if [[ "${MGA_DRY_RUN:-0}" == "1" ]]; then
  common_args+=(--dry-run)
fi

primary_suite_args=()
if [[ "$scope" == "p123" ]]; then
  primary_suite_args=(--suites
    p1_force_15n p1_force_30n
    p2_push_nominal p2_push_ood p3_unjam)
fi
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
  "$python_bin" -m genedynamics.experiments.runner "$config" \
    "${common_args[@]}" "${primary_suite_args[@]}"
done

aligned_ablation_configs=(
  configs/humanoid/push_to_line/ablation/no_rl_prior.yaml
  configs/humanoid/push_to_line/ablation/no_learned_reliability.yaml
  configs/humanoid/push_to_line/ablation/no_controllability_geometry.yaml
  configs/humanoid/push_to_line/ablation/no_retraction.yaml
)
for config in "${aligned_ablation_configs[@]}"; do
  "$python_bin" -m genedynamics.experiments.runner "$config" \
    "${common_args[@]}" "${primary_suite_args[@]}"
done
# P1 keeps the task-specific impedance ablation in addition to the four
# cross-task ablations above.
"$python_bin" -m genedynamics.experiments.runner \
  configs/humanoid/push_to_line/ablation/no_stiffness.yaml "${common_args[@]}" \
  --suites p1_force_15n p1_force_30n
