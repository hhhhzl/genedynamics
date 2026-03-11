#!/usr/bin/env python3
"""
Verify SoftZoo installation and environment.

Checks: code root, configs, assets, PCD, and a minimal rollout.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    print("=== SoftZoo Verification ===\n")

    # 1. Path resolution
    from genedynamics.envs.external.softzoo.bootstrap import (
        validate_softzoo_environment,
        ensure_softzoo_on_path,
    )

    ok, msgs = validate_softzoo_environment(str(ROOT), require_pcd="Caterpillar")
    for m in msgs:
        print(f"  {m}")
    if not ok:
        print("\n  PCD required for crawling. Download from:")
        print("  https://drive.google.com/drive/folders/1AYeZsr2ZMb1DkeOndQM0nBlNfx7dorUL")
        print("  Place Caterpillar.pcd in data/softzoo/assets/meshes/pcd/")
        print("\n  Continuing with path-only check (rollout may return 0)...\n")

    # 2. Import softzoo
    ensure_softzoo_on_path(str(ROOT))
    try:
        import softzoo  # noqa: F401
        print("  softzoo import: OK")
    except ImportError as e:
        print(f"  softzoo import: FAIL - {e}")
        return 1

    # 3. Single rollout
    from genedynamics.envs.evaluators import SoftZooRolloutEvaluator, SoftZooEvaluatorConfig
    from genedynamics.envs.evaluators.protocols import RolloutBatchRequest, RolloutRequest
    import numpy as np

    eval_cfg = SoftZooEvaluatorConfig(max_workers=0, cache_size=0)
    evaluator = SoftZooRolloutEvaluator(config=eval_cfg, project_root=str(ROOT))
    req = RolloutBatchRequest(
        task_id="crawling_ground",
        requests=[
            RolloutRequest(
                morphology_params=np.array([1.0, 1.0, 1.0]),
                controller_params=np.ones(20) * 10.0,
                mode_id=0,
                fidelity_level=0,
                seed=0,
                num_repeats=1,
                record=False,
            )
        ],
    )
    res = evaluator.evaluate_batch(req, parallel=False)
    r = res.results[0]
    print(f"  Single rollout: return={r.return_:.4f}, success={r.success}")
    if r.failure_code:
        print(f"  Failure code: {r.failure_code}")

    print("\nNote: return=0, success=False expected without Caterpillar.pcd.")
    print("Download PCD from https://drive.google.com/drive/folders/1AYeZsr2ZMb1DkeOndQM0nBlNfx7dorUL")
    print("Place in data/softzoo/assets/meshes/pcd/Caterpillar.pcd")
    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
