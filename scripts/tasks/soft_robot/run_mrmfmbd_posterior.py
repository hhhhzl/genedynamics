#!/usr/bin/env python3
"""
Run MRMFMBD posterior bridge (theta = x, phi co-design).

Uses SoftZooRolloutEvaluator and MRMFMBDPosteriorBackendJax for
true posterior bridge with S1 mode marginalization and S3 multi-fidelity.

Example:
  python scripts/tasks/soft_robot/run_mrmfmbd_posterior.py --K 20 --M 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description="MRMFMBD posterior bridge")
    parser.add_argument("--K", type=int, default=20, help="Bridge steps")
    parser.add_argument("--M", type=int, default=4, help="Proposals per step")
    parser.add_argument("--task", type=str, default="crawling_ground")
    parser.add_argument("--x-dim", type=int, default=3, help="Morphology dim (e.g. geometry, softness, actuator)")
    parser.add_argument("--phi-dim", type=int, default=4, help="Controller dim (e.g. omega params)")
    parser.add_argument("--num-modes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="results/mrmfmbd_posterior")
    parser.add_argument("--show-tqdm", action="store_true")
    args = parser.parse_args()

    # Add project root
    root = Path(__file__).resolve().parents[3]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from genedynamics.envs.evaluators import SoftZooRolloutEvaluator, SoftZooEvaluatorConfig
    from genedynamics.solvers.single.mrmfmbd import (
        ThetaParametrization,
        ThetaPrior,
        ThetaPriorConfig,
        MRMFMBDPosteriorBackendJax,
        PosteriorBridgeConfig,
        ModeMarginalizerS1,
        default_mode_system_config,
        create_fidelity_ladder,
    )

    evaluator = SoftZooRolloutEvaluator(
        config=SoftZooEvaluatorConfig(max_workers=0, cache_size=64),
        project_root=str(root),
    )

    theta_param = ThetaParametrization(
        x_dim=args.x_dim,
        phi_dim=args.phi_dim,
        x_bounds=(0.01, 2.0),
        phi_bounds=(5.0, 150.0),
    )
    theta_prior = ThetaPrior(theta_param, ThetaPriorConfig(x_std=0.5, phi_std=30.0))

    mode_marginalizer = ModeMarginalizerS1(
        default_mode_system_config(args.num_modes),
        backend="jax",
    )
    fidelity_ladder = create_fidelity_ladder(
        K=args.K,
        num_levels=3,
        ladder_type="geometric",
        step_ratio=1.5,
    )

    backend = MRMFMBDPosteriorBackendJax(
        evaluator=evaluator,
        theta_param=theta_param,
        theta_prior=theta_prior,
        mode_marginalizer=mode_marginalizer,
        fidelity_ladder=fidelity_ladder,
        config=PosteriorBridgeConfig(
            K=args.K,
            M=args.M,
            reward_temperature=0.1,
            top_k_fine=2,
            fine_fidelity_level=2,
        ),
        task_id=args.task,
        num_modes=args.num_modes,
        seed=args.seed,
        show_tqdm=args.show_tqdm,
    )

    print("Running MRMFMBD posterior bridge...")
    result = backend.plan()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "result.json"

    # Serialize for JSON (convert numpy)
    def _to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _to_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_to_serializable(v) for v in obj]
        return obj

    with open(out_path, "w") as f:
        json.dump(_to_serializable(result), f, indent=2)

    print(f"Result saved to {out_path}")
    print(f"  theta: {result['theta'][:8]}...")
    print(f"  wall_clock: {result['wall_clock']:.2f}s")
    print(f"  fidelity_history: {result['fidelity_history'][:5]}...")
    return 0


if __name__ == "__main__":
    sys.exit(main())
