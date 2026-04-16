#!/usr/bin/env python3
"""
Run co-design smoke test (phase_a_smoke.yaml).

Validates the full MRMFMBD pipeline end-to-end:
  YAML → scheduler → platform → baseline → evaluator → SoftZoo rollout → result
"""

import os
import sys
import yaml
import json
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Resolve project root: script lives at scripts/tasks/soft_robot/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
sys.path.insert(0, _PROJECT_ROOT)
os.chdir(_PROJECT_ROOT)


def main():
    from genedynamics.experiments.framework.baseline_platform import (
        BaselineExperimentConfig,
        BaselineExperimentPlatform,
    )

    config_path = os.path.join(_PROJECT_ROOT, "configs/co_design/phase_a_smoke.yaml")
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    # Import baselines to trigger registration
    import genedynamics.experiments.framework.baselines  # noqa: F401

    cfg = BaselineExperimentConfig(
        baseline_name=raw["baseline_name"],
        task_domain=raw.get("task_domain", "softzoo"),
        task_id=raw.get("task_id", "crawling_ground"),
        seeds=raw.get("seeds", [0]),
        output_dir=raw.get("output_dir", "results/phase_a/smoke"),
        cache_dir=raw.get("cache_dir", raw.get("checkpoint_dir")),
        scheduler_config=raw.get("scheduler_config"),
        method_params=raw.get("method_params", {}),
        baseline_params=raw.get("baseline_params", {}),
        evaluator_params=raw.get("evaluator_runtime", raw.get("evaluator_params", {})),
        save_gif=raw.get("save_gif", False),
    )

    print(f"Config: baseline={cfg.baseline_name}, task={cfg.task_id}, seeds={cfg.seeds}")
    print(f"Scheduler: {cfg.scheduler_config is not None}")
    print(f"Method params: {cfg.method_params}")

    platform = BaselineExperimentPlatform(
        cfg,
        project_root=_PROJECT_ROOT,
        use_checkpointing=False,
        use_logging=True,
    )

    print("Running smoke test...")
    try:
        results = platform.run_all()
        print(f"Completed {len(results)} seed(s)")
        for r in results:
            res = r.get("result", {})
            print(f"  seed={r['seed']}: return={res.get('return_', '?'):.4f}, "
                  f"wall_time={r['wall_time']:.2f}s, "
                  f"success={res.get('success', '?')}")
            # Print bridge history summary
            bh = res.get("bridge_history", res.get("metadata", {}).get("bridge_history", []))
            if bh and isinstance(bh, list):
                steps = [s for s in bh if isinstance(s, dict)]
                if steps:
                    print(f"    bridge_steps={len(steps)}, "
                          f"mean_reward={steps[-1].get('mean_reward', '?')}, "
                          f"final_ess={steps[-1].get('ess', '?')}")
        print("SMOKE TEST PASSED")
    except Exception as e:
        print(f"SMOKE TEST FAILED: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
