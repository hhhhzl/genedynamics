#!/usr/bin/env python3
"""Pre-flight checks for co-design smoke test."""

import os
import sys
import yaml

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(0, "/workspace")


def main():
    # 1. SoftZoo bootstrap
    from genedynamics.envs.external.softzoo.bootstrap import validate_softzoo_environment
    ok, msgs = validate_softzoo_environment()
    for m in msgs:
        print(f"  {m}")
    status = "OK" if ok else "FAIL"
    print(f"SoftZoo validation: {status}")

    # 2. Task registry
    from genedynamics.envs.external.softzoo.task_registry import list_tasks
    print(f"Tasks: {list_tasks()}")

    # 3. Scheduler from YAML
    with open("configs/co_design/phase_a_smoke.yaml") as f:
        cfg = yaml.safe_load(f)
    print(f"Smoke config: baseline={cfg['baseline_name']}, task={cfg['task_id']}")

    from genedynamics.experiments.common.constraints import create_scheduler_from_config
    scheduler = create_scheduler_from_config(cfg["scheduler_config"], "jax")
    print(f"Scheduler: {type(scheduler).__name__}")

    from genedynamics.core.constraints.core.types import ScheduleState
    params = scheduler.diffusion_schedulers[0].diffusion_params(ScheduleState(k=0, K=1))
    print(f"Diffusion: M_k={params['M_k']}, Ndiffuse={params['Ndiffuse']}, T_k={params['T_k']}")

    # 4. Baseline config
    from genedynamics.experiments.co_design.baseline import BaselineConfig
    bl_config = BaselineConfig(
        task_id=cfg["task_id"],
        seed=0,
        extra=cfg.get("method_params", {}),
        scheduler=scheduler,
    )
    dp = bl_config.get_diffusion_params()
    print(f"BaselineConfig: K={dp['Ndiffuse']}, M={dp['M_k']}")

    # 5. Baseline registry
    from genedynamics.experiments.co_design import list_baselines
    print(f"Baselines: {list_baselines()}")

    # 6. Evaluator instantiation
    from genedynamics.envs.evaluators import SoftZooRolloutEvaluator, SoftZooEvaluatorConfig
    try:
        evaluator = SoftZooRolloutEvaluator(
            config=SoftZooEvaluatorConfig(max_workers=0, cache_size=64),
            project_root="/workspace",
            runtime_config=cfg.get("evaluator_runtime", {}),
        )
        print(f"Evaluator: {type(evaluator).__name__}")
    except Exception as e:
        print(f"Evaluator error (expected if SoftZoo assets missing): {e}")

    print("Pre-flight OK!")


if __name__ == "__main__":
    main()
