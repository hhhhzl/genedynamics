"""Plan or run the task-local T1 AdaptSim development smoke."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from genedynamics.tasks.transfer.obstacle_policy_search import (
    DEVELOPMENT_VERSION, META_DEVELOPMENT_VERSION, NOMINAL_DIAGNOSIS_VERSION,
    NOMINAL_EVIDENCE_VERSION, SOURCE_NOMINAL_EVIDENCE_VERSION,
    development_plan, nominal_diagnosis_plan,
    run_adaptsim_development, run_adaptsim_nominal_diagnosis,
)
from genedynamics.tasks.transfer.mating_policy_search import (
    DEVELOPMENT_VERSION as MATING_DEVELOPMENT_VERSION,
    POPULATION_DEVELOPMENT_VERSION as MATING_POPULATION_DEVELOPMENT_VERSION,
    development_plan as mating_development_plan,
    run_development as run_mating_development,
)
from genedynamics.tasks.transfer.mobile_mating_policy_search import (
    VERSION as MOBILE_FEASIBILITY_VERSION,
    TRAINING_VERSION as MOBILE_TRAINING_VERSION,
    audit_feasibility as audit_mobile_feasibility,
    feasibility_plan as mobile_feasibility_plan,
    run_feasibility as run_mobile_feasibility,
    training_plan as mobile_training_plan,
    run_training as run_mobile_training,
)
from genedynamics.tasks.transfer.guide_policy_search import (
    VERSION as GUIDE_FEASIBILITY_VERSION,
    TRAINING_VERSION as GUIDE_TRAINING_VERSION,
    audit_feasibility as audit_guide_feasibility,
    feasibility_plan as guide_feasibility_plan,
    run_feasibility as run_guide_feasibility,
    training_plan as guide_training_plan,
    run_training as run_guide_training,
)


ROOT = Path(__file__).resolve().parents[3]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-only", action="store_true")
    mode.add_argument("--run", action="store_true")
    mode.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    path = args.config if args.config.is_absolute() else ROOT / args.config
    config = json.loads(path.read_text())
    if args.audit and config.get("version") not in {
        MOBILE_FEASIBILITY_VERSION, GUIDE_FEASIBILITY_VERSION,
    }:
        raise ValueError("audit is available only for task-policy feasibility")
    if config.get("version") in {DEVELOPMENT_VERSION, META_DEVELOPMENT_VERSION}:
        result = development_plan(config, ROOT) if args.plan_only else run_adaptsim_development(config, ROOT)
    elif config.get("version") in {
        NOMINAL_DIAGNOSIS_VERSION, NOMINAL_EVIDENCE_VERSION,
        SOURCE_NOMINAL_EVIDENCE_VERSION,
    }:
        result = nominal_diagnosis_plan(config, ROOT) if args.plan_only else run_adaptsim_nominal_diagnosis(config, ROOT)
    elif config.get("version") in {
        MATING_DEVELOPMENT_VERSION, MATING_POPULATION_DEVELOPMENT_VERSION,
    }:
        result = mating_development_plan(config, ROOT) if args.plan_only else run_mating_development(config, ROOT)
    elif config.get("version") == MOBILE_FEASIBILITY_VERSION:
        result = (
            mobile_feasibility_plan(config, ROOT) if args.plan_only else
            audit_mobile_feasibility(config, ROOT) if args.audit else
            run_mobile_feasibility(config, ROOT)
        )
    elif config.get("version") == MOBILE_TRAINING_VERSION:
        result = mobile_training_plan(config, ROOT) if args.plan_only else run_mobile_training(config, ROOT)
    elif config.get("version") == GUIDE_FEASIBILITY_VERSION:
        result = (
            guide_feasibility_plan(config, ROOT) if args.plan_only else
            audit_guide_feasibility(config, ROOT) if args.audit else
            run_guide_feasibility(config, ROOT)
        )
    elif config.get("version") == GUIDE_TRAINING_VERSION:
        result = (
            guide_training_plan(config, ROOT) if args.plan_only else
            run_guide_training(config, ROOT)
        )
    else:
        raise ValueError("unknown AdaptSim task-local recipe version")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
