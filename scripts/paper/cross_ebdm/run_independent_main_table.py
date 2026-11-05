"""Plan or execute the sealed GPU0 independent four-task main table."""

import argparse
import json
import os
from pathlib import Path

from genedynamics.tasks.transfer.main_table_evaluation import GPU0_UUID
from genedynamics.tasks.transfer.main_table_execution import (
    ExecutionSettings,
    plan,
    run,
)
from scripts.paper.cross_ebdm.assemble_independent_main_table import assemble
from scripts.paper.cross_ebdm.audit_main_table_admission import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan-only", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument(
        "--output", type=Path,
        default=Path("results/_analysis/four_task_transfer/independent_main_table_execution_v3"),
    )
    parser.add_argument(
        "--table-output", type=Path,
        default=Path("results/_analysis/four_task_transfer/independent_main_table_formal_v3"),
    )
    parser.add_argument("--site-packages", type=Path)
    parser.add_argument("--temporary-directory", type=Path)
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
    )
    parser.add_argument("--physical-gpu-uuid", default=GPU0_UUID)
    parser.add_argument("--maximum-job-seconds", type=int, default=43_200)
    args = parser.parse_args()
    settings = ExecutionSettings(
        output=args.output,
        table_output=args.table_output,
        site_packages=args.site_packages,
        temporary_directory=args.temporary_directory,
        cuda_visible_devices=args.cuda_visible_devices,
        physical_gpu_uuid=args.physical_gpu_uuid,
        maximum_job_seconds=args.maximum_job_seconds,
    )
    admission = audit()
    if args.plan_only:
        prepared = plan(admission, settings)
        print(json.dumps({
            "status": prepared["status"], "jobs": len(prepared["jobs"]),
            "physical_outcomes": prepared["expected_physical_outcomes"],
            "physical_gpu_index": prepared["physical_gpu_index"],
            "sources": len(prepared["shared_source_and_task_config_sha256"]),
        }, indent=2))
    else:
        run(admission, settings, assemble)


if __name__ == "__main__":
    main()
