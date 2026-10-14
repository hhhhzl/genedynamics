"""Compare ported policy cores against pinned upstream sources.

Core parity is a P3 prerequisite, not native task or four-task reproduction.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

from scripts.baselines.sources import LOCK, ROOT, verify

TESTS = [
    "test/baselines/test_fada_source_parity.py",
    "test/baselines/test_human2sim2robot_source_parity.py",
    "test/baselines/test_acp_source_parity.py",
    "test/baselines/test_policy_execution.py",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-task-contracts", action="store_true",
                        help="also run real CPU task, learner, asset and lifecycle regressions")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT / "latex" or ROOT / "latex" in output.parents:
        parser.error("qualification reports belong outside latex/")
    output.mkdir(parents=True, exist_ok=True)
    sources = [verify(spec) for spec in json.loads(LOCK.read_text())["sources"]]
    (output / "sources.lock.json").write_bytes(LOCK.read_bytes())
    env = {**os.environ, "GENEDYNAMICS_BASELINE_TEST_DEVICE": args.device,
           "JAX_PLATFORMS": "cpu", "CUDA_VISIBLE_DEVICES": args.gpu if args.device == "cuda" else "",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    tests = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "test/baselines").glob("test_*.py")) if args.include_task_contracts else TESTS
    command = [sys.executable, "-m", "pytest", "-q", *tests,
               f"--junitxml={output / 'parity.xml'}"]
    with (output / "parity.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                stderr=subprocess.STDOUT, timeout=600)
    xml = output / "parity.xml"
    cases = list(ET.parse(xml).getroot().iter("testcase")) if xml.exists() else []
    skipped = sum(case.find("skipped") is not None for case in cases)
    failures = sum(case.find("failure") is not None or case.find("error") is not None for case in cases)
    passed = result.returncode == 0 and bool(cases) and not skipped and not failures
    source_paths = set()
    for directory in ("fada", "human2sim2robot", "acp"):
        source_paths.update(p for p in (ROOT / "genedynamics/solvers/single" / directory).rglob("*")
                            if p.is_file() and (p.suffix in {".py", ".json", ".md"}
                                                or p.name.startswith("LICENSE") or p.name == "NOTICE"))
    source_paths.update((ROOT / "genedynamics/learning/baselines").rglob("*.py"))
    source_paths.update((ROOT / "scripts/baselines").rglob("*.py"))
    source_paths.add(ROOT / "scripts/fada_source_train.py")
    source_paths.update((ROOT / "genedynamics/experiments/baselines").rglob("*.py"))
    if args.include_task_contracts:
        source_paths.update((ROOT / "genedynamics/learning/cross_embodiment").rglob("*.py"))
        source_paths.update((ROOT / "genedynamics/envs/domains/manipulation/cross_embodiment").rglob("*.py"))
        source_paths.add(ROOT / "genedynamics/experiments/artifacts.py")
    source_paths.update(p for p in (ROOT / "requirements/baselines").rglob("*") if p.is_file())
    source_paths.update(ROOT / name for name in tests)
    source_paths.add(ROOT / "genedynamics/solvers/common/policy_execution.py")
    source_paths.update(ROOT / name for name in (
        "genedynamics/solvers/__init__.py", "pyproject.toml", "THIRD_PARTY_NOTICES.md",
    ))
    import torch
    report = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "source_core_parity_passed" if passed else "failed",
        "scope": "same-device upstream/local core parity; native training, assets, task adaptation and physical backend qualification remain separate",
        "includes_task_contracts": args.include_task_contracts,
        "device": args.device, "visible_gpu": env["CUDA_VISIBLE_DEVICES"],
        "python": sys.version,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "tests": len(cases), "skipped": skipped, "failures": failures,
        "returncode": result.returncode, "sources": sources, "command": command,
        "local_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in sorted(source_paths) if p.is_file()},
    }
    (output / "qualification.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "requirements.lock.txt").write_text(subprocess.check_output(
        [sys.executable, "-m", "pip", "freeze", "--all"], text=True,
    ))
    print(json.dumps({key: report[key] for key in ("status", "device", "tests", "skipped", "failures")}, indent=2))
    if not passed:
        raise SystemExit(f"source parity failed; see {output / 'parity.log'}")


if __name__ == "__main__":
    main()
