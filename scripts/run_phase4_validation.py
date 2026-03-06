#!/usr/bin/env python3
"""
Phase 4: Closed-loop validation and fair comparison.

Runs MBD, MD-COAS, and three quadruped tasks. Produces a validation report.
Uses .venv_arm64 by default. Set VENV env for custom Python.

Run: python scripts/run_phase4_validation.py
Or:  VENV=conda_run_fedguide python scripts/run_phase4_validation.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VENV = os.environ.get("VENV", str(ROOT / ".venv_arm64" / "bin" / "python"))


def run_deploy(config: str, episodes: int = 1, max_steps: int = 20) -> tuple[bool, float, str]:
    """Run deploy and return (success, elapsed_sec, output_tail)."""
    cmd = [
        VENV, "-m", "genedynamics.deploy.cli",
        "--config", str(ROOT / config),
        "--episodes", str(episodes),
        "--max-steps", str(max_steps),
        "--no-record",
    ]
    t0 = time.time()
    try:
        r = subprocess.run(
            cmd,
            cwd=str(ROOT),
            capture_output=True,
            text=True,
            timeout=300,
        )
        elapsed = time.time() - t0
        out = (r.stdout + r.stderr).strip()
        tail = "\n".join(out.split("\n")[-8:]) if out else ""
        return r.returncode == 0, elapsed, tail
    except subprocess.TimeoutExpired:
        return False, 300.0, "TIMEOUT"
    except Exception as e:
        return False, time.time() - t0, str(e)


def main() -> int:
    print("=== Phase 4: Closed-loop validation and fair comparison ===\n")
    print(f"Python: {VENV}")
    print(f"Root:   {ROOT}\n")

    tests = [
        ("MBD deploy (baseline)", "configs/quadruped/flat/mbd_deploy_quick.yaml"),
        ("MD-COAS deploy (level 0, fair)", "configs/quadruped/flat/mdcoas_deploy_quick.yaml"),
        ("obstacle_avoid task", "configs/quadruped/obstacle_avoid/mdcoas_deploy_quick.yaml"),
        ("rough_terrain task", "configs/quadruped/rough_terrain/mbd_deploy_quick.yaml"),
        ("push_recovery task", "configs/quadruped/push_recovery/mbd_deploy_quick.yaml"),
    ]

    results = []
    for name, config in tests:
        print(f">>> {name}")
        ok, elapsed, tail = run_deploy(config)
        results.append((name, ok, elapsed))
        status = "OK" if ok else "FAIL"
        print(f"    {status} ({elapsed:.1f}s)")
        if not ok and tail:
            print(f"    ---\n{tail}\n---")
        print()

    # Summary
    print("=== Summary ===")
    all_ok = all(r[1] for r in results)
    for name, ok, elapsed in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name} ({elapsed:.1f}s)")
    print()
    if all_ok:
        print("All Phase 4 validation tests passed.")
        return 0
    print("Some tests failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
