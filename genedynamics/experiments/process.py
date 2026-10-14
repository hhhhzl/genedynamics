"""Run an isolated research stage with bounded logs, time and artifact usage."""
import os
from pathlib import Path
import signal
import subprocess
import time
from .artifacts import active_policy, atomic_json, ResourceExhausted


def run_stage(name, command, output, *, env=None, timeout=3600, max_log_bytes=16 * 1024**2):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policy = active_policy()
    if policy:
        policy.check(2 * 1024**2)
    start = time.monotonic()
    row = {"name": name, "command": list(map(str, command)), "status": "running",
           "log": str(output / (name + ".log"))}
    report_path = output / (name + ".stage.json")
    atomic_json(report_path, row)
    child = None
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError("stage interrupted; charged probes are retained")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        with Path(row["log"]).open("w") as log:
            child = subprocess.Popen(row["command"], stdout=log, stderr=subprocess.STDOUT,
                                     env=env, start_new_session=True)
            while child.poll() is None:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f"stage exceeded {timeout}s")
                if Path(row["log"]).stat().st_size > max_log_bytes:
                    raise ResourceExhausted("stage log quota exhausted")
                if policy:
                    policy.check(2 * 1024**2)
                time.sleep(2)
            row.update(exit_code=child.returncode, status="passed" if child.returncode == 0 else "failed")
    except BaseException as error:
        row.update(status="resource_exhausted" if isinstance(error, ResourceExhausted) else "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                   error_type=type(error).__name__, error=str(error), exit_code=None)
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        signal.signal(signal.SIGTERM, previous)
        row["seconds"] = time.monotonic() - start
        # Tiny emergency status must still be publishable after a quota stop.
        atomic_json(report_path, row, enforce_policy=row["status"] != "resource_exhausted")
    print(f"{name}: {row['status']} ({row['seconds']:.1f}s)", flush=True)
    return row
