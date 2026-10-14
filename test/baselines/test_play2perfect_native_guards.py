"""The prepared native runner must never grant itself EULA/startup permission."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "results/_development/p3_20261014/results/dexterous/tight_insertion/qualification/play2perfect_native/level_source/seed_0/assets/tight_insertion_model.pth"


@pytest.fixture
def native_command(tmp_path):
    if not CHECKPOINT.is_file(): pytest.skip("published native checkpoint not fetched")
    return [sys.executable, "-m", "scripts.baselines.play2perfect_native_rollout", "--checkpoint", str(CHECKPOINT),
            "--output", str(tmp_path / "native"), "--gpu", "1"]


def test_unaccepted_eula_rejects_before_any_isaac_import_or_attempt(native_command, tmp_path):
    result = subprocess.run(native_command, cwd=ROOT, env=dict(os.environ, OMNI_KIT_ACCEPT_EULA="NO"), capture_output=True, text=True)
    assert result.returncode != 0
    assert "EULA has not been explicitly accepted" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not (tmp_path / "native/native_attempt.json").exists()
    assert not (tmp_path / "native/native_closed_loop.json").exists()


def test_plan_only_is_concrete_bounded_gpu1_and_does_not_start_native_attempt(native_command, tmp_path):
    result = subprocess.run(native_command + ["--plan-only"], cwd=ROOT,
                            env=dict(os.environ, OMNI_KIT_ACCEPT_EULA="NO"), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    plan = json.loads((tmp_path / "native/native_rollout_plan.json").read_text())
    assert plan["AppLauncher"]["device"] == plan["actor_device"] == plan["env_device"] == "cuda:0"
    assert plan["physical_gpu"] == 1
    assert "UUID" in plan["cuda_visibility"]
    assert plan["AppLauncher"]["multi_gpu"] is False
    assert plan["seed"] == 0 and plan["coefficient_mode"] == "exploitation"
    assert all(row["coefficient"] == 0 and row["resolved_parameter_id"] == 5 and row["exact_match"]
               for row in plan["coefficient_assignment"])
    assert plan["limits"] == {"wall_seconds_including_setup": 600, "owned_gpu_mib": 10240, "owned_rss_gib": 20}
    assert "--portable-root" in plan["Kit_args"]
    assert not (tmp_path / "native/native_attempt.json").exists()


def test_native_full_success_requires_terminal_retract_and_every_trial_stays_in_denominator():
    from scripts.baselines.play2perfect_native_rollout import native_fixed_goal_outcome, full_denominator_summary

    full = native_fixed_goal_outcome(2, 2, retract_enabled=True, terminal_retract_success=True)
    insertion_only = native_fixed_goal_outcome(2, 2, retract_enabled=True, terminal_retract_success=False)
    early_drop = native_fixed_goal_outcome(0, 0, retract_enabled=True, terminal_retract_success=False)
    assert full["success"] and full["insertion_success"]
    assert insertion_only["insertion_success"] and not insertion_only["success"]
    assert not early_drop["success"]
    rows = [dict(ended=True, **row) for row in (full, insertion_only, early_drop)]
    rows.append(dict(ended=False, success=False))
    result = full_denominator_summary(rows, 4)
    assert result["requested_trials"] == 4 and result["completed_trials"] == 3
    assert result["unfinished_trials"] == 1 and result["insertion_successes"] == 2
    assert result["successes"] == 1 and result["full_denominator_success_rate"] == .25
    with pytest.raises(RuntimeError, match="terminal retract snapshot"):
        native_fixed_goal_outcome(2, 2, retract_enabled=True, terminal_retract_success=None)


def test_source_mixed_rejects_unmatched_eight_slot_upstream_coefficients(native_command):
    from scripts.baselines.play2perfect_native_rollout import coefficient_assignment

    original_eight = coefficient_assignment([50 - 50 * i / 7 for i in range(8)], [50, 40, 30, 20, 10, 0])
    assert sum(row["exact_match"] for row in original_eight) == 2
    assert all(row["resolved_parameter_id"] == 0 for row in original_eight[1:-1])
    bad = subprocess.run(native_command + ["--plan-only", "--coefficient-mode", "source-mixed"],
                         cwd=ROOT, capture_output=True, text=True)
    assert bad.returncode != 0 and "must exactly match trained IDs" in bad.stderr
    good = subprocess.run(native_command + ["--plan-only", "--coefficient-mode", "source-mixed", "--num-envs", "6"],
                          cwd=ROOT, capture_output=True, text=True)
    assert good.returncode == 0, good.stderr
    assert [row["resolved_parameter_id"] for row in json.loads(good.stdout)["coefficient_assignment"]] == list(range(6))


def test_watchdog_tool_timeout_retains_every_requested_trial_and_attempt(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.baselines import play2perfect_native_rollout as native

    args = SimpleNamespace(output=tmp_path, gpu=1, num_envs=4, max_seconds=600, max_gpu_mib=10240, max_rss_gib=20)
    class Worker:
        pid = 99999999
        returncode = None
        def poll(self): return self.returncode
        def wait(self, timeout=None): self.returncode = -15
    worker = Worker()
    # This early failure occurs before process-tree sampling; core test runtime
    # need not install the native runtime's psutil dependency.
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace())
    monkeypatch.setattr(native.subprocess, "Popen", lambda *args, **kwargs: worker)
    def hung_tool(*args, **kwargs):
        assert kwargs["timeout"] == 5
        raise subprocess.TimeoutExpired("nvidia-smi", 5)
    monkeypatch.setattr(native.subprocess, "check_output", hung_tool)
    monkeypatch.setattr(native.os, "killpg", lambda *args: None)
    with pytest.raises(RuntimeError, match="Native attempt failed"):
        native.launch_bounded(args, {}, {"schema": "unit-test-without-native-startup"})
    attempt = json.loads((tmp_path / "native_attempt.json").read_text())
    partial = json.loads((tmp_path / "native_partial_closed_loop.json").read_text())
    assert attempt["watchdog_exception"]["type"] == "TimeoutExpired"
    assert attempt["worker_returncode"] == -15
    assert partial["requested_trials"] == partial["unfinished_trials"] == 4
    assert partial["completed_trials"] == partial["successes"] == 0
    assert all(row["status"] == "not_started" and row["censored"] for row in partial["trials"])
    with pytest.raises(RuntimeError, match="cannot be reused"):
        native.launch_bounded(args, {}, {})


def test_vulkan_active_uuid_is_verified_independently_of_cuda_ordinal(tmp_path):
    from scripts.baselines.play2perfect_native_rollout import verify_vulkan_gpu_log
    selected = "GPU-58b54684-3577-11d3-f039-0e8469190739"
    path = tmp_path / "kit.log"
    path.write_text("| GPU | Name | Active | LDA | Memory | Vendor-ID | LUID |\n"
                    "| 0 | RTX 4500 Ada | No | | 24576 MB | 10de | 0 |\n"
                    "| | | | | | 27b1 | 253f0813-4432-2d51-a149-3e73b38bf77a |\n"
                    "| 1 | RTX 4500 Ada | Yes:0 | | 24576 MB | 10de | 0 |\n"
                    "| | | | | | 27b1 | 58b54684-3577-11d3-f039-0e8469190739 |\n")
    assert verify_vulkan_gpu_log(path, selected)["verified"]
    with pytest.raises(RuntimeError, match="Vulkan"):
        verify_vulkan_gpu_log(path, "GPU-253f0813-4432-2d51-a149-3e73b38bf77a")
    path.write_text("| 1 | RTX 4500 Ada | Yes: 0 | | 24816 MB | 10de | 0 |\n"
                    "| | | | | | 27b1 | 58b54684.. |\n"
                    "| | | | | | 82 | |\n")
    assert verify_vulkan_gpu_log(path, selected, expected_pci_bus_id="00000000:82:00.0")["verified"]
    with pytest.raises(RuntimeError, match="Vulkan"):
        verify_vulkan_gpu_log(path, selected, expected_pci_bus_id="00000000:41:00.0")


def test_worker_exit_zero_without_rollout_evidence_remains_failed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.baselines import play2perfect_native_rollout as native
    args = SimpleNamespace(output=tmp_path, gpu=1, num_envs=8, max_seconds=600, max_gpu_mib=10240, max_rss_gib=20)
    worker = SimpleNamespace(pid=99999999, returncode=0, poll=lambda: 0)
    monkeypatch.setitem(sys.modules, "psutil", SimpleNamespace())
    monkeypatch.setattr(native.subprocess, "Popen", lambda *a, **kw: worker)
    monkeypatch.setattr(native.subprocess, "check_output", lambda *a, **kw: "1, GPU-selected\n")
    with pytest.raises(RuntimeError, match="Native attempt failed"):
        native.launch_bounded(args, {}, {})
    attempt = json.loads((tmp_path / "native_attempt.json").read_text())
    assert attempt["status"] == "worker_failed" and attempt["worker_returncode"] == 0
    partial = json.loads((tmp_path / "native_partial_closed_loop.json").read_text())
    assert partial["requested_trials"] == partial["unfinished_trials"] == 8


def test_error_is_persisted_before_kit_close_can_exit_zero(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from scripts.baselines import play2perfect_native_rollout as native
    import torch
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: None)
    class App:
        def close(self): raise SystemExit(0)
    monkeypatch.setitem(sys.modules, "isaaclab.app", SimpleNamespace(AppLauncher=lambda **kw: SimpleNamespace(app=App())))
    def fail(*a): raise ValueError("setup failure before teardown")
    monkeypatch.setattr(native, "_run_with_app", fail)
    state = {"status": "running"}
    saved = []
    with pytest.raises(SystemExit):
        native._run_native_worker(SimpleNamespace(runtime=tmp_path), {}, state, lambda: saved.append(dict(state)))
    assert saved[-1]["status"] == "failed"
    assert saved[-1]["exception"]["message"] == "setup failure before teardown"
