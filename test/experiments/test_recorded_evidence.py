"""Synthetic persistence regressions: no simulator, policy, or GPU execution."""
import gzip
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from genedynamics.core.types import Trajectory
from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.experiments.framework.experiment import ExperimentRunner


def _runner(tmp_path, seeds=(0, 1)):
    config = ExperimentConfig(
        name="recorded-evidence", output_dir=tmp_path, env_name="synthetic",
        method="synthetic", backend="numpy", seeds=list(seeds),
        metrics=["task"], visualizations=[], auto_report=False,
        metadata={"protocol": "synthetic-evidence-v1"},
    )
    runner = ExperimentRunner(config)
    runner._save_protocol_manifest = lambda: None
    runner.registry.get_plugin = lambda *args: SimpleNamespace(
        execution_failure_metrics={"success": 0.},
    )
    return runner


def _result(runner, seed, *, aborted=False):
    status = ({"state": "aborted_unrecoverable", "reason": "synthetic partial failure",
               "error_type": "RuntimeError", "executed_control_steps": 1}
              if aborted else {})
    trajectory = Trajectory(
        states=[np.array([1., 2., 3.]), np.array([2., 3., 4.])],
        actions=[np.array([.1])], info={},
    )
    result = {
        "level": 0, "seed": seed, "planning_time": 1.,
        "config_snapshot": runner.config.to_dict(),
        "metrics": {"task": {"success": float(not aborted)}},
        "result": {
            "trajectory": trajectory, "state_layout": "public_flat",
            "state_fields": [{"name": "position", "shape": [3], "start": 0, "stop": 3}],
            "recorded_evidence": {"episodes": {
                "requested_mode": 1, "physical_steps": 1,
                "failure_value": {"__nonfinite_float__": "nan"},
                "qpos": np.array([1., 2.]),
            }},
        },
    }
    if aborted:
        result["execution_status"] = status
        result["result"]["execution_status"] = status
    return result


def test_flat_trajectory_values_and_plugin_layout_survive_persistence(tmp_path):
    runner = _runner(tmp_path)
    saved = runner._save_result(_result(runner, 0, aborted=True))
    trajectory = json.loads((tmp_path / "level_0/seed_0/trajectory/trajectory.json").read_text())
    assert trajectory["states"] == [[1., 2., 3.], [2., 3., 4.]]
    assert trajectory["state_layout"] == "public_flat"
    assert trajectory["state_fields"][0]["name"] == "position"
    assert trajectory["execution_status"]["executed_control_steps"] == 1
    pointer = saved["recorded_evidence"]["episodes"]
    compressed = (tmp_path / "level_0/seed_0/trajectory/episodes.json.gz").read_bytes()
    assert hashlib.sha256(compressed).hexdigest() == pointer["sha256"]
    decoded = json.loads(gzip.decompress(compressed))
    assert decoded["requested_mode"] == 1 and decoded["physical_steps"] == 1
    assert decoded["failure_value"] == {"__nonfinite_float__": "nan"}
    assert decoded["qpos"] == [1., 2.]


@pytest.mark.parametrize("state,expected", [
    ({"position": np.array([4., 5.])}, {"position": [4., 5.]}),
    (SimpleNamespace(obs=np.array([2.]), pipeline_state=SimpleNamespace(
        q=np.array([3.]), qd=np.array([4.]))), {"obs": [2.], "q": [3.], "qd": [4.]}),
])
def test_dictionary_and_structured_states_keep_their_own_representation(tmp_path, state, expected):
    runner = _runner(tmp_path)
    result = _result(runner, 0)
    result["result"]["trajectory"] = Trajectory(states=[state], actions=[], info={})
    runner._save_result(result)
    saved = json.loads((tmp_path / "level_0/seed_0/trajectory/trajectory.json").read_text())
    assert saved["states"] == [expected]


def test_aborted_runs_stay_in_denominator_and_count_once_after_reread(tmp_path):
    runner = _runner(tmp_path)
    calls = []
    def execute(level, seed):
        calls.append(seed)
        return _result(runner, seed, aborted=seed == 1)
    runner.run_single_experiment = execute
    assert len(runner.run_all()) == 2
    summary = json.loads((tmp_path / "overall_summary.json").read_text())
    assert calls == [0, 1]
    assert summary["total_experiments"] == 2
    assert summary["failed_experiments"] == 1
    assert summary["metrics"]["task"]["success"]["mean"] == .5
    assert summary["metrics"]["task"]["success"]["n"] == 2
    assert len(runner._collect_completed_results()) == 1
    assert len(runner._collect_completed_results(include_aborted=True)) == 2
    assert runner.failures[0]["recorded_evidence"]["episodes"]["sha256"]

    # A later invocation executes only its new request; the historical abort
    # remains a failed row, not an omitted sample or a second failure count.
    later = _runner(tmp_path, seeds=(2,))
    later.run_single_experiment = lambda level, seed: _result(later, seed)
    later.run_all()
    summary = json.loads((tmp_path / "overall_summary.json").read_text())
    assert summary["total_experiments"] == 3
    assert summary["failed_experiments"] == 1
    assert summary["metrics"]["task"]["success"]["mean"] == pytest.approx(2/3)
    assert len(later.failures) == 1


def test_setup_exception_retains_charged_prefix_and_cannot_be_replaced(tmp_path):
    runner = _runner(tmp_path)
    calls = []
    def execute(level, seed):
        calls.append(seed)
        if seed == 0:
            error = RuntimeError("synthetic failed preparation")
            error.recorded_evidence = {"episodes": {"requested_mode": 1, "preparation": [1, 2, 3]}}
            error.execution_accounting = {"reset_worlds": 1, "preparation_control_steps": 3}
            raise error
        return _result(runner, seed)
    runner.run_single_experiment = execute
    runner.run_all()
    saved = json.loads((tmp_path / "level_0/seed_0/results.json").read_text())
    assert saved["execution_status"]["state"] == "aborted_unrecoverable"
    assert saved["execution_status"]["execution_accounting"]["preparation_control_steps"] == 3
    assert saved["execution_status"]["planning_time_measured"] is False
    assert runner.failures[0]["recorded_evidence"] == saved["recorded_evidence"]
    summary = json.loads((tmp_path / "overall_summary.json").read_text())
    assert summary["total_experiments"] == 2 and summary["failed_experiments"] == 1
    assert summary["avg_planning_time"] == 1.  # unmeasured setup time is not a zero timing sample
    assert summary["metrics"]["task"]["success"]["mean"] == .5
    with pytest.raises(ValueError, match="Refusing to resume/overwrite aborted"):
        runner.run_all(resume=True)
    assert calls == [0, 1]
