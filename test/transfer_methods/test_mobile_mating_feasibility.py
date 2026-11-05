"""Public mobile baseline screens must preserve the frozen execution contract."""

from copy import deepcopy
import json
from pathlib import Path
import re
import runpy

import pytest

from genedynamics.tasks.transfer.contact import MobileContactMechanismSpec
from genedynamics.tasks.transfer.mobile_mating_policy_search import (
    MobileMatingPolicySearchTask, MobileMatingSearchTask,
    feasibility_plan,
)


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / (
    "configs/quadruped/mobile_keyed_insertion_recovery/validation/"
    "adaptsim_mobile_feedback_feasibility.json"
)


def test_mobile_screen_preflight_accepts_only_unique_public_requests(tmp_path):
    config = json.loads(RECIPE.read_text())
    plan = feasibility_plan(config, tmp_path)
    assert plan["maximum_physics_episodes"] == 8
    assert plan["formal_performance_evidence"] is False

    duplicate = deepcopy(config)
    duplicate["requests"][1]["mode"] = 0
    with pytest.raises(ValueError, match="cell/mode"):
        feasibility_plan(duplicate, tmp_path)


def test_mobile_screen_rejects_changed_body_contract_and_policy(tmp_path):
    config = json.loads(RECIPE.read_text())
    changed_body = deepcopy(config)
    changed_body["shared_execution"]["body_velocity_world"] = [.01, .005, 0.]
    with pytest.raises(ValueError, match="shared reset/body intent"):
        feasibility_plan(changed_body, tmp_path)

    changed_policy = deepcopy(config)
    changed_policy["policies"][1]["values"][0] = 4.
    with pytest.raises(ValueError, match="outside declared support"):
        feasibility_plan(changed_policy, tmp_path)


@pytest.mark.parametrize("error", [
    ValueError("Go2+Z1 source pose unreachable: mock"),
    RuntimeError("floating reset stance preparation failed: 3, mock"),
])
def test_adaptsim_t4_simulator_reset_is_charged_but_target_reset_aborts(
    monkeypatch, error,
):
    import genedynamics.tasks.transfer.mobile_mating_policy_search as module

    class FailingTask:
        def reset(self, *args, **kwargs):
            raise error

    monkeypatch.setattr(module, "instantiate_task", lambda *args, **kwargs: FailingTask())
    binding = MobileMatingPolicySearchTask.__new__(MobileMatingPolicySearchTask)
    binding.shared_execution = {}
    binding.reset_seed_pool = ()
    mechanism = MobileContactMechanismSpec()
    simulated = binding.evaluate_policy(
        [0] * 7, [MobileMatingSearchTask(mechanism, (11, 12))], 1,
    )
    assert simulated.reward == -2.
    assert simulated.simulation_steps == 0
    assert [row["outcome"] for row in simulated.episodes] == [
        "reset_unreachable", "reset_unreachable",
    ]
    target_simulation, rows = binding.evaluate_target(
        [0] * 7, mechanism, 11, simulator=True,
    )
    assert target_simulation.target_control_steps == 0
    assert [row["outcome"] for row in rows] == [
        "reset_unreachable", "reset_unreachable",
    ]
    with pytest.raises(type(error), match=re.escape(str(error))):
        binding.evaluate_target([0] * 7, mechanism, 11)


def test_adaptsim_t4_unrecognized_reset_error_aborts_simulator(monkeypatch):
    import genedynamics.tasks.transfer.mobile_mating_policy_search as module

    class FailingTask:
        def reset(self, *args, **kwargs):
            raise RuntimeError("controller stepped before reset")

    monkeypatch.setattr(module, "instantiate_task", lambda *args, **kwargs: FailingTask())
    binding = MobileMatingPolicySearchTask.__new__(MobileMatingPolicySearchTask)
    binding.shared_execution = {}
    binding.reset_seed_pool = ()
    with pytest.raises(RuntimeError, match="controller stepped before reset"):
        binding.evaluate_policy(
            [0] * 7,
            [MobileMatingSearchTask(MobileContactMechanismSpec(), (11, 12))], 1,
        )


def test_adaptsim_t4_independent_ledger_rejects_deployment_reset_failure():
    check = runpy.run_path(str(
        ROOT / "scripts/paper/cross_ebdm/audit_adaptsim_population.py"
    ))["_check_episode"]
    row = {
        "mode": 0, "reset_seed": 11, "success": False,
        "constraint_violation": False, "outcome": "reset_unreachable",
        "control_steps": 0, "reward": -2.,
        "reset_error": "floating reset stance preparation failed: 3, mock",
    }
    check(row, "T4", simulator=True)
    with pytest.raises(ValueError, match="invalid episode"):
        check(row, "T4", simulator=False)
