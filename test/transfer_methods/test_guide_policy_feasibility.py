"""T2 baseline policy screens must preserve the shared physical contract."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from genedynamics.tasks.transfer.guide_policy_search import feasibility_plan


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / (
    "configs/arm/guided_contact_alignment/validation/"
    "adaptsim_guide_policy_feasibility.json"
)


def _copy_task_config(recipe, tmp_path):
    relative = Path(recipe["task_config"]["path"])
    target = tmp_path / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes((ROOT / relative).read_bytes())


def test_guide_preflight_freezes_task_config_and_public_support(tmp_path):
    recipe = json.loads(RECIPE.read_text())
    _copy_task_config(recipe, tmp_path)
    plan = feasibility_plan(recipe, tmp_path)
    assert plan["maximum_physics_episodes"] == 12
    assert plan["formal_performance_evidence"] is False

    changed = deepcopy(recipe)
    changed["task_config"]["sha256"] = "0" * 64
    with pytest.raises((FileNotFoundError, ValueError)):
        feasibility_plan(changed, tmp_path)

    changed = deepcopy(recipe)
    changed["policies"][0]["values"][2] = .08
    with pytest.raises(ValueError, match="outside declared public support"):
        feasibility_plan(changed, tmp_path)


def test_guide_preflight_requires_all_four_paired_requests(tmp_path):
    recipe = json.loads(RECIPE.read_text())
    _copy_task_config(recipe, tmp_path)
    changed = deepcopy(recipe)
    changed["requests"][1]["mode"] = 0
    with pytest.raises(ValueError, match="source/nominal modes"):
        feasibility_plan(changed, tmp_path)
