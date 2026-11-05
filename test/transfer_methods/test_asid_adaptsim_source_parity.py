"""Compare new solver kernels to pinned public author code when available."""

from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from genedynamics.solvers.single.asid import ASIDSolver
from genedynamics.solvers.single.asid.learning import ASIDBoundedCEM
from genedynamics.solvers.single.adaptsim import (
    AdaptSimSolver,
    ParameterDistribution,
    ParameterSpec,
)
from genedynamics.solvers.single.adaptsim.backends.branching_q_torch import BranchingQNetwork
from genedynamics.solvers.single.adaptsim.learning import (
    AdaptSimAdaptation,
    AdaptSimMetaTraining,
    AdaptationEvaluation,
    AdaptationTraining,
    CEMTaskPolicySearch,
    PolicySearchEvaluation,
)


ROOT = Path(__file__).resolve().parents[2]


def test_mobile_asid_only_counts_known_zero_control_simulator_resets() -> None:
    from genedynamics.tasks.transfer.mobile_asid_training import _simulator_reset_rejection

    assert _simulator_reset_rejection(ValueError("Go2+Z1 source pose unreachable: IK"))
    assert _simulator_reset_rejection(RuntimeError("floating reset stance preparation failed: 11"))
    assert not _simulator_reset_rejection(RuntimeError("controller crashed after one control"))
    assert not _simulator_reset_rejection(ValueError("floating reset stance preparation failed: wrong exception"))


def test_adaptsim_t3_public_binding_replays_audited_actions() -> None:
    """The common actor factory must execute the audited trained policy."""
    import gzip

    from genedynamics.envs.adapters.transfer_task import instantiate_task
    from genedynamics.experiments.framework import ExperimentConfig
    from genedynamics.tasks.transfer.contact import ContactMechanismSpec
    from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor

    config_path = ROOT / "configs/arm/keyed_insertion_recovery/baseline/adaptsim.yaml"
    evidence_path = ROOT / (
        "results/arm/keyed_insertion_recovery/baseline/adaptsim/validation/"
        "feedback_population_heldout_diagnosis_8737217/evidence/"
        "0_0_0_adaptsim_development.json.gz"
    )
    if not evidence_path.is_file():
        pytest.skip("audited development evidence is unavailable")
    expected = np.asarray(json.loads(gzip.decompress(evidence_path.read_bytes()))[
        "records"
    ][0]["requested_actions"])
    config = ExperimentConfig.from_yaml(config_path)
    seed = 3_200_400_000
    task = instantiate_task(
        "T3", robot="xarm7",
        mechanism_spec=ContactMechanismSpec(
            servo_gain=.85, payload_mass=.65, delay_steps=0,
        ), seed=seed, shared_execution=config.env_params["shared_execution"],
    )
    public = task.reset((0,), seed=seed, initial="jammed", modes=np.asarray([0]))
    actor = create_transfer_actor(task, "adaptsim", config.method_params)
    actual = []
    for _ in range(task.config.timeout_steps):
        command = actor.act(public)
        actual.append(command[0])
        transition = task.step(command)
        public = transition.observation
        if bool(transition.done[0]):
            break
    np.testing.assert_array_equal(actual, expected)

    changed = deepcopy(config.method_params)
    changed["actor_binding"]["training_report"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 validation"):
        create_transfer_actor(task, "adaptsim", changed)


def test_asid_t1_public_binding_replays_audited_actions() -> None:
    """The common ASID actor must preserve the diagnosed downstream policy."""
    import gzip

    from genedynamics.envs.adapters.transfer_task import instantiate_task
    from genedynamics.experiments.framework import ExperimentConfig
    from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
    from genedynamics.tasks.transfer.runtime import MechanismSpec

    run = ROOT / (
        "results/arm/obstacle_transfer/baseline/asid/validation/"
        "nominal_diagnosis_grid_seeded_3817251"
    )
    evidence_path = run / "episodes/00_asid/evidence.json.gz"
    if not evidence_path.is_file():
        pytest.skip("audited ASID development evidence is unavailable")
    request = json.loads((run / "report.json").read_text())["rows"][0]["request"]
    expected = np.asarray(json.loads(gzip.decompress(evidence_path.read_bytes()))[
        "records"
    ][0]["requested_actions"])
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/obstacle_transfer/baseline/asid.yaml"
    )
    task = instantiate_task(
        "T1", robot="xarm7",
        mechanism_spec=MechanismSpec(**request["mechanism"]),
        seed=request["reset_seed"],
    )
    public = task.reset(
        (request["physical_initial_slot"],), seed=request["reset_seed"],
        modes=np.asarray([request["requested_mode"]]),
    )
    parameters = {**config.method_params, "mode": request["requested_mode"]}
    actor = create_transfer_actor(task, "asid", parameters)
    actual = []
    for _ in range(task.config.timeout_steps):
        command = actor.act(public)
        actual.append(command[0])
        transition = task.step(command)
        public = transition.observation
        if bool(transition.done[0]):
            break
    np.testing.assert_array_equal(actual, expected)

    changed = deepcopy(parameters)
    changed["actor_binding"]["training_report"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 validation"):
        create_transfer_actor(task, "asid", changed)


def test_asid_t2_public_binding_requires_audited_checkpoint(monkeypatch) -> None:
    """The T2 public actor binds the searched policy to its audited report."""
    from genedynamics.tasks.transfer.controllers.asid import public_guide_checkpoint_actor
    from genedynamics.tasks.transfer.guide_asid import TRAINING_VERSION

    task = object()
    policy = [.02, .008, .03]
    report_pointer = {"path": "results/t2/report.json", "sha256": "a" * 64}
    recipe = {
        "version": TRAINING_VERSION, "task": "T2", "robot": "xarm7",
        "task_policy": {"lower": [.01, .006, .02], "upper": [.04, .01, .04]},
    }
    report = {
        "version": TRAINING_VERSION, "status": "complete_development_chain",
        "formal_performance_evidence": False,
        "identification": {"best_parameters": [.8, .6, 1]},
        "task_policy": {"best_policy": policy},
    }
    audit = {
        "schema_version": "asid-t2-full-chain-independent-audit-v1",
        "status": "passed_development_only",
        "report_sha256": report_pointer["sha256"],
        "formal_performance_evidence": False,
    }
    checkpoint = {
        "schema_version": "asid-guide-public-policy-checkpoint-v1",
        "task": "T2", "robot": "xarm7", "development_only": True,
        "source_report": report_pointer,
        "identified_mechanism": [.8, .6, 1], "policy": policy,
    }
    monkeypatch.setattr(
        "genedynamics.tasks.transfer.controllers.adaptsim.public_guide_actor",
        lambda bound_task, vector, *, mode: (bound_task, vector.tolist(), mode),
    )
    assert public_guide_checkpoint_actor(
        task, checkpoint, recipe, report, audit,
        report_pointer=report_pointer, mode=1,
    ) == (task, policy, 1)
    with pytest.raises(ValueError, match="identities differ"):
        public_guide_checkpoint_actor(
            task, checkpoint, recipe, report,
            {**audit, "report_sha256": "b" * 64},
            report_pointer=report_pointer, mode=1,
        )


def test_asid_t3_public_binding_replays_audited_actions() -> None:
    """The T3 public actor must replay its independently audited policy."""
    import gzip

    from genedynamics.envs.adapters.transfer_task import instantiate_task
    from genedynamics.experiments.framework import ExperimentConfig
    from genedynamics.tasks.transfer.contact import ContactMechanismSpec
    from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor

    run = ROOT / (
        "results/arm/keyed_insertion_recovery/baseline/asid/validation/"
        "fresh_actor_pilot_3302500000"
    )
    if not (run / "independent_audit.json").is_file():
        pytest.skip("audited ASID T3 development evidence is unavailable")
    report = json.loads((run / "report.json").read_text())
    row = report["results"][0]
    request = row["request"]
    evidence_path = ROOT / row["evidence"]["path"]
    expected = np.asarray(json.loads(gzip.decompress(evidence_path.read_bytes()))[
        "records"
    ][0]["requested_actions"])
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/keyed_insertion_recovery/baseline/asid.yaml"
    )
    task = instantiate_task(
        "T3", robot="xarm7",
        mechanism_spec=ContactMechanismSpec(**request["mechanism_parameters"]),
        seed=request["reset_seed"],
        shared_execution=config.env_params["shared_execution"],
    )
    mode = request["requested_mode"]
    public = task.reset(
        (request["physical_initial_slot"],), seed=request["reset_seed"],
        initial="jammed", modes=np.asarray([mode]),
    )
    actor = create_transfer_actor(
        task, "asid", {**config.method_params, "mode": mode},
    )
    actual = []
    for _ in range(task.config.timeout_steps):
        command = actor.act(public)
        actual.append(command[0])
        transition = task.step(command)
        public = transition.observation
        if bool(transition.done[0]):
            break
    np.testing.assert_array_equal(actual, expected)
    assert task.evaluate()["episodes"][-1]["outcome"] == row["outcome"]

    changed = deepcopy(config.method_params)
    changed["actor_binding"]["training_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 validation"):
        create_transfer_actor(task, "asid", changed)


def test_adaptsim_t4_binding_requires_complete_audited_mobile_training() -> None:
    """A T3 policy or a ledger without its audit cannot enter T4's actor."""
    from genedynamics.tasks.transfer.controllers.adaptsim import public_mobile_mating_actor
    from genedynamics.tasks.transfer.mobile_mating_policy_search import (
        MOBILE_CONFIG, TRAINING_VERSION,
    )
    from genedynamics.tasks.transfer.specs import task_spec

    recipe = json.loads((
        ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/training/"
        "adaptsim_mobile_population_development_v2.json"
    ).read_text())
    spec = task_spec("T4", robot="go2_z1", config_overrides=MOBILE_CONFIG)
    task = SimpleNamespace(spec=spec, shared_execution=recipe["shared_execution"])
    policy = recipe["task_policy_search"]["anchor"]
    checkpoint_sha = "a" * 64
    report_sha = "b" * 64
    checkpoint = {"version": TRAINING_VERSION, "final_policy": policy}
    report = {
        "version": TRAINING_VERSION, "task": "T4", "status": "complete_development",
        "checkpoint": {"sha256": checkpoint_sha},
        "deployment": {"final_policy": policy},
        "formal_performance_evidence": False,
    }
    audit = {
        "schema_version": "adaptsim-t4-population-independent-ledger-audit-v1",
        "status": "passed", "report_sha256": report_sha,
        "checkpoint_sha256": checkpoint_sha, "physical_episodes_accounted": 882,
    }
    actor = public_mobile_mating_actor(
        task, checkpoint, recipe, report, audit,
        checkpoint_sha256=checkpoint_sha, report_sha256=report_sha,
        mode=1, seed=33,
    )
    assert actor.spec.alias == "T4"
    with pytest.raises(ValueError, match="identit"):
        public_mobile_mating_actor(
            task, checkpoint, recipe, report,
            {**audit, "physical_episodes_accounted": 881},
            checkpoint_sha256=checkpoint_sha, report_sha256=report_sha,
            mode=1, seed=33,
        )


def test_adaptsim_t4_training_plan_freezes_population_and_source() -> None:
    """The mobile training budget and bound source code are checked before physics."""
    from genedynamics.tasks.transfer.mobile_mating_policy_search import training_plan

    old_recipe = json.loads((
        ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/training/"
        "adaptsim_mobile_population_development_v2.json"
    ).read_text())
    with pytest.raises(ValueError, match="source binding differs"):
        training_plan({**old_recipe, "output": (
            "results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/"
            "training/old_plan_only_test_never_run"
        )}, ROOT)
    recipe = json.loads((
        ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/training/"
        "adaptsim_independent_training_8814108.json"
    ).read_text())
    recipe["output"] = (
        "results/quadruped/mobile_keyed_insertion_recovery/baseline/adaptsim/"
        "training/plan_only_test_never_run"
    )
    plan = training_plan(recipe, ROOT)
    assert plan["maximum_physics_episodes"] == 882
    assert plan["candidate_physics_episodes"] == 840
    changed = deepcopy(recipe)
    changed["source_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source binding differs"):
        training_plan(changed, ROOT)


def test_asid_t4_training_plan_and_mobile_actor_audit_gate() -> None:
    """The mobile ASID actor accepts only its exact full-chain audit receipt."""
    from genedynamics.tasks.transfer.controllers.asid import public_mobile_mating_checkpoint_actor
    from genedynamics.tasks.transfer.mobile_asid_training import VERSION, training_plan
    from genedynamics.tasks.transfer.mobile_mating_policy_search import MOBILE_CONFIG
    from genedynamics.tasks.transfer.specs import task_spec

    recipe = json.loads((
        ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/training/"
        "asid_full_chain_development_v2.json"
    ).read_text())
    planned = deepcopy(recipe)
    planned["output"] = (
        "results/quadruped/mobile_keyed_insertion_recovery/baseline/asid/"
        "training/plan_only_test_never_run"
    )
    # Historical recipe bytes stay frozen; test the current task code contract.
    for row in planned["source_bindings"]:
        if row["path"] in {
            "genedynamics/tasks/transfer/mobile_asid_training.py",
            "genedynamics/tasks/transfer/mobile_mating_policy_search.py",
        }:
            row["sha256"] = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
    assert training_plan(planned, ROOT)["maximum_target_episodes"] == 3
    changed = deepcopy(planned)
    changed["source_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source SHA differs"):
        training_plan(changed, ROOT)

    task = SimpleNamespace(
        spec=task_spec("T4", robot="go2_z1", config_overrides=MOBILE_CONFIG),
        shared_execution=recipe["shared_execution"],
    )
    pointer = {"path": "results/t4/asid/report.json", "sha256": "a" * 64}
    policy = recipe["task_policy"]["anchor"]
    report = {
        "version": VERSION, "status": "complete_development_chain",
        "formal_performance_evidence": False,
        "identification": {"best_parameters": [.7, .5, .9, 3]},
        "task_policy": {"best_policy": policy},
    }
    checkpoint = {
        "schema_version": "asid-mobile-public-policy-checkpoint-v2",
        "task": "T4", "robot": "go2_z1", "development_only": True,
        "source_report": pointer, "identified_mechanism": [.7, .5, .9, 3],
        "policy": policy,
    }
    audit = {
        "schema_version": "asid-t4-full-chain-independent-audit-v2",
        "status": "passed_development_only",
        "report_sha256": pointer["sha256"],
        "formal_performance_evidence": False,
    }
    actor = public_mobile_mating_checkpoint_actor(
        task, checkpoint, recipe, report, audit,
        report_pointer=pointer, mode=0, seed=44,
    )
    assert actor.spec.alias == "T4"
    with pytest.raises(ValueError, match="identit"):
        public_mobile_mating_checkpoint_actor(
            task, checkpoint, recipe, report,
            {**audit, "report_sha256": "b" * 64},
            report_pointer=pointer, mode=0, seed=44,
        )


def test_adaptsim_t1_public_binding_replays_audited_policy() -> None:
    """The T1 AdaptSim final policy must survive the common actor binding."""
    import gzip

    from genedynamics.envs.adapters.transfer_task import instantiate_task
    from genedynamics.experiments.framework import ExperimentConfig
    from genedynamics.tasks.transfer.controllers.bindings import create_transfer_actor
    from genedynamics.tasks.transfer.runtime import MechanismSpec

    run = ROOT / (
        "results/arm/obstacle_transfer/baseline/adaptsim/validation/"
        "source_nominal_evidence_ledger_replay_928505"
    )
    evidence_path = run / "evidence/000_adaptsim_pilot.json.gz"
    if not evidence_path.is_file():
        pytest.skip("audited AdaptSim T1 evidence is unavailable")
    row = json.loads((run / "report.json").read_text())["rows"][0]
    scene = json.loads((run / "recipe.json").read_text())["scene_cells"][0]
    expected = np.asarray(json.loads(gzip.decompress(evidence_path.read_bytes()))[
        "records"
    ][0]["requested_actions"])
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/obstacle_transfer/baseline/adaptsim.yaml"
    )
    task = instantiate_task(
        "T1", robot="xarm7",
        mechanism_spec=MechanismSpec(**scene["mechanism"]),
        seed=row["reset_seed"],
    )
    public = task.reset(
        (row["physical_initial_slot"],), seed=row["reset_seed"],
        modes=np.asarray([row["mode"]]),
    )
    actor = create_transfer_actor(
        task, "adaptsim", {**config.method_params, "mode": row["mode"]},
    )
    actual = []
    for _ in range(task.config.timeout_steps):
        command = actor.act(public)
        actual.append(command[0])
        transition = task.step(command)
        public = transition.observation
        if bool(transition.done[0]):
            break
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert task.evaluate()["episodes"][-1]["outcome"] == row["outcome"]

    changed = deepcopy(config.method_params)
    changed["actor_binding"]["training_audit"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256 validation"):
        create_transfer_actor(task, "adaptsim", changed)


def test_asid_bounded_search_retains_every_candidate_and_is_reproducible() -> None:
    search = ASIDBoundedCEM([-1., -1.], [1., 1.], [0., 0.],
                            population=7, elite=2, iterations=3, maximize=False)

    def evaluate(candidate, iteration, index):
        return float(np.sum((candidate - [.35, -.25])**2)), {
            "episodes": [{"iteration": iteration, "index": index,
                          "outcome": "simulated"}],
        }

    first = search.run(evaluate, seed=27182)
    second = search.run(evaluate, seed=27182)
    assert len(first.ledger) == 21
    assert first.best_score <= first.ledger[0]["score"]
    assert first.ledger == second.ledger
    np.testing.assert_array_equal(first.best, second.best)
    assert all(len(row["episodes"]) == 1 for row in first.ledger)


def test_asid_full_chain_plan_binds_budget_and_sources(tmp_path) -> None:
    from genedynamics.tasks.transfer.obstacle_asid import plan

    config = json.loads((
        ROOT / "configs/arm/obstacle_transfer/training/asid_grid_seeded_development.json"
    ).read_text())
    config["output"] = str(tmp_path / "asid")
    # Keep the executed recipe immutable; this plan-only regression uses the
    # current task files after their later nominal-evidence extensions.
    for name in (
        "genedynamics/tasks/transfer/obstacle_asid.py",
        "genedynamics/tasks/transfer/obstacle_policy_search.py",
    ):
        next(row for row in config["source_bindings"]
             if row["path"] == name)["sha256"] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
    frozen = plan(config)
    assert frozen["maximum_simulator_episodes"] == 347
    assert frozen["maximum_target_episodes"] == 3
    assert len(frozen["source_bindings"]) == 6
    assert frozen["formal_performance_evidence"] is False
    changed = deepcopy(config)
    changed["identification_grid"]["payload_mass"].remove(.5)
    with pytest.raises(ValueError, match="global identification grid changed"):
        plan(changed)
    changed = deepcopy(config)
    changed["source_bindings"][2]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source binding changed"):
        plan(changed)


def test_asid_nominal_plan_binds_disjoint_roster_and_checkpoint(tmp_path) -> None:
    from genedynamics.tasks.transfer.obstacle_asid import nominal_plan

    config = json.loads((
        ROOT / "configs/arm/obstacle_transfer/validation/asid_nominal_diagnosis.json"
    ).read_text())
    config["output"] = str(tmp_path / "asid_nominal")
    # The archived diagnosis remains SHA-bound to its original task adapter.
    for row in config["source_bindings"]:
        if row["path"] == "genedynamics/tasks/transfer/obstacle_asid.py":
            row["sha256"] = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
    frozen = nominal_plan(config)
    assert frozen["planned_physics_episodes"] == 48
    assert len(frozen["requests"]) == 24
    assert min(row["reset_seed"] for row in frozen["requests"]) >= 3_000_200_000
    assert frozen["formal_performance_evidence"] is False
    changed = deepcopy(config)
    changed["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source binding changed"):
        nominal_plan(changed)


def test_asid_t1_independent_cohort_requires_extended_source_snapshot(tmp_path) -> None:
    from genedynamics.tasks.transfer.obstacle_asid import INDEPENDENT_COHORT, plan

    recipe = json.loads((
        ROOT / "configs/arm/obstacle_transfer/training/"
        "asid_grid_seeded_independent_training_8801001.json"
    ).read_text())
    recipe["output"] = str(tmp_path / "asid_independent")
    frozen = plan(recipe)
    assert frozen["training_cohort"] == INDEPENDENT_COHORT
    assert len(frozen["source_bindings"]) == 16
    changed = deepcopy(recipe)
    changed["source_bindings"].pop()
    with pytest.raises(ValueError, match="six exact source bindings"):
        plan(changed)


def test_asid_t3_independent_cohort_binds_disjoint_target_seeds() -> None:
    from genedynamics.tasks.transfer.mating_asid import (
        INDEPENDENT_COHORT, training_plan,
    )

    recipe = json.loads((
        ROOT / "configs/arm/keyed_insertion_recovery/training/"
        "asid_independent_training_8803001.json"
    ).read_text())
    recipe["output"] = (
        "results/arm/keyed_insertion_recovery/baseline/asid/training/"
        "plan_only_test_never_run/independent_training_8803001"
    )
    frozen = training_plan(recipe, ROOT)
    assert frozen["training_cohort"] == INDEPENDENT_COHORT
    assert frozen["maximum_target_episodes"] == 3
    changed = deepcopy(recipe)
    changed["target_deployment_seeds"][1] += 1
    with pytest.raises(ValueError, match="development contract changed"):
        training_plan(changed, ROOT)


def test_asid_t2_independent_cohort_binds_disjoint_target_seeds() -> None:
    from genedynamics.tasks.transfer.guide_asid import (
        INDEPENDENT_COHORT, training_plan,
    )

    recipe = json.loads((
        ROOT / "configs/arm/guided_contact_alignment/training/"
        "asid_independent_training_8802001.json"
    ).read_text())
    # The historical executed recipe remains frozen; test the current binding.
    for row in recipe["source_bindings"]:
        if row["path"] == "genedynamics/tasks/transfer/controllers/adaptsim.py":
            row["sha256"] = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
    recipe["output"] = (
        "results/arm/guided_contact_alignment/baseline/asid/training/"
        "plan_only_test_never_run/independent_training_8802001"
    )
    frozen = training_plan(recipe, ROOT)
    assert frozen["training_cohort"] == INDEPENDENT_COHORT
    assert frozen["maximum_target_episodes"] == 3
    changed = deepcopy(recipe)
    changed["target_exploration_seed"] += 1
    with pytest.raises(ValueError, match="development contract changed"):
        training_plan(changed, ROOT)


def test_asid_t4_independent_reset_pool_is_predeclared_without_physics() -> None:
    script = ROOT / "scripts/paper/cross_ebdm/qualify_mobile_training_reset.py"
    spec = importlib.util.spec_from_file_location("qualify_mobile_training_reset", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    recipe = json.loads((
        ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/validation/"
        "asid_independent_training_reset_feasibility_3404000000.json"
    ).read_text())
    recipe["output"] = (
        "results/quadruped/mobile_keyed_insertion_recovery/baseline/asid/"
        "diagnostics/plan_only_test_never_run"
    )
    assert module.plan(recipe)["maximum_reset_calls"] == 256
    changed = deepcopy(recipe)
    changed["seed_count"] = 63
    with pytest.raises(ValueError, match="contract differs"):
        module.plan(changed)


def test_asid_t4_independent_cohort_requires_fresh_safe_pool() -> None:
    from genedynamics.tasks.transfer.mobile_asid_training import (
        INDEPENDENT_COHORT_V2, training_plan,
    )

    recipe = json.loads((
        ROOT / "configs/quadruped/mobile_keyed_insertion_recovery/training/"
        "asid_independent_training_8804101.json"
    ).read_text())
    recipe["output"] = (
        "results/quadruped/mobile_keyed_insertion_recovery/baseline/asid/"
        "training/plan_only_test_never_run/independent_training_8804101"
    )
    with pytest.raises(ValueError, match="source SHA differs"):
        training_plan(recipe, ROOT)
    for row in recipe["source_bindings"]:
        if row["path"] == "genedynamics/tasks/transfer/mobile_mating_policy_search.py":
            row["sha256"] = hashlib.sha256((ROOT / row["path"]).read_bytes()).hexdigest()
    frozen = training_plan(recipe, ROOT)
    assert frozen["training_cohort"] == INDEPENDENT_COHORT_V2
    assert frozen["maximum_target_episodes"] == 3
    changed = deepcopy(recipe)
    changed["target_deployment_seeds"][0] += 1
    with pytest.raises(ValueError, match="training contract differs"):
        training_plan(changed, ROOT)


def test_asid_public_grid_fit_uses_observations_to_rank_ambiguous_parameters() -> None:
    from genedynamics.tasks.transfer.obstacle_identifiability import fit_grid

    observed = np.zeros((2, 6), dtype=np.float64)
    candidates = {
        (.7, .5): observed.copy(),
        (.7, .6): observed + np.array([[.0006, 0, 0, .0007, 0, 0],
                                       [.0005, 0, 0, .0005, 0, 0]]),
        (.8, .5): observed + np.array([[.002, 0, 0, .004, 0, 0],
                                       [.002, 0, 0, .004, 0, 0]]),
    }
    ranked = fit_grid(observed, candidates)
    assert [(row["servo_gain"], row["payload_mass"]) for row in ranked] == [
        (.7, .5), (.7, .6), (.8, .5),
    ]
    assert ranked[1]["position_error_mm"][0] == pytest.approx(.6)


def _source(path: str) -> Path:
    source = ROOT / "baselines" / path
    if not source.is_file():
        pytest.skip("pinned upstream baseline source is unavailable")
    return source


def test_adaptsim_nominal_diagnosis_binds_disjoint_resets_and_source(tmp_path):
    from genedynamics.tasks.transfer.obstacle_policy_search import nominal_diagnosis_plan

    config = json.loads((
        ROOT / "configs/arm/obstacle_transfer/validation/adaptsim_nominal_diagnosis.json"
    ).read_text())
    config["output"] = str(tmp_path / "nominal")
    plan = nominal_diagnosis_plan(config, ROOT)
    assert len(plan["requests"]) == 16
    assert plan["planned_physics_episodes"] == 32
    assert min(row["reset_seed"] for row in plan["requests"]) >= 2**31
    assert plan["policies"]["adaptsim_pilot"] != plan["policies"]["public_anchor"]
    assert plan["formal_performance_evidence"] is False

    changed = deepcopy(config)
    changed["checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint changed"):
        nominal_diagnosis_plan(changed, ROOT)
    changed = deepcopy(config)
    changed["seed_anchor"] = 928504
    with pytest.raises(ValueError, match="outside training sampler range"):
        nominal_diagnosis_plan(changed, ROOT)


def test_adaptsim_meta_development_freezes_rollout_budget_before_execution(tmp_path):
    from genedynamics.tasks.transfer.obstacle_policy_search import development_plan

    config = json.loads((
        ROOT / "configs/arm/obstacle_transfer/training/adaptsim_meta_coordinate_development.json"
    ).read_text())
    config["output"] = str(tmp_path / "meta")
    plan = development_plan(config, ROOT)
    assert plan["version"] == "adaptsim-obstacle-meta-development-v2"
    assert plan["outer_meta_iterations"] == 4
    assert plan["policy_training_calls"] == 42
    assert plan["simulated_target_rollouts"] == 36
    assert plan["deployment_target_rollouts"] == 6
    assert plan["maximum_physics_episodes"] == 672

    changed = deepcopy(config)
    changed["meta_training"]["inner_episodes_per_iteration"] = 3
    with pytest.raises(ValueError, match="schedule changed"):
        development_plan(changed, ROOT)
    changed = deepcopy(config)
    changed["execution"]["maximum_wall_seconds"] = 2000
    with pytest.raises(ValueError, match="limits"):
        development_plan(changed, ROOT)


def test_adaptsim_t3_feedback_binding_freezes_physics_budget_and_source(tmp_path):
    from genedynamics.tasks.transfer.mating_policy_search import (
        MatingPolicySearchTask, POLICY_FIELDS, development_plan,
    )

    config = json.loads((
        ROOT / "configs/arm/keyed_insertion_recovery/training/adaptsim_feedback_development.json"
    ).read_text())
    # The saved v1 recipe is immutable evidence for its original source. Rebind
    # only this in-memory copy when exercising the current implementation.
    for binding in config["source_bindings"]:
        binding["sha256"] = hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
    config["output"] = str(tmp_path / "mating")
    plan = development_plan(config, ROOT)
    assert plan["maximum_physics_episodes"] == 72
    assert plan["policy_training_calls"] == 6
    assert plan["formal_performance_evidence"] is False

    task = MatingPolicySearchTask(
        robot="xarm7", feedback_base=config["feedback_base"],
        shared_execution=config["shared_execution"],
    )
    feedback = task.feedback(config["task_policy_search"]["anchor"])
    assert [getattr(feedback, field) for field in POLICY_FIELDS] == config["task_policy_search"]["anchor"]
    with pytest.raises(ValueError, match="declared public support"):
        task.feedback([0.] * 7)

    changed = deepcopy(config)
    changed["source_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source binding changed"):
        development_plan(changed, ROOT)
    changed = deepcopy(config)
    changed["task_policy_search"]["population"] = 4
    with pytest.raises(ValueError, match="training budget differs"):
        development_plan(changed, ROOT)


def test_adaptsim_t3_population_training_charges_all_candidates(tmp_path):
    from genedynamics.tasks.transfer.mating_policy_search import development_plan

    config = json.loads((
        ROOT / "configs/arm/keyed_insertion_recovery/training/"
        "adaptsim_feedback_population_development.json"
    ).read_text())
    for binding in config["source_bindings"]:
        binding["sha256"] = hashlib.sha256((ROOT / binding["path"]).read_bytes()).hexdigest()
    config["output"] = str(tmp_path / "population")
    plan = development_plan(config, ROOT)
    assert plan["policy_training_calls"] == 21
    assert plan["candidate_physics_episodes"] == 840
    assert plan["simulated_target_rollouts"] == 16
    assert plan["deployment_target_rollouts"] == 5
    assert plan["maximum_physics_episodes"] == 882
    assert plan["formal_performance_evidence"] is False

    changed = deepcopy(config)
    changed["task_policy_search"]["tasks_per_candidate"] = 1
    with pytest.raises(ValueError, match="training budget differs"):
        development_plan(changed, ROOT)
    changed = deepcopy(config)
    changed["meta_training"]["inner_episodes"] = 1
    with pytest.raises(ValueError, match="meta/deployment budget differs"):
        development_plan(changed, ROOT)
    changed = deepcopy(config)
    changed["source_bindings"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source binding changed"):
        development_plan(changed, ROOT)


def test_asid_fisher_reward_matches_author_code() -> None:
    source = _source("ASID/env/wrappers/asid_reward.py")
    spec = importlib.util.spec_from_file_location("author_asid_reward", source)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def transition(state, action, theta):
        return np.array([
            state[0] + theta[0] * action[0] + theta[1] ** 2,
            state[1] + theta[1] * action[1] + theta[0] * theta[1],
        ])

    class AuthorEnv:
        def __init__(self):
            self.parameters = np.array([0.7, 1.2])
            self.state = np.array([0.2, -0.4])

        def get_parameters(self):
            return self.parameters.copy()

        def set_parameters(self, parameters):
            self.parameters = np.asarray(parameters).copy()

        def set_full_state(self, state):
            self.state = np.asarray(state).copy()

        def step(self, action):
            return transition(self.state, action, self.parameters), 0.0, False, {}

    env = AuthorEnv()
    state = env.state.copy()
    action = np.array([0.3, -0.8])
    author = module.ASIDRewardWrapper(env, delta=0.04, normalization=1.7, max_reward=9.0)
    port = ASIDSolver(transition, delta=0.04, normalization=1.7, max_reward=9.0)
    np.testing.assert_allclose(port.observation_jacobian(state, action, env.parameters),
                               author.estimate_loss_grad(state, action, env.parameters), atol=1e-12)
    np.testing.assert_allclose(port.information_reward(state, action, env.parameters),
                               author.get_reward(state, action, env.parameters), atol=1e-12)


def _author_apply_action():
    source = ast.parse(_source("AdaptSim/adaptsim/param/parameter_base.py").read_text())
    klass = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ParameterBase")
    method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == "apply_action")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)

    class Uniform:
        def __init__(self, lb_array, ub_array):
            self.lb = np.asarray(lb_array)
            self.ub = np.asarray(ub_array)

    class Gaussian:
        def __init__(self, m, L):
            self.mean = np.asarray(m)
            self.diag_std = np.asarray(L)

    scope = {"np": np, "Uniform": Uniform, "Gaussian": Gaussian}
    exec(compile(module, str(source), "exec"), scope)
    return scope["apply_action"], Uniform, Gaussian


@pytest.mark.parametrize("kind,action", [("uniform", (0, 3)), ("uniform", (2, 4)), ("gaussian", (0, 2))])
def test_adaptsim_parameter_actions_match_author_code(kind: str, action: tuple[int, int]) -> None:
    author_apply, Uniform, Gaussian = _author_apply_action()
    specs = (
        ParameterSpec("mass", 0.0, 2.0, 0.2, kind, 0.05, 0.5),
        ParameterSpec("friction", 0.2, 1.2, 0.1, kind, 0.05, 0.5),
    )
    solver = AdaptSimSolver(specs, feature_dim=1, hidden=(16, 16))
    left = np.array([0.8, 0.5])
    right = np.array([1.2, 0.9]) if kind == "uniform" else np.array([0.2, 0.1])
    distribution = ParameterDistribution(left, right, kind)
    original_dist = Uniform(left.copy(), right.copy()) if kind == "uniform" else Gaussian(left.copy(), right.copy())
    particle = SimpleNamespace(dist=original_dist)
    author = SimpleNamespace(
        dist_type=kind,
        num_adapt_param=2,
        _adapt_param_cfg={spec.name: SimpleNamespace(**vars(spec), discrete=False) for spec in specs},
    )
    author_apply(author, action, particle)
    updated = solver.apply_action(distribution, action)
    if kind == "uniform":
        np.testing.assert_allclose(updated.left, particle.dist.lb)
        np.testing.assert_allclose(updated.right, particle.dist.ub)
    else:
        np.testing.assert_allclose(updated.left, particle.dist.mean)
        np.testing.assert_allclose(updated.right, particle.dist.diag_std)


def test_adaptsim_branching_value_advantage_identity_and_gradient() -> None:
    torch.manual_seed(7)
    network = BranchingQNetwork(4, (5, 5), (16, 16))
    state = torch.randn(8, 4)
    q = network(state)
    assert q.shape == (8, 2, 5)
    torch.testing.assert_close(q.mean(dim=2), network.value(state).expand(-1, 2))
    q.sum().backward()
    assert any(parameter.grad is not None for parameter in network.parameters())


def test_adaptsim_branching_outputs_match_author_torch_network() -> None:
    _source("AdaptSim/adaptsim/model/branching_q_network.py")
    source_root = str(ROOT / "baselines" / "AdaptSim")
    sys.path.insert(0, source_root)
    try:
        author_class = importlib.import_module("adaptsim.model.branching_q_network").BranchingQNetwork
    finally:
        sys.path.remove(source_root)
    author = author_class(mlp_dim=[4, 16, 16], num_bin=[5, 5], activation_type="elu", verbose=False)
    port = BranchingQNetwork(4, (5, 5), (16, 16))
    with torch.no_grad():
        for index, layer in enumerate((port.model[0], port.model[2])):
            source_layer = author.model.moduleList[index].linear_1
            source_layer.weight.copy_(layer.weight)
            source_layer.bias.copy_(layer.bias)
        author.value_head.load_state_dict(port.value_head.state_dict())
        for original, adapted in zip(author.adv_heads, port.adv_heads):
            original.load_state_dict(adapted.state_dict())
    state = torch.randn(5, 4)
    torch.testing.assert_close(port(state), author(state))


def test_adaptsim_target_loop_charges_every_rollout_and_trains_on_observed_reward() -> None:
    torch.manual_seed(11)
    solver = AdaptSimSolver((ParameterSpec("mass", 0.0, 2.0, 0.1),), feature_dim=1, hidden=(8, 8))
    evaluations: list[int] = []

    def train(distribution, warm_start, seed):
        return AdaptationTraining(policy=float(distribution.left[0]), simulation_steps=17)

    def evaluate(policy, seed):
        evaluations.append(seed)
        return AdaptationEvaluation(reward=1.0 - abs(policy - 0.8),
                                    features=np.array([policy], dtype=np.float32),
                                    target_control_steps=19)

    loop = AdaptSimAdaptation(solver, train, evaluate, seed=31)
    result = loop.run(ParameterDistribution(np.array([0.6]), np.array([1.0]), "uniform"),
                      updates=3, train_inference=True, simulated_meta_target=True)
    assert result["simulation_steps"] == 4 * 17
    assert result["target_control_steps"] == 4 * 19
    assert result["target_rollouts"] == 4
    assert evaluations == [32, 34, 36, 38]
    assert len(result["records"]) == 4
    assert all("action" in row for row in result["records"][:3])
    assert "action" not in result["records"][3]


def test_adaptsim_deployment_never_updates_inference_from_target_reward() -> None:
    solver = AdaptSimSolver((ParameterSpec("mass", 0.0, 2.0, 0.1),), feature_dim=1, hidden=(8, 8))
    loop = AdaptSimAdaptation(
        solver,
        lambda distribution, warm_start, seed: AdaptationTraining(0, 1),
        lambda policy, seed: AdaptationEvaluation(0.0, np.zeros(1), 1),
        seed=1,
    )
    with pytest.raises(ValueError, match="simulated meta-targets"):
        loop.run(ParameterDistribution(np.array([0.6]), np.array([1.0]), "uniform"),
                 updates=1, train_inference=True)


def test_adaptsim_selects_trained_particle_on_shared_target_reset_and_charges_it() -> None:
    solver = AdaptSimSolver((ParameterSpec("mass", 0.0, 2.0, 0.1, "gaussian", 0.05, 0.25),),
                           feature_dim=1, hidden=(8, 8))
    training_calls: list[tuple[float, float | None, int]] = []
    target_calls: list[tuple[float, int]] = []

    def train(distribution, warm_start, seed):
        mean = float(distribution.left[0])
        training_calls.append((mean, warm_start, seed))
        return AdaptationTraining(mean, 11)

    def evaluate(policy, seed):
        target_calls.append((policy, seed))
        return AdaptationEvaluation(1.0 - abs(policy - 0.9), np.array([policy]), 17)

    particles = [
        ParameterDistribution(np.array([0.3]), np.array([0.1]), "gaussian"),
        ParameterDistribution(np.array([0.9]), np.array([0.1]), "gaussian"),
    ]
    loop = AdaptSimAdaptation(solver, train, evaluate, seed=41)
    selected = loop.select_particle(particles)
    assert selected.selected_index == 1
    assert selected.policy == 0.9
    assert selected.simulation_steps == 22
    assert selected.target_control_steps == 34
    assert [call[1] for call in target_calls] == [1000041, 1000041]
    assert [row["target_reward"] for row in selected.records] == pytest.approx([0.4, 1.0])
    continuation = loop.run(selected.distribution, updates=0, initial_policy=selected.policy)
    assert training_calls[-1][1] == selected.policy
    assert continuation["simulation_steps"] == 11
    assert continuation["target_control_steps"] == 17


def test_adaptsim_meta_training_uses_simulated_targets_and_accounts_for_every_rollout() -> None:
    torch.manual_seed(13)
    solver = AdaptSimSolver(
        (ParameterSpec("mass", 0.0, 2.0, 0.1),), feature_dim=1,
        hidden=(8, 8), single_step=False,
    )
    sampled_targets: list[int] = []
    trained: list[tuple[int, float]] = []
    evaluated: list[tuple[float, int, int]] = []

    def sample_target(seed: int) -> float:
        sampled_targets.append(seed)
        return 0.7 if len(sampled_targets) % 2 else 1.1

    def train(distribution, warm_start, seed):
        trained.append((seed, float(distribution.left[0])))
        return AdaptationTraining(float(distribution.left[0]), 23)

    def evaluate(policy, target, seed):
        evaluated.append((target, seed, len(evaluated)))
        return AdaptationEvaluation(
            reward=1.0 - abs(policy - target),
            features=np.array([policy - target], dtype=np.float32),
            target_control_steps=17,
        )

    result = AdaptSimMetaTraining(solver, train, sample_target, evaluate, seed=47).run(
        [ParameterDistribution(np.array([0.6]), np.array([1.0]), "uniform")],
        episodes=3, adaptation_steps=2, batch_size=2, update_every=2,
        gradient_steps=2, exploration_probability=1.0,
    )
    assert sampled_targets == [47, 100047, 200047]
    assert result.episodes == 3
    assert result.transitions == 6
    assert result.inference_updates == 6
    assert len(trained) == len(evaluated) == result.simulated_target_rollouts == 9
    assert result.simulation_steps == 9 * 23
    assert result.simulated_target_control_steps == 9 * 17
    assert all(len(record["steps"]) == 2 for record in result.records)
    assert all(isinstance(record["steps"][0]["reward"], float) for record in result.records)
    assert sum(record["initial_simulation_steps"] + sum(
        step["simulation_steps"] for step in record["steps"]
    ) for record in result.records) == result.simulation_steps
    assert sum(record["initial_simulated_target_control_steps"] + sum(
        step["simulated_target_control_steps"] for step in record["steps"]
    ) for record in result.records) == result.simulated_target_control_steps


def test_adaptsim_task_policy_search_optimizes_and_charges_candidates() -> None:
    search = CEMTaskPolicySearch(
        lower=[-2., -2.], upper=[2., 2.], anchor=[-1., -1.],
        sample_tasks=lambda distribution, rng, count: np.full(count, 0.75),
        evaluate_policy=lambda candidate, tasks, seed: PolicySearchEvaluation(
            reward=-float(np.sum((candidate - np.array([tasks[0], -0.5])) ** 2)),
            simulation_steps=11 * len(tasks),
        ),
        population=16, elite=4, iterations=5, tasks_per_candidate=3,
    )
    result = search(ParameterDistribution(np.array([0.5]), np.array([1.0]), "uniform"),
                    None, 76)
    np.testing.assert_allclose(result.policy, [0.75, -0.5], atol=0.25)
    assert len(search.last_ledger) == 80
    assert result.simulation_steps == sum(row["simulation_steps"] for row in search.last_ledger)
    assert result.simulation_steps == 80 * 3 * 11


def test_adaptsim_candidate_ledger_retains_each_mode_failure() -> None:
    search = CEMTaskPolicySearch(
        lower=[0.1], upper=[0.9], anchor=[0.5],
        sample_tasks=lambda distribution, rng, count: None,
        evaluate_policy=lambda candidate, tasks, seed: PolicySearchEvaluation(
            reward=0.1, simulation_steps=10,
            episodes=(
                {"mode": 0, "reset_seed": seed, "outcome": "success", "control_steps": 4},
                {"mode": 1, "reset_seed": seed + 1, "outcome": "overforce", "control_steps": 6},
            ),
        ),
        population=2, elite=1, iterations=1, tasks_per_candidate=1,
    )
    result = search(ParameterDistribution(np.array([0.5]), np.array([1.0]), "uniform"),
                    None, 81)
    assert result.simulation_steps == 20
    assert len(search.last_ledger) == 2
    assert all([row["outcome"] for row in candidate["episodes"]]
               == ["success", "overforce"] for candidate in search.last_ledger)
    assert sum(row["control_steps"] for candidate in search.last_ledger
               for row in candidate["episodes"]) == result.simulation_steps


def test_adaptsim_cold_start_does_not_pay_twice_for_anchor() -> None:
    search = CEMTaskPolicySearch(
        lower=[-1.], upper=[1.], anchor=[0.],
        sample_tasks=lambda distribution, rng, count: None,
        evaluate_policy=lambda candidate, tasks, seed: PolicySearchEvaluation(
            -abs(float(candidate[0]) - 0.3), 5,
        ),
        population=3, elite=1, iterations=1, tasks_per_candidate=1,
    )
    result = search(ParameterDistribution(np.array([0.5]), np.array([1.0]), "uniform"),
                    None, 12)
    assert result.simulation_steps == 15
    assert len({row["candidate"][0] for row in search.last_ledger}) == 3


def test_adaptsim_coordinate_probes_preserve_anchor_and_charge_each_candidate() -> None:
    search = CEMTaskPolicySearch(
        lower=[0., 0., 0.5, 0.5], upper=[1., 1., 1.5, 1.5],
        anchor=[0.5, 0.5, 1., 1.],
        sample_tasks=lambda distribution, rng, count: None,
        evaluate_policy=lambda policy, tasks, seed: PolicySearchEvaluation(
            reward=-abs(float(policy[2]) - 1.1), simulation_steps=13,
        ),
        population=5, elite=1, iterations=1, tasks_per_candidate=1,
        coordinate_probes=(2, 3), probe_fraction=0.1,
    )
    result = search(ParameterDistribution(np.array([0.5]), np.array([1.0]), "uniform"), None, 88)
    assert result.simulation_steps == 65
    assert len(search.last_ledger) == 5
    np.testing.assert_allclose([row["candidate"][2:] for row in search.last_ledger],
                               [[1., 1.], [0.9, 1.], [1.1, 1.], [1., 0.9], [1., 1.1]])
    np.testing.assert_allclose(result.policy, [0.5, 0.5, 1.1, 1.])


def test_adaptsim_gaussian_particle_clone_matches_author_source() -> None:
    source = ast.parse(_source("AdaptSim/adaptsim/agent/particle.py").read_text())
    factory = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "ParticleFactory")
    method = next(node for node in factory.body if isinstance(node, ast.FunctionDef) and node.name == "clone_particle")
    module = ast.Module(body=[method], type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {"np": np, "deepcopy": deepcopy}
    exec(compile(module, "author_particle.py", "exec"), namespace)
    seed = 81
    original = ParameterDistribution(np.array([0.8, 0.5]), np.array([0.1, 0.15]), "gaussian")
    author_particle = SimpleNamespace(
        dist=SimpleNamespace(mean=original.left.copy(), diag_std=original.right.copy()),
        twin=False, id=0,
    )
    author_factory = SimpleNamespace(
        rng=np.random.default_rng(seed), num_adapt_param=2,
        adapt_param_range=np.array([1.0, 1.0]),
        l_bounds=np.array([0.4, 0.2]), u_bounds=np.array([1.4, 1.2]),
    )
    author = namespace["clone_particle"](author_factory, author_particle, perturbation=0.2)
    solver = AdaptSimSolver((
        ParameterSpec("servo_gain", 0.4, 1.4, 0.1, "gaussian", 0.05, 0.25),
        ParameterSpec("payload_mass", 0.2, 1.2, 0.1, "gaussian", 0.05, 0.25),
    ), feature_dim=0, hidden=(8, 8))
    port = solver.clone_particle(original, np.random.default_rng(seed), perturbation=0.2)
    np.testing.assert_allclose(port.left, author.dist.mean)
    np.testing.assert_allclose(port.right, author.dist.diag_std)
    assert author.twin


def test_adaptsim_outer_population_replaces_worst_and_charges_every_evaluation() -> None:
    solver = AdaptSimSolver((
        ParameterSpec("mass", 0.0, 2.0, 0.1, "gaussian", 0.05, 0.25),
    ), feature_dim=1, hidden=(8, 8))
    training_calls: list[tuple[float, float | None, int]] = []
    evaluation_calls: list[tuple[float, int]] = []

    def train(distribution, warm_start, seed):
        policy = float(distribution.left[0])
        training_calls.append((policy, warm_start, seed))
        return AdaptationTraining(policy, 7)

    def evaluate(policy, target, seed):
        evaluation_calls.append((policy, seed))
        return AdaptationEvaluation(1.0 - abs(policy - target), np.array([policy - target]), 9)

    initial = [
        ParameterDistribution(np.array([0.3]), np.array([0.1]), "gaussian"),
        ParameterDistribution(np.array([0.9]), np.array([0.1]), "gaussian"),
    ]
    result = AdaptSimMetaTraining(
        solver, train, lambda seed: 0.9, evaluate, seed=53,
    ).run_population(
        initial, iterations=2, inner_episodes=1, adaptation_steps=1,
        kill_frequency=1, spawn_reward_threshold=0.5,
        meta_spawn_perturbation=0.1,
        batch_size=2, update_every=1, exploration_probability=1.0,
    )
    assert result.inference_updates == 1
    assert [row["inner_inference_updates"] for row in result.records] == [0, 1]
    assert len(training_calls) == len(evaluation_calls) == result.simulated_target_rollouts == 8
    assert result.simulation_steps == 8 * 7
    assert result.simulated_target_control_steps == 8 * 9
    assert result.records[0]["replacement"] is None
    replacement = result.records[1]["replacement"]
    assert replacement is not None
    assert replacement["replaced_index"] == 0
    assert replacement["parent_index"] == 1
    assert len(result.final_particles) == 2
    assert result.final_particles[0].left[0] != initial[0].left[0]
    for record in result.records:
        assert len({row["simulated_target_reset_seed"] for row in record["meta_evaluations"]}) == 1
    assert sum(row["simulation_steps"] for record in result.records
               for row in record["meta_evaluations"]) == 4 * 7
