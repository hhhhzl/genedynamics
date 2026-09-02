from pathlib import Path

import pytest

from genedynamics.experiments.framework.config import ExperimentConfig
from genedynamics.experiments.framework.experiment import ExperimentRunner


ROOT = Path(__file__).resolve().parents[2]


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_yaml_base_and_named_suite_resolution(tmp_path: Path):
    _write(
        tmp_path / "_base.yaml",
        """
name: contact
output_dir: results/contact
env_name: manipulator_surface_scan
method: mga
env_params: {dt: 0.02, medium: rigid}
execution_env_params: {sensor_delay_steps: 0}
method_params: {Nsample: 64, Hsample: 16, nested: {a: 1}}
seeds: [10, 11]
visualizations: []
metrics: []
suites:
  - name: soft_unseen
    level: unseen
    env_params: {medium: soft}
    execution_env_params: {sensor_delay_steps: 1}
    method_params: {nested: {b: 2}}
""",
    )
    child = _write(
        tmp_path / "mga.yaml",
        """
base: _base.yaml
name: mga
method_params: {Nsample: 64}
""",
    )

    cfg = ExperimentConfig.from_yaml(child)
    assert cfg.name == "mga"
    assert cfg.seeds == [10, 11]
    assert cfg.validate() == []

    suite_cfg = cfg.for_suite(cfg.suites[0])
    assert suite_cfg.suites == []
    assert suite_cfg.obstacle_levels == ["soft_unseen"]
    assert suite_cfg.env_params == {"dt": 0.02, "medium": "soft", "level": "unseen"}
    assert suite_cfg.execution_env_params["sensor_delay_steps"] == 1
    assert suite_cfg.method_params["nested"] == {"a": 1, "b": 2}
    assert cfg.env_params["medium"] == "rigid"


def test_yaml_base_cycle_is_rejected(tmp_path: Path):
    a = _write(tmp_path / "a.yaml", "base: b.yaml\n")
    _write(tmp_path / "b.yaml", "base: a.yaml\n")
    with pytest.raises(ValueError, match="Cyclic experiment config"):
        ExperimentConfig.from_yaml(a)


def test_duplicate_and_unsafe_suite_names_are_rejected(tmp_path: Path):
    cfg = ExperimentConfig(
        name="bad",
        output_dir=tmp_path,
        env_name="dummy",
        method="dummy",
        suites=[
            {"name": "same"},
            {"name": "same"},
            {"name": "../escape"},
        ],
    )
    errors = cfg.validate()
    assert any("Duplicate suite names" in error for error in errors)
    assert any("not path-safe" in error for error in errors)


def test_suite_cannot_change_fairness_budget(tmp_path: Path):
    cfg = ExperimentConfig(
        name="bad_budget",
        output_dir=tmp_path,
        env_name="dummy",
        method="dummy",
        suites=[{"name": "suite", "method_params": {"Nsample": 8}}],
    )
    assert any("cannot change fairness budget" in error for error in cfg.validate())


def test_existing_config_keeps_legacy_level_protocol():
    cfg = ExperimentConfig.from_yaml(ROOT / "configs/single_2d/mbd.yaml")
    assert cfg.name == "single2d_mbd_default"
    assert cfg.env_name == "single_integrator_box_2d"
    assert cfg.method == "mbd"
    assert cfg.obstacle_levels == list(range(11))
    assert cfg.seeds == list(range(10))
    assert cfg.suites == []
    assert cfg.method_params["num_modes"] == 20
    assert cfg.validate() == []


def test_two_suites_by_two_seeds_expand_without_mutating_config(tmp_path: Path):
    cfg = ExperimentConfig(
        name="matrix",
        output_dir=tmp_path,
        env_name="dummy",
        method="dummy",
        seeds=[3, 7],
        suites=[
            {"name": "first", "level": "plane", "env_params": {"medium": "rigid"}},
            {"name": "second", "level": "cylinder", "env_params": {"medium": "soft"}},
        ],
        metrics=[],
        visualizations=[],
        auto_report=False,
    )
    runner = ExperimentRunner(cfg)
    calls = []
    runner._save_protocol_manifest = lambda: None
    runner._save_result = lambda result: None
    runner._save_summary = lambda results: None

    def run_one(level, seed):
        calls.append((level, seed, runner.config.env_params["medium"],
                      runner.config.metadata["suite"]))
        return {"level": level, "seed": seed}

    runner.run_single_experiment = run_one
    results = runner.run_all()
    assert len(results) == 4
    assert calls == [
        ("first", 3, "rigid", "first"),
        ("first", 7, "rigid", "first"),
        ("second", 3, "soft", "second"),
        ("second", 7, "soft", "second"),
    ]
    assert runner.config is cfg
    assert cfg.env_params == {}
    assert [suite["name"] for suite in cfg.suites] == ["first", "second"]


def test_resume_reuses_only_complete_matching_results(tmp_path: Path):
    cfg = ExperimentConfig(
        name="resumable",
        output_dir=tmp_path,
        env_name="dummy",
        method="dummy",
        seeds=[0, 1],
        suites=[{"name": "first", "level": "plane"}],
        metrics=[],
        visualizations=[],
        auto_report=False,
    )
    suite_cfg = cfg.for_suite(cfg.suites[0])
    completed_dir = tmp_path / "level_first" / "seed_0"
    completed_dir.mkdir(parents=True)
    completed_snapshot = suite_cfg.to_dict()
    completed_snapshot["seeds"] = [0]
    completed = {
        "level": "first",
        "seed": 0,
        "planning_time": 1.25,
        "metrics": {},
        "config_snapshot": completed_snapshot,
    }
    import json
    (completed_dir / "results.json").write_text(
        json.dumps(completed, default=str), encoding="utf-8"
    )

    runner = ExperimentRunner(cfg)
    runner._save_protocol_manifest = lambda: None
    runner._save_result = lambda result: None
    runner._save_summary = lambda results: None
    calls = []

    def run_one(level, seed):
        calls.append((level, seed))
        return {
            "level": level,
            "seed": seed,
            "planning_time": 2.0,
            "metrics": {},
            "config_snapshot": runner.config.to_dict(),
        }

    runner.run_single_experiment = run_one
    results = runner.run_all(resume=True)
    assert [(result["level"], result["seed"]) for result in results] == [
        ("first", 0),
        ("first", 1),
    ]
    assert calls == [("first", 1)]


def test_resume_reruns_config_mismatch(tmp_path: Path):
    cfg = ExperimentConfig(
        name="resumable",
        output_dir=tmp_path,
        env_name="dummy",
        method="dummy",
        seeds=[0],
        metrics=[],
        visualizations=[],
        auto_report=False,
    )
    completed_dir = tmp_path / "level_0" / "seed_0"
    completed_dir.mkdir(parents=True)
    stale = {
        "level": 0,
        "seed": 0,
        "planning_time": 1.0,
        "metrics": {},
        "config_snapshot": {"name": "stale"},
    }
    import json
    (completed_dir / "results.json").write_text(
        json.dumps(stale), encoding="utf-8"
    )

    runner = ExperimentRunner(cfg)
    runner._save_protocol_manifest = lambda: None
    runner._save_result = lambda result: None
    runner._save_summary = lambda results: None
    calls = []
    runner.run_single_experiment = lambda level, seed: (
        calls.append((level, seed))
        or {"level": level, "seed": seed, "planning_time": 2.0, "metrics": {}}
    )
    runner.run_all(resume=True)
    assert calls == [(0, 0)]


def test_development_root_mirrors_canonical_output_and_marks_provenance(
    tmp_path: Path,
):
    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/surface_scan/baseline/dial.yaml"
    )
    canonical = cfg.output_dir
    cfg.use_development_output_root(tmp_path / "p7")
    assert cfg.output_dir == (
        tmp_path / "p7/results/arm/surface_scan/baseline/dial"
    ).resolve()
    assert cfg.metadata["run_class"] == "development"
    assert cfg.metadata["formal_seeds"] is False
    assert cfg.metadata["canonical_output_dir"] == str(canonical)


@pytest.mark.parametrize(
    "filename,controller_method,component",
    [
        ("no_controllability_geometry.yaml", "mga_component_gate",
         "use_controllability_geometry"),
        ("no_retraction.yaml", "mga_controllable_no_retraction",
         "use_retraction"),
        ("no_stiffness.yaml", "mga_controllable_no_stiffness",
         "use_stiffness"),
        ("fixed_or_euclidean_stiffness.yaml",
         "mga_controllable_euclid_stiffness", "log_spd_stiffness"),
    ],
)
def test_surface_ablation_inherits_full_contract_and_changes_one_component(
    filename: str, controller_method: str, component: str,
):
    from genedynamics.solvers.single.mga.core.method_registry import (
        diff_flags,
        resolve_method,
    )

    full = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/surface_scan/main/mga.yaml"
    )
    ablation = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/surface_scan/ablation" / filename
    )
    assert ablation.method == "mga"
    assert ablation.seeds == full.seeds
    assert ablation.suites == full.suites
    assert ablation.n_steps == full.n_steps
    assert ablation.method_params["controller_method"] == controller_method
    inherited = dict(ablation.method_params)
    inherited["controller_method"] = full.method_params["controller_method"]
    assert inherited == full.method_params
    assert diff_flags(
        resolve_method(full.method_params["controller_method"]),
        resolve_method(controller_method),
    ) == (component,)
    assert ablation.metadata["ablation_of"] == "mga"
    assert ablation.metadata["ablated_component"] == component


def test_peg_ablation_configs_change_only_the_named_component():
    from genedynamics.solvers.single.mga.core.method_registry import (
        diff_flags,
        resolve_method,
    )

    full = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/peg_insert/main/mga.yaml"
    )
    no_prior = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/peg_insert/ablation/no_rl_prior.yaml"
    )
    no_reliability = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/peg_insert/ablation/no_learned_reliability.yaml"
    )
    for ablation in (no_prior, no_reliability):
        assert ablation.method == "mga"
        assert ablation.seeds == full.seeds
        assert ablation.suites == full.suites
        assert ablation.n_steps == full.n_steps
        assert {
            key: ablation.method_params[key]
            for key in ("Nsample", "Hsample", "Hnode", "Ndiffuse", "Ndiffuse_init")
        } == {
            key: full.method_params[key]
            for key in ("Nsample", "Hsample", "Hnode", "Ndiffuse", "Ndiffuse_init")
        }
        assert ablation.metadata["ablation_of"] == "mga"

    assert diff_flags(
        resolve_method(full.method_params["controller_method"]),
        resolve_method(no_prior.method_params["controller_method"]),
    ) == ("use_rl_prior",)
    restored_prior = dict(no_prior.method_params)
    restored_prior.update({
        "controller_method": full.method_params["controller_method"],
        "policy_ckpt": full.method_params["policy_ckpt"],
    })
    assert restored_prior == full.method_params

    restored_reliability = dict(no_reliability.method_params)
    restored_reliability.update({
        "learned_reliability": True,
        "reliability_ckpt": full.method_params["reliability_ckpt"],
    })
    assert restored_reliability == full.method_params
