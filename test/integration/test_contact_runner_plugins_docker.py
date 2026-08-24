"""Unified contact-task plugin smoke tests (real Brax/MJX, CPU Docker)."""

from pathlib import Path

import pytest

from genedynamics.experiments.framework import ExperimentConfig, ExperimentRunner
from genedynamics.experiments.plugins.environments import (
    HumanoidBoxPushPlugin,
    ManipulatorPegInsertPlugin,
    ManipulatorSurfaceScanPlugin,
)
from genedynamics.experiments.plugins.methods import (
    ATACOMContactMethodPlugin,
    DIALContactMethodPlugin,
    FullMDACMethodPlugin,
    ISSAContactMethodPlugin,
    ModelBasedOnlyMethodPlugin,
    MPPIContactMethodPlugin,
    PegasusFlowContactMethodPlugin,
    StandaloneRLMethodPlugin,
)
from genedynamics.experiments.runner import register_all_plugins
from genedynamics.solvers.single.mdac.experiment import make_controller


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.requires_jax
@pytest.mark.requires_brax
@pytest.mark.parametrize(
    "config_path,plugin,controller_method",
    [
        ("configs/arm/surface_scan/baseline/dial.yaml", ManipulatorSurfaceScanPlugin, "dial"),
        ("configs/arm/peg_insert/baseline/dial.yaml", ManipulatorPegInsertPlugin, "dial"),
        ("configs/humanoid/push_to_line/baseline/dial.yaml", HumanoidBoxPushPlugin, "dial"),
    ],
)
def test_contact_plugin_creates_structured_reset_and_reuses_model_env(
    config_path, plugin, controller_method
):
    cfg = ExperimentConfig.from_yaml(ROOT / config_path)
    suite_cfg = cfg.for_suite(cfg.suites[0])
    adapter = plugin()
    env = adapter.create_env({
        **suite_cfg.env_params,
        "_experiment_seed": suite_cfg.seeds[0],
        "_controller_method": controller_method,
        "_execution_env_params": suite_cfg.execution_env_params,
    })
    x0 = adapter.reset_state(env, suite_cfg.seeds[0])
    assert hasattr(x0, "pipeline_state")
    assert int(env.action_size) > 0
    model_env, solver = make_controller(
        suite_cfg.env_name,
        controller_method,
        model_env=env,
        execution_env=getattr(env, "_experiment_execution_env", None),
        Nsample=2,
        Hsample=4,
        Hnode=2,
        Ndiffuse=1,
        Ndiffuse_init=1,
    )
    assert model_env is env
    assert solver is not None


@pytest.mark.requires_jax
@pytest.mark.requires_brax
def test_peg_plugin_preserves_hidden_execution_environment():
    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/peg_insert/baseline/dial.yaml"
    )
    suite = next(item for item in cfg.suites if item["name"] == "ood_pose")
    suite_cfg = cfg.for_suite(suite)
    adapter = ManipulatorPegInsertPlugin()
    env = adapter.create_env({
        **suite_cfg.env_params,
        "_experiment_seed": 10,
        "_controller_method": "dial",
        "_execution_env_params": suite_cfg.execution_env_params,
    })
    execution_env = adapter.execution_env(env)
    assert execution_env is not env
    assert int(execution_env.action_size) == int(env.action_size)
    assert int(execution_env.observation_size) == int(env.observation_size)


@pytest.mark.requires_jax
@pytest.mark.requires_brax
@pytest.mark.slow
@pytest.mark.parametrize(
    "config_path,plugin",
    [
        ("configs/arm/surface_scan/baseline/dial.yaml", ManipulatorSurfaceScanPlugin),
        ("configs/arm/peg_insert/baseline/dial.yaml", ManipulatorPegInsertPlugin),
        ("configs/humanoid/push_to_line/baseline/dial.yaml", HumanoidBoxPushPlugin),
    ],
)
def test_every_canonical_contact_suite_builds_and_resets(config_path, plugin):
    cfg = ExperimentConfig.from_yaml(ROOT / config_path)
    adapter = plugin()
    for suite in cfg.suites:
        suite_cfg = cfg.for_suite(suite)
        env = adapter.create_env({
            **suite_cfg.env_params,
            "_experiment_seed": suite_cfg.seeds[0],
            "_controller_method": "dial",
            "_execution_env_params": suite_cfg.execution_env_params,
        })
        state = adapter.reset_state(env, suite_cfg.seeds[0])
        execution_env = adapter.execution_env(env)
        assert hasattr(state, "pipeline_state"), suite["name"]
        assert int(env.action_size) == int(execution_env.action_size), suite["name"]
        assert int(env.observation_size) == int(execution_env.observation_size), suite["name"]


def test_all_eight_algorithm_plugins_are_independently_registered(tmp_path):
    cfg = ExperimentConfig(
        name="registry", output_dir=tmp_path,
        env_name="manipulator_peg_insert", method="dial",
        metrics=[], visualizations=[], obstacle_config={"generator": "none"},
    )
    runner = ExperimentRunner(cfg)
    register_all_plugins(runner)
    names = set(runner.registry.list_plugins("method"))
    assert {
        "full_mdac", "model_based_only", "standalone_rl", "dial", "mppi",
        "pegasusflow", "issa", "atacom",
    }.issubset(names)


def test_full_mdac_rejects_missing_learned_component_contract():
    class Env:
        _experiment_task = "manipulator_peg_insert"

    with pytest.raises(ValueError, match="frozen learned components"):
        FullMDACMethodPlugin().create_planner(
            Env(), None,
            {"task": "manipulator_peg_insert",
             "controller_method": "mdac_controllable_gate"},
        )


@pytest.mark.parametrize(
    "plugin,controller_method",
    [
        (FullMDACMethodPlugin, "mdac"),
        (ModelBasedOnlyMethodPlugin, "mdac_controllable"),
        (DIALContactMethodPlugin, "dial"),
        (MPPIContactMethodPlugin, "mppi"),
        (PegasusFlowContactMethodPlugin, "pegasusflow"),
        (ISSAContactMethodPlugin, "issa"),
        (ATACOMContactMethodPlugin, "atacom"),
        (StandaloneRLMethodPlugin, "rl"),
    ],
)
def test_each_contact_plugin_dispatches_its_own_controller(
    monkeypatch, plugin, controller_method
):
    class Env:
        _experiment_task = "manipulator_peg_insert"
        _experiment_execution_env = None

    captured = {}

    def fake_make_controller(task, method, **kwargs):
        captured.update(task=task, method=method, kwargs=kwargs)
        return kwargs["model_env"], object()

    monkeypatch.setattr(
        "genedynamics.experiments.plugins.methods.contact_receding.make_controller",
        fake_make_controller,
    )
    config = {"task": "manipulator_peg_insert", "n_steps": 1}
    if plugin is FullMDACMethodPlugin:
        config.update(policy_ckpt="policy.pkl", reliability_ckpt="reliability.json")
    planner = plugin().create_planner(Env(), None, config)
    assert planner.controller_method == controller_method
    assert captured["method"] == controller_method
    assert captured["kwargs"]["model_env"] is planner.env


@pytest.mark.requires_jax
@pytest.mark.requires_brax
@pytest.mark.slow
@pytest.mark.parametrize(
    "config_path,suite_name,metric_name,headline",
    [
        ("configs/arm/surface_scan/baseline/dial.yaml", "rigid_cylinder",
         "arm_surface_scan_metrics", "trajectory_path_coverage"),
        ("configs/arm/peg_insert/baseline/dial.yaml", "id_wide",
         "peg_insert_metrics", "safe_insertion_success"),
        ("configs/humanoid/push_to_line/baseline/dial.yaml", "p3_unjam",
         "humanoid_box_push_metrics", "goal_error"),
    ],
)
def test_unified_runner_executes_and_saves_one_contact_step(
    tmp_path, config_path, suite_name, metric_name, headline
):
    cfg = ExperimentConfig.from_yaml(ROOT / config_path)
    cfg.name = f"{suite_name}_one_step"
    cfg.output_dir = tmp_path
    cfg.seeds = [0]
    cfg.n_steps = 1
    cfg.visualizations = []
    cfg.suites = [next(item for item in cfg.suites if item["name"] == suite_name)]
    cfg.method_params.update({
        "Nsample": 2, "Hsample": 4, "Hnode": 2,
        "Ndiffuse": 1, "Ndiffuse_init": 1,
    })
    runner = ExperimentRunner(cfg)
    register_all_plugins(runner)
    results = runner.run_all()
    assert len(results) == 1
    seed_dir = tmp_path / f"level_{suite_name}" / "seed_0"
    assert (seed_dir / "results.json").is_file()
    assert (seed_dir / "trajectory" / "trajectory.json").is_file()
    with open(seed_dir / "results.json") as handle:
        payload = __import__("json").load(handle)
    assert headline in payload["metrics"][metric_name]
    with open(tmp_path / "overall_summary.json") as handle:
        overall = __import__("json").load(handle)
    aggregate = overall["metrics"][metric_name][headline]
    assert aggregate["n"] == 1
    assert aggregate["missing"] == 0
    assert aggregate["mean"] is not None


@pytest.mark.requires_jax
@pytest.mark.requires_brax
@pytest.mark.slow
def test_full_mdac_result_persists_receding_diagnostics(tmp_path):
    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/peg_insert/main/full_mdac.yaml"
    )
    cfg.output_dir = tmp_path
    cfg.seeds = [10]
    cfg.n_steps = 1
    cfg.visualizations = []
    cfg.suites = [next(item for item in cfg.suites if item["name"] == "id_wide")]
    cfg.method_params.update({
        "Nsample": 2, "Hsample": 4, "Hnode": 2,
        "Ndiffuse": 1, "Ndiffuse_init": 1,
        "prior_stochastic_samples": 1, "prior_atacom_samples": 0,
    })
    runner = ExperimentRunner(cfg)
    register_all_plugins(runner)
    runner.run_all()
    result_path = tmp_path / "level_id_wide" / "seed_10" / "results.json"
    with open(result_path) as handle:
        result = __import__("json").load(handle)
    diagnostics = result["diagnostics"]
    assert "prior_acceptance_rate" in diagnostics
    assert "selected_revalidated_safe" in diagnostics
