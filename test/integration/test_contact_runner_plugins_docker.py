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
from genedynamics.experiments.runner import main as runner_main, register_all_plugins
from genedynamics.experiments.plugins.methods.contact_receding import make_controller


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


def test_runner_dry_run_supports_isolated_multi_seed_multi_suite(
    monkeypatch, tmp_path, capsys,
):
    monkeypatch.setattr("sys.argv", [
        "runner",
        str(ROOT / "configs/arm/surface_scan/main/full_mdac.yaml"),
        "--development-root", str(tmp_path / "p7"),
        "--seeds", "0", "1",
        "--suites", "rigid_convex", "soft_convex",
        "--dry-run",
    ])
    runner_main()
    output = capsys.readouterr().out
    assert "Suites: ['rigid_convex', 'soft_convex']" in output
    assert "Seeds: [0, 1]" in output
    assert str(tmp_path / "p7/results/arm/surface_scan/main/full_mdac") in output
    assert "Run class: development" in output


@pytest.mark.parametrize(
    "filename,controller_method,stiffness_mode",
    [
        ("no_controllability_geometry.yaml", "mdac_component_gate", "log_spd"),
        ("no_retraction.yaml", "mdac_controllable_no_retraction", "log_spd"),
        ("no_stiffness.yaml", "mdac_controllable_no_stiffness", "none"),
        ("fixed_or_euclidean_stiffness.yaml",
         "mdac_controllable_euclid_stiffness", "euclid"),
    ],
)
def test_surface_ablation_yaml_dispatches_controller_and_stiffness_chart(
    monkeypatch, filename, controller_method, stiffness_mode,
):
    from genedynamics.experiments.plugins.environments._contact_task import (
        _env_kwargs,
    )

    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/surface_scan/ablation" / filename
    )
    captured = {}

    def fake_make_controller(task, method, **kwargs):
        captured.update(task=task, method=method, kwargs=kwargs)
        return kwargs["model_env"], object()

    monkeypatch.setattr(
        "genedynamics.experiments.plugins.methods.contact_receding.make_controller",
        fake_make_controller,
    )

    class Env:
        _experiment_task = "manipulator_surface_scan"
        _experiment_execution_env = None

    planner = FullMDACMethodPlugin().create_planner(
        Env(), None, {
            **cfg.method_params,
            "task": cfg.env_name,
            "n_steps": cfg.n_steps,
        },
    )
    assert planner.controller_method == controller_method
    assert captured["method"] == controller_method
    assert _env_kwargs(cfg.env_name, {
        **cfg.env_params,
        "_controller_method": controller_method,
    })["stiffness_mode"] == stiffness_mode


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
    "filename,controller_method,learned_reliability",
    [
        ("no_rl_prior.yaml", "mdac_controllable_gate_no_rl_prior", True),
        ("no_learned_reliability.yaml", "mdac_controllable_gate", False),
    ],
)
def test_peg_ablation_yaml_dispatches_exact_component_contract(
    monkeypatch, filename, controller_method, learned_reliability,
):
    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/peg_insert/ablation" / filename
    )
    captured = {}

    def fake_make_controller(task, method, **kwargs):
        captured.update(task=task, method=method, kwargs=kwargs)
        return kwargs["model_env"], object()

    monkeypatch.setattr(
        "genedynamics.experiments.plugins.methods.contact_receding.make_controller",
        fake_make_controller,
    )

    class Env:
        _experiment_task = "manipulator_peg_insert"
        _experiment_execution_env = None

    planner = FullMDACMethodPlugin().create_planner(
        Env(), None, {
            **cfg.method_params,
            "task": cfg.env_name,
            "n_steps": cfg.n_steps,
        },
    )
    assert planner.controller_method == controller_method
    assert captured["method"] == controller_method
    assert planner.component_contract["learned_reliability"] is learned_reliability
    assert (
        planner.component_contract["policy_ckpt"] is not None
    ) is planner.component_contract["mdac_flags"]["use_rl_prior"]
    assert (
        planner.component_contract["reliability_ckpt"] is not None
    ) is learned_reliability


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


def test_h1_suite_checkpoint_resolution_and_truthful_components(monkeypatch):
    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/humanoid/push_to_line/main/full_mdac.yaml"
    )
    captured = {}

    def fake_make_controller(task, method, **kwargs):
        captured.update(task=task, method=method, kwargs=kwargs)
        return kwargs["model_env"], object()

    monkeypatch.setattr(
        "genedynamics.experiments.plugins.methods.contact_receding.make_controller",
        fake_make_controller,
    )

    class Env:
        _experiment_task = "humanoid_box_push"
        _experiment_execution_env = None

    planner = FullMDACMethodPlugin().create_planner(Env(), None, {
        **cfg.method_params, "task": cfg.env_name, "suite": "p4_walk_push",
    })
    assert planner.component_contract["policy_ckpt"].endswith(
        "walk_ppo_seed0.pkl"
    )
    assert planner.component_contract["rl_prior"] is True
    assert planner.component_contract["learned_reliability"] is True

    mbo = ModelBasedOnlyMethodPlugin().create_planner(Env(), None, {
        "task": cfg.env_name, "suite": "p4_walk_push",
        "controller_method": "mdac_controllable",
    })
    assert mbo.component_contract["rl_prior"] is False
    assert mbo.component_contract["learned_reliability"] is False


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
    with open(seed_dir / "trajectory" / "trajectory.json") as handle:
        trajectory_payload = __import__("json").load(handle)
    assert trajectory_payload["schema_version"] == 2
    assert trajectory_payload["task_signals"]
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
    cfg.seeds = [0]
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
    result_path = tmp_path / "level_id_wide" / "seed_0" / "results.json"
    with open(result_path) as handle:
        result = __import__("json").load(handle)
    diagnostics = result["diagnostics"]
    assert "prior_acceptance_rate" in diagnostics
    assert "selected_revalidated_safe" in diagnostics


@pytest.mark.requires_jax
@pytest.mark.requires_brax
@pytest.mark.slow
@pytest.mark.parametrize("suite_name", [
    "p1_force_15n", "p3_unjam", "p4_walk_push",
])
@pytest.mark.parametrize("config_path", [
    "configs/humanoid/push_to_line/main/full_mdac.yaml",
    "configs/humanoid/push_to_line/baseline/model_based_only.yaml",
    "configs/humanoid/push_to_line/baseline/standalone_rl.yaml",
    "configs/humanoid/push_to_line/baseline/dial.yaml",
    "configs/humanoid/push_to_line/baseline/mppi.yaml",
    "configs/humanoid/push_to_line/baseline/pegasusflow.yaml",
    "configs/humanoid/push_to_line/baseline/issa.yaml",
    "configs/humanoid/push_to_line/baseline/atacom.yaml",
])
def test_h1_paper_algorithm_checkpoints_execute_one_development_step(
    tmp_path, config_path, suite_name,
):
    """Exercise every H1 algorithm and all three checkpoint/action schemas."""
    cfg = ExperimentConfig.from_yaml(ROOT / config_path)
    cfg.output_dir = tmp_path / cfg.name
    cfg.seeds = [0]
    cfg.n_steps = 1
    cfg.visualizations = []
    cfg.suites = [
        next(item for item in cfg.suites if item["name"] == suite_name)
    ]
    cfg.method_params.update({
        "Nsample": 2, "Hsample": 4, "Hnode": 3,
        "Ndiffuse": 1, "Ndiffuse_init": 1,
        "prior_stochastic_samples": 1, "prior_atacom_samples": 0,
        "n_dirs": 2, "n_iters": 1,
    })
    runner = ExperimentRunner(cfg)
    register_all_plugins(runner)
    results = runner.run_all()
    assert len(results) == 1
    seed_dir = cfg.output_dir / f"level_{suite_name}" / "seed_0"
    with open(seed_dir / "results.json") as handle:
        persisted = __import__("json").load(handle)
    assert persisted["component_contract"]["controller_method"]
    trajectory = seed_dir / "trajectory" / "trajectory.json"
    assert trajectory.is_file()


@pytest.mark.requires_jax
@pytest.mark.requires_brax
@pytest.mark.slow
@pytest.mark.parametrize("config_path,suite_name", [
    ("configs/humanoid/push_to_line/ablation/no_rl_prior.yaml", "p3_unjam"),
    ("configs/humanoid/push_to_line/ablation/no_learned_reliability.yaml", "p2_push_ood"),
    ("configs/humanoid/push_to_line/ablation/no_tangent.yaml", "p3_unjam"),
    ("configs/humanoid/push_to_line/ablation/no_retraction.yaml", "p3_unjam"),
    ("configs/humanoid/push_to_line/ablation/no_stiffness.yaml", "p1_force_15n"),
])
def test_h1_causal_ablation_executes_one_development_step(
    tmp_path, config_path, suite_name,
):
    cfg = ExperimentConfig.from_yaml(ROOT / config_path)
    cfg.output_dir = tmp_path / cfg.name
    cfg.seeds = [0]
    cfg.n_steps = 1
    cfg.visualizations = []
    cfg.suites = [
        next(item for item in cfg.suites if item["name"] == suite_name)
    ]
    cfg.method_params.update({
        "Nsample": 2, "Hsample": 4, "Hnode": 3,
        "Ndiffuse": 1, "Ndiffuse_init": 1,
        "prior_stochastic_samples": 1, "prior_atacom_samples": 0,
    })
    runner = ExperimentRunner(cfg)
    register_all_plugins(runner)
    results = runner.run_all()
    assert len(results) == 1
    result_file = (
        cfg.output_dir / f"level_{suite_name}" / "seed_0" / "results.json"
    )
    with open(result_file) as handle:
        persisted = __import__("json").load(handle)
    assert persisted["component_contract"]["controller_method"]
