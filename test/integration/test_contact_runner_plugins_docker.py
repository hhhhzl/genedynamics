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
    MGAMethodPlugin,
    ISSAContactMethodPlugin,
    ModelBasedOnlyMethodPlugin,
    MPPIContactMethodPlugin,
    PegasusFlowContactMethodPlugin,
    StandaloneRLMethodPlugin,
)
from genedynamics.experiments.runner import (
    _formal_readiness_errors,
    main as runner_main,
    register_all_plugins,
)
from genedynamics.experiments.plugins.methods.contact_receding import make_controller


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("resume", [False, True])
@pytest.mark.parametrize("with_suites", [False, True])
def test_aborted_attempt_preflight_preserves_all_output_before_first_seed(
    tmp_path, monkeypatch, resume, with_suites,
):
    """No physics: a later failed seed protects the complete output tree."""
    import json

    cfg = ExperimentConfig(
        name="failed_attempt_guard", output_dir=tmp_path,
        env_name="dummy", method="dummy", seeds=[110, 111],
        obstacle_levels=["legacy"], metrics=[], visualizations=[],
        suites=[{"name": "first"}, {"name": "second"}] if with_suites else [],
        auto_report=False,
    )
    failed_level = "second" if with_suites else "legacy"
    failed_dir = tmp_path / f"level_{failed_level}" / "seed_111"
    (failed_dir / "trajectory").mkdir(parents=True)
    manifest = tmp_path / "protocol_manifest.json"
    manifest.write_text('{"original_protocol": "do not overwrite"}\n')
    result_path = failed_dir / "results.json"
    result_path.write_text(json.dumps({
        "execution_status": {"state": "aborted_unrecoverable"},
        "original_attempt": True,
    }))
    trace_path = failed_dir / "trajectory" / "trajectory.json"
    trace_path.write_text('{"actions": [], "original_prefix": true}\n')
    original = {path: path.read_bytes() for path in tmp_path.rglob("*")
                if path.is_file()}
    original_paths = set(tmp_path.rglob("*"))
    runner = ExperimentRunner(cfg)
    calls = []
    monkeypatch.setattr(runner, "run_single_experiment",
                        lambda *args: calls.append(args))
    with pytest.raises(ValueError, match="Refusing to resume/overwrite aborted"):
        runner.run_all(resume=resume)
    assert calls == []
    assert runner.results == []
    assert runner.config is cfg
    assert set(tmp_path.rglob("*")) == original_paths
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("executed_steps", [0, 2])
@pytest.mark.parametrize("artifact_failure", [False, True])
def test_h1_explicit_rejection_saves_prefix_and_counts_failed_seed(
    tmp_path, monkeypatch, executed_steps, artifact_failure,
):
    """Runner contract with synthetic prefixes; this does not test physics."""
    import json
    from types import SimpleNamespace
    import numpy as np
    from genedynamics.core.types import ExecutionRejected
    from genedynamics.solvers.common.receding_horizon import RecedingHorizonResult
    from genedynamics.experiments.plugins.methods.contact_receding import _ContactPlanner
    from genedynamics.experiments.utils.metrics import collect_results

    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/humanoid/push_to_line/baseline/dial.yaml"
    )
    cfg.output_dir, cfg.seeds, cfg.n_steps = tmp_path, [110], 3
    cfg.suites = [dict(cfg.suites[0], n_steps=3)]
    cfg.visualizations, cfg.auto_report = [], False
    runner = ExperimentRunner(cfg)
    register_all_plugins(runner)
    method = runner.registry.get_plugin("method", "dial")

    def create_planner(env, energy, config):
        def run_receding(initial, steps, rng, **kwargs):
            error = ExecutionRejected("invalid sealed geometry", {"entry_valid": False})
            action = np.zeros(env.action_size, np.float32)
            error.partial_result = RecedingHorizonResult(
                states=[initial] * (executed_steps + 1),
                actions=[action] * executed_steps,
                infos=[{"executed": 1}] * executed_steps,
            )
            error.partial_states_complete = True
            error.rejected_action = action
            error.rejected_info = {"rejected": 1}
            error.rejected_step = executed_steps
            error.requested_steps = steps
            raise error
        return _ContactPlanner(
            solver=SimpleNamespace(run_receding=run_receding), env=env,
            execution_env=env, task="humanoid_box_push", controller_method="dial",
            n_steps=3, seed=110, component_contract={"synthetic_contract_test": True},
        )

    monkeypatch.setattr(method, "create_planner", create_planner)
    metric_calls = []
    def prefix_metrics(trajectory, *args, **kwargs):
        metric_calls.append(len(trajectory.actions))
        if artifact_failure:
            raise RuntimeError("synthetic prefix metric failure")
        return {"humanoid_box_push_metrics": {
            "safe_success": 1., "task_success": 1., "physics_force_peak": 2.,
        }}
    monkeypatch.setattr(runner, "_compute_metrics", prefix_metrics)
    if artifact_failure:
        cfg.visualizations = ["synthetic_failure"]
        def broken_visuals(*args, **kwargs):
            raise RuntimeError("synthetic prefix visual failure")
        monkeypatch.setattr(runner, "_generate_visualizations", broken_visuals)
    attempts = runner.run_all()
    assert len(attempts) == 1
    assert metric_calls == ([executed_steps] if executed_steps else [])
    suite = cfg.suites[0]["name"]
    seed_dir = tmp_path / f"level_{suite}" / "seed_110"
    payload = json.loads((seed_dir / "results.json").read_text())
    trace = json.loads((seed_dir / "trajectory" / "trajectory.json").read_text())
    assert payload["execution_status"]["state"] == "aborted_unrecoverable"
    assert payload["execution_status"]["executed_steps"] == executed_steps
    assert payload["execution_status"]["requested_steps"] == 3
    assert payload["metrics"]["humanoid_box_push_metrics"] == {
        "task_success": 0., "safe_success": 0.,
    }
    assert len(trace["states"]) == executed_steps + 1
    assert len(trace["actions"]) == len(trace["infos"]) == executed_steps
    assert trace["execution_status"]["rejected_info"] == {"rejected": 1}
    if executed_steps and not artifact_failure:
        assert payload["partial_metrics"]["humanoid_box_push_metrics"]["physics_force_peak"] == 2.
    else:
        assert payload["partial_metrics"] == {}
        assert "task_signals" not in trace
    if artifact_failure:
        assert payload["execution_status"]["visualization_error"] == "synthetic prefix visual failure"
        if executed_steps:
            assert payload["execution_status"]["partial_metrics_error"] == "synthetic prefix metric failure"
    overall = json.loads((tmp_path / "overall_summary.json").read_text())
    assert overall["total_experiments"] == 1
    summary = overall["metrics"]["humanoid_box_push_metrics"]["safe_success"]
    assert summary["n"] == 1 and summary["missing"] == 0 and summary["mean"] == 0.
    runner.config = cfg.for_suite(cfg.suites[0])
    with pytest.raises(ValueError, match="Refusing to resume/overwrite aborted"):
        runner._load_completed_result(suite, 110)
    records = collect_results([str(tmp_path)])
    assert len(records) == 1
    assert "physics_force_peak" not in records[0]["metrics"]["humanoid_box_push_metrics"]


@pytest.mark.parametrize("has_initializer", [False, True])
@pytest.mark.parametrize("is_planning_controller", [False, True])
def test_contact_plan_initializer_is_only_forwarded_to_planners(
    has_initializer, is_planning_controller,
):
    """The optional task seam must not change raw/tangent policy call contracts."""
    from types import SimpleNamespace
    import numpy as np
    from genedynamics.experiments.plugins.methods.contact_receding import (
        RecedingContactMethodPlugin, _ContactPlanner,
    )

    recorded = {}

    def run_receding(state, n_steps, rng, **kwargs):
        recorded.update(kwargs)
        return SimpleNamespace(
            states=[state, state], actions=[np.zeros(1)], infos=[], costs=[],
        )

    initialize = lambda state, nodes: nodes
    env = SimpleNamespace()
    if has_initializer:
        env.plan_initializer = initialize
    solver = SimpleNamespace(run_receding=run_receding)
    if is_planning_controller:
        solver.make_controller = lambda *args, **kwargs: None
    planner = _ContactPlanner(
        solver=solver, env=env, execution_env=env,
        task="humanoid_box_push", controller_method="dial",
        n_steps=1, seed=110, component_contract={},
    )
    RecedingContactMethodPlugin("dial", "dial").plan(
        planner, np.zeros(1), rng=None,
    )
    expected = dict(collect_states=True, synchronize_steps=True)
    if has_initializer and is_planning_controller:
        expected["initialize_plan"] = initialize
    assert recorded == expected


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
        "mga", "model_based_only", "standalone_rl", "dial", "mppi",
        "pegasusflow", "issa", "atacom",
    }.issubset(names)


def test_runner_dry_run_supports_isolated_multi_seed_multi_suite(
    monkeypatch, tmp_path, capsys,
):
    monkeypatch.setattr("sys.argv", [
        "runner",
        str(ROOT / "configs/arm/surface_scan/main/mga.yaml"),
        "--development-root", str(tmp_path / "p7"),
        "--seeds", "0", "1",
        "--suites", "rigid_convex", "soft_convex",
        "--dry-run",
    ])
    runner_main()
    output = capsys.readouterr().out
    assert "Suites: ['rigid_convex', 'soft_convex']" in output
    assert "Seeds: [0, 1]" in output
    assert str(tmp_path / "p7/results/arm/surface_scan/main/mga") in output
    assert "Run class: development" in output


def test_frozen_core_only_peg_protocol_allows_canonical_formal_execution():
    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/peg_insert/baseline/mppi.yaml"
    )
    assert cfg.metadata["protocol_status"] == "frozen_core_only"
    assert _formal_readiness_errors(cfg) == []

    cfg.use_development_output_root(ROOT / "results/_development/p1_guard_test")
    assert _formal_readiness_errors(cfg) == []


@pytest.mark.parametrize(
    "filename,controller_method,stiffness_mode",
    [
        ("no_controllability_geometry.yaml", "mga_component_gate", "log_spd"),
        ("no_retraction.yaml", "mga_controllable_no_retraction", "log_spd"),
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

    planner = MGAMethodPlugin().create_planner(
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


def test_mga_rejects_missing_learned_component_contract():
    class Env:
        _experiment_task = "manipulator_peg_insert"

    with pytest.raises(ValueError, match="frozen learned components"):
        MGAMethodPlugin().create_planner(
            Env(), None,
            {"task": "manipulator_peg_insert",
             "controller_method": "mga_controllable_gate"},
        )


def test_peg_insert_candidate_can_be_authoritative_only_for_development(
    monkeypatch,
):
    from types import SimpleNamespace

    candidate_metadata = {
        "task": "manipulator_peg_insert",
        "fit_scope": "development_fit_only",
        "performance_validated": False,
        "promotion_eligible": False,
    }

    def fake_make_controller(task, method, **kwargs):
        solver = SimpleNamespace(
            reliability_model=SimpleNamespace(metadata=candidate_metadata),
            reliability_validation_authoritative=bool(
                kwargs.get("reliability_validation_authoritative", False)
            ),
        )
        return kwargs["model_env"], solver

    monkeypatch.setattr(
        "genedynamics.experiments.plugins.methods.contact_receding.make_controller",
        fake_make_controller,
    )

    class Env:
        _experiment_task = "manipulator_peg_insert"
        _experiment_execution_env = None

    common = {
        "task": "manipulator_peg_insert",
        "controller_method": "mga_controllable_gate",
        "learned_reliability": True,
        "policy_ckpt": "policy.pkl",
        "reliability_ckpt": "candidate.json",
        "reliability_ood_policy": "model_based",
        "reliability_validation_authoritative": True,
        "experiment_formal_experiment": False,
    }
    with pytest.raises(ValueError, match="development-only PegInsert"):
        MGAMethodPlugin().create_planner(
            Env(), None, {**common, "experiment_run_class": "formal"}
        )

    planner = MGAMethodPlugin().create_planner(
        Env(), None, {**common, "experiment_run_class": "development"}
    )
    assert planner.component_contract["reliability_gate_authoritative"] is True
    assert (
        planner.component_contract["reliability_validation_authoritative"]
        is True
    )
    assert planner.component_contract["reliability_promotion_eligible"] is False


@pytest.mark.parametrize(
    "family,task,filename,controller_method,learned_reliability",
    [
        ("surface_scan", "manipulator_surface_scan", "no_rl_prior.yaml",
         "mga_controllable_gate_no_rl_prior", True),
        ("surface_scan", "manipulator_surface_scan",
         "no_learned_reliability.yaml", "mga_controllable_gate", False),
        ("peg_insert", "manipulator_peg_insert", "no_rl_prior.yaml",
         "mga_controllable_gate_no_rl_prior", False),
        ("peg_insert", "manipulator_peg_insert",
         "no_learned_reliability.yaml", "mga_controllable_gate", False),
    ],
)
def test_arm_learned_ablation_yaml_dispatches_exact_component_contract(
    monkeypatch, family, task, filename, controller_method, learned_reliability,
):
    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/arm" / family / "ablation" / filename
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
        _experiment_task = task
        _experiment_execution_env = None

    planner = MGAMethodPlugin().create_planner(
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
    ) is planner.component_contract["mga_flags"]["use_rl_prior"]
    assert (
        planner.component_contract["reliability_ckpt"] is not None
    ) is learned_reliability


@pytest.mark.parametrize(
    "plugin,controller_method",
    [
        (MGAMethodPlugin, "mga_controllable_gate"),
        (ModelBasedOnlyMethodPlugin, "mga_controllable"),
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
    if plugin is MGAMethodPlugin:
        config.update(policy_ckpt="policy.pkl", reliability_ckpt="reliability.json")
    planner = plugin().create_planner(Env(), None, config)
    assert planner.controller_method == controller_method
    assert captured["method"] == controller_method
    assert captured["kwargs"]["model_env"] is planner.env


def test_contact_runner_seed_routes_to_environment_and_stochastic_solver(monkeypatch):
    """One formal seed owns both domain randomization and planner sampling."""
    class Env:
        _experiment_task = "humanoid_box_push"
        _experiment_execution_env = None

    captured = {}

    def fake_make_controller(task, method, **kwargs):
        captured.update(task=task, method=method, kwargs=kwargs)
        return kwargs["model_env"], object()

    monkeypatch.setattr(
        "genedynamics.experiments.plugins.methods.contact_receding.make_controller",
        fake_make_controller,
    )
    planner = DIALContactMethodPlugin().create_planner(Env(), None, {
        "task": "humanoid_box_push",
        "n_steps": 1,
        "seed": 0,
        "np_random_seed": 17,
    })
    assert captured["kwargs"]["surface_seed"] == 17
    assert captured["kwargs"]["seed"] == 17
    assert planner.seed == 17


def test_h1_suite_checkpoint_resolution_and_truthful_components(monkeypatch):
    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/humanoid/push_to_line/main/mga.yaml"
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

    planner = MGAMethodPlugin().create_planner(Env(), None, {
        **cfg.method_params, "task": cfg.env_name, "suite": "p4_walk_push",
    })
    assert planner.component_contract["policy_ckpt"].endswith(
        "walk_ppo_seed0.pkl"
    )
    assert planner.component_contract["rl_prior"] is True
    assert planner.component_contract["learned_reliability"] is True

    mbo = ModelBasedOnlyMethodPlugin().create_planner(Env(), None, {
        "task": cfg.env_name, "suite": "p4_walk_push",
        "controller_method": "mga_controllable",
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
def test_mga_result_persists_receding_diagnostics(tmp_path):
    cfg = ExperimentConfig.from_yaml(
        ROOT / "configs/arm/peg_insert/main/mga.yaml"
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
    "p1_force_15n", "p2_push_ood", "p3_unjam", "p4_walk_push",
])
@pytest.mark.parametrize("config_path", [
    "configs/humanoid/push_to_line/main/mga.yaml",
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
    ("configs/humanoid/push_to_line/ablation/no_controllability_geometry.yaml", "p3_unjam"),
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
