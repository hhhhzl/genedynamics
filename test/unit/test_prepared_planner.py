from pathlib import Path

import numpy as np
from genedynamics.experiments.framework import (
    ExperimentConfig,
    ExperimentRunner,
    PreparedPlanner,
)
from genedynamics.experiments.plugins.environments.single_integrator_2d import (
    SingleIntegrator2DPlugin,
)
from genedynamics.experiments.plugins.methods.mdoc import MDOCMethodPlugin
from genedynamics.experiments.plugins.obstacles.empty import (
    EmptyObstacleGeneratorPlugin,
)


def _runner(output: Path) -> ExperimentRunner:
    config = ExperimentConfig(
        name="prepared_planner",
        output_dir=output,
        env_name="single_integrator_box_2d",
        method="mdoc",
        env_params={
            "dt": 0.05,
            "horizon": 4,
            "p_max": 2.0,
            "control_limit": 1.0,
        },
        method_params={
            "Nsample": 4,
            "Ndiffuse": 2,
            "num_modes": 1,
            "show_tqdm": False,
        },
        obstacle_levels=[0],
        obstacle_config={"generator": "none"},
        seeds=[7],
        backend="numpy",
        device="cpu",
        metrics=[],
        visualizations=[],
        auto_report=False,
    )
    runner = ExperimentRunner(config)
    runner.register_plugin(SingleIntegrator2DPlugin(), "environment")
    runner.register_plugin(MDOCMethodPlugin(), "method")
    runner.register_plugin(EmptyObstacleGeneratorPlugin(), "obstacle_generator")
    return runner


def test_prepared_planner_exposes_native_plugin_and_plan(tmp_path):
    prepared = _runner(tmp_path).prepare_planner(level=0, seed=7)

    assert isinstance(prepared, PreparedPlanner)
    assert prepared.method_plugin.name == "mdoc"
    assert prepared.method_config["Nsample"] == 4
    assert prepared.method_config["np_random_seed"] == 7
    result = prepared.plan()
    assert len(result["actions"]) == 4
    np.testing.assert_allclose(result["initial_state"], prepared.initial_state)


def test_native_experiment_uses_public_prepared_planner_boundary(
    tmp_path, monkeypatch
):
    runner = _runner(tmp_path)
    original = runner.prepare_planner
    calls = []

    def prepare(level, seed):
        calls.append((level, seed))
        return original(level, seed)

    monkeypatch.setattr(runner, "prepare_planner", prepare)
    result = runner.run_single_experiment(level=0, seed=7)

    assert calls == [(0, 7)]
    assert result["seed"] == 7
    assert result["level"] == 0
    assert result["planning_time"] >= 0.0
