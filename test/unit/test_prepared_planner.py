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

ROOT = Path(__file__).resolve().parents[2]


def _runner(output: Path) -> ExperimentRunner:
    config = ExperimentConfig.from_yaml(
        ROOT / "configs/validation/single_2d/mdoc_prepared_smoke.yaml"
    )
    config.output_dir = output
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
