from genedynamics.experiments.plugins.environments._contact_task import (
    ContactTaskEnvironmentPlugin,
)
from genedynamics.experiments.plugins.environments.single_integrator_2d import (
    SingleIntegrator2DPlugin,
)


def test_only_context_aware_plugins_receive_runner_private_fields():
    assert not SingleIntegrator2DPlugin().accepts_experiment_context
    assert ContactTaskEnvironmentPlugin(
        "manipulator_peg_insert"
    ).accepts_experiment_context
