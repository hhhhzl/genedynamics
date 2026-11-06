"""Cross-backend contracts for the Torch 2GO geometry layer."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from genedynamics.genemetry import (
    CfsRetraction,
    MultimodalGate,
    ProbePipeline,
    ScheduleOverlay,
    SdfManifold,
    SlidingWindow,
)
from genedynamics.genemetry.schedule.config import OverlayConfig


def test_torch_manifold_matches_jax_geometry_and_projection():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    values = np.asarray(
        [
            [0.2, -0.1, 0.3],
            [0.9, 0.2, -0.4],
            [-0.3, 0.7, 0.1],
            [0.1, -0.2, 0.05],
        ],
        dtype=np.float32,
    )
    vectors = np.asarray([[0.4, -0.2, 0.1], [-0.3, 0.5, 0.7]], dtype=np.float32)
    torch_manifold = SdfManifold(backend="torch")
    jax_manifold = SdfManifold(backend="jax")
    torch_bundle = torch_manifold.geometry(torch.tensor(values), 2, 1.0e-6)
    jax_bundle = jax_manifold.geometry(jnp.asarray(values), 2, 1.0e-6)

    np.testing.assert_allclose(
        torch_bundle.A_rows.numpy(), np.asarray(jax_bundle.A_rows), atol=2.0e-6
    )
    np.testing.assert_allclose(
        torch_bundle.metric_sys.numpy(),
        np.asarray(jax_bundle.metric_sys),
        atol=2.0e-6,
    )
    for mode in ("metric", "tangent"):
        actual = torch_manifold.project(torch.tensor(vectors), torch_bundle, mode=mode)
        expected = jax_manifold.project(jnp.asarray(vectors), jax_bundle, mode=mode)
        np.testing.assert_allclose(actual.numpy(), np.asarray(expected), atol=3.0e-5)


def test_torch_gate_matches_jax_route_proxy():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    generator = np.random.default_rng(1009)
    samples = generator.normal(size=(12, 8, 4)).astype(np.float32)
    samples[:6, -2:, 0] -= 2.0
    samples[6:, -2:, 0] += 2.0
    reference = np.zeros((8, 4), dtype=np.float32)
    mask = np.asarray([0, 0, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    torch_result = MultimodalGate(backend="torch").evaluate(
        torch.tensor(samples),
        torch.tensor(reference),
        torch.tensor(mask),
        torch.tensor(0.2),
        torch.tensor(0.4),
    )
    jax_result = MultimodalGate(backend="jax").evaluate(
        jnp.asarray(samples),
        jnp.asarray(reference),
        jnp.asarray(mask),
        jnp.asarray(0.2),
        jnp.asarray(0.4),
    )
    np.testing.assert_allclose(
        float(torch_result.meta["pi_multi"]),
        float(jax_result.meta["pi_multi"]),
        atol=2.0e-5,
    )
    assert float(torch_result.gamma) == float(jax_result.gamma)


def test_torch_schedule_overlay_matches_numpy_backend():
    config = OverlayConfig(
        rho_ref=200.0,
        kappa0=0.7,
        kappa_rho_gain=0.03,
        delta_margin_scale=0.6,
        delta_rho_gain=0.04,
        sigma_max=0.4,
        theta_min=0.1,
        theta_max=0.8,
    )
    torch_overlay = ScheduleOverlay(config=config, backend="torch")
    numpy_overlay = ScheduleOverlay(config=config, backend="numpy")
    rho = np.asarray([0.0, 5.0, 200.0], dtype=np.float32)
    margin = np.asarray([0.01, 0.04, 0.2], dtype=np.float32)
    hard_t = torch_overlay.hardness(torch.tensor(rho))
    hard_n = numpy_overlay.hardness(rho)
    np.testing.assert_allclose(hard_t.numpy(), hard_n, atol=1.0e-6)
    for actual, expected in zip(
        torch_overlay.constraint_overlay(torch.tensor(margin), torch.tensor(rho)),
        numpy_overlay.constraint_overlay(margin, rho),
    ):
        np.testing.assert_allclose(actual.numpy(), expected, atol=1.0e-6)
    for actual, expected in zip(
        torch_overlay.diffusion_overlay(hard_t, 0.08),
        numpy_overlay.diffusion_overlay(hard_n, 0.08),
    ):
        np.testing.assert_allclose(actual.numpy(), expected, atol=1.0e-6)


def test_torch_probe_pipeline_retracts_and_scatters_selected_candidates():
    trajectories = torch.arange(6 * 3 * 2, dtype=torch.float32).reshape(6, 3, 2)
    costs = torch.tensor([0.1, 4.0, 0.3, 3.0, 0.2, 2.0])
    retraction = CfsRetraction(
        backend="torch",
        filter_fn=lambda state, value, schedule_state, schedule_params: value * 0.5,
    )
    pipeline = ProbePipeline(
        backend="torch", tail_mix=1.0, pool_ratio=0.5, max_probes=2
    )
    result = pipeline.sample_and_retract(
        trajectories,
        costs,
        retraction,
        torch.zeros(1),
        {"M_eff": 2, "sched_state": {}, "sched_params": {}},
        {"generator": torch.Generator().manual_seed(7)},
    )
    selected = result.meta["idx_m"]
    assert selected.numel() == 2
    assert set(selected.tolist()).issubset({1, 3, 5})
    for index in range(len(trajectories)):
        expected = trajectories[index] * (0.5 if index in selected else 1.0)
        torch.testing.assert_close(result.fixed_trajectories[index], expected)
    torch.testing.assert_close(
        result.residual_geometry,
        (trajectories.index_select(0, selected) * -0.5).mean(dim=0),
    )


def test_sliding_window_has_torch_parity():
    window = SlidingWindow(3, 2)
    expected = window.mask_numpy(4, 7)
    np.testing.assert_array_equal(window.mask_torch(4, 7).numpy(), expected)


def test_twogo_solver_retains_all_backend_method_knobs():
    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.core.energy import EnergyTerm, LegacyEnergyFunctional
    from genedynamics.solvers.single.twogo.twogo import TwoGOSolver

    class Dynamics:
        action_dim = 2

        def transition(self, state, action, dt=None):
            return state + action

    RuntimeBackendManager.set_backend("torch", device="cpu")
    backend = RuntimeBackendManager.get_backend()
    solver = TwoGOSolver(
        dynamics=Dynamics(),
        energy=LegacyEnergyFunctional(
            {"zero": EnergyTerm(lambda state, action, info: 0.0)}
        ),
        backend=backend,
        horizon=4,
        Nsample=4,
        Ndiffuse=2,
        twogo_gate_ema_beta=0.31,
        twogo_gate_risk_theta=0.17,
        twogo_task_dir_alpha=0.23,
        twogo_cluster_reweight_alpha=0.41,
        twogo_cluster_retract_mu=0.07,
        twogo_enable_geometry=False,
        twogo_goal_hardstop_margin=0.09,
    )
    assert solver.config["twogo_gate_ema_beta"] == pytest.approx(0.31)
    assert solver.config["twogo_gate_risk_theta"] == pytest.approx(0.17)
    assert solver.config["twogo_task_dir_alpha"] == pytest.approx(0.23)
    assert solver.config["twogo_cluster_reweight_alpha"] == pytest.approx(0.41)
    assert solver.config["twogo_cluster_retract_mu"] == pytest.approx(0.07)
    assert solver.config["twogo_enable_geometry"] is False
    assert solver.config["twogo_goal_hardstop_margin"] == pytest.approx(0.09)


def test_torch_twogo_executes_full_geometry_path_with_exact_call_accounting():
    from genedynamics.core.backends.runtime import RuntimeBackendManager
    from genedynamics.core.constraints.action_filters import (
        TorchTrajectoryFeasibilityFilter,
    )
    from genedynamics.core.energy import EnergyTerm, LegacyEnergyFunctional
    from genedynamics.solvers.single.twogo.twogo import TwoGOSolver

    class EvaluatorDynamics:
        act_dim = 2
        device = torch.device("cpu")

        def __init__(self):
            self.model_call_count = 0

        def transition(self, state, action, dt=None):
            return np.asarray(state) + np.asarray(action)

        def evaluate_action_sequences(self, actions):
            batch, horizon, _ = actions.shape
            calls = batch * horizon
            self.model_call_count += calls
            positions = torch.cumsum(actions, dim=1)
            states = torch.cat((torch.zeros(batch, 1, 2), positions), dim=1)
            violation = actions.abs().amax(dim=(1, 2)) - 0.35
            costs = (positions[:, -1] - 0.15).square().sum(dim=1)
            costs = costs + 20.0 * violation.clamp_min(0).square()
            return SimpleNamespace(
                costs=costs,
                physical_states=states,
                maximum_violation=violation,
                predicted_feasible=violation <= 0,
                model_calls=calls,
                model_call_unit="scalar_dynamics_transition",
            )

        def constraint_geometry(self, actions):
            horizon = actions.shape[0]
            self.model_call_count += horizon
            vectors = torch.zeros_like(actions)
            vectors[:, 0] = 1.0
            vectors[:, 1] = torch.linspace(0.2, 0.8, horizon)
            return SimpleNamespace(
                vectors=vectors,
                model_calls=horizon,
                model_call_unit="scalar_dynamics_transition",
            )

    RuntimeBackendManager.set_backend("torch", device="cpu")
    backend = RuntimeBackendManager.get_backend()
    dynamics = EvaluatorDynamics()
    solver = TwoGOSolver(
        dynamics=dynamics,
        energy=LegacyEnergyFunctional(
            {"zero": EnergyTerm(lambda state, action, info: 0.0)}
        ),
        backend=backend,
        horizon=4,
        Nsample=8,
        Ndiffuse=3,
        seed=1009,
        constraint_filter=TorchTrajectoryFeasibilityFilter((0.5, 0.0)),
        twogo_probe_b=2,
        twogo_active_topk=2,
        twogo_transport_schedule="adaptive",
    )
    result = solver._get_backend_impl().plan(np.zeros(2, dtype=np.float32), 1009)
    assert result["actions"].shape == (4, 2)
    assert result["states"].shape == (5, 2)
    assert result["diffusion_sampled_actions"].shape == (3, 8, 4, 2)
    assert result["probe_b_hist"].tolist() == [2.0, 2.0, 2.0]
    assert result["model_calls"] == dynamics.model_call_count
    assert result["model_calls"] > 3 * 8 * 4
    assert result["model_call_unit"] == "scalar_dynamics_transition"
    assert np.isfinite(result["actions"]).all()
