"""Numerically compare the pinned DIAL author reverse step with the local port."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import torch

from genedynamics.solvers.single.dial.backends.dial_jax import (
    make_sigma_control, reverse_once,
)
from genedynamics.solvers.single.dial.spline import NodeSpline
from genedynamics.tasks.transfer.controllers.dial import TorchDIALPlanner


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "genedynamics/solvers/single/dial/source_manifest.json"


def test_pinned_dial_author_checkout_matches_source_lock():
    manifest = json.loads(MANIFEST.read_text())
    lock = json.loads((ROOT / "requirements/baselines/sources.lock.json").read_text())
    source = next(row for row in lock["sources"] if row["method"] == "dial_transfer")
    checkout = ROOT / "baselines" / source["directory"]
    commit = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True,
    ).strip()
    assert manifest["repository"] == source["repository"]
    assert manifest["commit"] == source["commit"] == commit
    assert manifest["author_core"]["path"] == str(
        Path("baselines") / source["directory"] / source["required_source_files"][0]
    )
    assert hashlib.sha256((checkout / "LICENSE").read_bytes()).hexdigest() == (
        source["licenses"]["LICENSE"]
    )


def _author_reverse_once():
    manifest = json.loads(MANIFEST.read_text())
    source_path = ROOT / manifest["author_core"]["path"]
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == manifest["author_core"]["sha256"]
    source = ast.parse(source_path.read_text())
    cls = next(node for node in source.body
               if isinstance(node, ast.ClassDef) and node.name == "MBDPI")
    reverse = next(node for node in cls.body
                   if isinstance(node, ast.FunctionDef) and node.name == "reverse_once")
    update = next(node for node in source.body
                  if isinstance(node, ast.FunctionDef) and node.name == "softmax_update")
    reverse.decorator_list = []
    update.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[update, reverse], type_ignores=[]))
    namespace = {"jax": jax, "jnp": jnp}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["reverse_once"], namespace["softmax_update"]


def test_pinned_author_reverse_step_matches_local_jax_port():
    author, author_update = _author_reverse_once()
    spline = NodeSpline.build(4, 16)
    count, horizon, nu = 32, 17, 3

    def reward(state, controls, t0):
        del state, t0
        return -jnp.sum((controls - .27) ** 2, axis=-1)

    def author_rollout(state, controls):
        rews = reward(state, controls, 0.)
        zeros = jnp.zeros((count + 1, horizon, nu), dtype=jnp.float32)
        trajectory = SimpleNamespace(
            q=zeros, qd=zeros, x=SimpleNamespace(pos=zeros[:, :, None, :]),
        )
        return rews, trajectory

    args = SimpleNamespace(Nsample=count, Hnode=4, temp_sample=.06)
    upstream = SimpleNamespace(
        args=args, nu=nu, update_fn=author_update,
        node2u_vvmap=lambda nodes: jnp.einsum("hn,bnu->bhu", spline.N2U, nodes),
        rollout_us_vmap=author_rollout,
    )
    nodes = jnp.asarray(np.random.default_rng(12).normal(0, .15, (5, nu)), dtype=jnp.float32)
    nodes = nodes.at[0, 0].set(1.3).at[2, 1].set(-1.4)
    rng = jax.random.PRNGKey(123)
    sigma = make_sigma_control(4, .9, 1.) * .5
    _, expected, author_info = author(upstream, None, rng, nodes, sigma)
    actual, local_info = reverse_once(
        None, rng, nodes, sigma, rollout_fn=reward, N2U=spline.N2U,
        Nsample=count, temp_sample=.06, action_limit=1.,
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(local_info["rews"], author_info["rews"], rtol=2e-6, atol=2e-6)
    np.testing.assert_allclose(actual[0], jnp.clip(nodes[0], -1, 1), rtol=0, atol=2e-6)


def test_pinned_author_reverse_step_matches_deployed_torch_planner(monkeypatch):
    author, author_update = _author_reverse_once()
    planner = TorchDIALPlanner(seed=19)
    # Isolate one reverse step; the deployed actor's 10/2-step schedule has
    # separate accounting tests and repeats this same update.
    planner.n_diffuse_init = 1
    count, horizon, nu = planner.n_samples, 17, 13
    nodes = jnp.asarray(np.random.default_rng(23).normal(0, .08, (5, nu)), dtype=jnp.float32)
    nodes = nodes.at[0, 0].set(1.3).at[2, 4].set(-1.4)
    planner.nodes = torch.tensor(np.asarray(nodes).tolist(), dtype=torch.float32)
    sigma = jnp.asarray(planner.sigma_control.tolist())
    rng = jax.random.PRNGKey(29)
    _, noise_rng = jax.random.split(rng)
    eps = np.asarray(jax.random.normal(noise_rng, (count, 5, nu)))
    monkeypatch.setattr(torch, "randn", lambda *args, **kwargs: torch.tensor(eps.tolist()))

    def reward(state, controls, t0):
        del state, t0
        return -jnp.sum((controls - .27) ** 2, axis=-1)

    def author_rollout(state, controls):
        rews = reward(state, controls, 0.)
        zeros = jnp.zeros((count + 1, horizon, nu), dtype=jnp.float32)
        return rews, SimpleNamespace(
            q=zeros, qd=zeros, x=SimpleNamespace(pos=zeros[:, :, None, :]),
        )

    upstream = SimpleNamespace(
        args=SimpleNamespace(Nsample=count, Hnode=4, temp_sample=planner.temp_sample),
        nu=nu, update_fn=author_update,
        node2u_vvmap=lambda values: jnp.einsum(
            "hn,bnu->bhu", jnp.asarray(planner.n2u.tolist()), values,
        ),
        rollout_us_vmap=author_rollout,
    )
    # GPU JAX may otherwise use reduced-precision matmul while Torch uses fp32.
    with jax.default_matmul_precision("highest"):
        _, expected_nodes, _ = author(upstream, None, rng, nodes, sigma)
        cost = lambda values: (values - .27).square().sum(-1).mean(-1)
        action = planner.replan(cost)
        expected_dense = np.asarray(jnp.asarray(planner.n2u.tolist()) @ expected_nodes)
        expected_shifted = np.asarray(
            jnp.asarray(planner.u2n.tolist()) @
            jnp.roll(jnp.asarray(expected_dense), -1, axis=0).at[-1].set(0.)
        )
    np.testing.assert_allclose(action.tolist(), expected_dense[0], rtol=2e-5, atol=2e-5)
    np.testing.assert_allclose(planner.nodes.tolist(), expected_shifted, rtol=2e-5, atol=2e-5)
    assert planner.candidate_rollouts == count + 1
