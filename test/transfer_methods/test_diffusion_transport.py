"""Few-step transport preserves request ownership and full-chain gradients."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from genedynamics.solvers.single.cross_ebdm.learning.diffusion_transport import PublicTaskDiffusionTransport
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import PublicTransportSpec
from genedynamics.solvers.single.cross_ebdm.learning.response import PublicResponseSpec
from genedynamics.solvers.single.cross_ebdm.learning.task_transport import PublicTaskResidualTransport, T2_TASK_OWNED_COORDINATES


@pytest.mark.parametrize("steps", [2, 4, 8])
def test_fixed_latent_transport_initializes_at_shared_base_and_backpropagates(steps):
    response = PublicResponseSpec()
    spec = PublicTransportSpec(residual_scale=.025, hidden=(64, 64))
    torch.manual_seed(7)
    diffusion = PublicTaskDiffusionTransport(response, spec, denoising_steps=steps, latent_seed=991)
    torch.manual_seed(7)
    mlp = PublicTaskResidualTransport(response, spec)
    assert abs(sum(p.numel() for p in diffusion.parameters()) / sum(p.numel() for p in mlp.parameters()) - 1) < .1
    batch = 3
    target = torch.zeros(batch, response.feature_dim)
    source = torch.zeros_like(target)
    commands = torch.zeros(batch, response.horizon, 13)
    base = torch.zeros_like(commands)
    phase = torch.ones(batch, dtype=torch.long)
    context = torch.zeros(batch, 9)
    initial = diffusion(target, source, commands, base, phase, context)
    assert torch.equal(initial, base)
    initial[..., 0].sum().backward()
    assert torch.isfinite(diffusion.network[-1].weight.grad).all()
    assert torch.count_nonzero(diffusion.network[-1].weight.grad) > 0
    with torch.no_grad():
        diffusion.network[-1].weight.add_(.0005)
    moved = diffusion(target, source, commands, base, phase, context)
    assert torch.count_nonzero(moved[..., 0]) > 0
    for coordinate in T2_TASK_OWNED_COORDINATES:
        torch.testing.assert_close(moved[..., coordinate], base[..., coordinate], rtol=0, atol=0)
    checkpoint = diffusion.checkpoint()
    restored = PublicTaskDiffusionTransport.from_checkpoint(checkpoint)
    torch.testing.assert_close(restored(target, source, commands, base, phase, context), moved, rtol=0, atol=0)
    with pytest.raises(ValueError, match="checkpoint"):
        PublicTaskDiffusionTransport.from_checkpoint(dict(checkpoint, stochastic_noise=True))


def test_task_binding_loads_diffusion_checkpoint_without_changing_source_or_response(tmp_path):
    from genedynamics.solvers.single.cross_ebdm.learning.task_response import PublicTaskIssuedResponseEnsemble
    from genedynamics.tasks.transfer.controllers.bindings import _cross_components
    from genedynamics.tasks.transfer.specs import task_spec

    root = Path(__file__).resolve().parents[2]
    recipe = json.loads((root / "configs/arm/guided_contact_alignment/training/policy_instances_v3.json").read_text())
    response_pointer = recipe["instances"]["instance_01"]["response_checkpoint"]
    response_payload = torch.load(root / response_pointer["path"], map_location="cpu", weights_only=True)
    response = PublicTaskIssuedResponseEnsemble.from_checkpoint(response_payload)
    policy = PublicTaskDiffusionTransport(response.spec, PublicTransportSpec(residual_scale=.025))
    checkpoint = tmp_path / "policy.pt"
    torch.save(policy.checkpoint(), checkpoint)
    pointer = {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    binding = {"policy_checkpoint": pointer,
               "response_checkpoint": response_pointer,
               "source_manifest": recipe["source_manifest"]}
    task = SimpleNamespace(spec=task_spec("T2", robot="xarm7"))
    loaded, observed, bank = _cross_components(task, binding, "cpu")
    assert isinstance(loaded, PublicTaskDiffusionTransport)
    assert observed.spec == loaded.response_spec == bank.spec
