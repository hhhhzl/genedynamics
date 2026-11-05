"""Public 10-step physical and controller-context response ensemble."""

from __future__ import annotations

from dataclasses import asdict

import torch
from torch import nn

from genedynamics.learning.checkpoints.torch import to_cpu
from .issued_response import project_contact_requests
from .response import PublicResponseSpec


PHYSICAL_RESPONSE_DIM = 20
CONTROLLER_CONTEXT_DIM = 53
OUTPUT_DIM = PHYSICAL_RESPONSE_DIM + CONTROLLER_CONTEXT_DIM + 1


def _rotation_exp(vector: torch.Tensor) -> torch.Tensor:
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((
        zero, -z, y, z, zero, -x, -y, x, zero,
    ), dim=-1).reshape(*vector.shape[:-1], 3, 3)
    theta2 = vector.square().sum(-1)
    theta = theta2.sqrt()
    small = theta2 < 1e-8
    a = torch.where(
        small, 1. - theta2 / 6. + theta2.square() / 120.,
        torch.sin(theta) / theta.clamp_min(torch.finfo(theta.dtype).tiny),
    )
    b = torch.where(
        small, .5 - theta2 / 24. + theta2.square() / 720.,
        (1. - torch.cos(theta)) / theta2.clamp_min(torch.finfo(theta.dtype).tiny),
    )
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return eye + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


def project_t3_endpoint_phase(
    physical_prediction: torch.Tensor,
    *,
    current_position: torch.Tensor,
    current_rotation: torch.Tensor,
    current_context: torch.Tensor,
    normalized_contact_count: torch.Tensor,
    mating_clearance: float,
) -> torch.Tensor:
    """Apply the public T3 controller's known endpoint transition geometry."""
    members, batch = physical_prediction.shape[:2]
    if (
        physical_prediction.shape != (members, batch, 20)
        or current_position.shape != (batch, 3)
        or current_rotation.shape != (batch, 3, 3)
        or current_context.shape != (batch, 53)
        or normalized_contact_count.shape != (members, batch)
        or not 0. < mating_clearance < .01
    ):
        raise ValueError("aligned public T3 endpoint projection inputs required")
    from .response import DELTA_SCALE
    delta = physical_prediction[..., :18] * physical_prediction.new_tensor(DELTA_SCALE)
    position = current_position.unsqueeze(0) + delta[..., :3]
    rotation = _rotation_exp(delta[..., 3:6]) @ current_rotation.unsqueeze(0)
    desired = rotation.new_tensor(((1., 0., 0.), (0., -1., 0.), (0., 0., -1.)))
    angle = (rotation - desired).square().sum((-2, -1)).sqrt()
    vertices = rotation.new_tensor([
        [x, y, z]
        for x in (-.010, .010) for y in (-.006, .006) for z in (.045, .115)
    ] + [
        [x, y, z]
        for x in (.009, .015) for y in (-.003, .003) for z in (.045, .115)
    ])
    points = torch.einsum("...ij,kj->...ki", rotation, vertices) + position[..., None, :]
    relative = points[..., :2] - position.new_tensor((.44, -.20))
    clearance = float(mating_clearance) - .0001
    base = (
        (relative[..., :8, 0].abs() < .010 + clearance)
        & (relative[..., :8, 1].abs() < .006 + clearance)
    ).all(-1)
    key = (
        (relative[..., 8:, 0] > -.010 - clearance)
        & (relative[..., 8:, 0] < .015 + clearance)
        & (relative[..., 8:, 1].abs() < .003 + clearance)
    ).all(-1)
    phase = current_context[:, 22:26].argmax(-1).unsqueeze(0).expand(members, batch)
    # The contact state is discrete {0,.5,1}; .75 is its fixed midpoint decoder.
    phase = torch.where(
        (phase == 0) & (normalized_contact_count >= .75), 1, phase,
    )
    phase = torch.where(
        (phase == 1) & (position[..., 2] > .609) & (angle < .045), 2, phase,
    )
    phase = torch.where(
        (phase == 2) & base & key & (angle < .035), 3, phase,
    )
    return torch.nn.functional.one_hot(phase, 4).to(physical_prediction.dtype)


class PublicControllerContextResponseEnsemble(nn.Module):
    """Predict public physical endpoint and causal controller-state endpoint."""

    VERSION = "public-issued-controller-context-response-v2"

    def __init__(self, spec: PublicResponseSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or PublicResponseSpec()
        self.members = nn.ModuleList()
        for _ in range(self.spec.members):
            sizes = (
                self.spec.feature_dim + CONTROLLER_CONTEXT_DIM
                + 13 * self.spec.horizon,
                *self.spec.hidden,
                OUTPUT_DIM,
            )
            layers: list[nn.Module] = []
            for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
                layers.append(nn.Linear(source, target))
                if index < len(sizes) - 2:
                    layers.append(nn.SiLU())
            self.members.append(nn.Sequential(*layers))

    def forward(
        self,
        features: torch.Tensor,
        commands: torch.Tensor,
        controller_context: torch.Tensor,
    ) -> torch.Tensor:
        if (
            features.shape[-1] != self.spec.feature_dim
            or commands.shape[-2:] != (self.spec.horizon, 13)
            or controller_context.shape != (*features.shape[:-1], CONTROLLER_CONTEXT_DIM)
        ):
            raise ValueError("public context-response input schema mismatch")
        offset = (self.spec.history_len - 1) * (self.spec.frame_dim + 1) + 60
        previous = features[..., offset:offset + 13]
        issued = project_contact_requests(commands, previous)
        enabled = (commands[..., 12:13] > (.1 / 25.)).to(commands.dtype)
        chart = torch.cat(
            (issued[..., :12], issued[..., 12:13] + 2 * enabled), dim=-1,
        )
        inputs = torch.cat(
            (features, controller_context, chart.flatten(start_dim=-2)), dim=-1,
        )
        return torch.stack([member(inputs) for member in self.members])

    @staticmethod
    def endpoint(
        prediction: torch.Tensor,
        controller_context: torch.Tensor,
        *,
        current_position: torch.Tensor | None = None,
        current_rotation: torch.Tensor | None = None,
        mating_clearance: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return physical20D, projected future context, normalized contact."""
        if prediction.shape[-1] != OUTPUT_DIM:
            raise ValueError("74D public context response required")
        if controller_context.shape != (*prediction.shape[1:-1], CONTROLLER_CONTEXT_DIM):
            raise ValueError("batch-aligned decision-time controller context required")
        current = controller_context.unsqueeze(0).expand(
            prediction.shape[0], *controller_context.shape,
        )
        endpoint = current + prediction[..., 20:73]
        # Phase is categorical and mode/subcycle are known invariants over the
        # fixed ten-control response horizon.
        phase = endpoint[..., 22:26].clamp(min=0.)
        phase = phase / phase.sum(-1, keepdim=True).clamp_min(1e-6)
        if any(value is not None for value in (
            current_position, current_rotation, mating_clearance,
        )):
            if any(value is None for value in (
                current_position, current_rotation, mating_clearance,
            )):
                raise ValueError("complete public T3 geometry projection required")
            phase = project_t3_endpoint_phase(
                prediction[..., :20], current_position=current_position,
                current_rotation=current_rotation,
                current_context=controller_context,
                normalized_contact_count=prediction[..., 73].clamp(0., 1.),
                mating_clearance=float(mating_clearance),
            )
        endpoint = torch.cat((
            endpoint[..., :22], phase,
            current[..., 26:28],
            endpoint[..., 28:34].clamp(-1., 1.),
            current[..., 34:35],
            endpoint[..., 35:],
        ), dim=-1)
        contact = prediction[..., 73].clamp(0., 1.)
        return prediction[..., :20], endpoint, contact

    def checkpoint(self) -> dict:
        return {
            "version": self.VERSION,
            "spec": asdict(self.spec),
            "state_dict": to_cpu(self.state_dict()),
            "private_state": False,
            "command_chart": (
                "known issued13D from public previous_action with raw "
                "force-enable bit; decision-time public controller context"
            ),
            "output_semantics": (
                "normalized20D public physical response, 53D controller-context "
                "delta, and normalized future public contact count"
            ),
        }

    @classmethod
    def from_checkpoint(
        cls, payload: dict, device: str | torch.device = "cpu",
    ) -> "PublicControllerContextResponseEnsemble":
        if (
            payload.get("version") != cls.VERSION
            or payload.get("private_state") is not False
        ):
            raise ValueError("public controller-context response checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicResponseSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class PublicPhaseHeadControllerContextResponseEnsemble(
    PublicControllerContextResponseEnsemble
):
    """Context response with an independent categorical endpoint-phase head."""

    VERSION = "public-issued-controller-context-phase-head-response-v3"
    PHASE_OUTPUT_DIM = OUTPUT_DIM + 4

    def __init__(self, spec: PublicResponseSpec | None = None) -> None:
        nn.Module.__init__(self)
        self.spec = spec or PublicResponseSpec()
        self.members = nn.ModuleList()
        for _ in range(self.spec.members):
            sizes = (
                self.spec.feature_dim + CONTROLLER_CONTEXT_DIM
                + 13 * self.spec.horizon,
                *self.spec.hidden,
                self.PHASE_OUTPUT_DIM,
            )
            layers: list[nn.Module] = []
            for index, (source, target) in enumerate(zip(sizes[:-1], sizes[1:])):
                layers.append(nn.Linear(source, target))
                if index < len(sizes) - 2:
                    layers.append(nn.SiLU())
            self.members.append(nn.Sequential(*layers))

    @staticmethod
    def endpoint(
        prediction: torch.Tensor,
        controller_context: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prediction.shape[-1] != PublicPhaseHeadControllerContextResponseEnsemble.PHASE_OUTPUT_DIM:
            raise ValueError("78D public phase-head context response required")
        if kwargs:
            raise ValueError("phase-head response does not use geometry projection")
        physical, endpoint, contact = PublicControllerContextResponseEnsemble.endpoint(
            prediction[..., :OUTPUT_DIM], controller_context,
        )
        phase = torch.nn.functional.one_hot(
            prediction[..., OUTPUT_DIM:].argmax(-1), 4,
        ).to(prediction.dtype)
        endpoint = torch.cat((endpoint[..., :22], phase, endpoint[..., 26:]), dim=-1)
        return physical, endpoint, contact

    def checkpoint(self) -> dict:
        payload = super().checkpoint()
        payload.update({
            "version": self.VERSION,
            "output_semantics": (
                "normalized20D public physical response, 53D controller-context "
                "delta, normalized future contact count, and independent 4D "
                "categorical endpoint-phase logits"
            ),
        })
        return payload

    @classmethod
    def from_checkpoint(
        cls, payload: dict, device: str | torch.device = "cpu",
    ) -> "PublicPhaseHeadControllerContextResponseEnsemble":
        if (
            payload.get("version") != cls.VERSION
            or payload.get("private_state") is not False
        ):
            raise ValueError("public controller-context phase-head checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicResponseSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


__all__ = [
    "CONTROLLER_CONTEXT_DIM", "OUTPUT_DIM", "PHYSICAL_RESPONSE_DIM",
    "PublicControllerContextResponseEnsemble",
    "PublicPhaseHeadControllerContextResponseEnsemble",
    "project_t3_endpoint_phase",
]
