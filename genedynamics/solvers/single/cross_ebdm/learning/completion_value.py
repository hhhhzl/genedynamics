"""Public T3 completion value and response-endpoint projection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from genedynamics.learning.checkpoints.torch import to_cpu
from .response import DELTA_SCALE


@dataclass(frozen=True)
class PublicCompletionValueSpec:
    hidden: tuple[int, ...] = (128, 128)
    members: int = 5
    maximum_remaining_controls: int = 150
    learning_rate: float = 3e-4

    def __post_init__(self) -> None:
        if (
            not self.hidden
            or any(width < 1 for width in self.hidden)
            or self.members < 2
            or self.maximum_remaining_controls < 1
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
        ):
            raise ValueError("positive public completion-value configuration required")

    @property
    def feature_dim(self) -> int:
        # Goal-relative position/rotation, public twist/wrench, controller
        # feedback, and the requested mode.  Elapsed time is intentionally
        # absent so candidate endpoints at a common horizon cannot gain by a
        # clock shortcut.
        return 3 + 3 + 6 + 6 + 2 + 2


@dataclass(frozen=True)
class PublicPhaseCompletionValueSpec(PublicCompletionValueSpec):
    maximum_remaining_controls: int = 750

    @property
    def feature_dim(self) -> int:
        return super().feature_dim + 4


@dataclass(frozen=True)
class PublicContextCompletionValueSpec(PublicCompletionValueSpec):
    """Full-episode value chart with the frozen public controller context."""

    maximum_remaining_controls: int = 750

    @property
    def feature_dim(self) -> int:
        # The shared PublicControllerMemory context is 53D.  contact_count is
        # included separately because it is causal actor-owned state used by
        # the public phase transition but intentionally absent from context_v1.
        return super().feature_dim + 53 + 1


def _orientation_error(current: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    if current.shape != desired.shape or current.shape[-2:] != (3, 3):
        raise ValueError("matching public rotation matrices required")
    return .5 * torch.cross(
        current.transpose(-1, -2), desired.transpose(-1, -2), dim=-1,
    ).sum(dim=-2)


def completion_value_features(
    *,
    position: torch.Tensor,
    rotation: torch.Tensor,
    twist: torch.Tensor,
    wrench: torch.Tensor,
    goal_position: torch.Tensor,
    goal_rotation: torch.Tensor,
    controller_feedback: torch.Tensor,
    mode: torch.Tensor,
) -> torch.Tensor:
    """Map strict public observations to a fixed SI-scaled completion chart."""
    lead = position.shape[:-1]
    if (
        position.shape != (*lead, 3)
        or rotation.shape != (*lead, 3, 3)
        or twist.shape != (*lead, 6)
        or wrench.shape != (*lead, 6)
        or goal_position.shape != (*lead, 3)
        or goal_rotation.shape != (*lead, 3, 3)
        or controller_feedback.shape != (*lead, 2)
        or mode.shape != lead
    ):
        raise ValueError("one aligned public completion-value state required")
    if mode.dtype == torch.bool or torch.is_floating_point(mode):
        raise ValueError("integer requested mode required")
    if bool(((mode < 0) | (mode > 1)).any()):
        raise ValueError("requested mode must be zero or one")
    position_error = (goal_position - position) / .02
    rotation_error = _orientation_error(rotation, goal_rotation) / .2
    twist_scale = twist.new_tensor((.08, .08, .08, .4, .4, .4))
    wrench_scale = wrench.new_tensor((25., 25., 25., 4., 4., 4.))
    feedback_scale = controller_feedback.new_tensor((.05, 1.))
    mode_one_hot = torch.nn.functional.one_hot(mode.long(), 2).to(position.dtype)
    value = torch.cat((
        position_error, rotation_error, twist / twist_scale,
        wrench / wrench_scale, controller_feedback / feedback_scale,
        mode_one_hot,
    ), dim=-1)
    if not bool(torch.isfinite(value).all()):
        raise ValueError("finite public completion-value features required")
    return value


def phase_completion_value_features(
    *,
    position: torch.Tensor,
    rotation: torch.Tensor,
    twist: torch.Tensor,
    wrench: torch.Tensor,
    goal_position: torch.Tensor,
    goal_rotation: torch.Tensor,
    controller_feedback: torch.Tensor,
    mode: torch.Tensor,
    phase: torch.Tensor,
) -> torch.Tensor:
    """Add the actor-owned causal phase to the public completion chart."""
    value = completion_value_features(
        position=position, rotation=rotation, twist=twist, wrench=wrench,
        goal_position=goal_position, goal_rotation=goal_rotation,
        controller_feedback=controller_feedback, mode=mode,
    )
    lead = position.shape[:-1]
    if phase.shape != lead or phase.dtype == torch.bool or torch.is_floating_point(phase):
        raise ValueError("one aligned integer public controller phase required")
    if bool(((phase < 0) | (phase > 3)).any()):
        raise ValueError("public controller phase must be zero through three")
    one_hot = torch.nn.functional.one_hot(phase.long(), 4).to(value.dtype)
    return torch.cat((value, one_hot), dim=-1)


def context_completion_value_features(
    *,
    position: torch.Tensor,
    rotation: torch.Tensor,
    twist: torch.Tensor,
    wrench: torch.Tensor,
    goal_position: torch.Tensor,
    goal_rotation: torch.Tensor,
    controller_feedback: torch.Tensor,
    mode: torch.Tensor,
    controller_context: torch.Tensor,
    contact_count: torch.Tensor,
) -> torch.Tensor:
    """Join candidate-sensitive public state with causal controller state."""
    value = completion_value_features(
        position=position, rotation=rotation, twist=twist, wrench=wrench,
        goal_position=goal_position, goal_rotation=goal_rotation,
        controller_feedback=controller_feedback, mode=mode,
    )
    lead = position.shape[:-1]
    if controller_context.shape != (*lead, 53):
        raise ValueError("one aligned 53D public controller context required")
    if contact_count.shape != lead:
        raise ValueError("one aligned public contact count required")
    if not bool(torch.isfinite(controller_context).all()):
        raise ValueError("finite public controller context required")
    if contact_count.dtype == torch.bool:
        raise ValueError("numeric public contact count required")
    if torch.is_floating_point(contact_count):
        if (
            not bool(torch.isfinite(contact_count).all())
            or bool(((contact_count < 0.) | (contact_count > 1.)).any())
        ):
            raise ValueError("normalized public contact count must be in zero to one")
        normalized_count = contact_count.to(value.dtype)
    else:
        if bool((contact_count < 0).any()):
            raise ValueError("nonnegative public contact count required")
        # Two consecutive contact observations trigger the public recovery
        # phase; larger values are equivalent under that transition rule.
        normalized_count = contact_count.clamp(max=2).to(value.dtype) / 2.
    return torch.cat((value, controller_context.to(value.dtype),
                      normalized_count[..., None]), dim=-1)


def _rotation_exp(vector: torch.Tensor) -> torch.Tensor:
    if vector.shape[-1] != 3:
        raise ValueError("rotation vector required")
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((
        zero, -z, y,
        z, zero, -x,
        -y, x, zero,
    ), dim=-1).reshape(*vector.shape[:-1], 3, 3)
    theta2 = vector.square().sum(-1)
    theta = theta2.sqrt()
    small = theta2 < 1e-8
    a = torch.where(
        small,
        1. - theta2 / 6. + theta2.square() / 120.,
        torch.sin(theta) / theta.clamp_min(torch.finfo(theta.dtype).tiny),
    )
    b = torch.where(
        small,
        .5 - theta2 / 24. + theta2.square() / 720.,
        (1. - torch.cos(theta)) / theta2.clamp_min(torch.finfo(theta.dtype).tiny),
    )
    eye = torch.eye(3, device=vector.device, dtype=vector.dtype)
    return eye + a[..., None, None] * skew + b[..., None, None] * (skew @ skew)


class PublicCompletionValueEnsemble(nn.Module):
    """Ensemble estimate of normalized controls remaining to the recorded end."""

    VERSION = "public-T3-completion-value-v1"

    def __init__(self, spec: PublicCompletionValueSpec | None = None) -> None:
        super().__init__()
        self.spec = spec or PublicCompletionValueSpec()
        self.members = nn.ModuleList()
        for _ in range(self.spec.members):
            widths = (self.spec.feature_dim, *self.spec.hidden, 1)
            layers: list[nn.Module] = []
            for index, (source, target) in enumerate(zip(widths[:-1], widths[1:])):
                layers.append(nn.Linear(source, target))
                if index < len(widths) - 2:
                    layers.append(nn.SiLU())
            self.members.append(nn.Sequential(*layers))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.spec.feature_dim:
            raise ValueError("public completion-value feature schema mismatch")
        # Softplus keeps the physical value nonnegative while retaining a
        # smooth local gradient for candidate endpoint ranking.
        return torch.stack([
            torch.nn.functional.softplus(member(features).squeeze(-1))
            for member in self.members
        ])

    def checkpoint(self, *, calibration: dict | None = None) -> dict:
        return {
            "version": self.VERSION,
            "spec": asdict(self.spec),
            "state_dict": to_cpu(self.state_dict()),
            "calibration": calibration,
            "private_state": False,
            "target_semantics": (
                "remaining controls to the unfiltered recorded public episode end, "
                "normalized by maximum_remaining_controls"
            ),
        }

    @classmethod
    def from_checkpoint(cls, payload: dict, device: str | torch.device = "cpu"):
        if (
            payload.get("version") != cls.VERSION
            or payload.get("private_state") is not False
        ):
            raise ValueError("public T3 completion-value checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicCompletionValueSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class PublicPhaseCompletionValueEnsemble(PublicCompletionValueEnsemble):
    """Full-episode completion value conditioned on causal controller phase."""

    VERSION = "public-T3-phase-completion-value-v2"

    def __init__(self, spec: PublicPhaseCompletionValueSpec | None = None) -> None:
        super().__init__(spec or PublicPhaseCompletionValueSpec())

    @classmethod
    def from_checkpoint(cls, payload: dict, device: str | torch.device = "cpu"):
        if (
            payload.get("version") != cls.VERSION
            or payload.get("private_state") is not False
        ):
            raise ValueError("public T3 phase completion-value checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicPhaseCompletionValueSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class PublicContextCompletionValueEnsemble(PublicCompletionValueEnsemble):
    """Completion value conditioned on all causal public controller state."""

    VERSION = "public-T3-context-completion-value-v3"

    def __init__(self, spec: PublicContextCompletionValueSpec | None = None) -> None:
        super().__init__(spec or PublicContextCompletionValueSpec())

    @classmethod
    def from_checkpoint(cls, payload: dict, device: str | torch.device = "cpu"):
        if (
            payload.get("version") != cls.VERSION
            or payload.get("private_state") is not False
        ):
            raise ValueError("public T3 context completion-value checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicContextCompletionValueSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class PublicActionConditionalCompletionValueEnsemble(
    PublicContextCompletionValueEnsemble
):
    """Endpoint value chart conditioned on the decision-time context.

    A candidate response predicts public endpoint state, while controller
    context at that hypothetical endpoint is not observed.  Training therefore
    holds the decision-time context fixed across each temporal pair, matching
    runtime candidate scoring exactly.
    """

    VERSION = "public-T3-action-conditional-completion-value-v4"

    def checkpoint(self, *, calibration: dict | None = None) -> dict:
        payload = super().checkpoint(calibration=calibration)
        payload["target_semantics"] = (
            "remaining controls to the unfiltered recorded public episode end; "
            "temporal endpoint differences hold decision-time public controller "
            "context fixed"
        )
        payload["endpoint_context"] = "decision_time_fixed"
        return payload

    @classmethod
    def from_checkpoint(cls, payload: dict, device: str | torch.device = "cpu"):
        if (
            payload.get("version") != cls.VERSION
            or payload.get("private_state") is not False
            or payload.get("endpoint_context") != "decision_time_fixed"
        ):
            raise ValueError("public T3 action-conditional value checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicContextCompletionValueSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class PublicEndpointAnchoredCompletionValueEnsemble(
    PublicActionConditionalCompletionValueEnsemble
):
    """Action-conditional endpoint chart anchored by future-time labels."""

    VERSION = "public-T3-endpoint-anchored-completion-value-v5"

    def checkpoint(self, *, calibration: dict | None = None) -> dict:
        payload = super().checkpoint(calibration=calibration)
        payload["endpoint_anchor"] = "recorded_future_remaining_controls"
        return payload

    @classmethod
    def from_checkpoint(cls, payload: dict, device: str | torch.device = "cpu"):
        if (
            payload.get("version") != cls.VERSION
            or payload.get("private_state") is not False
            or payload.get("endpoint_context") != "decision_time_fixed"
            or payload.get("endpoint_anchor")
            != "recorded_future_remaining_controls"
        ):
            raise ValueError("public T3 endpoint-anchored value checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicContextCompletionValueSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


class PublicQualifiedContextCompletionValueEnsemble(
    PublicContextCompletionValueEnsemble
):
    """Context value selected directly against every frozen cell gate."""

    VERSION = "public-T3-qualified-context-completion-value-v6"

    @classmethod
    def from_checkpoint(cls, payload: dict, device: str | torch.device = "cpu"):
        if (
            payload.get("version") != cls.VERSION
            or payload.get("private_state") is not False
        ):
            raise ValueError("qualified public T3 context value checkpoint required")
        values = dict(payload["spec"])
        values["hidden"] = tuple(values["hidden"])
        model = cls(PublicContextCompletionValueSpec(**values)).to(device)
        model.load_state_dict(payload["state_dict"], strict=True)
        return model


def response_endpoint_value(
    model: PublicCompletionValueEnsemble,
    prediction: torch.Tensor,
    *,
    current: dict[str, torch.Tensor],
    mode: int,
    phase: int | None = None,
    controller_context: torch.Tensor | None = None,
    contact_count: torch.Tensor | None = None,
) -> torch.Tensor:
    """Evaluate every response-member endpoint with every value member.

    ``prediction`` is response-member x candidate x normalized20D.  The return
    shape is value-member x response-member x candidate, so runtime admission
    can require a gain for every model pairing.
    """
    if prediction.ndim != 3 or prediction.shape[-1] != 20:
        raise ValueError("response-member x candidate x normalized20D required")
    response_members, candidates = prediction.shape[:2]
    scale = prediction.new_tensor(DELTA_SCALE)
    delta = prediction[..., :18] * scale

    def expanded(name: str, width: tuple[int, ...]) -> torch.Tensor:
        value = current[name]
        if value.shape not in {(1, *width), (candidates, *width)}:
            raise ValueError(
                f"one or candidate-aligned current public {name} value required"
            )
        return value.expand(response_members, candidates, *width)

    position = expanded("position", (3,)) + delta[..., :3]
    rotation = _rotation_exp(delta[..., 3:6]) @ expanded("rotation", (3, 3))
    twist = expanded("twist", (6,)) + delta[..., 6:12]
    wrench = expanded("wrench", (6,)) + delta[..., 12:18]
    goal_position = expanded("goal_position", (3,))
    goal_rotation = expanded("goal_rotation", (3, 3))
    controller_feedback = expanded("controller_feedback", (2,))
    modes = torch.full(
        (response_members, candidates), int(mode),
        device=prediction.device, dtype=torch.long,
    )
    arguments = dict(
        position=position, rotation=rotation, twist=twist, wrench=wrench,
        goal_position=goal_position, goal_rotation=goal_rotation,
        controller_feedback=controller_feedback, mode=modes,
    )
    if isinstance(model, PublicContextCompletionValueEnsemble):
        if phase is not None:
            raise ValueError("context completion endpoint receives phase in its context")
        if controller_context is None or contact_count is None:
            raise ValueError("context completion endpoint requires causal controller state")
        if controller_context.shape not in {
            (1, 53), (candidates, 53),
            (response_members, candidates, 53),
        }:
            raise ValueError("one, candidate-, or response-aligned controller context required")
        if contact_count.shape not in {
            (1,), (candidates,), (response_members, candidates),
        }:
            raise ValueError("one, candidate-, or response-aligned contact count required")
        features = context_completion_value_features(
            **arguments,
            controller_context=controller_context.expand(
                response_members, candidates, 53,
            ),
            contact_count=contact_count.expand(response_members, candidates),
        )
    elif isinstance(model, PublicPhaseCompletionValueEnsemble):
        if controller_context is not None or contact_count is not None:
            raise ValueError("controller context is unsupported by phase value")
        if phase is None or isinstance(phase, bool) or not 0 <= phase <= 3:
            raise ValueError("phase completion endpoint requires phase zero through three")
        features = phase_completion_value_features(
            **arguments,
            phase=torch.full_like(modes, int(phase)),
        )
    else:
        if phase is not None or controller_context is not None or contact_count is not None:
            raise ValueError("controller state is unsupported by the v1 completion value")
        features = completion_value_features(**arguments)
    return model(features)


__all__ = [
    "PublicActionConditionalCompletionValueEnsemble",
    "PublicCompletionValueEnsemble", "PublicCompletionValueSpec",
    "PublicContextCompletionValueEnsemble", "PublicContextCompletionValueSpec",
    "PublicEndpointAnchoredCompletionValueEnsemble",
    "PublicQualifiedContextCompletionValueEnsemble",
    "PublicPhaseCompletionValueEnsemble", "PublicPhaseCompletionValueSpec",
    "completion_value_features", "context_completion_value_features",
    "phase_completion_value_features",
    "response_endpoint_value",
]
