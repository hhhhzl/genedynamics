"""Task-first, support-gated response anchoring for a new method version.

The existing Full objective is intentionally left unchanged.  This module
defines a separately versioned training rule with three properties:

* the task objective remains active inside the model task band;
* response matching is applied only to samples that are task compatible and
  lie inside declared public-condition/disagreement thresholds; and
* final checkpoint selection is relative to the matched task-only policy.

The gates use public task-space histories and response-ensemble predictions.
They are model-side filters, not physical safety, reachability, or event-margin
certificates.  Thresholds must be frozen from source/known-nominal development
data before an unknown target cohort is accessed.
"""
from dataclasses import dataclass
import math

import torch

from .recovery_transport import recovery_mask,recovery_policy_commands
from .residual_transport import (
    BOUNDS,PublicTransportTrainer,decode_pose_transport,decoder_context,task_proxy,
)


@dataclass(frozen=True)
class PublicAnchoringGateSpec:
    task_tolerance: float
    maximum_condition_distance: float
    maximum_response_disagreement: float
    band_penalty: float = 100.0

    def __post_init__(self):
        values=(self.task_tolerance,self.maximum_condition_distance,
                self.maximum_response_disagreement,self.band_penalty)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("finite nonnegative anchoring-gate values required")
        if self.band_penalty == 0:
            raise ValueError("positive task-band penalty required")


@dataclass(frozen=True)
class PublicRelativeResponseSpec:
    """Normalize anchoring by the matched task-only response discrepancy.

    The denominator is detached: it is a fixed, same-state control reference,
    not another optimization target.  The floor only prevents a nearly exact
    task-only match from assigning unbounded weight to one sample.
    """

    reference_floor: float = 1e-4

    def __post_init__(self):
        if not math.isfinite(self.reference_floor) or self.reference_floor <= 0:
            raise ValueError("finite positive response-reference floor required")


@dataclass(frozen=True)
class PublicDirectionalResponseSpec:
    """Relative response anchoring admitted only for task-helpful source laws."""

    reference_floor: float = 1e-4
    minimum_source_task_advantage: float = 0.0
    source_task_state: str = "source"

    def __post_init__(self):
        values = (self.reference_floor, self.minimum_source_task_advantage)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("finite directional-response values required")
        if self.reference_floor <= 0 or self.minimum_source_task_advantage < 0:
            raise ValueError("positive floor and nonnegative source advantage required")
        if self.source_task_state not in {"source", "target"}:
            raise ValueError("source task comparison state must be source or target")


@dataclass(frozen=True)
class PublicPairedProgressSpec:
    """Require transport to improve over the frozen paired task-only action."""

    linear_margin: float = 1e-5
    angular_margin: float = 1e-4
    weight: float = 1.0
    tail_fraction: float = 0.0
    tail_weight: float = 0.0
    requested_mode: int | None = None

    def __post_init__(self):
        values = (self.linear_margin, self.angular_margin, self.weight)
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("finite positive paired-progress values required")
        if (
            not math.isfinite(self.tail_fraction)
            or not 0. <= self.tail_fraction <= 1.
            or not math.isfinite(self.tail_weight)
            or self.tail_weight < 0.
            or (self.tail_fraction == 0.) != (self.tail_weight == 0.)
        ):
            raise ValueError("paired-progress tail fraction and weight must be jointly positive")
        if (
            self.requested_mode is not None
            and (
                isinstance(self.requested_mode, bool)
                or not isinstance(self.requested_mode, int)
                or self.requested_mode < 0
            )
        ):
            raise ValueError("paired-progress requested mode must be a nonnegative integer")


@dataclass(frozen=True)
class PublicEnsembleRobustTaskSpec:
    """Optimize the worst paired member gap used by public runtime admission."""

    reduction: str = "worst_paired_member"

    def __post_init__(self):
        if self.reduction != "worst_paired_member":
            raise ValueError("ensemble task reduction must be worst_paired_member")


def ensemble_robust_paired_task(member_task, task_only_member_task, spec):
    """Worst same-member task gap; minimizing it improves every member."""
    if not isinstance(spec,PublicEnsembleRobustTaskSpec):
        raise ValueError("explicit ensemble-robust task spec required")
    if member_task.ndim != 2 or task_only_member_task.shape != member_task.shape:
        raise ValueError("matching member x batch task proxies required")
    gap=member_task-task_only_member_task.detach()
    return gap,gap.amax(0)


@dataclass(frozen=True)
class PublicCompletionProgressSpec:
    """Public T4 completion objective aligned with the deployed first action.

    Directional progress is compared with the common controller in physical
    task-space units.  The loss is active only after the public recovery gate;
    near the terminal set it additionally contracts the learned residual back
    to the common action that owns the hold behavior.
    """

    linear_progress_scale: float = 1e-4
    angular_progress_scale: float = 1e-3
    progress_weight: float = 1.0
    terminal_weight: float = 1.0
    position_tolerance_m: float = .002
    orientation_tolerance_rad: float = math.radians(2.)
    terminal_multiplier: float = 2.0
    progress_tolerance: float = 1e-7
    minimum_candidate_delta: float = 1e-4
    tail_fraction: float = 0.0
    tail_weight: float = 0.0

    def __post_init__(self):
        positive = (
            self.linear_progress_scale, self.angular_progress_scale,
            self.progress_weight, self.terminal_weight,
            self.position_tolerance_m, self.orientation_tolerance_rad,
            self.terminal_multiplier,
        )
        nonnegative = (
            self.progress_tolerance, self.minimum_candidate_delta,
            self.tail_fraction, self.tail_weight,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("finite positive completion-progress values required")
        if any(not math.isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("finite nonnegative completion tolerances required")
        if self.tail_fraction > 1:
            raise ValueError("completion tail fraction must be at most one")


@dataclass(frozen=True)
class PublicFiniteTimeProgressSpec:
    """Public T3 phase-3 progress with an explicit terminal-time reserve."""

    control_dt_seconds: float = .02
    hold_controls: int = 15
    phase: int = 3
    linear_progress_scale: float = 1e-5
    angular_progress_scale: float = 1e-4
    progress_weight: float = 1.0
    terminal_weight: float = 1.0
    position_tolerance_m: float = .002
    orientation_tolerance_rad: float = math.radians(2.)
    progress_tolerance: float = 1e-7
    minimum_candidate_delta: float = 1e-4
    tail_fraction: float = .1
    tail_weight: float = 4.0
    candidate_residual_scale: float = .025

    def __post_init__(self):
        positive = (
            self.control_dt_seconds, self.linear_progress_scale,
            self.angular_progress_scale, self.progress_weight,
            self.terminal_weight, self.position_tolerance_m,
            self.orientation_tolerance_rad,
            self.candidate_residual_scale,
        )
        nonnegative = (
            self.progress_tolerance, self.minimum_candidate_delta,
            self.tail_fraction, self.tail_weight,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive):
            raise ValueError("finite positive finite-time progress values required")
        if any(not math.isfinite(value) or value < 0 for value in nonnegative):
            raise ValueError("finite nonnegative finite-time tolerances required")
        if (
            isinstance(self.hold_controls, bool)
            or not isinstance(self.hold_controls, int)
            or self.hold_controls < 1
        ):
            raise ValueError("positive integer hold reserve required")
        if self.phase != 3:
            raise ValueError("T3 finite-time progress is a phase-3 objective")
        if self.tail_fraction > 1:
            raise ValueError("finite-time tail fraction must be at most one")


def _batched_orientation_error(current, desired):
    if current.shape != desired.shape or current.ndim != 3 or current.shape[-2:] != (3, 3):
        raise ValueError("matching batch x 3 x 3 rotations required")
    return .5 * torch.cross(
        current.transpose(-1, -2), desired.transpose(-1, -2), dim=-1,
    ).sum(dim=-2)


def recenter_saturated_transport_output(
    policy, batch, feedback, *, maximum_median_absolute_logit,
):
    """Restore trainable output logits using only frozen public calibration states.

    A row-wise positive rescaling of the final affine layer preserves each
    output's sign and ordering while moving saturated ``tanh`` logits back into
    a region with usable derivatives.  The result remains an ordinary transport
    checkpoint; deployment has no special forward or gradient path.
    """
    cap = float(maximum_median_absolute_logit)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("finite positive median-logit cap required")
    if len(batch["target_features"]) < 1:
        raise ValueError("nonempty public calibration states required")
    output = policy.network[-1]
    if not isinstance(output, torch.nn.Linear):
        raise ValueError("transport network must end in one affine output layer")
    inputs = torch.cat((
        batch["target_features"], batch["source_features"],
        batch["source_commands"].flatten(start_dim=1),
        batch["base_proto"].flatten(start_dim=1),
        torch.nn.functional.one_hot(
            batch["decoder"]["phase"].long(), 4,
        ).to(batch["target_features"].dtype),
        decoder_context(batch["decoder"], feedback),
    ), dim=1)
    with torch.no_grad():
        logits_before = policy.network(inputs)
        normalized_before, executed_before = recovery_policy_commands(
            policy, feedback, batch,
        )
        median_before = logits_before.abs().median(dim=0).values
        scale = torch.minimum(
            torch.ones_like(median_before),
            median_before.new_full(median_before.shape, cap)
            / median_before.clamp_min(torch.finfo(median_before.dtype).tiny),
        )
        output.weight.mul_(scale[:, None])
        if output.bias is not None:
            output.bias.mul_(scale)
        logits_after = policy.network(inputs)
        normalized_after, executed_after = recovery_policy_commands(
            policy, feedback, batch,
        )
        median_after = logits_after.abs().median(dim=0).values
        normalized_change = (normalized_after - normalized_before).abs()
        executed_change = (executed_after - executed_before).abs()
    if float(median_after.max()) > cap + 1e-5:
        raise RuntimeError("output recentering did not enforce the logit cap")
    return {
        "version": "public_output_row_recenter_v1",
        "calibration_states": int(len(batch["target_features"])),
        "maximum_median_absolute_logit": cap,
        "scaled_outputs": int((scale < 1).sum()),
        "minimum_output_scale": float(scale.min()),
        "median_absolute_logit_before_max": float(median_before.max()),
        "median_absolute_logit_after_max": float(median_after.max()),
        "normalized_command_change_p95": float(torch.quantile(
            normalized_change.flatten(), .95,
        )),
        "normalized_command_change_max": float(normalized_change.max()),
        "executed_command_change_p95": float(torch.quantile(
            executed_change.flatten(), .95,
        )),
        "executed_command_change_max": float(executed_change.max()),
    }


def public_completion_progress(candidate, batch, spec):
    """Per-state T4 progress deficits and candidate utility from public data."""
    if not isinstance(spec, PublicCompletionProgressSpec):
        raise ValueError("explicit public completion-progress spec required")
    if candidate.ndim != 3 or candidate.shape[-1] != 13:
        raise ValueError("candidate commands must be batch x horizon x 13")
    common = batch["common_action"]
    if common.shape != (candidate.shape[0], 13):
        raise ValueError("one normalized common action per candidate state required")
    current = batch["current"]
    reference = batch["completion_reference"]
    if reference.shape != (candidate.shape[0], 12):
        raise ValueError("one public goal position and rotation per candidate state required")

    scale = candidate.new_tensor(BOUNDS[:6])
    candidate_action = candidate[:, 0, :6] * scale
    common_action = common[:, :6] * scale
    position_error = reference[:, :3] - current["position"]
    goal_rotation = reference[:, 3:12].reshape(-1, 3, 3)
    rotation_error = _batched_orientation_error(current["rotation"], goal_rotation)
    candidate_linear = (candidate_action[:, :3] * position_error).sum(-1)
    common_linear = (common_action[:, :3] * position_error).sum(-1)
    candidate_angular = (candidate_action[:, 3:6] * rotation_error).sum(-1)
    common_angular = (common_action[:, 3:6] * rotation_error).sum(-1)
    linear_deficit = common_linear - candidate_linear
    angular_deficit = common_angular - candidate_angular
    enabled = recovery_mask(batch)
    terminal = (
        position_error.norm(dim=-1)
        <= spec.terminal_multiplier * spec.position_tolerance_m
    ) & (
        rotation_error.norm(dim=-1)
        <= spec.terminal_multiplier * spec.orientation_tolerance_rad
    )
    residual = candidate[:, 0, :6] - common[:, :6]
    delta = residual.square().mean(-1).sqrt()
    admissible = (
        enabled & ~terminal
        & (linear_deficit <= spec.progress_tolerance)
        & (angular_deficit <= spec.progress_tolerance)
    )
    useful = admissible & (delta >= spec.minimum_candidate_delta)
    return dict(
        enabled=enabled, terminal=terminal,
        linear_deficit=linear_deficit, angular_deficit=angular_deficit,
        linear_progress_candidate=candidate_linear,
        linear_progress_common=common_linear,
        angular_progress_candidate=candidate_angular,
        angular_progress_common=common_angular,
        candidate_delta=delta, admissible=admissible, useful=useful,
    )


def completion_progress_loss(candidate, batch, spec):
    """Return a differentiable common-relative completion loss and metrics."""
    values = public_completion_progress(candidate, batch, spec)
    enabled = values["enabled"]
    progress_mask = enabled & ~values["terminal"]
    count = progress_mask.to(candidate.dtype).sum().clamp(min=1.)
    linear = (
        (values["linear_deficit"] - spec.progress_tolerance).clamp(min=0)
        / spec.linear_progress_scale
    )
    angular = (
        (values["angular_deficit"] - spec.progress_tolerance).clamp(min=0)
        / spec.angular_progress_scale
    )
    progress = ((linear + angular) * progress_mask.to(candidate.dtype)).sum() / count
    tail = candidate.new_zeros(())
    if spec.tail_weight > 0 and spec.tail_fraction > 0 and bool(progress_mask.any()):
        tail_count = max(1, math.ceil(int(progress_mask.sum()) * spec.tail_fraction))
        tail = (
            torch.topk(linear[progress_mask], tail_count).values.mean()
            + torch.topk(angular[progress_mask], tail_count).values.mean()
        )
    terminal_mask = (enabled & values["terminal"]).to(candidate.dtype)
    enabled_count = enabled.to(candidate.dtype).sum().clamp(min=1.)
    terminal = (
        values["candidate_delta"].square() * terminal_mask
    ).sum() / enabled_count
    loss = (
        spec.progress_weight * (progress + spec.tail_weight * tail)
        + spec.terminal_weight * terminal
    )
    metrics = dict(
        completion_progress_loss=progress,
        completion_tail_progress_loss=tail,
        completion_terminal_loss=terminal,
        completion_enabled_fraction=enabled.to(candidate.dtype).mean(),
        completion_terminal_fraction=(enabled & values["terminal"]).to(candidate.dtype).mean(),
        completion_linear_deficit=(values["linear_deficit"].clamp(min=0)
                                   * progress_mask.to(candidate.dtype)).sum()/count,
        completion_angular_deficit=(values["angular_deficit"].clamp(min=0)
                                    * progress_mask.to(candidate.dtype)).sum()/count,
        completion_admissible_fraction=(values["admissible"].to(candidate.dtype).sum()
                                        / count),
        completion_useful_fraction=(values["useful"].to(candidate.dtype).sum()
                                    / count),
        completion_candidate_delta=(values["candidate_delta"]
                                    * enabled.to(candidate.dtype)).sum()/count,
    )
    return loss, metrics


def _public_execution_progress_support(
    batch, position_error, rotation_error, feedback, spec,
):
    """Exact first-action progress support of the public pose decoder.

    The decoder is coordinatewise monotone and piecewise affine in the pose
    request and phase-3 force request.  Vectorized Box-corner evaluation is
    therefore exact while retaining every state-dependent public lead,
    compensation, waiting, retreating and admittance clamp.
    """
    batch_size = position_error.shape[0]
    base = batch.get("base_proto")
    if base is None or base.ndim != 3 or base.shape[0] != batch_size:
        raise ValueError("candidate support requires public base proto commands")
    base = base[:, 0]
    lower = base - spec.candidate_residual_scale
    upper = base + spec.candidate_residual_scale
    lower = torch.cat((lower[:, :12].clamp(-1., 1.), lower[:, 12:].clamp(0., 1.)), -1)
    upper = torch.cat((upper[:, :12].clamp(-1., 1.), upper[:, 12:].clamp(0., 1.)), -1)
    bounds = base.new_tensor(BOUNDS)
    lower, upper = lower * bounds, upper * bounds

    linear_corners = position_error.new_tensor([
        [x, y, z, force]
        for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)
        for force in (0., 1.)
    ])
    linear_proto = position_error.new_zeros((batch_size, 16, 13))
    linear_proto[..., :3] = (
        lower[:, None, :3] + linear_corners[None, :, :3]
        * (upper - lower)[:, None, :3]
    )
    linear_proto[..., 12] = (
        lower[:, None, 12] + linear_corners[None, :, 3]
        * (upper - lower)[:, None, 12]
    )
    linear_actions = decode_pose_transport(
        linear_proto, batch["decoder"], feedback,
    )[..., :3]
    linear_support = (
        linear_actions * position_error[:, None, :]
    ).sum(-1).max(-1).values

    angular_corners = rotation_error.new_tensor([
        [x, y, z] for x in (0., 1.) for y in (0., 1.) for z in (0., 1.)
    ])
    angular_proto = rotation_error.new_zeros((batch_size, 8, 13))
    angular_proto[..., 3:6] = (
        lower[:, None, 3:6] + angular_corners[None]
        * (upper - lower)[:, None, 3:6]
    )
    angular_actions = decode_pose_transport(
        angular_proto, batch["decoder"], feedback,
    )[..., 3:6]
    angular_support = (
        angular_actions * rotation_error[:, None, :]
    ).sum(-1).max(-1).values
    return linear_support, angular_support


def public_finite_time_progress(candidate, batch, spec, feedback):
    """Return public T3 progress deficits relative to time and common action."""
    if not isinstance(spec, PublicFiniteTimeProgressSpec):
        raise ValueError("explicit public finite-time progress spec required")
    if candidate.ndim != 3 or candidate.shape[-1] != 13:
        raise ValueError("candidate commands must be batch x horizon x 13")
    common = batch["common_action"]
    if common.shape != (candidate.shape[0], 13):
        raise ValueError("one normalized common action per candidate state required")
    remaining = batch.get("remaining_controls")
    if remaining is None or remaining.reshape(-1).shape != (candidate.shape[0],):
        raise ValueError("one public remaining-control count per candidate state required")
    phase = batch["decoder"]["phase"].reshape(-1)
    if phase.shape != (candidate.shape[0],):
        raise ValueError("one public decoder phase per candidate state required")
    current = batch["current"]
    reference = batch["completion_reference"]
    if reference.shape != (candidate.shape[0], 12):
        raise ValueError("one public goal position and rotation per candidate state required")

    scale = candidate.new_tensor(BOUNDS[:6])
    candidate_action = candidate[:, 0, :6] * scale
    common_action = common[:, :6] * scale
    position_error = reference[:, :3] - current["position"]
    goal_rotation = reference[:, 3:12].reshape(-1, 3, 3)
    rotation_error = _batched_orientation_error(current["rotation"], goal_rotation)
    position_norm = position_error.norm(dim=-1)
    rotation_norm = rotation_error.norm(dim=-1)
    candidate_linear = (candidate_action[:, :3] * position_error).sum(-1)
    common_linear = (common_action[:, :3] * position_error).sum(-1)
    candidate_angular = (candidate_action[:, 3:6] * rotation_error).sum(-1)
    common_angular = (common_action[:, 3:6] * rotation_error).sum(-1)

    available = (
        remaining.reshape(-1).to(candidate.dtype) - spec.hold_controls
    ).clamp(min=1.)
    seconds = available * spec.control_dt_seconds
    required_linear_raw = position_norm * (
        (position_norm - spec.position_tolerance_m).clamp(min=0.) / seconds
    )
    required_angular_raw = rotation_norm * (
        (rotation_norm - spec.orientation_tolerance_rad).clamp(min=0.) / seconds
    )
    # The one-step target must remain inside the same normalized 13D action
    # support used by every compared actor.  Preserve the unclipped rate as a
    # diagnostic: a false feasibility flag means no one-step command can meet
    # the remaining-time rate from that recorded state.
    support_linear, support_angular = _public_execution_progress_support(
        batch, position_error, rotation_error, feedback, spec,
    )
    target_linear_raw = torch.maximum(common_linear, required_linear_raw)
    target_angular_raw = torch.maximum(common_angular, required_angular_raw)
    target_linear = torch.minimum(target_linear_raw, support_linear)
    target_angular = torch.minimum(target_angular_raw, support_angular)
    linear_deficit = target_linear - candidate_linear
    angular_deficit = target_angular - candidate_angular
    enabled = recovery_mask(batch) & (phase == spec.phase)
    terminal = (
        position_norm <= spec.position_tolerance_m
    ) & (
        rotation_norm <= spec.orientation_tolerance_rad
    )
    residual = candidate[:, 0, :6] - common[:, :6]
    delta = residual.square().mean(-1).sqrt()
    admissible = (
        enabled & ~terminal
        & (linear_deficit <= spec.progress_tolerance)
        & (angular_deficit <= spec.progress_tolerance)
    )
    useful = admissible & (delta >= spec.minimum_candidate_delta)
    return dict(
        enabled=enabled, terminal=terminal,
        remaining_controls=remaining.reshape(-1),
        linear_deficit=linear_deficit, angular_deficit=angular_deficit,
        linear_progress_candidate=candidate_linear,
        linear_progress_common=common_linear,
        linear_progress_required=target_linear,
        linear_progress_required_raw=required_linear_raw,
        linear_progress_support=support_linear,
        linear_rate_feasible=target_linear_raw <= support_linear,
        angular_progress_candidate=candidate_angular,
        angular_progress_common=common_angular,
        angular_progress_required=target_angular,
        angular_progress_required_raw=required_angular_raw,
        angular_progress_support=support_angular,
        angular_rate_feasible=target_angular_raw <= support_angular,
        candidate_delta=delta, admissible=admissible, useful=useful,
    )


def finite_time_progress_loss(candidate, batch, spec, feedback):
    """Differentiable T3 finite-time arrival and terminal-hold objective."""
    values = public_finite_time_progress(candidate, batch, spec, feedback)
    enabled = values["enabled"]
    progress_mask = enabled & ~values["terminal"]
    count = progress_mask.to(candidate.dtype).sum().clamp(min=1.)
    linear = (
        (values["linear_deficit"] - spec.progress_tolerance).clamp(min=0.)
        / spec.linear_progress_scale
    )
    angular = (
        (values["angular_deficit"] - spec.progress_tolerance).clamp(min=0.)
        / spec.angular_progress_scale
    )
    progress = ((linear + angular) * progress_mask.to(candidate.dtype)).sum() / count
    tail = candidate.new_zeros(())
    if spec.tail_weight > 0 and spec.tail_fraction > 0 and bool(progress_mask.any()):
        tail_count = max(1, math.ceil(int(progress_mask.sum()) * spec.tail_fraction))
        tail = (
            torch.topk(linear[progress_mask], tail_count).values.mean()
            + torch.topk(angular[progress_mask], tail_count).values.mean()
        )
    terminal_mask = (enabled & values["terminal"]).to(candidate.dtype)
    terminal_count = terminal_mask.sum().clamp(min=1.)
    enabled_count = enabled.to(candidate.dtype).sum().clamp(min=1.)
    terminal = (
        values["candidate_delta"].square() * terminal_mask
    ).sum() / terminal_count
    loss = (
        spec.progress_weight * (progress + spec.tail_weight * tail)
        + spec.terminal_weight * terminal
    )
    metrics = dict(
        finite_time_progress_loss=progress,
        finite_time_tail_loss=tail,
        finite_time_terminal_loss=terminal,
        finite_time_enabled_fraction=enabled.to(candidate.dtype).mean(),
        finite_time_terminal_fraction=(enabled & values["terminal"]).to(candidate.dtype).mean(),
        finite_time_linear_deficit=(values["linear_deficit"].clamp(min=0.)
                                    * progress_mask.to(candidate.dtype)).sum()/count,
        finite_time_angular_deficit=(values["angular_deficit"].clamp(min=0.)
                                     * progress_mask.to(candidate.dtype)).sum()/count,
        finite_time_admissible_fraction=(values["admissible"].to(candidate.dtype).sum()
                                         / count),
        finite_time_useful_fraction=(values["useful"].to(candidate.dtype).sum()
                                     / count),
        finite_time_candidate_delta=(values["candidate_delta"]
                                     * enabled.to(candidate.dtype)).sum()/count,
        finite_time_remaining_controls=(values["remaining_controls"].to(candidate.dtype)
                                        * enabled.to(candidate.dtype)).sum()/enabled_count,
        finite_time_linear_rate_feasible_fraction=(
            values["linear_rate_feasible"].to(candidate.dtype)
            * progress_mask.to(candidate.dtype)
        ).sum()/count,
        finite_time_angular_rate_feasible_fraction=(
            values["angular_rate_feasible"].to(candidate.dtype)
            * progress_mask.to(candidate.dtype)
        ).sum()/count,
    )
    return loss, metrics


def paired_finite_time_progress_loss(
    candidate, task_only, batch, finite_time_spec, paired_spec, feedback,
):
    """Penalize failure to beat task-only progress on both public task axes."""
    if not isinstance(paired_spec, PublicPairedProgressSpec):
        raise ValueError("explicit paired-progress spec required")
    if candidate.shape != task_only.shape:
        raise ValueError("paired progress requires matching command horizons")
    values = public_finite_time_progress(
        candidate, batch, finite_time_spec, feedback,
    )
    with torch.no_grad():
        reference = public_finite_time_progress(
            task_only, batch, finite_time_spec, feedback,
        )
    mask = values["enabled"] & ~values["terminal"]
    count = mask.to(candidate.dtype).sum().clamp(min=1.)
    linear_gap = (
        reference["linear_progress_candidate"].detach()
        + paired_spec.linear_margin
        - values["linear_progress_candidate"]
    )
    angular_gap = (
        reference["angular_progress_candidate"].detach()
        + paired_spec.angular_margin
        - values["angular_progress_candidate"]
    )
    linear_loss = (
        linear_gap.clamp(min=0.) / finite_time_spec.linear_progress_scale
    )
    angular_loss = (
        angular_gap.clamp(min=0.) / finite_time_spec.angular_progress_scale
    )
    paired = (
        (linear_loss + angular_loss) * mask.to(candidate.dtype)
    ).sum() / count
    tail = candidate.new_zeros(())
    tail_modes = candidate.new_zeros(())
    if paired_spec.tail_fraction > 0.:
        requested_mode = batch.get("requested_mode")
        if requested_mode is None:
            raise ValueError("mode-tail paired progress requires requested_mode")
        requested_mode = requested_mode.reshape(-1).to(mask.device)
        if requested_mode.shape != mask.shape:
            raise ValueError("requested_mode shape differs from paired progress mask")
        mode_tails = []
        for mode in torch.unique(requested_mode[mask], sorted=True):
            mode_mask = mask & (requested_mode == mode)
            mode_count = int(mode_mask.sum())
            if mode_count < 1:
                continue
            tail_count = max(
                1, math.ceil(mode_count * paired_spec.tail_fraction),
            )
            mode_tails.append(
                torch.topk(linear_loss[mode_mask], tail_count).values.mean()
                + torch.topk(angular_loss[mode_mask], tail_count).values.mean()
            )
        if mode_tails:
            tail = torch.stack(mode_tails).mean()
            tail_modes = tail.new_tensor(float(len(mode_tails)))
    return paired_spec.weight * (
        paired + paired_spec.tail_weight * tail
    ), {
        "paired_progress_loss": paired,
        "paired_progress_tail_loss": tail,
        "paired_progress_tail_modes": tail_modes,
        "paired_linear_gain": (
            (values["linear_progress_candidate"]
             - reference["linear_progress_candidate"].detach())
            * mask.to(candidate.dtype)
        ).sum() / count,
        "paired_angular_gain": (
            (values["angular_progress_candidate"]
             - reference["angular_progress_candidate"].detach())
            * mask.to(candidate.dtype)
        ).sum() / count,
        "paired_linear_margin_fraction": (
            (linear_gap <= 0).to(candidate.dtype) * mask.to(candidate.dtype)
        ).sum() / count,
        "paired_angular_margin_fraction": (
            (angular_gap <= 0).to(candidate.dtype) * mask.to(candidate.dtype)
        ).sum() / count,
    }


def select_completion_compatible_checkpoint(
        *, candidate_proxy, task_only_proxy, task_tolerance,
        linear_deficit_p95, angular_deficit_p95, progress_tolerance,
        useful_fraction, minimum_useful_fraction, candidate, task_only):
    """Select a T4 candidate only when task compatibility and utility hold."""
    values = (
        candidate_proxy, task_only_proxy, task_tolerance,
        linear_deficit_p95, angular_deficit_p95, progress_tolerance,
        useful_fraction, minimum_useful_fraction,
    )
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("finite completion selection values required")
    if any(value < 0 for value in (
            task_tolerance, progress_tolerance, useful_fraction,
            minimum_useful_fraction)):
        raise ValueError("nonnegative completion selection thresholds required")
    if useful_fraction > 1 or minimum_useful_fraction > 1:
        raise ValueError("completion utility fractions must be at most one")
    checks = dict(
        task_compatible=(candidate_proxy <= task_only_proxy + task_tolerance),
        linear_progress_compatible=(linear_deficit_p95 <= progress_tolerance),
        angular_progress_compatible=(angular_deficit_p95 <= progress_tolerance),
        candidate_utility=(useful_fraction >= minimum_useful_fraction),
    )
    accepted = all(checks.values())
    return dict(
        accepted=accepted, selected=candidate if accepted else task_only,
        reference="matched_task_only_and_common_completion",
        checks=checks,
        task_only_relative_gap=float(candidate_proxy)-float(task_only_proxy),
        linear_deficit_p95=float(linear_deficit_p95),
        angular_deficit_p95=float(angular_deficit_p95),
        useful_fraction=float(useful_fraction),
    )


def select_finite_time_compatible_checkpoint(
        *, candidate_proxy, task_only_proxy, task_tolerance,
        linear_deficit_p95, angular_deficit_p95, progress_tolerance,
        useful_fraction, minimum_useful_fraction, candidate, task_only):
    """Select a T3 candidate only when task, arrival and utility gates pass."""
    result = select_completion_compatible_checkpoint(
        candidate_proxy=candidate_proxy,
        task_only_proxy=task_only_proxy,
        task_tolerance=task_tolerance,
        linear_deficit_p95=linear_deficit_p95,
        angular_deficit_p95=angular_deficit_p95,
        progress_tolerance=progress_tolerance,
        useful_fraction=useful_fraction,
        minimum_useful_fraction=minimum_useful_fraction,
        candidate=candidate,
        task_only=task_only,
    )
    result["reference"] = "matched_task_only_and_public_finite_time_progress"
    return result


def _source_current(source_features,response_spec):
    if source_features.shape[-1] != response_spec.feature_dim:
        raise ValueError("public source feature chart differs")
    offset=(response_spec.history_len-1)*(response_spec.frame_dim+1)
    if not bool(torch.all(source_features[...,offset+response_spec.frame_dim] == 1)):
        raise ValueError("source current history frame must be valid")
    return dict(
        position=source_features[...,offset:offset+3]*.5,
        rotation=source_features[...,offset+3:offset+12].reshape(-1,3,3),
        wrench=source_features[...,offset+18:offset+24]
            *source_features.new_tensor([25.]*3+[4.]*3),
    )


def public_condition_distance(batch,response_spec):
    """Normalized current task-space distance to the selected source history."""
    source=_source_current(batch["source_features"],response_spec)
    current=batch["current"]
    position=((current["position"]-source["position"])/.02).square().mean(-1)
    rotation=((current["rotation"]-source["rotation"])/.1).square().mean((-1,-2))
    wrench=((current["wrench"]-source["wrench"])
            /current["wrench"].new_tensor([5.]*3+[1.]*3)).square().mean(-1)
    return ((position+rotation+wrench)/3).sqrt()


def response_disagreement(predicted,source):
    """RMS ensemble disagreement in the normalized 18D public response chart."""
    if predicted.ndim != 3 or source.shape != predicted.shape or predicted.shape[-1] < 18:
        raise ValueError("matching response ensembles must have shape members x batch x outputs")
    target=predicted[...,:18].std(0,unbiased=False).square().mean(-1).sqrt()
    reference=source[...,:18].std(0,unbiased=False).square().mean(-1).sqrt()
    return torch.maximum(target,reference)


def response_matching(predicted, source):
    """Per-sample MSE in the normalized public 18D response chart."""
    if predicted.ndim != 3 or source.shape != predicted.shape or predicted.shape[-1] < 18:
        raise ValueError("matching response ensembles must have shape members x batch x outputs")
    return ((predicted[..., :18] - source[..., :18]) ** 2).mean((0, 2))


def relative_response_matching(matching, task_only_reference, spec):
    """Dimensionless discrepancy relative to a frozen matched-control value."""
    if not isinstance(spec, (PublicRelativeResponseSpec, PublicDirectionalResponseSpec)):
        raise ValueError("explicit relative-response spec required")
    if matching.ndim != 1 or task_only_reference.shape != matching.shape:
        raise ValueError("one matched response reference per sample required")
    if not bool(torch.isfinite(task_only_reference).all()) or bool((task_only_reference < 0).any()):
        raise ValueError("finite nonnegative task-only response references required")
    denominator = task_only_reference.detach().clamp_min(spec.reference_floor)
    return matching / denominator


@torch.no_grad()
def recovery_response_reference(policy, response, feedback, batch):
    """Matched task-only response discrepancy for the exact public states."""
    _, actual = recovery_policy_commands(policy, feedback, batch)
    predicted = response(batch["target_features"], actual)
    source = response(batch["source_features"], batch["source_commands"])
    return response_matching(predicted, source)


def conditional_anchoring_mask(*,task,task_only_reference,condition_distance,
                               disagreement,spec,extra_mask=None):
    """Detached per-sample gate; no gradient crosses a discrete gate decision."""
    shape=task.shape
    if any(value.shape != shape for value in
           (task_only_reference,condition_distance,disagreement)):
        raise ValueError("one task/reference/support/disagreement value per sample required")
    compatible=task.detach() <= task_only_reference.detach()+spec.task_tolerance
    supported=condition_distance.detach() <= spec.maximum_condition_distance
    certain=disagreement.detach() <= spec.maximum_response_disagreement
    mask=compatible & supported & certain
    if extra_mask is not None:
        if extra_mask.reshape(-1).shape != shape:
            raise ValueError("one extra gate value per sample required")
        mask=mask & extra_mask.reshape(-1).bool()
    return mask,dict(task_compatible=compatible,support_accepted=supported,
                     disagreement_accepted=certain)


def select_task_compatible_checkpoint(*,candidate_proxy,task_only_proxy,tolerance,
                                      candidate,task_only):
    """Select the candidate only inside the task-only-relative validation band."""
    values=(candidate_proxy,task_only_proxy,tolerance)
    if any(not math.isfinite(float(value)) for value in values) or tolerance < 0:
        raise ValueError("finite proxies and nonnegative tolerance required")
    accepted=float(candidate_proxy) <= float(task_only_proxy)+float(tolerance)
    return dict(accepted=accepted,selected=candidate if accepted else task_only,
                reference="matched_task_only",
                task_only_relative_gap=float(candidate_proxy)-float(task_only_proxy))


def directional_source_task_advantage(*,response,batch,task_only_reference,
                                      spec,ensemble_task_spec=None,source_prediction=None):
    """Evaluate source-command task value with an explicit state convention.

    Historical directional checkpoints omit ``source_task_state`` and retain
    the old source-state proxy. New experiments can request a same-target-state
    counterfactual without changing their archived objective.
    """
    with torch.no_grad():
        if spec.source_task_state == "target":
            source_task_prediction = response(
                batch["target_features"], batch["source_commands"],
            )
        elif source_prediction is not None:
            source_task_prediction = source_prediction
        else:
            source_task_prediction = response(
                batch["source_features"], batch["source_commands"],
            )
        source_member_task = task_proxy(
            source_task_prediction,batch["current"],batch["task_reference"],
        )
        return ((task_only_reference.detach()-source_member_task).amin(0)
                if ensemble_task_spec is not None else
                task_only_reference.detach()-source_member_task.mean(0))


def _gated_objective(*,policy,response,batch,normalized,actual,response_weight,
                     task_only_reference,gate_spec,change_mask=None,
                     task_only_response_reference=None,response_spec=None,
                     apply_anchoring_gate=True,ensemble_task_spec=None):
    if task_only_reference is None:
        raise ValueError("gated anchoring requires a matched task-only reference")
    predicted=response(batch["target_features"],actual)
    with torch.no_grad():
        source=response(batch["source_features"],batch["source_commands"])
    member_task=task_proxy(predicted,batch["current"],batch["task_reference"])
    robust_member_gap=None
    if ensemble_task_spec is None:
        task=member_task.mean(0)
        task_reference=task_only_reference
    else:
        robust_member_gap,task=ensemble_robust_paired_task(
            member_task,task_only_reference,ensemble_task_spec,
        )
        task_reference=torch.zeros_like(task)
    matching=response_matching(predicted,source)
    if response_spec is None:
        response_term=matching
        relative_matching=None
    else:
        if task_only_response_reference is None:
            raise ValueError("relative anchoring requires a matched task-only response reference")
        relative_matching=relative_response_matching(
            matching,task_only_response_reference,response_spec,
        )
        response_term=relative_matching
    disagreement=response_disagreement(predicted,source)
    condition=public_condition_distance(batch,response.spec)
    gated_mask,parts=conditional_anchoring_mask(
        task=task,task_only_reference=task_reference,
        condition_distance=condition,disagreement=disagreement,spec=gate_spec,
        extra_mask=change_mask,
    )
    if apply_anchoring_gate:
        mask=gated_mask
    elif change_mask is None:
        mask=torch.ones_like(gated_mask)
    else:
        mask=change_mask.reshape(-1).bool()
    source_task_advantage=None
    source_task_advantage_mask=None
    if isinstance(response_spec,PublicDirectionalResponseSpec):
        source_task_advantage=directional_source_task_advantage(
            response=response,batch=batch,task_only_reference=task_only_reference,
            spec=response_spec,ensemble_task_spec=ensemble_task_spec,
            source_prediction=source,
        )
        source_task_advantage_mask=(
            source_task_advantage>=response_spec.minimum_source_task_advantage
        )
        mask=mask & source_task_advantage_mask
    peaks=torch.expm1(predicted[...,18:20]).clamp(min=0)
    risk=((peaks[...,0]/25.-1).clamp(min=0)**2
          +(peaks[...,1]/10.-1).clamp(min=0)**2).amax(0)
    change=((normalized-batch["base_proto"])**2).mean((1,2))
    if change_mask is not None:
        change=change*change_mask.reshape(-1).to(change.dtype)
    weight=policy.spec.response_weight if response_weight is None else response_weight
    violation=(task-task_reference-gate_spec.task_tolerance).clamp(min=0)
    # Keep the task objective active throughout the band.  The detached gate
    # only controls the extra response term.
    loss=(task+gate_spec.band_penalty*violation
          +weight*mask.to(matching.dtype)*response_term
          +policy.spec.risk_weight*risk+policy.spec.change_weight*change).mean()
    metrics=dict(
        task_proxy=member_task.mean(),response_matching=matching.mean(),
        gated_response_matching=(mask.to(matching.dtype)*matching).mean(),
        anchoring_fraction=mask.to(task.dtype).mean(),
        task_compatible_fraction=parts["task_compatible"].to(task.dtype).mean(),
        support_accepted_fraction=parts["support_accepted"].to(task.dtype).mean(),
        disagreement_accepted_fraction=parts["disagreement_accepted"].to(task.dtype).mean(),
        condition_distance=condition.mean(),response_disagreement=disagreement.mean(),
        task_band_violation=violation.mean(),risk_proxy=risk.mean(),
        command_change=change.mean(),
    )
    if relative_matching is not None:
        metrics.update(
            relative_response_matching=relative_matching.mean(),
            gated_relative_response_matching=(
                mask.to(relative_matching.dtype)*relative_matching
            ).mean(),
            task_only_response_matching=task_only_response_reference.mean(),
        )
    if source_task_advantage is not None:
        metrics.update(
            source_task_advantage=source_task_advantage.mean(),
            source_task_advantage_fraction=(
                source_task_advantage_mask.to(task.dtype).mean()
            ),
            directional_anchoring_fraction=mask.to(task.dtype).mean(),
        )
    if robust_member_gap is not None:
        metrics.update(
            robust_paired_task_gap=task.mean(),
            robust_all_members_nonworse_fraction=(
                (robust_member_gap <= 0.).all(0).to(task.dtype).mean()
            ),
            robust_worst_member_task_proxy=member_task.amax(0).mean(),
        )
    return loss,metrics


class PublicAdditiveBandTransportTrainer(PublicTransportTrainer):
    """Ungated C+H+wR control aligned with the gated-v2 task objective."""
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None):
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError("explicit public anchoring-gate spec required")
        normalized=self.policy(
            batch["target_features"],batch["source_features"],
            batch["source_commands"],batch["base_proto"],
            batch["decoder"]["phase"],decoder_context(batch["decoder"],self.feedback),
        )
        actual=decode_pose_transport(
            normalized*normalized.new_tensor(BOUNDS),batch["decoder"],self.feedback,
        )/normalized.new_tensor(BOUNDS)
        return _gated_objective(
            policy=self.policy,response=self.response,batch=batch,normalized=normalized,
            actual=actual,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
            apply_anchoring_gate=False,
        )


class PublicRecoveryAdditiveBandTransportTrainer(PublicTransportTrainer):
    """Recovery-effective ungated C+H+wR control for the aligned ablation."""
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None):
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError("explicit public anchoring-gate spec required")
        normalized,actual=recovery_policy_commands(self.policy,self.feedback,batch)
        return _gated_objective(
            policy=self.policy,response=self.response,batch=batch,normalized=normalized,
            actual=actual,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
            change_mask=recovery_mask(batch),apply_anchoring_gate=False,
        )


class PublicGatedTransportTrainer(PublicTransportTrainer):
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None):
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError("explicit public anchoring-gate spec required")
        normalized=self.policy(
            batch["target_features"],batch["source_features"],
            batch["source_commands"],batch["base_proto"],
            batch["decoder"]["phase"],decoder_context(batch["decoder"],self.feedback),
        )
        actual=decode_pose_transport(
            normalized*normalized.new_tensor(BOUNDS),batch["decoder"],self.feedback,
        )/normalized.new_tensor(BOUNDS)
        return _gated_objective(
            policy=self.policy,response=self.response,batch=batch,normalized=normalized,
            actual=actual,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
        )


class PublicRecoveryGatedTransportTrainer(PublicTransportTrainer):
    def objective(self,batch,*,response_weight=None,task_only_reference=None,
                  gate_spec=None,completion_spec=None,finite_time_spec=None,
                  task_only_response_reference=None,response_spec=None,
                  task_only_commands=None,paired_progress_spec=None,
                  ensemble_task_spec=None,apply_anchoring_gate=True):
        if not isinstance(gate_spec,PublicAnchoringGateSpec):
            raise ValueError("explicit public anchoring-gate spec required")
        normalized,actual=recovery_policy_commands(self.policy,self.feedback,batch)
        loss,metrics=_gated_objective(
            policy=self.policy,response=self.response,batch=batch,normalized=normalized,
            actual=actual,response_weight=response_weight,
            task_only_reference=task_only_reference,gate_spec=gate_spec,
            change_mask=recovery_mask(batch),
            task_only_response_reference=task_only_response_reference,
            response_spec=response_spec,ensemble_task_spec=ensemble_task_spec,
            apply_anchoring_gate=apply_anchoring_gate,
        )
        if completion_spec is not None and finite_time_spec is not None:
            raise ValueError("choose one terminal-progress objective")
        if completion_spec is None and finite_time_spec is None:
            return loss,metrics
        progress_actual = actual
        progress_task_only = task_only_commands
        progress_batch = batch
        if (
            paired_progress_spec is not None
            and paired_progress_spec.requested_mode is not None
        ):
            requested_mode = batch.get("requested_mode")
            if requested_mode is None:
                raise ValueError("mode-specific progress requires requested_mode")
            requested_mode = requested_mode.reshape(-1)
            mode_mask = requested_mode == paired_progress_spec.requested_mode
            if not bool(mode_mask.any()):
                raise ValueError("mode-specific progress batch has no requested state")

            def select_mode(value):
                if isinstance(value, dict):
                    return {key: select_mode(child) for key, child in value.items()}
                return value[mode_mask]

            progress_actual = actual[mode_mask]
            progress_task_only = (
                None if task_only_commands is None else task_only_commands[mode_mask]
            )
            progress_batch = select_mode(batch)
        if completion_spec is not None:
            progress,progress_metrics=completion_progress_loss(
                progress_actual,progress_batch,completion_spec,
            )
        else:
            progress,progress_metrics=finite_time_progress_loss(
                progress_actual,progress_batch,finite_time_spec,self.feedback,
            )
        paired = actual.new_zeros(())
        paired_metrics = {}
        if paired_progress_spec is not None:
            if finite_time_spec is None or task_only_commands is None:
                raise ValueError(
                    "paired progress requires finite-time task-only commands"
                )
            paired, paired_metrics = paired_finite_time_progress_loss(
                progress_actual, progress_task_only, progress_batch, finite_time_spec,
                paired_progress_spec, self.feedback,
            )
        elif task_only_commands is not None:
            raise ValueError("task-only progress commands require a paired spec")
        return loss+progress+paired,{
            **metrics, **progress_metrics, **paired_metrics,
        }

