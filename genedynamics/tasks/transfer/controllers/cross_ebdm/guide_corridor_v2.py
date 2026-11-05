"""T2 finite-state contact binding with source-compatible phase categories.

The base request is the same mode-specific public route used by the frozen T2
sentinel.  The learned policy may transport only task-authorized coordinates;
the normal contact unit, impedance and force fields remain task-owned.
"""
import numpy as np
import torch
from genedynamics.solvers.single.cross_ebdm.learning.response import public_frame,features
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS
from genedynamics.solvers.single.cross_ebdm.learning.substep_risk import (
    PublicJointAdmissionCalibration,
    PublicJointAdmissionEnsemble,
    PublicSubstepRiskCalibration,
    PublicSubstepRiskEnsemble,
    RiskSetEmptyError,
    calibrated_force_upper_bound,
    calibrated_joint_upper_bound,
    risk_admission_candidates,
    select_admissible_candidate,
    select_joint_admissible_candidate,
)
from genedynamics.solvers.single.response_mpc.arm import PublicArmTaskAdamMPCActor
from genedynamics.tasks.transfer.controllers.guide_contact import public_finite_state_guide_contact_policy
from genedynamics.tasks.transfer.controllers.guide_lift_route import CLEAR_Z
from .arm_dense import PublicArmTaskDenseActor, PublicArmTaskSourcePriorActor
from .arm_noise import PublicArmTaskNoiseActor
from .guide_lift import lift_state

PHASES=np.asarray([0,0,1,1,3],np.int64)


def corridor_state(base,public):
    goal=np.asarray(base.spec.config.goal)
    rotation=np.broadcast_to(np.diag([1.,-1.,-1.]),(1,3,3))
    if hasattr(base, "route"):
        if base.switched:
            active = base.completion
            phase = np.asarray([3 if active.stage >= 2 else 1], np.int64)
            integral = active.compensation.copy()
            target = active.initial.copy()
            target[:, 2] = CLEAR_Z
            if active.stage >= 1:
                target[:, :2] = goal[:2]
            if active.stage >= 2:
                target[:, 2] = goal[2]
            force = np.full(
                (1, 1), base.spec.config.desired_force if active.stage >= 2 else 0.,
            )
            reference = np.concatenate(
                (target, rotation.reshape(1, 9), force), axis=1,
            ).astype(np.float32)
            return phase, integral, reference
        active = base.route
    elif hasattr(base,'switched'):
        if base.switched:
            _,integral,reference=lift_state(base.candidate,public)
            phase=np.asarray([3 if base.candidate.stage==3 else 1],np.int64)
            return phase,integral,reference
        active=base.base
    else:
        active=base
    phase=PHASES[np.asarray([active.stage])]
    integral=active.compensation.copy()
    target=np.asarray([[.38,-.20,goal[2]]],dtype=float)
    if active.stage>=2:target[:,active.mode]=goal[active.mode]
    if active.stage>=3:target[:,:2]=goal[:2]
    transit=getattr(active,'TRANSIT_FORCE_N',active.transit_force_N)
    if transit is None:transit=base.spec.config.desired_force
    force=np.full((1,1),base.spec.config.desired_force if active.stage in (0,4) else transit)
    reference=np.concatenate((target,rotation.reshape(1,9),force),axis=1).astype(np.float32)
    return phase,integral,reference


class _CorridorPrepare:
    requires_source_query=True
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.spec.alias!='T2':raise ValueError('explicit T2 finite-state contact binding')
        self.base=public_finite_state_guide_contact_policy(self.spec,mode=self.mode)
    def prepare(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/self.spec.config.control_dt))
        if len(public['tcp_position'])!=1 or tick<self.last_tick or tick>self.last_tick+1:raise ValueError('one world and every public tick required')
        if tick!=self.last_tick:self.frames.append(public_frame(public,self.spec.config.robot,self.policy.response_spec)[0]);self.frames=self.frames[-self.policy.response_spec.history_len:];self.last_tick=tick
        base=self.base.act(public);phase,integral,reference=corridor_state(self.base,public);target=features(np.stack(self.frames),[len(self.frames)-1],self.spec.config.robot,'T2',[0,0,0],self.policy.response_spec)
        batch=dict(target_features=target,base_proto=np.repeat(base[:,None],self.policy.response_spec.horizon,axis=1)/BOUNDS,phase=phase,own_context=np.concatenate((integral/.2,np.zeros((1,3))),axis=1).astype(np.float32),current=dict(position=public['tcp_position'].copy(),rotation=public['tcp_rotation'].copy(),wrench=public['measured_wrench'].copy()),task_reference=reference,source_query_distance=np.zeros(1))
        if self.requires_source_query:
            source,distance,_=self.bank.query(public,phase,[self.mode]);batch.update(source_features=source['features'],source_commands=source['commands'],source_query_distance=distance)
        return batch,base
class PublicCorridorGuideDenseActor(_CorridorPrepare,PublicArmTaskDenseActor):
    VERSION='public-T2-finite-state-contact-dense-v3'


class PublicModeConditionedContactEstablishedCorridorGuideDenseActor(
        PublicCorridorGuideDenseActor):
    """Admit mode-0 tangent transport only after public contact establishment.

    The task-owned finite-state route declares contact established when its
    measured-load dwell advances to stage 2.  Before that event, the actor
    projects only the declared transport coordinates back to the public base
    request.  Mode 1 retains the original transport stream because its ordered
    route has a different feasible support and no establishment projection.
    The mode-0 admission latches after establishment; it is deliberately
    distinct from the rejected contact-loss recovery guard below.
    """

    VERSION = "public-T2-mode0-contact-established-transport-dense-v2"
    PROJECTED_TO_PUBLIC_BASE = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_contact_establishment_admission()

    def _reset_contact_establishment_admission(self):
        self.contact_establishment_applicable = self.mode == 0
        self.contact_establishment_admitted = not self.contact_establishment_applicable
        self.contact_establishment_projected_controls = 0
        self.contact_establishment_transport_controls = 0
        self.contact_establishment_tick = None

    def reset(self):
        super().reset()
        self._reset_contact_establishment_admission()

    def get_state(self):
        state = super().get_state()
        state["contact_establishment_admission"] = {
            "applicable": self.contact_establishment_applicable,
            "admitted": self.contact_establishment_admitted,
            "projected_controls": self.contact_establishment_projected_controls,
            "transport_controls": self.contact_establishment_transport_controls,
            "establishment_tick": self.contact_establishment_tick,
        }
        return state

    def restore(self, state):
        value = dict(state)
        admission = value.pop("contact_establishment_admission")
        super().restore(value)
        if bool(admission["applicable"]) != (self.mode == 0):
            raise ValueError("mode-conditioned contact admission differs")
        self.contact_establishment_applicable = bool(admission["applicable"])
        self.contact_establishment_admitted = bool(admission["admitted"])
        self.contact_establishment_projected_controls = int(
            admission["projected_controls"]
        )
        self.contact_establishment_transport_controls = int(
            admission["transport_controls"]
        )
        self.contact_establishment_tick = admission["establishment_tick"]

    def _public_contact_established(self):
        route = self.base
        if bool(getattr(route, "switched", False)):
            return True
        active = (
            route.route if hasattr(route, "route") else
            route.base if hasattr(route, "base") else route
        )
        return int(getattr(active, "stage", 0)) >= 2

    def act(self, public):
        tick = int(np.rint(
            np.asarray(public["elapsed_seconds"]).item()
            / self.spec.config.control_dt
        ))
        if tick == self.last_action_tick:
            return self.cached_action.copy()
        action = super().act(public)
        if not self.contact_establishment_admitted and self._public_contact_established():
            self.contact_establishment_admitted = True
            self.contact_establishment_tick = tick
        if self.contact_establishment_admitted:
            self.contact_establishment_transport_controls += 1
            self.last_selected_action = self.cached_action.copy()
            self.last_selector_code = 0
            return self.cached_action.copy()

        from genedynamics.solvers.single.cross_ebdm.learning.task_transport import (
            T2_TRANSPORT_COORDINATES,
        )
        selected = np.asarray(action, dtype=np.float32).copy()
        coordinates = list(T2_TRANSPORT_COORDINATES)
        selected[:, coordinates] = np.asarray(
            self.last_common_action, dtype=np.float32,
        )[:, coordinates]
        self.cached_action = selected
        self.last_selected_action = selected.copy()
        self.last_selector_code = self.PROJECTED_TO_PUBLIC_BASE
        self.contact_establishment_projected_controls += 1
        return selected.copy()


class PublicContactGuardedCorridorGuideDenseActor(PublicCorridorGuideDenseActor):
    """Project learned transport to the public base while contact recovers.

    The guard uses only the current public normal wrench and the public
    finite-state route's contact/outlet memory.  It never reads the evaluator's
    axial-support truth.  Task-owned normal motion, impedance, and force are
    already identical to the base; the projection therefore changes only the
    declared T2 transport coordinates.
    """

    VERSION = "public-T2-measured-contact-guarded-dense-v1"
    RECOVERY_LOADED_TICKS = 5

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._reset_contact_guard()

    def _reset_contact_guard(self):
        self.contact_guard_active = False
        self.contact_guard_recovery_ticks = 0
        self.contact_guard_controls = 0
        self.contact_guard_activations = 0

    def reset(self):
        super().reset()
        self._reset_contact_guard()

    def get_state(self):
        state = super().get_state()
        state["contact_guard"] = {
            "active": self.contact_guard_active,
            "recovery_ticks": self.contact_guard_recovery_ticks,
            "controls": self.contact_guard_controls,
            "activations": self.contact_guard_activations,
        }
        return state

    def restore(self, state):
        value = dict(state)
        guard = value.pop("contact_guard")
        super().restore(value)
        self.contact_guard_active = bool(guard["active"])
        self.contact_guard_recovery_ticks = int(guard["recovery_ticks"])
        self.contact_guard_controls = int(guard["controls"])
        self.contact_guard_activations = int(guard["activations"])

    def act(self, public):
        tick = int(np.rint(
            np.asarray(public["elapsed_seconds"]).item()
            / self.spec.config.control_dt
        ))
        if tick == self.last_action_tick:
            return self.cached_action.copy()
        action = super().act(public)
        route = self.base
        contact = (
            route.route if hasattr(route, "route") else
            route.base if hasattr(route, "base") else route
        )
        before_outlet = not bool(getattr(route, "switched", False))
        established = getattr(contact, "first_contact_tick", None) is not None
        normal_force = float(np.asarray(public["measured_wrench"])[0, 2])
        loaded = normal_force > max(.5, self.spec.config.contact_threshold)

        if established and before_outlet:
            if not loaded:
                if not self.contact_guard_active:
                    self.contact_guard_activations += 1
                self.contact_guard_active = True
                self.contact_guard_recovery_ticks = 0
            elif self.contact_guard_active:
                self.contact_guard_recovery_ticks += 1
                if self.contact_guard_recovery_ticks >= self.RECOVERY_LOADED_TICKS:
                    self.contact_guard_active = False
                    self.contact_guard_recovery_ticks = 0
        elif not before_outlet:
            self.contact_guard_active = False
            self.contact_guard_recovery_ticks = 0

        if self.contact_guard_active:
            public_base = np.asarray(route.cached, dtype=np.float32)
            action = np.asarray(action, dtype=np.float32).copy()
            from genedynamics.solvers.single.cross_ebdm.learning.task_transport import (
                T2_TRANSPORT_COORDINATES,
            )
            coordinates = list(T2_TRANSPORT_COORDINATES)
            action[:, coordinates] = public_base[:, coordinates]
            self.contact_guard_controls += 1
            self.cached_action = action.copy()
            self.last_selected_action = self.cached_action.copy()
            self.last_selector_code = 1
        else:
            self.last_selected_action = self.cached_action.copy()
            self.last_selector_code = 0
        return self.cached_action.copy()


class PublicRiskAdmittedCorridorGuideDenseActor(PublicCorridorGuideDenseActor):
    """Admit a transported T2 horizon using public 1 ms peak-risk bounds."""

    VERSION = "public-T2-substep-risk-admitted-dense-v2"

    def __init__(self, *args, risk_model: PublicSubstepRiskEnsemble,
                 risk_calibration: PublicSubstepRiskCalibration, **kwargs):
        super().__init__(*args, **kwargs)
        if risk_model.spec != self.policy.response_spec:
            raise ValueError("risk and transport public-history schemas differ")
        self.risk_model = risk_model.to(self.device).eval()
        for parameter in self.risk_model.parameters():
            parameter.requires_grad_(False)
        if not isinstance(risk_calibration, PublicSubstepRiskCalibration):
            raise TypeError("explicit public substep-risk calibration required")
        self.risk_calibration = risk_calibration
        self.risk_fallback_controls = 0
        self.risk_emergency_controls = 0
        self.risk_empty_controls = 0
        self.irreversible_risk_model_calls = 0
        self.last_risk_fraction = 0.0
        self.last_risk_index = len(risk_calibration.interpolation_fractions) - 1
        self.last_risk_upper_bounds_N = np.zeros(
            (len(risk_calibration.interpolation_fractions), 2), np.float32,
        )
        self.risk_fraction_counts = np.zeros(
            len(risk_calibration.interpolation_fractions), np.int64,
        )

    def reset(self):
        super().reset()
        self.risk_fallback_controls = 0
        self.risk_emergency_controls = 0
        self.risk_empty_controls = 0
        self.last_risk_fraction = 0.0
        self.last_risk_index = len(self.risk_calibration.interpolation_fractions) - 1
        self.last_risk_upper_bounds_N.fill(0.0)
        self.risk_fraction_counts.fill(0)

    def get_state(self):
        state = super().get_state()
        state.update(
            risk_fallback_controls=self.risk_fallback_controls,
            risk_emergency_controls=self.risk_emergency_controls,
            risk_empty_controls=self.risk_empty_controls,
            irreversible_risk_model_calls=self.irreversible_risk_model_calls,
            last_risk_fraction=self.last_risk_fraction,
            last_risk_index=self.last_risk_index,
            last_risk_upper_bounds_N=self.last_risk_upper_bounds_N.copy(),
            risk_fraction_counts=self.risk_fraction_counts.copy(),
        )
        return state

    def restore(self, state):
        super().restore(state)
        self.risk_fallback_controls = int(state["risk_fallback_controls"])
        self.risk_emergency_controls = int(state["risk_emergency_controls"])
        self.risk_empty_controls = int(state["risk_empty_controls"])
        self.irreversible_risk_model_calls = int(state["irreversible_risk_model_calls"])
        self.last_risk_fraction = float(state["last_risk_fraction"])
        self.last_risk_index = int(state["last_risk_index"])
        self.last_risk_upper_bounds_N = np.asarray(
            state["last_risk_upper_bounds_N"], dtype=np.float32,
        ).copy()
        self.risk_fraction_counts = np.asarray(
            state["risk_fraction_counts"], dtype=np.int64,
        ).copy()

    def act(self, public):
        tick = int(np.rint(
            np.asarray(public["elapsed_seconds"]).item() / self.spec.config.control_dt
        ))
        if tick == self.last_action_tick:
            return self.cached_action.copy()
        batch, base = self.prepare(public)
        if batch["source_query_distance"][0] > self.maximum_query_distance:
            self.fallback_controls += 1
            self.cached_action = base.copy()
            self.last_action_tick = tick
            return self.cached_action.copy()
        with torch.inference_mode():
            arguments = [
                torch.as_tensor(batch[key], device=self.device, dtype=torch.float32)
                for key in ("target_features", "source_features", "source_commands", "base_proto")
            ]
            phase = torch.as_tensor(batch["phase"], device=self.device)
            context = torch.as_tensor(
                batch["own_context"], device=self.device, dtype=torch.float32,
            )
            candidate = self.policy(*arguments, phase, context)
            self.irreversible_network_calls += 1
            chunks = risk_admission_candidates(
                candidate, arguments[3], self.risk_calibration,
            )
            batch_size, choices, horizon, width = chunks.shape
            risk_features = arguments[0][:, None].expand(
                batch_size, choices, arguments[0].shape[-1],
            ).reshape(batch_size * choices, -1)
            predicted = self.risk_model(
                risk_features, chunks.reshape(batch_size * choices, horizon, width),
            )
            upper = calibrated_force_upper_bound(
                predicted, self.risk_calibration,
            ).reshape(batch_size, choices, 2)
            index, admitted = select_admissible_candidate(
                upper, self.risk_calibration,
            )
            self.irreversible_risk_model_calls += 1
        if batch_size != 1:
            raise ValueError("T2 risk-admitted actor requires one world")
        if not bool(admitted.item()):
            self.last_risk_index = -1
            self.last_risk_fraction = float("nan")
            self.last_risk_upper_bounds_N = upper[0].cpu().numpy().astype(np.float32)
            self.risk_empty_controls += 1
            raise RiskSetEmptyError()
        chosen = int(index.item())
        self.last_risk_index = chosen
        fraction_count = len(self.risk_calibration.interpolation_fractions)
        self.last_risk_fraction = (
            float(self.risk_calibration.interpolation_fractions[chosen])
            if chosen < fraction_count else float("nan")
        )
        self.last_risk_upper_bounds_N = upper[0].cpu().numpy().astype(np.float32)
        if chosen < fraction_count:
            self.risk_fraction_counts[chosen] += 1
        if chosen == fraction_count - 1:
            self.risk_fallback_controls += 1
        elif chosen == fraction_count:
            self.risk_emergency_controls += 1
        self.cached_action = (
            chunks[0, chosen, 0] * chunks.new_tensor(BOUNDS)
        ).cpu().numpy()[None].astype(np.float32)
        self.last_action_tick = tick
        return self.cached_action.copy()


class PublicJointRiskAdmittedCorridorGuideDenseActor(PublicCorridorGuideDenseActor):
    """Admit T2 candidates only when force and guide contact both qualify."""

    VERSION = "public-T2-joint-force-contact-admitted-dense-v3"

    def __init__(self, *args, risk_model: PublicJointAdmissionEnsemble,
                 risk_calibration: PublicJointAdmissionCalibration, **kwargs):
        super().__init__(*args, **kwargs)
        if risk_model.spec != self.policy.response_spec:
            raise ValueError("joint risk and transport public-history schemas differ")
        if not isinstance(risk_calibration, PublicJointAdmissionCalibration):
            raise TypeError("explicit public joint-admission calibration required")
        self.risk_model = risk_model.to(self.device).eval()
        for parameter in self.risk_model.parameters():
            parameter.requires_grad_(False)
        self.risk_calibration = risk_calibration
        self.risk_fallback_controls = 0
        self.risk_emergency_controls = 0
        self.risk_empty_controls = 0
        self.irreversible_risk_model_calls = 0
        self.last_risk_fraction = 0.0
        self.last_risk_index = len(risk_calibration.interpolation_fractions) - 1
        choices = len(risk_calibration.interpolation_fractions) + 1
        self.last_risk_upper_bounds = np.zeros((choices, 4), np.float32)
        self.risk_fraction_counts = np.zeros(choices, np.int64)

    def reset(self):
        super().reset()
        self.risk_fallback_controls = 0
        self.risk_emergency_controls = 0
        self.risk_empty_controls = 0
        self.last_risk_fraction = 0.0
        self.last_risk_index = len(self.risk_calibration.interpolation_fractions) - 1
        self.last_risk_upper_bounds.fill(0.0)
        self.risk_fraction_counts.fill(0)

    def get_state(self):
        state = super().get_state()
        state.update(
            risk_fallback_controls=self.risk_fallback_controls,
            risk_emergency_controls=self.risk_emergency_controls,
            risk_empty_controls=self.risk_empty_controls,
            irreversible_risk_model_calls=self.irreversible_risk_model_calls,
            last_risk_fraction=self.last_risk_fraction,
            last_risk_index=self.last_risk_index,
            last_risk_upper_bounds=self.last_risk_upper_bounds.copy(),
            risk_fraction_counts=self.risk_fraction_counts.copy(),
        )
        return state

    def restore(self, state):
        super().restore(state)
        self.risk_fallback_controls = int(state["risk_fallback_controls"])
        self.risk_emergency_controls = int(state["risk_emergency_controls"])
        self.risk_empty_controls = int(state["risk_empty_controls"])
        self.irreversible_risk_model_calls = int(state["irreversible_risk_model_calls"])
        self.last_risk_fraction = float(state["last_risk_fraction"])
        self.last_risk_index = int(state["last_risk_index"])
        self.last_risk_upper_bounds = np.asarray(
            state["last_risk_upper_bounds"], dtype=np.float32,
        ).copy()
        self.risk_fraction_counts = np.asarray(
            state["risk_fraction_counts"], dtype=np.int64,
        ).copy()

    def act(self, public):
        tick = int(np.rint(
            np.asarray(public["elapsed_seconds"]).item() / self.spec.config.control_dt
        ))
        if tick == self.last_action_tick:
            return self.cached_action.copy()
        batch, base = self.prepare(public)
        with torch.inference_mode():
            arguments = [
                torch.as_tensor(batch[key], device=self.device, dtype=torch.float32)
                for key in ("target_features", "source_features", "source_commands", "base_proto")
            ]
            if batch["source_query_distance"][0] > self.maximum_query_distance:
                candidate = arguments[3]
                self.fallback_controls += 1
            else:
                candidate = self.policy(
                    *arguments,
                    torch.as_tensor(batch["phase"], device=self.device),
                    torch.as_tensor(batch["own_context"], device=self.device, dtype=torch.float32),
                )
                self.irreversible_network_calls += 1
            chunks = risk_admission_candidates(
                candidate, arguments[3], self.risk_calibration,
            )
            batch_size, choices, horizon, width = chunks.shape
            risk_features = arguments[0][:, None].expand(
                batch_size, choices, arguments[0].shape[-1],
            ).reshape(batch_size * choices, -1)
            predicted = self.risk_model(
                risk_features, chunks.reshape(batch_size * choices, horizon, width),
            )
            upper = calibrated_joint_upper_bound(
                predicted, self.risk_calibration,
            ).reshape(batch_size, choices, 4)
            index, admitted = select_joint_admissible_candidate(
                upper, self.risk_calibration,
            )
            self.irreversible_risk_model_calls += 1
        if batch_size != 1:
            raise ValueError("T2 joint-admitted actor requires one world")
        self.last_risk_upper_bounds = upper[0].cpu().numpy().astype(np.float32)
        if not bool(admitted.item()):
            self.last_risk_index = -1
            self.last_risk_fraction = float("nan")
            self.risk_empty_controls += 1
            raise RiskSetEmptyError()
        chosen = int(index.item())
        fraction_count = len(self.risk_calibration.interpolation_fractions)
        self.last_risk_index = chosen
        self.last_risk_fraction = (
            float(self.risk_calibration.interpolation_fractions[chosen])
            if chosen < fraction_count else float("nan")
        )
        self.risk_fraction_counts[chosen] += 1
        if chosen == fraction_count - 1:
            self.risk_fallback_controls += 1
        elif chosen == fraction_count:
            self.risk_emergency_controls += 1
        self.cached_action = (
            chunks[0, chosen, 0] * chunks.new_tensor(BOUNDS)
        ).cpu().numpy()[None].astype(np.float32)
        self.last_action_tick = tick
        return self.cached_action.copy()
class PublicCorridorGuideMPCActor(_CorridorPrepare,PublicArmTaskAdamMPCActor):
    VERSION='public-T2-finite-state-contact-sourcefree-AdamMPC-v3'
    requires_source_query=False
    def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.maximum_query_distance=np.inf
class PublicCorridorGuideNoiseActor(PublicArmTaskNoiseActor):
    VERSION='public-T2-finite-state-contact-local-noise-v3'
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        if self.spec.alias!='T2':raise ValueError('explicit T2 corridor excitation')
        self.base=public_finite_state_guide_contact_policy(self.spec,mode=self.mode)


class PublicCorridorGuideSourcePriorActor(_CorridorPrepare,PublicArmTaskSourcePriorActor):
    VERSION='public-T2-finite-state-contact-source-prior-v3'
