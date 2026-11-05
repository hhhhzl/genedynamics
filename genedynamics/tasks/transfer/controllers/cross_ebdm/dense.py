"""Public-history dense transport execution through the shared NumPy decoder."""
from copy import deepcopy
import numpy as np
import torch

from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy
from genedynamics.tasks.transfer.behavior import teacher_reference
from genedynamics.core.control.transfer import orientation_error
from genedynamics.solvers.single.cross_ebdm.learning.response import public_frame,features
from genedynamics.solvers.single.cross_ebdm.learning.residual_transport import BOUNDS,decoder_state,decoder_context,decode_pose_transport,task_proxy
from genedynamics.solvers.single.cross_ebdm.learning.completion_value import response_endpoint_value


class ProtoInterceptor:
    def __init__(self):self.override=None;self.raw=None
    def apply(self,value,phase):
        self.raw=value.copy()
        return value if self.override is None else self.override.copy()


def canonical_proto(raw):
    value=np.array(raw,copy=True);value[...,:3]=np.clip(value[...,:3],-.02,.02);value[...,3:6]=np.clip(value[...,3:6],-.12,.12)
    value[...,6:12]=np.clip(value[...,6:12],-1.,1.);value[...,12]=np.clip(value[...,12],0.,25.)
    return value


class PublicDenseTransportActor:
    """One independent world; optional batching shares immutable neural weights."""
    def __init__(self,spec,policy,source_bank,feedback,*,prior_path=None,mode=0,seed=0,device='cpu',maximum_query_distance=5.,retreat_release_version='legacy'):
        if spec.alias!='T3':raise ValueError('dense decoder currently physically binds T3; other tasks require separate qualification')
        if source_bank.spec!=policy.response_spec:raise ValueError('source query and learned policy schemas differ')
        self.spec,self.policy,self.source_bank,self.feedback=spec,policy,source_bank,feedback;self.device=torch.device(device);self.mode=int(mode);self.seed=int(seed);self.maximum_query_distance=maximum_query_distance
        if retreat_release_version == 'legacy':
            base_type = PublicBiasFeedbackPolicy
        elif retreat_release_version == 'safe_settled_v1':
            from genedynamics.tasks.transfer.controllers.mating_feedback import PublicLivenessMatingFeedbackPolicy
            base_type = PublicLivenessMatingFeedbackPolicy
        else:
            raise ValueError('unknown T3 retreat release version')
        self.base=base_type.load(spec,prior_path,feedback=feedback,modes=[mode],seed=seed);self.interceptor=ProtoInterceptor();self.base.pose_transport=self.interceptor
        self.policy.to(self.device).eval();self.frames=[];self.last_tick=-1;self.last_query_distance=None;self.fallback_controls=0;self.irreversible_network_calls=0

    def get_state(self):return deepcopy(dict(base=self.base.get_state(),frames=self.frames,last_tick=self.last_tick,last_query_distance=self.last_query_distance,fallback_controls=self.fallback_controls))

    def restore(self,state):
        self.base.restore(state['base']);self.frames=deepcopy(state['frames']);self.last_tick=state['last_tick'];self.last_query_distance=state['last_query_distance'];self.fallback_controls=state['fallback_controls'];self.interceptor.override=None

    def reset(self):
        self.base.reset(modes=[self.mode],seed=self.seed);self.frames=[];self.last_tick=-1;self.last_query_distance=None;self.fallback_controls=0;self.interceptor.override=None

    def prepare(self,public):
        if len(public['tcp_position'])!=1:raise ValueError('one causal history per actor; use independent-world batching')
        tick=int(round(float(np.asarray(public['elapsed_seconds']).reshape(-1)[0])/self.spec.config.control_dt))
        if tick<self.last_tick or (self.last_tick>=0 and tick>self.last_tick+1):raise ValueError('explicit reset/reorder and every observation tick required')
        if tick!=self.last_tick:
            self.frames.append(public_frame(public,self.spec.config.robot,self.policy.response_spec)[0]);self.frames=self.frames[-self.policy.response_spec.history_len:];self.last_tick=tick
        self.interceptor.override=None;base_action=self.base.act(public);raw=canonical_proto(self.interceptor.raw)
        state=decoder_state(self.base,public);source,distance,index=self.source_bank.query(public,state['phase'],[self.mode]);self.last_query_distance=float(distance[0])
        target=features(np.stack(self.frames),[len(self.frames)-1],self.spec.config.robot,self.spec.alias,[0,0,0],self.policy.response_spec)
        return dict(target_features=target,source_features=source['features'],source_commands=source['commands'],base_proto=np.repeat(raw,self.policy.response_spec.horizon,axis=1)/BOUNDS,
            decoder=state,current=dict(
                position=np.asarray(public['tcp_position']).copy(),
                rotation=np.asarray(public['tcp_rotation']).copy(),
                twist=np.asarray(public['tcp_twist']).copy(),
                wrench=np.asarray(public['measured_wrench']).copy(),
                goal_position=np.asarray(public['goal_position']).copy(),
                goal_rotation=np.asarray(public['goal_rotation']).copy(),
                controller_feedback=np.asarray(
                    public['controller_feedback']
                ).copy(),
            ),
            source_current=dict(position=source['position'],rotation=source['rotation'],wrench=source['wrench']),source_query_index=index,source_query_distance=distance),base_action

    def act(self,public):
        batch,base_action=self.prepare(public)
        if self.last_query_distance>self.maximum_query_distance:
            self.fallback_controls+=1;return base_action
        state={k:torch.as_tensor(v,device=self.device) for k,v in batch['decoder'].items()};state={k:v.float() if v.is_floating_point() else v for k,v in state.items()}
        with torch.inference_mode():
            values=[torch.as_tensor(batch[k],device=self.device,dtype=torch.float32) for k in ('target_features','source_features','source_commands','base_proto')]
            normalized=self.policy(*values,state['phase'],decoder_context(state,self.feedback));self.irreversible_network_calls+=1
            self.interceptor.override=(normalized[:,0:1]*normalized.new_tensor(BOUNDS)).cpu().numpy()
        return self.base.act(public)


class PublicTaskDominantDenseTransportActor(PublicDenseTransportActor):
    """Use transport only when it dominates its paired task-only action.

    Every comparison uses the current public history, the frozen public
    response model, and the controller's own public task reference.  The
    paired task-only checkpoint is therefore an executable control action,
    rather than only an offline loss reference.
    """

    VERSION = 'public-T3-transport-over-task-only-dominance-v1'
    BLENDS = (1., .5, .25, 0.)
    TASK_TOLERANCE = 1e-7
    PROGRESS_TOLERANCE = 1e-9

    def __init__(self, *args, task_only_policy, response, **kwargs):
        super().__init__(*args, **kwargs)
        if task_only_policy.response_spec != self.policy.response_spec:
            raise ValueError('paired task-only policy response chart differs')
        if response.spec != self.policy.response_spec:
            raise ValueError('task-dominance response chart differs')
        self.task_only_policy = task_only_policy.to(self.device).eval()
        self.response = response.to(self.device).eval()
        for parameter in self.response.parameters():
            parameter.requires_grad_(False)
        self.task_dominance_blend_counts = np.zeros(len(self.BLENDS), np.int64)
        self.transport_admission_controls = 0
        self.task_only_fallback_controls = 0
        self.last_task_dominance_blend = 0.
        self.last_transport_task_proxy = None
        self.last_task_only_task_proxy = None

    def reset(self):
        super().reset()
        self.task_dominance_blend_counts = np.zeros(len(self.BLENDS), np.int64)
        self.transport_admission_controls = 0
        self.task_only_fallback_controls = 0
        self.last_task_dominance_blend = 0.
        self.last_transport_task_proxy = None
        self.last_task_only_task_proxy = None

    def get_state(self):
        state = super().get_state()
        state['task_dominance'] = deepcopy({
            'version': self.VERSION,
            'blend_counts': self.task_dominance_blend_counts,
            'transport_admission_controls': self.transport_admission_controls,
            'task_only_fallback_controls': self.task_only_fallback_controls,
            'last_blend': self.last_task_dominance_blend,
            'last_transport_task_proxy': self.last_transport_task_proxy,
            'last_task_only_task_proxy': self.last_task_only_task_proxy,
        })
        return state

    def restore(self, state):
        value = dict(state)
        dominance = value.pop('task_dominance')
        if dominance['version'] != self.VERSION:
            raise ValueError('task-dominance execution version differs')
        super().restore(value)
        self.task_dominance_blend_counts = np.asarray(
            dominance['blend_counts'], dtype=np.int64,
        ).copy()
        self.transport_admission_controls = int(
            dominance['transport_admission_controls']
        )
        self.task_only_fallback_controls = int(
            dominance['task_only_fallback_controls']
        )
        self.last_task_dominance_blend = float(dominance['last_blend'])
        self.last_transport_task_proxy = dominance['last_transport_task_proxy']
        self.last_task_only_task_proxy = dominance['last_task_only_task_proxy']

    def _task_reference(self):
        reference = teacher_reference(
            self.base.observer.memory, goal_z=self.spec.config.goal[2],
        )
        cosine, sine = np.cos(reference[:, 3]), np.sin(reference[:, 3])
        rotation = np.zeros((len(reference), 3, 3), dtype=np.float32)
        rotation[:, 0, 0] = cosine
        rotation[:, 0, 1] = sine
        rotation[:, 1, 0] = sine
        rotation[:, 1, 1] = -cosine
        rotation[:, 2, 2] = -1.
        return np.concatenate(
            (reference[:, :3], rotation.reshape(-1, 9), reference[:, 4:5]),
            axis=1,
        ).astype(np.float32)

    def _progress(self, command, batch, reference):
        position_error = reference[:, :3] - batch['current']['position']
        desired_rotation = reference[:, 3:12].reshape(-1, 3, 3)
        rotation_error = np.stack([
            orientation_error(np, current, desired)
            for current, desired in zip(
                batch['current']['rotation'], desired_rotation,
            )
        ])
        linear = (command[:, 0, :3] * torch.as_tensor(
            position_error, device=command.device, dtype=command.dtype,
        )).sum(-1)
        angular = (command[:, 0, 3:6] * torch.as_tensor(
            rotation_error, device=command.device, dtype=command.dtype,
        )).sum(-1)
        return linear, angular

    def act(self, public):
        batch, base_action = self.prepare(public)
        if self.last_query_distance > self.maximum_query_distance:
            self.fallback_controls += 1
            return base_action
        state = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in batch['decoder'].items()
        }
        state = {
            key: value.float() if value.is_floating_point() else value
            for key, value in state.items()
        }
        with torch.inference_mode():
            values = [
                torch.as_tensor(batch[key], device=self.device, dtype=torch.float32)
                for key in (
                    'target_features', 'source_features',
                    'source_commands', 'base_proto',
                )
            ]
            context = decoder_context(state, self.feedback)
            transport = self.policy(*values, state['phase'], context)
            task_only = self.task_only_policy(*values, state['phase'], context)
            reference = self._task_reference()
            reference_tensor = torch.as_tensor(
                reference, device=self.device, dtype=torch.float32,
            )
            current = {
                key: torch.as_tensor(value, device=self.device, dtype=torch.float32)
                for key, value in batch['current'].items()
            }
            task_command = decode_pose_transport(
                task_only * task_only.new_tensor(BOUNDS), state, self.feedback,
            )
            task_normalized = task_command / task_command.new_tensor(BOUNDS)
            task_prediction = self.response(values[0], task_normalized)
            task_score = task_proxy(
                task_prediction, current, reference_tensor,
            ).mean()
            task_linear, task_angular = self._progress(
                task_command, batch, reference,
            )
            selected = task_only
            selected_index = len(self.BLENDS) - 1
            transport_score = task_score
            for index, fraction in enumerate(self.BLENDS[:-1]):
                trial = task_only + fraction * (transport - task_only)
                command = decode_pose_transport(
                    trial * trial.new_tensor(BOUNDS), state, self.feedback,
                )
                normalized = command / command.new_tensor(BOUNDS)
                prediction = self.response(values[0], normalized)
                score = task_proxy(prediction, current, reference_tensor).mean()
                linear, angular = self._progress(command, batch, reference)
                no_worse = bool(
                    score <= task_score + self.TASK_TOLERANCE
                    and linear >= task_linear - self.PROGRESS_TOLERANCE
                    and angular >= task_angular - self.PROGRESS_TOLERANCE
                )
                strict = bool(
                    score < task_score - self.TASK_TOLERANCE
                    or linear > task_linear + self.PROGRESS_TOLERANCE
                    or angular > task_angular + self.PROGRESS_TOLERANCE
                )
                if no_worse and strict:
                    selected = trial
                    selected_index = index
                    transport_score = score
                    break
            self.irreversible_network_calls += 2
            self.interceptor.override = (
                selected[:, 0:1] * selected.new_tensor(BOUNDS)
            ).cpu().numpy()
        blend = self.BLENDS[selected_index]
        self.task_dominance_blend_counts[selected_index] += 1
        self.last_task_dominance_blend = float(blend)
        self.last_transport_task_proxy = float(transport_score.cpu())
        self.last_task_only_task_proxy = float(task_score.cpu())
        if blend > 0.:
            self.transport_admission_controls += 1
        else:
            self.task_only_fallback_controls += 1
        return self.base.act(public)


class PublicFullTaskDominantDenseTransportActor(
    PublicTaskDominantDenseTransportActor
):
    """One batched full-transport comparison against paired task-only."""

    VERSION = 'public-T3-full-transport-over-task-only-dominance-v2'
    BLENDS = (1., 0.)

    def act(self, public):
        batch, base_action = self.prepare(public)
        if self.last_query_distance > self.maximum_query_distance:
            self.fallback_controls += 1
            return base_action
        state = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in batch['decoder'].items()
        }
        state = {
            key: value.float() if value.is_floating_point() else value
            for key, value in state.items()
        }
        with torch.inference_mode():
            values = [
                torch.as_tensor(batch[key], device=self.device, dtype=torch.float32)
                for key in (
                    'target_features', 'source_features',
                    'source_commands', 'base_proto',
                )
            ]
            context = decoder_context(state, self.feedback)
            transport = self.policy(*values, state['phase'], context)
            task_only = self.task_only_policy(*values, state['phase'], context)
            pair = torch.cat((transport, task_only), dim=0)
            paired_state = {
                key: value.expand(2, *value.shape[1:])
                for key, value in state.items()
            }
            commands = decode_pose_transport(
                pair * pair.new_tensor(BOUNDS), paired_state, self.feedback,
            )
            normalized = commands / commands.new_tensor(BOUNDS)
            features_pair = values[0].expand(2, *values[0].shape[1:])
            prediction = self.response(features_pair, normalized)
            reference = self._task_reference()
            reference_pair = torch.as_tensor(
                reference, device=self.device, dtype=torch.float32,
            ).expand(2, -1)
            current_pair = {
                key: torch.as_tensor(
                    value, device=self.device, dtype=torch.float32,
                ).expand(2, *np.asarray(value).shape[1:])
                for key, value in batch['current'].items()
            }
            scores = task_proxy(
                prediction, current_pair, reference_pair,
            ).mean(0)
            linear, angular = self._progress(commands, batch, reference)
            no_worse = bool(
                scores[0] <= scores[1] + self.TASK_TOLERANCE
                and linear[0] >= linear[1] - self.PROGRESS_TOLERANCE
                and angular[0] >= angular[1] - self.PROGRESS_TOLERANCE
            )
            strict = bool(
                scores[0] < scores[1] - self.TASK_TOLERANCE
                or linear[0] > linear[1] + self.PROGRESS_TOLERANCE
                or angular[0] > angular[1] + self.PROGRESS_TOLERANCE
            )
            selected_index = 0 if no_worse and strict else 1
            selected = pair[selected_index:selected_index + 1]
            self.irreversible_network_calls += 2
            self.interceptor.override = (
                selected[:, 0:1] * selected.new_tensor(BOUNDS)
            ).cpu().numpy()
        blend = self.BLENDS[selected_index]
        self.task_dominance_blend_counts[selected_index] += 1
        self.last_task_dominance_blend = float(blend)
        self.last_transport_task_proxy = float(scores[0].cpu())
        self.last_task_only_task_proxy = float(scores[1].cpu())
        if blend > 0.:
            self.transport_admission_controls += 1
        else:
            self.task_only_fallback_controls += 1
        return self.base.act(public)


class PublicAdaptiveStepTaskDominantDenseTransportActor(
    PublicTaskDominantDenseTransportActor
):
    """Line-search along transport and require measurable task progress.

    The learned transport output defines a direction away from the paired
    task-only action.  A fixed full step made that direction statistically
    favorable but too small to change completion time materially.  This actor
    evaluates a frozen public trust-region grid and takes the admissible step
    with the largest normalized finite-time progress gain.  Task-only remains
    an explicit candidate and owns every rejected control.
    """

    VERSION = 'public-T3-adaptive-step-transport-dominance-v3'
    BLENDS = (4., 2., 1., .5, .25, 0.)
    LINEAR_PROGRESS_SCALE = 1e-5
    ANGULAR_PROGRESS_SCALE = 1e-4
    MINIMUM_NORMALIZED_PROGRESS_GAIN = 1e-4

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_linear_progress_gain = 0.
        self.last_angular_progress_gain = 0.
        self.last_normalized_progress_gain = 0.

    def reset(self):
        super().reset()
        self.last_linear_progress_gain = 0.
        self.last_angular_progress_gain = 0.
        self.last_normalized_progress_gain = 0.

    def get_state(self):
        state = super().get_state()
        state['adaptive_step'] = deepcopy({
            'version': self.VERSION,
            'last_linear_progress_gain': self.last_linear_progress_gain,
            'last_angular_progress_gain': self.last_angular_progress_gain,
            'last_normalized_progress_gain': self.last_normalized_progress_gain,
        })
        return state

    def restore(self, state):
        value = dict(state)
        adaptive = value.pop('adaptive_step')
        if adaptive['version'] != self.VERSION:
            raise ValueError('adaptive-step execution version differs')
        super().restore(value)
        self.last_linear_progress_gain = float(
            adaptive['last_linear_progress_gain']
        )
        self.last_angular_progress_gain = float(
            adaptive['last_angular_progress_gain']
        )
        self.last_normalized_progress_gain = float(
            adaptive['last_normalized_progress_gain']
        )

    def _candidate_admission_mask(self, admissible, phase):
        """Apply execution-owned authority limits after public model gates."""
        del phase
        return admissible

    def act(self, public):
        batch, base_action = self.prepare(public)
        if self.last_query_distance > self.maximum_query_distance:
            self.fallback_controls += 1
            return base_action
        state = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in batch['decoder'].items()
        }
        state = {
            key: value.float() if value.is_floating_point() else value
            for key, value in state.items()
        }
        with torch.inference_mode():
            values = [
                torch.as_tensor(batch[key], device=self.device, dtype=torch.float32)
                for key in (
                    'target_features', 'source_features',
                    'source_commands', 'base_proto',
                )
            ]
            context = decoder_context(state, self.feedback)
            transport = self.policy(*values, state['phase'], context)
            task_only = self.task_only_policy(*values, state['phase'], context)
            fractions = transport.new_tensor(self.BLENDS).reshape(-1, 1, 1)
            trials = task_only + fractions * (transport - task_only)
            trials = torch.cat((
                trials[..., :12].clamp(-1., 1.),
                trials[..., 12:].clamp(0., 1.),
            ), dim=-1)
            count = len(self.BLENDS)
            paired_state = {
                key: value.expand(count, *value.shape[1:])
                for key, value in state.items()
            }
            commands = decode_pose_transport(
                trials * trials.new_tensor(BOUNDS), paired_state, self.feedback,
            )
            normalized = commands / commands.new_tensor(BOUNDS)
            prediction = self.response(
                values[0].expand(count, *values[0].shape[1:]), normalized,
            )
            reference = self._task_reference()
            reference_pair = torch.as_tensor(
                reference, device=self.device, dtype=torch.float32,
            ).expand(count, -1)
            current_pair = {
                key: torch.as_tensor(
                    value, device=self.device, dtype=torch.float32,
                ).expand(count, *np.asarray(value).shape[1:])
                for key, value in batch['current'].items()
            }
            scores = task_proxy(
                prediction, current_pair, reference_pair,
            ).mean(0)
            linear, angular = self._progress(commands, batch, reference)
            task_index = count - 1
            linear_gain = linear - linear[task_index]
            angular_gain = angular - angular[task_index]
            normalized_gain = (
                linear_gain / self.LINEAR_PROGRESS_SCALE
                + angular_gain / self.ANGULAR_PROGRESS_SCALE
            )
            admissible = (
                (scores <= scores[task_index] + self.TASK_TOLERANCE)
                & (linear >= linear[task_index] - self.PROGRESS_TOLERANCE)
                & (angular >= angular[task_index] - self.PROGRESS_TOLERANCE)
                & (
                    normalized_gain
                    >= self.MINIMUM_NORMALIZED_PROGRESS_GAIN
                )
            )
            admissible = self._candidate_admission_mask(
                admissible, int(state['phase'].reshape(-1)[0].item()),
            )
            admissible[task_index] = False
            ranked = torch.where(
                admissible, normalized_gain,
                normalized_gain.new_full(normalized_gain.shape, -torch.inf),
            )
            selected_index = (
                int(ranked.argmax().item())
                if bool(admissible.any()) else task_index
            )
            selected = trials[selected_index:selected_index + 1]
            self.irreversible_network_calls += 2
            self.interceptor.override = (
                selected[:, 0:1] * selected.new_tensor(BOUNDS)
            ).cpu().numpy()
        blend = self.BLENDS[selected_index]
        self.task_dominance_blend_counts[selected_index] += 1
        self.last_task_dominance_blend = float(blend)
        self.last_transport_task_proxy = float(scores[selected_index].cpu())
        self.last_task_only_task_proxy = float(scores[task_index].cpu())
        self.last_linear_progress_gain = float(linear_gain[selected_index].cpu())
        self.last_angular_progress_gain = float(angular_gain[selected_index].cpu())
        self.last_normalized_progress_gain = float(
            normalized_gain[selected_index].cpu()
        )
        if blend > 0.:
            self.transport_admission_controls += 1
        else:
            self.task_only_fallback_controls += 1
        return self.base.act(public)


class PublicTransientPhaseAdaptiveStepDenseTransportActor(
    PublicAdaptiveStepTaskDominantDenseTransportActor
):
    """Give terminal convergence to the paired task-only controller.

    Transport retains adaptive, publicly gated authority during contact,
    recovery and keyway alignment (phases zero through two).  Once the shared
    public controller reaches terminal insertion phase three, the paired
    task-only action owns every remaining control.
    """

    VERSION = 'public-T3-transient-phase-adaptive-step-dominance-v12'

    def _candidate_admission_mask(self, admissible, phase):
        if phase not in (0, 1, 2, 3):
            raise ValueError('T3 controller phase must be zero through three')
        if phase == 3:
            return torch.zeros_like(admissible, dtype=torch.bool)
        return admissible


class PublicPhaseScheduledAdaptiveStepDenseTransportActor(
    PublicAdaptiveStepTaskDominantDenseTransportActor
):
    """Schedule bounded transport authority over the public task phases.

    The keyway-alignment phase has the densest useful transport support and
    receives one additional bounded trial.  Terminal mode-zero insertion keeps
    the previously qualified grid, while terminal mode one is capped because
    its phase-three transport support is sparse.
    """

    VERSION = 'public-T3-phase-scheduled-adaptive-step-dominance-v13'
    BLENDS = (6., 4., 2., 1., .5, .25, 0.)

    def _candidate_admission_mask(self, admissible, phase):
        if phase not in (0, 1, 2, 3):
            raise ValueError('T3 controller phase must be zero through three')
        limited = admissible.clone()
        if phase != 2:
            limited[0] = False
        if phase == 3 and self.mode == 1:
            limited[:2] = False
        return limited


class PublicExpandedAdaptiveStepTaskDominantDenseTransportActor(
    PublicAdaptiveStepTaskDominantDenseTransportActor
):
    """Adaptive transport with one larger, still publicly gated trust step."""

    VERSION = 'public-T3-expanded-adaptive-step-transport-dominance-v4'
    BLENDS = (8., 4., 2., 1., .5, .25, 0.)


class PublicEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor(
    PublicAdaptiveStepTaskDominantDenseTransportActor
):
    """Admit transport only when every public response member predicts gain."""

    VERSION = 'public-T3-ensemble-robust-adaptive-step-dominance-v5'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_worst_member_task_proxy_gain = 0.

    def reset(self):
        super().reset()
        self.last_worst_member_task_proxy_gain = 0.

    def get_state(self):
        state = super().get_state()
        state['ensemble_robust'] = {
            'version': self.VERSION,
            'last_worst_member_task_proxy_gain': (
                self.last_worst_member_task_proxy_gain
            ),
        }
        return state

    def restore(self, state):
        value = dict(state)
        robust = value.pop('ensemble_robust')
        if robust['version'] != self.VERSION:
            raise ValueError('ensemble-robust execution version differs')
        super().restore(value)
        self.last_worst_member_task_proxy_gain = float(
            robust['last_worst_member_task_proxy_gain']
        )

    def _blend_fractions(self, transport, task_only):
        return transport.new_tensor(self.BLENDS).reshape(-1, 1, 1)

    def _clear_execution_commitment(self):
        """Hook for actors that execute an admitted finite-horizon prefix."""

    def _committed_action(self, batch, state, values, context):
        return None

    def _record_selected_sequence(self, selected, blend, state):
        """Hook called after one freshly evaluated candidate is selected."""

    def _candidate_member_scores(
        self, prediction, current_pair, reference_pair, batch,
    ):
        """Return lower-is-better member scores for every trial candidate."""
        return task_proxy(prediction, current_pair, reference_pair)

    def _candidate_statistics(
        self, member_scores, linear, angular, task_index,
    ):
        return self._robust_candidate_statistics(
            member_scores, linear, angular, task_index,
        )

    def _candidate_admission_mask(self, admissible, phase):
        """Apply execution-owned public support limits after robust scoring."""
        del phase
        return admissible

    @classmethod
    def _robust_candidate_statistics(
        cls, member_scores, linear, angular, task_index,
    ):
        """Return the frozen ensemble lower bound and physical admission mask."""
        linear_gain = linear - linear[task_index]
        angular_gain = angular - angular[task_index]
        normalized_gain = (
            linear_gain / cls.LINEAR_PROGRESS_SCALE
            + angular_gain / cls.ANGULAR_PROGRESS_SCALE
        )
        member_gain = (
            member_scores[:, task_index:task_index + 1] - member_scores
        )
        worst_member_gain = member_gain.amin(0)
        admissible = (
            (worst_member_gain >= cls.TASK_TOLERANCE)
            & (linear >= linear[task_index] - cls.PROGRESS_TOLERANCE)
            & (angular >= angular[task_index] - cls.PROGRESS_TOLERANCE)
            & (normalized_gain >= cls.MINIMUM_NORMALIZED_PROGRESS_GAIN)
        )
        admissible = admissible.clone()
        admissible[task_index] = False
        return linear_gain, angular_gain, normalized_gain, worst_member_gain, admissible

    def act(self, public):
        batch, base_action = self.prepare(public)
        if self.last_query_distance > self.maximum_query_distance:
            self._clear_execution_commitment()
            self.fallback_controls += 1
            return base_action
        state = {
            key: torch.as_tensor(value, device=self.device)
            for key, value in batch['decoder'].items()
        }
        state = {
            key: value.float() if value.is_floating_point() else value
            for key, value in state.items()
        }
        with torch.inference_mode():
            values = [
                torch.as_tensor(batch[key], device=self.device, dtype=torch.float32)
                for key in (
                    'target_features', 'source_features',
                    'source_commands', 'base_proto',
                )
            ]
            context = decoder_context(state, self.feedback)
            committed = self._committed_action(
                batch, state, values, context,
            )
            if committed is not None:
                self.interceptor.override = (
                    committed * committed.new_tensor(BOUNDS)
                ).cpu().numpy()
                return self.base.act(public)
            transport = self.policy(*values, state['phase'], context)
            task_only = self.task_only_policy(*values, state['phase'], context)
            fractions = self._blend_fractions(transport, task_only)
            trials = task_only + fractions * (transport - task_only)
            trials = torch.cat((
                trials[..., :12].clamp(-1., 1.),
                trials[..., 12:].clamp(0., 1.),
            ), dim=-1)
            count = len(self.BLENDS)
            paired_state = {
                key: value.expand(count, *value.shape[1:])
                for key, value in state.items()
            }
            commands = decode_pose_transport(
                trials * trials.new_tensor(BOUNDS), paired_state, self.feedback,
            )
            normalized = commands / commands.new_tensor(BOUNDS)
            prediction = self.response(
                values[0].expand(count, *values[0].shape[1:]), normalized,
            )
            reference = self._task_reference()
            reference_pair = torch.as_tensor(
                reference, device=self.device, dtype=torch.float32,
            ).expand(count, -1)
            current_pair = {
                key: torch.as_tensor(
                    value, device=self.device, dtype=torch.float32,
                ).expand(count, *np.asarray(value).shape[1:])
                for key, value in batch['current'].items()
            }
            member_scores = self._candidate_member_scores(
                prediction, current_pair, reference_pair, batch,
            )
            scores = member_scores.mean(0)
            linear, angular = self._progress(commands, batch, reference)
            task_index = count - 1
            (
                linear_gain, angular_gain, normalized_gain,
                worst_member_gain, admissible,
            ) = self._candidate_statistics(
                member_scores, linear, angular, task_index,
            )
            admissible = self._candidate_admission_mask(
                admissible, int(state['phase'].reshape(-1)[0].item()),
            )
            ranked = torch.where(
                admissible, normalized_gain,
                normalized_gain.new_full(normalized_gain.shape, -torch.inf),
            )
            selected_index = (
                int(ranked.argmax().item())
                if bool(admissible.any()) else task_index
            )
            selected = trials[selected_index:selected_index + 1]
            self.irreversible_network_calls += 2
            self.interceptor.override = (
                selected[:, 0:1] * selected.new_tensor(BOUNDS)
            ).cpu().numpy()
        blend = float(fractions[selected_index].reshape(-1)[0].cpu())
        self.task_dominance_blend_counts[selected_index] += 1
        self.last_task_dominance_blend = float(blend)
        self.last_transport_task_proxy = float(scores[selected_index].cpu())
        self.last_task_only_task_proxy = float(scores[task_index].cpu())
        self.last_linear_progress_gain = float(linear_gain[selected_index].cpu())
        self.last_angular_progress_gain = float(angular_gain[selected_index].cpu())
        self.last_normalized_progress_gain = float(
            normalized_gain[selected_index].cpu()
        )
        self.last_worst_member_task_proxy_gain = float(
            worst_member_gain[selected_index].cpu()
        )
        if blend > 0.:
            self.transport_admission_controls += 1
        else:
            self.task_only_fallback_controls += 1
        self._record_selected_sequence(selected, blend, state)
        return self.base.act(public)


class PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor(
    PublicEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor
):
    """Ensemble-consensus admission over the previously frozen expanded grid."""

    VERSION = 'public-T3-expanded-ensemble-robust-adaptive-step-dominance-v6'
    BLENDS = (8., 4., 2., 1., .5, .25, 0.)


class PublicCompletionValueEnsembleRobustDenseTransportActor(
    PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor
):
    """Admit transport only under calibrated remaining-completion-time gain."""

    VERSION = 'public-T3-completion-value-ensemble-robust-dominance-v11'

    def __init__(
        self, *args, completion_value, minimum_admission_gain_controls, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if completion_value.spec.maximum_remaining_controls < 1:
            raise ValueError('positive completion-value normalization required')
        margin = float(minimum_admission_gain_controls)
        if not np.isfinite(margin) or margin <= 0.:
            raise ValueError('positive finite completion-value admission margin required')
        self.completion_value = completion_value.to(self.device).eval()
        for parameter in self.completion_value.parameters():
            parameter.requires_grad_(False)
        self.minimum_admission_gain_controls = margin
        self.minimum_admission_gain_normalized = (
            margin / self.completion_value.spec.maximum_remaining_controls
        )
        self.last_worst_member_completion_gain_controls = 0.
        self._candidate_completion_gains = None

    def reset(self):
        super().reset()
        self.last_worst_member_completion_gain_controls = 0.
        self._candidate_completion_gains = None

    def get_state(self):
        state = super().get_state()
        state['completion_value_admission'] = {
            'version': self.VERSION,
            'minimum_admission_gain_controls': self.minimum_admission_gain_controls,
            'last_worst_member_completion_gain_controls': (
                self.last_worst_member_completion_gain_controls
            ),
        }
        return state

    def restore(self, state):
        value = dict(state)
        admission = value.pop('completion_value_admission')
        if admission['version'] != self.VERSION:
            raise ValueError('completion-value execution version differs')
        if not np.isclose(
            admission['minimum_admission_gain_controls'],
            self.minimum_admission_gain_controls,
            rtol=0., atol=1e-12,
        ):
            raise ValueError('completion-value admission margin differs')
        super().restore(value)
        self.last_worst_member_completion_gain_controls = float(
            admission['last_worst_member_completion_gain_controls']
        )
        self._candidate_completion_gains = None

    def _candidate_member_scores(
        self, prediction, current_pair, reference_pair, batch,
    ):
        del reference_pair, batch
        values = response_endpoint_value(
            self.completion_value, prediction,
            current={
                name: current_pair[name]
                for name in (
                    'position', 'rotation', 'twist', 'wrench',
                    'goal_position', 'goal_rotation', 'controller_feedback',
                )
            },
            mode=self.mode,
        )
        return values.flatten(0, 1)

    def _candidate_statistics(
        self, member_scores, linear, angular, task_index,
    ):
        values = super()._candidate_statistics(
            member_scores, linear, angular, task_index,
        )
        linear_gain, angular_gain, normalized_gain, worst_gain, admissible = values
        admissible = admissible & (
            worst_gain >= self.minimum_admission_gain_normalized
        )
        admissible = admissible.clone()
        admissible[task_index] = False
        self._candidate_completion_gains = (
            worst_gain * self.completion_value.spec.maximum_remaining_controls
        ).detach().cpu()
        return (
            linear_gain, angular_gain, normalized_gain, worst_gain, admissible,
        )

    def act(self, public):
        self._candidate_completion_gains = None
        result = super().act(public)
        if self._candidate_completion_gains is not None:
            index = self.BLENDS.index(self.last_task_dominance_blend)
            self.last_worst_member_completion_gain_controls = float(
                self._candidate_completion_gains[index]
            )
        return result


class PublicModeConditionedCompletionValueDenseTransportActor(
    PublicCompletionValueEnsembleRobustDenseTransportActor
):
    """Mode-zero branch of the frozen public mode-conditioned controller."""

    VERSION = 'public-T3-mode-conditioned-finite-time-dominance-v14'


class PublicModeConditionedAdaptiveStepDenseTransportActor(
    PublicAdaptiveStepTaskDominantDenseTransportActor
):
    """Mode-one branch of the frozen public mode-conditioned controller."""

    VERSION = 'public-T3-mode-conditioned-finite-time-dominance-v14'


class PublicModeSpecificRecordedCompletionDenseTransportActor(
    PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor
):
    """Mode-specific recorded-completion policy with robust public admission."""

    VERSION = 'public-T3-mode-specific-recorded-completion-dominance-v15'


class PublicModeSpecificSupportedPhaseConvexDenseTransportActor(
    PublicModeSpecificRecordedCompletionDenseTransportActor
):
    """Use mode-one transport only on its bounded public support domain."""

    VERSION = 'public-T3-mode-specific-supported-phase-convex-dominance-v16'
    MODE_ONE_MAX_BLEND = 1.

    def _candidate_admission_mask(self, admissible, phase):
        if phase not in (0, 1, 2, 3):
            raise ValueError('T3 controller phase must be zero through three')
        if self.mode == 0:
            return admissible
        limited = admissible.clone()
        if phase in (0, 3):
            return torch.zeros_like(limited, dtype=torch.bool)
        for index, blend in enumerate(self.BLENDS):
            if blend > self.MODE_ONE_MAX_BLEND:
                limited[index] = False
        return limited


class PublicModeSpecificSupportedPhaseCappedExpansionDenseTransportActor(
    PublicModeSpecificSupportedPhaseConvexDenseTransportActor
):
    """One bounded mode-one expansion inside the supported public phases."""

    VERSION = 'public-T3-mode-specific-supported-phase-capped-expansion-dominance-v17'
    MODE_ONE_MAX_BLEND = 2.


class PublicModeSpecificCalibratedAuthorityDenseTransportActor(
    PublicModeSpecificSupportedPhaseConvexDenseTransportActor
):
    """Use development-calibrated mode-one authority on its support phases."""

    VERSION = 'public-T3-mode-specific-calibrated-authority-dominance-v18'

    def __init__(self, *args, mode_one_max_blend, **kwargs):
        value = float(mode_one_max_blend)
        if value not in (0., 1., 2., 8.):
            raise ValueError('calibrated mode-one blend must be 0, 1, 2, or 8')
        self.MODE_ONE_MAX_BLEND = value
        super().__init__(*args, **kwargs)


class PublicCappedExpansionEnsembleRobustDenseTransportActor(
    PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor
):
    """One bounded 1.5x trust-region expansion beyond the admitted v6 grid."""

    VERSION = 'public-T3-capped-expansion-ensemble-robust-dominance-v8'
    BLENDS = (12., 8., 4., 2., 1., .5, .25, 0.)


class PublicHorizonConsistentEnsembleRobustDenseTransportActor(
    PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor
):
    """Execute an admitted response horizon while public progress stays non-worse."""

    VERSION = 'public-T3-horizon-consistent-ensemble-robust-dominance-v9'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.committed_sequence = None
        self.committed_index = 0
        self.committed_phase = None
        self.committed_blend = 0.
        self.horizon_commitment_remaining = 0
        self.horizon_commitment_controls = 0
        self.horizon_commitment_aborts = 0

    def reset(self):
        super().reset()
        self.committed_sequence = None
        self.committed_index = 0
        self.committed_phase = None
        self.committed_blend = 0.
        self.horizon_commitment_remaining = 0
        self.horizon_commitment_controls = 0
        self.horizon_commitment_aborts = 0

    def get_state(self):
        state = super().get_state()
        state['horizon_consistent'] = deepcopy({
            'version': self.VERSION,
            'sequence': self.committed_sequence,
            'index': self.committed_index,
            'phase': self.committed_phase,
            'blend': self.committed_blend,
            'controls': self.horizon_commitment_controls,
            'aborts': self.horizon_commitment_aborts,
        })
        return state

    def restore(self, state):
        value = dict(state)
        commitment = value.pop('horizon_consistent')
        if commitment['version'] != self.VERSION:
            raise ValueError('horizon-consistent execution version differs')
        super().restore(value)
        sequence = commitment['sequence']
        self.committed_sequence = (
            None if sequence is None else np.asarray(sequence, np.float32).copy()
        )
        self.committed_index = int(commitment['index'])
        self.committed_phase = commitment['phase']
        self.committed_blend = float(commitment['blend'])
        self.horizon_commitment_controls = int(commitment['controls'])
        self.horizon_commitment_aborts = int(commitment['aborts'])
        self.horizon_commitment_remaining = (
            0 if self.committed_sequence is None
            else len(self.committed_sequence) - self.committed_index
        )

    def _clear_execution_commitment(self):
        self.committed_sequence = None
        self.committed_index = 0
        self.committed_phase = None
        self.committed_blend = 0.
        self.horizon_commitment_remaining = 0

    def _record_selected_sequence(self, selected, blend, state):
        self._clear_execution_commitment()
        if blend <= 0. or selected.shape[1] <= 1:
            return
        self.committed_sequence = (
            selected[0, 1:].detach().cpu().numpy().astype(np.float32, copy=True)
        )
        self.committed_phase = int(state['phase'].reshape(-1)[0].item())
        self.committed_blend = float(blend)
        self.horizon_commitment_remaining = len(self.committed_sequence)

    def _committed_action(self, batch, state, values, context):
        if self.committed_sequence is None:
            return None
        phase = int(state['phase'].reshape(-1)[0].item())
        if phase != self.committed_phase:
            self.horizon_commitment_aborts += 1
            self._clear_execution_commitment()
            return None
        candidate = torch.as_tensor(
            self.committed_sequence[
                self.committed_index:self.committed_index + 1
            ], device=self.device, dtype=torch.float32,
        ).unsqueeze(0)
        task_only = self.task_only_policy(
            *values, state['phase'], context,
        )[:, :1]
        trials = torch.cat((candidate, task_only), dim=0)
        paired_state = {
            key: value.expand(2, *value.shape[1:])
            for key, value in state.items()
        }
        commands = decode_pose_transport(
            trials * trials.new_tensor(BOUNDS), paired_state, self.feedback,
        )
        reference = self._task_reference()
        linear, angular = self._progress(commands, batch, reference)
        linear_gain = linear[0] - linear[1]
        angular_gain = angular[0] - angular[1]
        normalized_gain = (
            linear_gain / self.LINEAR_PROGRESS_SCALE
            + angular_gain / self.ANGULAR_PROGRESS_SCALE
        )
        admissible = bool(
            linear[0] >= linear[1] - self.PROGRESS_TOLERANCE
            and angular[0] >= angular[1] - self.PROGRESS_TOLERANCE
            and normalized_gain >= self.MINIMUM_NORMALIZED_PROGRESS_GAIN
        )
        self.irreversible_network_calls += 1
        if not admissible:
            self.horizon_commitment_aborts += 1
            self._clear_execution_commitment()
            return None
        self.last_task_dominance_blend = self.committed_blend
        self.last_transport_task_proxy = None
        self.last_task_only_task_proxy = None
        self.last_linear_progress_gain = float(linear_gain.cpu())
        self.last_angular_progress_gain = float(angular_gain.cpu())
        self.last_normalized_progress_gain = float(normalized_gain.cpu())
        self.last_worst_member_task_proxy_gain = 0.
        blend_index = self.BLENDS.index(self.committed_blend)
        self.task_dominance_blend_counts[blend_index] += 1
        self.transport_admission_controls += 1
        self.horizon_commitment_controls += 1
        self.committed_index += 1
        self.horizon_commitment_remaining = (
            len(self.committed_sequence) - self.committed_index
        )
        selected = candidate
        if self.committed_index == len(self.committed_sequence):
            self._clear_execution_commitment()
        return selected


class PublicValidatedBridgeEnsembleRobustDenseTransportActor(
    PublicHorizonConsistentEnsembleRobustDenseTransportActor
):
    """Use a validation-scale robust margin and refresh after one public history."""

    VERSION = 'public-T3-validated-bridge-ensemble-robust-dominance-v10'
    TASK_TOLERANCE = 0.031

    def _record_selected_sequence(self, selected, blend, state):
        super()._record_selected_sequence(selected, blend, state)
        if self.committed_sequence is None:
            return
        continuation = max(0, self.policy.response_spec.history_len - 1)
        self.committed_sequence = self.committed_sequence[:continuation].copy()
        if len(self.committed_sequence) == 0:
            self._clear_execution_commitment()
            return
        self.horizon_commitment_remaining = len(self.committed_sequence)


class PublicBoundedEndpointEnsembleRobustDenseTransportActor(
    PublicExpandedEnsembleRobustAdaptiveStepTaskDominantDenseTransportActor
):
    """Add the exact feasible endpoint along the learned transport direction."""

    VERSION = 'public-T3-bounded-endpoint-ensemble-robust-dominance-v7'
    BLENDS = (float('inf'), 8., 4., 2., 1., .5, .25, 0.)

    def _blend_fractions(self, transport, task_only):
        direction = transport - task_only
        lower = direction.new_tensor((-1.,) * 12 + (0.,))
        upper = direction.new_ones(13)
        positive = direction > 0.
        negative = direction < 0.
        limits = torch.where(
            positive, (upper - task_only) / direction.clamp_min(1e-12),
            torch.where(
                negative, (lower - task_only) / direction.clamp_max(-1e-12),
                direction.new_full(direction.shape, torch.inf),
            ),
        )
        endpoint = limits.amin().clamp_min(0.)
        endpoint = torch.where(
            torch.isfinite(endpoint), endpoint, endpoint.new_zeros(()),
        )
        fixed = transport.new_tensor(self.BLENDS[1:])
        return torch.cat((endpoint.reshape(1), fixed)).reshape(-1, 1, 1)


class IndependentWorldDenseSolver:
    """Explicit per-world memory, stable under reordering and partial reset."""
    def __init__(self,actors):self.actors=list(actors)
    def act(self,public):return np.concatenate([actor.act({k:np.asarray(v)[i:i+1] for k,v in public.items()}) for i,actor in enumerate(self.actors)])
    def get_state(self):return [a.get_state() for a in self.actors]
    def restore(self,states):
        if len(states)!=len(self.actors):raise ValueError('actor state/world count differs')
        for actor,state in zip(self.actors,states):actor.restore(state)
    def reorder(self,indices):self.actors=[self.actors[int(i)] for i in indices]
    def reset(self,mask=None):
        mask=np.ones(len(self.actors),bool) if mask is None else np.asarray(mask,bool)
        if mask.shape!=(len(self.actors),):raise ValueError('one reset mask entry per world')
        for actor,selected in zip(self.actors,mask):
            if selected:actor.reset()
