"""Shared, public contact/withdrawal prefix for arm-skill comparisons.

The fixed task classifies recovery mode before first12mm withdrawal. This gate
uses only control-rate wrench/position and a conservative public upper bound
on first contact height. It supplies no evaluator contact bit or plant state.
"""
from copy import deepcopy
import numpy as np
from .mating_feedback import PublicLivenessMatingFeedbackPolicy, PublicMatingFeedbackPolicy
from genedynamics.tasks.transfer.controllers.cross_ebdm.policies.bias_feedback import PublicBiasFeedbackPolicy
from genedynamics.core.control.transfer import orientation_error
class PublicRecoveryGate:
    VERSION='public-contact-withdrawal-prefix-v1'
    def __init__(self):self.reset()
    def reset(self):
        self.tick=-1;self.contact_seen=False;self.maximum_precontact_z=-np.inf;self.enabled=False
    def get_state(self):return deepcopy(vars(self))
    def restore(self,state):self.__dict__.update(deepcopy(state))
    def observe(self,public):
        tick=int(np.rint(np.asarray(public['elapsed_seconds']).item()/.02))
        if tick==self.tick:return self.enabled
        if tick!=self.tick+1:raise ValueError('every public recovery tick required')
        z=float(np.asarray(public['tcp_position'])[0,2]);fz=float(np.asarray(public['measured_wrench'])[0,2])
        if not self.contact_seen:
            self.maximum_precontact_z=max(self.maximum_precontact_z,z)
            if tick>=1 and fz>.3001:self.contact_seen=True
        if self.contact_seen and z>self.maximum_precontact_z+.012+.0002:self.enabled=True
        self.tick=tick;return self.enabled
class PublicRecoveryExecution:
    VERSION='public-common-recovery-execution-v1'
    COMMON_T4_POLICY=PublicMatingFeedbackPolicy
    def __init__(self,candidate,spec,feedback,*,mode,seed):
        if spec.alias not in {'T3','T4'}:raise ValueError('explicit insertion/recovery binding')
        self.candidate,self.spec,self.mode,self.seed=candidate,spec,int(mode),int(seed)
        cls=self.COMMON_T4_POLICY if spec.alias=='T4' else PublicBiasFeedbackPolicy
        self.common=cls.load(spec,None,feedback=feedback,modes=[mode],seed=seed);self.gate=PublicRecoveryGate();self.common_controls=0;self.learned_controls=0
    def __getattr__(self,name):return getattr(self.candidate,name)
    def get_state(self):return deepcopy(dict(candidate=self.candidate.get_state(),common=self.common.get_state(),gate=self.gate.get_state(),common_controls=self.common_controls,learned_controls=self.learned_controls))
    def restore(self,state):
        self.candidate.restore(state['candidate']);self.common.restore(state['common']);self.gate.restore(state['gate']);self.common_controls=state['common_controls'];self.learned_controls=state['learned_controls']
    def reset(self):
        self.candidate.reset();self.common.reset(modes=[self.mode],seed=self.seed);self.gate.reset();self.common_controls=self.learned_controls=0
    def act(self,public):
        enabled=self.gate.observe(public);common=self.common.act(public);candidate=self.candidate.act(public)
        if enabled:self.learned_controls+=1;return candidate
        self.common_controls+=1;return common


class PublicLivenessRecoveryExecution(PublicRecoveryExecution):
    """Versioned common prefix using the same P5 release as every candidate."""

    VERSION='public-common-recovery-execution-safe-settled-v1'
    COMMON_T4_POLICY=PublicLivenessMatingFeedbackPolicy


class PublicCompletionDominantRecoveryExecution(PublicLivenessRecoveryExecution):
    """Admit a T4 candidate only while it preserves common completion cues."""

    VERSION = 'public-completion-dominant-recovery-execution-v1'
    BLENDS = (1., .5, .25, 0.)
    TERMINAL_MULTIPLIER = 2.
    PROGRESS_TOLERANCE = 1e-7

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.spec.alias != 'T4':
            raise ValueError('completion-dominant execution is a T4 binding')
        self._reset_completion_admission()

    def _reset_completion_admission(self):
        self.last_tick = -1
        self.cached = None
        self.last_common_action = np.zeros((1, 13), np.float32)
        self.last_candidate_action = np.zeros((1, 13), np.float32)
        self.last_selected_action = np.zeros((1, 13), np.float32)
        self.last_completion_blend = 0.
        self.last_completion_code = 0
        self.last_terminal_near = False
        self.completion_admission_controls = 0
        self.completion_common_fallback_controls = 0
        self.completion_terminal_controls = 0
        self.completion_blend_counts = np.zeros(len(self.BLENDS), np.int64)

    def reset(self):
        super().reset()
        self._reset_completion_admission()

    def get_state(self):
        state = super().get_state()
        state['completion_admission'] = deepcopy({
            'version': self.VERSION,
            'last_tick': self.last_tick,
            'cached': self.cached,
            'last_common_action': self.last_common_action,
            'last_candidate_action': self.last_candidate_action,
            'last_selected_action': self.last_selected_action,
            'last_completion_blend': self.last_completion_blend,
            'last_completion_code': self.last_completion_code,
            'last_terminal_near': self.last_terminal_near,
            'completion_admission_controls': self.completion_admission_controls,
            'completion_common_fallback_controls': self.completion_common_fallback_controls,
            'completion_terminal_controls': self.completion_terminal_controls,
            'completion_blend_counts': self.completion_blend_counts,
        })
        return state

    def restore(self, state):
        value = dict(state)
        completion = value.pop('completion_admission')
        if completion['version'] != self.VERSION:
            raise ValueError('completion admission binding differs')
        super().restore(value)
        self.last_tick = int(completion['last_tick'])
        self.cached = deepcopy(completion['cached'])
        for name in (
            'last_common_action', 'last_candidate_action',
            'last_selected_action', 'completion_blend_counts',
        ):
            setattr(self, name, np.asarray(completion[name]).copy())
        self.last_completion_blend = float(completion['last_completion_blend'])
        self.last_completion_code = int(completion['last_completion_code'])
        self.last_terminal_near = bool(completion['last_terminal_near'])
        self.completion_admission_controls = int(completion['completion_admission_controls'])
        self.completion_common_fallback_controls = int(completion['completion_common_fallback_controls'])
        self.completion_terminal_controls = int(completion['completion_terminal_controls'])

    def _goal_errors(self, public):
        position = np.asarray(public['tcp_position'], dtype=float)[0]
        goal = np.asarray(public['goal_position'], dtype=float)[0]
        rotation = np.asarray(public['tcp_rotation'], dtype=float)[0]
        goal_rotation = np.asarray(public['goal_rotation'], dtype=float)[0]
        return goal - position, orientation_error(np, rotation, goal_rotation)

    def _preserves_common_progress(self, blend, common, position_error, rotation_error):
        common_linear = float(np.dot(common[0, :3], position_error))
        blend_linear = float(np.dot(blend[0, :3], position_error))
        common_angular = float(np.dot(common[0, 3:6], rotation_error))
        blend_angular = float(np.dot(blend[0, 3:6], rotation_error))
        return (
            blend_linear + self.PROGRESS_TOLERANCE >= common_linear
            and blend_angular + self.PROGRESS_TOLERANCE >= common_angular
        )

    def act(self, public):
        tick = int(np.rint(
            np.asarray(public['elapsed_seconds']).item()
            / self.spec.config.control_dt
        ))
        if tick == self.last_tick:
            return self.cached.copy()
        if tick != self.last_tick + 1:
            raise ValueError('every public recovery tick required')
        enabled = self.gate.observe(public)
        common = np.asarray(self.common.act(public), dtype=np.float32)
        candidate = np.asarray(self.candidate.act(public), dtype=np.float32)
        self.last_common_action = common.copy()
        self.last_candidate_action = candidate.copy()
        self.last_terminal_near = False
        if not enabled:
            selected = common
            blend_value = 0.
            blend_index = len(self.BLENDS) - 1
            code = 0
            self.common_controls += 1
        else:
            position_error, rotation_error = self._goal_errors(public)
            terminal_near = bool(
                np.linalg.norm(position_error)
                <= self.TERMINAL_MULTIPLIER * self.spec.config.position_tolerance
                and np.linalg.norm(rotation_error)
                <= self.TERMINAL_MULTIPLIER * self.spec.config.orientation_tolerance
            )
            self.last_terminal_near = terminal_near
            if terminal_near:
                selected = common
                blend_value = 0.
                blend_index = len(self.BLENDS) - 1
                code = 2
                self.completion_terminal_controls += 1
                self.completion_common_fallback_controls += 1
            else:
                selected = common
                blend_value = 0.
                blend_index = len(self.BLENDS) - 1
                code = 3
                for index, fraction in enumerate(self.BLENDS):
                    trial = common + fraction * (candidate - common)
                    if self._preserves_common_progress(
                        trial, common, position_error, rotation_error,
                    ):
                        selected = trial.astype(np.float32)
                        blend_value = float(fraction)
                        blend_index = index
                        code = 1 if fraction > 0. else 3
                        break
                if blend_value == 0.:
                    self.completion_common_fallback_controls += 1
                else:
                    self.completion_admission_controls += 1
            self.learned_controls += 1
        self.last_completion_blend = blend_value
        self.last_completion_code = code
        self.completion_blend_counts[blend_index] += 1
        self.last_selected_action = np.asarray(selected, dtype=np.float32).copy()
        self.cached = self.last_selected_action.copy()
        self.last_tick = tick
        return self.cached.copy()
