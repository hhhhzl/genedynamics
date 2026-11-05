"""Explicit T4 FADA task adaptation with a public initial-body position chart.

Source Panda body motion is zero. Mobile future-body coordinates are expressed
relative to this episode's observed initial body position; matching an absolute
Panda base origin to a Go2 standing height would be an invalid correspondence.
This changes no source weights/statistics/native Planner-IDM/LoRA algorithm.
"""
from copy import deepcopy
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from .base import PublicFADATaskActor

BODY_CHART='public-body-position-relative-to-observed-reset-v1'


def mobile_checkpoint_binding(payload):
    result=deepcopy(payload);adapter=result['genedynamics_task_adapter'];adapter['source_task_name_before_mobile_binding']=adapter.get('task_name');adapter['task_name']=task_spec('T4',robot='go2_z1').name;adapter['portable_body_position_chart']=BODY_CHART;adapter['scope']='explicit T3Panda toT4Go2 arm-under-stance taskadaptation; nooriginalhumanoidquantitative reproduction';return result


def anchor_public_body(public,anchor):
    result={k:np.asarray(v).copy() for k,v in public.items()};result['body_position']-=anchor;return result


class PublicMobileFADATaskActor(PublicFADATaskActor):
    def __init__(self,spec,payload,*,mode,device='cpu'):
        if spec.alias!='T4' or spec.config.robot!='go2_z1' or payload['genedynamics_task_adapter'].get('portable_body_position_chart')!=BODY_CHART:raise ValueError('explicit actualT4 publicbody chart required')
        super().__init__(spec,payload,mode=mode,device=device)
    def reset(self):super().reset();self.body_anchor=None
    def get_state(self):return dict(super().get_state(),body_anchor=deepcopy(self.body_anchor),body_chart=BODY_CHART)
    def restore(self,state):
        if state['body_chart']!=BODY_CHART:raise ValueError('mobile bodychart differs')
        super().restore(state);self.body_anchor=deepcopy(state['body_anchor'])
    def act(self,public):
        if self.body_anchor is None:self.body_anchor=np.asarray(public['body_position']).copy()
        return super().act(anchor_public_body(public,self.body_anchor))
