"""Explicit new T4 reset binding: settle feet with the tool above the fixture.

The frozen original reset remains available. This version changes temporary
preparation posture only and must be shared by every compared method. Runtime
posture/controller/plant/physical criteria remain the original ones; new
post-preparation states need fresh scene/data/method qualification.
"""
from dataclasses import replace
from copy import deepcopy
import numpy as np

VERSION='T4-raised-tool-stance-preparation-v1'
STAGING_POSITION=np.asarray([.44,-.2,.64])


def bind_staged_reset(provider):
    if provider.config.robot!='go2_z1' or provider.config.socket_profile!='matched_keyway':raise ValueError('explicit T4 matched fixture only')
    if getattr(provider,'staged_reset_version',None) is not None:raise ValueError('cannot silently rebind a staged provider')
    original=provider.reset_batch
    def reset(*args,**kwargs):
        scene=provider.scene
        # Environment-owner IK, fixed public staging point, no actor input or
        # actual mechanism passed to any planner/solver. No physics here.
        high=provider.arm_ik(STAGING_POSITION.copy(),0.,scene.start_qpos.copy())
        manifest=deepcopy(scene.manifest);manifest['reset_binding_version']=VERSION;manifest['preparation_staging_position']=STAGING_POSITION.tolist()
        provider.scene=replace(scene,start_qpos=high,manifest=manifest)
        try:return original(*args,**kwargs)
        finally:provider.scene=scene
    provider.reset_batch=reset;provider.staged_reset_version=VERSION
