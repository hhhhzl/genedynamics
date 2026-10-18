"""Explicit new T4 reset binding: settle feet with the canonical folded arm.

The frozen original reset remains available. This version changes temporary
preparation posture only and must be shared by every compared method. Runtime
posture/controller/plant/physical criteria remain the original ones; new
post-preparation states need fresh scene/data/method qualification.
"""
from dataclasses import replace
from copy import deepcopy
import numpy as np

VERSION='T4-canonical-folded-arm-stance-preparation-v1'
CANONICAL_ARM_QPOS=np.asarray([0.,.785,-.261,-.523,0.,0.])


def bind_folded_reset(provider):
    if provider.config.robot!='go2_z1' or provider.config.socket_profile!='matched_keyway':raise ValueError('explicit T4 matched fixture only')
    if getattr(provider,'staged_reset_version',None) is not None:raise ValueError('cannot silently rebind a staged provider')
    original=provider.reset_batch
    def reset(*args,**kwargs):
        scene=provider.scene
        # Original composite builder's canonical Z1 joint pose, chosen before
        # its Cartesian start IK. No guessed Cartesian height or IK search.
        high=scene.start_qpos.copy();high[np.asarray(provider.qindices)[12:]]=CANONICAL_ARM_QPOS
        manifest=deepcopy(scene.manifest);manifest['reset_binding_version']=VERSION;manifest['preparation_arm_qpos']=CANONICAL_ARM_QPOS.tolist()
        provider.scene=replace(scene,start_qpos=high,manifest=manifest)
        try:return original(*args,**kwargs)
        finally:provider.scene=scene
    provider.reset_batch=reset;provider.staged_reset_version=VERSION
