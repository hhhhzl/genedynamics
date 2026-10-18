"""Temporary preparation posture cannot leak into runtime/nullspace posture."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest
from genedynamics.tasks.transfer.staged_preparation import bind_staged_reset,VERSION

@dataclass
class Scene:
    start_qpos:np.ndarray
    manifest:dict


def test_staging_restores_original_scene_even_on_failed_preparation():
    scene=Scene(np.asarray([1.,2.,3.]),{});seen=[]
    p=SimpleNamespace(config=SimpleNamespace(robot='go2_z1',socket_profile='matched_keyway'),scene=scene)
    p.arm_ik=lambda pos,yaw,seed:seed+pos
    def reset(*args,**kwargs):
        seen.append(p.scene);assert p.scene.manifest['reset_binding_version']==VERSION
        if kwargs.get('fail'):raise RuntimeError('physical preparation failed')
        return p.scene.start_qpos.copy()
    p.reset_batch=reset;bind_staged_reset(p);value=p.reset_batch();np.testing.assert_allclose(value,scene.start_qpos+[.44,-.2,.64]);assert p.scene is scene and scene.manifest=={}
    with pytest.raises(RuntimeError):p.reset_batch(fail=True)
    assert p.scene is scene
    with pytest.raises(ValueError):bind_staged_reset(p)
