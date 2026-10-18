"""Canonical folded preparation is temporary and has no Cartesian IK query."""
from dataclasses import dataclass
from types import SimpleNamespace
import numpy as np
import pytest
from genedynamics.tasks.transfer.folded_preparation import bind_folded_reset,CANONICAL_ARM_QPOS

@dataclass
class Scene:
    start_qpos:np.ndarray
    manifest:dict


def test_native_folded_joint_pose_preserves_runtime_scene_and_restores_failures():
    scene=Scene(np.arange(25,dtype=float),{});p=SimpleNamespace(config=SimpleNamespace(robot='go2_z1',socket_profile='matched_keyway'),scene=scene,qindices=np.arange(7,25))
    def reset(*args,**kwargs):
        np.testing.assert_array_equal(p.scene.start_qpos[19:],CANONICAL_ARM_QPOS);np.testing.assert_array_equal(p.scene.start_qpos[:19],scene.start_qpos[:19])
        if kwargs.get('fail'):raise RuntimeError('setup failed')
        return p.scene.start_qpos.copy()
    p.reset_batch=reset;bind_folded_reset(p);p.reset_batch();assert p.scene is scene
    with pytest.raises(RuntimeError):p.reset_batch(fail=True)
    assert p.scene is scene and scene.manifest=={}
