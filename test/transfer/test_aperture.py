import numpy as np
import pytest

from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.aperture import (
    OBJECT_VERTICES_TCP, PublicApertureBrakePolicy, object_vertices,
)
from genedynamics.tasks.transfer.runtime import MechanismSpec, ObstacleTransferConfig
from genedynamics.tasks.transfer.specs import task_spec


def test_aperture_config_and_semantic_task_are_explicit():
    with pytest.raises(ValueError, match="scene profile"):
        ObstacleTransferConfig(scene_profile="unknown")
    spec = task_spec("aperture_brake", robot="panda")
    assert spec.name == "aperture_brake"
    assert spec.alias == "T1"
    assert spec.config.scene_profile == "aperture_brake_v1"


def test_object_envelope_is_complete_rigid_payload_geometry():
    rotation = np.diag([1., -1., -1.])
    position = np.asarray([.4, -.09, .55])
    vertices = object_vertices(position, rotation)
    assert vertices.shape == OBJECT_VERTICES_TCP.shape == (16, 3)
    np.testing.assert_allclose(vertices.min(axis=0), [.34, -.16, .47])
    np.testing.assert_allclose(vertices.max(axis=0), [.46, -.08, .49])


def test_aperture_adapter_exposes_public_object_and_substep_evidence():
    task = instantiate_task(
        "aperture_brake", robot="panda", mechanism_spec=MechanismSpec(), seed=7,
    )
    public = task.reset((7,), seed=7, modes=[0])
    for key in (
        "carried_object_position", "carried_object_rotation",
        "carried_object_twist", "aperture_geometry",
    ):
        assert key in public
    actor = PublicApertureBrakePolicy(task.spec, mode=0)
    task.step(actor.act(public))
    trace = task.records[-1].traces[-1]
    assert trace["tcp_position_substeps"].shape == (task.config.substeps, 3)
    assert trace["frame_contact_substeps"].shape == (task.config.substeps,)
