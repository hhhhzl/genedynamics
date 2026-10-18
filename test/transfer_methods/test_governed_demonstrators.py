"""Native ACP setup-force and virtual-target increments are publicly bounded."""
import numpy as np
import pytest

pytest.importorskip("mujoco")
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.demonstrators.acp import RateLimitedACPTeacher


def test_native_startup_limits_do_not_change_the_spd_action_contract():
    task=instantiate_task("T3",robot="panda");public=task.reset((31300,),seed=31300)
    recipe=dict(force_feedback_gain=1.,force_integral_gain=2.,force_integral_limit=5.,force_command_limit=15.,normal_force_rate=10.,virtual_linear_speed=.02,log_stiffness_rate=2.)
    teacher=RateLimitedACPTeacher(task.spec,0,recipe)
    for i in range(4):
        command=teacher.act(public);row=teacher.limiter_receipts[-1]
        assert abs(row["force_setpoint"])<=(i+1)*.2+1e-12
        assert row["virtual_translation_step"]<=.0004+1e-12
        assert np.linalg.eigvalsh(command["stiffness_tool"][0,0,0]).min()>0
        public["elapsed_seconds"][:]=(i+1)*.02
    assert teacher.integral==0. # no integral windup while force slew-limited
    assert not hasattr(teacher.teacher.task,"state")
