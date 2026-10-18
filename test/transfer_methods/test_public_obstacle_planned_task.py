"""Shared path reaches dense references and FADA memory without source drift."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.cross_ebdm.obstacle_planned import PublicPlannedObstacleNoiseActor
from genedynamics.tasks.transfer.controllers.cross_ebdm.arm_noise import PublicArmTaskNoiseActor
from genedynamics.experiments.plugins.methods.transfer.fada.obstacle_planned import planned_body_memory
from genedynamics.experiments.plugins.methods.transfer.fada.obstacle import obstacle_body_memory

PATH=[.5425236683058019,.6480255644870154,.6177498942095562,.6013262630830858]


def test_common_plan_keeps_source_and_other_mode_and_binds_same_target_route():
    spec=task_spec('T1',robot='panda');old=PublicArmTaskNoiseActor(spec,mode=0,seed=1);new=PublicPlannedObstacleNoiseActor(spec,mode=0,seed=1,path=PATH);np.testing.assert_array_equal(old.base.routes,new.base.routes);assert type(obstacle_body_memory(spec,[0])) is type(planned_body_memory(spec,[0],PATH))
    spec=task_spec('T1',robot='xarm7');actor=PublicPlannedObstacleNoiseActor(spec,mode=0,seed=1,path=PATH);memory=planned_body_memory(spec,[0],PATH);np.testing.assert_array_equal(actor.base.routes,memory.base.routes);np.testing.assert_array_equal(actor.base.routes[0,1,[0,2]],PATH[:2]);np.testing.assert_array_equal(actor.base.routes[0,2,[0,2]],PATH[2:]);snapshot=actor.get_state();actor.base.phase[:]=3;actor.restore(snapshot);np.testing.assert_array_equal(actor.base.phase,snapshot['base']['phase'])
