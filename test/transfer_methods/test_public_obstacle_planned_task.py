"""Shared path reaches dense references and FADA memory without source drift."""
import numpy as np
from genedynamics.tasks.transfer.specs import task_spec
from genedynamics.tasks.transfer.controllers.cross_ebdm.obstacle_planned import PublicPlannedObstacleNoiseActor
from genedynamics.tasks.transfer.controllers.cross_ebdm.arm_noise import PublicArmTaskNoiseActor
from genedynamics.tasks.transfer.controllers.fada.obstacle_planned import planned_body_memory
from genedynamics.tasks.transfer.controllers.fada.obstacle import obstacle_body_memory
from genedynamics.envs.adapters.transfer_task import instantiate_task
from genedynamics.tasks.transfer.controllers.obstacle_path import PublicObstaclePathPolicy
from genedynamics.tasks.transfer.runtime import MechanismSpec

PATH=[.5425236683058019,.6480255644870154,.6177498942095562,.6013262630830858]


def test_common_plan_keeps_source_and_other_mode_and_binds_same_target_route():
    spec=task_spec('T1',robot='panda');old=PublicArmTaskNoiseActor(spec,mode=0,seed=1);new=PublicPlannedObstacleNoiseActor(spec,mode=0,seed=1,path=PATH);np.testing.assert_array_equal(old.base.routes,new.base.routes);assert type(obstacle_body_memory(spec,[0])) is type(planned_body_memory(spec,[0],PATH))
    spec=task_spec('T1',robot='xarm7');actor=PublicPlannedObstacleNoiseActor(spec,mode=0,seed=1,path=PATH);memory=planned_body_memory(spec,[0],PATH);np.testing.assert_array_equal(actor.base.routes,memory.base.routes);np.testing.assert_array_equal(actor.base.routes[0,1,[0,2]],PATH[:2]);np.testing.assert_array_equal(actor.base.routes[0,2,[0,2]],PATH[2:]);snapshot=actor.get_state();actor.base.phase[:]=3;actor.restore(snapshot);np.testing.assert_array_equal(actor.base.phase,snapshot['base']['phase'])


def test_ur5e_common_path_completes_both_modes_without_relaxing_contract():
    spec=task_spec('T1',robot='ur5e',config_overrides={'reset_joint_noise':0.})
    task=instantiate_task('T1',robot='ur5e',mechanism_spec=MechanismSpec(),seed=376800,config_overrides={'reset_joint_noise':0.})
    public=task.reset((376800,376801),seed=376800,modes=(0,1));actor=PublicObstaclePathPolicy(spec,modes=(0,1),path=[.54,.65,.57,.62])
    for _ in range(spec.config.timeout_steps):
        result=task.step(actor.act(public));public=result.observation
        if np.asarray(result.done).all():break
    episodes=task.evaluate()['episodes']
    assert [row['success'] for row in episodes]==[True,True]
    assert not any(row['constraint_violation'] for row in episodes)
    assert max(row['physical_control_steps'] for row in episodes)<spec.config.timeout_steps
    assert actor.phase.tolist()==[3,3]
