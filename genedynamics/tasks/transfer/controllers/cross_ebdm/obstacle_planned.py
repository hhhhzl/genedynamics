"""Shared, model-selected T1 path binding for dense, MPC and data actors."""
import numpy as np
from genedynamics.solvers.single.response_mpc.arm import PublicArmTaskAdamMPCActor
from genedynamics.tasks.transfer.controllers.obstacle_path import PublicObstaclePathPolicy
from .arm_dense import PublicArmTaskDenseActor, PublicArmTaskSourcePriorActor
from .arm_noise import PublicArmTaskNoiseActor


def install_path(actor,path):
    if actor.spec.alias!='T1':raise ValueError('explicit T1 common path only')
    actor.base=PublicObstaclePathPolicy(actor.spec,modes=[actor.mode],path=path)


class PublicPlannedObstacleDenseActor(PublicArmTaskDenseActor):
    VERSION='public-T1-common-planned-dense-v1'
    def __init__(self,*args,path,**kwargs):super().__init__(*args,**kwargs);install_path(self,path)


class PublicPlannedObstacleMPCActor(PublicArmTaskAdamMPCActor):
    VERSION='public-T1-common-planned-adam-mpc-v1'
    def __init__(self,*args,path,**kwargs):super().__init__(*args,**kwargs);install_path(self,path)


class PublicPlannedObstacleNoiseActor(PublicArmTaskNoiseActor):
    VERSION='public-T1-common-planned-local-noise-v1'
    def __init__(self,*args,path,**kwargs):super().__init__(*args,**kwargs);install_path(self,path)


class PublicPlannedObstacleSourcePriorActor(PublicArmTaskSourcePriorActor):
    VERSION = 'public-T1-common-planned-source-prior-v1'

    def __init__(self, *args, path, **kwargs):
        super().__init__(*args, **kwargs)
        install_path(self, path)
