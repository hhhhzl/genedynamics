import numpy as np
import pinocchio

from environments.d3il.d3il_sim.utils.sim_path import d3il_path


class RobotKinematicsInterface:
    def getForwardKinematics(self, q=None):
        pass

    def getJacobian(self, q=None):
        pass


class RobotDynamicsInterface:
    def get_gravity(self, q=None):
        pass

    def get_coriolis(self, q=None, qd=None):
        pass

    def get_mass_matrix(self, q=None):
        pass


class RobotModelFromPinochio(RobotKinematicsInterface, RobotDynamicsInterface):
    def __init__(self, obj_urdf="./models/common/robots/panda_arm_hand_pinocchio.urdf"):
        obj_urdf = d3il_path(obj_urdf)
        self.pin_model = pinocchio.buildModelFromUrdf(obj_urdf)
        self.pin_data = self.pin_model.createData()

        self.pin_end_effector_frame_id = self.pin_model.getFrameId("panda_grasptarget")

        self.pin_q = np.zeros(self.pin_model.nv)
        self.pin_qd = np.zeros(self.pin_model.nv)

    def getForwardKinematics(self, q):

        # account for additional joints (e.g. finger)
        self.pin_q[:7] = q
        pinocchio.framesForwardKinematics(self.pin_model, self.pin_data, self.pin_q)

        current_c_pos = np.array(
            self.pin_data.oMf[self.pin_end_effector_frame_id].translation
        )

        quat_pin = pinocchio.Quaternion(
            self.pin_data.oMf[self.pin_end_effector_frame_id].rotation
        ).coeffs()  # [ x, y, z, w]
        current_c_quat = np.zeros(4)
        current_c_quat[1:] = quat_pin[:3]
        current_c_quat[0] = quat_pin[-1]

        return current_c_pos, current_c_quat

    def getJacobian(self, q):
        self.pin_q[:7] = q

        pinocchio.computeJointJacobians(self.pin_model, self.pin_data, self.pin_q)
        pinocchio.framesForwardKinematics(self.pin_model, self.pin_data, self.pin_q)
        return pinocchio.getFrameJacobian(
            self.pin_model,
            self.pin_data,
            self.pin_end_effector_frame_id,
            pinocchio.LOCAL_WORLD_ALIGNED,
        )[:, :7]

    def get_gravity(self, q=None):
        self.pin_q[:7] = q

        vq = np.zeros(self.pin_q.shape)
        aq0 = np.zeros(self.pin_q.shape)

        # compute dynamic drift -- Coriolis, centrifugal, gravity -- as velocity is zero this should only be gravity
        b = pinocchio.rnea(self.pin_model, self.pin_data, self.pin_q, vq, aq0)
        return b[:7]

    def get_coriolis(self, q, qd):
        self.pin_q[:7] = q
        self.pin_qd[:7] = qd

        aq0 = np.zeros(self.pin_q.shape)

        # compute dynamic drift -- Coriolis, centrifugal, gravity
        b = pinocchio.rnea(self.pin_model, self.pin_data, self.pin_q, self.pin_qd, aq0)
        return b[:7] - self.get_gravity(q)

    def get_mass_matrix(self, q):
        if q.shape[0] == 7:
            q_tmp = q.copy()
            q = np.zeros(9)
            q[:7] = q_tmp
            # compute mass matrix M
        M = pinocchio.crba(self.pin_model, self.pin_data, q)
        return M[:7, :7]
