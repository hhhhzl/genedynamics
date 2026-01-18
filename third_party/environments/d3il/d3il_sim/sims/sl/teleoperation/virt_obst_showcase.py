import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import make_interp_spline

from environments.d3il.d3il_sim.sims.sl.teleoperation.demo_teacher import connect_sl_robot
from environments.d3il.d3il_sim.sims.sl.teleoperation.virt_obst_sim import set_context, virtual_obst_setup


def play_scene(n, algo, seed):
    RES_DF = os.path.join(
        os.path.dirname(__file__), "vo_data", "vo_n{}_s{}_res_long.pkl".format(n, seed)
    )
    df = pd.read_pickle(RES_DF)

    robot = connect_sl_robot(
        robot_name="panda1",
        backend_addr="tcp://141.3.53.151:51468",
        local_addr="141.3.53.158",
        gripper_actuation=True,
    )

    # GOOD = 7, 11, 14
    for i, row in df.iterrows():
        if i not in [14]:
            pass
            continue
        print()
        print("########")
        print(i)
        print()

        traj = row[algo + "_traj"]

        c = {
            "obst_0": row["context"][0],
            "obst_1": row["context"][1],
            "obst_2": row["context"][2],
            "goal": row["context"][3],
        }

        robot.gotoCartPositionAndQuat_ImpedanceCtrl(
            [0.43507827, -0.49354531, 0.17805743],
            [0.00284432, 0.74617086, -0.66556491, -0.01563032],
        )

        quat = robot.current_c_quat

        print(robot.current_c_pos)
        print(robot.current_c_quat)

        z = robot.current_c_pos[2]

        trajectoryVel = np.diff(traj, 1, axis=0) / robot.dt
        trajectoryAcc = np.diff(traj, 2, axis=0) / (robot.dt**2)
        spline_traj = np.zeros((1000, 2))

        for axii in range(2):
            l, r = [(1, 0.0), (2, 0.0)], [
                (1, trajectoryVel[0][axii]),
                (2, trajectoryAcc[0][axii]),
            ]
            bsplinef = make_interp_spline(
                x=[0, 1],
                y=[robot.current_c_pos[axii], traj[0, axii]],
                bc_type=(l, r),
                k=5,
            )
            spline_traj[:, axii] = bsplinef(np.linspace(0, 1, 1000))

        traj = np.vstack((spline_traj, traj[1:]))
        robot.activeController = robot.cartesianPosQuatTrackingController
        robot.cartesianPosQuatTrackingController.initController(
            robot, traj.shape[0] // 1000
        )

        cur_c = robot.current_c_pos
        for ii in range(len(traj)):
            robot.cartesianPosQuatTrackingController.setSetPoint(
                np.hstack(([traj[ii][0], traj[ii][1], z], quat))
            )
            robot.nextStep()
        robot.hold_joint_position()


if __name__ == "__main__":
    N_COMPONENTS = 12
    algos = []
    algos.append("ml-cur-dual")
    for a in algos:
        play_scene(N_COMPONENTS, a, 0)
