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

    scene, robot = virtual_obst_setup(render=True, proc_id=0, hint_alpha=0.5)
    scene.start()

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

        ### TRAJECTORY 2D PLOT
        # fig, ax = plt.subplots()
        # ax.plot(traj[:, 1], -1 * traj[:, 0])
        # for x, y in zip([-0.25, 0.0, 0.25, 0.5], row["context"]):
        #     ax.add_patch(plt.Circle((x, -1 * y), 0.05, fill=False))

        # plt.pause(0.005)

        scene.reset()
        set_context(scene, c)

        robot.gotoCartPositionAndQuat_ImpedanceCtrl(
            [0.43507827, -0.49354531, 0.17805743],
            [0.00284432, 0.74617086, -0.66556491, -0.01563032],
        )

        # Z coord is ~ 0.178
        # robot.gotoJointPosition(
        #     [-0.2028, 0.6555, -0.7554, -1.7188, 0.5067, 2.1350, 1.0885]
        # )

        quat = robot.current_c_quat

        print(robot.current_c_pos)
        print(robot.current_c_quat)

        z = robot.current_c_pos[2]

        # robot.gotoCartPositionAndQuat_ImpedanceCtrl([traj[0][0], traj[0][1], z], quat)

        # This creates a b spline with 0 1st and 2nd order derivatives at the boundaries
        trajectoryVel = np.diff(traj, 1, axis=0) / robot.dt
        trajectoryAcc = np.diff(traj, 2, axis=0) / (robot.dt**2)
        spline_traj = np.zeros((1000, 2))

        ### RAW TRAJECTORY POS, VEL, ACC PLOT
        # fig, axs = plt.subplots(3,2)
        # for axi in range(2):
        #     axs[0][axi].plot(np.linspace(0,1, traj.shape[0]), traj[:, axi])
        #     axs[1][axi].plot(np.linspace(0,1, trajectoryVel.shape[0]), trajectoryVel[:, axi])
        #     axs[2][axi].plot(np.linspace(0,1, trajectoryAcc.shape[0]),trajectoryAcc[:, axi])
        # plt.pause(0.005)

        fig, axs = plt.subplots(3, 2)
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
            # print((np.diff(spline_traj[:, axii], 1, axis=0) / robot.dt)[-1])
            # print(trajectoryVel[0])
            print(spline_traj[-1, axii])
            print(traj[0])

            extend_traj = np.hstack((spline_traj[:, axii], traj[1:, axii]))
            axs[0][axii].plot(extend_traj)
            spline_vel = np.diff(extend_traj, 1, axis=0) / robot.dt
            spline_acc = np.diff(extend_traj, 2, axis=0) / (robot.dt**2)
            axs[1][axii].plot(spline_vel)
            axs[2][axii].plot(spline_acc)
            # axs[axii].plot(spline_traj[:, axii])
        # plt.pause(0.005)

        traj = np.vstack((spline_traj, traj[1:]))
        robot.activeController = robot.cartesianPosQuatTrackingController
        robot.cartesianPosQuatTrackingController.initController(
            robot, traj.shape[0] // 1000
        )

        cur_c = robot.current_c_pos
        for ii in range(len(traj)):
            robot.cartesianPosQuatTrackingController.setSetPoint(
                np.hstack(([traj[ii][0], traj[ii][1], z], quat))
                # np.hstack([cur_c, quat])
            )
            robot.nextStep()
        robot.hold_joint_position()
        robot.wait(4)
        break
    plt.show()


if __name__ == "__main__":
    N_COMPONENTS = 12
    algos = []
    algos.append("ml-cur-dual")
    for a in algos:
        play_scene(N_COMPONENTS, a, 0)
