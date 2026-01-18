import argparse
import os
from pathlib import Path
import time

import matplotlib.pyplot as plt
from d3il_sim.utils.sim_path import d3il_path
from gamepad_control.logger.logger import Logger as data_logger
from gamepad_control.src import tcp_control

tasks = ["table", "pushing", "sorting", "aligning", "avoiding", "stacking"]

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-t", "--task", type=str, required=True, help="name of the task", choices=tasks)
args = arg_parser.parse_args()
task = args.task

sv_dir = d3il_path("../dataset/data", task, "recorded_data")
print(sv_dir)
Path(sv_dir).mkdir(parents=True, exist_ok=True)

exit()


def show_image(image):
    plt.imshow(image)
    plt.ion()
    plt.axis("off")
    plt.pause(0.000001)
    plt.show()


if __name__ == "__main__":
    n_substeps = 35

    if task == "table":
        from envs.gym_table_env.gym_table.envs.table import Table_Env

        env = Table_Env(n_substeps=n_substeps)

    ########################################################################
    elif task == "pushing":
        from envs.gym_pushing_env.gym_pushing.envs.pushing import \
            Block_Push_Env

        env = Block_Push_Env(n_substeps=n_substeps)

    ########################################################################
    elif task == "sorting":
        from envs.gym_sorting_env.gym_sorting.envs.sorting import Sorting_Env

        env = Sorting_Env(n_substeps=n_substeps, num_boxes=6)

    ########################################################################
    elif task == "aligning":
        from envs.gym_aligning_env.gym_aligning.envs.aligning import \
            Robot_Push_Env

        env = Robot_Push_Env(n_substeps=n_substeps)

    ########################################################################
    elif task == "avoiding":
        from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import \
            ObstacleAvoidanceEnv

        env = ObstacleAvoidanceEnv(n_substeps=n_substeps)

        ########################################################################
    elif task == "stacking":
        from envs.gym_stacking_env.gym_stacking.envs.stacking import \
            CubeStacking_Env

        env = CubeStacking_Env(n_substeps=n_substeps)

    ########################################################################
    else:
        assert False, "no such task"

    ia_controller = tcp_control.TcpGamepadController(env.scene, env.robot)

    env.start()

    logger = data_logger(
        env.scene,
        env.manager,
        env.robot,
        root_dir=os.path.join(os.path.dirname(__file__), sv_dir),
        cam_loggers=env.cam_dict,
        n_substeps=n_substeps,
        **env.log_dict
    )

    # One step to init
    ia_controller.reset_pose()
    env.manager.start()
    env.scene.next_step(log=False)
    logger.start()
    i = 0
    while True:
        # On Stop
        i += 1
        if ia_controller.start():
            break

        elif ia_controller.btn_x():
            logger.start()

        elif ia_controller.btn_a():
            logger.save()

        elif ia_controller.btn_b():
            logger.abort()

        elif ia_controller.btn_y():
            logger.abort()

            time.sleep(0.1)

            for log_name, s in logger.obj_loggers.items():
                s.reset()

            for log_name, s in logger.cam_loggers.items():
                s.reset()

            ia_controller.reset_pose()
            env.manager.start()
            env.scene.next_step(log=False)
            logger.start()
        else:   
            env.robot.open_fingers()
            target_pos, target_quat, des_joints = ia_controller.move(
                timesteps=n_substeps
            )
            logger.log(env.robot.time_stamp, target_pos.copy())
