import glob
import os
import numpy as np
import cv2
import pickle


class Logger:
    def __init__(self, scene, manager, robot, root_dir="./", cam_loggers=None, n_substeps=35, **obj_loggers) -> None:
        os.makedirs(root_dir, exist_ok=True)
        self.root_dir = root_dir

        if not os.path.isdir(root_dir):
            os.makedirs(root_dir)

        self.scene = scene
        self.manager = manager
        self.robot = robot
        self.cam_loggers = cam_loggers
        self.obj_loggers = obj_loggers
        self.n_substeps = n_substeps

        self.log_duration = 100000

        self.log_counter = len(glob.glob(os.path.join(self.root_dir, "env_*")))

        self.is_logging = False

        self._target_pos = []
        self._ts = []

    def start(self):
        if self.is_logging:
            print("Log already started.")
            return

        self.is_logging = True
        # use step_count as the interval
        self.scene.start_logging(duration=self.log_duration, log_interval=self.n_substeps)

    def abort(self):
        if not self.is_logging:
            print("No log in process.")
            return

        self.is_logging = False
        self.scene.stop_logging()
        self._target_pos = []
        self._ts = []

    def save(self, filename=None):
        if not self.is_logging:
            print("No log in process.")
            return

        self.scene.stop_logging()

        robot_logger = self.robot.robot_logger

        env_state = dict()

        env_state["context"] = self.manager.context

        env_state["robot"] = {}

        env_state["robot"]["time_stamp"] = robot_logger.time_stamp
        env_state["robot"]["sim_step"] = robot_logger.step
        env_state["robot"]["wall_clock"] = robot_logger.wall_clock
        env_state["robot"]["j_pos"] = robot_logger.joint_pos
        env_state["robot"]["j_vel"] = robot_logger.joint_vel
        env_state["robot"]["c_pos"] = robot_logger.cart_pos
        env_state["robot"]["c_vel"] = robot_logger.cart_vel
        env_state["robot"]["c_quat"] = robot_logger.cart_quat
        env_state["robot"]["des_c_pos"] = robot_logger.des_c_pos
        env_state["robot"]["des_c_quat"] = robot_logger.des_quat
        env_state["robot"]["des_j_pos"] = robot_logger.des_joint_pos
        env_state["robot"]["des_j_vel"] = robot_logger.des_joint_vel
        env_state["robot"]["des_j_acc"] = robot_logger.des_joint_acc
        env_state["robot"]["gripper_width"] = robot_logger.gripper_width

        for log_name, logger in self.cam_loggers.items():
            env_state[log_name] = {}
            env_state[log_name]["sim_step"] = logger.step

            image_dirs = []

            images = np.stack(logger.color_image)

            for num, image in enumerate(images):

                if filename is None:
                    sv_dir = self.root_dir + '/' + log_name + '/env_{:03d}_{:02d}'.format(self.log_counter, self.manager.index)
                else:
                    sv_dir = self.root_dir + '/' + log_name + '/' + filename.split('.')[0]

                if not os.path.isdir(sv_dir):
                    os.makedirs(sv_dir)

                sv_img = os.path.join(sv_dir, '{}.jpg'.format(num))

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(sv_img, image)
                image_dirs.append(sv_img)

        for log_name, logger in self.obj_loggers.items():
            env_state[log_name] = {}

            env_state[log_name]["sim_step"] = logger.step
            env_state[log_name]["pos"] = logger.pos
            env_state[log_name]["quat"] = logger.orientation

        state_dir = self.root_dir
        if not os.path.isdir(state_dir):
            os.makedirs(state_dir)

        if filename is None:
            filename = "env_{:03d}_{:02d}.pkl".format(self.log_counter, self.manager.index)

        with open(
            os.path.join(
                state_dir,
                filename,
            ),
            "wb",
        ) as f:
            pickle.dump(env_state, f)

        self.is_logging = False
        self._target_pos = []
        self._ts = []
        self.log_counter += 1

    def log(self, ts, target):
        if not self.is_logging:
            return
        self._ts.append(ts)
        self._target_pos.append(target)
