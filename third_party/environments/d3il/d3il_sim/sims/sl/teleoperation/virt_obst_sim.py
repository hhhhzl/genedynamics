import numpy as np
import pandas as pd

from environments.d3il.d3il_sim.sims.mujoco.mj_interactive.ia_objects.pushing_object import (
    VirtualPushObject,
)
from environments.d3il.d3il_sim.sims.mujoco.mj_interactive.ia_robots.mj_push_robot import MjPushRobot
from environments.d3il.d3il_sim.sims.SimFactory import SimRepository
from environments.d3il.d3il_sim.sims.universal_sim.PrimitiveObjects import Cylinder, Sphere
from environments.d3il.d3il_sim.utils.geometric_transformation import euler2quat

GOAL_Y = 0.5
GOAL_X_BOUNDS = [0.25, 0.7]
OBST_Y = [-0.25, 0.0, 0.25]
OBST_X_BOUNDS = [0.3, 0.6]


def virtual_obst_setup(
    orientation=None, pos_x=None, pos_y=None, render=True, proc_id="", hint_alpha=0.3
):
    if orientation is None:
        orientation = 0

    if pos_x is None:
        pos_x = 0.33

    if pos_y is None:
        pos_y = 0.0

    # np.random.seed(seed)
    goal_angle = [0, 0, orientation * np.pi / 180]
    quat = euler2quat(goal_angle)

    obj_list = [
        Sphere(
            name="goal_indicator",
            init_pos=[0.5, GOAL_Y, 0.0],
            init_quat=[0, 1, 0, 0],
            size=[0.05],
            rgba=[0, 1, 0, hint_alpha],
            visual_only=True,
            static=True,
        ),
        Sphere(
            None,
            init_pos=[GOAL_X_BOUNDS[0], GOAL_Y, 0.0],
            init_quat=[0, 1, 0, 0],
            size=[0.02],
            rgba=[0, 0, 1, hint_alpha],
            visual_only=True,
            static=True,
        ),
        Sphere(
            None,
            init_pos=[GOAL_X_BOUNDS[1], GOAL_Y, 0.0],
            init_quat=[0, 1, 0, 0],
            size=[0.02],
            rgba=[0, 0, 1, hint_alpha],
            visual_only=True,
            static=True,
        ),
        Sphere(
            "start_indicator",
            init_pos=[0.35, -1 * GOAL_Y, 0.0],
            init_quat=[0, 1, 0, 0],
            size=[0.02],
            rgba=[1, 1, 0, hint_alpha],
            visual_only=True,
            static=True,
        ),
    ]

    for i in range(len(OBST_Y)):
        obj_list.append(
            Cylinder(
                "obst_{}".format(i),
                init_pos=[0.3, OBST_Y[i], 0.0],
                init_quat=[0, 1, 0, 0],
                size=[0.05, 0.15],
                rgba=[1, 0, 0, 1],
                static=True,
            )
        )

    factory = SimRepository.get_factory("mujoco")

    rm = factory.RenderMode.BLIND
    if render:
        rm = factory.RenderMode.HUMAN

    s = factory.create_scene(object_list=obj_list, render=rm, proc_id=proc_id)
    virtual_bot = MjPushRobot(scene=s)
    # s.start()
    return s, virtual_bot


def remix_goal(scene) -> dict:
    degree = np.random.randint(-60, 60)
    rads = [0, 0, degree * np.pi / 180]
    quat = euler2quat(rads)

    context = {}

    x = np.random.uniform(*GOAL_X_BOUNDS)
    y = GOAL_Y
    context["goal"] = x
    scene.set_obj_pos_and_quat([x, y, 0], quat, obj_name="goal_indicator")

    for i in range(len(OBST_Y)):
        x = np.random.uniform(*OBST_X_BOUNDS)

        obst_name = "obst_{}".format(i)
        context[obst_name] = x
        scene.set_obj_pos_and_quat([x, OBST_Y[i], 0], quat, obj_name=obst_name)
    return context


def set_context(scene, context_dict):
    x = context_dict["goal"]
    y = GOAL_Y
    scene.set_obj_pos_and_quat([x, y, 0], [0, 1, 0, 0], obj_name="goal_indicator")

    for i in range(len(OBST_Y)):
        x = np.random.uniform(*OBST_X_BOUNDS)

        obst_name = "obst_{}".format(i)
        x = context_dict[obst_name]
        scene.set_obj_pos_and_quat([x, OBST_Y[i], 0], [0, 1, 0, 0], obj_name=obst_name)


if __name__ == "__main__":
    df = pd.DataFrame()
    s, r = virtual_obst_setup()
    s.start()
    for i in range(10):
        s.reset()
        c = remix_goal(s)
        df = df.append(c, ignore_index=True)
        # df.to_pickle("./df_test.pkl")
        r.wait(2)
