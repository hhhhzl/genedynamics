import os

import mujoco_py as mj
from packaging import version

### HACKY MUJOCO 2.1 / 2.0 OFFSCREEN RENDERING FIX
### MJ 2.1.2.14 crashes with id=-1, but is not truly offscreen
### On the cluster, however, id=-1 seems to work and 0 doesn't
### TODO: Remove once fixed!!
MJ_RENDER_DEVICE_ID = -1
mj_version = version.parse(mj.__version__)
if (
    mj_version >= version.parse("2.1")
    and mj_version < version.parse("2.2")
    and "CLUSTER" not in os.environ
):
    MJ_RENDER_DEVICE_ID = 0


__RENDER_CTX_MAP = {}


def get_renderer(sim: mj.MjSim, name: str):
    global __RENDER_CTX_MAP
    if name not in __RENDER_CTX_MAP:
        ctx = mj.MjRenderContextOffscreen(sim, MJ_RENDER_DEVICE_ID)
        __RENDER_CTX_MAP[name] = ctx
    ctx = __RENDER_CTX_MAP[name]
    ctx.update_sim(sim)
    return ctx


def render(
    sim: mj.MjSim,
    cam_name: str,
    width: int,
    height: int,
    depth: bool,
    segmentation: bool,
):
    """
    renders an image in mujoco
    :param sim: current MjSim
    :param width: width in pixels
    :param height: height in pixels
    :param cam_name: name of the camera.
    :param depth: bool for depth data
    :param segmentation: bool for object segmentation
    :return: pixel array
    """
    ctx = get_renderer(sim, cam_name)
    cam_id = sim.model._camera_name2id[cam_name]

    ctx._set_mujoco_buffers()

    ctx.render(width, height, cam_id, segmentation=segmentation)
    return ctx.read_pixels(width, height, depth=depth, segmentation=segmentation)
