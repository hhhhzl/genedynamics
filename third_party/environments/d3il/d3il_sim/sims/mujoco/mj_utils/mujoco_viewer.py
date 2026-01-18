import os.path

import imageio
import mujoco_py

from environments.d3il.d3il_sim.utils.sim_path import d3il_path


class MujocoViewer:
    def __init__(self, sim, width=512, height=512, cam_id=-1, rm=True):
        self.sim = sim
        self.width = width
        self.height = height
        self.cam_id = cam_id
        self.mode = rm

        self.recording = False
        self.nframes = 0

        self._viewers = {}
        self.frames = []

        self.fps = 1 / (sim.model.opt.timestep * sim.nsubsteps)

        self._get_viewer()

    def render(self):
        rm = self.mode
        if rm == "blind":
            return

        viewer = self._get_viewer()
        args = [self.width, self.height] if self.mode == "offscreen" else []
        viewer.render(*args)
        if self.recording:
            data = self._get_viewer().read_pixels(self.width, self.height, depth=False)
            self.frames.append(
                data[::-1, :, :]
            )  # original image is upside-down, so flip it

            if self.frames.__len__() == self.nframes:
                self.stop_recording()

    def _get_viewer(self):
        self.viewer = self._viewers.get(self.mode)
        if self.viewer is None:
            if self.mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif self.mode == "offscreen":
                self.viewer = mujoco_py.MjRenderContextOffscreen(
                    self.sim, device_id=self.cam_id
                )
            self._viewer_setup()
            self._viewers[self.mode] = self.viewer
        return self.viewer

    def _viewer_setup(self, distance=3, azimuth=180, elevation=-14):
        if self.mode == "blind":
            return

        body_id = self.sim.model.body_name2id("panda_rb0_hand")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = distance
        self.viewer.cam.azimuth = azimuth
        self.viewer.cam.elevation = elevation

    def save_video(self):
        fname = d3il_path("videos", self.get_file_name())
        writer = imageio.get_writer(fname, fps=self.fps)
        for f in self.frames:
            writer.append_data(f)
        writer.close()
        self.frames = []  # reset frames

    def start_recording(self, nframes=-1):
        if self.mode == "blind":
            raise Exception("Cannot Record videos when RenderMode == BLIND")

        self.nframes = nframes
        self.recording = True

    def stop_recording(self):
        self.save_video()
        self.recording = False

    @staticmethod
    def get_file_name():
        if not os.path.exists(d3il_path("videos")):
            os.mkdir(d3il_path("videos"))
            return "vid_0.mp4"
        else:
            vids = os.listdir(d3il_path("videos"))
            nums = [int(v.split("vid_")[1].split(".")[0]) for v in vids if "vid_" in v]
            num = max(nums) + 1 if nums.__len__() > 0 else 0
            return "vid_" + str(num) + ".mp4"
