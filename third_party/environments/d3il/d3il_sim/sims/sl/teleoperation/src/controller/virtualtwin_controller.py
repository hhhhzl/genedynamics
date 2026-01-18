from .teleop_controller import TeleopReplicaController


class VirtualTwinController(TeleopReplicaController):
    """
    Digital Twin Controller, as digital robots do not compute Kalman Filter correctly.
    """

    def __init__(self, digital_scene):
        self.scene = digital_scene
        self.robot = digital_scene.robots[0]

    def initialize(self):
        self.scene.start()

    def get_load(self):
        return 0

    def reset(self):
        self.scene.reset()
