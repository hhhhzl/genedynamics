from enum import Enum
from typing import List, Tuple

from environments.d3il.d3il_sim.core.sim_object.sim_object import SimObject


class PrimitiveObject(SimObject):
    class Shape(Enum):
        BOX = "box"
        SPHERE = "sphere"
        CYLINDER = "cylinder"

    def __init__(
        self,
        name: str,
        init_pos,
        init_quat,
        mass: float,
        size,
        rgba,
        static: bool,
        visual_only: bool,
        obj_type: Shape,
        solimp: List[float] = None,
        solref: List[float] = None,
    ):
        super(PrimitiveObject, self).__init__(name, init_pos, init_quat)
        self.type = obj_type
        self.mass = mass
        self.size = size
        self.static = static
        self.visual_only = visual_only

        if rgba is None:
            rgba = [0.5, 0.5, 0.5, 1.0]
        self.rgba = rgba

        self.solimp = solimp
        self.solref = solref

    def get_poi(self) -> list:
        return [self.name]

    # PrimitiveObject Loading function is registered by each Simulators Factory!


class Box(PrimitiveObject):
    def __init__(
        self,
        name: str,
        init_pos,
        init_quat,
        mass: float = 0.1,
        size=None,
        rgba=None,
        static: bool = False,
        visual_only: bool = False,
        solimp: List[float] = None,
        solref: List[float] = None,
    ):
        if size is None:
            size = [0.02, 0.02, 0.02]

        if len(size) != 3:
            raise ValueError("Expected list of size three for size attribute.")
        super(Box, self).__init__(
            name,
            init_pos,
            init_quat,
            mass,
            size,
            rgba,
            static,
            visual_only,
            self.Shape.BOX,
            solimp,
            solref,
        )


class Sphere(PrimitiveObject):
    def __init__(
        self,
        name: str,
        init_pos,
        init_quat,
        mass: float = 0.1,
        size=None,
        rgba=None,
        static: bool = False,
        visual_only: bool = False,
        solimp: List[float] = None,
        solref: List[float] = None,
    ):
        if size is None:
            size = [0.02]

        if len(size) != 1:
            raise ValueError("Expected list of size one for size attribute")
        super(Sphere, self).__init__(
            name,
            init_pos,
            init_quat,
            mass,
            size,
            rgba,
            static,
            visual_only,
            self.Shape.SPHERE,
            solimp,
            solref,
        )


class Cylinder(PrimitiveObject):
    def __init__(
        self,
        name: str,
        init_pos,
        init_quat,
        mass: float = 0.1,
        size=None,
        rgba=None,
        static: bool = False,
        visual_only: bool = False,
        solimp: List[float] = None,
        solref: List[float] = None,
    ):
        if size is None:
            size = [0.02, 0.02]

        if len(size) != 2:
            raise ValueError("Expected list of size two for size attribute.")
        super(Cylinder, self).__init__(
            name,
            init_pos,
            init_quat,
            mass,
            size,
            rgba,
            static,
            visual_only,
            self.Shape.CYLINDER,
            solimp,
            solref,
        )
