import inputs

from . import rt_device

# Maximum values for Analogstick Axis and Triggers
AXIS_MAXVALUE = 32767.0
TRIGGER_MAXVALUE = 1023.0

BUTTON_CODES = [
    "BTN_MODE",
    "BTN_START",
    "BTN_SELECT",
    "BTN_NORTH",  # X
    "BTN_SOUTH",  # A
    "BTN_EAST",  # B
    "BTN_WEST",  # Y
    "BTN_TR",
    "BTN_TL",
    "BTN_THUMBR",
    "BTN_THUMBL",
]

TRIGGER_CODES = ["ABS_Z", "ABS_RZ"]

AXIS_CODES = [
    "ABS_X",
    "ABS_Y",
    "ABS_RX",
    "ABS_RY",
    "ABS_HAT0X",  # D-Pad
    "ABS_HAT0Y",  # D-Pad
]


class GamePad(rt_device.RtDeviceMonitor):
    """
    Class to handle all Gamepads
    """

    def __init__(self):
        # Initialize all available Gamepad inputs as 0
        for attr in BUTTON_CODES + TRIGGER_CODES + AXIS_CODES:
            setattr(self, attr, 0)

        super(GamePad, self).__init__("Gamepad")

    def monitor(self):
        # Receive all Events from the connected gamepad. Update the associated fields
        # Axes are normalized to [-1, 1], Triggers to [0, 1]
        while True:
            events = inputs.get_gamepad()
            for e in events:
                if e.code in BUTTON_CODES:
                    setattr(self, e.code, e.state)
                elif e.code in TRIGGER_CODES:
                    setattr(self, e.code, e.state / TRIGGER_MAXVALUE)
                elif e.code in AXIS_CODES:
                    setattr(self, e.code, e.state / AXIS_MAXVALUE)

    def check_connection(self) -> bool:
        try:
            var = inputs.devices.gamepads[0]
            return True
        except IndexError:
            return False
