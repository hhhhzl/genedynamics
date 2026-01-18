import abc
import threading


class RtDeviceMonitor(abc.ABC):
    """
    Base Class for monitoring a device sending data in realtime
    """

    def __init__(self, device_type: str):
        if not self.check_connection():
            raise DeviceNotFoundError("No {} connection detected.".format(device_type))

        # Create and start a thread for receiving data
        self._monitor_thread = threading.Thread(target=self.monitor)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    @abc.abstractmethod
    def monitor(self):
        """
        Thread function to monitor the connected device
        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def check_connection(self) -> bool:
        """
        check if the device is connected correctly
        Returns:
            bool: connection success
        """
        pass


class DeviceNotFoundError(Exception):
    """
    Error class when a connection cannot be established to the RT Device
    """

    pass
