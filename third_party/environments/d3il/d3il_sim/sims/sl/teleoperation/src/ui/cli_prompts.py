import sched

from environments.d3il.d3il_sim.sims.sl.teleoperation.mockup_tests.follow_trajectory.trajectory_scheduler import (
    MetaTrajectoryScheduler,
)

from ..schedulers.human_scheduler import SchedulerInterface
from ..schedulers.teleop_scheduler import TeleopMetaScheduler


class CliMap:
    def __init__(self, _map: dict):
        self._map = _map

    def get_input(self) -> bool:
        self.print_instructions()
        cmd = str(input("Enter Command... ")).upper()

        if cmd not in self._map:
            print("Wrong Command!")
            return self.get_input()

        func = self._map[cmd][1]

        if func is None:
            return False

        cli_continue = func()
        if cli_continue is None:
            cli_continue = True
        return cli_continue

    def print_instructions(self):
        print()
        for key, (label, _) in self._map.items():
            print("({}) {}".format(key, label))

    def register_function(self, key: str, label: str, fn):
        self._map[key] = (label, fn)


class HumanTeacherCliMap(CliMap):
    def __init__(self, scheduler: SchedulerInterface):
        def fn():
            scheduler.stop_logging()
            scheduler.reset_position()

        _map = {
            "L": ("Start Logging", scheduler.start_logging),
            "P": ("End Logging", fn),
            "R": ("Reset Robot", scheduler.reset_position),
            "Q": ("Exit", lambda: False),
        }
        super(HumanTeacherCliMap, self).__init__(_map)


class HumanTeacherReacherCliMap(CliMap):
    def __init__(self, scheduler: SchedulerInterface, object_reset_callback):
        def composer_function():
            scheduler.stop_logging()
            scheduler.reset_position()
            object_reset_callback()

        _map = {
            "L": ("Start Logging", scheduler.start_logging),
            "P": ("End Logging", scheduler.stop_logging),
            "R": ("Reset Robot", scheduler.reset_position),
            "S": ("Reset Spheres", object_reset_callback),
            "C": ("End Logging and Reset all", composer_function),
            "Q": ("Exit", lambda: False),
        }
        super(HumanTeacherReacherCliMap, self).__init__(_map)


class TeleopCliMap(HumanTeacherCliMap):
    def __init__(self, teleop_scheduler: TeleopMetaScheduler):
        super(TeleopCliMap, self).__init__(teleop_scheduler)

        for i, fb in enumerate(teleop_scheduler.fb_modes):
            # internal_i argument important: otherwise only the last i is used for all selections!
            self._map[str(i)] = (
                fb.name,
                lambda internal_i=i: teleop_scheduler.set_fb_mode(internal_i),
            )
        self.scheduler = teleop_scheduler


class TrajectoryCliMap(CliMap):
    def __init__(self, scheduler: MetaTrajectoryScheduler):
        self.scheduler = scheduler
        _map = {
            "A": ("Start following Trajectory and start logging", self.start_all),
            "S": ("Stop following Trajectory and stop logging", self.stop_all),
            "Q": ("Exit", lambda: False),
        }
        super().__init__(_map)

    def start_all(self):
        self.scheduler.primary_thread.set_follow_trajectory(True)
        self.scheduler.start_logging()

    def stop_all(self):
        self.scheduler.primary_thread.set_follow_trajectory(False)
        self.scheduler.stop_logging()
