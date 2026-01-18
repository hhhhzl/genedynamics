# Mockup Tests

This package is used for debugging and fine-tuning the gains for the teleoperation.

## Follow Trajectory
The `demo_follow_trajectory.py` script starts an environment in which the replica robot follows a primary robot over the
teleoperation system. However, the primary robot is replaced by a fixed trajectory. This can be used to adjust the gains
for the teleoperation, since the fixed trajectory is a deterministic environment. Note that in this environment there is
no physical primary robot, the replica robot only follows the pre-recorded positions.

In order to use this package, a trajectory of the primary robot is needed. It can be recorded via the `demo_teacher.py`
in the `teleoperation` package. Log a trajectory and enter its path and name into the `demo_follow_trajectory.py`. Then,
the replica robot should replicate the saved trajectory as if a human controlled the primary roboter.

