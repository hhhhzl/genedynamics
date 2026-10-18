"""Deterministic development routes, NOT a source prior or transport baseline.

Uses only public geometry and command constraints. Useful for checking the scene,
13D decoder and terminal accounting before adding the learned models.
"""


def reference_commands(task, skill, route="upper", steps=None):
    import numpy as np
    from genedynamics.core.control.transfer import bounded_command
    if route == "upper":
        waypoints = [(task.start_position[0], task.start_position[1], .72),
                     (.67, task.start_position[1], .72), (.67, task.goal_position[1], .72),
                     task.goal_position]
    elif route == "side":
        waypoints = [(task.start_position[0], -.20, task.start_position[2]),
                     (.68, -.20, task.start_position[2]), (.68, task.goal_position[1], task.goal_position[2]),
                     task.goal_position]
    else:
        raise ValueError("route must be upper or side")
    reference, yaw, previous, index = np.array(task.start_position), 0., np.zeros(13), 0
    commands, dwell = [], 0
    for _ in range(steps or task.timeout_steps):
        target = np.asarray(waypoints[index])
        if np.linalg.norm(reference-target) < .008 and index < len(waypoints)-1:
            dwell += 1
            if dwell >= round(2./task.control_dt):
                index += 1
                dwell = 0
                target = np.asarray(waypoints[index])
        command = np.zeros(13)
        command[:3] = np.clip(3.*(target-reference), -skill.linear_speed, skill.linear_speed)
        command[5] = np.clip(3.*(task.goal_yaw-yaw), -skill.angular_speed, skill.angular_speed)
        previous = bounded_command(np, command, previous, skill, task.control_dt)
        reference = reference + previous[:3] * task.control_dt
        yaw += previous[5] * task.control_dt
        commands.append(command)
    return np.asarray(commands, dtype=np.float32)


def observed_route_modes(position, valid):
    """Classify public TCP crossings as upper, side, unknown, or ambiguous."""
    import numpy as np

    crossing = valid & (position[..., 0] >= .465) & (position[..., 0] <= .535)
    upper = (crossing & (position[..., 2] >= .64)).any(axis=(1, 2))
    side = (crossing & (position[..., 1] <= -.15)).any(axis=(1, 2))
    return np.where(
        upper & side, 2, np.where(upper, 0, np.where(side, 1, -1))
    ).astype(np.int32)
