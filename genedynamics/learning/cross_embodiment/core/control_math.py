"""Small array-namespace control operations, shared by reference and JAX kernels.

No tensor library is imported here; xp is chosen by the backend. The public
orchestrator never depends on these implementation helpers.
"""


def bounded_command(xp, command, previous, skill, dt):
    amplitude = xp.asarray([skill.linear_speed] * 3 + [skill.angular_speed] * 3
                           + [skill.log_stiffness_limit] * 6 + [0.], dtype=command.dtype)
    rate = xp.asarray([skill.linear_acceleration] * 3 + [skill.angular_acceleration] * 3
                      + [skill.log_stiffness_rate] * 6 + [0.], dtype=command.dtype) * dt
    bounded = xp.clip(command, -amplitude, amplitude)
    return previous + xp.clip(bounded - previous, -rate, rate)


def integrate_rotation(xp, matrix, angular, dt):
    v = angular * dt
    squared = xp.sum(v*v)
    safe_squared = xp.maximum(squared, 1e-12)
    theta = xp.sqrt(safe_squared)
    x, y, z = v[0], v[1], v[2]
    zero = xp.asarray(0., dtype=v.dtype)
    skew = xp.stack((xp.stack((zero, -z, y)), xp.stack((z, zero, -x)), xp.stack((-y, x, zero))))
    # Both branches have finite derivatives at zero; norm(v) followed by sinc
    # would still introduce a NaN gradient through the norm at v=0.
    a = xp.where(squared < 1e-6, 1.-squared/6.+squared*squared/120., xp.sin(theta)/theta)
    b = xp.where(squared < 1e-6, .5-squared/24.+squared*squared/720., (1.-xp.cos(theta))/safe_squared)
    delta = xp.eye(3, dtype=v.dtype) + a * skew + b * (skew @ skew)
    return delta @ matrix


def orientation_error(xp, current, desired):
    return .5 * xp.sum(xp.cross(current.T, desired.T), axis=0)


def orientation_distance(xp, current, desired):
    # atan2 is well-conditioned near zero, unlike acos(trace).
    sine = xp.linalg.norm(orientation_error(xp, current, desired))
    cosine = xp.clip((xp.trace(desired @ current.T) - 1.) / 2., -1., 1.)
    return xp.arctan2(sine, cosine)


def impedance(xp, position, rotation, qvel, jacp, jacr, gravity,
              reference_position, reference_rotation, command, integral_error, skill, gain, limits,
              qpos, posture):
    stiffness = xp.asarray(skill.reference_stiffness, dtype=qvel.dtype) * xp.exp(command[6:12])
    error = xp.concatenate((reference_position - position,
                            orientation_error(xp, rotation, reference_rotation)))
    jac = xp.concatenate((jacp, jacr), axis=0)  # [6,nv]
    integral_gain = xp.asarray([400.]*3 + [20.]*3, dtype=qvel.dtype)
    wrench = stiffness * error + integral_gain * integral_error + 2. * xp.sqrt(stiffness) * (command[:6] - jac @ qvel)
    posture_tau = 10. * (posture-qpos)
    null_tau = posture_tau - jac.T @ xp.linalg.solve(jac @ jac.T + .001*xp.eye(6, dtype=qvel.dtype), jac @ posture_tau)
    tau = gain * (jac.T @ wrench + null_tau) + gravity - 2. * qvel
    return xp.clip(tau, -limits, limits), xp.max(xp.abs(tau) / limits)


def integrate_error(xp, previous, position, rotation, refpos, refrot, dt):
    limit = xp.asarray([.1]*3 + [.1]*3, dtype=previous.dtype)
    error = xp.concatenate((refpos-position, orientation_error(xp, rotation, refrot)))
    return xp.clip(previous + dt*error, -limit, limit)


def terminal_status(xp, old_status, steps, hold_count, collision, joint_limit, speed_limit,
                    finite, position_error, angle_error, task):
    active = old_status == 0
    reached = (position_error <= task.position_tolerance) & (angle_error <= task.orientation_tolerance)
    hold = xp.where(reached, hold_count + 1, 0)
    code = xp.where(steps >= task.timeout_steps, 5, 0)
    code = xp.where(hold >= task.hold_steps, 1, code)
    code = xp.where(speed_limit, 4, code)
    code = xp.where(joint_limit, 3, code)
    code = xp.where(collision, 2, code)
    code = xp.where(~finite, 6, code)
    return xp.where(active, code, old_status), xp.where(active, hold, hold_count)
