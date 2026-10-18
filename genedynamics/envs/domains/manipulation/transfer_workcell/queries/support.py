"""Array-namespace source support checks shared by numerical providers."""
from .source import SupportReason
from genedynamics.core.control.transfer import orientation_distance


def semantic_finite_and_rotation(xp, position, rotation, twist, phase):
    finite = (xp.all(xp.isfinite(position)) & xp.all(xp.isfinite(rotation))
              & xp.all(xp.isfinite(twist)) & xp.isfinite(phase))
    proper = (xp.max(xp.abs(rotation.T @ rotation - xp.eye(3, dtype=rotation.dtype))) < 1e-3)
    proper &= xp.abs(xp.linalg.det(rotation) - 1) < 1e-3
    return finite & proper


def support_result(xp, valid, phase, position, rotation, twist, actual_position,
                   actual_rotation, actual_twist, qpos, qvel, ranges, clearance, task, query):
    pe = xp.linalg.norm(actual_position - position)
    ae = orientation_distance(xp, actual_rotation, rotation)
    ve = xp.linalg.norm(actual_twist - twist)
    pose_ok = ((pe <= query.position_tolerance) & (ae <= query.orientation_tolerance)
               & xp.all(qpos >= ranges[:, 0]) & xp.all(qpos <= ranges[:, 1]))
    velocity_ok = (ve <= query.twist_tolerance) & xp.all(xp.abs(qvel) <= task.joint_speed_limit)
    reason = xp.where(~velocity_ok, int(SupportReason.VELOCITY_UNREACHABLE), int(SupportReason.SUPPORTED))
    reason = xp.where(clearance < -task.collision_tolerance, int(SupportReason.COLLISION), reason)
    reason = xp.where(~pose_ok, int(SupportReason.POSE_UNREACHABLE), reason)
    reason = xp.where(phase != 0, int(SupportReason.UNSUPPORTED_PHASE), reason)
    reason = xp.where(~valid, int(SupportReason.INVALID_SEMANTICS), reason)
    return reason == 0, reason, pe, ae, ve
