# Derived from Yifan Hou ACP (MIT), commit 74d68eca3bfb5cb59da131c6351d9e1c9b129ff6.
# Copyright (c) 2024 Yifan Hou@Stanford. See LICENSE and SOURCE.json.
"""Source-derived virtual-target labels and anisotropic stiffness decoding."""
import numpy as np
from . import spatial as su


def decode_stiffness_tool(
    reference_pose_world,
    virtual_target_pose_world,
    scalar_stiffness,
    current_tool_pose_world,
    *,
    perpendicular_stiffness=5000.0,
    rotational_stiffness=100.0,
):
    """Reconstruct the author's anisotropic 6x6 stiffness in the current tool frame.

    Inputs are broadcastable (..., 4, 4) world-from-tool poses, scalar
    stiffness in N/m. The translational block softens along the displacement
    from reference to virtual target; rotation uses Nm/rad. The projector
    expression equals upstream S @ diag(k, 5000, 5000) @ inv(S), and remains
    defined for directions parallel to the upstream basis vector (world z).
    The upstream fallback direction below 1 mm is preserved.
    """
    reference = np.asarray(reference_pose_world)
    virtual = np.asarray(virtual_target_pose_world)
    current = np.asarray(current_tool_pose_world)
    for name, pose in [("reference", reference), ("virtual target", virtual), ("current tool", current)]:
        if pose.shape[-2:] != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f"{name} must contain finite world-from-tool 4x4 poses")
    relative_reference = su.SE3_inv(current) @ reference
    relative_virtual = su.SE3_inv(current) @ virtual
    direction = relative_virtual[..., :3, 3] - relative_reference[..., :3, 3]
    norm = np.linalg.norm(direction, axis=-1, keepdims=True)
    direction = np.where(norm < 0.001, np.array([1.0, 0.0, 0.0]), direction)
    direction = direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    projector = direction[..., :, None] * direction[..., None, :]
    stiffness = np.asarray(scalar_stiffness)
    if not np.isfinite(stiffness).all():
        raise ValueError("scalar stiffness must be finite, in N/m")
    translation = perpendicular_stiffness * np.eye(3) + (
        stiffness[..., None, None] - perpendicular_stiffness
    ) * projector
    result = np.zeros(translation.shape[:-2] + (6, 6), dtype=translation.dtype)
    result[..., :3, :3] = translation
    result[..., 3:, 3:] = rotational_stiffness * np.eye(3)
    return result


def decode_action19(action, current_tool_pose_world):
    """Decode native [reference pose9, virtual target pose9, k] for 1 or 2 arms.

    ``action`` is (B, H, 19*R), ``current_tool_pose_world`` is (B, R, 4, 4).
    Returns (B, H, R, 4, 4) poses and (B, H, R) scalar stiffness. Rotation6d
    follows the author's row convention, rather than the common column convention.
    """
    action = np.asarray(action)
    current = np.asarray(current_tool_pose_world)
    if action.ndim != 3 or action.shape[-1] not in (19, 38):
        raise ValueError("native ACP actions must have shape (B, H, 19) or (B, H, 38)")
    arms = action.shape[-1] // 19
    if current.shape != (action.shape[0], arms, 4, 4):
        raise ValueError("current_tool_pose_world must have shape (B, number_of_arms, 4, 4)")
    if not np.isfinite(action).all() or not np.isfinite(current).all():
        raise ValueError("native ACP actions and current poses must be finite")
    packed = action.reshape(action.shape[:2] + (arms, 19))
    reference = current[:, None] @ su.pose9_to_SE3(packed[..., :9])
    virtual = current[:, None] @ su.pose9_to_SE3(packed[..., 9:18])
    return reference, virtual, packed[..., 18]

class VirtualTargetEstimator:
    def __init__(
        self,
        k_max,
        k_min,
        f_low,
        f_high,
        dim=3,
        characteristic_length=0.1,
        vel_tol=0.001,
    ):
        """
        k_max: maximum stiffness
        k_min: minimum stiffness
        f_low: lower bound of the force
        f_high: upper bound of the force
        dim: 3 or 6, 3 for translational, 6 for full 6D
        characteristic_length: the characteristic length for rotational stiffness
        vel_tol: velocity larger than this will trigger the adjustment
        """

        self.k_max = k_max
        self.k_min = k_min
        self.f_low = f_low
        self.f_high = f_high
        self.dim = dim
        self.characteristic_length = characteristic_length
        self.vel_tol = vel_tol

        # internal variables
        self.last_wrench_T = np.zeros(6)
        self.last_pose_WT = np.array([0, 0, 0, 1, 0, 0, 0])

        assert self.dim in [3, 6]

    def update_legacy(self, wrench_T, vel_WT):
        f = -wrench_T[:3]
        f_norm = np.linalg.norm(f)
        if f_norm < self.f_low:
            k = self.k_max
            pos_TC = np.zeros(3)
        elif f_norm > self.f_high:
            k = self.k_min
            pos_TC = f / k
        else:
            k = self.k_max - (self.k_max - self.k_min) * (f_norm - self.f_low) / (
                self.f_high - self.f_low
            )
            pos_TC = f / k
        return k, pos_TC, False

    def update(self, wrench_T, vel_WT):
        """
        wrench_T: 6x1, wrench in the tool frame
        vel_WT: 6x1, body velocity
        """
        wrench_T = np.asarray(wrench_T).copy()
        vel_WT = np.asarray(vel_WT).copy()
        # reg: convert rotational units to translational units
        # The goal is to use the same stiffness coefficient.
        # Real to Reg for characteristic length = 0.1m:
        #   angle: rad -> rad * m = m,  10 rad -> 1 m
        #   torque: Nm -> Nm / m = N,   10 Nm -> 100 N
        #   rotational stiffness: Nm/rad -> Nm/rad / m^2 = N / m, 1Nm/rad -> 100 N/m
        f = -wrench_T[: self.dim]
        f_reg = f.copy()
        if self.dim == 6:
            f_reg[3:] = f_reg[3:] / self.characteristic_length

        # compute stiffness
        f_norm = np.linalg.norm(f_reg)
        if f_norm < self.f_low:
            k = self.k_max
            twist_reg_TC = np.zeros(self.dim)
        elif f_norm > self.f_high:
            k = self.k_min
            twist_reg_TC = f_reg / k
        else:
            k = self.k_max - (self.k_max - self.k_min) * (f_norm - self.f_low) / (
                self.f_high - self.f_low
            )
            twist_reg_TC = f_reg / k

        # reduce influence of friction and gravity
        delta_wrench_T = wrench_T - self.last_wrench_T
        # SE3_TTnew = su.SE3_inv(su.pose7_to_SE3(self.last_pose_WT)) @ su.pose7_to_SE3(
        #     pose_WT
        # )
        # twist_TTnew = su.SE3_to_spt(SE3_TTnew)

        self.last_wrench_T = wrench_T.copy()
        # self.last_pose_WT = pose_WT

        if self.dim == 6:
            delta_wrench_T[3:] = delta_wrench_T[3:] / self.characteristic_length
            # twist_TTnew[3:] = twist_TTnew[3:] * self.characteristic_length
            vel_WT[3:] = vel_WT[3:] * self.characteristic_length
        else:
            delta_wrench_T = delta_wrench_T[:3]
            # twist_TTnew = twist_TTnew[:3]
            vel_WT = vel_WT[:3]

        flag_adjusted = False
        if np.linalg.norm(vel_WT) > self.vel_tol:
            # dot_product = delta_wrench_T.dot(vel_WT)
            # if dot_product > 0:
            # force in delta_twist direction is not spring force
            # adjust the virtual target estimation
            vel_WT_unit = vel_WT / np.linalg.norm(vel_WT)
            twist_reg_TC_motion_component = vel_WT_unit.dot(twist_reg_TC) * vel_WT_unit
            twist_reg_TC_other_component = twist_reg_TC - twist_reg_TC_motion_component

            twist_reg_TC = twist_reg_TC_other_component
            flag_adjusted = True

        twist_TC = twist_reg_TC

        if self.dim == 6:
            twist_TC[3:] = twist_TC[3:] / self.characteristic_length
            SE3_TC = su.spt_to_SE3(twist_TC)
            return k, SE3_TC, flag_adjusted
        else:
            pos_TC = twist_TC
            return k, pos_TC, flag_adjusted
