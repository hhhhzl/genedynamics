# Copyright (c) 2025 Tyler Lum. MIT; see solver LICENSE.upstream.
# Source: tylerlum/human2sim2robot commit 894eae2ec3ae39a573b81bd1860d14cc6bdfa6df.
# Native reward/reset and trajectory-clipping definitions, unchanged.
from typing import Dict
import numpy as np
import torch

@torch.jit.script
def compute_reward_jit(
    rew_buf: torch.Tensor,
    is_object_lifted: torch.Tensor,
    is_goal_object_lifted: torch.Tensor,
    smallest_this_episode_indexfingertip_goal_distance: torch.Tensor,
    indexfingertip_goal_distance: torch.Tensor,
    smallest_this_episode_fingertips_object_distance: torch.Tensor,
    fingertips_object_distance: torch.Tensor,
    smallest_this_episode_object_goal_distance: torch.Tensor,
    object_goal_distance: torch.Tensor,
    is_fingertips_object_close: torch.Tensor,
    is_in_success_region: torch.Tensor,
    has_enough_consecutive_successes_to_end_episode: torch.Tensor,
    max_episode_length: int,
    progress_buf: torch.Tensor,
    raw_actions: torch.Tensor,
    prev_raw_actions: torch.Tensor,
    object_and_goal_far_apart_need_stop_reference_motion: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    # Make sure this aligns with _setup_reward_weights
    reward_dict = {}

    reward_dict["Indexfingertip-Goal Distance Reward"] = torch.where(
        smallest_this_episode_indexfingertip_goal_distance
        < indexfingertip_goal_distance,
        torch.zeros_like(rew_buf),
        smallest_this_episode_indexfingertip_goal_distance
        - indexfingertip_goal_distance,
    )

    reward_dict["Fingertips-Object Distance Reward"] = torch.where(
        smallest_this_episode_fingertips_object_distance < fingertips_object_distance,
        torch.zeros_like(rew_buf),
        smallest_this_episode_fingertips_object_distance - fingertips_object_distance,
    )

    robot_is_lifting_object_and_should = torch.logical_and(
        torch.logical_and(is_object_lifted, is_goal_object_lifted),
        is_fingertips_object_close,
    )

    reward_dict["Object-Goal Distance Reward"] = torch.where(
        smallest_this_episode_object_goal_distance < object_goal_distance,
        torch.zeros_like(rew_buf),
        smallest_this_episode_object_goal_distance - object_goal_distance,
    )
    reward_dict["Success Reward"] = torch.where(
        is_in_success_region,
        torch.ones_like(rew_buf),
        torch.zeros_like(rew_buf),
    )
    reward_dict["Success Reward"] = torch.where(
        robot_is_lifting_object_and_should,
        reward_dict["Success Reward"],
        torch.zeros_like(rew_buf),
    )

    reward_dict["Consecutive Success Reward"] = torch.where(
        has_enough_consecutive_successes_to_end_episode,
        torch.ones_like(rew_buf) * (max_episode_length - progress_buf),
        torch.zeros_like(rew_buf),
    )
    reward_dict["Consecutive Success Reward"] = torch.where(
        robot_is_lifting_object_and_should,
        reward_dict["Consecutive Success Reward"],
        torch.zeros_like(rew_buf),
    )

    # Keypoint errors range from 0 to 0.5 most of the time
    #   * Roughly 0.1 keypoint error is still not great, but starting to get good
    #   * Roughly 0.01 keypoint error is very good
    # Thus we want the following approximate values
    #   * 0.01 keypoint error -> ~1.0 reward
    #   * 0.1 keypoint error -> ~0.3 reward
    # r = exp(-10 * e) roughly satisfies this
    TRACKING_ERROR_SCALE = 10.0
    reward_dict["Object Tracking Reward"] = torch.where(
        is_fingertips_object_close,
        torch.exp(-object_goal_distance * TRACKING_ERROR_SCALE),
        torch.zeros_like(rew_buf),
    )

    GIVE_LIFT_REWARD = False
    if GIVE_LIFT_REWARD:
        # Give a boost to the object tracking reward if the robot is lifting the object and should
        reward_dict["Object Tracking Reward"] = torch.where(
            robot_is_lifting_object_and_should,
            5 * reward_dict["Object Tracking Reward"],
            reward_dict["Object Tracking Reward"],
        )

    # Downscale the reward if the object tracking is bad
    reward_dict["Object Tracking Reward"] = torch.where(
        object_and_goal_far_apart_need_stop_reference_motion,
        0.1 * reward_dict["Object Tracking Reward"],
        reward_dict["Object Tracking Reward"],
    )

    # Action smoothing penalty
    reward_dict["Action Smoothing Penalty"] = (raw_actions - prev_raw_actions).norm(
        p=2, dim=-1
    ) ** 2

    reward_dict["Hand Tracking Reward"] = torch.zeros_like(rew_buf)

    return reward_dict

@torch.jit.script
def compute_reset_jit(
    reset_buf: torch.Tensor,
    progress_buf: torch.Tensor,
    max_episode_length: int,
    object_fallen_off_table: torch.Tensor,
    reference_motion_done: torch.Tensor,
    object_and_goal_far_apart_need_reset: torch.Tensor,
    has_enough_consecutive_successes_to_end_episode: torch.Tensor,
    is_fingertips_object_close: torch.Tensor,
    FORCE_REFERENCE_TRAJECTORY_TRACKING: bool,
    EARLY_RESET_BASED_ON_FINGERTIPS_OBJECT_DISTANCE: bool,
) -> torch.Tensor:
    reset = torch.zeros_like(reset_buf)
    reset[progress_buf >= max_episode_length] = 1
    reset[object_fallen_off_table] = 1
    if FORCE_REFERENCE_TRAJECTORY_TRACKING:
        reset[reference_motion_done] = 1
        reset[object_and_goal_far_apart_need_reset] = 1
    reset[has_enough_consecutive_successes_to_end_episode] = 1
    if (
        FORCE_REFERENCE_TRAJECTORY_TRACKING
        and EARLY_RESET_BASED_ON_FINGERTIPS_OBJECT_DISTANCE
    ):
        reset[~is_fingertips_object_close] = 1  # If the object is not close, reset
    return reset

def clip_T_list(raw_T_list: torch.Tensor, data_dt: float) -> torch.Tensor:
    N_raw = raw_T_list.shape[0]
    assert raw_T_list.shape == (
        N_raw,
        4,
        4,
    ), f"Expected shape {(N_raw, 4, 4)}, got {raw_T_list.shape}"

    all_xyz = raw_T_list[:, :3, 3].cpu().numpy()

    # Use a window for smoother speed calculation
    WINDOW = 30
    speeds = np.linalg.norm(all_xyz[WINDOW:] - all_xyz[:-WINDOW], axis=-1) / (
        WINDOW * data_dt
    )

    # MIN_SPEED = 0.05  # BRITTLE: May need to tune per task
    MIN_SPEED = np.percentile(speeds, 50)
    is_moving = speeds > MIN_SPEED

    N_CONSECUTIVE_MOVING = 10
    # Create a rolling sum over a window of size N_CONSECUTIVE_MOVING
    rolling_sum = np.convolve(
        is_moving, np.ones(N_CONSECUTIVE_MOVING, dtype=int), mode="valid"
    )
    is_moving_consecutively = rolling_sum == N_CONSECUTIVE_MOVING

    PLOT = False
    if PLOT:
        import matplotlib.pyplot as plt

        plt.plot(speeds, label="speeds")
        plt.plot([MIN_SPEED] * N_raw, label="MIN_SPEED")
        plt.plot(
            is_moving_consecutively.astype(float) * np.max(speeds),
            label="is_moving_consecutively",
        )
        plt.legend()
        plt.show()
        breakpoint()

    is_moving_idxs = np.where(is_moving_consecutively)[0]
    if is_moving_idxs.size == 0:
        raise ValueError(
            f"No moving object found in given trajectory, speeds = {speeds}, moving = {is_moving}, np.max(speeds) = {np.max(speeds)}"
        )

    start_step = is_moving_idxs[0]

    END_EARLY = False
    if END_EARLY:
        end_step = is_moving_idxs[-1] + WINDOW
    else:
        end_step = N_raw

    return raw_T_list[start_step:end_step]
