"""Target-delta mapping with physics-verified rotation and TCP/tool offset closure."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from forcewipe_v4.path_pose_control import quaternion_wxyz_to_matrix
from forcewipe_v4.scene_geometry import V4_TOOL_TCP_CENTER_OFFSET_M
from forcewipe_v6.geometry_target_pose import (
    GeometryTargetPoseConfig,
    GeometryTargetPoseError,
    GeometryTargetPoseProposal,
)


def proposal_to_target_delta_action_r3(
    proposal: GeometryTargetPoseProposal,
    *,
    root_quaternion_world_wxyz: np.ndarray,
    target_quaternion_root_wxyz: np.ndarray,
    config: GeometryTargetPoseConfig = GeometryTargetPoseConfig(),
) -> np.ndarray:
    """Map desired tool-center motion to a TCP target-delta action.

    The tool center is connected to the TCP at ``[0, 0, offset]`` in the TCP
    frame.  Rotation therefore moves the tool center unless the TCP position
    receives the equal-and-opposite rigid-offset correction.
    """

    config.validate()
    world_from_root = quaternion_wxyz_to_matrix(root_quaternion_world_wxyz)
    root_from_target = quaternion_wxyz_to_matrix(target_quaternion_root_wxyz)
    desired_world_from_tool = np.asarray(proposal.desired_world_from_tool_after, dtype=np.float64)
    if desired_world_from_tool.shape != (3, 3) or not np.all(np.isfinite(desired_world_from_tool)):
        raise GeometryTargetPoseError("desired tool rotation is invalid")
    desired_root_from_tool = world_from_root.T @ desired_world_from_tool
    action_space_delta = root_from_target @ desired_root_from_tool.T
    rotation_delta = Rotation.from_matrix(action_space_delta).as_euler("XYZ", degrees=False)
    rotation_norm = float(np.linalg.norm(rotation_delta))
    if rotation_norm > config.maximum_rotation_step_rad:
        rotation_delta *= config.maximum_rotation_step_rad / rotation_norm
    executed_action_delta = Rotation.from_euler("XYZ", rotation_delta).as_matrix()
    predicted_root_from_target = executed_action_delta.T @ root_from_target

    offset_local = np.array([0.0, 0.0, V4_TOOL_TCP_CENTER_OFFSET_M], dtype=np.float64)
    rotation_induced_center_delta_root = (
        predicted_root_from_target - root_from_target
    ) @ offset_local
    desired_center_delta_world = np.asarray(proposal.total_position_delta_world_m, dtype=np.float64)
    desired_center_delta_root = world_from_root.T @ desired_center_delta_world
    tcp_position_delta_root = desired_center_delta_root - rotation_induced_center_delta_root

    normalized_position = tcp_position_delta_root / config.action_position_scale_m
    normalized_rotation = rotation_delta / config.action_rotation_scale_rad
    action = np.concatenate((normalized_position, normalized_rotation, np.array([-1.0])))
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise GeometryTargetPoseError("target-delta action is invalid")
    if np.any(np.abs(action[:6]) > 1.0 + 1.0e-12):
        raise GeometryTargetPoseError("target-delta action exceeds normalized bounds")
    return action.astype(np.float32)
