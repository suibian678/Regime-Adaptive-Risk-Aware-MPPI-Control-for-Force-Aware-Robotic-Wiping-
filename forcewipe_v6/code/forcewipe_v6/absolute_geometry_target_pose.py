"""Absolute geometry reference for curved-surface target-pose control."""

from __future__ import annotations

import math

import numpy as np
from scipy.spatial.transform import Rotation

from forcewipe_v4.path_pose_control import quaternion_wxyz_to_matrix
from forcewipe_v4.scene_geometry import V4_TOOL_TCP_CENTER_OFFSET_M
from forcewipe_v6.geometry_target_pose import (
    GeometryTargetPoseConfig,
    GeometryTargetPoseError,
    GeometryTargetPoseProposal,
)


def proposed_normal_offset(
    current_offset_m: float,
    requested_increment_m: float,
    *,
    minimum_offset_m: float = -0.01,
    maximum_offset_m: float = 0.03,
) -> tuple[float, bool]:
    values = tuple(float(value) for value in (
        current_offset_m, requested_increment_m, minimum_offset_m, maximum_offset_m
    ))
    if not all(math.isfinite(value) for value in values):
        raise GeometryTargetPoseError("normal-offset state must be finite")
    current, increment, minimum, maximum = values
    if minimum >= maximum or not minimum <= current <= maximum:
        raise GeometryTargetPoseError("normal-offset state or bounds are invalid")
    raw = current + increment
    bounded = min(maximum, max(minimum, raw))
    return bounded, abs(bounded - raw) > 1.0e-15


def absolute_tool_center_target_world(
    proposal: GeometryTargetPoseProposal,
    *,
    initial_target_tool_center_world_m: np.ndarray,
    path_start_world_m: np.ndarray,
    normal_offset_after_m: float,
) -> np.ndarray:
    initial = np.asarray(initial_target_tool_center_world_m, dtype=np.float64)
    path_start = np.asarray(path_start_world_m, dtype=np.float64)
    path_after = np.asarray(proposal.path_point_after_world_m, dtype=np.float64)
    outward = np.asarray(proposal.committed_outward_normal_after_world, dtype=np.float64)
    if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in (initial, path_start, path_after, outward)):
        raise GeometryTargetPoseError("absolute geometry vectors are invalid")
    offset = float(normal_offset_after_m)
    if not math.isfinite(offset):
        raise GeometryTargetPoseError("normal offset is invalid")
    return initial + (path_after - path_start) - offset * outward


def absolute_geometry_target_delta_action(
    proposal: GeometryTargetPoseProposal,
    *,
    initial_target_tool_center_world_m: np.ndarray,
    path_start_world_m: np.ndarray,
    normal_offset_after_m: float,
    root_position_world_m: np.ndarray,
    root_quaternion_world_wxyz: np.ndarray,
    target_position_root_m: np.ndarray,
    target_quaternion_root_wxyz: np.ndarray,
    config: GeometryTargetPoseConfig = GeometryTargetPoseConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Return an action and its absolute desired tool-center target."""

    config.validate()
    root_position_world = np.asarray(root_position_world_m, dtype=np.float64)
    target_position_root = np.asarray(target_position_root_m, dtype=np.float64)
    if root_position_world.shape != (3,) or target_position_root.shape != (3,):
        raise GeometryTargetPoseError("root/target positions must be three-vectors")
    world_from_root = quaternion_wxyz_to_matrix(root_quaternion_world_wxyz)
    root_from_target = quaternion_wxyz_to_matrix(target_quaternion_root_wxyz)
    desired_world_from_tool = np.asarray(proposal.desired_world_from_tool_after, dtype=np.float64)
    desired_root_from_tool = world_from_root.T @ desired_world_from_tool
    action_space_delta = root_from_target @ desired_root_from_tool.T
    rotation_delta = Rotation.from_matrix(action_space_delta).as_euler("XYZ", degrees=False)
    rotation_norm = float(np.linalg.norm(rotation_delta))
    if rotation_norm > config.maximum_rotation_step_rad:
        rotation_delta *= config.maximum_rotation_step_rad / rotation_norm
    executed_action_delta = Rotation.from_euler("XYZ", rotation_delta).as_matrix()
    predicted_root_from_target = executed_action_delta.T @ root_from_target

    desired_center_world = absolute_tool_center_target_world(
        proposal,
        initial_target_tool_center_world_m=initial_target_tool_center_world_m,
        path_start_world_m=path_start_world_m,
        normal_offset_after_m=normal_offset_after_m,
    )
    desired_center_root = world_from_root.T @ (desired_center_world - root_position_world)
    local_offset = np.array([0.0, 0.0, V4_TOOL_TCP_CENTER_OFFSET_M], dtype=np.float64)
    desired_tcp_root = desired_center_root - predicted_root_from_target @ local_offset
    tcp_delta_root = desired_tcp_root - target_position_root
    normalized_position = tcp_delta_root / config.action_position_scale_m
    normalized_rotation = rotation_delta / config.action_rotation_scale_rad
    action = np.concatenate((normalized_position, normalized_rotation, np.array([-1.0])))
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise GeometryTargetPoseError("absolute target-delta action is invalid")
    if np.any(np.abs(action[:6]) > 1.0 + 1.0e-12):
        raise GeometryTargetPoseError("absolute target-delta action exceeds normalized bounds")
    return action.astype(np.float32), desired_center_world
