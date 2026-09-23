"""Revision-2 target-delta action mapping for the frozen geometry sequencer.

The R1 physics trace established that ManiSkill's root-aligned target-delta
rotation action updates the SAPIEN target matrix as ``D.T @ R_target``.  The
action-space correction must therefore be ``R_target @ R_desired.T``.  R1 used
the opposite order and is preserved unchanged for provenance.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from forcewipe_v4.path_pose_control import quaternion_wxyz_to_matrix
from forcewipe_v6.geometry_target_pose import (
    GeometryTargetPoseConfig,
    GeometryTargetPoseError,
    GeometryTargetPoseProposal,
)


def proposal_to_target_delta_action_r2(
    proposal: GeometryTargetPoseProposal,
    *,
    root_quaternion_world_wxyz: np.ndarray,
    target_quaternion_root_wxyz: np.ndarray,
    config: GeometryTargetPoseConfig = GeometryTargetPoseConfig(),
) -> np.ndarray:
    """Map a proposal using the physics-verified inverse action convention."""

    config.validate()
    world_from_root = quaternion_wxyz_to_matrix(root_quaternion_world_wxyz)
    root_from_target = quaternion_wxyz_to_matrix(target_quaternion_root_wxyz)
    desired_world_from_tool = np.asarray(proposal.desired_world_from_tool_after, dtype=np.float64)
    if desired_world_from_tool.shape != (3, 3) or not np.all(np.isfinite(desired_world_from_tool)):
        raise GeometryTargetPoseError("desired tool rotation is invalid")
    desired_root_from_tool = world_from_root.T @ desired_world_from_tool

    # The target controller applies the action-space rotation D as D.T in the
    # SAPIEN matrix convention.  Setting D = R_target R_desired.T therefore
    # moves the target toward R_desired rather than away from it.
    action_space_delta = root_from_target @ desired_root_from_tool.T
    rotation_delta = Rotation.from_matrix(action_space_delta).as_euler("XYZ", degrees=False)
    rotation_norm = float(np.linalg.norm(rotation_delta))
    if rotation_norm > config.maximum_rotation_step_rad:
        rotation_delta *= config.maximum_rotation_step_rad / rotation_norm

    world_position_delta = np.asarray(proposal.total_position_delta_world_m, dtype=np.float64)
    root_position_delta = world_from_root.T @ world_position_delta
    normalized_position = root_position_delta / config.action_position_scale_m
    normalized_rotation = rotation_delta / config.action_rotation_scale_rad
    action = np.concatenate((normalized_position, normalized_rotation, np.array([-1.0])))
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise GeometryTargetPoseError("target-delta action is invalid")
    if np.any(np.abs(action[:6]) > 1.0 + 1.0e-12):
        raise GeometryTargetPoseError("target-delta action exceeds normalized bounds")
    return action.astype(np.float32)
