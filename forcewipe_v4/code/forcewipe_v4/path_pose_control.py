"""Causal surface-path and normal-force pose controller for V4 pilots."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from .disturbances import reference_noise_offset_world
from .low_level_control import (
    CausalNormalForceController,
    ForceControlCommand,
    ForceControllerConfig,
    LowLevelControlError,
)
from .low_level_baselines import BaselineControlOutput, SolverAudit
from .scenarios import ScenarioPath, ScenarioSpec, surface_normal


@dataclass(frozen=True)
class PathPoseControllerConfig:
    dt_s: float = 0.01
    tangential_speed_m_s: float = 0.025
    lookahead_m: float = 0.004
    tangent_error_gain: float = 0.35
    max_tangent_step_m: float = 0.0035
    path_motion_min_force_n: float = 3.0
    path_motion_headroom_force_n: float = 13.5
    max_rotation_step_rad: float = 0.05
    action_position_scale_m: float = 0.1
    action_rotation_scale_rad: float = 0.1

    def validate(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values())
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise LowLevelControlError("all path-pose controller parameters must be positive")
        if self.max_rotation_step_rad > self.action_rotation_scale_rad:
            raise LowLevelControlError("rotation step exceeds the pose-action scale")


@dataclass(frozen=True)
class PathForcePoseCommand:
    normalized_pose_action: np.ndarray
    force_command: ForceControlCommand
    progress: float
    commanded_progress: float
    target_point_xyz_m: np.ndarray
    tangent_xyz: np.ndarray
    outward_normal_xyz: np.ndarray
    tangent_step_xyz_m: np.ndarray
    rotation_delta_xyz_rad: np.ndarray
    solver_audit: SolverAudit | None


def quaternion_wxyz_to_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise LowLevelControlError("quaternion must be a finite wxyz four-vector")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise LowLevelControlError("quaternion must be nonzero")
    w, x, y, z = quaternion / norm
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def desired_tool_rotation(tangent_xyz: np.ndarray, outward_normal_xyz: np.ndarray) -> np.ndarray:
    """Return world-from-tool rotation with local +x tangent and +z inward."""

    tangent = np.asarray(tangent_xyz, dtype=np.float64)
    normal = np.asarray(outward_normal_xyz, dtype=np.float64)
    if tangent.shape != (3,) or normal.shape != (3,):
        raise LowLevelControlError("tangent and normal must be three-vectors")
    normal_norm = float(np.linalg.norm(normal))
    if normal_norm <= 0.0:
        raise LowLevelControlError("surface normal must be nonzero")
    normal = normal / normal_norm
    tangent = tangent - float(np.dot(tangent, normal)) * normal
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 0.0:
        raise LowLevelControlError("path tangent must not be parallel to the normal")
    x_axis = tangent / tangent_norm
    z_axis = -normal
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(rotation) < 0.999999:
        raise LowLevelControlError("desired tool frame is not right-handed")
    return rotation


def root_aligned_rotation_delta_xyz(
    current_quaternion_wxyz: np.ndarray,
    desired_world_from_tool: np.ndarray,
    *,
    max_step_rad: float,
) -> np.ndarray:
    """Return the delta action that reduces world-frame TCP orientation error.

    ManiSkill's Panda differential IK exposes the root-aligned rotation action
    with the inverse matrix convention relative to the world-frame SAPIEN pose
    returned by ``agent.tcp.pose``.  Consequently the action-space correction
    is ``R_current R_desired^T``.  A local one-step physics probe checks this
    sign before the controller is used in the path pilot.
    """

    current = quaternion_wxyz_to_matrix(current_quaternion_wxyz)
    desired = np.asarray(desired_world_from_tool, dtype=np.float64)
    if desired.shape != (3, 3) or not np.all(np.isfinite(desired)):
        raise LowLevelControlError("desired rotation must be a finite 3x3 matrix")
    delta = current @ desired.T
    euler_xyz = Rotation.from_matrix(delta).as_euler("XYZ", degrees=False)
    return np.clip(euler_xyz, -float(max_step_rad), float(max_step_rad))


class CausalPathForcePoseController:
    def __init__(
        self,
        *,
        force_config: ForceControllerConfig = ForceControllerConfig(),
        path_config: PathPoseControllerConfig = PathPoseControllerConfig(),
        normal_force_controller: object | None = None,
    ):
        path_config.validate()
        self.force_controller = (
            CausalNormalForceController(force_config)
            if normal_force_controller is None
            else normal_force_controller
        )
        if not callable(getattr(self.force_controller, "command", None)) or not callable(
            getattr(self.force_controller, "reset", None)
        ):
            raise LowLevelControlError(
                "normal force controller must expose reset() and command()"
            )
        self.path_config = path_config
        self.reset()

    def reset(self) -> None:
        self.force_controller.reset()
        self._best_progress = 0.0

    def command(
        self,
        *,
        measured_force_n: float,
        target_force_n: float,
        tool_position_xyz_m: np.ndarray,
        tcp_quaternion_wxyz: np.ndarray,
        scenario: ScenarioSpec,
        path: ScenarioPath,
        surface_top_origin_z_m: float,
    ) -> PathForcePoseCommand:
        tool_position = np.asarray(tool_position_xyz_m, dtype=np.float64)
        if tool_position.shape != (3,) or not np.all(np.isfinite(tool_position)):
            raise LowLevelControlError("tool position must be a finite three-vector")

        relative = tool_position.copy()
        relative[2] -= float(surface_top_origin_z_m)
        projection = path.project(relative)
        self._best_progress = max(self._best_progress, projection.progress)
        cfg = self.path_config
        advance = (cfg.lookahead_m + cfg.tangential_speed_m_s * cfg.dt_s) / path.total_length
        commanded_progress = min(1.0, self._best_progress + advance)
        target_relative, tangent = path.at(commanded_progress)
        target_point = target_relative.copy()
        target_point[2] += float(surface_top_origin_z_m)
        normal = np.asarray(surface_normal(scenario, target_relative[0]), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        target_point += reference_noise_offset_world(
            scenario,
            progress=commanded_progress,
            tangent_world=tangent,
            outward_normal_world=normal,
        )

        planar_error = target_point - tool_position
        planar_error -= float(np.dot(planar_error, normal)) * normal
        force_output = self.force_controller.command(
            measured_force_n=measured_force_n,
            target_force_n=target_force_n,
            outward_normal_xyz=normal,
        )
        if isinstance(force_output, BaselineControlOutput):
            force_command = force_output.command
            solver_audit = force_output.audit
        elif isinstance(force_output, ForceControlCommand):
            force_command = force_output
            solver_audit = None
        else:
            raise LowLevelControlError(
                "normal force controller returned an unsupported command type"
            )
        if (
            measured_force_n < cfg.path_motion_min_force_n
            or force_command.mode == "headroom_lift"
            or measured_force_n >= cfg.path_motion_headroom_force_n
            or not bool(getattr(force_command, "tangential_motion_allowed", True))
        ):
            tangent_step = np.zeros(3, dtype=np.float64)
        else:
            tangent_step = (
                tangent * cfg.tangential_speed_m_s * cfg.dt_s
                + cfg.tangent_error_gain * planar_error
            )
            tangent_step -= float(np.dot(tangent_step, normal)) * normal
            tangent_norm = float(np.linalg.norm(tangent_step))
            if tangent_norm > cfg.max_tangent_step_m:
                tangent_step *= cfg.max_tangent_step_m / tangent_norm

        normal_delta = -normal * force_command.normal_step_m
        position_delta = tangent_step + normal_delta
        normalized_position = np.clip(
            position_delta / cfg.action_position_scale_m,
            -1.0,
            1.0,
        )

        desired_rotation = desired_tool_rotation(tangent, normal)
        rotation_delta = root_aligned_rotation_delta_xyz(
            tcp_quaternion_wxyz,
            desired_rotation,
            max_step_rad=cfg.max_rotation_step_rad,
        )
        normalized_rotation = np.clip(
            rotation_delta / cfg.action_rotation_scale_rad,
            -1.0,
            1.0,
        )
        pose_action = np.concatenate((normalized_position, normalized_rotation))
        return PathForcePoseCommand(
            normalized_pose_action=pose_action,
            force_command=force_command,
            progress=self._best_progress,
            commanded_progress=commanded_progress,
            target_point_xyz_m=target_point,
            tangent_xyz=tangent,
            outward_normal_xyz=normal,
            tangent_step_xyz_m=tangent_step,
            rotation_delta_xyz_rad=rotation_delta,
            solver_audit=solver_audit,
        )
