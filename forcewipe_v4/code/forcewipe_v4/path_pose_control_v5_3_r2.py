"""V5.3-r2 first-pass path controller with bounded geometric re-entry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np

from .disturbances import reference_noise_offset_world
from .low_level_control import LowLevelControlError
from .low_level_control_v5_3 import ConditionalControllerState
from .low_level_control_v5_3_r2 import (
    ConditionalCausalNormalForceControllerV53R2,
    ConditionalForceControlCommandV53R2,
    ConditionalForceControllerConfigV53R2,
)
from .path_pose_control import (
    PathForcePoseCommand,
    PathPoseControllerConfig,
    desired_tool_rotation,
    root_aligned_rotation_delta_xyz,
)
from .scenarios import ScenarioPath, ScenarioSpec, surface_normal


class GeometricRecoveryPhaseV53R2(str, Enum):
    CONTACT_ACQUIRE = "CONTACT_ACQUIRE"
    CONFIRM = "CONFIRM"
    TRACK = "TRACK"
    LIFT = "LIFT"
    HOVER_REPOSITION = "HOVER_REPOSITION"


@dataclass(frozen=True)
class GeometricRecoveryConfigV53R2:
    recovery_timeout_samples: int = 50
    track_confirm_samples: int = 3
    release_force_n: float = 0.2
    hover_clearance_m: float = 0.035
    hover_position_tolerance_m: float = 0.004
    lift_step_m: float = 0.0035
    hover_reposition_step_m: float = 0.006

    def validate(self) -> None:
        counts = (self.recovery_timeout_samples, self.track_confirm_samples)
        if not all(isinstance(value, int) and not isinstance(value, bool) and value > 0 for value in counts):
            raise LowLevelControlError("geometric recovery counts must be positive integers")
        reals = (
            self.release_force_n,
            self.hover_clearance_m,
            self.hover_position_tolerance_m,
            self.lift_step_m,
            self.hover_reposition_step_m,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in reals):
            raise LowLevelControlError("geometric recovery parameters must be positive")
        if self.hover_position_tolerance_m >= self.hover_clearance_m:
            raise LowLevelControlError("hover tolerance must be below hover clearance")


@dataclass(frozen=True)
class PathForcePoseCommandV53R2(PathForcePoseCommand):
    geometric_phase: str
    geometric_transition_reason: str
    committed_progress: float
    instantaneous_progress: float


def _bounded(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float64).copy()
    norm = float(np.linalg.norm(result))
    if norm > float(maximum_norm):
        result *= float(maximum_norm) / norm
    return result


def _manual_command(
    *,
    normal_step_m: float,
    normal_xyz: np.ndarray,
    state: str,
    reason: str,
    target_force_n: float,
    controller_config: ConditionalForceControllerConfigV53R2,
) -> ConditionalForceControlCommandV53R2:
    normal = np.asarray(normal_xyz, dtype=np.float64)
    normalized = np.clip(
        (-normal * float(normal_step_m)) / controller_config.action_position_scale_m,
        -1.0,
        1.0,
    )
    severe, recovery_exit = (
        ConditionalCausalNormalForceControllerV53R2._thresholds(
            controller_config, float(target_force_n)
        )
    )
    return ConditionalForceControlCommandV53R2(
        normalized_position_action=normalized,
        normal_step_m=float(normal_step_m),
        force_error_n=0.0,
        force_rate_n_s=0.0,
        integral_error_n_s=0.0,
        mode=state.lower(),
        controller_state=state,
        state_transition_reason=reason,
        proportional_step_m=0.0,
        integral_step_m=0.0,
        derivative_step_m=0.0,
        raw_normal_step_m=float(normal_step_m),
        clipped_normal_step_m=float(normal_step_m),
        confirmation_count=0,
        tangential_motion_allowed=False,
        severe_drop_threshold_n=severe,
        recovery_exit_threshold_n=recovery_exit,
    )


class CausalPathForcePoseControllerV53R2:
    """Commit path progress only in TRACK and recover through hover geometry."""

    def __init__(
        self,
        *,
        force_config: ConditionalForceControllerConfigV53R2 = ConditionalForceControllerConfigV53R2(),
        path_config: PathPoseControllerConfig = PathPoseControllerConfig(),
        recovery_config: GeometricRecoveryConfigV53R2 = GeometricRecoveryConfigV53R2(),
    ) -> None:
        force_config.validate()
        path_config.validate()
        recovery_config.validate()
        self.force_config = force_config
        self.path_config = path_config
        self.recovery_config = recovery_config
        self.force_controller = ConditionalCausalNormalForceControllerV53R2(force_config)
        self.reset()

    @property
    def geometric_phase(self) -> GeometricRecoveryPhaseV53R2:
        return self._phase

    def reset(self) -> None:
        self.force_controller.reset()
        self._phase = GeometricRecoveryPhaseV53R2.CONTACT_ACQUIRE
        self._committed_progress = 0.0
        self._recovery_steps = 0
        self._track_confirmation_count = 0

    def _reference_geometry(
        self,
        *,
        progress: float,
        scenario: ScenarioSpec,
        path: ScenarioPath,
        surface_top_origin_z_m: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        relative, tangent = path.at(float(progress))
        point = relative.copy()
        point[2] += float(surface_top_origin_z_m)
        normal = np.asarray(surface_normal(scenario, float(relative[0])), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        point += reference_noise_offset_world(
            scenario,
            progress=float(progress),
            tangent_world=tangent,
            outward_normal_world=normal,
        )
        return point, tangent, normal

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
    ) -> PathForcePoseCommandV53R2:
        tool = np.asarray(tool_position_xyz_m, dtype=np.float64)
        if tool.shape != (3,) or not np.all(np.isfinite(tool)):
            raise LowLevelControlError("tool position must be a finite three-vector")
        relative_tool = tool.copy()
        relative_tool[2] -= float(surface_top_origin_z_m)
        projection = path.project(relative_tool)
        instantaneous_progress = float(projection.progress)
        transition_reason = "geometric_phase_hold"
        cfg = self.path_config
        recovery = self.recovery_config

        reference_progress = self._committed_progress
        target_point, tangent, normal = self._reference_geometry(
            progress=reference_progress,
            scenario=scenario,
            path=path,
            surface_top_origin_z_m=surface_top_origin_z_m,
        )
        planar_error = target_point - tool
        planar_error -= float(np.dot(planar_error, normal)) * normal

        tangent_step = np.zeros(3, dtype=np.float64)
        position_delta = np.zeros(3, dtype=np.float64)
        force_command: ConditionalForceControlCommandV53R2

        if self._phase is GeometricRecoveryPhaseV53R2.LIFT:
            clearance = float(np.dot(tool - target_point, normal))
            released = float(measured_force_n) <= recovery.release_force_n
            high_enough = clearance >= (
                recovery.hover_clearance_m - recovery.hover_position_tolerance_m
            )
            if released and high_enough:
                self._phase = GeometricRecoveryPhaseV53R2.HOVER_REPOSITION
                transition_reason = "lift_complete_to_hover_reposition"
                position_delta = np.zeros(3, dtype=np.float64)
                force_command = _manual_command(
                    normal_step_m=0.0,
                    normal_xyz=normal,
                    state=self._phase.value,
                    reason=transition_reason,
                    target_force_n=target_force_n,
                    controller_config=self.force_config,
                )
            else:
                position_delta = normal * recovery.lift_step_m
                force_command = _manual_command(
                    normal_step_m=-recovery.lift_step_m,
                    normal_xyz=normal,
                    state=self._phase.value,
                    reason="bounded_geometric_lift",
                    target_force_n=target_force_n,
                    controller_config=self.force_config,
                )

        elif self._phase is GeometricRecoveryPhaseV53R2.HOVER_REPOSITION:
            hover_target = target_point + normal * recovery.hover_clearance_m
            hover_error = hover_target - tool
            if float(np.linalg.norm(hover_error)) <= recovery.hover_position_tolerance_m:
                self._phase = GeometricRecoveryPhaseV53R2.CONTACT_ACQUIRE
                self.force_controller.reset()
                transition_reason = "hover_reposition_complete_to_contact_acquire"
                position_delta = np.zeros(3, dtype=np.float64)
            else:
                position_delta = _bounded(
                    hover_error, recovery.hover_reposition_step_m
                )
            force_command = _manual_command(
                normal_step_m=0.0,
                normal_xyz=normal,
                state=self._phase.value,
                reason=transition_reason,
                target_force_n=target_force_n,
                controller_config=self.force_config,
            )

        else:
            force_command = self.force_controller.command(
                measured_force_n=measured_force_n,
                target_force_n=target_force_n,
                outward_normal_xyz=normal,
            )
            controller_state = force_command.controller_state
            if controller_state == ConditionalControllerState.RECOVER.value:
                self._recovery_steps += 1
            else:
                self._recovery_steps = 0

            if self._recovery_steps >= recovery.recovery_timeout_samples:
                self._phase = GeometricRecoveryPhaseV53R2.LIFT
                self.force_controller.reset()
                self._track_confirmation_count = 0
                transition_reason = "recovery_timeout_to_lift"
                position_delta = normal * recovery.lift_step_m
                force_command = _manual_command(
                    normal_step_m=-recovery.lift_step_m,
                    normal_xyz=normal,
                    state=self._phase.value,
                    reason=transition_reason,
                    target_force_n=target_force_n,
                    controller_config=self.force_config,
                )
            elif self._phase is GeometricRecoveryPhaseV53R2.CONTACT_ACQUIRE:
                position_delta = -normal * force_command.normal_step_m
                if controller_state == ConditionalControllerState.TRACK.value:
                    self._phase = GeometricRecoveryPhaseV53R2.CONFIRM
                    self._track_confirmation_count = 1
                    transition_reason = "contact_acquire_to_confirm"
            elif self._phase is GeometricRecoveryPhaseV53R2.CONFIRM:
                position_delta = -normal * force_command.normal_step_m
                if (
                    controller_state == ConditionalControllerState.TRACK.value
                    and float(np.linalg.norm(planar_error))
                    <= recovery.hover_position_tolerance_m
                ):
                    self._track_confirmation_count += 1
                    if self._track_confirmation_count >= recovery.track_confirm_samples:
                        self._phase = GeometricRecoveryPhaseV53R2.TRACK
                        transition_reason = "confirm_complete_to_track"
                else:
                    self._track_confirmation_count = 0
                    if controller_state in {
                        ConditionalControllerState.ACQUIRE.value,
                        ConditionalControllerState.RECOVER.value,
                    }:
                        self._phase = GeometricRecoveryPhaseV53R2.CONTACT_ACQUIRE
                        transition_reason = "confirm_lost_to_contact_acquire"
            else:
                if controller_state == ConditionalControllerState.TRACK.value:
                    # Progress is committed only after the force controller and
                    # geometric supervisor both remain in confirmed TRACK.
                    self._committed_progress = max(
                        self._committed_progress, instantaneous_progress
                    )
                    advance = (
                        cfg.lookahead_m + cfg.tangential_speed_m_s * cfg.dt_s
                    ) / path.total_length
                    commanded_progress = min(1.0, self._committed_progress + advance)
                    target_point, tangent, normal = self._reference_geometry(
                        progress=commanded_progress,
                        scenario=scenario,
                        path=path,
                        surface_top_origin_z_m=surface_top_origin_z_m,
                    )
                    planar_error = target_point - tool
                    planar_error -= float(np.dot(planar_error, normal)) * normal
                    tangent_step = (
                        tangent * cfg.tangential_speed_m_s * cfg.dt_s
                        + cfg.tangent_error_gain * planar_error
                    )
                    tangent_step -= float(np.dot(tangent_step, normal)) * normal
                    tangent_step = _bounded(tangent_step, cfg.max_tangent_step_m)
                position_delta = tangent_step - normal * force_command.normal_step_m

        commanded_progress = self._committed_progress
        if self._phase is GeometricRecoveryPhaseV53R2.TRACK:
            commanded_progress = min(
                1.0,
                self._committed_progress
                + (cfg.lookahead_m + cfg.tangential_speed_m_s * cfg.dt_s)
                / path.total_length,
            )
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
        return PathForcePoseCommandV53R2(
            normalized_pose_action=pose_action,
            force_command=force_command,
            progress=float(self._committed_progress),
            commanded_progress=float(commanded_progress),
            target_point_xyz_m=target_point,
            tangent_xyz=tangent,
            outward_normal_xyz=normal,
            tangent_step_xyz_m=tangent_step,
            rotation_delta_xyz_rad=rotation_delta,
            solver_audit=None,
            geometric_phase=self._phase.value,
            geometric_transition_reason=transition_reason,
            committed_progress=float(self._committed_progress),
            instantaneous_progress=instantaneous_progress,
        )
