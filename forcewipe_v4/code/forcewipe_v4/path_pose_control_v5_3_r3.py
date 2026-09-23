"""V5.3-r3 geometric supervisor and self-contained diagnostic log."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .low_level_control_v5_3 import ConditionalControllerState
from .path_pose_control import (
    desired_tool_rotation,
    root_aligned_rotation_delta_xyz,
)
from .path_pose_control_v5_3_r2 import (
    CausalPathForcePoseControllerV53R2,
    GeometricRecoveryPhaseV53R2,
    PathForcePoseCommandV53R2,
)


class CausalPathForcePoseControllerV53R3(CausalPathForcePoseControllerV53R2):
    """Enforce HEADROOM globally and commit progress only after confirmed TRACK."""

    def reset(self) -> None:
        super().reset()
        self._diagnostic_rows: list[dict[str, object]] = []

    @property
    def diagnostic_rows(self) -> tuple[dict[str, object], ...]:
        return tuple(self._diagnostic_rows)

    @staticmethod
    def _state_value(state) -> str | None:
        return None if state is None else str(getattr(state, "value", state))

    def _zero_tangent_and_restore_progress(
        self,
        command: PathForcePoseCommandV53R2,
        *,
        committed_progress: float,
        phase: GeometricRecoveryPhaseV53R2,
        reason: str,
    ) -> PathForcePoseCommandV53R2:
        normal_delta = -command.outward_normal_xyz * command.force_command.normal_step_m
        normalized_position = np.clip(
            normal_delta / self.path_config.action_position_scale_m,
            -1.0,
            1.0,
        )
        pose_action = np.concatenate(
            (normalized_position, command.normalized_pose_action[3:])
        )
        self._committed_progress = float(committed_progress)
        self._phase = phase
        return replace(
            command,
            normalized_pose_action=pose_action,
            progress=float(committed_progress),
            commanded_progress=float(committed_progress),
            tangent_step_xyz_m=np.zeros(3, dtype=np.float64),
            geometric_phase=phase.value,
            geometric_transition_reason=reason,
            committed_progress=float(committed_progress),
        )

    def _headroom_preemption(
        self,
        *,
        measured_force_n: float,
        target_force_n: float,
        tool_position_xyz_m: np.ndarray,
        tcp_quaternion_wxyz: np.ndarray | None,
        scenario,
        path,
        surface_top_origin_z_m: float,
    ) -> PathForcePoseCommandV53R2:
        tool = np.asarray(tool_position_xyz_m, dtype=np.float64)
        relative_tool = tool.copy()
        relative_tool[2] -= float(surface_top_origin_z_m)
        projection = path.project(relative_tool)
        target_point, tangent, normal = self._reference_geometry(
            progress=self._committed_progress,
            scenario=scenario,
            path=path,
            surface_top_origin_z_m=surface_top_origin_z_m,
        )
        force_command = self.force_controller.command(
            measured_force_n=measured_force_n,
            target_force_n=target_force_n,
            outward_normal_xyz=normal,
        )
        controller_state = force_command.controller_state
        if controller_state == ConditionalControllerState.HEADROOM.value:
            reason = f"headroom_preempts_{self._phase.value.lower()}"
            position_delta = -normal * force_command.normal_step_m
        else:
            # The HEADROOM exit-confirmation sample is a geometric hold.  The
            # original LIFT/HOVER phase may resume on the following sample.
            reason = f"headroom_exit_hold_{self._phase.value.lower()}"
            position_delta = np.zeros(3, dtype=np.float64)
        normalized_position = np.clip(
            position_delta / self.path_config.action_position_scale_m,
            -1.0,
            1.0,
        )
        if tcp_quaternion_wxyz is None:
            rotation_delta = np.zeros(3, dtype=np.float64)
        else:
            rotation_delta = root_aligned_rotation_delta_xyz(
                tcp_quaternion_wxyz,
                desired_tool_rotation(tangent, normal),
                max_step_rad=self.path_config.max_rotation_step_rad,
            )
        normalized_rotation = np.clip(
            rotation_delta / self.path_config.action_rotation_scale_rad,
            -1.0,
            1.0,
        )
        return PathForcePoseCommandV53R2(
            normalized_pose_action=np.concatenate(
                (normalized_position, normalized_rotation)
            ),
            force_command=force_command,
            progress=float(self._committed_progress),
            commanded_progress=float(self._committed_progress),
            target_point_xyz_m=target_point,
            tangent_xyz=tangent,
            outward_normal_xyz=normal,
            tangent_step_xyz_m=np.zeros(3, dtype=np.float64),
            rotation_delta_xyz_rad=rotation_delta,
            solver_audit=None,
            geometric_phase=self._phase.value,
            geometric_transition_reason=reason,
            committed_progress=float(self._committed_progress),
            instantaneous_progress=float(projection.progress),
        )

    def _append_diagnostic(
        self,
        command: PathForcePoseCommandV53R2,
        *,
        phase_before: str,
        force_state_before: str | None,
        orientation_input_available: bool,
    ) -> None:
        force = command.force_command
        actual_tangent = float(np.linalg.norm(command.tangent_step_xyz_m))
        permitted = bool(
            phase_before == GeometricRecoveryPhaseV53R2.TRACK.value
            and force_state_before == ConditionalControllerState.TRACK.value
            and command.geometric_phase == GeometricRecoveryPhaseV53R2.TRACK.value
            and force.controller_state == ConditionalControllerState.TRACK.value
        )
        self._diagnostic_rows.append(
            {
                "step": len(self._diagnostic_rows),
                "controller_state": force.controller_state,
                "controller_reason": force.state_transition_reason,
                "geometric_phase_before": phase_before,
                "geometric_phase": command.geometric_phase,
                "geometric_reason": command.geometric_transition_reason,
                "committed_progress": float(command.committed_progress),
                "instantaneous_progress": float(command.instantaneous_progress),
                "commanded_progress": float(command.commanded_progress),
                "recovery_count": int(self._recovery_steps),
                "geometric_confirmation_count": int(
                    self._track_confirmation_count
                ),
                "tangential_motion_permitted": permitted,
                "tangent_step_norm_m": actual_tangent,
                "orientation_input_available": orientation_input_available,
                "force_rate_n_s": float(force.force_rate_n_s),
                "integral_error_n_s": float(force.integral_error_n_s),
                "p_step_m": float(force.proportional_step_m),
                "i_step_m": float(force.integral_step_m),
                "d_step_m": float(force.derivative_step_m),
                "raw_normal_step_m": float(force.raw_normal_step_m),
                "clipped_normal_step_m": float(force.clipped_normal_step_m),
            }
        )

    def command(self, **kwargs) -> PathForcePoseCommandV53R2:
        phase_before = self._phase.value
        force_state_before = self._state_value(self.force_controller.state)
        progress_before = float(self._committed_progress)
        measured_force = float(kwargs["measured_force_n"])
        orientation_input_available = kwargs.get("tcp_quaternion_wxyz") is not None

        manual_phase = self._phase in {
            GeometricRecoveryPhaseV53R2.LIFT,
            GeometricRecoveryPhaseV53R2.HOVER_REPOSITION,
        }
        headroom_active = (
            measured_force >= self.force_config.headroom_enter_n
            or self.force_controller.state is ConditionalControllerState.HEADROOM
        )
        if manual_phase and headroom_active:
            command = self._headroom_preemption(**kwargs)
        else:
            # Historical replay artifacts archive causal force and position but
            # not the pre-action TCP quaternion.  Orientation is irrelevant to
            # the force/geometric state machine audited by replay.  A neutral
            # value is supplied only to the inherited orientation calculation,
            # after which rotational authority is explicitly removed.
            inherited_kwargs = dict(kwargs)
            if not orientation_input_available:
                inherited_kwargs["tcp_quaternion_wxyz"] = np.array(
                    [1.0, 0.0, 0.0, 0.0], dtype=np.float64
                )
            command = super().command(**inherited_kwargs)

        if not orientation_input_available:
            command = replace(
                command,
                normalized_pose_action=np.concatenate(
                    (command.normalized_pose_action[:3], np.zeros(3, dtype=np.float64))
                ),
                rotation_delta_xyz_rad=np.zeros(3, dtype=np.float64),
            )

        state_after = command.force_command.controller_state
        if phase_before == GeometricRecoveryPhaseV53R2.TRACK.value:
            if state_after != ConditionalControllerState.TRACK.value:
                self._track_confirmation_count = 0
                command = self._zero_tangent_and_restore_progress(
                    command,
                    committed_progress=progress_before,
                    phase=GeometricRecoveryPhaseV53R2.CONTACT_ACQUIRE,
                    reason="track_force_state_lost_to_contact_acquire",
                )
            elif force_state_before != ConditionalControllerState.TRACK.value:
                self._track_confirmation_count = 1
                command = self._zero_tangent_and_restore_progress(
                    command,
                    committed_progress=progress_before,
                    phase=GeometricRecoveryPhaseV53R2.CONFIRM,
                    reason="force_recovered_to_geometric_confirm",
                )

        # A transition into TRACK is a hold; only the following sample can
        # receive tangential authority and commit a new projection.
        if (
            phase_before != GeometricRecoveryPhaseV53R2.TRACK.value
            and command.geometric_phase == GeometricRecoveryPhaseV53R2.TRACK.value
        ):
            command = self._zero_tangent_and_restore_progress(
                command,
                committed_progress=progress_before,
                phase=GeometricRecoveryPhaseV53R2.TRACK,
                reason="geometric_track_transition_hold",
            )

        self._append_diagnostic(
            command,
            phase_before=phase_before,
            force_state_before=force_state_before,
            orientation_input_available=orientation_input_available,
        )
        return command
