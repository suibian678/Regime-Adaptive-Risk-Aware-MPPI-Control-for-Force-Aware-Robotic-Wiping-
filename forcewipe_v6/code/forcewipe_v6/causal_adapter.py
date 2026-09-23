"""Physics-agnostic causal adapter for the ForceWipe V6 supervisor.

The future simulator binding owns raw sensor acquisition and actuator
execution.  This module owns causal ordering, frame/sign semantics, complete
state provenance, and the all-or-nothing in-memory evidence transaction.

Positive normal commands point inward.  ``command_outward_normal_xyz`` is the
instantaneous actuation normal, whereas
``committed_recovery_outward_normal_xyz`` anchors the recovery hover target.
The Cartesian normal component is ``-u_n * n_out``.  Rotation deltas use
ManiSkill Euler XYZ; no rotvec field exists at this boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import math
import numbers
from typing import Any

from .controller import (
    SupervisorCommand,
    SupervisorInput,
    SupervisorSnapshot,
    SupervisorState,
    UnifiedCausalForceRecoverySupervisor,
)


class V6AdapterError(RuntimeError):
    """Base error for causal-adapter contract failures."""


class V6AdapterContractError(V6AdapterError):
    """The adapter boundary received inconsistent evidence."""


class V6AdapterExecutionError(V6AdapterError):
    """The command executor failed before a closed bundle existed."""


def _integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if not isinstance(value, numbers.Integral) or isinstance(value, bool):
        raise V6AdapterContractError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise V6AdapterContractError(f"{name} must be at least {minimum}")
    return result


def _real(value: Any, *, name: str, minimum: float | None = None) -> float:
    if not isinstance(value, numbers.Real) or isinstance(value, bool):
        raise V6AdapterContractError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise V6AdapterContractError(f"{name} must be finite")
    if minimum is not None and result < minimum:
        raise V6AdapterContractError(f"{name} must be at least {minimum}")
    return result


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise V6AdapterContractError(f"{name} must be a nonempty string")
    return value


def _vector(value: Any, *, name: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, (tuple, list)) or len(value) != length:
        raise V6AdapterContractError(f"{name} must be a length-{length} vector")
    return tuple(_real(item, name=f"{name}[{index}]") for index, item in enumerate(value))


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return float(sum(a * b for a, b in zip(left, right)))


def _norm(value: tuple[float, ...]) -> float:
    return float(math.sqrt(_dot(value, value)))


def _add(*values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(sum(items) for items in zip(*values))


def _sub(left: tuple[float, ...], right: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(a - b for a, b in zip(left, right))


def _scale(value: tuple[float, ...], scalar: float) -> tuple[float, ...]:
    return tuple(scalar * item for item in value)


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _quat_multiply(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return (
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    )


def _quat_rotate(
    quaternion: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    conjugate = (quaternion[0], -quaternion[1], -quaternion[2], -quaternion[3])
    rotated = _quat_multiply(
        _quat_multiply(quaternion, (0.0, *vector)), conjugate
    )
    return (rotated[1], rotated[2], rotated[3])


def _close(left: float, right: float, tolerance: float) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def _vector_close(
    left: tuple[float, ...], right: tuple[float, ...], tolerance: float
) -> bool:
    return len(left) == len(right) and all(
        _close(a, b, tolerance) for a, b in zip(left, right)
    )


def _quaternion_close(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    # q and -q are the same orientation.
    return _vector_close(left, right, tolerance) or _vector_close(
        left, tuple(-item for item in right), tolerance
    )


@dataclass(frozen=True, order=True)
class ControlStepKey:
    run_id: str
    scenario_id: int
    evaluation_id: int
    segment_index: int
    control_step_index: int

    def validate(self) -> None:
        _text(self.run_id, name="run_id")
        _integer(self.scenario_id, name="scenario_id")
        _integer(self.evaluation_id, name="evaluation_id")
        _integer(self.segment_index, name="segment_index")
        _integer(self.control_step_index, name="control_step_index")

    @property
    def stream_key(self) -> tuple[str, int, int, int]:
        self.validate()
        return (
            self.run_id,
            int(self.scenario_id),
            int(self.evaluation_id),
            int(self.segment_index),
        )

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class CausalPreStepSample:
    key: ControlStepKey
    pre_time_ns: int
    state_source_kind: str
    force_source_native_sample_index: int | None
    force_source_time_ns: int
    bootstrap_id: str
    measured_force_n: float
    target_force_n: float
    normal_velocity_outward_m_s: float
    contact_observed: bool
    instantaneous_progress: float
    geometric_track_error_m: float
    requested_tangential_step_m: float
    recovery_hover_pose_error_m: float
    recovery_clearance_m: float
    geometry_frame_id: str
    command_frame_id: str
    contact_tool_frame_id: str
    robot_tcp_frame_id: str
    task_path_id: str
    surface_id: str
    surface_reference_point_xyz_m: tuple[float, float, float]
    projection_outward_normal_xyz: tuple[float, float, float]
    command_outward_normal_xyz: tuple[float, float, float]
    task_projection_point_xyz_m: tuple[float, float, float]
    task_projection_progress: float
    task_projection_arc_length_m: float
    projection_tangent_unit_xyz: tuple[float, float, float]
    command_tangent_unit_xyz: tuple[float, float, float]
    task_path_length_m: float
    recovery_hover_target_position_xyz_m: tuple[float, float, float]
    requested_recovery_delta_xyz_m: tuple[float, float, float]
    requested_recovery_rotation_delta_euler_xyz_rad: tuple[float, float, float]
    contact_tool_position_xyz_m: tuple[float, float, float]
    contact_tool_quaternion_wxyz: tuple[float, float, float, float]
    contact_tool_linear_velocity_xyz_m_s: tuple[float, float, float]
    contact_tool_angular_velocity_xyz_rad_s: tuple[float, float, float]
    robot_tcp_position_xyz_m: tuple[float, float, float]
    robot_tcp_quaternion_wxyz: tuple[float, float, float, float]
    robot_tcp_linear_velocity_xyz_m_s: tuple[float, float, float]
    robot_tcp_angular_velocity_xyz_rad_s: tuple[float, float, float]
    # Recovery geometry remains anchored at committed progress.  Normal and
    # tangent actuation may instead use the instantaneous projection frame.
    committed_recovery_outward_normal_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)
    # Independent TRACK correction request; never aliases recovery authority.
    requested_cross_track_correction_delta_xyz_m: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )

    def validate(self) -> None:
        self.key.validate()
        _integer(self.pre_time_ns, name="pre_time_ns")
        _text(self.state_source_kind, name="state_source_kind")
        if self.state_source_kind not in {
            "bootstrap_sensor_snapshot",
            "prior_native_sample",
        }:
            raise V6AdapterContractError("state_source_kind is unsupported")
        _integer(self.force_source_time_ns, name="force_source_time_ns")
        if self.force_source_native_sample_index is not None:
            _integer(
                self.force_source_native_sample_index,
                name="force_source_native_sample_index",
            )
        if self.force_source_time_ns > self.pre_time_ns:
            raise V6AdapterContractError("state source cannot be newer than pre-step time")
        _text(self.bootstrap_id, name="bootstrap_id")
        _real(self.measured_force_n, name="measured_force_n", minimum=0.0)
        target = _real(self.target_force_n, name="target_force_n", minimum=0.0)
        if target <= 0.0:
            raise V6AdapterContractError("target_force_n must be positive")
        _real(self.normal_velocity_outward_m_s, name="normal_velocity_outward_m_s")
        if not isinstance(self.contact_observed, bool):
            raise V6AdapterContractError("contact_observed must be boolean")
        progress = _real(self.instantaneous_progress, name="instantaneous_progress")
        if not 0.0 <= progress <= 1.0:
            raise V6AdapterContractError("instantaneous_progress must lie in [0,1]")
        for name in (
            "geometric_track_error_m",
            "requested_tangential_step_m",
            "recovery_hover_pose_error_m",
            "recovery_clearance_m",
        ):
            _real(getattr(self, name), name=name, minimum=0.0)
        _text(self.geometry_frame_id, name="geometry_frame_id")
        _text(self.command_frame_id, name="command_frame_id")
        _text(self.contact_tool_frame_id, name="contact_tool_frame_id")
        _text(self.robot_tcp_frame_id, name="robot_tcp_frame_id")
        _text(self.task_path_id, name="task_path_id")
        _text(self.surface_id, name="surface_id")
        _real(self.task_path_length_m, name="task_path_length_m", minimum=0.0)
        if self.task_path_length_m <= 0.0:
            raise V6AdapterContractError("task_path_length_m must be positive")
        projection_progress = _real(
            self.task_projection_progress, name="task_projection_progress"
        )
        if not 0.0 <= projection_progress <= 1.0:
            raise V6AdapterContractError("task_projection_progress must lie in [0,1]")
        _real(
            self.task_projection_arc_length_m,
            name="task_projection_arc_length_m",
            minimum=0.0,
        )
        for name in (
            "surface_reference_point_xyz_m",
            "projection_outward_normal_xyz",
            "command_outward_normal_xyz",
            "committed_recovery_outward_normal_xyz",
            "task_projection_point_xyz_m",
            "projection_tangent_unit_xyz",
            "command_tangent_unit_xyz",
            "recovery_hover_target_position_xyz_m",
            "requested_recovery_delta_xyz_m",
            "requested_cross_track_correction_delta_xyz_m",
            "requested_recovery_rotation_delta_euler_xyz_rad",
            "contact_tool_position_xyz_m",
            "contact_tool_linear_velocity_xyz_m_s",
            "contact_tool_angular_velocity_xyz_rad_s",
            "robot_tcp_position_xyz_m",
            "robot_tcp_linear_velocity_xyz_m_s",
            "robot_tcp_angular_velocity_xyz_rad_s",
        ):
            _vector(getattr(self, name), name=name, length=3)
        for name in (
            "contact_tool_quaternion_wxyz",
            "robot_tcp_quaternion_wxyz",
        ):
            quaternion = _vector(getattr(self, name), name=name, length=4)
            if _norm(quaternion) <= 0.0:
                raise V6AdapterContractError(f"{name} must have nonzero norm")

    def to_supervisor_input(self) -> SupervisorInput:
        self.validate()
        return SupervisorInput(
            measured_force_n=float(self.measured_force_n),
            target_force_n=float(self.target_force_n),
            normal_velocity_outward_m_s=float(self.normal_velocity_outward_m_s),
            contact_observed=self.contact_observed,
            instantaneous_progress=float(self.instantaneous_progress),
            geometric_track_error_m=float(self.geometric_track_error_m),
            requested_tangential_step_m=float(self.requested_tangential_step_m),
            recovery_hover_pose_error_m=float(self.recovery_hover_pose_error_m),
            recovery_clearance_m=float(self.recovery_clearance_m),
        )

    def to_log_row(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.key.as_dict(),
            **{key: value for key, value in asdict(self).items() if key != "key"},
        }


@dataclass(frozen=True)
class IssuedCartesianCommand:
    key: ControlStepKey
    command_id: str
    issued_time_ns: int
    command_frame_id: str
    command_mode: str
    normal_step_m: float
    task_tangential_step_m: float
    recovery_reposition_permitted: bool
    normal_delta_xyz_m: tuple[float, float, float]
    task_delta_xyz_m: tuple[float, float, float]
    recovery_delta_xyz_m: tuple[float, float, float]
    cartesian_delta_xyz_m: tuple[float, float, float]
    recovery_rotation_delta_euler_xyz_rad: tuple[float, float, float]
    rotation_delta_euler_xyz_rad: tuple[float, float, float]
    cross_track_correction_permitted: bool = False
    cross_track_correction_delta_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        self.key.validate()
        _text(self.command_id, name="command_id")
        _integer(self.issued_time_ns, name="issued_time_ns")
        _text(self.command_frame_id, name="command_frame_id")
        _text(self.command_mode, name="command_mode")
        _real(self.normal_step_m, name="normal_step_m")
        _real(self.task_tangential_step_m, name="task_tangential_step_m", minimum=0.0)
        if not isinstance(self.recovery_reposition_permitted, bool):
            raise V6AdapterContractError("recovery_reposition_permitted must be boolean")
        if not isinstance(self.cross_track_correction_permitted, bool):
            raise V6AdapterContractError(
                "cross_track_correction_permitted must be boolean"
            )
        for name in (
            "normal_delta_xyz_m",
            "task_delta_xyz_m",
            "recovery_delta_xyz_m",
            "cross_track_correction_delta_xyz_m",
            "cartesian_delta_xyz_m",
            "recovery_rotation_delta_euler_xyz_rad",
            "rotation_delta_euler_xyz_rad",
        ):
            _vector(getattr(self, name), name=name, length=3)

    def to_log_row(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.key.as_dict(),
            **{key: value for key, value in asdict(self).items() if key != "key"},
        }


@dataclass(frozen=True)
class CommandReadback:
    key: ControlStepKey
    command_id: str
    readback_time_ns: int
    accepted: bool
    readback_stage: str
    command_frame_id: str
    command_mode: str
    normal_step_m: float
    task_tangential_step_m: float
    recovery_reposition_permitted: bool
    normal_delta_xyz_m: tuple[float, float, float]
    task_delta_xyz_m: tuple[float, float, float]
    recovery_delta_xyz_m: tuple[float, float, float]
    cartesian_delta_xyz_m: tuple[float, float, float]
    recovery_rotation_delta_euler_xyz_rad: tuple[float, float, float]
    rotation_delta_euler_xyz_rad: tuple[float, float, float]
    cross_track_correction_permitted: bool = False
    cross_track_correction_delta_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def validate(self) -> None:
        self.key.validate()
        _text(self.command_id, name="command_id")
        _integer(self.readback_time_ns, name="readback_time_ns")
        if not isinstance(self.accepted, bool):
            raise V6AdapterContractError("accepted must be boolean")
        _text(self.readback_stage, name="readback_stage")
        _text(self.command_frame_id, name="readback command_frame_id")
        _text(self.command_mode, name="readback command_mode")
        _real(self.normal_step_m, name="readback normal_step_m")
        _real(
            self.task_tangential_step_m,
            name="readback task_tangential_step_m",
            minimum=0.0,
        )
        if not isinstance(self.recovery_reposition_permitted, bool):
            raise V6AdapterContractError(
                "readback recovery_reposition_permitted must be boolean"
            )
        if not isinstance(self.cross_track_correction_permitted, bool):
            raise V6AdapterContractError(
                "readback cross_track_correction_permitted must be boolean"
            )
        for name in (
            "normal_delta_xyz_m",
            "task_delta_xyz_m",
            "recovery_delta_xyz_m",
            "cross_track_correction_delta_xyz_m",
            "cartesian_delta_xyz_m",
            "recovery_rotation_delta_euler_xyz_rad",
            "rotation_delta_euler_xyz_rad",
        ):
            _vector(getattr(self, name), name=f"readback {name}", length=3)

    def to_log_row(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.key.as_dict(),
            **{key: value for key, value in asdict(self).items() if key != "key"},
        }


@dataclass(frozen=True)
class NativeAuditSample:
    key: ControlStepKey
    command_id: str
    native_sample_index: int
    substep_index: int
    time_ns: int
    audit_force_n: float
    contact_observed: bool
    normal_velocity_outward_m_s: float
    instantaneous_progress: float
    geometric_track_error_m: float
    recovery_hover_pose_error_m: float
    recovery_clearance_m: float
    surface_reference_point_xyz_m: tuple[float, float, float]
    projection_outward_normal_xyz: tuple[float, float, float]
    command_outward_normal_xyz: tuple[float, float, float]
    task_projection_point_xyz_m: tuple[float, float, float]
    task_projection_progress: float
    task_projection_arc_length_m: float
    projection_tangent_unit_xyz: tuple[float, float, float]
    command_tangent_unit_xyz: tuple[float, float, float]
    recovery_hover_target_position_xyz_m: tuple[float, float, float]
    contact_tool_position_xyz_m: tuple[float, float, float]
    contact_tool_quaternion_wxyz: tuple[float, float, float, float]
    contact_tool_linear_velocity_xyz_m_s: tuple[float, float, float]
    contact_tool_angular_velocity_xyz_rad_s: tuple[float, float, float]
    robot_tcp_position_xyz_m: tuple[float, float, float]
    robot_tcp_quaternion_wxyz: tuple[float, float, float, float]
    robot_tcp_linear_velocity_xyz_m_s: tuple[float, float, float]
    robot_tcp_angular_velocity_xyz_rad_s: tuple[float, float, float]
    committed_recovery_outward_normal_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def validate(self) -> None:
        self.key.validate()
        _text(self.command_id, name="command_id")
        _integer(self.native_sample_index, name="native_sample_index")
        _integer(self.substep_index, name="substep_index")
        _integer(self.time_ns, name="native time_ns")
        _real(self.audit_force_n, name="audit_force_n", minimum=0.0)
        if not isinstance(self.contact_observed, bool):
            raise V6AdapterContractError("native contact_observed must be boolean")
        _real(self.normal_velocity_outward_m_s, name="native normal velocity")
        progress = _real(self.instantaneous_progress, name="native progress")
        if not 0.0 <= progress <= 1.0:
            raise V6AdapterContractError("native progress must lie in [0,1]")
        for name in (
            "geometric_track_error_m",
            "recovery_hover_pose_error_m",
            "recovery_clearance_m",
        ):
            _real(getattr(self, name), name=f"native {name}", minimum=0.0)
        projection_progress = _real(
            self.task_projection_progress, name="native task_projection_progress"
        )
        if not 0.0 <= projection_progress <= 1.0:
            raise V6AdapterContractError("native task projection progress must lie in [0,1]")
        _real(
            self.task_projection_arc_length_m,
            name="native task_projection_arc_length_m",
            minimum=0.0,
        )
        for name in (
            "surface_reference_point_xyz_m",
            "projection_outward_normal_xyz",
            "command_outward_normal_xyz",
            "committed_recovery_outward_normal_xyz",
            "task_projection_point_xyz_m",
            "projection_tangent_unit_xyz",
            "command_tangent_unit_xyz",
            "recovery_hover_target_position_xyz_m",
            "contact_tool_position_xyz_m",
            "contact_tool_linear_velocity_xyz_m_s",
            "contact_tool_angular_velocity_xyz_rad_s",
            "robot_tcp_position_xyz_m",
            "robot_tcp_linear_velocity_xyz_m_s",
            "robot_tcp_angular_velocity_xyz_rad_s",
        ):
            _vector(getattr(self, name), name=f"native {name}", length=3)
        for name in ("contact_tool_quaternion_wxyz", "robot_tcp_quaternion_wxyz"):
            quaternion = _vector(getattr(self, name), name=f"native {name}", length=4)
            if _norm(quaternion) <= 0.0:
                raise V6AdapterContractError(f"native {name} must have nonzero norm")

    def to_log_row(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.key.as_dict(),
            **{key: value for key, value in asdict(self).items() if key != "key"},
        }


@dataclass(frozen=True)
class CausalPostStepSample:
    key: ControlStepKey
    post_time_ns: int
    last_native_sample_index: int
    last_native_time_ns: int
    measured_force_n: float
    contact_observed: bool
    normal_velocity_outward_m_s: float
    instantaneous_progress: float
    geometric_track_error_m: float
    recovery_hover_pose_error_m: float
    recovery_clearance_m: float
    surface_reference_point_xyz_m: tuple[float, float, float]
    projection_outward_normal_xyz: tuple[float, float, float]
    command_outward_normal_xyz: tuple[float, float, float]
    task_projection_point_xyz_m: tuple[float, float, float]
    task_projection_progress: float
    task_projection_arc_length_m: float
    projection_tangent_unit_xyz: tuple[float, float, float]
    command_tangent_unit_xyz: tuple[float, float, float]
    recovery_hover_target_position_xyz_m: tuple[float, float, float]
    contact_tool_position_xyz_m: tuple[float, float, float]
    contact_tool_quaternion_wxyz: tuple[float, float, float, float]
    contact_tool_linear_velocity_xyz_m_s: tuple[float, float, float]
    contact_tool_angular_velocity_xyz_rad_s: tuple[float, float, float]
    robot_tcp_position_xyz_m: tuple[float, float, float]
    robot_tcp_quaternion_wxyz: tuple[float, float, float, float]
    robot_tcp_linear_velocity_xyz_m_s: tuple[float, float, float]
    robot_tcp_angular_velocity_xyz_rad_s: tuple[float, float, float]
    committed_recovery_outward_normal_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def validate(self) -> None:
        self.key.validate()
        _integer(self.post_time_ns, name="post_time_ns")
        _integer(self.last_native_sample_index, name="last_native_sample_index")
        _integer(self.last_native_time_ns, name="last_native_time_ns")
        if self.last_native_time_ns > self.post_time_ns:
            raise V6AdapterContractError("last native time cannot follow post-step time")
        _real(self.measured_force_n, name="post measured_force_n", minimum=0.0)
        if not isinstance(self.contact_observed, bool):
            raise V6AdapterContractError("post contact_observed must be boolean")
        _real(self.normal_velocity_outward_m_s, name="post normal velocity")
        progress = _real(self.instantaneous_progress, name="post progress")
        if not 0.0 <= progress <= 1.0:
            raise V6AdapterContractError("post progress must lie in [0,1]")
        for name in (
            "geometric_track_error_m",
            "recovery_hover_pose_error_m",
            "recovery_clearance_m",
        ):
            _real(getattr(self, name), name=f"post {name}", minimum=0.0)
        projection_progress = _real(
            self.task_projection_progress, name="post task_projection_progress"
        )
        if not 0.0 <= projection_progress <= 1.0:
            raise V6AdapterContractError("post task projection progress must lie in [0,1]")
        _real(
            self.task_projection_arc_length_m,
            name="post task_projection_arc_length_m",
            minimum=0.0,
        )
        for name in (
            "surface_reference_point_xyz_m",
            "projection_outward_normal_xyz",
            "command_outward_normal_xyz",
            "committed_recovery_outward_normal_xyz",
            "task_projection_point_xyz_m",
            "projection_tangent_unit_xyz",
            "command_tangent_unit_xyz",
            "recovery_hover_target_position_xyz_m",
            "contact_tool_position_xyz_m",
            "contact_tool_linear_velocity_xyz_m_s",
            "contact_tool_angular_velocity_xyz_rad_s",
            "robot_tcp_position_xyz_m",
            "robot_tcp_linear_velocity_xyz_m_s",
            "robot_tcp_angular_velocity_xyz_rad_s",
        ):
            _vector(getattr(self, name), name=f"post {name}", length=3)
        for name in ("contact_tool_quaternion_wxyz", "robot_tcp_quaternion_wxyz"):
            quaternion = _vector(getattr(self, name), name=f"post {name}", length=4)
            if _norm(quaternion) <= 0.0:
                raise V6AdapterContractError(f"post {name} must have nonzero norm")

    def to_log_row(self) -> dict[str, Any]:
        self.validate()
        return {
            **self.key.as_dict(),
            **{key: value for key, value in asdict(self).items() if key != "key"},
        }


@dataclass(frozen=True)
class BackendStepResult:
    readback: CommandReadback
    native_samples: tuple[NativeAuditSample, ...]
    post: CausalPostStepSample


@dataclass(frozen=True)
class AdapterConfig:
    expected_native_steps_per_control: int = 5
    control_period_ns: int = 10_000_000
    native_period_ns: int | None = None
    command_readback_tolerance_m: float = 1e-9
    rotation_readback_tolerance_rad: float = 1e-9
    force_alignment_tolerance_n: float = 1e-9
    state_alignment_tolerance: float = 1e-9
    basis_tolerance: float = 1e-9
    quaternion_tolerance: float = 1e-9
    maximum_task_tangential_step_m: float = 0.0035
    maximum_recovery_reposition_step_m: float = 0.006
    maximum_recovery_rotation_step_rad: float = 0.25
    recovery_hover_clearance_m: float = 0.01
    cross_track_correction_enter_m: float = 0.003
    maximum_cross_track_correction_step_m: float = 0.0005
    expected_command_frame_id: str = "world"
    expected_command_mode: str = "delta_pose_euler_xyz"
    required_readback_stage: str = "actuator_target_after_limits"

    def __post_init__(self) -> None:
        if self.native_period_ns is None:
            count = _integer(
                self.expected_native_steps_per_control,
                name="expected_native_steps_per_control",
                minimum=1,
            )
            control_period = _integer(
                self.control_period_ns, name="control_period_ns", minimum=1
            )
            if control_period % count != 0:
                raise V6AdapterContractError(
                    "control period must divide exactly by the native ratio"
                )
            object.__setattr__(self, "native_period_ns", control_period // count)

    def validate(self) -> None:
        count = _integer(
            self.expected_native_steps_per_control,
            name="expected_native_steps_per_control",
            minimum=1,
        )
        control_period = _integer(self.control_period_ns, name="control_period_ns", minimum=1)
        native_period = _integer(self.native_period_ns, name="native_period_ns", minimum=1)
        if count * native_period != control_period:
            raise V6AdapterContractError(
                "control_period_ns must equal native ratio times native_period_ns"
            )
        for name in (
            "command_readback_tolerance_m",
            "rotation_readback_tolerance_rad",
            "force_alignment_tolerance_n",
            "state_alignment_tolerance",
            "basis_tolerance",
            "quaternion_tolerance",
            "maximum_task_tangential_step_m",
            "maximum_recovery_reposition_step_m",
            "maximum_recovery_rotation_step_rad",
            "recovery_hover_clearance_m",
            "cross_track_correction_enter_m",
            "maximum_cross_track_correction_step_m",
        ):
            _real(getattr(self, name), name=name, minimum=0.0)
        _text(self.expected_command_frame_id, name="expected_command_frame_id")
        _text(self.expected_command_mode, name="expected_command_mode")
        _text(self.required_readback_stage, name="required_readback_stage")


@dataclass(frozen=True)
class AdapterStepBundle:
    pre: CausalPreStepSample
    supervisor_command: SupervisorCommand
    issued: IssuedCartesianCommand
    readback: CommandReadback
    native_samples: tuple[NativeAuditSample, ...]
    post: CausalPostStepSample

    @property
    def key(self) -> ControlStepKey:
        return self.pre.key


@dataclass(frozen=True)
class _LedgerSnapshot:
    bundles: tuple[AdapterStepBundle, ...]
    last_by_stream: tuple[tuple[tuple[str, int, int, int], AdapterStepBundle], ...]
    keys: frozenset[ControlStepKey]
    command_ids: frozenset[str]


class CausalEvidenceLedger:
    """In-memory transaction boundary and primary-key closure checker."""

    CONTROL_TABLES = (
        "causal_input",
        "supervisor_command",
        "issued_command",
        "command_readback",
        "post_step",
    )

    def __init__(self, config: AdapterConfig = AdapterConfig()) -> None:
        config.validate()
        self.config = config
        self._bundles: list[AdapterStepBundle] = []
        self._last_by_stream: dict[tuple[str, int, int, int], AdapterStepBundle] = {}
        self._keys: set[ControlStepKey] = set()
        self._command_ids: set[str] = set()

    @property
    def bundles(self) -> tuple[AdapterStepBundle, ...]:
        return tuple(self._bundles)

    def _snapshot(self) -> _LedgerSnapshot:
        return _LedgerSnapshot(
            bundles=tuple(self._bundles),
            last_by_stream=tuple(sorted(self._last_by_stream.items())),
            keys=frozenset(self._keys),
            command_ids=frozenset(self._command_ids),
        )

    def _restore(self, snapshot: _LedgerSnapshot) -> None:
        if not isinstance(snapshot, _LedgerSnapshot):
            raise V6AdapterContractError("unsupported ledger snapshot")
        self._bundles = list(snapshot.bundles)
        self._last_by_stream = dict(snapshot.last_by_stream)
        self._keys = set(snapshot.keys)
        self._command_ids = set(snapshot.command_ids)

    def _is_pristine(self) -> bool:
        return self._snapshot() == _LedgerSnapshot((), (), frozenset(), frozenset())

    def prior_for(self, key: ControlStepKey) -> AdapterStepBundle | None:
        return self._last_by_stream.get(key.stream_key)

    def _stream_bundles(self, key: ControlStepKey) -> tuple[AdapterStepBundle, ...]:
        return tuple(
            bundle for bundle in self._bundles if bundle.key.stream_key == key.stream_key
        )

    def _validate_basis_and_geometry(
        self,
        pre: CausalPreStepSample,
        *,
        validate_pre_step_requests: bool = True,
    ) -> None:
        cfg = self.config
        tolerance = cfg.state_alignment_tolerance
        projection_normal = tuple(
            float(item) for item in pre.projection_outward_normal_xyz
        )
        command_normal = tuple(float(item) for item in pre.command_outward_normal_xyz)
        recovery_normal = tuple(
            float(item) for item in pre.committed_recovery_outward_normal_xyz
        )
        projection_tangent = tuple(
            float(item) for item in pre.projection_tangent_unit_xyz
        )
        command_tangent = tuple(float(item) for item in pre.command_tangent_unit_xyz)
        for name, vector in (
            ("projection outward normal", projection_normal),
            ("command outward normal", command_normal),
            ("committed recovery outward normal", recovery_normal),
            ("projection tangent", projection_tangent),
            ("command tangent", command_tangent),
        ):
            if not _close(_norm(vector), 1.0, cfg.basis_tolerance):
                raise V6AdapterContractError(f"{name} must be unit length")
        if abs(_dot(projection_normal, projection_tangent)) > cfg.basis_tolerance:
            raise V6AdapterContractError(
                "projection tangent must be orthogonal to projection normal"
            )
        if abs(_dot(command_normal, command_tangent)) > cfg.basis_tolerance:
            raise V6AdapterContractError(
                "command tangent must be orthogonal to command normal"
            )
        for name in (
            "contact_tool_quaternion_wxyz",
            "robot_tcp_quaternion_wxyz",
        ):
            quaternion = tuple(float(item) for item in getattr(pre, name))
            if not _close(_norm(quaternion), 1.0, cfg.quaternion_tolerance):
                raise V6AdapterContractError(f"{name} must be normalized")
        if not _close(
            _dot(pre.contact_tool_linear_velocity_xyz_m_s, projection_normal),
            pre.normal_velocity_outward_m_s,
            tolerance,
        ):
            raise V6AdapterContractError(
                "outward normal velocity does not match contact-tool velocity and surface normal"
            )
        if (
            pre.requested_tangential_step_m
            > cfg.maximum_task_tangential_step_m + tolerance
        ):
            raise V6AdapterContractError(
                "requested task-tangential step exceeds the registered bound"
            )

        if not _close(
            pre.instantaneous_progress,
            pre.task_projection_progress,
            tolerance,
        ):
            raise V6AdapterContractError(
                "instantaneous progress differs from the registered path projection"
            )
        if not _close(
            pre.task_projection_arc_length_m,
            pre.task_projection_progress * pre.task_path_length_m,
            tolerance,
        ):
            raise V6AdapterContractError(
                "projected arc length differs from progress times frozen path length"
            )
        projection_displacement = _sub(
            pre.contact_tool_position_xyz_m,
            pre.task_projection_point_xyz_m,
        )
        tangential_plane_error = _sub(
            projection_displacement,
            _scale(
                projection_normal,
                _dot(projection_displacement, projection_normal),
            ),
        )
        expected_track_error = _norm(tangential_plane_error)
        expected_clearance = max(
            0.0,
            _dot(
                _sub(
                    pre.contact_tool_position_xyz_m,
                    pre.surface_reference_point_xyz_m,
                ),
                recovery_normal,
            ),
        )
        expected_hover_target = _add(
            pre.surface_reference_point_xyz_m,
            _scale(recovery_normal, cfg.recovery_hover_clearance_m),
        )
        expected_hover_error = _norm(
            _sub(
                pre.contact_tool_position_xyz_m,
                expected_hover_target,
            )
        )
        if not _vector_close(
            pre.recovery_hover_target_position_xyz_m,
            expected_hover_target,
            tolerance,
        ):
            raise V6AdapterContractError(
                "recovery hover target is inconsistent with committed surface reference and normal"
            )
        if not _close(expected_track_error, pre.geometric_track_error_m, tolerance):
            raise V6AdapterContractError(
                "geometric error is inconsistent with contact-tool pose and task path"
            )
        if not _close(expected_clearance, pre.recovery_clearance_m, tolerance):
            raise V6AdapterContractError(
                "recovery clearance is inconsistent with contact-tool pose and surface"
            )
        if not _close(expected_hover_error, pre.recovery_hover_pose_error_m, tolerance):
            raise V6AdapterContractError(
                "hover error is inconsistent with contact-tool pose and hover target"
            )

        if validate_pre_step_requests:
            recovery_delta = tuple(
                float(item) for item in pre.requested_recovery_delta_xyz_m
            )
            cross_track_delta = tuple(
                float(item)
                for item in pre.requested_cross_track_correction_delta_xyz_m
            )
            recovery_rotation = tuple(
                float(item)
                for item in pre.requested_recovery_rotation_delta_euler_xyz_rad
            )
            if _norm(recovery_delta) > cfg.maximum_recovery_reposition_step_m + tolerance:
                raise V6AdapterContractError(
                    "requested recovery translation exceeds its bound"
                )
            if _norm(recovery_rotation) > cfg.maximum_recovery_rotation_step_rad + tolerance:
                raise V6AdapterContractError(
                    "requested recovery Euler-XYZ rotation exceeds its bound"
                )
            if _dot(recovery_delta, recovery_normal) < -tolerance:
                raise V6AdapterContractError(
                    "requested recovery translation points inward"
                )
            toward_hover = _sub(
                pre.recovery_hover_target_position_xyz_m,
                pre.contact_tool_position_xyz_m,
            )
            if (
                _norm(recovery_delta) > tolerance
                and _dot(recovery_delta, toward_hover) <= 0.0
            ):
                raise V6AdapterContractError(
                    "requested recovery translation does not point toward hover"
                )
            if _norm(cross_track_delta) > (
                cfg.maximum_cross_track_correction_step_m + tolerance
            ):
                raise V6AdapterContractError(
                    "requested cross-track correction exceeds its bound"
                )
            if _norm(cross_track_delta) > tolerance:
                if pre.geometric_track_error_m < (
                    cfg.cross_track_correction_enter_m - tolerance
                ):
                    raise V6AdapterContractError(
                        "cross-track correction requested below its entry threshold"
                    )
                lateral = tangential_plane_error
                lateral_norm = _norm(lateral)
                if lateral_norm <= tolerance:
                    raise V6AdapterContractError(
                        "cross-track correction has no geometric residual"
                    )
                expected_scale = min(
                    1.0,
                    cfg.maximum_cross_track_correction_step_m / lateral_norm,
                )
                expected_cross_track = _scale(lateral, -expected_scale)
                if not _vector_close(
                    cross_track_delta,
                    expected_cross_track,
                    tolerance,
                ):
                    raise V6AdapterContractError(
                        "cross-track correction direction or magnitude is inconsistent"
                    )
                if abs(_dot(cross_track_delta, projection_normal)) > tolerance:
                    raise V6AdapterContractError(
                        "cross-track correction is not surface tangential"
                    )

    def _validate_static_stream_contract(
        self, pre: CausalPreStepSample, reference: CausalPreStepSample
    ) -> None:
        cfg = self.config
        if pre.target_force_n != reference.target_force_n:
            raise V6AdapterContractError("target force must remain fixed within a control stream")
        for name in (
            "bootstrap_id",
            "geometry_frame_id",
            "command_frame_id",
            "contact_tool_frame_id",
            "robot_tcp_frame_id",
            "task_path_id",
            "surface_id",
        ):
            if getattr(pre, name) != getattr(reference, name):
                raise V6AdapterContractError(f"{name} changed within a control stream")
        if not _close(
            pre.task_path_length_m,
            reference.task_path_length_m,
            cfg.state_alignment_tolerance,
        ):
            raise V6AdapterContractError("task path length changed within a control stream")

    def _validate_state_matches_native(
        self, pre: CausalPreStepSample, native: NativeAuditSample
    ) -> None:
        cfg = self.config
        scalar_pairs = (
            (pre.measured_force_n, native.audit_force_n, cfg.force_alignment_tolerance_n),
            (pre.normal_velocity_outward_m_s, native.normal_velocity_outward_m_s, cfg.state_alignment_tolerance),
            (pre.instantaneous_progress, native.instantaneous_progress, cfg.state_alignment_tolerance),
            (pre.geometric_track_error_m, native.geometric_track_error_m, cfg.state_alignment_tolerance),
            (pre.recovery_hover_pose_error_m, native.recovery_hover_pose_error_m, cfg.state_alignment_tolerance),
            (pre.recovery_clearance_m, native.recovery_clearance_m, cfg.state_alignment_tolerance),
            (pre.task_projection_progress, native.task_projection_progress, cfg.state_alignment_tolerance),
            (pre.task_projection_arc_length_m, native.task_projection_arc_length_m, cfg.state_alignment_tolerance),
        )
        if any(not _close(left, right, tol) for left, right, tol in scalar_pairs):
            raise V6AdapterContractError("pre-step state differs from its cited native sample")
        if pre.contact_observed != native.contact_observed:
            raise V6AdapterContractError("pre-step contact differs from its cited native sample")
        for name in (
            "surface_reference_point_xyz_m",
            "projection_outward_normal_xyz",
            "command_outward_normal_xyz",
            "committed_recovery_outward_normal_xyz",
            "task_projection_point_xyz_m",
            "projection_tangent_unit_xyz",
            "command_tangent_unit_xyz",
            "recovery_hover_target_position_xyz_m",
            "contact_tool_position_xyz_m",
            "contact_tool_linear_velocity_xyz_m_s",
            "contact_tool_angular_velocity_xyz_rad_s",
            "robot_tcp_position_xyz_m",
            "robot_tcp_linear_velocity_xyz_m_s",
            "robot_tcp_angular_velocity_xyz_rad_s",
        ):
            if not _vector_close(
                getattr(pre, name), getattr(native, name), cfg.state_alignment_tolerance
            ):
                raise V6AdapterContractError(f"pre-step {name} differs from cited native state")
        for name in ("contact_tool_quaternion_wxyz", "robot_tcp_quaternion_wxyz"):
            if not _quaternion_close(
                getattr(pre, name), getattr(native, name), cfg.quaternion_tolerance
            ):
                raise V6AdapterContractError(f"pre-step {name} differs from cited native state")

    def validate_pre(self, pre: CausalPreStepSample) -> None:
        pre.validate()
        self._validate_basis_and_geometry(pre)
        if pre.key in self._keys:
            raise V6AdapterContractError("duplicate control-step primary key")
        prior = self.prior_for(pre.key)
        if prior is None:
            if pre.key.control_step_index != 0:
                raise V6AdapterContractError("a control stream must start at step zero")
            if pre.state_source_kind != "bootstrap_sensor_snapshot":
                raise V6AdapterContractError("step zero requires an audited bootstrap sensor snapshot")
            if pre.force_source_native_sample_index is not None:
                raise V6AdapterContractError("the bootstrap state cannot cite a native sample")
            if pre.force_source_time_ns != pre.pre_time_ns:
                raise V6AdapterContractError("bootstrap state time must equal the pre-step time")
            return
        if pre.state_source_kind != "prior_native_sample":
            raise V6AdapterContractError("noninitial state must cite the prior native sample")
        if pre.key.control_step_index != prior.key.control_step_index + 1:
            raise V6AdapterContractError("control-step keys must be contiguous")
        expected_pre_time = prior.pre.pre_time_ns + self.config.control_period_ns
        if pre.pre_time_ns != expected_pre_time:
            raise V6AdapterContractError("control-step cadence differs from the registered period")
        last_native = prior.native_samples[-1]
        if pre.force_source_native_sample_index != last_native.native_sample_index:
            raise V6AdapterContractError("pre-step state must cite the preceding last native sample")
        if pre.force_source_time_ns != last_native.time_ns:
            raise V6AdapterContractError("pre-step state source time is misaligned")
        if pre.pre_time_ns != last_native.time_ns or pre.pre_time_ns != prior.post.post_time_ns:
            raise V6AdapterContractError("pre-step state is not aligned to the prior terminal native state")
        self._validate_state_matches_native(pre, last_native)
        self._validate_static_stream_contract(pre, self._stream_bundles(pre.key)[0].pre)

    @staticmethod
    def _same_key(expected: ControlStepKey, actual: ControlStepKey, *, name: str) -> None:
        if actual != expected:
            raise V6AdapterContractError(f"{name} key does not match the pre-step key")

    def _validate_supervisor_history(
        self, pre: CausalPreStepSample, command: SupervisorCommand
    ) -> None:
        history = self._stream_bundles(pre.key)
        previous = history[-1] if len(history) >= 1 else None
        previous_previous = history[-2] if len(history) >= 2 else None
        expected_previous_force = None if previous is None else previous.pre.measured_force_n
        expected_previous_previous_force = (
            None if previous_previous is None else previous_previous.pre.measured_force_n
        )
        expected_previous_command = 0.0 if previous is None else previous.readback.normal_step_m
        expected_previous_previous_command = (
            0.0 if previous_previous is None else previous_previous.readback.normal_step_m
        )
        tolerance = self.config.state_alignment_tolerance
        for actual, expected in (
            (command.previous_force_n, expected_previous_force),
            (command.previous_previous_force_n, expected_previous_previous_force),
        ):
            if (actual is None) != (expected is None):
                raise V6AdapterContractError("supervisor force history is not ledger-closed")
            if actual is not None and not _close(actual, expected, tolerance):
                raise V6AdapterContractError("supervisor force history is not ledger-closed")
        if not _close(command.previous_normal_command_m, expected_previous_command, tolerance):
            raise V6AdapterContractError("supervisor prior command history is not ledger-closed")
        if not _close(
            command.previous_previous_normal_command_m,
            expected_previous_previous_command,
            tolerance,
        ):
            raise V6AdapterContractError("supervisor second command history is not ledger-closed")
        direct_pairs = (
            (command.measured_force_n, pre.measured_force_n),
            (command.target_force_n, pre.target_force_n),
            (command.normal_velocity_outward_m_s, pre.normal_velocity_outward_m_s),
            (command.geometric_track_error_m, pre.geometric_track_error_m),
            (command.recovery_hover_pose_error_m, pre.recovery_hover_pose_error_m),
            (command.recovery_clearance_m, pre.recovery_clearance_m),
            (command.instantaneous_progress, pre.instantaneous_progress),
            (command.requested_tangential_step_m, pre.requested_tangential_step_m),
        )
        if any(not _close(left, right, tolerance) for left, right in direct_pairs):
            raise V6AdapterContractError("supervisor command does not log the complete pre-step input")
        if command.contact_observed != pre.contact_observed:
            raise V6AdapterContractError("supervisor contact input does not match pre-step state")

    def _validate_issued(
        self,
        pre: CausalPreStepSample,
        command: SupervisorCommand,
        issued: IssuedCartesianCommand,
    ) -> None:
        issued.validate()
        self._validate_supervisor_history(pre, command)
        self._same_key(pre.key, issued.key, name="issued command")
        if issued.command_id in self._command_ids:
            raise V6AdapterContractError("command_id must be globally unique")
        if issued.issued_time_ns != pre.pre_time_ns:
            raise V6AdapterContractError("command issue time must equal pre-step time")
        if (
            issued.command_frame_id != pre.command_frame_id
            or issued.command_frame_id != self.config.expected_command_frame_id
        ):
            raise V6AdapterContractError("issued command frame is not the registered frame")
        if issued.command_mode != self.config.expected_command_mode:
            raise V6AdapterContractError("issued command mode is not registered Euler XYZ")
        tolerance = self.config.command_readback_tolerance_m
        if not _close(issued.normal_step_m, command.executed_normal_step_m, tolerance):
            raise V6AdapterContractError("issued normal command differs from supervisor output")
        if not _close(
            issued.task_tangential_step_m,
            command.executed_tangential_step_m,
            tolerance,
        ):
            raise V6AdapterContractError("issued task-tangential command differs from supervisor output")
        if issued.recovery_reposition_permitted != command.recovery_reposition_permitted:
            raise V6AdapterContractError("issued recovery authority differs from supervisor output")
        if (
            issued.cross_track_correction_permitted
            != command.cross_track_correction_permitted
        ):
            raise V6AdapterContractError(
                "issued cross-track authority differs from supervisor output"
            )
        if (
            command.recovery_reposition_permitted
            and command.cross_track_correction_permitted
        ):
            raise V6AdapterContractError(
                "recovery and cross-track authorities cannot overlap"
            )

        expected_normal = _scale(
            pre.command_outward_normal_xyz, -float(command.executed_normal_step_m)
        )
        expected_task = _scale(
            pre.command_tangent_unit_xyz, float(command.executed_tangential_step_m)
        )
        zero = (0.0, 0.0, 0.0)
        expected_recovery = (
            pre.requested_recovery_delta_xyz_m
            if command.recovery_reposition_permitted
            else zero
        )
        expected_cross_track = (
            pre.requested_cross_track_correction_delta_xyz_m
            if command.cross_track_correction_permitted
            else zero
        )
        expected_recovery_rotation = (
            pre.requested_recovery_rotation_delta_euler_xyz_rad
            if command.recovery_reposition_permitted
            else zero
        )
        for actual, expected, name in (
            (issued.normal_delta_xyz_m, expected_normal, "normal Cartesian component"),
            (issued.task_delta_xyz_m, expected_task, "task Cartesian component"),
            (issued.recovery_delta_xyz_m, expected_recovery, "recovery Cartesian component"),
            (
                issued.cross_track_correction_delta_xyz_m,
                expected_cross_track,
                "cross-track Cartesian component",
            ),
        ):
            if not _vector_close(actual, expected, tolerance):
                raise V6AdapterContractError(f"issued {name} is inconsistent")
        expected_total = _add(
            expected_normal,
            expected_task,
            expected_recovery,
            expected_cross_track,
        )
        if not _vector_close(issued.cartesian_delta_xyz_m, expected_total, tolerance):
            raise V6AdapterContractError("issued Cartesian delta does not equal its authority components")
        rotation_tolerance = self.config.rotation_readback_tolerance_rad
        if not _vector_close(
            issued.recovery_rotation_delta_euler_xyz_rad,
            expected_recovery_rotation,
            rotation_tolerance,
        ):
            raise V6AdapterContractError("issued recovery Euler-XYZ component is inconsistent")
        if not _vector_close(
            issued.rotation_delta_euler_xyz_rad,
            expected_recovery_rotation,
            rotation_tolerance,
        ):
            raise V6AdapterContractError("issued Euler-XYZ delta is not recovery-owned")
        if command.recovery_reposition_permitted:
            if command.state_after != SupervisorState.RECOVERY_HOVER.value:
                raise V6AdapterContractError(
                    "recovery reposition is outside registered low-level authority"
                )
            if (
                abs(command.executed_normal_step_m) > tolerance
                or abs(command.executed_tangential_step_m) > tolerance
            ):
                raise V6AdapterContractError("recovery reposition overlaps normal or task authority")
        if command.cross_track_correction_permitted:
            track_cross_correction = bool(
                command.transition_reason
                == "track_cross_track_correction_projection"
                and command.state_before == SupervisorState.TRACK.value
                and command.state_after == SupervisorState.TRACK.value
                and not command.transitioned
                and not command.stable_track
                and not command.tangential_motion_permitted
                and command.projection_active
                and pre.geometric_track_error_m
                >= self.config.cross_track_correction_enter_m - tolerance
                and abs(
                    command.committed_progress_after
                    - command.committed_progress_before
                )
                <= tolerance
                and abs(command.integral_after_n_s - command.integral_before_n_s)
                <= tolerance
            )
            if not track_cross_correction:
                raise V6AdapterContractError(
                    "cross-track correction is outside registered TRACK authority"
                )
            if (
                abs(command.executed_normal_step_m) > tolerance
                or abs(command.executed_tangential_step_m) > tolerance
            ):
                raise V6AdapterContractError(
                    "cross-track correction overlaps normal or task authority"
                )
        if command.state_after == SupervisorState.SAFE_HOLD.value:
            if not _vector_close(issued.cartesian_delta_xyz_m, zero, tolerance):
                raise V6AdapterContractError("SAFE_HOLD Cartesian command must be zero")
            if not _vector_close(
                issued.rotation_delta_euler_xyz_rad, zero, rotation_tolerance
            ):
                raise V6AdapterContractError("SAFE_HOLD Euler-XYZ command must be zero")

    def _validate_readback(
        self, issued: IssuedCartesianCommand, readback: CommandReadback
    ) -> None:
        readback.validate()
        self._same_key(issued.key, readback.key, name="command readback")
        if readback.command_id != issued.command_id:
            raise V6AdapterContractError("command readback identity mismatch")
        if not readback.accepted:
            raise V6AdapterContractError("actuator interface rejected the command")
        if readback.readback_stage != self.config.required_readback_stage:
            raise V6AdapterContractError("readback is not from the final actuator target stage")
        if readback.readback_time_ns < issued.issued_time_ns:
            raise V6AdapterContractError("command readback predates command issue")
        if readback.command_frame_id != issued.command_frame_id:
            raise V6AdapterContractError("command-frame readback mismatch")
        if readback.command_mode != issued.command_mode:
            raise V6AdapterContractError("command-mode readback mismatch")
        tolerance = self.config.command_readback_tolerance_m
        if any(
            not _close(left, right, tolerance)
            for left, right in (
                (issued.normal_step_m, readback.normal_step_m),
                (issued.task_tangential_step_m, readback.task_tangential_step_m),
            )
        ):
            raise V6AdapterContractError("scalar command readback mismatch")
        for name in (
            "normal_delta_xyz_m",
            "task_delta_xyz_m",
            "recovery_delta_xyz_m",
            "cross_track_correction_delta_xyz_m",
            "cartesian_delta_xyz_m",
        ):
            if not _vector_close(getattr(issued, name), getattr(readback, name), tolerance):
                raise V6AdapterContractError(f"{name} readback mismatch")
        rotation_tolerance = self.config.rotation_readback_tolerance_rad
        for name in (
            "recovery_rotation_delta_euler_xyz_rad",
            "rotation_delta_euler_xyz_rad",
        ):
            if not _vector_close(
                getattr(issued, name), getattr(readback, name), rotation_tolerance
            ):
                raise V6AdapterContractError(f"{name} readback mismatch")
        if issued.recovery_reposition_permitted != readback.recovery_reposition_permitted:
            raise V6AdapterContractError("recovery-authority readback mismatch")
        if (
            issued.cross_track_correction_permitted
            != readback.cross_track_correction_permitted
        ):
            raise V6AdapterContractError("cross-track-authority readback mismatch")

    def _validate_native_geometry(
        self, pre: CausalPreStepSample, native: NativeAuditSample
    ) -> None:
        payload = asdict(pre)
        payload.update(
            measured_force_n=native.audit_force_n,
            normal_velocity_outward_m_s=native.normal_velocity_outward_m_s,
            contact_observed=native.contact_observed,
            instantaneous_progress=native.instantaneous_progress,
            geometric_track_error_m=native.geometric_track_error_m,
            recovery_hover_pose_error_m=native.recovery_hover_pose_error_m,
            recovery_clearance_m=native.recovery_clearance_m,
            surface_reference_point_xyz_m=native.surface_reference_point_xyz_m,
            projection_outward_normal_xyz=native.projection_outward_normal_xyz,
            command_outward_normal_xyz=native.command_outward_normal_xyz,
            committed_recovery_outward_normal_xyz=(
                native.committed_recovery_outward_normal_xyz
            ),
            task_projection_point_xyz_m=native.task_projection_point_xyz_m,
            task_projection_progress=native.task_projection_progress,
            task_projection_arc_length_m=native.task_projection_arc_length_m,
            projection_tangent_unit_xyz=native.projection_tangent_unit_xyz,
            command_tangent_unit_xyz=native.command_tangent_unit_xyz,
            recovery_hover_target_position_xyz_m=native.recovery_hover_target_position_xyz_m,
            contact_tool_position_xyz_m=native.contact_tool_position_xyz_m,
            contact_tool_quaternion_wxyz=native.contact_tool_quaternion_wxyz,
            contact_tool_linear_velocity_xyz_m_s=native.contact_tool_linear_velocity_xyz_m_s,
            contact_tool_angular_velocity_xyz_rad_s=native.contact_tool_angular_velocity_xyz_rad_s,
            robot_tcp_position_xyz_m=native.robot_tcp_position_xyz_m,
            robot_tcp_quaternion_wxyz=native.robot_tcp_quaternion_wxyz,
            robot_tcp_linear_velocity_xyz_m_s=native.robot_tcp_linear_velocity_xyz_m_s,
            robot_tcp_angular_velocity_xyz_rad_s=native.robot_tcp_angular_velocity_xyz_rad_s,
        )
        payload["key"] = pre.key
        state = CausalPreStepSample(**payload)
        state.validate()
        # Native/post packets describe the plant after the command has already
        # executed.  Reusing the pre-step recovery request against that later
        # pose would misclassify a legitimate arrival at, or overshoot of, the
        # hover target as an infrastructure contract failure.
        self._validate_basis_and_geometry(
            state,
            validate_pre_step_requests=False,
        )

    def _validate_post_matches_native(
        self, post: CausalPostStepSample, native: NativeAuditSample
    ) -> None:
        cfg = self.config
        scalar_pairs = (
            (post.measured_force_n, native.audit_force_n, cfg.force_alignment_tolerance_n),
            (post.normal_velocity_outward_m_s, native.normal_velocity_outward_m_s, cfg.state_alignment_tolerance),
            (post.instantaneous_progress, native.instantaneous_progress, cfg.state_alignment_tolerance),
            (post.geometric_track_error_m, native.geometric_track_error_m, cfg.state_alignment_tolerance),
            (post.recovery_hover_pose_error_m, native.recovery_hover_pose_error_m, cfg.state_alignment_tolerance),
            (post.recovery_clearance_m, native.recovery_clearance_m, cfg.state_alignment_tolerance),
            (post.task_projection_progress, native.task_projection_progress, cfg.state_alignment_tolerance),
            (post.task_projection_arc_length_m, native.task_projection_arc_length_m, cfg.state_alignment_tolerance),
        )
        if any(not _close(left, right, tol) for left, right, tol in scalar_pairs):
            raise V6AdapterContractError("post-step state differs from the last native state")
        if post.contact_observed != native.contact_observed:
            raise V6AdapterContractError("post-step contact differs from the last native state")
        for name in (
            "surface_reference_point_xyz_m",
            "projection_outward_normal_xyz",
            "command_outward_normal_xyz",
            "committed_recovery_outward_normal_xyz",
            "task_projection_point_xyz_m",
            "projection_tangent_unit_xyz",
            "command_tangent_unit_xyz",
            "recovery_hover_target_position_xyz_m",
            "contact_tool_position_xyz_m",
            "contact_tool_linear_velocity_xyz_m_s",
            "contact_tool_angular_velocity_xyz_rad_s",
            "robot_tcp_position_xyz_m",
            "robot_tcp_linear_velocity_xyz_m_s",
            "robot_tcp_angular_velocity_xyz_rad_s",
        ):
            if not _vector_close(
                getattr(post, name), getattr(native, name), cfg.state_alignment_tolerance
            ):
                raise V6AdapterContractError(f"post-step {name} differs from last native state")
        for name in ("contact_tool_quaternion_wxyz", "robot_tcp_quaternion_wxyz"):
            if not _quaternion_close(
                getattr(post, name), getattr(native, name), cfg.quaternion_tolerance
            ):
                raise V6AdapterContractError(f"post-step {name} differs from last native state")

    def validate_bundle(self, bundle: AdapterStepBundle) -> None:
        self.validate_pre(bundle.pre)
        self._validate_issued(bundle.pre, bundle.supervisor_command, bundle.issued)
        self._validate_readback(bundle.issued, bundle.readback)
        expected_count = self.config.expected_native_steps_per_control
        if len(bundle.native_samples) != expected_count:
            raise V6AdapterContractError("native sample count differs from adapter ratio")
        prior = self.prior_for(bundle.key)
        expected_first_native = (
            0 if prior is None else prior.native_samples[-1].native_sample_index + 1
        )
        if not (
            bundle.issued.issued_time_ns
            <= bundle.readback.readback_time_ns
            < bundle.pre.pre_time_ns + self.config.native_period_ns
        ):
            raise V6AdapterContractError("readback must precede the first registered native sample")
        for substep, native in enumerate(bundle.native_samples):
            native.validate()
            self._same_key(bundle.key, native.key, name="native sample")
            if native.command_id != bundle.issued.command_id:
                raise V6AdapterContractError("native sample command identity mismatch")
            if native.substep_index != substep:
                raise V6AdapterContractError("native substep indices must be contiguous")
            if native.native_sample_index != expected_first_native + substep:
                raise V6AdapterContractError("native sample indices must be contiguous")
            expected_time = (
                bundle.pre.pre_time_ns + (substep + 1) * self.config.native_period_ns
            )
            if native.time_ns != expected_time:
                raise V6AdapterContractError("native sample cadence differs from the registered period")
            self._validate_native_geometry(bundle.pre, native)
        bundle.post.validate()
        self._same_key(bundle.key, bundle.post.key, name="post-step sample")
        last_native = bundle.native_samples[-1]
        if bundle.post.last_native_sample_index != last_native.native_sample_index:
            raise V6AdapterContractError("post-step last native index mismatch")
        if bundle.post.last_native_time_ns != last_native.time_ns:
            raise V6AdapterContractError("post-step last native time mismatch")
        if bundle.post.post_time_ns != last_native.time_ns:
            raise V6AdapterContractError("post-step time must equal the final native sample time")
        self._validate_post_matches_native(bundle.post, last_native)

    def commit(self, bundle: AdapterStepBundle) -> None:
        snapshot = self._snapshot()
        try:
            self.validate_bundle(bundle)
            self._bundles.append(bundle)
            self._keys.add(bundle.key)
            self._command_ids.add(bundle.issued.command_id)
            self._last_by_stream[bundle.key.stream_key] = bundle
        except BaseException:
            self._restore(snapshot)
            raise

    @staticmethod
    def _control_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        return (
            row["run_id"],
            row["scenario_id"],
            row["evaluation_id"],
            row["segment_index"],
            row["control_step_index"],
        )

    def table_rows(self) -> dict[str, tuple[dict[str, Any], ...]]:
        tables: dict[str, list[dict[str, Any]]] = {
            name: [] for name in (*self.CONTROL_TABLES, "native_audit")
        }
        for bundle in self._bundles:
            tables["causal_input"].append(bundle.pre.to_log_row())
            tables["supervisor_command"].append(
                {**bundle.key.as_dict(), **asdict(bundle.supervisor_command)}
            )
            tables["issued_command"].append(bundle.issued.to_log_row())
            tables["command_readback"].append(bundle.readback.to_log_row())
            tables["post_step"].append(bundle.post.to_log_row())
            tables["native_audit"].extend(
                native.to_log_row() for native in bundle.native_samples
            )
        return {name: tuple(rows) for name, rows in tables.items()}

    def closure_summary(self) -> dict[str, Any]:
        tables = self.table_rows()
        key_sets = {
            name: {self._control_row_key(row) for row in tables[name]}
            for name in self.CONTROL_TABLES
        }
        reference = key_sets["causal_input"]
        native_keys = {self._control_row_key(row) for row in tables["native_audit"]}
        control_keys_closed = all(keys == reference for keys in key_sets.values())
        native_keys_closed = native_keys == reference if reference else not native_keys
        return {
            "control_steps": len(reference),
            "native_samples": len(tables["native_audit"]),
            "control_primary_keys_closed": control_keys_closed,
            "native_foreign_keys_closed": native_keys_closed,
            "duplicate_control_keys": len(self._bundles) - len(reference),
            "unique_command_ids": len(self._command_ids),
            "closed": bool(
                control_keys_closed
                and native_keys_closed
                and len(self._bundles) == len(reference)
                and len(self._command_ids) == len(reference)
            ),
        }


CommandBuilder = Callable[[SupervisorCommand, CausalPreStepSample], IssuedCartesianCommand]
CommandExecutor = Callable[[IssuedCartesianCommand], BackendStepResult]


class CausalControlAdapter:
    """Orders one causal controller decision and one backend control interval."""

    def __init__(
        self,
        supervisor: UnifiedCausalForceRecoverySupervisor,
        command_builder: CommandBuilder,
        *,
        config: AdapterConfig = AdapterConfig(),
        ledger: CausalEvidenceLedger | None = None,
    ) -> None:
        if not isinstance(supervisor, UnifiedCausalForceRecoverySupervisor):
            raise V6AdapterContractError("supervisor has an unsupported type")
        if not callable(command_builder):
            raise V6AdapterContractError("command_builder must be callable")
        config.validate()
        controller_period_ns = int(round(float(supervisor.config.dt_s) * 1_000_000_000))
        if controller_period_ns != config.control_period_ns:
            raise V6AdapterContractError("adapter control period differs from supervisor dt_s")
        self.supervisor = supervisor
        self.command_builder = command_builder
        self.config = config
        self.ledger = CausalEvidenceLedger(config) if ledger is None else ledger
        if self.ledger.config != config:
            raise V6AdapterContractError("ledger and adapter configs differ")
        if not self.ledger._is_pristine():
            raise V6AdapterContractError("a new adapter requires a pristine evidence ledger")
        fresh_snapshot = UnifiedCausalForceRecoverySupervisor(supervisor.config).snapshot()
        if supervisor.snapshot() != fresh_snapshot:
            raise V6AdapterContractError(
                "a new adapter requires a fresh-reset supervisor; resume is not implicit"
            )
        self._faulted = False
        self._fault_stage: str | None = None
        self._stream_key: tuple[str, int, int, int] | None = None
        self._expected_supervisor_snapshot = supervisor.snapshot()
        self._quarantined_backend_result: BackendStepResult | None = None

    @property
    def faulted(self) -> bool:
        return self._faulted

    @property
    def fault_stage(self) -> str | None:
        return self._fault_stage

    @property
    def quarantined_backend_result(self) -> BackendStepResult | None:
        """Invalid post-execution evidence retained for runner-side quarantine."""
        return self._quarantined_backend_result

    def _mark_fault(
        self,
        stage: str,
        supervisor_snapshot: SupervisorSnapshot,
        ledger_snapshot: _LedgerSnapshot,
        *,
        backend_result: BackendStepResult | None = None,
    ) -> None:
        self._faulted = True
        self._fault_stage = stage
        self._quarantined_backend_result = backend_result
        self.supervisor.restore(supervisor_snapshot)
        self.ledger._restore(ledger_snapshot)

    def _require_ledger_unchanged(
        self, expected: _LedgerSnapshot, *, owner: str
    ) -> None:
        if self.ledger._snapshot() != expected:
            raise V6AdapterContractError(f"{owner} mutated the evidence ledger")

    def step(
        self,
        pre: CausalPreStepSample,
        executor: CommandExecutor,
    ) -> AdapterStepBundle:
        if self._faulted:
            raise V6AdapterContractError("adapter is terminally faulted")
        if not callable(executor):
            raise V6AdapterContractError("executor must be callable")
        supervisor_snapshot = self.supervisor.snapshot()
        ledger_snapshot = self.ledger._snapshot()
        if supervisor_snapshot != self._expected_supervisor_snapshot:
            self._faulted = True
            self._fault_stage = "external_supervisor_state_mutation"
            raise V6AdapterContractError("supervisor state changed outside the adapter transaction")
        try:
            if self._stream_key is not None and pre.key.stream_key != self._stream_key:
                raise V6AdapterContractError("one adapter instance cannot cross control-stream identity")
            self.ledger.validate_pre(pre)
            command = self.supervisor.command(pre.to_supervisor_input())
            command_snapshot = self.supervisor.snapshot()
            issued = self.command_builder(command, pre)
            if self.supervisor.snapshot() != command_snapshot:
                raise V6AdapterContractError("command builder mutated supervisor state")
            self._require_ledger_unchanged(ledger_snapshot, owner="command builder")
            if not isinstance(issued, IssuedCartesianCommand):
                raise V6AdapterContractError("command_builder returned an unsupported packet")
            self.ledger._validate_issued(pre, command, issued)
        except BaseException:
            self._mark_fault("pre_or_command", supervisor_snapshot, ledger_snapshot)
            raise
        try:
            result = executor(issued)
        except BaseException as exc:
            self._mark_fault("executor", supervisor_snapshot, ledger_snapshot)
            raise V6AdapterExecutionError("command executor failed") from exc
        if self.supervisor.snapshot() != command_snapshot:
            self._mark_fault(
                "executor_supervisor_state_mutation",
                supervisor_snapshot,
                ledger_snapshot,
                backend_result=result if isinstance(result, BackendStepResult) else None,
            )
            raise V6AdapterContractError("command executor mutated supervisor state")
        try:
            self._require_ledger_unchanged(ledger_snapshot, owner="command executor")
        except BaseException:
            self._mark_fault(
                "executor_ledger_state_mutation",
                supervisor_snapshot,
                ledger_snapshot,
                backend_result=result if isinstance(result, BackendStepResult) else None,
            )
            raise
        if not isinstance(result, BackendStepResult):
            self._mark_fault("backend_result_type", supervisor_snapshot, ledger_snapshot)
            raise V6AdapterContractError("executor returned an unsupported result")
        bundle = AdapterStepBundle(
            pre=pre,
            supervisor_command=command,
            issued=issued,
            readback=result.readback,
            native_samples=tuple(result.native_samples),
            post=result.post,
        )
        try:
            self.ledger.commit(bundle)
            if not self.ledger.closure_summary()["closed"]:
                raise V6AdapterContractError("adapter cannot return a nonclosed evidence ledger")
        except BaseException:
            self._mark_fault(
                "evidence_validation",
                supervisor_snapshot,
                ledger_snapshot,
                backend_result=result,
            )
            raise
        if self._stream_key is None:
            self._stream_key = pre.key.stream_key
        self._expected_supervisor_snapshot = self.supervisor.snapshot()
        return bundle


__all__ = [
    "AdapterConfig",
    "AdapterStepBundle",
    "BackendStepResult",
    "CausalControlAdapter",
    "CausalEvidenceLedger",
    "CausalPostStepSample",
    "CausalPreStepSample",
    "CommandReadback",
    "ControlStepKey",
    "IssuedCartesianCommand",
    "NativeAuditSample",
    "V6AdapterContractError",
    "V6AdapterError",
    "V6AdapterExecutionError",
]
