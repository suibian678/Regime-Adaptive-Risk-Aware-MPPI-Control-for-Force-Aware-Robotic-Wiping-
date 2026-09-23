"""Causal SAPIEN binding for repeatable ForceWipe V6 SANDBOX runs.

This module deliberately imports neither SAPIEN nor ManiSkill.  The runtime
objects are accessed through the small duck-typed surface exercised by the
fake-environment tests.  Importing this module therefore cannot start or even
initialize physics.

The binding is intentionally narrow:

* simulation, control, and native audit rates are all exactly 100 Hz;
* one control decision owns exactly one native physics sample;
* controller input force is the preceding native audit sample (apart from the
  explicitly marked step-zero bootstrap);
* contact geometry is measured on the compliant tool actor, while the robot
  TCP is retained separately for actuator/readback provenance;
* task motion is tangent-only and recovery repositioning is bounded to 6 mm;
* rotation is frozen at zero for the first physical SANDBOX binding;
* command readback is reconstructed from the action received by
  ``agent.set_action`` and the arm controller target pose, never echoed from
  the issued packet; and
* raw native evidence is offered to a runner callback before the logical V6
  adapter validates or commits the interval.

The callbacks in this file do not write files.  A future SANDBOX runner owns
the crash-safe Parquet/Zstd staging and terminal-fault marker policy.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import MISSING, dataclass, fields
import math
from typing import Any

import numpy as np

from .causal_adapter import (
    AdapterConfig,
    AdapterStepBundle,
    BackendStepResult,
    CausalControlAdapter,
    CausalPostStepSample,
    CausalPreStepSample,
    CommandReadback,
    ControlStepKey,
    IssuedCartesianCommand,
    NativeAuditSample,
)
from .controller import (
    SupervisorCommand,
    SupervisorSnapshot,
    UnifiedCausalForceRecoverySupervisor,
)


class V6SapienSandboxBindingError(RuntimeError):
    """Base error for the non-qualification V6 physical binding."""


class V6SapienSandboxContractError(V6SapienSandboxBindingError):
    """The environment or an evidence packet violates the frozen interface."""


class V6SapienSandboxExecutionError(V6SapienSandboxBindingError):
    """A physical interval advanced without producing a valid closed bundle."""


def _array(value: Any, *, name: str, length: int) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (1, length):
        array = array[0]
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise V6SapienSandboxContractError(
            f"{name} must be one finite length-{length} vector"
        )
    return array.copy()


def _flat_vector(value: Any, *, name: str) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise V6SapienSandboxContractError(f"{name} must be one finite vector")
    return array.copy()


def _index_vector(value: Any, *, name: str) -> np.ndarray:
    numeric = _flat_vector(value, name=name)
    rounded = np.rint(numeric)
    if not np.allclose(numeric, rounded, rtol=0.0, atol=0.0):
        raise V6SapienSandboxContractError(f"{name} must contain integer indices")
    result = rounded.astype(np.int64)
    if np.any(result < 0) or len(np.unique(result)) != result.size:
        raise V6SapienSandboxContractError(f"{name} contains invalid indices")
    return result


def _scalar(value: Any, *, name: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 1 or not math.isfinite(float(array[0])):
        raise V6SapienSandboxContractError(f"{name} must contain one finite scalar")
    return float(array[0])


def _tuple(vector: np.ndarray) -> tuple[float, ...]:
    return tuple(float(item) for item in np.asarray(vector, dtype=np.float64))


def _unit(vector: Any, *, name: str) -> np.ndarray:
    value = _array(vector, name=name, length=3)
    norm = float(np.linalg.norm(value))
    if norm <= np.finfo(float).eps:
        raise V6SapienSandboxContractError(f"{name} must be nonzero")
    return value / norm


def _bounded(vector: np.ndarray, maximum_norm: float) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    if norm <= float(maximum_norm):
        return value.copy()
    return value * (float(maximum_norm) / norm)


def _quaternion(value: Any, *, name: str) -> np.ndarray:
    quaternion = _array(value, name=name, length=4)
    norm = float(np.linalg.norm(quaternion))
    # SAPIEN exposes these tensors as float32.  Accept only the bounded
    # round-off that a normalized float32 quaternion can acquire, then
    # canonicalize it before any frame transform or evidence write.
    if abs(norm - 1.0) > 1e-6:
        raise V6SapienSandboxContractError(f"{name} must be normalized")
    return quaternion / norm


def _quat_conjugate(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64)
    return np.array([value[0], -value[1], -value[2], -value[3]], dtype=np.float64)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.asarray(left, dtype=np.float64)
    rw, rx, ry, rz = np.asarray(right, dtype=np.float64)
    return np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )


def _quat_rotate(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    rotated = _quat_multiply(
        _quat_multiply(
            np.asarray(quaternion, dtype=np.float64),
            np.array([0.0, *np.asarray(vector, dtype=np.float64)], dtype=np.float64),
        ),
        _quat_conjugate(np.asarray(quaternion, dtype=np.float64)),
    )
    return rotated[1:].copy()


def _quaternion_same(left: np.ndarray, right: np.ndarray, tolerance: float) -> bool:
    return bool(
        np.allclose(left, right, rtol=0.0, atol=tolerance)
        or np.allclose(left, -right, rtol=0.0, atol=tolerance)
    )


def _bool_scalar(value: Any, *, name: str) -> bool:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value).reshape(-1)
    if array.size != 1:
        raise V6SapienSandboxContractError(f"{name} must contain one boolean")
    item = array[0]
    if not isinstance(item, (bool, np.bool_)):
        raise V6SapienSandboxContractError(f"{name} must contain one boolean")
    return bool(item)


def _construct_packet(packet_type: type, candidates: dict[str, Any]):
    """Construct a causal-adapter packet while tolerating additive API fields.

    The V6 causal adapter is maintained independently from this binding.  This
    helper filters compatibility aliases but still fails if a newly required
    field has no physical source here; it never supplies a guessed value.
    """

    packet_fields = tuple(fields(packet_type))
    allowed = {item.name for item in packet_fields}
    missing = [
        item.name
        for item in packet_fields
        if item.name not in candidates
        and item.default is MISSING
        and item.default_factory is MISSING
    ]
    if missing:
        raise V6SapienSandboxContractError(
            f"{packet_type.__name__} has unsupported required fields: {missing}"
        )
    return packet_type(**{name: value for name, value in candidates.items() if name in allowed})


@dataclass(frozen=True)
class SapienSandboxBindingConfig:
    sim_rate_hz: int = 100
    control_rate_hz: int = 100
    native_rate_hz: int = 100
    expected_native_steps_per_control: int = 1
    position_action_scale_m: float = 0.1
    maximum_task_tangent_step_m: float = 0.0035
    maximum_recovery_reposition_step_m: float = 0.006
    contact_threshold_n: float = 0.2
    gripper_command: float = -1.0
    rotation_frozen_zero: bool = True
    force_alignment_tolerance_n: float = 1e-9
    pose_alignment_tolerance_m: float = 1e-7
    command_readback_tolerance_m: float = 2e-7
    quaternion_readback_tolerance: float = 1e-8
    geometry_frame_id: str = "world"
    command_frame_id: str = "world"
    contact_tool_frame_id: str = "v4_tool"
    robot_tcp_frame_id: str = "agent_tcp"
    command_mode: str = "delta_pose_euler_xyz"
    readback_stage: str = "ik_and_joint_drive_target_after_limits"

    def validate(self) -> None:
        if (
            self.sim_rate_hz != 100
            or self.control_rate_hz != 100
            or self.native_rate_hz != 100
        ):
            raise V6SapienSandboxContractError(
                "V6 physical SANDBOX requires sim/control/native rates of 100 Hz"
            )
        if self.expected_native_steps_per_control != 1:
            raise V6SapienSandboxContractError(
                "V6 physical SANDBOX requires one native sample per control step"
            )
        positive = (
            self.position_action_scale_m,
            self.maximum_task_tangent_step_m,
            self.maximum_recovery_reposition_step_m,
            self.contact_threshold_n,
            self.force_alignment_tolerance_n,
            self.pose_alignment_tolerance_m,
            self.command_readback_tolerance_m,
            self.quaternion_readback_tolerance,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise V6SapienSandboxContractError("binding tolerances and bounds must be positive")
        if self.maximum_recovery_reposition_step_m != 0.006:
            raise V6SapienSandboxContractError(
                "the first V6 recovery-reposition bound is frozen at 6 mm"
            )
        if self.maximum_task_tangent_step_m > self.position_action_scale_m:
            raise V6SapienSandboxContractError("task tangent bound exceeds action scale")
        if self.maximum_recovery_reposition_step_m > self.position_action_scale_m:
            raise V6SapienSandboxContractError("recovery bound exceeds action scale")
        if self.gripper_command != -1.0:
            raise V6SapienSandboxContractError("the V6 SANDBOX gripper command is frozen at -1")
        if self.rotation_frozen_zero is not True:
            raise V6SapienSandboxContractError("rotation must remain frozen at zero")
        for name in (
            "geometry_frame_id",
            "command_frame_id",
            "contact_tool_frame_id",
            "robot_tcp_frame_id",
            "command_mode",
            "readback_stage",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise V6SapienSandboxContractError(f"{name} must be nonempty")
        if self.command_frame_id != "world":
            raise V6SapienSandboxContractError("generic V6 commands are frozen in world frame")
        if self.command_mode != "delta_pose_euler_xyz":
            raise V6SapienSandboxContractError("unexpected generic command mode")


@dataclass(frozen=True)
class ToolStateProvenance:
    time_ns: int
    measured_force_n: float
    contact_observed: bool
    contact_tool_position_xyz_m: tuple[float, float, float]
    contact_tool_quaternion_wxyz: tuple[float, float, float, float]
    contact_tool_linear_velocity_xyz_m_s: tuple[float, float, float]
    contact_tool_angular_velocity_xyz_rad_s: tuple[float, float, float]
    robot_tcp_position_xyz_m: tuple[float, float, float]
    robot_tcp_quaternion_wxyz: tuple[float, float, float, float]
    robot_tcp_linear_velocity_xyz_m_s: tuple[float, float, float]
    robot_tcp_angular_velocity_xyz_rad_s: tuple[float, float, float]
    instantaneous_progress: float
    geometric_track_error_m: float
    recovery_hover_pose_error_m: float
    recovery_clearance_m: float
    signed_recovery_clearance_m: float
    surface_reference_point_xyz_m: tuple[float, float, float]
    projection_outward_normal_xyz: tuple[float, float, float]
    command_outward_normal_xyz: tuple[float, float, float]
    task_projection_point_xyz_m: tuple[float, float, float]
    task_projection_progress: float
    task_projection_arc_length_m: float
    projection_tangent_unit_xyz: tuple[float, float, float]
    command_tangent_unit_xyz: tuple[float, float, float]
    task_path_length_m: float
    recovery_hover_target_xyz_m: tuple[float, float, float]
    committed_recovery_outward_normal_xyz: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class PhysicalPreStepEvidence:
    key: ControlStepKey
    force_source_native_sample_index: int | None
    force_source_time_ns: int
    requested_tangential_step_m: float
    requested_cross_track_correction_delta_xyz_m: tuple[float, float, float]
    state: ToolStateProvenance


@dataclass(frozen=True)
class PhysicalNativeEvidence:
    key: ControlStepKey
    command_id: str
    native_sample_index: int
    substep_index: int
    state: ToolStateProvenance


@dataclass(frozen=True)
class PhysicalPostStepEvidence:
    key: ControlStepKey
    last_native_sample_index: int
    last_native_time_ns: int
    state: ToolStateProvenance


@dataclass(frozen=True)
class PhysicalCommandReadbackEvidence:
    key: ControlStepKey
    command_id: str
    normalized_action: tuple[float, ...]
    arm_normalized_action: tuple[float, float, float, float, float, float]
    arm_target_position_before_m: tuple[float, float, float]
    arm_target_position_after_m: tuple[float, float, float]
    arm_target_quaternion_before_wxyz: tuple[float, float, float, float]
    arm_target_quaternion_after_wxyz: tuple[float, float, float, float]
    arm_root_quaternion_world_wxyz: tuple[float, float, float, float]
    target_cartesian_delta_root_xyz_m: tuple[float, float, float]
    target_cartesian_delta_world_xyz_m: tuple[float, float, float]
    ik_solution_qpos: tuple[float, ...]
    controller_target_qpos: tuple[float, ...]
    articulation_drive_target_qpos: tuple[float, ...]
    gripper_command: float


@dataclass(frozen=True)
class EnvironmentStepOutcome:
    reward: float
    terminated: bool
    truncated: bool
    tracking_force_n: float


@dataclass(frozen=True)
class PhysicalIntervalEvidence:
    pre: PhysicalPreStepEvidence
    readback: PhysicalCommandReadbackEvidence
    native: tuple[PhysicalNativeEvidence, ...]
    post: PhysicalPostStepEvidence
    environment_outcome: EnvironmentStepOutcome


@dataclass(frozen=True)
class RawIntervalEvent:
    event: str
    key: ControlStepKey
    command_id: str | None
    physical_pre: PhysicalPreStepEvidence | None = None
    physical_native: PhysicalNativeEvidence | None = None
    physical_post: PhysicalPostStepEvidence | None = None
    readback: PhysicalCommandReadbackEvidence | None = None
    environment_outcome: EnvironmentStepOutcome | None = None


@dataclass(frozen=True)
class PhysicalIntervalFault:
    key: ControlStepKey
    stage: str
    exception_type: str
    message: str
    command_id: str | None
    pre: PhysicalPreStepEvidence | None
    readback: PhysicalCommandReadbackEvidence | None
    native: tuple[PhysicalNativeEvidence, ...]
    post: PhysicalPostStepEvidence | None
    environment_outcome: EnvironmentStepOutcome | None


RawIntervalSink = Callable[[RawIntervalEvent], None]
IntervalFaultSink = Callable[[PhysicalIntervalFault], None]
TangentialRequestSource = Callable[[ControlStepKey, SupervisorSnapshot], float]
SurfaceNormalSource = Callable[[float], Any]


@dataclass(frozen=True)
class _Geometry:
    instantaneous_progress: float
    geometric_track_error_m: float
    recovery_hover_pose_error_m: float
    recovery_clearance_m: float
    signed_recovery_clearance_m: float
    surface_reference_point_xyz_m: np.ndarray
    projection_outward_normal_xyz: np.ndarray
    command_outward_normal_xyz: np.ndarray
    committed_recovery_outward_normal_xyz: np.ndarray
    task_projection_point_xyz_m: np.ndarray
    task_projection_progress: float
    task_projection_arc_length_m: float
    projection_tangent_unit_xyz: np.ndarray
    command_tangent_unit_xyz: np.ndarray
    task_path_length_m: float
    recovery_hover_target_xyz_m: np.ndarray


class SurfacePathGeometry:
    """Pure geometry adapter around the frozen V4 ``ScenarioPath`` API."""

    def __init__(
        self,
        path: Any,
        surface_normal_source: SurfaceNormalSource,
        *,
        surface_top_origin_z_m: float,
        recovery_hover_clearance_m: float,
    ) -> None:
        if not callable(getattr(path, "project", None)) or not callable(
            getattr(path, "at", None)
        ):
            raise V6SapienSandboxContractError("path must expose project() and at()")
        if not callable(surface_normal_source):
            raise V6SapienSandboxContractError("surface normal source must be callable")
        if not math.isfinite(float(surface_top_origin_z_m)):
            raise V6SapienSandboxContractError("surface origin must be finite")
        if not math.isfinite(float(recovery_hover_clearance_m)) or float(
            recovery_hover_clearance_m
        ) <= 0.0:
            raise V6SapienSandboxContractError("hover clearance must be positive")
        self.path = path
        self.surface_normal_source = surface_normal_source
        self.surface_top_origin_z_m = float(surface_top_origin_z_m)
        self.recovery_hover_clearance_m = float(recovery_hover_clearance_m)

    def evaluate(
        self,
        contact_tool_position_xyz_m: Any,
        *,
        committed_progress: float,
    ) -> _Geometry:
        tool = _array(
            contact_tool_position_xyz_m,
            name="contact-tool position",
            length=3,
        )
        relative = tool.copy()
        relative[2] -= self.surface_top_origin_z_m
        projection = self.path.project(relative)
        progress = float(projection.progress)
        if not 0.0 <= progress <= 1.0:
            raise V6SapienSandboxContractError("path projection progress is invalid")
        projected = _array(projection.point_xyz, name="projected path point", length=3)
        projection_normal = _unit(
            self.surface_normal_source(float(projected[0])),
            name="projected outward normal",
        )
        projection_tangent = _unit(
            projection.tangent_xyz,
            name="projected path tangent",
        )
        projection_tangent = projection_tangent - float(
            np.dot(projection_tangent, projection_normal)
        ) * projection_normal
        projection_tangent = _unit(
            projection_tangent,
            name="surface-projected path tangent",
        )
        projected_world = projected.copy()
        projected_world[2] += self.surface_top_origin_z_m
        residual = relative - projected
        lateral_residual = residual - float(
            np.dot(residual, projection_normal)
        ) * projection_normal
        geometric_error = float(np.linalg.norm(lateral_residual))

        reference_relative, reference_tangent = self.path.at(float(committed_progress))
        reference_relative = _array(
            reference_relative, name="recovery reference point", length=3
        )
        command_tangent = _unit(reference_tangent, name="command path tangent")
        command_normal = _unit(
            self.surface_normal_source(float(reference_relative[0])),
            name="command-reference outward normal",
        )
        command_tangent = command_tangent - float(
            np.dot(command_tangent, command_normal)
        ) * command_normal
        command_tangent = _unit(command_tangent, name="command surface tangent")
        reference_world = reference_relative.copy()
        reference_world[2] += self.surface_top_origin_z_m
        hover_target = (
            reference_world + self.recovery_hover_clearance_m * command_normal
        )
        signed_clearance = float(np.dot(tool - reference_world, command_normal))
        task_path_length = float(self.path.total_length)
        if not math.isfinite(task_path_length) or task_path_length <= 0.0:
            raise V6SapienSandboxContractError("path total_length must be positive")
        return _Geometry(
            instantaneous_progress=progress,
            geometric_track_error_m=geometric_error,
            recovery_hover_pose_error_m=float(np.linalg.norm(hover_target - tool)),
            recovery_clearance_m=max(0.0, signed_clearance),
            signed_recovery_clearance_m=signed_clearance,
            surface_reference_point_xyz_m=reference_world,
            projection_outward_normal_xyz=projection_normal,
            command_outward_normal_xyz=command_normal,
            committed_recovery_outward_normal_xyz=command_normal,
            task_projection_point_xyz_m=projected_world,
            task_projection_progress=progress,
            task_projection_arc_length_m=progress * task_path_length,
            projection_tangent_unit_xyz=projection_tangent,
            command_tangent_unit_xyz=command_tangent,
            task_path_length_m=task_path_length,
            recovery_hover_target_xyz_m=hover_target,
        )


class SapienSandboxBinding:
    """Bind one V6 supervisor to one continuous SAPIEN control stream."""

    def __init__(
        self,
        env: Any,
        raw_env: Any,
        supervisor: UnifiedCausalForceRecoverySupervisor,
        geometry: SurfacePathGeometry,
        *,
        run_id: str,
        scenario_id: int,
        evaluation_id: int,
        target_force_n: float,
        bootstrap_id: str,
        task_path_id: str,
        surface_id: str,
        segment_index: int = 0,
        tangential_request_source: TangentialRequestSource | None = None,
        raw_interval_sink: RawIntervalSink | None = None,
        interval_fault_sink: IntervalFaultSink | None = None,
        config: SapienSandboxBindingConfig = SapienSandboxBindingConfig(),
    ) -> None:
        config.validate()
        if not isinstance(supervisor, UnifiedCausalForceRecoverySupervisor):
            raise V6SapienSandboxContractError("unsupported V6 supervisor")
        if not math.isclose(
            config.contact_threshold_n,
            float(supervisor.config.contact_threshold_n),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise V6SapienSandboxContractError(
                "binding and supervisor contact thresholds differ"
            )
        if not math.isclose(
            geometry.recovery_hover_clearance_m,
            float(supervisor.config.recovery_hover_clearance_m),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise V6SapienSandboxContractError(
                "binding geometry and supervisor recovery-hover clearances differ"
            )
        if not isinstance(run_id, str) or not run_id.strip():
            raise V6SapienSandboxContractError("run_id must be nonempty")
        for name, value in (
            ("scenario_id", scenario_id),
            ("evaluation_id", evaluation_id),
            ("segment_index", segment_index),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise V6SapienSandboxContractError(f"{name} must be nonnegative integer")
        if not math.isfinite(float(target_force_n)) or float(target_force_n) <= 0.0:
            raise V6SapienSandboxContractError("target force must be positive")
        for name, value in (
            ("bootstrap_id", bootstrap_id),
            ("task_path_id", task_path_id),
            ("surface_id", surface_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise V6SapienSandboxContractError(f"{name} must be nonempty")
        if tangential_request_source is not None and not callable(tangential_request_source):
            raise V6SapienSandboxContractError("tangential request source must be callable")
        if raw_interval_sink is not None and not callable(raw_interval_sink):
            raise V6SapienSandboxContractError("raw interval sink must be callable")
        if interval_fault_sink is not None and not callable(interval_fault_sink):
            raise V6SapienSandboxContractError("fault sink must be callable")
        self.env = env
        self.raw = raw_env
        self.supervisor = supervisor
        self.geometry = geometry
        self.config = config
        self.run_id = run_id
        self.scenario_id = int(scenario_id)
        self.evaluation_id = int(evaluation_id)
        self.segment_index = int(segment_index)
        self.target_force_n = float(target_force_n)
        self.bootstrap_id = bootstrap_id
        self.task_path_id = task_path_id
        self.surface_id = surface_id
        self.tangential_request_source = tangential_request_source
        self.raw_interval_sink = raw_interval_sink
        self.interval_fault_sink = interval_fault_sink
        self._validate_runtime_contract()
        adapter_config = AdapterConfig(
            expected_native_steps_per_control=1,
            control_period_ns=10_000_000,
            native_period_ns=10_000_000,
            command_readback_tolerance_m=config.command_readback_tolerance_m,
            rotation_readback_tolerance_rad=config.command_readback_tolerance_m,
            force_alignment_tolerance_n=config.force_alignment_tolerance_n,
            state_alignment_tolerance=config.pose_alignment_tolerance_m,
            quaternion_tolerance=config.quaternion_readback_tolerance,
            maximum_task_tangential_step_m=config.maximum_task_tangent_step_m,
            maximum_recovery_reposition_step_m=(
                config.maximum_recovery_reposition_step_m
            ),
            maximum_recovery_rotation_step_rad=0.0,
            recovery_hover_clearance_m=geometry.recovery_hover_clearance_m,
            expected_command_frame_id=config.command_frame_id,
            expected_command_mode=config.command_mode,
            required_readback_stage=config.readback_stage,
        )
        self.adapter = CausalControlAdapter(
            supervisor,
            self._build_issued_command,
            config=adapter_config,
        )
        self._next_control_step = 0
        self._next_native_sample = 0
        self._time_ns = 0
        self._last_post: PhysicalPostStepEvidence | None = None
        self._last_bundle: AdapterStepBundle | None = None
        self._pre_by_key: dict[ControlStepKey, PhysicalPreStepEvidence] = {}
        self._geometry_by_key: dict[ControlStepKey, _Geometry] = {}
        self._pending_readback: PhysicalCommandReadbackEvidence | None = None
        self._pending_native: list[PhysicalNativeEvidence] = []
        self._pending_post: PhysicalPostStepEvidence | None = None
        self._pending_environment_outcome: EnvironmentStepOutcome | None = None
        self._pending_command_id: str | None = None
        self._intervals: list[PhysicalIntervalEvidence] = []
        self._fault_emitted = False
        self._binding_faulted = False
        self._environment_terminal = False

    @property
    def physical_intervals(self) -> tuple[PhysicalIntervalEvidence, ...]:
        return tuple(self._intervals)

    def _validate_runtime_contract(self) -> None:
        observed = {
            "sim": getattr(self.raw, "_sim_freq", None),
            "control": getattr(self.raw, "_control_freq", None),
            "ratio": getattr(self.raw, "_sim_steps_per_control", None),
        }
        expected = {"sim": 100, "control": 100, "ratio": 1}
        if observed != expected:
            raise V6SapienSandboxContractError(
                f"noncanonical SAPIEN timing contract: {observed}, expected {expected}"
            )
        if not callable(getattr(self.raw, "_normal_force", None)):
            raise V6SapienSandboxContractError("raw environment lacks _normal_force()")
        if not callable(getattr(self.raw, "_after_simulation_step", None)):
            raise V6SapienSandboxContractError(
                "raw environment lacks _after_simulation_step()"
            )
        if not callable(getattr(getattr(self.raw, "agent", None), "set_action", None)):
            raise V6SapienSandboxContractError("raw agent lacks set_action()")
        if not callable(getattr(self.env, "step", None)):
            raise V6SapienSandboxContractError("environment lacks step()")
        if not hasattr(self.raw, "v4_tool"):
            raise V6SapienSandboxContractError("raw environment lacks compliant v4_tool")
        if not hasattr(getattr(self.raw, "agent", None), "tcp"):
            raise V6SapienSandboxContractError("raw environment lacks agent.tcp")
        for owner, label in (
            (self.raw.v4_tool, "v4_tool"),
            (self.raw.agent.tcp, "agent.tcp"),
        ):
            for attribute in ("linear_velocity", "angular_velocity"):
                if not hasattr(owner, attribute):
                    raise V6SapienSandboxContractError(
                        f"{label} lacks native {attribute} provenance"
                    )
        shape = tuple(getattr(getattr(self.env, "action_space", None), "shape", ()))
        if shape != (7,):
            raise V6SapienSandboxContractError(
                "first V6 SANDBOX requires one flat seven-field action"
            )
        arm_controller, _mapping = self._arm_controller()
        if not hasattr(arm_controller, "root_link"):
            raise V6SapienSandboxContractError("arm controller lacks root_link")
        if not callable(
            getattr(getattr(arm_controller, "kinematics", None), "compute_ik", None)
        ):
            raise V6SapienSandboxContractError("arm controller lacks compute_ik")
        articulation = getattr(arm_controller, "articulation", None)
        if not callable(getattr(articulation, "get_drive_targets", None)):
            raise V6SapienSandboxContractError(
                "arm articulation lacks get_drive_targets()"
            )
        if not hasattr(arm_controller, "active_joint_indices"):
            raise V6SapienSandboxContractError(
                "arm controller lacks active_joint_indices"
            )
        arm_config = getattr(arm_controller, "config", None)
        if arm_config is None:
            raise V6SapienSandboxContractError("arm controller lacks config")
        if getattr(arm_config, "frame", None) != (
            "root_translation:root_aligned_body_rotation"
        ):
            raise V6SapienSandboxContractError(
                "arm translation must use the robot-root frame"
            )
        if getattr(arm_config, "normalize_action", None) is not True:
            raise V6SapienSandboxContractError("arm action must be normalized")
        if getattr(arm_config, "use_delta", None) is not True:
            raise V6SapienSandboxContractError("arm action must use delta pose")
        if getattr(arm_config, "interpolate", None) is not False:
            raise V6SapienSandboxContractError(
                "first V6 SANDBOX requires non-interpolated drive targets"
            )
        for attribute, expected in (("pos_lower", -0.1), ("pos_upper", 0.1)):
            observed = np.asarray(getattr(arm_config, attribute, np.nan), dtype=np.float64)
            if observed.size not in (1, 3) or not np.allclose(
                observed,
                expected,
                rtol=0.0,
                atol=1e-12,
            ):
                raise V6SapienSandboxContractError(
                    "arm normalized position scale is not exactly +/-0.1 m"
                )

    def _key(self) -> ControlStepKey:
        return ControlStepKey(
            run_id=self.run_id,
            scenario_id=self.scenario_id,
            evaluation_id=self.evaluation_id,
            segment_index=self.segment_index,
            control_step_index=self._next_control_step,
        )

    def _read_pose(self, owner: Any, *, name: str) -> tuple[np.ndarray, np.ndarray]:
        try:
            pose = owner.pose
        except AttributeError as exc:
            raise V6SapienSandboxContractError(f"{name} lacks pose") from exc
        return (
            _array(pose.p, name=f"{name} position", length=3),
            _quaternion(pose.q, name=f"{name} quaternion"),
        )

    def _read_motion(
        self, owner: Any, *, name: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position, quaternion = self._read_pose(owner, name=name)
        linear = _array(
            owner.linear_velocity,
            name=f"{name} linear velocity",
            length=3,
        )
        angular = _array(
            owner.angular_velocity,
            name=f"{name} angular velocity",
            length=3,
        )
        return position, quaternion, linear, angular

    def _arm_root_quaternion_world(self, arm_controller: Any) -> np.ndarray:
        _position, quaternion = self._read_pose(
            arm_controller.root_link,
            name="arm_controller.root_link",
        )
        return quaternion

    def _read_raw_state(
        self,
        *,
        time_ns: int,
        committed_progress: float,
    ) -> ToolStateProvenance:
        force = _scalar(self.raw._normal_force(), name="normal force")
        if force < 0.0:
            raise V6SapienSandboxContractError("normal force must be nonnegative")
        (
            tool_position,
            tool_quaternion,
            tool_linear_velocity,
            tool_angular_velocity,
        ) = self._read_motion(self.raw.v4_tool, name="v4_tool")
        (
            tcp_position,
            tcp_quaternion,
            tcp_linear_velocity,
            tcp_angular_velocity,
        ) = self._read_motion(
            self.raw.agent.tcp,
            name="agent.tcp",
        )
        geometry = self.geometry.evaluate(
            tool_position,
            committed_progress=float(committed_progress),
        )
        # Store the full vector; the scalar projected value is supplied to the
        # supervisor when the pre-step packet is assembled.
        state = ToolStateProvenance(
            time_ns=int(time_ns),
            measured_force_n=force,
            contact_observed=bool(force >= self.config.contact_threshold_n),
            contact_tool_position_xyz_m=_tuple(tool_position),
            contact_tool_quaternion_wxyz=_tuple(tool_quaternion),
            contact_tool_linear_velocity_xyz_m_s=_tuple(tool_linear_velocity),
            contact_tool_angular_velocity_xyz_rad_s=_tuple(tool_angular_velocity),
            robot_tcp_position_xyz_m=_tuple(tcp_position),
            robot_tcp_quaternion_wxyz=_tuple(tcp_quaternion),
            robot_tcp_linear_velocity_xyz_m_s=_tuple(tcp_linear_velocity),
            robot_tcp_angular_velocity_xyz_rad_s=_tuple(tcp_angular_velocity),
            instantaneous_progress=geometry.instantaneous_progress,
            geometric_track_error_m=geometry.geometric_track_error_m,
            recovery_hover_pose_error_m=geometry.recovery_hover_pose_error_m,
            recovery_clearance_m=geometry.recovery_clearance_m,
            signed_recovery_clearance_m=geometry.signed_recovery_clearance_m,
            surface_reference_point_xyz_m=_tuple(
                geometry.surface_reference_point_xyz_m
            ),
            projection_outward_normal_xyz=_tuple(
                geometry.projection_outward_normal_xyz
            ),
            command_outward_normal_xyz=_tuple(
                geometry.command_outward_normal_xyz
            ),
            committed_recovery_outward_normal_xyz=_tuple(
                geometry.committed_recovery_outward_normal_xyz
            ),
            task_projection_point_xyz_m=_tuple(
                geometry.task_projection_point_xyz_m
            ),
            task_projection_progress=geometry.task_projection_progress,
            task_projection_arc_length_m=geometry.task_projection_arc_length_m,
            projection_tangent_unit_xyz=_tuple(
                geometry.projection_tangent_unit_xyz
            ),
            command_tangent_unit_xyz=_tuple(
                geometry.command_tangent_unit_xyz
            ),
            task_path_length_m=geometry.task_path_length_m,
            recovery_hover_target_xyz_m=_tuple(geometry.recovery_hover_target_xyz_m),
        )
        return state

    @staticmethod
    def _normal_velocity(state: ToolStateProvenance) -> float:
        velocity = np.asarray(
            state.contact_tool_linear_velocity_xyz_m_s,
            dtype=np.float64,
        )
        normal = np.asarray(
            state.projection_outward_normal_xyz,
            dtype=np.float64,
        )
        return float(np.dot(velocity, normal))

    def _request_tangent(self, key: ControlStepKey, snapshot: SupervisorSnapshot) -> float:
        value = (
            0.0
            if self.tangential_request_source is None
            else float(self.tangential_request_source(key, snapshot))
        )
        if not math.isfinite(value) or not 0.0 <= value <= (
            self.config.maximum_task_tangent_step_m
        ):
            raise V6SapienSandboxContractError(
                "pre-step task tangent request exceeds the frozen bound"
            )
        return value

    def _build_pre_step(self) -> tuple[CausalPreStepSample, PhysicalPreStepEvidence]:
        key = self._key()
        snapshot = self.supervisor.snapshot()
        if self._last_post is None:
            live = self._read_raw_state(
                time_ns=0,
                committed_progress=snapshot.committed_progress,
            )
            source_index = None
            source_time = 0
            controller_force = live.measured_force_n
            state_source_kind = "bootstrap_sensor_snapshot"
        else:
            live = self._read_raw_state(
                time_ns=self._time_ns,
                committed_progress=snapshot.committed_progress,
            )
            prior = self._last_post
            if abs(live.measured_force_n - prior.state.measured_force_n) > (
                self.config.force_alignment_tolerance_n
            ):
                raise V6SapienSandboxContractError(
                    "live pre-step force differs from preceding native audit"
                )
            for label, current, previous in (
                (
                    "contact-tool position",
                    live.contact_tool_position_xyz_m,
                    prior.state.contact_tool_position_xyz_m,
                ),
                (
                    "contact-tool linear velocity",
                    live.contact_tool_linear_velocity_xyz_m_s,
                    prior.state.contact_tool_linear_velocity_xyz_m_s,
                ),
                (
                    "contact-tool angular velocity",
                    live.contact_tool_angular_velocity_xyz_rad_s,
                    prior.state.contact_tool_angular_velocity_xyz_rad_s,
                ),
                (
                    "robot-TCP position",
                    live.robot_tcp_position_xyz_m,
                    prior.state.robot_tcp_position_xyz_m,
                ),
                (
                    "robot-TCP linear velocity",
                    live.robot_tcp_linear_velocity_xyz_m_s,
                    prior.state.robot_tcp_linear_velocity_xyz_m_s,
                ),
                (
                    "robot-TCP angular velocity",
                    live.robot_tcp_angular_velocity_xyz_rad_s,
                    prior.state.robot_tcp_angular_velocity_xyz_rad_s,
                ),
            ):
                if not np.allclose(
                    current,
                    previous,
                    rtol=0.0,
                    atol=self.config.pose_alignment_tolerance_m,
                ):
                    raise V6SapienSandboxContractError(
                        f"live pre-step {label} differs from preceding post-step"
                    )
            for label, current, previous in (
                (
                    "contact-tool quaternion",
                    live.contact_tool_quaternion_wxyz,
                    prior.state.contact_tool_quaternion_wxyz,
                ),
                (
                    "robot-TCP quaternion",
                    live.robot_tcp_quaternion_wxyz,
                    prior.state.robot_tcp_quaternion_wxyz,
                ),
            ):
                if not _quaternion_same(
                    np.asarray(current, dtype=np.float64),
                    np.asarray(previous, dtype=np.float64),
                    self.config.quaternion_readback_tolerance,
                ):
                    raise V6SapienSandboxContractError(
                        f"live pre-step {label} differs from preceding post-step"
                    )
            # Geometry is recomputed at the current supervisor high-water
            # reference, while physical pose and force remain causally tied to
            # the preceding native sample.
            source_index = prior.last_native_sample_index
            source_time = prior.last_native_time_ns
            controller_force = prior.state.measured_force_n
            state_source_kind = "prior_native_sample"
        request = self._request_tangent(key, snapshot)
        physical = PhysicalPreStepEvidence(
            key=key,
            force_source_native_sample_index=source_index,
            force_source_time_ns=source_time,
            requested_tangential_step_m=request,
            requested_cross_track_correction_delta_xyz_m=(0.0, 0.0, 0.0),
            state=live,
        )
        geometry = self.geometry.evaluate(
            live.contact_tool_position_xyz_m,
            committed_progress=snapshot.committed_progress,
        )
        self._geometry_by_key[key] = geometry
        self._pre_by_key[key] = physical
        recovery_delta = geometry.recovery_hover_target_xyz_m - np.asarray(
            live.contact_tool_position_xyz_m,
            dtype=np.float64,
        )
        recovery_normal_component = float(
            np.dot(
                recovery_delta,
                geometry.committed_recovery_outward_normal_xyz,
            )
        )
        if recovery_normal_component < 0.0:
            # Recovery authority is never allowed to request inward motion.
            recovery_delta = (
                recovery_delta
                - recovery_normal_component
                * geometry.committed_recovery_outward_normal_xyz
            )
        recovery_delta = _bounded(
            recovery_delta,
            self.config.maximum_recovery_reposition_step_m,
        )
        candidates = {
            "key": key,
            "pre_time_ns": int(self._time_ns),
            "state_source_kind": state_source_kind,
            "force_source_native_sample_index": source_index,
            "force_source_time_ns": int(source_time),
            "bootstrap_id": self.bootstrap_id,
            "measured_force_n": float(controller_force),
            "target_force_n": self.target_force_n,
            "normal_velocity_outward_m_s": self._normal_velocity(live),
            "contact_observed": bool(controller_force >= self.config.contact_threshold_n),
            "instantaneous_progress": geometry.instantaneous_progress,
            "geometric_track_error_m": geometry.geometric_track_error_m,
            "requested_tangential_step_m": request,
            "recovery_hover_pose_error_m": geometry.recovery_hover_pose_error_m,
            "recovery_clearance_m": geometry.recovery_clearance_m,
            "geometry_frame_id": self.config.geometry_frame_id,
            "command_frame_id": self.config.command_frame_id,
            "contact_tool_frame_id": self.config.contact_tool_frame_id,
            "robot_tcp_frame_id": self.config.robot_tcp_frame_id,
            "task_path_id": self.task_path_id,
            "surface_id": self.surface_id,
            "surface_reference_point_xyz_m": _tuple(
                geometry.surface_reference_point_xyz_m
            ),
            "projection_outward_normal_xyz": _tuple(
                geometry.projection_outward_normal_xyz
            ),
            "command_outward_normal_xyz": _tuple(
                geometry.command_outward_normal_xyz
            ),
            "committed_recovery_outward_normal_xyz": _tuple(
                geometry.committed_recovery_outward_normal_xyz
            ),
            "task_projection_point_xyz_m": _tuple(
                geometry.task_projection_point_xyz_m
            ),
            "task_projection_progress": geometry.task_projection_progress,
            "task_projection_arc_length_m": geometry.task_projection_arc_length_m,
            "projection_tangent_unit_xyz": _tuple(
                geometry.projection_tangent_unit_xyz
            ),
            "command_tangent_unit_xyz": _tuple(
                geometry.command_tangent_unit_xyz
            ),
            "task_path_length_m": geometry.task_path_length_m,
            "recovery_hover_target_position_xyz_m": _tuple(
                geometry.recovery_hover_target_xyz_m
            ),
            "requested_recovery_delta_xyz_m": _tuple(recovery_delta),
            "requested_cross_track_correction_delta_xyz_m": (0.0, 0.0, 0.0),
            "requested_recovery_rotation_delta_euler_xyz_rad": (0.0, 0.0, 0.0),
            "contact_tool_position_xyz_m": live.contact_tool_position_xyz_m,
            "contact_tool_quaternion_wxyz": live.contact_tool_quaternion_wxyz,
            "contact_tool_linear_velocity_xyz_m_s": (
                live.contact_tool_linear_velocity_xyz_m_s
            ),
            "contact_tool_angular_velocity_xyz_rad_s": (
                live.contact_tool_angular_velocity_xyz_rad_s
            ),
            "robot_tcp_position_xyz_m": live.robot_tcp_position_xyz_m,
            "robot_tcp_quaternion_wxyz": live.robot_tcp_quaternion_wxyz,
            "robot_tcp_linear_velocity_xyz_m_s": (
                live.robot_tcp_linear_velocity_xyz_m_s
            ),
            "robot_tcp_angular_velocity_xyz_rad_s": (
                live.robot_tcp_angular_velocity_xyz_rad_s
            ),
        }
        return _construct_packet(CausalPreStepSample, candidates), physical

    def _build_issued_command(
        self,
        command: SupervisorCommand,
        pre: CausalPreStepSample,
    ) -> IssuedCartesianCommand:
        geometry = self._geometry_by_key.get(pre.key)
        physical = self._pre_by_key.get(pre.key)
        if geometry is None or physical is None:
            raise V6SapienSandboxContractError("issued command lacks pre-step geometry")
        normal_step = float(command.executed_normal_step_m)
        tangent_step = float(command.executed_tangential_step_m)
        cross_track_permitted = bool(
            getattr(command, "cross_track_correction_permitted", False)
        )
        normal = geometry.command_outward_normal_xyz
        tangent = geometry.command_tangent_unit_xyz
        if tangent_step < -1e-15 or tangent_step > (
            self.config.maximum_task_tangent_step_m + 1e-15
        ):
            raise V6SapienSandboxContractError("supervisor tangent exceeds binding bound")
        if (
            command.recovery_reposition_permitted
            and cross_track_permitted
        ):
            raise V6SapienSandboxContractError(
                "recovery and cross-track authorities overlap"
            )
        cross_track_delta = np.zeros(3, dtype=np.float64)
        if command.recovery_reposition_permitted:
            if abs(normal_step) > 1e-15 or abs(tangent_step) > 1e-15:
                raise V6SapienSandboxContractError(
                    "recovery reposition overlaps normal or task-tangent authority"
                )
            normal_delta = np.zeros(3, dtype=np.float64)
            task_delta = np.zeros(3, dtype=np.float64)
            recovery_delta = np.asarray(
                pre.requested_recovery_delta_xyz_m,
                dtype=np.float64,
            )
        elif cross_track_permitted:
            if abs(normal_step) > 1e-15 or abs(tangent_step) > 1e-15:
                raise V6SapienSandboxContractError(
                    "cross-track correction overlaps normal or task-tangent authority"
                )
            normal_delta = np.zeros(3, dtype=np.float64)
            task_delta = np.zeros(3, dtype=np.float64)
            recovery_delta = np.zeros(3, dtype=np.float64)
            cross_track_delta = np.asarray(
                pre.requested_cross_track_correction_delta_xyz_m,
                dtype=np.float64,
            )
        else:
            normal_delta = -normal * normal_step
            task_delta = tangent * tangent_step
            recovery_delta = np.zeros(3, dtype=np.float64)
        delta = normal_delta + task_delta + recovery_delta + cross_track_delta
        if str(command.state_after) == "SAFE_HOLD" and float(np.linalg.norm(delta)) > 1e-15:
            raise V6SapienSandboxContractError("SAFE_HOLD attempted Cartesian motion")
        normalized = delta / self.config.position_action_scale_m
        if np.any(np.abs(normalized) > 1.0 + 1e-12):
            raise V6SapienSandboxContractError("Cartesian command would be clipped downstream")
        command_id = ":".join(
            (
                pre.key.run_id,
                str(pre.key.scenario_id),
                str(pre.key.evaluation_id),
                str(pre.key.segment_index),
                str(pre.key.control_step_index),
            )
        )
        candidates = {
            "key": pre.key,
            "command_id": command_id,
            "issued_time_ns": int(pre.pre_time_ns),
            "command_frame_id": self.config.command_frame_id,
            "command_mode": self.config.command_mode,
            "normal_step_m": normal_step,
            "task_tangential_step_m": tangent_step,
            "recovery_reposition_permitted": bool(
                command.recovery_reposition_permitted
            ),
            "cross_track_correction_permitted": bool(
                cross_track_permitted
            ),
            "normal_delta_xyz_m": _tuple(normal_delta),
            "task_delta_xyz_m": _tuple(task_delta),
            "recovery_delta_xyz_m": _tuple(recovery_delta),
            "cross_track_correction_delta_xyz_m": _tuple(cross_track_delta),
            "cartesian_delta_xyz_m": _tuple(delta),
            "recovery_rotation_delta_euler_xyz_rad": (0.0, 0.0, 0.0),
            "rotation_delta_euler_xyz_rad": (0.0, 0.0, 0.0),
        }
        issued = _construct_packet(IssuedCartesianCommand, candidates)
        self._pending_command_id = command_id
        return issued

    def _arm_controller(self) -> tuple[Any, tuple[int, int]]:
        controller = getattr(self.raw.agent, "controller", None)
        controllers = getattr(controller, "controllers", None)
        mapping = getattr(controller, "action_mapping", None)
        if not isinstance(controllers, dict) or "arm" not in controllers:
            raise V6SapienSandboxContractError("combined controller lacks arm controller")
        if not isinstance(mapping, dict) or "arm" not in mapping:
            raise V6SapienSandboxContractError("combined controller lacks arm action mapping")
        start, end = mapping["arm"]
        if (int(start), int(end)) != (0, 6):
            raise V6SapienSandboxContractError("arm action must occupy the first six fields")
        if tuple(mapping.get("gripper", ())) != (6, 7):
            raise V6SapienSandboxContractError(
                "gripper action must occupy the seventh field"
            )
        return controllers["arm"], (int(start), int(end))

    def _full_action(
        self,
        issued: IssuedCartesianCommand,
    ) -> tuple[np.ndarray, np.ndarray]:
        shape = tuple(getattr(getattr(self.env, "action_space", None), "shape", ()))
        if shape != (7,):
            raise V6SapienSandboxContractError("environment action space must be flat 7D")
        action = np.zeros(shape, dtype=np.float32)
        world_delta = np.asarray(issued.cartesian_delta_xyz_m, dtype=np.float64)
        arm_controller, _mapping = self._arm_controller()
        root_quaternion_world = self._arm_root_quaternion_world(arm_controller)
        root_delta = _quat_rotate(_quat_conjugate(root_quaternion_world), world_delta)
        normalized = root_delta / self.config.position_action_scale_m
        if np.any(np.abs(normalized) > 1.0 + 1e-12):
            raise V6SapienSandboxContractError("issued action exceeds normalized position bounds")
        action[:3] = normalized.astype(np.float32)
        action[3:6] = 0.0
        action[-1] = np.float32(self.config.gripper_command)
        return action, root_quaternion_world

    def _selected_arm_drive_targets(self, arm_controller: Any) -> np.ndarray:
        indices = _index_vector(
            arm_controller.active_joint_indices,
            name="arm active_joint_indices",
        )
        all_targets = _flat_vector(
            arm_controller.articulation.get_drive_targets(),
            name="articulation drive targets",
        )
        if int(np.max(indices)) >= all_targets.size:
            raise V6SapienSandboxContractError(
                "arm active joint index exceeds articulation drive targets"
            )
        return all_targets[indices].copy()

    def _make_readback(
        self,
        issued: IssuedCartesianCommand,
        actual_action: np.ndarray,
        before_position: np.ndarray,
        before_quaternion: np.ndarray,
        after_target_position: np.ndarray,
        after_target_quaternion: np.ndarray,
        root_quaternion_world: np.ndarray,
        ik_solution_qpos: np.ndarray,
        controller_target_qpos: np.ndarray,
        articulation_drive_target_qpos: np.ndarray,
    ) -> tuple[CommandReadback, PhysicalCommandReadbackEvidence]:
        arm_action = np.asarray(actual_action[:6], dtype=np.float64)
        if np.any(np.abs(arm_action) > 1.0 + 1e-12):
            raise V6SapienSandboxContractError("agent received out-of-range arm action")
        if not np.allclose(arm_action[3:], 0.0, rtol=0.0, atol=1e-12):
            raise V6SapienSandboxContractError("rotation is not frozen at zero")
        target_delta_root = after_target_position - before_position
        target_delta_world = _quat_rotate(root_quaternion_world, target_delta_root)
        issued_delta = np.asarray(issued.cartesian_delta_xyz_m, dtype=np.float64)
        if not np.allclose(
            target_delta_world,
            issued_delta,
            rtol=0.0,
            atol=self.config.command_readback_tolerance_m,
        ):
            raise V6SapienSandboxContractError(
                "arm target-pose readback differs from issued Cartesian command"
            )
        if not np.allclose(
            arm_action[:3] * self.config.position_action_scale_m,
            target_delta_root,
            rtol=0.0,
            atol=self.config.command_readback_tolerance_m,
        ):
            raise V6SapienSandboxContractError(
                "normalized action and arm target-pose readback disagree"
            )
        if not _quaternion_same(
            before_quaternion,
            after_target_quaternion,
            self.config.quaternion_readback_tolerance,
        ):
            raise V6SapienSandboxContractError(
                "zero rotation action changed the arm target quaternion"
            )
        geometry = self._geometry_by_key[issued.key]
        if (
            issued.recovery_reposition_permitted
            and issued.cross_track_correction_permitted
        ):
            raise V6SapienSandboxContractError(
                "readback cannot carry overlapping Cartesian authorities"
            )
        cross_track_delta = np.zeros(3, dtype=np.float64)
        if issued.recovery_reposition_permitted:
            normal_step = 0.0
            tangent_step = 0.0
            normal_delta = np.zeros(3, dtype=np.float64)
            task_delta = np.zeros(3, dtype=np.float64)
            recovery_delta = target_delta_world.copy()
        elif issued.cross_track_correction_permitted:
            normal_step = 0.0
            tangent_step = 0.0
            normal_delta = np.zeros(3, dtype=np.float64)
            task_delta = np.zeros(3, dtype=np.float64)
            recovery_delta = np.zeros(3, dtype=np.float64)
            cross_track_delta = target_delta_world.copy()
        else:
            normal_step = -float(
                np.dot(target_delta_world, geometry.command_outward_normal_xyz)
            )
            tangent_step = float(
                np.dot(target_delta_world, geometry.command_tangent_unit_xyz)
            )
            if tangent_step < -self.config.command_readback_tolerance_m:
                raise V6SapienSandboxContractError(
                    "actuator readback contains reverse task motion"
                )
            tangent_step = max(0.0, tangent_step)
            normal_delta = -geometry.command_outward_normal_xyz * normal_step
            task_delta = geometry.command_tangent_unit_xyz * tangent_step
            recovery_delta = np.zeros(3, dtype=np.float64)
            residual = target_delta_world - normal_delta - task_delta
            if float(np.linalg.norm(residual)) > self.config.command_readback_tolerance_m:
                raise V6SapienSandboxContractError(
                    "actuator readback contains motion outside normal/task authority"
                )
        candidates = {
            "key": issued.key,
            "command_id": issued.command_id,
            "readback_time_ns": int(self._time_ns),
            "accepted": True,
            "readback_stage": self.config.readback_stage,
            "command_frame_id": self.config.command_frame_id,
            "command_mode": self.config.command_mode,
            "normal_step_m": normal_step,
            "task_tangential_step_m": tangent_step,
            "recovery_reposition_permitted": issued.recovery_reposition_permitted,
            "cross_track_correction_permitted": (
                issued.cross_track_correction_permitted
            ),
            "normal_delta_xyz_m": _tuple(normal_delta),
            "task_delta_xyz_m": _tuple(task_delta),
            "recovery_delta_xyz_m": _tuple(recovery_delta),
            "cross_track_correction_delta_xyz_m": _tuple(cross_track_delta),
            "cartesian_delta_xyz_m": _tuple(target_delta_world),
            "recovery_rotation_delta_euler_xyz_rad": (0.0, 0.0, 0.0),
            "rotation_delta_euler_xyz_rad": (0.0, 0.0, 0.0),
        }
        readback = _construct_packet(CommandReadback, candidates)
        evidence = PhysicalCommandReadbackEvidence(
            key=issued.key,
            command_id=issued.command_id,
            normalized_action=_tuple(actual_action),
            arm_normalized_action=_tuple(arm_action),
            arm_target_position_before_m=_tuple(before_position),
            arm_target_position_after_m=_tuple(after_target_position),
            arm_target_quaternion_before_wxyz=_tuple(before_quaternion),
            arm_target_quaternion_after_wxyz=_tuple(after_target_quaternion),
            arm_root_quaternion_world_wxyz=_tuple(root_quaternion_world),
            target_cartesian_delta_root_xyz_m=_tuple(target_delta_root),
            target_cartesian_delta_world_xyz_m=_tuple(target_delta_world),
            ik_solution_qpos=_tuple(ik_solution_qpos),
            controller_target_qpos=_tuple(controller_target_qpos),
            articulation_drive_target_qpos=_tuple(articulation_drive_target_qpos),
            gripper_command=float(actual_action[-1]),
        )
        return readback, evidence

    def _emit_raw(self, event: RawIntervalEvent) -> None:
        if self.raw_interval_sink is not None:
            self.raw_interval_sink(event)

    def _emit_fault(self, key: ControlStepKey, error: BaseException) -> None:
        if self._fault_emitted:
            return
        self._fault_emitted = True
        if self.interval_fault_sink is None:
            return
        root_error = error
        seen: set[int] = set()
        while root_error.__cause__ is not None and id(root_error) not in seen:
            seen.add(id(root_error))
            root_error = root_error.__cause__
        self.interval_fault_sink(
            PhysicalIntervalFault(
                key=key,
                stage=str(self.adapter.fault_stage or "physical_binding"),
                exception_type=type(root_error).__name__,
                message=str(root_error),
                command_id=self._pending_command_id,
                pre=self._pre_by_key.get(key),
                readback=self._pending_readback,
                native=tuple(self._pending_native),
                post=self._pending_post,
                environment_outcome=self._pending_environment_outcome,
            )
        )

    def _execute(self, issued: IssuedCartesianCommand) -> BackendStepResult:
        action, root_quaternion_world = self._full_action(issued)
        original_set_action = self.raw.agent.set_action
        original_after_simulation_step = self.raw._after_simulation_step
        if getattr(original_set_action, "_v6_sandbox_binding", False) or getattr(
            original_after_simulation_step, "_v6_sandbox_binding", False
        ):
            raise V6SapienSandboxContractError("V6 physical hooks are already installed")
        readback_packet: CommandReadback | None = None
        readback_evidence: PhysicalCommandReadbackEvidence | None = None
        native_packets: list[NativeAuditSample] = []
        physical_native: list[PhysicalNativeEvidence] = []
        arm_controller, _arm_mapping = self._arm_controller()

        def wrapped_set_action(received_action: Any):
            nonlocal readback_packet, readback_evidence
            if readback_packet is not None:
                raise V6SapienSandboxContractError("agent.set_action called more than once")
            received = _array(
                received_action,
                name="action received by agent.set_action",
                length=7,
            )
            if not math.isclose(
                float(received[-1]),
                self.config.gripper_command,
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise V6SapienSandboxContractError(
                    "agent.set_action did not receive the frozen -1 gripper command"
                )
            use_target = bool(getattr(arm_controller.config, "use_target", False))
            prior_target = getattr(arm_controller, "_target_pose", None)
            before_pose = (
                prior_target
                if use_target and prior_target is not None
                else arm_controller.ee_pose_at_base
            )
            before = _array(
                before_pose.p,
                name="arm pre-target position",
                length=3,
            )
            before_quaternion = _quaternion(
                before_pose.q,
                name="arm pre-target quaternion",
            )
            current_root_quaternion_world = self._arm_root_quaternion_world(
                arm_controller
            )
            if not _quaternion_same(
                current_root_quaternion_world,
                root_quaternion_world,
                self.config.quaternion_readback_tolerance,
            ):
                raise V6SapienSandboxContractError(
                    "robot root frame changed between command construction and set_action"
                )
            ik_results: list[Any] = []
            original_compute_ik = arm_controller.kinematics.compute_ik

            def wrapped_compute_ik(*args: Any, **kwargs: Any):
                result = original_compute_ik(*args, **kwargs)
                ik_results.append(result)
                return result

            arm_controller.kinematics.compute_ik = wrapped_compute_ik
            try:
                result = original_set_action(received_action)
            finally:
                ik_hook_intact = (
                    arm_controller.kinematics.compute_ik is wrapped_compute_ik
                )
                arm_controller.kinematics.compute_ik = original_compute_ik
                if not ik_hook_intact:
                    raise V6SapienSandboxContractError(
                        "arm compute_ik hook was replaced"
                    )
            if len(ik_results) != 1:
                raise V6SapienSandboxContractError(
                    "agent.set_action did not produce exactly one arm IK result"
                )
            if ik_results[0] is None:
                raise V6SapienSandboxContractError("arm IK returned no solution")
            ik_solution = _flat_vector(ik_results[0], name="raw arm IK solution")
            controller_target = _flat_vector(
                getattr(arm_controller, "_target_qpos", None),
                name="arm controller target qpos",
            )
            start_qpos = _flat_vector(
                getattr(arm_controller, "_start_qpos", None),
                name="arm controller start qpos",
            )
            drive_target = self._selected_arm_drive_targets(arm_controller)
            if not (
                ik_solution.shape
                == controller_target.shape
                == start_qpos.shape
                == drive_target.shape
            ):
                raise V6SapienSandboxContractError(
                    "IK/controller/drive target dimensions disagree"
                )
            joint_tolerance = 1e-8
            if not np.allclose(
                ik_solution,
                controller_target,
                rtol=0.0,
                atol=joint_tolerance,
            ):
                raise V6SapienSandboxContractError(
                    "controller target qpos differs from raw IK solution"
                )
            if not np.allclose(
                controller_target,
                drive_target,
                rtol=0.0,
                atol=joint_tolerance,
            ):
                raise V6SapienSandboxContractError(
                    "articulation drive target was not updated to controller target qpos"
                )
            issued_motion = float(
                np.linalg.norm(np.asarray(issued.cartesian_delta_xyz_m, dtype=np.float64))
            )
            if issued_motion > self.config.command_readback_tolerance_m and np.allclose(
                controller_target,
                start_qpos,
                rtol=0.0,
                atol=joint_tolerance,
            ):
                raise V6SapienSandboxContractError(
                    "nonzero Cartesian command did not update the joint target"
                )
            post_action_root_quaternion_world = self._arm_root_quaternion_world(
                arm_controller
            )
            if not _quaternion_same(
                post_action_root_quaternion_world,
                root_quaternion_world,
                self.config.quaternion_readback_tolerance,
            ):
                raise V6SapienSandboxContractError(
                    "arm root frame changed during agent.set_action"
                )
            target_pose = getattr(arm_controller, "_target_pose", None)
            if target_pose is None:
                raise V6SapienSandboxContractError("arm controller exposes no target pose")
            after = _array(
                target_pose.p,
                name="arm target position",
                length=3,
            )
            after_quaternion = _quaternion(
                target_pose.q,
                name="arm target quaternion",
            )
            readback_packet, readback_evidence = self._make_readback(
                issued,
                received,
                before,
                before_quaternion,
                after,
                after_quaternion,
                root_quaternion_world,
                ik_solution,
                controller_target,
                drive_target,
            )
            self._pending_readback = readback_evidence
            self._emit_raw(
                RawIntervalEvent(
                    event="command_readback",
                    key=issued.key,
                    command_id=issued.command_id,
                    readback=readback_evidence,
                )
            )
            return result

        def wrapped_after_simulation_step(*args: Any, **kwargs: Any):
            result = original_after_simulation_step(*args, **kwargs)
            if readback_packet is None:
                raise V6SapienSandboxContractError(
                    "native physics advanced before command readback"
                )
            self._time_ns += 10_000_000
            state = self._read_raw_state(
                time_ns=self._time_ns,
                committed_progress=self.supervisor.snapshot().committed_progress,
            )
            physical = PhysicalNativeEvidence(
                key=issued.key,
                command_id=issued.command_id,
                native_sample_index=self._next_native_sample,
                substep_index=len(physical_native),
                state=state,
            )
            candidates = {
                "key": issued.key,
                "command_id": issued.command_id,
                "native_sample_index": self._next_native_sample,
                "substep_index": len(physical_native),
                "time_ns": self._time_ns,
                "audit_force_n": state.measured_force_n,
                "contact_observed": state.contact_observed,
                "normal_velocity_outward_m_s": self._normal_velocity(state),
                "instantaneous_progress": state.instantaneous_progress,
                "geometric_track_error_m": state.geometric_track_error_m,
                "recovery_hover_pose_error_m": state.recovery_hover_pose_error_m,
                "recovery_clearance_m": state.recovery_clearance_m,
                "surface_reference_point_xyz_m": (
                    state.surface_reference_point_xyz_m
                ),
                "projection_outward_normal_xyz": (
                    state.projection_outward_normal_xyz
                ),
                "command_outward_normal_xyz": state.command_outward_normal_xyz,
                "committed_recovery_outward_normal_xyz": (
                    state.committed_recovery_outward_normal_xyz
                ),
                "task_projection_point_xyz_m": state.task_projection_point_xyz_m,
                "task_projection_progress": state.task_projection_progress,
                "task_projection_arc_length_m": state.task_projection_arc_length_m,
                "projection_tangent_unit_xyz": (
                    state.projection_tangent_unit_xyz
                ),
                "command_tangent_unit_xyz": state.command_tangent_unit_xyz,
                "recovery_hover_target_position_xyz_m": (
                    state.recovery_hover_target_xyz_m
                ),
                "contact_tool_position_xyz_m": state.contact_tool_position_xyz_m,
                "contact_tool_quaternion_wxyz": (
                    state.contact_tool_quaternion_wxyz
                ),
                "contact_tool_linear_velocity_xyz_m_s": (
                    state.contact_tool_linear_velocity_xyz_m_s
                ),
                "contact_tool_angular_velocity_xyz_rad_s": (
                    state.contact_tool_angular_velocity_xyz_rad_s
                ),
                "robot_tcp_position_xyz_m": state.robot_tcp_position_xyz_m,
                "robot_tcp_quaternion_wxyz": state.robot_tcp_quaternion_wxyz,
                "robot_tcp_linear_velocity_xyz_m_s": (
                    state.robot_tcp_linear_velocity_xyz_m_s
                ),
                "robot_tcp_angular_velocity_xyz_rad_s": (
                    state.robot_tcp_angular_velocity_xyz_rad_s
                ),
            }
            native_packets.append(_construct_packet(NativeAuditSample, candidates))
            physical_native.append(physical)
            self._pending_native.append(physical)
            self._next_native_sample += 1
            self._emit_raw(
                RawIntervalEvent(
                    event="native_sample",
                    key=issued.key,
                    command_id=issued.command_id,
                    physical_native=physical,
                )
            )
            return result

        setattr(wrapped_set_action, "_v6_sandbox_binding", True)
        setattr(wrapped_after_simulation_step, "_v6_sandbox_binding", True)
        self.raw.agent.set_action = wrapped_set_action
        self.raw._after_simulation_step = wrapped_after_simulation_step
        environment_result: Any = None
        try:
            environment_result = self.env.step(action)
        finally:
            set_action_intact = self.raw.agent.set_action is wrapped_set_action
            native_hook_intact = (
                self.raw._after_simulation_step is wrapped_after_simulation_step
            )
            self.raw.agent.set_action = original_set_action
            self.raw._after_simulation_step = original_after_simulation_step
            if not set_action_intact:
                raise V6SapienSandboxContractError("agent.set_action hook was replaced")
            if not native_hook_intact:
                raise V6SapienSandboxContractError("native hook was replaced")
        if not isinstance(environment_result, tuple) or len(environment_result) != 5:
            raise V6SapienSandboxContractError(
                "env.step must return observation/reward/terminated/truncated/info"
            )
        _observation, reward, terminated, truncated, info = environment_result
        if not isinstance(info, Mapping) or "normal_force" not in info:
            raise V6SapienSandboxContractError(
                "env.step info must return normal_force"
            )
        tracking_force = _scalar(info["normal_force"], name="info normal_force")
        if tracking_force < 0.0:
            raise V6SapienSandboxContractError(
                "tracking normal force must be nonnegative"
            )
        outcome = EnvironmentStepOutcome(
            reward=_scalar(reward, name="environment reward"),
            terminated=_bool_scalar(terminated, name="environment terminated"),
            truncated=_bool_scalar(truncated, name="environment truncated"),
            tracking_force_n=tracking_force,
        )
        self._pending_environment_outcome = outcome
        self._emit_raw(
            RawIntervalEvent(
                event="environment_outcome",
                key=issued.key,
                command_id=issued.command_id,
                environment_outcome=outcome,
            )
        )
        if readback_packet is None or readback_evidence is None:
            raise V6SapienSandboxContractError("control interval lacks command readback")
        if len(native_packets) != 1 or len(physical_native) != 1:
            raise V6SapienSandboxContractError(
                "control interval did not produce exactly one native sample"
            )
        last_packet = native_packets[-1]
        last_physical = physical_native[-1]
        # ``tracking_force_n`` and the last native audit force are separate
        # timestamped streams.  Preserve both; never require equality or
        # backfill the control-rate value into native evidence.
        post_physical = PhysicalPostStepEvidence(
            key=issued.key,
            last_native_sample_index=last_packet.native_sample_index,
            last_native_time_ns=last_packet.time_ns,
            state=last_physical.state,
        )
        post_candidates = {
            "key": issued.key,
            "post_time_ns": int(last_packet.time_ns),
            "last_native_sample_index": int(last_packet.native_sample_index),
            "last_native_time_ns": int(last_packet.time_ns),
            "measured_force_n": float(last_packet.audit_force_n),
            "contact_observed": bool(last_physical.state.contact_observed),
            "normal_velocity_outward_m_s": self._normal_velocity(
                last_physical.state
            ),
            "instantaneous_progress": last_physical.state.instantaneous_progress,
            "geometric_track_error_m": (
                last_physical.state.geometric_track_error_m
            ),
            "recovery_hover_pose_error_m": (
                last_physical.state.recovery_hover_pose_error_m
            ),
            "recovery_clearance_m": last_physical.state.recovery_clearance_m,
            "surface_reference_point_xyz_m": (
                last_physical.state.surface_reference_point_xyz_m
            ),
            "projection_outward_normal_xyz": (
                last_physical.state.projection_outward_normal_xyz
            ),
            "command_outward_normal_xyz": (
                last_physical.state.command_outward_normal_xyz
            ),
            "committed_recovery_outward_normal_xyz": (
                last_physical.state.committed_recovery_outward_normal_xyz
            ),
            "task_projection_point_xyz_m": (
                last_physical.state.task_projection_point_xyz_m
            ),
            "task_projection_progress": (
                last_physical.state.task_projection_progress
            ),
            "task_projection_arc_length_m": (
                last_physical.state.task_projection_arc_length_m
            ),
            "projection_tangent_unit_xyz": (
                last_physical.state.projection_tangent_unit_xyz
            ),
            "command_tangent_unit_xyz": (
                last_physical.state.command_tangent_unit_xyz
            ),
            "recovery_hover_target_position_xyz_m": (
                last_physical.state.recovery_hover_target_xyz_m
            ),
            "contact_tool_position_xyz_m": (
                last_physical.state.contact_tool_position_xyz_m
            ),
            "contact_tool_quaternion_wxyz": (
                last_physical.state.contact_tool_quaternion_wxyz
            ),
            "contact_tool_linear_velocity_xyz_m_s": (
                last_physical.state.contact_tool_linear_velocity_xyz_m_s
            ),
            "contact_tool_angular_velocity_xyz_rad_s": (
                last_physical.state.contact_tool_angular_velocity_xyz_rad_s
            ),
            "robot_tcp_position_xyz_m": (
                last_physical.state.robot_tcp_position_xyz_m
            ),
            "robot_tcp_quaternion_wxyz": (
                last_physical.state.robot_tcp_quaternion_wxyz
            ),
            "robot_tcp_linear_velocity_xyz_m_s": (
                last_physical.state.robot_tcp_linear_velocity_xyz_m_s
            ),
            "robot_tcp_angular_velocity_xyz_rad_s": (
                last_physical.state.robot_tcp_angular_velocity_xyz_rad_s
            ),
        }
        post_packet = _construct_packet(CausalPostStepSample, post_candidates)
        self._pending_post = post_physical
        self._emit_raw(
            RawIntervalEvent(
                event="post_step",
                key=issued.key,
                command_id=issued.command_id,
                physical_post=post_physical,
            )
        )
        return BackendStepResult(
            readback=readback_packet,
            native_samples=tuple(native_packets),
            post=post_packet,
        )

    def step(self) -> AdapterStepBundle:
        if self.adapter.faulted or self._binding_faulted:
            raise V6SapienSandboxContractError("physical binding is terminally faulted")
        if self._environment_terminal:
            raise V6SapienSandboxContractError(
                "environment already returned a terminal control interval"
            )
        key = self._key()
        self._pending_readback = None
        self._pending_native = []
        self._pending_post = None
        self._pending_environment_outcome = None
        self._pending_command_id = None
        try:
            pre, physical_pre = self._build_pre_step()
            self._emit_raw(
                RawIntervalEvent(
                    event="pre_step",
                    key=key,
                    command_id=None,
                    physical_pre=physical_pre,
                )
            )
            bundle = self.adapter.step(pre, self._execute)
            if (
                self._pending_readback is None
                or self._pending_post is None
                or self._pending_environment_outcome is None
            ):
                raise V6SapienSandboxExecutionError(
                    "logical bundle closed without physical provenance"
                )
            interval = PhysicalIntervalEvidence(
                pre=physical_pre,
                readback=self._pending_readback,
                native=tuple(self._pending_native),
                post=self._pending_post,
                environment_outcome=self._pending_environment_outcome,
            )
            self._intervals.append(interval)
            self._last_post = self._pending_post
            self._last_bundle = bundle
            self._next_control_step += 1
            self._environment_terminal = bool(
                self._pending_environment_outcome.terminated
                or self._pending_environment_outcome.truncated
            )
            return bundle
        except BaseException as error:
            self._binding_faulted = True
            self._emit_fault(key, error)
            raise


__all__ = [
    "EnvironmentStepOutcome",
    "IntervalFaultSink",
    "PhysicalCommandReadbackEvidence",
    "PhysicalIntervalEvidence",
    "PhysicalIntervalFault",
    "PhysicalNativeEvidence",
    "PhysicalPostStepEvidence",
    "PhysicalPreStepEvidence",
    "RawIntervalEvent",
    "RawIntervalSink",
    "SapienSandboxBinding",
    "SapienSandboxBindingConfig",
    "SurfacePathGeometry",
    "ToolStateProvenance",
    "V6SapienSandboxBindingError",
    "V6SapienSandboxContractError",
    "V6SapienSandboxExecutionError",
]
