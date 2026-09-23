"""Transactional storage for repeatable ForceWipe V6 physics sandboxes.

This module is deliberately simulator agnostic.  It owns only the permission,
identity, tabular-evidence, and atomic-commit boundary for development runs.
SANDBOX runs are repeatable: a permission is not consumed, but every attempt
uses a new run id and an existing run is never overwritten.

The frozen physical SANDBOX emits exactly one 100-Hz native sample for each
100-Hz control decision.  A control row therefore cites the native sample
produced by its command and, from step one onward, cites the preceding command's
native sample as the source of its pre-step state.  The validator below
preserves that actual data model instead of manufacturing an N+1 observation
stream.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from typing import Any
from uuid import uuid4

import polars as pl

from .authority_metrics_s3 import (
    AUTHORITY_METRIC_SPLIT_V3,
    authority_violations_s3,
)
from .authority_metrics_s5 import (
    AUTHORITY_METRIC_PATH_FRAME_V4,
    authority_violations_s5,
)
from .authority_metrics_precision_r5 import (
    AUTHORITY_METRIC_PRECISION_R5,
    authority_violations_precision_r5,
)
from .authority_metrics_system_r4 import (
    AUTHORITY_METRIC_SYSTEM_R4,
    authority_violations_system_r4,
)


PERMISSION_FORMAT = "forcewipe_v6_sandbox_permission_v1"
RUN_DEFINITION_FORMAT_LEGACY = "forcewipe_v6_sandbox_run_definition_v1"
RUN_DEFINITION_FORMAT = "forcewipe_v6_sandbox_run_definition_v2"
RUN_DEFINITION_FORMAT_S3 = "forcewipe_v6_s3_sandbox_run_definition_v1"
RUN_DEFINITION_FORMAT_S3_R2 = "forcewipe_v6_s3_sandbox_run_definition_v2"
RUN_DEFINITION_FORMAT_S5 = "forcewipe_v6_s5_sandbox_run_definition_v1"
RUN_DEFINITION_FORMAT_SYSTEM_R3_REV2 = (
    "forcewipe_v6_system_r3_rev2_sandbox_run_definition_v1"
)
RUN_DEFINITION_FORMAT_SYSTEM_R4 = (
    "forcewipe_v6_system_r4_sandbox_run_definition_v1"
)
COMMIT_FORMAT = "forcewipe_v6_sandbox_commit_v1"
PERMISSION_SCOPE = "V6_S2_SANDBOX_ONLY"
PERMISSION_SCOPE_S3 = "V6_S3_SANDBOX_ONLY"
PERMISSION_SCOPE_S5 = "V6_S5_SANDBOX_ONLY"
PERMISSION_SCOPE_SYSTEM_R3_REV2 = "V6_SYSTEM_R3_REV2_SANDBOX_ONLY"
PERMISSION_SCOPE_SYSTEM_R4 = "V6_SYSTEM_R4_SANDBOX_ONLY"
AUTHORITY_METRIC_LEGACY = "legacy_nonstable_integral_zero_v1"
AUTHORITY_METRIC_STATE_AWARE = "state_aware_integrator_v2"

STATUS_DIAGNOSTIC = "completed_sandbox_diagnostic"
STATUS_SCIENTIFIC_FAIL = "completed_scientific_fail"
STATUS_INFRASTRUCTURE_ABORT = "aborted_infrastructure"
TERMINAL_STATUSES = frozenset(
    {STATUS_DIAGNOSTIC, STATUS_SCIENTIFIC_FAIL, STATUS_INFRASTRUCTURE_ABORT}
)

_RUN_ID = re.compile(r"^sbx_[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_INFRASTRUCTURE_KEYS = frozenset({"adapter", "binding", "runner", "storage"})
SANDBOX_OUTPUT_ROOT = Path("/tmp/forcewipe_v6_sandbox")
SUPERVISOR_STATES = frozenset(
    {
        "APPROACH",
        "CONTACT_VERIFY",
        "TRACK",
        "REBOUND_GUARD",
        "HEADROOM",
        "RECOVERY_LIFT",
        "RECOVERY_HOVER",
        "RECOVERY_ACQUIRE",
        "SAFE_HOLD",
    }
)
_DOWNSTREAM_PERMISSION_KEYS = (
    "qualification_dev_execution_permitted",
    "cal_execution_permitted",
    "train_execution_permitted",
    "test_execution_permitted",
)


class SandboxStorageError(RuntimeError):
    """Base error for the SANDBOX evidence boundary."""


class SandboxPermissionError(SandboxStorageError):
    """Execution was not authorized for this exact storage boundary."""


class SandboxIntegrityError(SandboxStorageError):
    """Evidence or identity violates the registered schema/closure contract."""


class SandboxInfrastructureAbort(SandboxStorageError):
    """A run was transactionally closed as an infrastructure abort."""

    def __init__(self, message: str, *, final_path: Path | None = None) -> None:
        super().__init__(message)
        self.final_path = final_path


class SandboxUncommittedError(SandboxStorageError):
    """A directory has no valid last-write COMMIT marker."""


NATIVE_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "scenario_id": pl.Int64,
        "evaluation_id": pl.Int64,
        "segment_index": pl.Int32,
        "control_step_index": pl.Int64,
        "command_id": pl.String,
        "native_sample_index": pl.Int64,
        "substep_index": pl.Int32,
        "time_ns": pl.Int64,
        "audit_force_n": pl.Float64,
        "contact_observed": pl.Boolean,
        "normal_velocity_outward_m_s": pl.Float64,
        "instantaneous_progress": pl.Float64,
        "geometric_track_error_m": pl.Float64,
        "recovery_hover_pose_error_m": pl.Float64,
        "recovery_clearance_m": pl.Float64,
        "signed_recovery_clearance_m": pl.Float64,
        "geometry_frame_id": pl.String,
        "contact_tool_frame_id": pl.String,
        "robot_tcp_frame_id": pl.String,
        "task_path_id": pl.String,
        "surface_id": pl.String,
        "surface_reference_x_m": pl.Float64,
        "surface_reference_y_m": pl.Float64,
        "surface_reference_z_m": pl.Float64,
        "projection_outward_normal_x": pl.Float64,
        "projection_outward_normal_y": pl.Float64,
        "projection_outward_normal_z": pl.Float64,
        "command_outward_normal_x": pl.Float64,
        "command_outward_normal_y": pl.Float64,
        "command_outward_normal_z": pl.Float64,
        "committed_recovery_outward_normal_x": pl.Float64,
        "committed_recovery_outward_normal_y": pl.Float64,
        "committed_recovery_outward_normal_z": pl.Float64,
        "task_projection_point_x_m": pl.Float64,
        "task_projection_point_y_m": pl.Float64,
        "task_projection_point_z_m": pl.Float64,
        "task_projection_progress": pl.Float64,
        "task_projection_arc_length_m": pl.Float64,
        "projection_tangent_x": pl.Float64,
        "projection_tangent_y": pl.Float64,
        "projection_tangent_z": pl.Float64,
        "command_tangent_x": pl.Float64,
        "command_tangent_y": pl.Float64,
        "command_tangent_z": pl.Float64,
        "task_path_length_m": pl.Float64,
        "recovery_hover_target_x_m": pl.Float64,
        "recovery_hover_target_y_m": pl.Float64,
        "recovery_hover_target_z_m": pl.Float64,
        "contact_tool_x_m": pl.Float64,
        "contact_tool_y_m": pl.Float64,
        "contact_tool_z_m": pl.Float64,
        "contact_tool_qw": pl.Float64,
        "contact_tool_qx": pl.Float64,
        "contact_tool_qy": pl.Float64,
        "contact_tool_qz": pl.Float64,
        "contact_tool_vx_m_s": pl.Float64,
        "contact_tool_vy_m_s": pl.Float64,
        "contact_tool_vz_m_s": pl.Float64,
        "contact_tool_wx_rad_s": pl.Float64,
        "contact_tool_wy_rad_s": pl.Float64,
        "contact_tool_wz_rad_s": pl.Float64,
        "robot_tcp_x_m": pl.Float64,
        "robot_tcp_y_m": pl.Float64,
        "robot_tcp_z_m": pl.Float64,
        "robot_tcp_qw": pl.Float64,
        "robot_tcp_qx": pl.Float64,
        "robot_tcp_qy": pl.Float64,
        "robot_tcp_qz": pl.Float64,
        "robot_tcp_vx_m_s": pl.Float64,
        "robot_tcp_vy_m_s": pl.Float64,
        "robot_tcp_vz_m_s": pl.Float64,
        "robot_tcp_wx_rad_s": pl.Float64,
        "robot_tcp_wy_rad_s": pl.Float64,
        "robot_tcp_wz_rad_s": pl.Float64,
    }
)


CONTROL_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "scenario_id": pl.Int64,
        "evaluation_id": pl.Int64,
        "segment_index": pl.Int32,
        "control_step_index": pl.Int64,
        "command_id": pl.String,
        "pre_time_ns": pl.Int64,
        "state_source_kind": pl.String,
        "force_source_native_sample_index": pl.Int64,
        "force_source_time_ns": pl.Int64,
        "bootstrap_id": pl.String,
        "pre_force_n": pl.Float64,
        "target_force_n": pl.Float64,
        "pre_contact_observed": pl.Boolean,
        "normal_velocity_outward_m_s": pl.Float64,
        "instantaneous_progress": pl.Float64,
        "geometric_track_error_m": pl.Float64,
        "requested_tangential_step_m": pl.Float64,
        "recovery_hover_pose_error_m": pl.Float64,
        "recovery_clearance_m": pl.Float64,
        "geometry_frame_id": pl.String,
        "command_frame_id": pl.String,
        "contact_tool_frame_id": pl.String,
        "robot_tcp_frame_id": pl.String,
        "task_path_id": pl.String,
        "surface_id": pl.String,
        "surface_reference_x_m": pl.Float64,
        "surface_reference_y_m": pl.Float64,
        "surface_reference_z_m": pl.Float64,
        "task_projection_point_x_m": pl.Float64,
        "task_projection_point_y_m": pl.Float64,
        "task_projection_point_z_m": pl.Float64,
        "task_projection_progress": pl.Float64,
        "task_projection_arc_length_m": pl.Float64,
        "task_path_length_m": pl.Float64,
        "pre_contact_tool_x_m": pl.Float64,
        "pre_contact_tool_y_m": pl.Float64,
        "pre_contact_tool_z_m": pl.Float64,
        "pre_contact_tool_qw": pl.Float64,
        "pre_contact_tool_qx": pl.Float64,
        "pre_contact_tool_qy": pl.Float64,
        "pre_contact_tool_qz": pl.Float64,
        "pre_contact_tool_vx_m_s": pl.Float64,
        "pre_contact_tool_vy_m_s": pl.Float64,
        "pre_contact_tool_vz_m_s": pl.Float64,
        "pre_contact_tool_wx_rad_s": pl.Float64,
        "pre_contact_tool_wy_rad_s": pl.Float64,
        "pre_contact_tool_wz_rad_s": pl.Float64,
        "pre_robot_tcp_x_m": pl.Float64,
        "pre_robot_tcp_y_m": pl.Float64,
        "pre_robot_tcp_z_m": pl.Float64,
        "pre_robot_tcp_qw": pl.Float64,
        "pre_robot_tcp_qx": pl.Float64,
        "pre_robot_tcp_qy": pl.Float64,
        "pre_robot_tcp_qz": pl.Float64,
        "pre_robot_tcp_vx_m_s": pl.Float64,
        "pre_robot_tcp_vy_m_s": pl.Float64,
        "pre_robot_tcp_vz_m_s": pl.Float64,
        "pre_robot_tcp_wx_rad_s": pl.Float64,
        "pre_robot_tcp_wy_rad_s": pl.Float64,
        "pre_robot_tcp_wz_rad_s": pl.Float64,
        "signed_recovery_clearance_m": pl.Float64,
        "projection_outward_normal_x": pl.Float64,
        "projection_outward_normal_y": pl.Float64,
        "projection_outward_normal_z": pl.Float64,
        "command_outward_normal_x": pl.Float64,
        "command_outward_normal_y": pl.Float64,
        "command_outward_normal_z": pl.Float64,
        "committed_recovery_outward_normal_x": pl.Float64,
        "committed_recovery_outward_normal_y": pl.Float64,
        "committed_recovery_outward_normal_z": pl.Float64,
        "projection_tangent_x": pl.Float64,
        "projection_tangent_y": pl.Float64,
        "projection_tangent_z": pl.Float64,
        "command_tangent_x": pl.Float64,
        "command_tangent_y": pl.Float64,
        "command_tangent_z": pl.Float64,
        "recovery_hover_target_x_m": pl.Float64,
        "recovery_hover_target_y_m": pl.Float64,
        "recovery_hover_target_z_m": pl.Float64,
        "requested_recovery_dx_m": pl.Float64,
        "requested_recovery_dy_m": pl.Float64,
        "requested_recovery_dz_m": pl.Float64,
        "requested_cross_track_dx_m": pl.Float64,
        "requested_cross_track_dy_m": pl.Float64,
        "requested_cross_track_dz_m": pl.Float64,
        "requested_recovery_rx_rad": pl.Float64,
        "requested_recovery_ry_rad": pl.Float64,
        "requested_recovery_rz_rad": pl.Float64,
        "issued_time_ns": pl.Int64,
        "issued_command_frame_id": pl.String,
        "issued_command_mode": pl.String,
        "issued_normal_step_m": pl.Float64,
        "issued_tangential_step_m": pl.Float64,
        "issued_recovery_reposition_permitted": pl.Boolean,
        "issued_cross_track_correction_permitted": pl.Boolean,
        "issued_normal_dx_m": pl.Float64,
        "issued_normal_dy_m": pl.Float64,
        "issued_normal_dz_m": pl.Float64,
        "issued_task_dx_m": pl.Float64,
        "issued_task_dy_m": pl.Float64,
        "issued_task_dz_m": pl.Float64,
        "issued_recovery_dx_m": pl.Float64,
        "issued_recovery_dy_m": pl.Float64,
        "issued_recovery_dz_m": pl.Float64,
        "issued_cross_track_dx_m": pl.Float64,
        "issued_cross_track_dy_m": pl.Float64,
        "issued_cross_track_dz_m": pl.Float64,
        "issued_dx_m": pl.Float64,
        "issued_dy_m": pl.Float64,
        "issued_dz_m": pl.Float64,
        "issued_recovery_rx_rad": pl.Float64,
        "issued_recovery_ry_rad": pl.Float64,
        "issued_recovery_rz_rad": pl.Float64,
        "issued_rotation_rx_rad": pl.Float64,
        "issued_rotation_ry_rad": pl.Float64,
        "issued_rotation_rz_rad": pl.Float64,
        "readback_time_ns": pl.Int64,
        "readback_accepted": pl.Boolean,
        "readback_stage": pl.String,
        "applied_command_frame_id": pl.String,
        "applied_command_mode": pl.String,
        "applied_normal_step_m": pl.Float64,
        "applied_tangential_step_m": pl.Float64,
        "applied_recovery_reposition_permitted": pl.Boolean,
        "applied_cross_track_correction_permitted": pl.Boolean,
        "applied_normal_dx_m": pl.Float64,
        "applied_normal_dy_m": pl.Float64,
        "applied_normal_dz_m": pl.Float64,
        "applied_task_dx_m": pl.Float64,
        "applied_task_dy_m": pl.Float64,
        "applied_task_dz_m": pl.Float64,
        "applied_recovery_dx_m": pl.Float64,
        "applied_recovery_dy_m": pl.Float64,
        "applied_recovery_dz_m": pl.Float64,
        "applied_cross_track_dx_m": pl.Float64,
        "applied_cross_track_dy_m": pl.Float64,
        "applied_cross_track_dz_m": pl.Float64,
        "applied_dx_m": pl.Float64,
        "applied_dy_m": pl.Float64,
        "applied_dz_m": pl.Float64,
        "applied_recovery_rx_rad": pl.Float64,
        "applied_recovery_ry_rad": pl.Float64,
        "applied_recovery_rz_rad": pl.Float64,
        "applied_rotation_rx_rad": pl.Float64,
        "applied_rotation_ry_rad": pl.Float64,
        "applied_rotation_rz_rad": pl.Float64,
        "post_time_ns": pl.Int64,
        "last_native_sample_index": pl.Int64,
        "last_native_time_ns": pl.Int64,
        "post_force_n": pl.Float64,
        "post_contact_observed": pl.Boolean,
        "post_normal_velocity_outward_m_s": pl.Float64,
        "post_instantaneous_progress": pl.Float64,
        "post_geometric_track_error_m": pl.Float64,
        "post_recovery_hover_pose_error_m": pl.Float64,
        "post_recovery_clearance_m": pl.Float64,
        "post_contact_tool_x_m": pl.Float64,
        "post_contact_tool_y_m": pl.Float64,
        "post_contact_tool_z_m": pl.Float64,
        "post_contact_tool_qw": pl.Float64,
        "post_contact_tool_qx": pl.Float64,
        "post_contact_tool_qy": pl.Float64,
        "post_contact_tool_qz": pl.Float64,
        "post_contact_tool_vx_m_s": pl.Float64,
        "post_contact_tool_vy_m_s": pl.Float64,
        "post_contact_tool_vz_m_s": pl.Float64,
        "post_contact_tool_wx_rad_s": pl.Float64,
        "post_contact_tool_wy_rad_s": pl.Float64,
        "post_contact_tool_wz_rad_s": pl.Float64,
        "post_robot_tcp_x_m": pl.Float64,
        "post_robot_tcp_y_m": pl.Float64,
        "post_robot_tcp_z_m": pl.Float64,
        "post_robot_tcp_qw": pl.Float64,
        "post_robot_tcp_qx": pl.Float64,
        "post_robot_tcp_qy": pl.Float64,
        "post_robot_tcp_qz": pl.Float64,
        "post_robot_tcp_vx_m_s": pl.Float64,
        "post_robot_tcp_vy_m_s": pl.Float64,
        "post_robot_tcp_vz_m_s": pl.Float64,
        "post_robot_tcp_wx_rad_s": pl.Float64,
        "post_robot_tcp_wy_rad_s": pl.Float64,
        "post_robot_tcp_wz_rad_s": pl.Float64,
    }
)


DIAGNOSTIC_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "scenario_id": pl.Int64,
        "evaluation_id": pl.Int64,
        "segment_index": pl.Int32,
        "control_step_index": pl.Int64,
        "state_before": pl.String,
        "state_after": pl.String,
        "transition_reason": pl.String,
        "transitioned": pl.Boolean,
        "state_dwell_samples": pl.Int64,
        "measured_force_n": pl.Float64,
        "target_force_n": pl.Float64,
        "contact_observed": pl.Boolean,
        "previous_force_n": pl.Float64,
        "previous_previous_force_n": pl.Float64,
        "force_rate_n_s": pl.Float64,
        "normal_velocity_outward_m_s": pl.Float64,
        "previous_normal_command_m": pl.Float64,
        "previous_previous_normal_command_m": pl.Float64,
        "geometric_track_error_m": pl.Float64,
        "recovery_hover_pose_error_m": pl.Float64,
        "recovery_clearance_m": pl.Float64,
        "force_track_ready": pl.Boolean,
        "geometric_track_ready": pl.Boolean,
        "stable_track": pl.Boolean,
        "p_step_m": pl.Float64,
        "i_step_m": pl.Float64,
        "d_step_m": pl.Float64,
        "nominal_normal_step_m": pl.Float64,
        "actuator_limited_normal_step_m": pl.Float64,
        "projected_normal_step_m": pl.Float64,
        "executed_normal_step_m": pl.Float64,
        "envelope_base_upper_n": pl.Float64,
        "envelope_command_upper_n": pl.Float64,
        "safe_press_limit_m": pl.Float64,
        "projection_active": pl.Boolean,
        "actuator_saturation_active": pl.Boolean,
        "safety_projection_active": pl.Boolean,
        "integral_before_n_s": pl.Float64,
        "integral_after_n_s": pl.Float64,
        "rebound_drop_n": pl.Float64,
        "rebound_guard_triggered": pl.Boolean,
        "headroom_triggered": pl.Boolean,
        "tangential_motion_permitted": pl.Boolean,
        "recovery_reposition_permitted": pl.Boolean,
        "cross_track_correction_permitted": pl.Boolean,
        "requested_tangential_step_m": pl.Float64,
        "executed_tangential_step_m": pl.Float64,
        "instantaneous_progress": pl.Float64,
        "committed_progress_before": pl.Float64,
        "committed_progress_after": pl.Float64,
        "in_recovery_episode_before": pl.Boolean,
        "in_recovery_episode_after": pl.Boolean,
        "recovery_sample_counted_this_step": pl.Boolean,
        "recovery_sample_budget_exhausted": pl.Boolean,
        "pre_recovery_highwater_progress_before": pl.Float64,
        "pre_recovery_highwater_progress_after": pl.Float64,
        "verified_return_count_before": pl.Int64,
        "verified_return_count": pl.Int64,
        "verified_return_required_progress": pl.Float64,
        "verified_return_progress_gate_passed": pl.Boolean,
        "verified_return_completed": pl.Boolean,
        "episode_total_recovery_cycles": pl.Int64,
        "episode_total_recovery_samples": pl.Int64,
        "sampled_force_violation_observed": pl.Boolean,
    }
)


EPISODE_SCHEMA = pl.Schema(
    {
        "run_id": pl.String,
        "scenario_id": pl.Int64,
        "evaluation_id": pl.Int64,
        "target_force_n": pl.Float64,
        "terminal_status": pl.String,
        "scientific_failure": pl.Boolean,
        "scientific_failure_reason": pl.String,
        "lifecycle_complete": pl.Boolean,
        "control_steps": pl.Int64,
        "native_samples": pl.Int64,
        "peak_force_n": pl.Float64,
        "force_limit_violation_count": pl.Int64,
        "authority_violation_count": pl.Int64,
        "command_readback_violation_count": pl.Int64,
        "tracking_rmse_n": pl.Float64,
        "band_fraction": pl.Float64,
        "contact_fraction": pl.Float64,
        "final_committed_progress": pl.Float64,
        "final_supervisor_state": pl.String,
        "wall_time_s": pl.Float64,
    }
)


SCHEMAS: dict[str, pl.Schema] = {
    "native": NATIVE_SCHEMA,
    "control": CONTROL_SCHEMA,
    "diagnostic": DIAGNOSTIC_SCHEMA,
    "episode": EPISODE_SCHEMA,
}

NULLABLE_FIELDS: dict[str, frozenset[str]] = {
    "native": frozenset(),
    "control": frozenset({"force_source_native_sample_index"}),
    "diagnostic": frozenset(
        {
            "previous_force_n",
            "previous_previous_force_n",
            "pre_recovery_highwater_progress_before",
            "pre_recovery_highwater_progress_after",
            "verified_return_required_progress",
        }
    ),
    "episode": frozenset({"tracking_rmse_n", "band_fraction"}),
}

SOURCE_MANIFEST_FIELDS = ("relative_path", "size_bytes", "sha256", "role")
SCENARIO_MANIFEST_FIELDS = (
    "scenario_id",
    "evaluation_id",
    "family",
    "target_force_n",
    "seed",
    "max_control_steps",
    "expected_trigger",
    "parameters_json",
)
FILE_MANIFEST_FIELDS = (
    "relative_path",
    "role",
    "size_bytes",
    "sha256",
    "row_count",
    "schema_sha256",
)

ParquetWriter = Callable[[str, pl.DataFrame, Path], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise SandboxIntegrityError("metadata is not finite canonical JSON") from exc
    return (text + "\n").encode("utf-8")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _schema_payload(name: str, schema: pl.Schema) -> list[dict[str, Any]]:
    nullable = NULLABLE_FIELDS[name]
    return [
        {"name": field, "dtype": str(dtype), "nullable": field in nullable}
        for field, dtype in schema.items()
    ]


def schema_sha256(name: str | None = None) -> str:
    """Return a stable schema fingerprint for one table or the whole registry."""

    if name is not None:
        if name not in SCHEMAS:
            raise SandboxIntegrityError(f"unknown schema {name!r}")
        payload: Any = _schema_payload(name, SCHEMAS[name])
    else:
        payload = {
            key: _schema_payload(key, value) for key, value in SCHEMAS.items()
        }
    return _sha256_bytes(_canonical_json_bytes(payload))


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


def _atomic_write_json(path: Path, value: Any) -> None:
    _atomic_write_bytes(path, _canonical_json_bytes(value))


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        if set(row) != set(fields):
            raise SandboxIntegrityError("CSV row fields do not match its contract")
        writer.writerow({name: row[name] for name in fields})
    return stream.getvalue().encode("utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _reject_symlink_components(path: Path, *, label: str) -> None:
    current = path.absolute()
    while True:
        if current.exists() and current.is_symlink():
            raise SandboxPermissionError(f"{label} contains a symbolic link")
        if current.parent == current:
            break
        current = current.parent


def _safe_relative_path(value: Any, *, label: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SandboxIntegrityError(f"{label} must be a nonempty POSIX relative path")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(part in ("", ".", "..") for part in relative.parts):
        raise SandboxIntegrityError(f"{label} escapes its registered root")
    return relative


def _checked_sha(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SandboxIntegrityError(f"{label} must be a lowercase SHA-256")
    return value


def _checked_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise SandboxIntegrityError(f"{label} must be an integer >= {minimum}")
    return value


def _checked_real(value: Any, *, label: str, minimum: float | None = None) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SandboxIntegrityError(f"{label} must be a real number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise SandboxIntegrityError(f"{label} is outside its finite domain")
    return result


def _validate_permission(permission: Mapping[str, Any]) -> Path:
    if permission.get("format") != PERMISSION_FORMAT:
        raise SandboxPermissionError("unsupported SANDBOX permission format")
    if permission.get("scope") not in {
        PERMISSION_SCOPE,
        PERMISSION_SCOPE_S3,
        PERMISSION_SCOPE_S5,
        PERMISSION_SCOPE_SYSTEM_R3_REV2,
        PERMISSION_SCOPE_SYSTEM_R4,
    }:
        raise SandboxPermissionError("permission scope is not SANDBOX-only")
    if permission.get("sandbox_execution_permitted") is not True:
        raise SandboxPermissionError("SANDBOX execution permission is closed")
    for key in _DOWNSTREAM_PERMISSION_KEYS:
        if permission.get(key) is not False:
            raise SandboxPermissionError(f"downstream permission {key} must remain false")
    if permission.get("scope") == PERMISSION_SCOPE_SYSTEM_R3_REV2:
        from .authorization_system_r3 import (
            validate_approval_identity,
            validate_review_binding,
        )

        try:
            validate_approval_identity(
                permission.get("reviewer"),
                permission.get("approved_utc"),
                permission.get("approval_id"),
            )
            validate_review_binding(permission)
        except ValueError as exc:
            raise SandboxPermissionError(str(exc)) from exc
        expected = {
            "controller_revision": (
                "V6-system-contact-candidate-v3-rev2-local-cycle-accounting"
            ),
            "evaluation_id_namespace": (
                "forcewipe-v6-system-r3-rev2-sandbox-v1"
            ),
            "run_prefix": "sbx_v6systemr3r2_",
            "run_definition_format": RUN_DEFINITION_FORMAT_SYSTEM_R3_REV2,
        }
        for name, value in expected.items():
            if permission.get(name) != value:
                raise SandboxPermissionError(
                    f"System R3 rev2 permission does not bind {name}"
                )
    if permission.get("scope") == PERMISSION_SCOPE_SYSTEM_R4:
        from .authorization_system_r4 import (
            validate_approval_identity,
            validate_review_binding,
        )

        try:
            validate_approval_identity(
                permission.get("reviewer"),
                permission.get("approved_utc"),
                permission.get("approval_id"),
            )
            validate_review_binding(permission)
        except ValueError as exc:
            raise SandboxPermissionError(str(exc)) from exc
        expected = {
            "controller_revision": (
                "V6-system-contact-candidate-v4-bounded-nested-recovery"
            ),
            "evaluation_id_namespace": "forcewipe-v6-system-r4-sandbox-v1",
            "run_prefix": "sbx_v6systemr4_",
            "run_definition_format": RUN_DEFINITION_FORMAT_SYSTEM_R4,
        }
        for name, value in expected.items():
            if permission.get(name) != value:
                raise SandboxPermissionError(
                    f"System R4 permission does not bind {name}"
                )
    raw_root = permission.get("allowed_output_root")
    if not isinstance(raw_root, str) or not raw_root:
        raise SandboxPermissionError("allowed_output_root is missing")
    root = Path(raw_root)
    if not root.is_absolute():
        raise SandboxPermissionError("allowed_output_root must be absolute")
    _reject_symlink_components(root, label="allowed_output_root")
    if root.resolve(strict=False) != SANDBOX_OUTPUT_ROOT.resolve(strict=False):
        raise SandboxPermissionError(
            "allowed_output_root must equal /tmp/forcewipe_v6_sandbox"
        )
    _checked_int(
        permission.get("max_evaluations_per_run"),
        label="max_evaluations_per_run",
        minimum=1,
    )
    _checked_int(
        permission.get("max_control_steps_per_evaluation"),
        label="max_control_steps_per_evaluation",
        minimum=1,
    )
    infrastructure = permission.get("trusted_infrastructure_sha256")
    if not isinstance(infrastructure, Mapping) or set(infrastructure) != _INFRASTRUCTURE_KEYS:
        raise SandboxPermissionError("trusted infrastructure identity is incomplete")
    for name, digest in infrastructure.items():
        _checked_sha(digest, label=f"trusted infrastructure {name}")
    expected_schema = _checked_sha(
        permission.get("schema_sha256"), label="permission schema_sha256"
    )
    if expected_schema != schema_sha256():
        raise SandboxPermissionError("permission does not bind the active schemas")
    return root


def _validate_run_definition(
    definition: Mapping[str, Any], *, run_id: str, permission: Mapping[str, Any]
) -> None:
    definition_format = definition.get("format")
    if definition_format not in {
        RUN_DEFINITION_FORMAT_LEGACY,
        RUN_DEFINITION_FORMAT,
        RUN_DEFINITION_FORMAT_S3,
        RUN_DEFINITION_FORMAT_S3_R2,
        RUN_DEFINITION_FORMAT_S5,
        RUN_DEFINITION_FORMAT_SYSTEM_R3_REV2,
        RUN_DEFINITION_FORMAT_SYSTEM_R4,
    }:
        raise SandboxIntegrityError("unsupported run-definition format")
    if definition.get("run_id") != run_id:
        raise SandboxIntegrityError("run definition and requested run id differ")
    authority_metric = definition.get(
        "authority_metric_version", AUTHORITY_METRIC_LEGACY
    )
    if authority_metric not in {
        AUTHORITY_METRIC_LEGACY,
        AUTHORITY_METRIC_STATE_AWARE,
        AUTHORITY_METRIC_SPLIT_V3,
        AUTHORITY_METRIC_PATH_FRAME_V4,
        AUTHORITY_METRIC_PRECISION_R5,
        AUTHORITY_METRIC_SYSTEM_R4,
    }:
        raise SandboxIntegrityError("unsupported authority metric version")
    if (
        definition_format == RUN_DEFINITION_FORMAT_LEGACY
        and authority_metric != AUTHORITY_METRIC_LEGACY
    ):
        raise SandboxIntegrityError("legacy run definition must use legacy authority metric")
    if (
        definition_format == RUN_DEFINITION_FORMAT
        and authority_metric != AUTHORITY_METRIC_STATE_AWARE
    ):
        raise SandboxIntegrityError(
            "current run definition must use state-aware authority metric"
        )
    if (
        definition_format in {RUN_DEFINITION_FORMAT_S3, RUN_DEFINITION_FORMAT_S3_R2}
        and authority_metric != AUTHORITY_METRIC_SPLIT_V3
    ):
        raise SandboxIntegrityError(
            "S3 run definition must use split-authority metric v3"
        )
    if (
        definition_format
        in {RUN_DEFINITION_FORMAT_S5, RUN_DEFINITION_FORMAT_SYSTEM_R3_REV2}
        and authority_metric
        not in {AUTHORITY_METRIC_PATH_FRAME_V4, AUTHORITY_METRIC_PRECISION_R5}
    ):
        raise SandboxIntegrityError(
            "S5 run definition must use path-frame authority metric v4"
        )
    if (
        definition_format == RUN_DEFINITION_FORMAT_SYSTEM_R4
        and authority_metric != AUTHORITY_METRIC_SYSTEM_R4
    ):
        raise SandboxIntegrityError(
            "System R4 run definition must use nested-recovery authority metric v5"
        )
    permission_scope = permission.get("scope")
    if definition_format in {RUN_DEFINITION_FORMAT_S3, RUN_DEFINITION_FORMAT_S3_R2}:
        if permission_scope != PERMISSION_SCOPE_S3:
            raise SandboxPermissionError("S3 run definition requires S3 permission scope")
    elif definition_format == RUN_DEFINITION_FORMAT_S5:
        if permission_scope != PERMISSION_SCOPE_S5:
            raise SandboxPermissionError("S5 run definition requires S5 permission scope")
    elif definition_format == RUN_DEFINITION_FORMAT_SYSTEM_R3_REV2:
        if permission_scope != PERMISSION_SCOPE_SYSTEM_R3_REV2:
            raise SandboxPermissionError(
                "System R3 rev2 run definition requires System R3 rev2 permission scope"
            )
        expected = {
            "controller_revision": permission.get("controller_revision"),
            "evaluation_id_namespace": permission.get("evaluation_id_namespace"),
            "run_prefix": permission.get("run_prefix"),
        }
        for name, value in expected.items():
            if definition.get(name) != value:
                raise SandboxIntegrityError(
                    f"System R3 rev2 run definition does not bind {name}"
                )
    elif definition_format == RUN_DEFINITION_FORMAT_SYSTEM_R4:
        if permission_scope != PERMISSION_SCOPE_SYSTEM_R4:
            raise SandboxPermissionError(
                "System R4 run definition requires System R4 permission scope"
            )
        expected = {
            "controller_revision": permission.get("controller_revision"),
            "evaluation_id_namespace": permission.get("evaluation_id_namespace"),
            "run_prefix": permission.get("run_prefix"),
        }
        for name, value in expected.items():
            if definition.get(name) != value:
                raise SandboxIntegrityError(
                    f"System R4 run definition does not bind {name}"
                )
    elif permission_scope != PERMISSION_SCOPE:
        raise SandboxPermissionError("S2 run definition requires S2 permission scope")
    _checked_sha(definition.get("controller_sha256"), label="controller_sha256")
    if definition_format in {
        RUN_DEFINITION_FORMAT_S3_R2,
        RUN_DEFINITION_FORMAT_S5,
        RUN_DEFINITION_FORMAT_SYSTEM_R3_REV2,
        RUN_DEFINITION_FORMAT_SYSTEM_R4,
    }:
        external_runtime = definition.get("external_python_runtime")
        if not isinstance(external_runtime, Mapping):
            raise SandboxIntegrityError(
                "run lacks its required external runtime identity"
            )
        external_digest = _checked_sha(
            definition.get("external_runtime_sha256"),
            label="external_runtime_sha256",
        )
        if hashlib.sha256(_canonical_json_bytes(external_runtime)).hexdigest() != external_digest:
            raise SandboxIntegrityError("S3-r2 external runtime digest is inconsistent")
    adapter_sha = _checked_sha(definition.get("adapter_sha256"), label="adapter_sha256")
    if definition.get("schema_sha256") != schema_sha256():
        raise SandboxIntegrityError("run definition does not bind the active schemas")
    force_limit_n = _checked_real(
        definition.get("force_limit_n"), label="force_limit_n", minimum=0.0
    )
    if force_limit_n != 15.0:
        raise SandboxIntegrityError("V6 SANDBOX force limit must remain 15 N")
    _checked_real(
        definition.get("command_readback_tolerance_m"),
        label="command_readback_tolerance_m",
        minimum=0.0,
    )
    _checked_real(
        definition.get("pose_alignment_tolerance_m"),
        label="pose_alignment_tolerance_m",
        minimum=0.0,
    )
    for name in (
        "basis_tolerance",
        "quaternion_tolerance",
        "rotation_readback_tolerance_rad",
        "maximum_recovery_reposition_step_m",
        "maximum_recovery_rotation_step_rad",
        "maximum_task_tangential_step_m",
        "recovery_hover_clearance_m",
    ):
        _checked_real(definition.get(name), label=name, minimum=0.0)
    if float(definition["maximum_recovery_reposition_step_m"]) != 0.006:
        raise SandboxIntegrityError("V6 SANDBOX recovery translation bound must be 6 mm")
    if float(definition["maximum_recovery_rotation_step_rad"]) != 0.0:
        raise SandboxIntegrityError("V6 SANDBOX recovery rotation must remain frozen")
    if float(definition["maximum_task_tangential_step_m"]) != 0.0035:
        raise SandboxIntegrityError("V6 SANDBOX task step bound must remain 3.5 mm")
    if float(definition["recovery_hover_clearance_m"]) <= 0.0:
        raise SandboxIntegrityError("recovery hover clearance must be positive")
    if "cross_track_correction_enter_m" in definition:
        if float(definition["cross_track_correction_enter_m"]) != 0.003:
            raise SandboxIntegrityError(
                "cross-track correction entry threshold must remain 3 mm"
            )
        if float(definition["maximum_cross_track_correction_step_m"]) != 0.0005:
            raise SandboxIntegrityError(
                "cross-track correction step bound must remain 0.5 mm"
            )
    for name in (
        "expected_command_frame_id",
        "expected_command_mode",
        "required_readback_stage",
    ):
        if not isinstance(definition.get(name), str) or not definition[name]:
            raise SandboxIntegrityError(f"{name} must be nonempty text")
    native_steps = _checked_int(
        definition.get("expected_native_steps_per_control"),
        label="expected_native_steps_per_control",
        minimum=1,
    )
    native_period_ns = _checked_int(
        definition.get("native_period_ns"), label="native_period_ns", minimum=1
    )
    control_period_ns = _checked_int(
        definition.get("control_period_ns"), label="control_period_ns", minimum=1
    )
    if control_period_ns != native_steps * native_period_ns:
        raise SandboxIntegrityError(
            "control period must equal native count times native period"
        )
    if (
        native_steps != 1
        or native_period_ns != 10_000_000
        or control_period_ns != 10_000_000
    ):
        raise SandboxIntegrityError(
            "V6 physical SANDBOX requires one 100-Hz native sample per control step"
        )
    infrastructure = definition.get("trusted_infrastructure_sha256")
    if infrastructure != permission.get("trusted_infrastructure_sha256"):
        raise SandboxPermissionError("run and permission infrastructure identities differ")
    if adapter_sha != infrastructure["adapter"]:
        raise SandboxPermissionError("run adapter identity is not the trusted adapter")
    parent = definition.get("parent_run_id")
    if parent is not None and (not isinstance(parent, str) or _RUN_ID.fullmatch(parent) is None):
        raise SandboxIntegrityError("parent_run_id is invalid")
    _canonical_json_bytes(definition)


def _validate_source_manifest(
    rows: Sequence[Mapping[str, Any]], *, source_root: Path
) -> list[dict[str, Any]]:
    if not rows:
        raise SandboxIntegrityError("source manifest must not be empty")
    _reject_symlink_components(source_root, label="source_root")
    source_root = source_root.resolve(strict=True)
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in rows:
        if set(row) != set(SOURCE_MANIFEST_FIELDS):
            raise SandboxIntegrityError("source manifest fields are invalid")
        relative = _safe_relative_path(row["relative_path"], label="source relative_path")
        relative_text = relative.as_posix()
        if relative_text in seen:
            raise SandboxIntegrityError("source manifest contains duplicate paths")
        seen.add(relative_text)
        path = source_root.joinpath(*relative.parts)
        _reject_symlink_components(path, label="source manifest path")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(source_root):
            raise SandboxIntegrityError("source manifest path escapes source_root")
        if not resolved.is_file():
            raise SandboxIntegrityError("source manifest entry is not a file")
        size = _checked_int(row["size_bytes"], label="source size_bytes")
        digest = _checked_sha(row["sha256"], label="source sha256")
        role = row["role"]
        if not isinstance(role, str) or not role:
            raise SandboxIntegrityError("source role must be nonempty")
        if resolved.stat().st_size != size or _sha256_file(resolved) != digest:
            raise SandboxIntegrityError("source manifest identity mismatch")
        normalized.append(
            {
                "relative_path": relative_text,
                "size_bytes": size,
                "sha256": digest,
                "role": role,
            }
        )
    return normalized


def _validate_scenario_manifest(
    rows: Sequence[Mapping[str, Any]], *, permission: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]]]:
    maximum_evaluations = int(permission["max_evaluations_per_run"])
    maximum_steps = int(permission["max_control_steps_per_evaluation"])
    if not rows or len(rows) > maximum_evaluations:
        raise SandboxIntegrityError("scenario manifest violates the run-size bound")
    normalized: list[dict[str, Any]] = []
    by_evaluation: dict[int, dict[str, Any]] = {}
    scenario_keys: set[tuple[int, float, int]] = set()
    for row in rows:
        if set(row) != set(SCENARIO_MANIFEST_FIELDS):
            raise SandboxIntegrityError("scenario manifest fields are invalid")
        scenario_id = _checked_int(row["scenario_id"], label="scenario_id")
        evaluation_id = _checked_int(row["evaluation_id"], label="evaluation_id")
        target_force_n = _checked_real(
            row["target_force_n"], label="target_force_n", minimum=0.0
        )
        if target_force_n <= 0.0:
            raise SandboxIntegrityError("target_force_n must be positive")
        seed = _checked_int(row["seed"], label="seed")
        max_control_steps = _checked_int(
            row["max_control_steps"], label="max_control_steps", minimum=1
        )
        if max_control_steps > maximum_steps:
            raise SandboxIntegrityError("scenario exceeds permission step bound")
        family = row["family"]
        expected_trigger = row["expected_trigger"]
        parameters_json = row["parameters_json"]
        if not isinstance(family, str) or not family:
            raise SandboxIntegrityError("scenario family must be nonempty")
        if not isinstance(expected_trigger, str):
            raise SandboxIntegrityError("expected_trigger must be text")
        if not isinstance(parameters_json, str):
            raise SandboxIntegrityError("parameters_json must be canonical JSON text")
        try:
            parameters = json.loads(parameters_json)
        except json.JSONDecodeError as exc:
            raise SandboxIntegrityError("parameters_json is malformed") from exc
        if _canonical_json_bytes(parameters).decode().strip() != parameters_json:
            raise SandboxIntegrityError("parameters_json must use canonical encoding")
        if evaluation_id in by_evaluation:
            raise SandboxIntegrityError("duplicate evaluation_id in scenario manifest")
        scenario_key = (scenario_id, target_force_n, seed)
        if scenario_key in scenario_keys:
            raise SandboxIntegrityError("duplicate scenario/target/seed identity")
        scenario_keys.add(scenario_key)
        normalized_row = {
            "scenario_id": scenario_id,
            "evaluation_id": evaluation_id,
            "family": family,
            "target_force_n": target_force_n,
            "seed": seed,
            "max_control_steps": max_control_steps,
            "expected_trigger": expected_trigger,
            "parameters_json": parameters_json,
        }
        normalized.append(normalized_row)
        by_evaluation[evaluation_id] = normalized_row
    return normalized, by_evaluation


def _validate_trusted_source_roles(
    rows: Sequence[Mapping[str, Any]],
    *,
    definition: Mapping[str, Any],
    permission: Mapping[str, Any],
) -> None:
    expected = {
        "controller": definition["controller_sha256"],
        "adapter": permission["trusted_infrastructure_sha256"]["adapter"],
        "binding": permission["trusted_infrastructure_sha256"]["binding"],
        "runner": permission["trusted_infrastructure_sha256"]["runner"],
        "storage": permission["trusted_infrastructure_sha256"]["storage"],
    }
    observed: dict[str, str] = {}
    for row in rows:
        role = str(row["role"])
        if role not in expected:
            continue
        if role in observed:
            raise SandboxIntegrityError(f"source manifest duplicates trusted role {role}")
        observed[role] = str(row["sha256"])
    if observed != expected:
        raise SandboxPermissionError("source manifest does not close trusted source identities")


def _frame(rows: Sequence[Mapping[str, Any]], schema: pl.Schema, *, table: str) -> pl.DataFrame:
    if not rows:
        raise SandboxIntegrityError(f"{table} must contain at least one row")
    expected = set(schema.names())
    for index, row in enumerate(rows):
        if set(row) != expected:
            missing = sorted(expected - set(row))
            extra = sorted(set(row) - expected)
            raise SandboxIntegrityError(
                f"{table} row {index} schema mismatch: missing={missing}, extra={extra}"
            )
    try:
        result = pl.from_dicts(list(rows), schema=schema, strict=True)
    except Exception as exc:
        raise SandboxIntegrityError(f"{table} cannot be represented by its schema") from exc
    nullable = NULLABLE_FIELDS[table]
    for name in schema.names():
        if name not in nullable and result[name].null_count():
            raise SandboxIntegrityError(f"{table}.{name} contains missing evidence")
    for name, dtype in schema.items():
        if dtype in (pl.Float32, pl.Float64):
            invalid = result.select(
                (pl.col(name).is_not_null() & ~pl.col(name).is_finite()).any()
            ).item()
            if invalid:
                raise SandboxIntegrityError(f"{table}.{name} contains a nonfinite value")
    return result


def _key(row: Mapping[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(row["run_id"]),
        int(row["scenario_id"]),
        int(row["evaluation_id"]),
        int(row["segment_index"]),
        int(row["control_step_index"]),
    )


def _identity_check(
    rows: Sequence[Mapping[str, Any]], *, run_id: str, scenario_id: int, evaluation_id: int
) -> None:
    for row in rows:
        if (
            row["run_id"] != run_id
            or int(row["scenario_id"]) != scenario_id
            or int(row["evaluation_id"]) != evaluation_id
        ):
            raise SandboxIntegrityError("table row identity differs from its scenario")


def _values(row: Mapping[str, Any], names: Sequence[str]) -> tuple[float, ...]:
    return tuple(float(row[name]) for name in names)


def _distance(left: Sequence[float], right: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)))


def _optional_close(left: Any, right: Any, tolerance: float) -> bool:
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= tolerance


def _quaternion_distance(left: Sequence[float], right: Sequence[float]) -> float:
    direct = _distance(left, right)
    antipodal = _distance(left, tuple(-float(value) for value in right))
    return min(direct, antipodal)


def _validate_quaternion(
    value: Sequence[float], *, label: str, tolerance: float
) -> None:
    norm = math.sqrt(sum(float(item) ** 2 for item in value))
    if abs(norm - 1.0) > tolerance:
        raise SandboxIntegrityError(f"{label} quaternion is not normalized")


def _validate_unit_geometry(
    row: Mapping[str, Any],
    *,
    tolerance: float,
    basis_tolerance: float,
    definition: Mapping[str, Any],
    validate_command_time_requests: bool = True,
) -> None:
    projection_normal = _values(
        row,
        (
            "projection_outward_normal_x",
            "projection_outward_normal_y",
            "projection_outward_normal_z",
        ),
    )
    command_normal = _values(
        row,
        (
            "command_outward_normal_x",
            "command_outward_normal_y",
            "command_outward_normal_z",
        ),
    )
    recovery_normal = _values(
        row,
        (
            "committed_recovery_outward_normal_x",
            "committed_recovery_outward_normal_y",
            "committed_recovery_outward_normal_z",
        ),
    )
    projection_tangent = _values(
        row,
        ("projection_tangent_x", "projection_tangent_y", "projection_tangent_z"),
    )
    command_tangent = _values(
        row, ("command_tangent_x", "command_tangent_y", "command_tangent_z")
    )
    for label, vector in (
        ("projection normal", projection_normal),
        ("command normal", command_normal),
        ("committed recovery normal", recovery_normal),
        ("projection tangent", projection_tangent),
        ("command tangent", command_tangent),
    ):
        if (
            abs(math.sqrt(sum(value * value for value in vector)) - 1.0)
            > basis_tolerance
        ):
            raise SandboxIntegrityError(f"{label} must be a unit vector")
    if (
        abs(sum(a * b for a, b in zip(projection_normal, projection_tangent)))
        > basis_tolerance
    ):
        raise SandboxIntegrityError("projection normal and tangent are not orthogonal")
    if (
        abs(sum(a * b for a, b in zip(command_normal, command_tangent)))
        > basis_tolerance
    ):
        raise SandboxIntegrityError("command normal and tangent are not orthogonal")
    velocity = _values(
        row,
        (
            "pre_contact_tool_vx_m_s",
            "pre_contact_tool_vy_m_s",
            "pre_contact_tool_vz_m_s",
        ),
    )
    projected_velocity = sum(
        left * right for left, right in zip(velocity, projection_normal)
    )
    if abs(projected_velocity - float(row["normal_velocity_outward_m_s"])) > tolerance:
        raise SandboxIntegrityError("normal velocity does not match tool velocity projection")
    contact_tool_position = _values(
        row,
        ("pre_contact_tool_x_m", "pre_contact_tool_y_m", "pre_contact_tool_z_m"),
    )
    projection_point = _values(
        row,
        (
            "task_projection_point_x_m",
            "task_projection_point_y_m",
            "task_projection_point_z_m",
        ),
    )
    surface_reference = _values(
        row,
        ("surface_reference_x_m", "surface_reference_y_m", "surface_reference_z_m"),
    )
    hover_target = _values(
        row,
        (
            "recovery_hover_target_x_m",
            "recovery_hover_target_y_m",
            "recovery_hover_target_z_m",
        ),
    )
    path_length = float(row["task_path_length_m"])
    if path_length <= 0.0:
        raise SandboxIntegrityError("task path length must be positive")
    projection_progress = float(row["task_projection_progress"])
    if not 0.0 <= projection_progress <= 1.0:
        raise SandboxIntegrityError("task projection progress lies outside [0,1]")
    if abs(float(row["instantaneous_progress"]) - projection_progress) > tolerance:
        raise SandboxIntegrityError("instantaneous and projected progress differ")
    requested_tangential_step = float(row["requested_tangential_step_m"])
    if (
        requested_tangential_step < 0.0
        or requested_tangential_step
        > float(definition["maximum_task_tangential_step_m"]) + tolerance
    ):
        raise SandboxIntegrityError("requested task step lies outside its frozen bound")
    if abs(
        float(row["task_projection_arc_length_m"]) - projection_progress * path_length
    ) > tolerance:
        raise SandboxIntegrityError("path arc length does not close to projected progress")
    displacement = tuple(
        contact_tool_position[index] - projection_point[index] for index in range(3)
    )
    normal_displacement = sum(
        displacement[index] * projection_normal[index] for index in range(3)
    )
    lateral = tuple(
        displacement[index] - normal_displacement * projection_normal[index]
        for index in range(3)
    )
    clearance_vector = tuple(
        contact_tool_position[index] - surface_reference[index] for index in range(3)
    )
    expected_signed_clearance = sum(
        clearance_vector[index] * recovery_normal[index] for index in range(3)
    )
    expected_clearance = max(0.0, expected_signed_clearance)
    expected_hover_target = tuple(
        surface_reference[index]
        + float(definition["recovery_hover_clearance_m"]) * recovery_normal[index]
        for index in range(3)
    )
    expected_hover_error = _distance(contact_tool_position, hover_target)
    if _distance(hover_target, expected_hover_target) > tolerance:
        raise SandboxIntegrityError("recovery hover target does not close to surface normal")
    if abs(math.sqrt(sum(item * item for item in lateral)) - float(row["geometric_track_error_m"])) > tolerance:
        raise SandboxIntegrityError(
            "geometric error does not close to contact tool and task path"
        )
    if abs(
        expected_signed_clearance - float(row["signed_recovery_clearance_m"])
    ) > tolerance:
        raise SandboxIntegrityError(
            "signed recovery clearance does not close to contact tool and surface"
        )
    if abs(expected_clearance - float(row["recovery_clearance_m"])) > tolerance:
        raise SandboxIntegrityError(
            "recovery clearance does not close to contact tool and surface"
        )
    if abs(expected_hover_error - float(row["recovery_hover_pose_error_m"])) > tolerance:
        raise SandboxIntegrityError("hover error does not close to contact tool and target")
    if validate_command_time_requests:
        recovery_delta = _values(
            row,
            ("requested_recovery_dx_m", "requested_recovery_dy_m", "requested_recovery_dz_m"),
        )
        recovery_rotation = _values(
            row,
            (
                "requested_recovery_rx_rad",
                "requested_recovery_ry_rad",
                "requested_recovery_rz_rad",
            ),
        )
        if _distance(recovery_delta, (0.0, 0.0, 0.0)) > float(
            definition["maximum_recovery_reposition_step_m"]
        ) + tolerance:
            raise SandboxIntegrityError("requested recovery translation exceeds its bound")
        if _distance(recovery_rotation, (0.0, 0.0, 0.0)) > float(
            definition["maximum_recovery_rotation_step_rad"]
        ) + tolerance:
            raise SandboxIntegrityError("requested recovery rotation exceeds its bound")
        if sum(recovery_delta[index] * recovery_normal[index] for index in range(3)) < -tolerance:
            raise SandboxIntegrityError("requested recovery translation points inward")
        toward_hover = tuple(
            hover_target[index] - contact_tool_position[index] for index in range(3)
        )
        if _distance(recovery_delta, (0.0, 0.0, 0.0)) > tolerance and sum(
            recovery_delta[index] * toward_hover[index] for index in range(3)
        ) <= 0.0:
            raise SandboxIntegrityError("requested recovery translation does not point to hover")
        cross_track = _values(
            row,
            (
                "requested_cross_track_dx_m",
                "requested_cross_track_dy_m",
                "requested_cross_track_dz_m",
            ),
        )
        cross_norm = _distance(cross_track, (0.0, 0.0, 0.0))
        cross_bound = float(definition.get("maximum_cross_track_correction_step_m", 0.0005))
        cross_enter = float(definition.get("cross_track_correction_enter_m", 0.003))
        if cross_norm > cross_bound + tolerance:
            raise SandboxIntegrityError("requested cross-track correction exceeds its bound")
        if cross_norm > tolerance:
            if float(row["geometric_track_error_m"]) < cross_enter - tolerance:
                raise SandboxIntegrityError("cross-track correction is below its entry threshold")
            lateral_norm = math.sqrt(sum(item * item for item in lateral))
            scale = min(1.0, cross_bound / lateral_norm)
            expected_cross = tuple(-scale * item for item in lateral)
            if _distance(cross_track, expected_cross) > tolerance:
                raise SandboxIntegrityError("cross-track correction geometry is inconsistent")


def _validate_command_components(
    row: Mapping[str, Any],
    *,
    prefix: str,
    tolerance: float,
    rotation_tolerance: float,
    allow_precision_r5_normal_cross_track: bool = False,
) -> None:
    normal_axis = _values(
        row,
        (
            "command_outward_normal_x",
            "command_outward_normal_y",
            "command_outward_normal_z",
        ),
    )
    tangent_axis = _values(
        row, ("command_tangent_x", "command_tangent_y", "command_tangent_z")
    )
    total = _values(row, (f"{prefix}_dx_m", f"{prefix}_dy_m", f"{prefix}_dz_m"))
    normal = _values(
        row,
        (
            f"{prefix}_normal_dx_m",
            f"{prefix}_normal_dy_m",
            f"{prefix}_normal_dz_m",
        ),
    )
    task = _values(
        row,
        (f"{prefix}_task_dx_m", f"{prefix}_task_dy_m", f"{prefix}_task_dz_m"),
    )
    recovery = _values(
        row,
        (
            f"{prefix}_recovery_dx_m",
            f"{prefix}_recovery_dy_m",
            f"{prefix}_recovery_dz_m",
        ),
    )
    cross_track = _values(
        row,
        (
            f"{prefix}_cross_track_dx_m",
            f"{prefix}_cross_track_dy_m",
            f"{prefix}_cross_track_dz_m",
        ),
    )
    component_sum = tuple(
        normal[index] + task[index] + recovery[index] + cross_track[index]
        for index in range(3)
    )
    if _distance(total, component_sum) > tolerance:
        raise SandboxIntegrityError(f"{prefix} command components do not close")
    normal_step = float(row[f"{prefix}_normal_step_m"])
    tangent_step = float(row[f"{prefix}_tangential_step_m"])
    expected_normal = tuple(-axis * normal_step for axis in normal_axis)
    expected_task = tuple(axis * tangent_step for axis in tangent_axis)
    if _distance(normal, expected_normal) > tolerance:
        raise SandboxIntegrityError(f"{prefix} normal component has the wrong geometry")
    if _distance(task, expected_task) > tolerance:
        raise SandboxIntegrityError(f"{prefix} task component has the wrong geometry")
    recovery_permitted = bool(row[f"{prefix}_recovery_reposition_permitted"])
    cross_track_permitted = bool(
        row[f"{prefix}_cross_track_correction_permitted"]
    )
    if recovery_permitted and cross_track_permitted:
        raise SandboxIntegrityError(
            f"{prefix} recovery and cross-track authority overlap"
        )
    if recovery_permitted:
        if _distance(normal, (0.0, 0.0, 0.0)) > tolerance or _distance(
            task, (0.0, 0.0, 0.0)
        ) > tolerance:
            raise SandboxIntegrityError(
                f"{prefix} recovery overlaps normal or task authority"
            )
    elif _distance(recovery, (0.0, 0.0, 0.0)) > tolerance:
        raise SandboxIntegrityError(f"{prefix} recovery component lacks authority")
    if cross_track_permitted:
        forbidden = (task, recovery)
        if not allow_precision_r5_normal_cross_track:
            forbidden = (normal, *forbidden)
        if any(
            _distance(component, (0.0, 0.0, 0.0)) > tolerance
            for component in forbidden
        ):
            raise SandboxIntegrityError(
                f"{prefix} cross-track correction overlaps another authority"
            )
    elif _distance(cross_track, (0.0, 0.0, 0.0)) > tolerance:
        raise SandboxIntegrityError(
            f"{prefix} cross-track component lacks authority"
        )
    if prefix == "issued":
        requested_recovery = _values(
            row,
            (
                "requested_recovery_dx_m",
                "requested_recovery_dy_m",
                "requested_recovery_dz_m",
            ),
        )
        expected_recovery = (
            requested_recovery if recovery_permitted else (0.0, 0.0, 0.0)
        )
        if _distance(recovery, expected_recovery) > tolerance:
            raise SandboxIntegrityError("issued recovery component differs from request")
        requested_cross_track = _values(
            row,
            (
                "requested_cross_track_dx_m",
                "requested_cross_track_dy_m",
                "requested_cross_track_dz_m",
            ),
        )
        expected_cross_track = (
            requested_cross_track
            if cross_track_permitted
            else (0.0, 0.0, 0.0)
        )
        if _distance(cross_track, expected_cross_track) > tolerance:
            raise SandboxIntegrityError(
                "issued cross-track component differs from request"
            )
        requested_rotation = _values(
            row,
            (
                "requested_recovery_rx_rad",
                "requested_recovery_ry_rad",
                "requested_recovery_rz_rad",
            ),
        )
        expected_rotation = (
            requested_rotation if recovery_permitted else (0.0, 0.0, 0.0)
        )
        issued_recovery_rotation = _values(
            row,
            (
                "issued_recovery_rx_rad",
                "issued_recovery_ry_rad",
                "issued_recovery_rz_rad",
            ),
        )
        issued_rotation = _values(
            row,
            (
                "issued_rotation_rx_rad",
                "issued_rotation_ry_rad",
                "issued_rotation_rz_rad",
            ),
        )
        if _distance(issued_recovery_rotation, expected_rotation) > rotation_tolerance or _distance(
            issued_rotation, expected_rotation
        ) > rotation_tolerance:
            raise SandboxIntegrityError("issued recovery rotation does not close")


def _command_readback_violations(
    control_rows: Sequence[Mapping[str, Any]], *, definition: Mapping[str, Any]
) -> int:
    count = 0
    translation_pairs = (
        ("issued_normal_step_m", "applied_normal_step_m"),
        ("issued_tangential_step_m", "applied_tangential_step_m"),
        *( 
            (f"issued_{component}_{axis}_m", f"applied_{component}_{axis}_m")
            for component in ("normal", "task", "recovery", "cross_track")
            for axis in ("d" + letter for letter in "xyz")
        ),
        ("issued_dx_m", "applied_dx_m"),
        ("issued_dy_m", "applied_dy_m"),
        ("issued_dz_m", "applied_dz_m"),
    )
    rotation_pairs = (
        *( 
            (f"issued_recovery_r{axis}_rad", f"applied_recovery_r{axis}_rad")
            for axis in "xyz"
        ),
        *( 
            (f"issued_rotation_r{axis}_rad", f"applied_rotation_r{axis}_rad")
            for axis in "xyz"
        ),
    )
    tolerance = float(definition["command_readback_tolerance_m"])
    rotation_tolerance = float(definition["rotation_readback_tolerance_rad"])
    for row in control_rows:
        mismatch = not bool(row["readback_accepted"])
        mismatch = mismatch or any(
            abs(float(row[left]) - float(row[right])) > tolerance
            for left, right in translation_pairs
        )
        mismatch = mismatch or any(
            abs(float(row[left]) - float(row[right])) > rotation_tolerance
            for left, right in rotation_pairs
        )
        mismatch = mismatch or (
            bool(row["issued_recovery_reposition_permitted"])
            != bool(row["applied_recovery_reposition_permitted"])
        )
        mismatch = mismatch or (
            bool(row["issued_cross_track_correction_permitted"])
            != bool(row["applied_cross_track_correction_permitted"])
        )
        mismatch = mismatch or (
            row["issued_command_frame_id"] != row["applied_command_frame_id"]
            or row["issued_command_mode"] != row["applied_command_mode"]
            or row["readback_stage"] != definition["required_readback_stage"]
        )
        count += int(mismatch)
    return count


def _authority_violations(
    diagnostic_rows: Sequence[Mapping[str, Any]],
    *,
    metric_version: str = AUTHORITY_METRIC_STATE_AWARE,
) -> int:
    if metric_version not in {
        AUTHORITY_METRIC_LEGACY,
        AUTHORITY_METRIC_STATE_AWARE,
        AUTHORITY_METRIC_SPLIT_V3,
        AUTHORITY_METRIC_PATH_FRAME_V4,
        AUTHORITY_METRIC_PRECISION_R5,
        AUTHORITY_METRIC_SYSTEM_R4,
    }:
        raise SandboxIntegrityError("unsupported authority metric version")
    if metric_version == AUTHORITY_METRIC_SPLIT_V3:
        return authority_violations_s3(diagnostic_rows)
    if metric_version == AUTHORITY_METRIC_PATH_FRAME_V4:
        return authority_violations_s5(diagnostic_rows)
    if metric_version == AUTHORITY_METRIC_PRECISION_R5:
        return authority_violations_precision_r5(diagnostic_rows)
    if metric_version == AUTHORITY_METRIC_SYSTEM_R4:
        # The R4 diagnostic schema intentionally remains the immutable base
        # schema.  Its bounded subattempt ledgers are reconstructed from the
        # versioned transition reasons by the metric.
        return authority_violations_system_r4(
            diagnostic_rows,
        )
    count = 0
    tolerance = 1e-12
    for row in diagnostic_rows:
        stable = bool(row["stable_track"])
        permission = bool(row["tangential_motion_permitted"])
        committed = float(row["committed_progress_after"]) - float(
            row["committed_progress_before"]
        )
        violation = permission != stable
        violation = violation or (
            bool(row["transitioned"])
            != (str(row["state_before"]) != str(row["state_after"]))
        )
        violation = violation or (
            stable
            and (
                row["state_after"] != "TRACK"
                or not bool(row["force_track_ready"])
                or not bool(row["geometric_track_ready"])
            )
        )
        violation = violation or (
            not permission and abs(float(row["executed_tangential_step_m"])) > tolerance
        )
        violation = violation or (not stable and abs(committed) > tolerance)
        if not stable and metric_version == AUTHORITY_METRIC_LEGACY:
            violation = violation or (
                abs(float(row["integral_after_n_s"])) > tolerance
            )
        elif not stable:
            track_hold = (
                str(row["state_before"]) == "TRACK"
                and str(row["state_after"]) == "TRACK"
            )
            if track_hold:
                violation = violation or (
                    abs(
                        float(row["integral_after_n_s"])
                        - float(row["integral_before_n_s"])
                    )
                    > tolerance
                )
            else:
                violation = violation or (
                    abs(float(row["integral_after_n_s"])) > tolerance
                )
        violation = violation or (
            permission
            and (
                bool(row["recovery_reposition_permitted"])
                or bool(row.get("cross_track_correction_permitted", False))
            )
        )
        violation = violation or (
            bool(row["transitioned"])
            and (
                permission
                or abs(float(row["executed_tangential_step_m"])) > tolerance
                or abs(committed) > tolerance
            )
        )
        violation = violation or (
            row["state_after"] == "SAFE_HOLD"
            and (
                abs(float(row["executed_normal_step_m"])) > tolerance
                or abs(float(row["executed_tangential_step_m"])) > tolerance
                or bool(row["recovery_reposition_permitted"])
                or bool(row.get("cross_track_correction_permitted", False))
                or abs(float(row["integral_after_n_s"])) > tolerance
                or abs(committed) > tolerance
            )
        )
        violation = violation or (
            row["state_after"] in {"HEADROOM", "REBOUND_GUARD"}
            and float(row["executed_normal_step_m"]) > tolerance
        )
        count += int(violation)
    return count


def _validate_evaluation(
    *,
    run_id: str,
    scenario: Mapping[str, Any],
    definition: Mapping[str, Any],
    native: pl.DataFrame,
    control: pl.DataFrame,
    diagnostic: pl.DataFrame,
    episode: pl.DataFrame,
) -> bool:
    native_rows = native.to_dicts()
    control_rows = control.to_dicts()
    diagnostic_rows = diagnostic.to_dicts()
    episode_rows = episode.to_dicts()
    evaluation_id = int(scenario["evaluation_id"])
    scenario_id = int(scenario["scenario_id"])
    _identity_check(
        native_rows, run_id=run_id, scenario_id=scenario_id, evaluation_id=evaluation_id
    )
    _identity_check(
        control_rows, run_id=run_id, scenario_id=scenario_id, evaluation_id=evaluation_id
    )
    _identity_check(
        diagnostic_rows,
        run_id=run_id,
        scenario_id=scenario_id,
        evaluation_id=evaluation_id,
    )
    if len(episode_rows) != 1:
        raise SandboxIntegrityError("episode table must contain exactly one row")
    _identity_check(
        episode_rows, run_id=run_id, scenario_id=scenario_id, evaluation_id=evaluation_id
    )
    if any(float(row["audit_force_n"]) < 0.0 for row in native_rows):
        raise SandboxIntegrityError("native force must be nonnegative")
    if any(
        int(row["native_sample_index"]) < 0
        or int(row["substep_index"]) < 0
        or int(row["time_ns"]) < 0
        for row in native_rows
    ):
        raise SandboxIntegrityError("native sample indices and times must be nonnegative")

    control_keys = [_key(row) for row in control_rows]
    diagnostic_keys = [_key(row) for row in diagnostic_rows]
    if len(set(control_keys)) != len(control_keys):
        raise SandboxIntegrityError("duplicate control primary key")
    if len(set(diagnostic_keys)) != len(diagnostic_keys):
        raise SandboxIntegrityError("duplicate diagnostic primary key")
    if set(control_keys) != set(diagnostic_keys):
        raise SandboxIntegrityError("control and diagnostic primary keys do not close")
    control_command_ids = [str(row["command_id"]) for row in control_rows]
    if len(set(control_command_ids)) != len(control_command_ids):
        raise SandboxIntegrityError("command_id is not unique within the evaluation")

    native_primary_keys = [
        (
            row["run_id"],
            int(row["scenario_id"]),
            int(row["evaluation_id"]),
            int(row["segment_index"]),
            int(row["native_sample_index"]),
        )
        for row in native_rows
    ]
    if len(set(native_primary_keys)) != len(native_primary_keys):
        raise SandboxIntegrityError("duplicate native primary key")

    expected_native_steps = int(definition["expected_native_steps_per_control"])
    native_period_ns = int(definition["native_period_ns"])
    control_period_ns = int(definition["control_period_ns"])
    pose_tolerance = float(definition["pose_alignment_tolerance_m"])
    command_tolerance = float(definition["command_readback_tolerance_m"])
    diagnostics = {_key(row): row for row in diagnostic_rows}
    streams: dict[tuple[str, int, int, int], list[Mapping[str, Any]]] = {}
    for row in control_rows:
        stream = _key(row)[:-1]
        streams.setdefault(stream, []).append(row)
    for stream, rows in streams.items():
        rows.sort(key=lambda item: int(item["control_step_index"]))
        steps = [int(item["control_step_index"]) for item in rows]
        if steps != list(range(len(steps))):
            raise SandboxIntegrityError("control steps are not contiguous from zero")
        native_stream = [
            row
            for row in native_rows
            if (
                row["run_id"],
                int(row["scenario_id"]),
                int(row["evaluation_id"]),
                int(row["segment_index"]),
            )
            == stream
        ]
        native_stream.sort(key=lambda item: int(item["native_sample_index"]))
        native_indices = [int(item["native_sample_index"]) for item in native_stream]
        if native_indices != list(range(len(native_indices))):
            raise SandboxIntegrityError("native sample indices are not contiguous from zero")
        native_times = [int(item["time_ns"]) for item in native_stream]
        if native_times != sorted(native_times) or len(set(native_times)) != len(native_times):
            raise SandboxIntegrityError("native stream times are not strictly increasing")
        native_control_keys = {
            (
                row["run_id"],
                int(row["scenario_id"]),
                int(row["evaluation_id"]),
                int(row["segment_index"]),
                int(row["control_step_index"]),
            )
            for row in native_stream
        }
        if native_control_keys != {
            _key(row) for row in rows
        }:
            raise SandboxIntegrityError("native foreign keys do not close to control keys")
        previous_last: Mapping[str, Any] | None = None
        previous_post_time: int | None = None
        previous_diag: Mapping[str, Any] | None = None
        expected_first_native_index = 0
        command_ids: set[str] = set()
        reference_row = rows[0]
        for row in rows:
            key = _key(row)
            step = int(row["control_step_index"])
            if row["command_id"] in command_ids:
                raise SandboxIntegrityError("command_id is not unique in its stream")
            command_ids.add(str(row["command_id"]))
            diag = diagnostics[key]
            if not str(diag["state_before"]) or not str(diag["state_after"]):
                raise SandboxIntegrityError("diagnostic state identity is empty")
            if (
                diag["state_before"] not in SUPERVISOR_STATES
                or diag["state_after"] not in SUPERVISOR_STATES
            ):
                raise SandboxIntegrityError("diagnostic supervisor state is unknown")
            if not str(diag["transition_reason"]):
                raise SandboxIntegrityError("diagnostic transition reason is empty")
            if any(
                not str(row[name])
                for name in (
                    "bootstrap_id",
                    "geometry_frame_id",
                    "contact_tool_frame_id",
                    "robot_tcp_frame_id",
                    "task_path_id",
                    "surface_id",
                )
            ):
                raise SandboxIntegrityError("physical provenance identity is empty")
            if (
                row["command_frame_id"] != definition["expected_command_frame_id"]
                or row["issued_command_frame_id"] != row["command_frame_id"]
                or row["issued_command_mode"] != definition["expected_command_mode"]
            ):
                raise SandboxIntegrityError("issued command frame or mode is unregistered")
            for name in (
                "bootstrap_id",
                "geometry_frame_id",
                "command_frame_id",
                "contact_tool_frame_id",
                "robot_tcp_frame_id",
                "task_path_id",
                "surface_id",
                "task_path_length_m",
            ):
                left, right = row[name], reference_row[name]
                if isinstance(left, str):
                    changed = left != right
                else:
                    changed = abs(float(left) - float(right)) > pose_tolerance
                if changed:
                    raise SandboxIntegrityError(f"static stream field {name} changed")
            if abs(float(diag["measured_force_n"]) - float(row["pre_force_n"])) > 1e-12:
                raise SandboxIntegrityError("diagnostic force differs from pre-step force")
            if abs(float(diag["target_force_n"]) - float(row["target_force_n"])) > 1e-12:
                raise SandboxIntegrityError("diagnostic target differs from control target")
            if bool(diag["contact_observed"]) != bool(row["pre_contact_observed"]):
                raise SandboxIntegrityError("diagnostic contact differs from pre-step state")
            diagnostic_input_pairs = (
                ("normal_velocity_outward_m_s", "normal_velocity_outward_m_s"),
                ("instantaneous_progress", "instantaneous_progress"),
                ("geometric_track_error_m", "geometric_track_error_m"),
                ("requested_tangential_step_m", "requested_tangential_step_m"),
                ("recovery_hover_pose_error_m", "recovery_hover_pose_error_m"),
                ("recovery_clearance_m", "recovery_clearance_m"),
            )
            if any(
                abs(float(diag[diagnostic_name]) - float(row[control_name]))
                > pose_tolerance
                for diagnostic_name, control_name in diagnostic_input_pairs
            ):
                raise SandboxIntegrityError(
                    "diagnostic supervisor input differs from pre-step evidence"
                )
            if abs(
                float(diag["executed_normal_step_m"])
                - float(row["issued_normal_step_m"])
            ) > 1e-12:
                raise SandboxIntegrityError("issued normal command differs from diagnostic")
            if abs(
                float(diag["executed_tangential_step_m"])
                - float(row["issued_tangential_step_m"])
            ) > 1e-12:
                raise SandboxIntegrityError("issued tangential command differs from diagnostic")
            if bool(diag["recovery_reposition_permitted"]) != bool(
                row["issued_recovery_reposition_permitted"]
            ):
                raise SandboxIntegrityError("issued recovery authority differs from diagnostic")
            if bool(diag["cross_track_correction_permitted"]) != bool(
                row["issued_cross_track_correction_permitted"]
            ):
                raise SandboxIntegrityError(
                    "issued cross-track authority differs from diagnostic"
                )
            if previous_diag is not None:
                if diag["state_before"] != previous_diag["state_after"]:
                    raise SandboxIntegrityError(
                        "diagnostic state chain is discontinuous"
                    )
                if not _optional_close(
                    diag["integral_before_n_s"],
                    previous_diag["integral_after_n_s"],
                    pose_tolerance,
                ):
                    raise SandboxIntegrityError(
                        "diagnostic integral chain is discontinuous"
                    )
                if not _optional_close(
                    diag["committed_progress_before"],
                    previous_diag["committed_progress_after"],
                    pose_tolerance,
                ):
                    raise SandboxIntegrityError(
                        "diagnostic committed-progress chain is discontinuous"
                    )
                if int(diag["verified_return_count_before"]) != int(
                    previous_diag["verified_return_count"]
                ):
                    raise SandboxIntegrityError(
                        "diagnostic verified-return chain is discontinuous"
                    )
                if not _optional_close(
                    diag["pre_recovery_highwater_progress_before"],
                    previous_diag["pre_recovery_highwater_progress_after"],
                    pose_tolerance,
                ):
                    raise SandboxIntegrityError(
                        "diagnostic recovery-highwater chain is discontinuous"
                    )
                if int(diag["episode_total_recovery_cycles"]) < int(
                    previous_diag["episode_total_recovery_cycles"]
                ) or int(diag["episode_total_recovery_samples"]) < int(
                    previous_diag["episode_total_recovery_samples"]
                ):
                    raise SandboxIntegrityError(
                        "diagnostic episode recovery counters decreased"
                    )
            _validate_unit_geometry(
                row,
                tolerance=pose_tolerance,
                basis_tolerance=float(definition["basis_tolerance"]),
                definition=definition,
            )
            _validate_command_components(
                row,
                prefix="issued",
                tolerance=command_tolerance,
                rotation_tolerance=float(definition["rotation_readback_tolerance_rad"]),
                allow_precision_r5_normal_cross_track=(
                    definition["authority_metric_version"]
                    == AUTHORITY_METRIC_PRECISION_R5
                ),
            )
            _validate_command_components(
                row,
                prefix="applied",
                tolerance=command_tolerance,
                rotation_tolerance=float(definition["rotation_readback_tolerance_rad"]),
                allow_precision_r5_normal_cross_track=(
                    definition["authority_metric_version"]
                    == AUTHORITY_METRIC_PRECISION_R5
                ),
            )
            pre_robot_tcp_quaternion = _values(
                row,
                (
                    "pre_robot_tcp_qw",
                    "pre_robot_tcp_qx",
                    "pre_robot_tcp_qy",
                    "pre_robot_tcp_qz",
                ),
            )
            pre_contact_tool_quaternion = _values(
                row,
                (
                    "pre_contact_tool_qw",
                    "pre_contact_tool_qx",
                    "pre_contact_tool_qy",
                    "pre_contact_tool_qz",
                ),
            )
            _validate_quaternion(
                pre_robot_tcp_quaternion,
                label="pre-step robot TCP",
                tolerance=float(definition["quaternion_tolerance"]),
            )
            _validate_quaternion(
                pre_contact_tool_quaternion,
                label="pre-step contact tool",
                tolerance=float(definition["quaternion_tolerance"]),
            )
            if abs(
                max(0.0, float(row["signed_recovery_clearance_m"]))
                - float(row["recovery_clearance_m"])
            ) > pose_tolerance:
                raise SandboxIntegrityError(
                    "signed and nonnegative recovery clearances do not close"
                )
            if int(row["issued_time_ns"]) != int(row["pre_time_ns"]):
                raise SandboxIntegrityError("issued command is not aligned to pre-step time")
            if min(
                int(row["pre_time_ns"]),
                int(row["force_source_time_ns"]),
                int(row["issued_time_ns"]),
                int(row["readback_time_ns"]),
                int(row["post_time_ns"]),
            ) < 0:
                raise SandboxIntegrityError("control times must be nonnegative")
            if int(row["force_source_time_ns"]) > int(row["pre_time_ns"]):
                raise SandboxIntegrityError("force source time follows the pre-step sample")
            if float(row["pre_force_n"]) < 0.0 or float(row["post_force_n"]) < 0.0:
                raise SandboxIntegrityError("control force must be nonnegative")
            if not 0.0 <= float(row["instantaneous_progress"]) <= 1.0:
                raise SandboxIntegrityError("instantaneous progress lies outside [0,1]")
            if int(row["readback_time_ns"]) < int(row["issued_time_ns"]):
                raise SandboxIntegrityError("readback predates command issue")
            if previous_post_time is not None and int(row["pre_time_ns"]) != previous_post_time:
                raise SandboxIntegrityError("pre-step time does not equal the prior post-step time")
            if int(row["post_time_ns"]) - int(row["pre_time_ns"]) != control_period_ns:
                raise SandboxIntegrityError("control interval duration differs from definition")
            produced = [
                native_row
                for native_row in native_stream
                if int(native_row["control_step_index"]) == step
            ]
            produced.sort(key=lambda item: int(item["substep_index"]))
            if len(produced) != expected_native_steps:
                raise SandboxIntegrityError(
                    "control step native-sample count differs from definition"
                )
            if [int(item["substep_index"]) for item in produced] != list(
                range(len(produced))
            ):
                raise SandboxIntegrityError("native substep indices are not contiguous")
            produced_indices = [int(item["native_sample_index"]) for item in produced]
            if produced_indices != list(
                range(expected_first_native_index, expected_first_native_index + len(produced))
            ):
                raise SandboxIntegrityError("native samples are interleaved across controls")
            expected_first_native_index += len(produced)
            if any(item["command_id"] != row["command_id"] for item in produced):
                raise SandboxIntegrityError("native sample command identity mismatch")
            for native_row in produced:
                if any(
                    native_row[name] != row[name]
                    for name in (
                        "geometry_frame_id",
                        "contact_tool_frame_id",
                        "robot_tcp_frame_id",
                        "task_path_id",
                        "surface_id",
                    )
                ):
                    raise SandboxIntegrityError(
                        "native physical provenance identity differs from control"
                    )
                _validate_quaternion(
                    _values(
                        native_row,
                        (
                            "robot_tcp_qw",
                            "robot_tcp_qx",
                            "robot_tcp_qy",
                            "robot_tcp_qz",
                        ),
                    ),
                    label="native robot TCP",
                    tolerance=float(definition["quaternion_tolerance"]),
                )
                _validate_quaternion(
                    _values(
                        native_row,
                        (
                            "contact_tool_qw",
                            "contact_tool_qx",
                            "contact_tool_qy",
                            "contact_tool_qz",
                        ),
                    ),
                    label="native contact tool",
                    tolerance=float(definition["quaternion_tolerance"]),
                )
                native_projection = dict(row)
                native_projection.update(
                    {
                        "normal_velocity_outward_m_s": native_row[
                            "normal_velocity_outward_m_s"
                        ],
                        "instantaneous_progress": native_row[
                            "instantaneous_progress"
                        ],
                        "geometric_track_error_m": native_row[
                            "geometric_track_error_m"
                        ],
                        "recovery_hover_pose_error_m": native_row[
                            "recovery_hover_pose_error_m"
                        ],
                        "recovery_clearance_m": native_row[
                            "recovery_clearance_m"
                        ],
                        "signed_recovery_clearance_m": native_row[
                            "signed_recovery_clearance_m"
                        ],
                    }
                )
                for name in (
                    "geometry_frame_id",
                    "contact_tool_frame_id",
                    "robot_tcp_frame_id",
                    "task_path_id",
                    "surface_id",
                    "surface_reference_x_m",
                    "surface_reference_y_m",
                    "surface_reference_z_m",
                    "projection_outward_normal_x",
                    "projection_outward_normal_y",
                    "projection_outward_normal_z",
                    "command_outward_normal_x",
                    "command_outward_normal_y",
                    "command_outward_normal_z",
                    "committed_recovery_outward_normal_x",
                    "committed_recovery_outward_normal_y",
                    "committed_recovery_outward_normal_z",
                    "task_projection_point_x_m",
                    "task_projection_point_y_m",
                    "task_projection_point_z_m",
                    "task_projection_progress",
                    "task_projection_arc_length_m",
                    "projection_tangent_x",
                    "projection_tangent_y",
                    "projection_tangent_z",
                    "command_tangent_x",
                    "command_tangent_y",
                    "command_tangent_z",
                    "task_path_length_m",
                    "recovery_hover_target_x_m",
                    "recovery_hover_target_y_m",
                    "recovery_hover_target_z_m",
                ):
                    native_projection[name] = native_row[name]
                for destination, source in (
                    ("pre_contact_tool_x_m", "contact_tool_x_m"),
                    ("pre_contact_tool_y_m", "contact_tool_y_m"),
                    ("pre_contact_tool_z_m", "contact_tool_z_m"),
                    ("pre_contact_tool_qw", "contact_tool_qw"),
                    ("pre_contact_tool_qx", "contact_tool_qx"),
                    ("pre_contact_tool_qy", "contact_tool_qy"),
                    ("pre_contact_tool_qz", "contact_tool_qz"),
                    ("pre_contact_tool_vx_m_s", "contact_tool_vx_m_s"),
                    ("pre_contact_tool_vy_m_s", "contact_tool_vy_m_s"),
                    ("pre_contact_tool_vz_m_s", "contact_tool_vz_m_s"),
                    ("pre_contact_tool_wx_rad_s", "contact_tool_wx_rad_s"),
                    ("pre_contact_tool_wy_rad_s", "contact_tool_wy_rad_s"),
                    ("pre_contact_tool_wz_rad_s", "contact_tool_wz_rad_s"),
                    ("pre_robot_tcp_x_m", "robot_tcp_x_m"),
                    ("pre_robot_tcp_y_m", "robot_tcp_y_m"),
                    ("pre_robot_tcp_z_m", "robot_tcp_z_m"),
                    ("pre_robot_tcp_qw", "robot_tcp_qw"),
                    ("pre_robot_tcp_qx", "robot_tcp_qx"),
                    ("pre_robot_tcp_qy", "robot_tcp_qy"),
                    ("pre_robot_tcp_qz", "robot_tcp_qz"),
                    ("pre_robot_tcp_vx_m_s", "robot_tcp_vx_m_s"),
                    ("pre_robot_tcp_vy_m_s", "robot_tcp_vy_m_s"),
                    ("pre_robot_tcp_vz_m_s", "robot_tcp_vz_m_s"),
                    ("pre_robot_tcp_wx_rad_s", "robot_tcp_wx_rad_s"),
                    ("pre_robot_tcp_wy_rad_s", "robot_tcp_wy_rad_s"),
                    ("pre_robot_tcp_wz_rad_s", "robot_tcp_wz_rad_s"),
                ):
                    native_projection[destination] = native_row[source]
                _validate_unit_geometry(
                    native_projection,
                    tolerance=pose_tolerance,
                    basis_tolerance=float(definition["basis_tolerance"]),
                    definition=definition,
                    validate_command_time_requests=False,
                )
            times = [int(item["time_ns"]) for item in produced]
            expected_times = [
                int(row["pre_time_ns"]) + (substep + 1) * native_period_ns
                for substep in range(expected_native_steps)
            ]
            if times != expected_times:
                raise SandboxIntegrityError("native sample cadence differs from definition")
            if times[0] <= int(row["readback_time_ns"]):
                raise SandboxIntegrityError("first native sample does not follow readback")
            last = produced[-1]
            if int(row["last_native_sample_index"]) != int(last["native_sample_index"]):
                raise SandboxIntegrityError("post-step last-native index mismatch")
            if int(row["last_native_time_ns"]) != int(last["time_ns"]):
                raise SandboxIntegrityError("post-step last-native time mismatch")
            if int(row["post_time_ns"]) != int(last["time_ns"]):
                raise SandboxIntegrityError("post-step time differs from last native time")
            if abs(float(row["post_force_n"]) - float(last["audit_force_n"])) > 1e-12:
                raise SandboxIntegrityError("post-step force differs from last native force")
            post_scalar_pairs = (
                ("post_normal_velocity_outward_m_s", "normal_velocity_outward_m_s"),
                ("post_instantaneous_progress", "instantaneous_progress"),
                ("post_geometric_track_error_m", "geometric_track_error_m"),
                ("post_recovery_hover_pose_error_m", "recovery_hover_pose_error_m"),
                ("post_recovery_clearance_m", "recovery_clearance_m"),
            )
            if any(
                abs(float(row[post_name]) - float(last[native_name])) > pose_tolerance
                for post_name, native_name in post_scalar_pairs
            ):
                raise SandboxIntegrityError(
                    "post-step scalar state differs from last native evidence"
                )
            if bool(row["post_contact_observed"]) != bool(last["contact_observed"]):
                raise SandboxIntegrityError(
                    "post-step contact differs from last native evidence"
                )
            post_native_pairs = (
                (("post_contact_tool_x_m", "post_contact_tool_y_m", "post_contact_tool_z_m"), ("contact_tool_x_m", "contact_tool_y_m", "contact_tool_z_m"), False),
                (("post_contact_tool_qw", "post_contact_tool_qx", "post_contact_tool_qy", "post_contact_tool_qz"), ("contact_tool_qw", "contact_tool_qx", "contact_tool_qy", "contact_tool_qz"), True),
                (("post_contact_tool_vx_m_s", "post_contact_tool_vy_m_s", "post_contact_tool_vz_m_s"), ("contact_tool_vx_m_s", "contact_tool_vy_m_s", "contact_tool_vz_m_s"), False),
                (("post_contact_tool_wx_rad_s", "post_contact_tool_wy_rad_s", "post_contact_tool_wz_rad_s"), ("contact_tool_wx_rad_s", "contact_tool_wy_rad_s", "contact_tool_wz_rad_s"), False),
                (("post_robot_tcp_x_m", "post_robot_tcp_y_m", "post_robot_tcp_z_m"), ("robot_tcp_x_m", "robot_tcp_y_m", "robot_tcp_z_m"), False),
                (("post_robot_tcp_qw", "post_robot_tcp_qx", "post_robot_tcp_qy", "post_robot_tcp_qz"), ("robot_tcp_qw", "robot_tcp_qx", "robot_tcp_qy", "robot_tcp_qz"), True),
                (("post_robot_tcp_vx_m_s", "post_robot_tcp_vy_m_s", "post_robot_tcp_vz_m_s"), ("robot_tcp_vx_m_s", "robot_tcp_vy_m_s", "robot_tcp_vz_m_s"), False),
                (("post_robot_tcp_wx_rad_s", "post_robot_tcp_wy_rad_s", "post_robot_tcp_wz_rad_s"), ("robot_tcp_wx_rad_s", "robot_tcp_wy_rad_s", "robot_tcp_wz_rad_s"), False),
            )
            for post_names, native_names, quaternion in post_native_pairs:
                post_value = _values(row, post_names)
                native_value = _values(last, native_names)
                difference = (
                    _quaternion_distance(post_value, native_value)
                    if quaternion
                    else _distance(post_value, native_value)
                )
                if difference > pose_tolerance:
                    raise SandboxIntegrityError(
                        "post-step pose/velocity differs from last native evidence"
                    )
            if step == 0:
                if row["state_source_kind"] != "bootstrap_sensor_snapshot":
                    raise SandboxIntegrityError("step zero lacks bootstrap state provenance")
                if row["force_source_native_sample_index"] is not None:
                    raise SandboxIntegrityError("initial force cites an unavailable native sample")
                if int(row["force_source_time_ns"]) != int(row["pre_time_ns"]):
                    raise SandboxIntegrityError("bootstrap state time differs from pre-step time")
            else:
                if row["state_source_kind"] != "prior_native_sample":
                    raise SandboxIntegrityError("noninitial state lacks native provenance")
                if previous_last is None:
                    raise SandboxIntegrityError("preceding native sample is missing")
                if int(row["force_source_native_sample_index"]) != int(
                    previous_last["native_sample_index"]
                ):
                    raise SandboxIntegrityError("pre-step native source index is misaligned")
                if int(row["force_source_time_ns"]) != int(previous_last["time_ns"]):
                    raise SandboxIntegrityError("pre-step native source time is misaligned")
                if abs(float(row["pre_force_n"]) - float(previous_last["audit_force_n"])) > 1e-12:
                    raise SandboxIntegrityError("pre-step force differs from cited native force")
                scalar_pairs = (
                    ("normal_velocity_outward_m_s", "normal_velocity_outward_m_s"),
                    ("instantaneous_progress", "instantaneous_progress"),
                    ("geometric_track_error_m", "geometric_track_error_m"),
                    ("recovery_hover_pose_error_m", "recovery_hover_pose_error_m"),
                    ("recovery_clearance_m", "recovery_clearance_m"),
                )
                if any(
                    abs(float(row[pre_name]) - float(previous_last[native_name]))
                    > pose_tolerance
                    for pre_name, native_name in scalar_pairs
                ):
                    raise SandboxIntegrityError(
                        "pre-step scalar state differs from cited native evidence"
                    )
                if bool(row["pre_contact_observed"]) != bool(
                    previous_last["contact_observed"]
                ):
                    raise SandboxIntegrityError(
                        "pre-step contact differs from cited native evidence"
                    )
                pre_native_geometry_pairs = (
                    (("surface_reference_x_m", "surface_reference_y_m", "surface_reference_z_m"), ("surface_reference_x_m", "surface_reference_y_m", "surface_reference_z_m")),
                    (("projection_outward_normal_x", "projection_outward_normal_y", "projection_outward_normal_z"), ("projection_outward_normal_x", "projection_outward_normal_y", "projection_outward_normal_z")),
                    (("command_outward_normal_x", "command_outward_normal_y", "command_outward_normal_z"), ("command_outward_normal_x", "command_outward_normal_y", "command_outward_normal_z")),
                    (("committed_recovery_outward_normal_x", "committed_recovery_outward_normal_y", "committed_recovery_outward_normal_z"), ("committed_recovery_outward_normal_x", "committed_recovery_outward_normal_y", "committed_recovery_outward_normal_z")),
                    (("task_projection_point_x_m", "task_projection_point_y_m", "task_projection_point_z_m"), ("task_projection_point_x_m", "task_projection_point_y_m", "task_projection_point_z_m")),
                    (("projection_tangent_x", "projection_tangent_y", "projection_tangent_z"), ("projection_tangent_x", "projection_tangent_y", "projection_tangent_z")),
                    (("command_tangent_x", "command_tangent_y", "command_tangent_z"), ("command_tangent_x", "command_tangent_y", "command_tangent_z")),
                    (("recovery_hover_target_x_m", "recovery_hover_target_y_m", "recovery_hover_target_z_m"), ("recovery_hover_target_x_m", "recovery_hover_target_y_m", "recovery_hover_target_z_m")),
                )
                for pre_names, native_names in pre_native_geometry_pairs:
                    if _distance(
                        _values(row, pre_names), _values(previous_last, native_names)
                    ) > pose_tolerance:
                        raise SandboxIntegrityError(
                            "pre-step dynamic geometry differs from cited native evidence"
                        )
                for name in (
                    "task_projection_progress",
                    "task_projection_arc_length_m",
                    "task_path_length_m",
                ):
                    if abs(float(row[name]) - float(previous_last[name])) > pose_tolerance:
                        raise SandboxIntegrityError(
                            "pre-step path geometry differs from cited native evidence"
                        )
                pre_native_pairs = (
                    (("pre_contact_tool_x_m", "pre_contact_tool_y_m", "pre_contact_tool_z_m"), ("contact_tool_x_m", "contact_tool_y_m", "contact_tool_z_m"), False),
                    (("pre_contact_tool_qw", "pre_contact_tool_qx", "pre_contact_tool_qy", "pre_contact_tool_qz"), ("contact_tool_qw", "contact_tool_qx", "contact_tool_qy", "contact_tool_qz"), True),
                    (("pre_contact_tool_vx_m_s", "pre_contact_tool_vy_m_s", "pre_contact_tool_vz_m_s"), ("contact_tool_vx_m_s", "contact_tool_vy_m_s", "contact_tool_vz_m_s"), False),
                    (("pre_contact_tool_wx_rad_s", "pre_contact_tool_wy_rad_s", "pre_contact_tool_wz_rad_s"), ("contact_tool_wx_rad_s", "contact_tool_wy_rad_s", "contact_tool_wz_rad_s"), False),
                    (("pre_robot_tcp_x_m", "pre_robot_tcp_y_m", "pre_robot_tcp_z_m"), ("robot_tcp_x_m", "robot_tcp_y_m", "robot_tcp_z_m"), False),
                    (("pre_robot_tcp_qw", "pre_robot_tcp_qx", "pre_robot_tcp_qy", "pre_robot_tcp_qz"), ("robot_tcp_qw", "robot_tcp_qx", "robot_tcp_qy", "robot_tcp_qz"), True),
                    (("pre_robot_tcp_vx_m_s", "pre_robot_tcp_vy_m_s", "pre_robot_tcp_vz_m_s"), ("robot_tcp_vx_m_s", "robot_tcp_vy_m_s", "robot_tcp_vz_m_s"), False),
                    (("pre_robot_tcp_wx_rad_s", "pre_robot_tcp_wy_rad_s", "pre_robot_tcp_wz_rad_s"), ("robot_tcp_wx_rad_s", "robot_tcp_wy_rad_s", "robot_tcp_wz_rad_s"), False),
                )
                for pre_names, native_names, quaternion in pre_native_pairs:
                    pre_value = _values(row, pre_names)
                    native_value = _values(previous_last, native_names)
                    difference = (
                        _quaternion_distance(pre_value, native_value)
                        if quaternion
                        else _distance(pre_value, native_value)
                    )
                    if difference > pose_tolerance:
                        raise SandboxIntegrityError(
                            "pre-step pose/velocity differs from cited native evidence"
                        )
            previous_last = last
            previous_post_time = int(row["post_time_ns"])
            previous_diag = diag

    if len(control_rows) > int(scenario["max_control_steps"]):
        raise SandboxIntegrityError("evaluation exceeds its registered control-step bound")
    if any(abs(float(row["target_force_n"]) - float(scenario["target_force_n"])) > 1e-12 for row in control_rows):
        raise SandboxIntegrityError("control target differs from scenario manifest")

    metrics = episode_rows[0]
    if abs(float(metrics["target_force_n"]) - float(scenario["target_force_n"])) > 1e-12:
        raise SandboxIntegrityError("episode target differs from scenario manifest")
    force_limit = float(definition["force_limit_n"])
    initial_pre_forces = [
        float(row["pre_force_n"])
        for row in control_rows
        if int(row["control_step_index"]) == 0
    ]
    all_observed_forces = [float(row["audit_force_n"]) for row in native_rows] + initial_pre_forces
    peak_force = max(all_observed_forces)
    force_violations = sum(force > force_limit for force in all_observed_forces)
    authority_violations = _authority_violations(
        diagnostic_rows,
        metric_version=str(
            definition.get("authority_metric_version", AUTHORITY_METRIC_LEGACY)
        ),
    )
    readback_violations = _command_readback_violations(
        control_rows, definition=definition
    )
    expected_metrics = {
        "control_steps": len(control_rows),
        "native_samples": len(native_rows),
        "force_limit_violation_count": force_violations,
        "authority_violation_count": authority_violations,
        "command_readback_violation_count": readback_violations,
    }
    for name, expected in expected_metrics.items():
        if int(metrics[name]) != expected:
            raise SandboxIntegrityError(f"episode {name} does not match raw evidence")
    if abs(float(metrics["peak_force_n"]) - peak_force) > 1e-12:
        raise SandboxIntegrityError("episode peak force does not match native evidence")
    if not 0.0 <= float(metrics["final_committed_progress"]) <= 1.0:
        raise SandboxIntegrityError("final committed progress lies outside [0,1]")
    if not str(metrics["terminal_status"]).strip():
        raise SandboxIntegrityError("episode terminal status is empty")
    if float(metrics["wall_time_s"]) < 0.0:
        raise SandboxIntegrityError("episode wall time is negative")
    if metrics["tracking_rmse_n"] is not None and float(metrics["tracking_rmse_n"]) < 0.0:
        raise SandboxIntegrityError("episode tracking RMSE is negative")
    if metrics["band_fraction"] is not None and not 0.0 <= float(
        metrics["band_fraction"]
    ) <= 1.0:
        raise SandboxIntegrityError("episode band fraction lies outside [0,1]")
    final_diagnostic = max(
        diagnostic_rows,
        key=lambda item: (int(item["segment_index"]), int(item["control_step_index"])),
    )
    if abs(
        float(metrics["final_committed_progress"])
        - float(final_diagnostic["committed_progress_after"])
    ) > pose_tolerance:
        raise SandboxIntegrityError(
            "episode final committed progress differs from diagnostic evidence"
        )
    if metrics["final_supervisor_state"] != final_diagnostic["state_after"]:
        raise SandboxIntegrityError(
            "episode final supervisor state differs from diagnostic evidence"
        )
    if metrics["contact_fraction"] is not None:
        observed_contact_fraction = sum(
            bool(row["contact_observed"]) for row in native_rows
        ) / len(native_rows)
        if abs(float(metrics["contact_fraction"]) - observed_contact_fraction) > 1e-12:
            raise SandboxIntegrityError(
                "episode contact fraction differs from native evidence"
            )
    sampled_force_flags = sum(
        bool(row["sampled_force_violation_observed"]) for row in diagnostic_rows
    )
    inconsistent_sampled_force_flags = any(
        bool(row["sampled_force_violation_observed"])
        != (float(row["measured_force_n"]) > force_limit)
        for row in diagnostic_rows
    )
    if inconsistent_sampled_force_flags:
        raise SandboxIntegrityError("diagnostic sampled-force flag is inconsistent")
    scientific_required = bool(
        force_violations
        or sampled_force_flags
        or authority_violations
        or readback_violations
        or any(row["state_after"] == "SAFE_HOLD" for row in diagnostic_rows)
        or not bool(metrics["lifecycle_complete"])
    )
    if scientific_required and not bool(metrics["scientific_failure"]):
        raise SandboxIntegrityError("episode attempts to hide a registered scientific failure")
    if bool(metrics["scientific_failure"]) and not str(
        metrics["scientific_failure_reason"]
    ).strip():
        raise SandboxIntegrityError("scientific failure lacks a reason")
    if not bool(metrics["scientific_failure"]) and str(
        metrics["scientific_failure_reason"]
    ).strip():
        raise SandboxIntegrityError("scientific pass carries a failure reason")
    return bool(metrics["scientific_failure"])


def _default_parquet_writer(table: str, frame: pl.DataFrame, path: Path) -> None:
    del table
    frame.write_parquet(path, compression="zstd", statistics=True)


def _role_for(relative: str) -> str:
    if relative.startswith("quarantine/"):
        return "raw_physical_interval_quarantine"
    if relative.startswith("raw_intervals/"):
        return "raw_physical_interval_journal"
    if relative.startswith("data/native/"):
        return "native"
    if relative.startswith("data/control/"):
        return "control"
    if relative.startswith("data/diagnostic/"):
        return "diagnostic"
    if relative.startswith("data/episode/"):
        return "episode"
    return {
        "RUN_DEFINITION.json": "run_definition",
        "SOURCE_MANIFEST.csv": "source_manifest",
        "SCENARIO_MANIFEST.csv": "scenario_manifest",
        "RUNTIME_SNAPSHOT.json": "runtime_snapshot",
        "PERMISSION_SNAPSHOT.json": "permission_snapshot",
        "CHANGE_RATIONALE.json": "change_rationale",
        "RUN_STATE.json": "run_state",
        "RUN_SUMMARY.json": "run_summary",
        "RUN_STATUS.json": "run_status",
    }.get(relative, "supporting_evidence")


@dataclass(frozen=True)
class VerificationResult:
    run_id: str
    status: str
    files_verified: int
    evaluations_verified: int
    scientific_failures: int


class SandboxRunStore:
    """One non-overwriting, repeatable SANDBOX run transaction."""

    def __init__(
        self,
        *,
        run_id: str,
        root: Path,
        staging_path: Path,
        final_path: Path,
        definition: dict[str, Any],
        scenarios: dict[int, dict[str, Any]],
        writer: ParquetWriter,
    ) -> None:
        self.run_id = run_id
        self.root = root
        self.staging_path = staging_path
        self.final_path = final_path
        self.definition = definition
        self.scenarios = scenarios
        self.writer = writer
        self._committed_evaluations: set[int] = set()
        self._scientific_failures: dict[int, bool] = {}
        self._quarantine_entries = 0
        self._terminal = False

    @classmethod
    def begin(
        cls,
        *,
        permission_path: Path,
        run_id: str,
        run_definition: Mapping[str, Any],
        source_root: Path,
        source_manifest: Sequence[Mapping[str, Any]],
        scenario_manifest: Sequence[Mapping[str, Any]],
        runtime_snapshot: Mapping[str, Any],
        change_rationale: Mapping[str, Any],
        writer: ParquetWriter | None = None,
    ) -> "SandboxRunStore":
        if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
            raise SandboxIntegrityError("run_id is not a safe SANDBOX identifier")
        permission_path = Path(permission_path)
        _reject_symlink_components(permission_path, label="permission_path")
        permission_bytes = permission_path.read_bytes()
        try:
            permission = json.loads(permission_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SandboxPermissionError("permission is not valid UTF-8 JSON") from exc
        if not isinstance(permission, Mapping):
            raise SandboxPermissionError("permission root must be an object")

        # All validation above directory creation is intentional: permission-off and
        # identity failures leave no staging/output artefact.
        root = _validate_permission(permission)
        _validate_run_definition(run_definition, run_id=run_id, permission=permission)
        sources = _validate_source_manifest(source_manifest, source_root=Path(source_root))
        _validate_trusted_source_roles(
            sources, definition=run_definition, permission=permission
        )
        scenarios, scenarios_by_evaluation = _validate_scenario_manifest(
            scenario_manifest, permission=permission
        )
        _canonical_json_bytes(runtime_snapshot)
        _canonical_json_bytes(change_rationale)

        runs_root = root / "runs"
        staging_root = root / "staging"
        final_path = runs_root / run_id
        staging_path = staging_root / f".{run_id}.creating"
        for path, label in (
            (runs_root, "runs root"),
            (staging_root, "staging root"),
            (final_path, "final run path"),
            (staging_path, "staging run path"),
        ):
            _reject_symlink_components(path, label=label)
        if final_path.exists() or staging_path.exists():
            raise SandboxIntegrityError("run_id already exists and cannot be overwritten")

        root.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(root, label="allowed_output_root")
        runs_root.mkdir(exist_ok=True)
        staging_root.mkdir(exist_ok=True)
        _reject_symlink_components(runs_root, label="runs root")
        _reject_symlink_components(staging_root, label="staging root")
        staging_path.mkdir(exist_ok=False)
        (staging_path / "data" / "native").mkdir(parents=True)
        (staging_path / "data" / "control").mkdir(parents=True)
        (staging_path / "data" / "diagnostic").mkdir(parents=True)
        (staging_path / "data" / "episode").mkdir(parents=True)

        instance = cls(
            run_id=run_id,
            root=root,
            staging_path=staging_path,
            final_path=final_path,
            definition=dict(run_definition),
            scenarios=scenarios_by_evaluation,
            writer=_default_parquet_writer if writer is None else writer,
        )
        try:
            _atomic_write_json(
                staging_path / "RUN_STATE.json",
                {
                    "format": "forcewipe_v6_sandbox_run_state_v1",
                    "run_id": run_id,
                    "status": "running",
                    "started_utc": _utc_now(),
                    "committed_evaluations": 0,
                },
            )
            _atomic_write_json(staging_path / "RUN_DEFINITION.json", run_definition)
            _atomic_write_bytes(
                staging_path / "SOURCE_MANIFEST.csv",
                _csv_bytes(sources, SOURCE_MANIFEST_FIELDS),
            )
            _atomic_write_bytes(
                staging_path / "SCENARIO_MANIFEST.csv",
                _csv_bytes(scenarios, SCENARIO_MANIFEST_FIELDS),
            )
            _atomic_write_json(staging_path / "RUNTIME_SNAPSHOT.json", runtime_snapshot)
            _atomic_write_bytes(staging_path / "PERMISSION_SNAPSHOT.json", permission_bytes)
            _atomic_write_json(staging_path / "CHANGE_RATIONALE.json", change_rationale)
        except BaseException:
            # An interrupted preamble intentionally remains uncommitted for audit.
            raise
        return instance

    def _write_parquet(self, table: str, frame: pl.DataFrame, destination: Path) -> None:
        temporary = destination.with_name(f".{destination.name}.{uuid4().hex}.part")
        try:
            self.writer(table, frame, temporary)
            if not temporary.is_file():
                raise SandboxIntegrityError("Parquet writer returned without a file")
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            observed_schema = pl.read_parquet_schema(temporary)
            if observed_schema != SCHEMAS[table]:
                raise SandboxIntegrityError(f"written {table} schema changed on disk")
            os.replace(temporary, destination)
            _fsync_directory(destination.parent)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise

    def append_evaluation(
        self,
        *,
        evaluation_id: int,
        native_rows: Sequence[Mapping[str, Any]],
        control_rows: Sequence[Mapping[str, Any]],
        diagnostic_rows: Sequence[Mapping[str, Any]],
        episode_metrics: Mapping[str, Any],
    ) -> None:
        if self._terminal:
            raise SandboxStorageError("run is already terminal")
        try:
            if evaluation_id not in self.scenarios:
                raise SandboxIntegrityError("evaluation is absent from scenario manifest")
            if evaluation_id in self._committed_evaluations:
                raise SandboxIntegrityError("evaluation was already appended")
            native = _frame(native_rows, NATIVE_SCHEMA, table="native")
            control = _frame(control_rows, CONTROL_SCHEMA, table="control")
            diagnostic = _frame(
                diagnostic_rows, DIAGNOSTIC_SCHEMA, table="diagnostic"
            )
            episode = _frame([episode_metrics], EPISODE_SCHEMA, table="episode")
            scientific_failure = _validate_evaluation(
                run_id=self.run_id,
                scenario=self.scenarios[evaluation_id],
                definition=self.definition,
                native=native,
                control=control,
                diagnostic=diagnostic,
                episode=episode,
            )
            filename = f"evaluation_{evaluation_id}.parquet"
            for table, frame in (
                ("native", native),
                ("control", control),
                ("diagnostic", diagnostic),
                ("episode", episode),
            ):
                self._write_parquet(
                    table, frame, self.staging_path / "data" / table / filename
                )
            self._committed_evaluations.add(evaluation_id)
            self._scientific_failures[evaluation_id] = scientific_failure
            _atomic_write_json(
                self.staging_path / "RUN_STATE.json",
                {
                    "format": "forcewipe_v6_sandbox_run_state_v1",
                    "run_id": self.run_id,
                    "status": "running",
                    "committed_evaluations": len(self._committed_evaluations),
                    "updated_utc": _utc_now(),
                },
            )
        except Exception as exc:
            final_path: Path | None = None
            try:
                final_path = self._abort_internal(
                    f"{type(exc).__name__}: {exc}", stage="append_evaluation"
                )
            except Exception:
                # If even the abort closure cannot be written, staging deliberately
                # remains uncommitted and cannot be mistaken for evidence.
                pass
            raise SandboxInfrastructureAbort(str(exc), final_path=final_path) from exc

    def record_quarantine_interval(
        self,
        *,
        evaluation_id: int,
        control_step_index: int,
        stage: str,
        raw_interval: Mapping[str, Any],
    ) -> Path:
        """Persist an untrusted physical interval before infrastructure abort.

        Quarantine entries are JSON rather than Parquet so a schema/serializer
        failure cannot erase the raw diagnostic packet that caused it.  Their
        presence prevents a normal/scientific completion and they are included
        in the infrastructure-abort file manifest.
        """

        if self._terminal:
            raise SandboxStorageError("run is already terminal")
        if evaluation_id not in self.scenarios:
            raise SandboxIntegrityError("quarantine evaluation is not registered")
        _checked_int(control_step_index, label="quarantine control_step_index")
        if (
            not isinstance(stage, str)
            or re.fullmatch(r"[A-Za-z0-9_.-]+", stage) is None
        ):
            raise SandboxIntegrityError("quarantine stage is not a safe identifier")
        _canonical_json_bytes(raw_interval)
        self._quarantine_entries += 1
        directory = self.staging_path / "quarantine"
        directory.mkdir(exist_ok=True)
        destination = directory / f"interval_{self._quarantine_entries:06d}.json"
        _atomic_write_json(
            destination,
            {
                "format": "forcewipe_v6_raw_interval_quarantine_v1",
                "run_id": self.run_id,
                "scenario_id": self.scenarios[evaluation_id]["scenario_id"],
                "evaluation_id": evaluation_id,
                "control_step_index": control_step_index,
                "stage": stage,
                "captured_utc": _utc_now(),
                "raw_interval": raw_interval,
            },
        )
        return destination

    def complete(self, *, summary: Mapping[str, Any] | None = None) -> Path:
        if self._terminal:
            raise SandboxStorageError("run is already terminal")
        if self._quarantine_entries:
            final_path = self._abort_internal(
                "raw physical intervals are quarantined",
                stage="complete",
            )
            raise SandboxInfrastructureAbort(
                "quarantine evidence forbids scientific completion",
                final_path=final_path,
            )
        missing = sorted(set(self.scenarios) - self._committed_evaluations)
        if missing:
            final_path = self._abort_internal(
                f"missing registered evaluations: {missing}", stage="complete"
            )
            raise SandboxInfrastructureAbort(
                "not all registered evaluations were appended", final_path=final_path
            )
        scientific_failures = sum(self._scientific_failures.values())
        status = STATUS_SCIENTIFIC_FAIL if scientific_failures else STATUS_DIAGNOSTIC
        user_summary = {} if summary is None else dict(summary)
        reserved = {
            "format",
            "run_id",
            "status",
            "evaluations",
            "scientific_failures",
            "completed_utc",
        }
        if reserved.intersection(user_summary):
            raise SandboxIntegrityError("summary attempts to replace computed identity")
        payload = {
            "format": "forcewipe_v6_sandbox_summary_v1",
            "run_id": self.run_id,
            "status": status,
            "evaluations": len(self._committed_evaluations),
            "scientific_failures": scientific_failures,
            "completed_utc": _utc_now(),
            **user_summary,
        }
        _atomic_write_json(self.staging_path / "RUN_SUMMARY.json", payload)
        _atomic_write_json(
            self.staging_path / "RUN_STATUS.json",
            {
                "format": "forcewipe_v6_sandbox_status_v1",
                "run_id": self.run_id,
                "status": status,
                "claim_boundary": (
                    "Repeatable development SANDBOX only; no qualification, CAL, "
                    "TRAIN, TEST, safety, tracking, or liveness claim."
                ),
            },
        )
        _atomic_write_json(
            self.staging_path / "RUN_STATE.json",
            {
                "format": "forcewipe_v6_sandbox_run_state_v1",
                "run_id": self.run_id,
                "status": status,
                "committed_evaluations": len(self._committed_evaluations),
                "completed_utc": _utc_now(),
            },
        )
        return self._finalize(status=status)

    def abort_infrastructure(self, *, reason: str, stage: str) -> Path:
        if self._terminal:
            raise SandboxStorageError("run is already terminal")
        if not isinstance(reason, str) or not reason.strip():
            raise SandboxIntegrityError("infrastructure abort reason must be nonempty")
        if not isinstance(stage, str) or not stage.strip():
            raise SandboxIntegrityError("infrastructure abort stage must be nonempty")
        return self._abort_internal(reason, stage=stage)

    def _abort_internal(self, reason: str, *, stage: str) -> Path:
        for path in self.staging_path.rglob("*.part"):
            path.unlink(missing_ok=True)
        payload = {
            "format": "forcewipe_v6_sandbox_summary_v1",
            "run_id": self.run_id,
            "status": STATUS_INFRASTRUCTURE_ABORT,
            "evaluations": len(self._committed_evaluations),
            "scientific_failures": sum(self._scientific_failures.values()),
            "quarantine_entries": self._quarantine_entries,
            "abort_stage": stage,
            "abort_reason": reason,
            "completed_utc": _utc_now(),
        }
        _atomic_write_json(self.staging_path / "RUN_SUMMARY.json", payload)
        _atomic_write_json(
            self.staging_path / "RUN_STATUS.json",
            {
                "format": "forcewipe_v6_sandbox_status_v1",
                "run_id": self.run_id,
                "status": STATUS_INFRASTRUCTURE_ABORT,
                "claim_boundary": "Infrastructure abort; no scientific result.",
            },
        )
        _atomic_write_json(
            self.staging_path / "RUN_STATE.json",
            {
                "format": "forcewipe_v6_sandbox_run_state_v1",
                "run_id": self.run_id,
                "status": STATUS_INFRASTRUCTURE_ABORT,
                "committed_evaluations": len(self._committed_evaluations),
                "completed_utc": _utc_now(),
            },
        )
        return self._finalize(status=STATUS_INFRASTRUCTURE_ABORT)

    def _finalize(self, *, status: str) -> Path:
        if status not in TERMINAL_STATUSES:
            raise SandboxIntegrityError("unsupported terminal status")
        if self.final_path.exists():
            raise SandboxIntegrityError("final run path appeared during execution")
        manifest_rows: list[dict[str, Any]] = []
        for path in sorted(self.staging_path.rglob("*")):
            if not path.is_file() or path.name in {"FILE_MANIFEST.csv", "COMMIT.json"}:
                continue
            if path.name.endswith(".part") or ".part." in path.name:
                continue
            relative = path.relative_to(self.staging_path).as_posix()
            role = _role_for(relative)
            row_count: int | str = ""
            fingerprint = ""
            if path.suffix == ".parquet":
                if role not in SCHEMAS:
                    raise SandboxIntegrityError("unregistered Parquet role in staging")
                observed_schema = pl.read_parquet_schema(path)
                if observed_schema != SCHEMAS[role]:
                    raise SandboxIntegrityError("Parquet schema drift before commit")
                row_count = pl.scan_parquet(path).select(pl.len()).collect().item()
                fingerprint = schema_sha256(role)
            manifest_rows.append(
                {
                    "relative_path": relative,
                    "role": role,
                    "size_bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                    "row_count": row_count,
                    "schema_sha256": fingerprint,
                }
            )
        manifest_bytes = _csv_bytes(manifest_rows, FILE_MANIFEST_FIELDS)
        _atomic_write_bytes(self.staging_path / "FILE_MANIFEST.csv", manifest_bytes)
        commit = {
            "format": COMMIT_FORMAT,
            "run_id": self.run_id,
            "status": status,
            "file_manifest_sha256": _sha256_bytes(manifest_bytes),
            "manifest_entries": len(manifest_rows),
            "committed_utc": _utc_now(),
        }
        # COMMIT is deliberately the final file written inside staging.
        _atomic_write_json(self.staging_path / "COMMIT.json", commit)
        _fsync_directory(self.staging_path)
        os.replace(self.staging_path, self.final_path)
        _fsync_directory(self.final_path.parent)
        self._terminal = True
        return self.final_path


def _parse_scenario_csv(rows: Sequence[Mapping[str, str]]) -> dict[int, dict[str, Any]]:
    parsed: dict[int, dict[str, Any]] = {}
    for row in rows:
        if set(row) != set(SCENARIO_MANIFEST_FIELDS):
            raise SandboxIntegrityError("committed scenario manifest fields changed")
        value = {
            "scenario_id": int(row["scenario_id"]),
            "evaluation_id": int(row["evaluation_id"]),
            "family": row["family"],
            "target_force_n": float(row["target_force_n"]),
            "seed": int(row["seed"]),
            "max_control_steps": int(row["max_control_steps"]),
            "expected_trigger": row["expected_trigger"],
            "parameters_json": row["parameters_json"],
        }
        if value["evaluation_id"] in parsed:
            raise SandboxIntegrityError("committed scenario manifest has duplicates")
        parsed[value["evaluation_id"]] = value
    return parsed


def _validate_committed_source_manifest(rows: Sequence[Mapping[str, str]]) -> None:
    if not rows:
        raise SandboxIntegrityError("committed source manifest is empty")
    seen: set[str] = set()
    for row in rows:
        if set(row) != set(SOURCE_MANIFEST_FIELDS):
            raise SandboxIntegrityError("committed source manifest fields changed")
        relative = _safe_relative_path(
            row["relative_path"], label="committed source relative_path"
        ).as_posix()
        if relative in seen:
            raise SandboxIntegrityError("committed source manifest has duplicate paths")
        seen.add(relative)
        try:
            size = int(row["size_bytes"])
        except ValueError as exc:
            raise SandboxIntegrityError("committed source size is not an integer") from exc
        _checked_int(size, label="committed source size_bytes")
        _checked_sha(row["sha256"], label="committed source sha256")
        if not row["role"]:
            raise SandboxIntegrityError("committed source role is empty")


def verify_committed_run(path: Path) -> VerificationResult:
    """Verify hashes, schemas, identities, and causal closure of a committed run."""

    path = Path(path)
    _reject_symlink_components(path, label="committed run path")
    if not path.is_dir():
        raise SandboxUncommittedError("committed run directory is absent")
    for item in path.rglob("*"):
        if item.is_symlink():
            raise SandboxIntegrityError("committed run contains a symbolic link")
    commit_path = path / "COMMIT.json"
    manifest_path = path / "FILE_MANIFEST.csv"
    if not commit_path.is_file() or not manifest_path.is_file():
        raise SandboxUncommittedError("last-write COMMIT or file manifest is absent")
    try:
        commit = json.loads(commit_path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SandboxUncommittedError("COMMIT is malformed") from exc
    if commit.get("format") != COMMIT_FORMAT:
        raise SandboxUncommittedError("COMMIT format is invalid")
    run_id = commit.get("run_id")
    status = commit.get("status")
    if not isinstance(run_id, str) or _RUN_ID.fullmatch(run_id) is None:
        raise SandboxIntegrityError("committed run_id is invalid")
    if path.name != run_id:
        raise SandboxIntegrityError("directory and committed run identities differ")
    if status not in TERMINAL_STATUSES:
        raise SandboxIntegrityError("committed status is invalid")
    manifest_bytes = manifest_path.read_bytes()
    if _sha256_bytes(manifest_bytes) != commit.get("file_manifest_sha256"):
        raise SandboxIntegrityError("file manifest hash mismatch")
    manifest = _read_csv(manifest_path)
    if len(manifest) != int(commit.get("manifest_entries", -1)):
        raise SandboxIntegrityError("file manifest entry count mismatch")
    if any(set(row) != set(FILE_MANIFEST_FIELDS) for row in manifest):
        raise SandboxIntegrityError("file manifest columns changed")
    relative_paths = [row["relative_path"] for row in manifest]
    if len(set(relative_paths)) != len(relative_paths):
        raise SandboxIntegrityError("file manifest contains duplicate paths")
    actual_files = {
        item.relative_to(path).as_posix()
        for item in path.rglob("*")
        if item.is_file() and item.name not in {"FILE_MANIFEST.csv", "COMMIT.json"}
    }
    if actual_files != set(relative_paths):
        raise SandboxIntegrityError("committed file set differs from manifest")
    for row in manifest:
        relative = _safe_relative_path(row["relative_path"], label="result path")
        file_path = path.joinpath(*relative.parts)
        _reject_symlink_components(file_path, label="result file")
        if file_path.stat().st_size != int(row["size_bytes"]):
            raise SandboxIntegrityError("committed file size mismatch")
        if _sha256_file(file_path) != row["sha256"]:
            raise SandboxIntegrityError("committed file hash mismatch")
        role = row["role"]
        if file_path.suffix == ".parquet":
            if role not in SCHEMAS:
                raise SandboxIntegrityError("committed Parquet role is unknown")
            if row["schema_sha256"] != schema_sha256(role):
                raise SandboxIntegrityError("committed schema fingerprint mismatch")
            if pl.read_parquet_schema(file_path) != SCHEMAS[role]:
                raise SandboxIntegrityError("committed Parquet schema mismatch")
            if pl.scan_parquet(file_path).select(pl.len()).collect().item() != int(
                row["row_count"]
            ):
                raise SandboxIntegrityError("committed Parquet row count mismatch")

    required = {
        "RUN_DEFINITION.json",
        "SOURCE_MANIFEST.csv",
        "SCENARIO_MANIFEST.csv",
        "RUNTIME_SNAPSHOT.json",
        "PERMISSION_SNAPSHOT.json",
        "CHANGE_RATIONALE.json",
        "RUN_STATE.json",
        "RUN_SUMMARY.json",
        "RUN_STATUS.json",
    }
    if not required.issubset(actual_files):
        raise SandboxIntegrityError("committed metadata closure is incomplete")
    definition = json.loads((path / "RUN_DEFINITION.json").read_bytes())
    summary = json.loads((path / "RUN_SUMMARY.json").read_bytes())
    run_status = json.loads((path / "RUN_STATUS.json").read_bytes())
    run_state = json.loads((path / "RUN_STATE.json").read_bytes())
    permission = json.loads((path / "PERMISSION_SNAPSHOT.json").read_bytes())
    runtime_snapshot = json.loads((path / "RUNTIME_SNAPSHOT.json").read_bytes())
    change_rationale = json.loads((path / "CHANGE_RATIONALE.json").read_bytes())
    del runtime_snapshot, change_rationale
    if any(item.get("run_id") != run_id for item in (definition, summary, run_status, run_state)):
        raise SandboxIntegrityError("committed metadata run identities differ")
    if any(item.get("status") != status for item in (summary, run_status, run_state)):
        raise SandboxIntegrityError("committed terminal statuses differ")
    permission_root = _validate_permission(permission)
    if permission_root.resolve() != path.parent.parent.resolve():
        raise SandboxIntegrityError("permission output root differs from committed location")
    _validate_run_definition(definition, run_id=run_id, permission=permission)
    committed_sources = _read_csv(path / "SOURCE_MANIFEST.csv")
    _validate_committed_source_manifest(committed_sources)
    _validate_trusted_source_roles(
        committed_sources, definition=definition, permission=permission
    )
    parsed_scenarios = _parse_scenario_csv(_read_csv(path / "SCENARIO_MANIFEST.csv"))
    normalized_scenarios, scenarios = _validate_scenario_manifest(
        list(parsed_scenarios.values()), permission=permission
    )
    del normalized_scenarios

    evaluations_verified = 0
    scientific_failures = 0
    if status != STATUS_INFRASTRUCTURE_ABORT:
        for evaluation_id, scenario in scenarios.items():
            filename = f"evaluation_{evaluation_id}.parquet"
            table_paths = {
                name: path / "data" / name / filename for name in SCHEMAS
            }
            if not all(item.is_file() for item in table_paths.values()):
                raise SandboxIntegrityError("committed evaluation datasets are incomplete")
            frames = {name: pl.read_parquet(item) for name, item in table_paths.items()}
            scientific_failures += int(
                _validate_evaluation(
                    run_id=run_id,
                    scenario=scenario,
                    definition=definition,
                    native=frames["native"],
                    control=frames["control"],
                    diagnostic=frames["diagnostic"],
                    episode=frames["episode"],
                )
            )
            evaluations_verified += 1
        expected_status = (
            STATUS_SCIENTIFIC_FAIL if scientific_failures else STATUS_DIAGNOSTIC
        )
        if expected_status != status:
            raise SandboxIntegrityError("terminal status hides its episode outcomes")
        if int(summary.get("evaluations", -1)) != evaluations_verified:
            raise SandboxIntegrityError("summary evaluation count mismatch")
        if int(summary.get("scientific_failures", -1)) != scientific_failures:
            raise SandboxIntegrityError("summary scientific-failure count mismatch")
    return VerificationResult(
        run_id=run_id,
        status=status,
        files_verified=len(manifest),
        evaluations_verified=evaluations_verified,
        scientific_failures=scientific_failures,
    )


__all__ = [
    "AUTHORITY_METRIC_SPLIT_V3",
    "AUTHORITY_METRIC_PATH_FRAME_V4",
    "AUTHORITY_METRIC_SYSTEM_R4",
    "AUTHORITY_METRIC_LEGACY",
    "AUTHORITY_METRIC_STATE_AWARE",
    "COMMIT_FORMAT",
    "CONTROL_SCHEMA",
    "DIAGNOSTIC_SCHEMA",
    "EPISODE_SCHEMA",
    "NATIVE_SCHEMA",
    "PERMISSION_FORMAT",
    "PERMISSION_SCOPE",
    "PERMISSION_SCOPE_S3",
    "PERMISSION_SCOPE_S5",
    "PERMISSION_SCOPE_SYSTEM_R3_REV2",
    "PERMISSION_SCOPE_SYSTEM_R4",
    "RUN_DEFINITION_FORMAT",
    "RUN_DEFINITION_FORMAT_LEGACY",
    "RUN_DEFINITION_FORMAT_S3",
    "RUN_DEFINITION_FORMAT_S3_R2",
    "RUN_DEFINITION_FORMAT_S5",
    "RUN_DEFINITION_FORMAT_SYSTEM_R3_REV2",
    "RUN_DEFINITION_FORMAT_SYSTEM_R4",
    "SANDBOX_OUTPUT_ROOT",
    "SCHEMAS",
    "STATUS_DIAGNOSTIC",
    "STATUS_INFRASTRUCTURE_ABORT",
    "STATUS_SCIENTIFIC_FAIL",
    "SandboxInfrastructureAbort",
    "SandboxIntegrityError",
    "SandboxPermissionError",
    "SandboxRunStore",
    "SandboxStorageError",
    "SandboxUncommittedError",
    "VerificationResult",
    "schema_sha256",
    "verify_committed_run",
]
