"""Scientific force-stream schemas and adapters for ForceWipe V4."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from .native_bridge import NativeStepRecord
from .sapien_adapter import ControlStepForceAudit
from .streaming import DATA_CONTRACT_VERSION, SchemaContract, StreamingDataError


def _column(
    name: str,
    dtype: str,
    *,
    nullable: bool = False,
    finite: bool = False,
    unit: str | None = None,
    source: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "dtype": dtype,
        "nullable": bool(nullable),
        "finite": bool(finite),
        "source": source,
    }
    if unit is not None:
        payload["unit"] = unit
    return payload


def _contract(
    *,
    table_name: str,
    columns: list[dict[str, Any]],
    primary_key: list[str],
    step_column: str,
    maximum_rows_per_evaluation: int,
) -> SchemaContract:
    maximum = int(maximum_rows_per_evaluation)
    if maximum <= 0:
        raise StreamingDataError("force-stream row maximum must be positive")
    return SchemaContract.from_dict(
        {
            "data_contract_version": DATA_CONTRACT_VERSION,
            "table_name": table_name,
            "table_version": "1.0.0-draft",
            "columns": columns,
            "primary_key": primary_key,
            "evaluation_id_column": "evaluation_id",
            "step_column": step_column,
            "target_rows": 131_072,
            "row_group_size": 65_536,
            "compression_level": 3,
            "data_page_size": 1_048_576,
            "minimum_rows_per_evaluation": 1,
            "maximum_rows_per_evaluation": maximum,
            "force_semantics": {
                "f_aud_n": "post-PhysX-substep safety audit",
                "f_ctrl_hold_n": "pre-control causal measurement held across the interval",
                "f_trk_n": "post-control info.normal_force tracking sample",
                "no_backfill": True,
                "equality_required": False,
            },
        }
    )


def native_trace_contract(maximum_rows_per_evaluation: int) -> SchemaContract:
    columns = [
        _column("evaluation_id", "UInt64", source="frozen_evaluation_manifest"),
        _column("native_sample_index", "UInt32", source="sapien_native_hook"),
        _column("control_step_index", "UInt32", source="sapien_native_hook"),
        _column("substep_index", "UInt8", source="sapien_native_hook"),
        _column("time_ns", "UInt64", unit="ns", source="native_clock"),
        _column("primitive_index", "Int16", source="physical_executor"),
        _column("phase_code", "UInt8", source="physical_executor"),
        _column("f_aud_n", "Float64", finite=True, unit="N", source="raw._normal_force_after_physx"),
        _column("f_ctrl_hold_n", "Float64", nullable=True, finite=True, unit="N", source="pre_control_measurement"),
        _column("f_ctrl_source_native_sample_index", "UInt32", nullable=True, source="pre_control_measurement"),
        _column("f_ctrl_measurement_time_ns", "UInt64", nullable=True, unit="ns", source="pre_control_measurement"),
        _column("f_ctrl_age_samples", "UInt32", nullable=True, source="causal_hold_audit"),
        _column("f_ctrl_age_ns", "UInt64", nullable=True, unit="ns", source="causal_hold_audit"),
        _column("f_ctrl_is_held", "Boolean", nullable=True, source="causal_hold_audit"),
        _column("tcp_x_m", "Float64", finite=True, unit="m", source="agent.tcp.pose"),
        _column("tcp_y_m", "Float64", finite=True, unit="m", source="agent.tcp.pose"),
        _column("tcp_z_m", "Float64", finite=True, unit="m", source="agent.tcp.pose"),
        _column("tcp_vx_m_s", "Float64", finite=True, unit="m/s", source="native_finite_difference"),
        _column("tcp_vy_m_s", "Float64", finite=True, unit="m/s", source="native_finite_difference"),
        _column("tcp_vz_m_s", "Float64", finite=True, unit="m/s", source="native_finite_difference"),
        _column("path_progress", "Float64", finite=True, source="scenario_path_projection"),
        _column("path_distance_m", "Float64", finite=True, unit="m", source="scenario_path_projection"),
        _column("tangential_speed_m_s", "Float64", finite=True, unit="m/s", source="scenario_path_projection"),
        _column("contact", "Boolean", source="f_aud_threshold"),
        _column("near_limit", "Boolean", source="f_aud_threshold"),
        _column("force_violation", "Boolean", source="f_aud_threshold"),
        _column("residual_force_quality", "Float64", finite=True, source="causal_truth_update"),
        _column("residual_traversal", "Float64", finite=True, source="causal_truth_update"),
        _column("residual_removed_mass", "Float64", finite=True, source="causal_truth_update"),
        _column("residual_mass_before", "Float64", finite=True, source="causal_truth_update"),
        _column("residual_mass_after", "Float64", finite=True, source="causal_truth_update"),
        _column("state_finite", "Boolean", source="adapter_validation"),
    ]
    return _contract(
        table_name="native_trace_v1",
        columns=columns,
        primary_key=["evaluation_id", "native_sample_index"],
        step_column="native_sample_index",
        maximum_rows_per_evaluation=maximum_rows_per_evaluation,
    )


def control_trace_contract(maximum_rows_per_evaluation: int) -> SchemaContract:
    columns = [
        _column("evaluation_id", "UInt64", source="frozen_evaluation_manifest"),
        _column("control_step_index", "UInt32", source="sapien_adapter"),
        _column("time_start_ns", "UInt64", unit="ns", source="pre_control_measurement"),
        _column("time_end_ns", "UInt64", unit="ns", source="native_clock"),
        _column("first_native_sample_index", "UInt32", source="sapien_adapter"),
        _column("last_native_sample_index", "UInt32", source="sapien_adapter"),
        _column("native_sample_count", "UInt8", source="sapien_adapter"),
        _column("f_ctrl_input_n", "Float64", finite=True, unit="N", source="pre_control_measurement"),
        _column("f_ctrl_source_native_sample_index", "UInt32", nullable=True, source="pre_control_measurement"),
        _column("f_ctrl_input_minus_previous_f_trk_n", "Float64", nullable=True, finite=True, unit="N", source="force_stream_continuity_audit"),
        _column("f_trk_n", "Float64", finite=True, unit="N", source="post_control_info_normal_force"),
        _column("last_f_aud_n", "Float64", finite=True, unit="N", source="native_trace"),
        _column("f_trk_minus_last_f_aud_n", "Float64", finite=True, unit="N", source="force_stream_comparison"),
        _column("force_target_n", "Float64", finite=True, unit="N", source="scenario_manifest"),
        _column("tcp_x_m", "Float64", finite=True, unit="m", source="last_native_sample"),
        _column("tcp_y_m", "Float64", finite=True, unit="m", source="last_native_sample"),
        _column("tcp_z_m", "Float64", finite=True, unit="m", source="last_native_sample"),
        _column("tcp_vx_m_s", "Float64", finite=True, unit="m/s", source="last_native_sample"),
        _column("tcp_vy_m_s", "Float64", finite=True, unit="m/s", source="last_native_sample"),
        _column("tcp_vz_m_s", "Float64", finite=True, unit="m/s", source="last_native_sample"),
        _column("path_progress", "Float64", finite=True, source="last_native_sample"),
        _column("path_distance_m", "Float64", finite=True, unit="m", source="last_native_sample"),
        _column("contact", "Boolean", source="last_native_sample"),
        _column("near_limit", "Boolean", source="last_native_sample"),
        _column("force_violation", "Boolean", source="interval_native_or"),
        _column("state_finite", "Boolean", source="adapter_validation"),
    ]
    return _contract(
        table_name="control_trace_v1",
        columns=columns,
        primary_key=["evaluation_id", "control_step_index"],
        step_column="control_step_index",
        maximum_rows_per_evaluation=maximum_rows_per_evaluation,
    )


def force_stream_contracts(
    *, maximum_native_rows: int, maximum_control_rows: int
) -> dict[str, SchemaContract]:
    return {
        "native_trace_v1": native_trace_contract(maximum_native_rows),
        "control_trace_v1": control_trace_contract(maximum_control_rows),
    }


def native_trace_frame(
    evaluation_id: int,
    records: Sequence[NativeStepRecord],
    *,
    contract: SchemaContract,
    primitive_indices: Sequence[int] | None = None,
    phase_codes: Sequence[int] | None = None,
) -> pl.DataFrame:
    if contract.table_name != "native_trace_v1":
        raise StreamingDataError("native frame received the wrong schema contract")
    count = len(records)
    primitive = [-1] * count if primitive_indices is None else list(primitive_indices)
    phases = [0] * count if phase_codes is None else list(phase_codes)
    if len(primitive) != count or len(phases) != count:
        raise StreamingDataError("native metadata length differs from record count")
    rows: list[dict[str, Any]] = []
    for record, primitive_index, phase_code in zip(records, primitive, phases):
        if record.control_step_index is None or record.substep_index is None:
            raise StreamingDataError("native record lacks control/substep provenance")
        update = record.residual_update
        px, py, pz = record.tcp_position_xyz
        vx, vy, vz = record.tcp_velocity_xyz
        rows.append(
            {
                "evaluation_id": int(evaluation_id),
                "native_sample_index": int(record.sample_index),
                "control_step_index": int(record.control_step_index),
                "substep_index": int(record.substep_index),
                "time_ns": int(record.time_ns),
                "primitive_index": int(primitive_index),
                "phase_code": int(phase_code),
                "f_aud_n": float(record.audit_force_n),
                "f_ctrl_hold_n": record.tracking_force_n,
                "f_ctrl_source_native_sample_index": record.tracking_force_measurement_sample_index,
                "f_ctrl_measurement_time_ns": record.tracking_force_measurement_time_ns,
                "f_ctrl_age_samples": record.tracking_force_age_samples,
                "f_ctrl_age_ns": record.tracking_force_age_ns,
                "f_ctrl_is_held": record.tracking_force_is_held,
                "tcp_x_m": px,
                "tcp_y_m": py,
                "tcp_z_m": pz,
                "tcp_vx_m_s": vx,
                "tcp_vy_m_s": vy,
                "tcp_vz_m_s": vz,
                "path_progress": float(record.progress),
                "path_distance_m": float(record.path_distance_m),
                "tangential_speed_m_s": float(record.tangential_speed_m_s),
                "contact": bool(record.contact),
                "near_limit": bool(record.near_limit),
                "force_violation": bool(record.force_violation),
                "residual_force_quality": float(update.force_quality),
                "residual_traversal": float(update.traversal),
                "residual_removed_mass": float(update.removed_mass),
                "residual_mass_before": float(update.mass_before),
                "residual_mass_after": float(update.mass_after),
                "state_finite": True,
            }
        )
    frame = pl.DataFrame(rows, schema=contract.schema)
    contract.validate_frame(frame, int(evaluation_id))
    return frame


def control_trace_frame(
    evaluation_id: int,
    audits: Sequence[ControlStepForceAudit],
    records: Sequence[NativeStepRecord],
    *,
    force_target_n: float,
    contract: SchemaContract,
) -> pl.DataFrame:
    if contract.table_name != "control_trace_v1":
        raise StreamingDataError("control frame received the wrong schema contract")
    by_control: dict[int, list[NativeStepRecord]] = {}
    for record in records:
        if record.control_step_index is None or record.substep_index is None:
            raise StreamingDataError("native record lacks control/substep provenance")
        by_control.setdefault(int(record.control_step_index), []).append(record)
    rows: list[dict[str, Any]] = []
    previous_end: float | None = None
    for audit in audits:
        members = by_control.get(int(audit.control_step_index), [])
        substeps = [int(record.substep_index) for record in members]
        if (
            len(members) != audit.native_sample_count
            or substeps != list(range(audit.native_sample_count))
            or not members
            or members[0].sample_index != audit.first_native_sample_index
            or members[-1].sample_index != audit.last_native_sample_index
        ):
            raise StreamingDataError("control/native provenance is inconsistent")
        last = members[-1]
        px, py, pz = last.tcp_position_xyz
        vx, vy, vz = last.tcp_velocity_xyz
        rows.append(
            {
                "evaluation_id": int(evaluation_id),
                "control_step_index": int(audit.control_step_index),
                "time_start_ns": int(audit.tracking_input_time_ns),
                "time_end_ns": int(last.time_ns),
                "first_native_sample_index": int(audit.first_native_sample_index),
                "last_native_sample_index": int(audit.last_native_sample_index),
                "native_sample_count": int(audit.native_sample_count),
                "f_ctrl_input_n": float(audit.tracking_input_force_n),
                "f_ctrl_source_native_sample_index": audit.tracking_input_source_native_sample_index,
                "f_ctrl_input_minus_previous_f_trk_n": (
                    None
                    if previous_end is None
                    else float(audit.tracking_input_force_n - previous_end)
                ),
                "f_trk_n": float(audit.end_tracking_force_n),
                "last_f_aud_n": float(audit.last_audit_force_n),
                "f_trk_minus_last_f_aud_n": float(
                    audit.end_tracking_minus_last_audit_n
                ),
                "force_target_n": float(force_target_n),
                "tcp_x_m": px,
                "tcp_y_m": py,
                "tcp_z_m": pz,
                "tcp_vx_m_s": vx,
                "tcp_vy_m_s": vy,
                "tcp_vz_m_s": vz,
                "path_progress": float(last.progress),
                "path_distance_m": float(last.path_distance_m),
                "contact": bool(last.contact),
                "near_limit": bool(last.near_limit),
                "force_violation": any(record.force_violation for record in members),
                "state_finite": True,
            }
        )
        previous_end = float(audit.end_tracking_force_n)
    frame = pl.DataFrame(rows, schema=contract.schema)
    contract.validate_frame(frame, int(evaluation_id))
    return frame
