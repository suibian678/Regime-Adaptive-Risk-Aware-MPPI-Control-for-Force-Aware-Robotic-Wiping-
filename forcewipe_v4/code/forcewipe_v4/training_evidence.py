"""Episode evidence and canonical Parquet schemas for V4 policy training."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import math
from typing import Any, Mapping

import polars as pl

from .streaming import DATA_CONTRACT_VERSION, SchemaContract, StreamingDataError


class ScientificFailureCode(IntEnum):
    NONE = 0
    FIRST_PASS_TRACKING_TIMEOUT = 1
    FIRST_PASS_FORCE_VIOLATION = 2
    FIRST_PASS_ENVIRONMENT_TERMINATION = 3
    PRIMITIVE_TIMEOUT = 4
    FORCE_LIMIT_VIOLATION = 5
    SPATIAL_CONSTRAINT_VIOLATION = 6
    PLANT_TERMINATION = 7
    STOP_RETURN_TIMEOUT = 8


class PhysicalPhaseCode(IntEnum):
    FIRST_PASS = 0
    PRIMITIVE = 1
    STOP_RETURN = 2


class ScientificEpisodeFailure(RuntimeError):
    """A recorded task outcome that must not be labeled infrastructure failure."""

    def __init__(self, code: ScientificFailureCode, message: str):
        super().__init__(message)
        self.code = ScientificFailureCode(code)


@dataclass(frozen=True)
class TrainingEpisodeEvidence:
    evaluation_id: int
    native_rows: tuple[Mapping[str, Any], ...]
    control_rows: tuple[Mapping[str, Any], ...]
    primitive_rows: tuple[Mapping[str, Any], ...]
    episode_row: Mapping[str, Any]

    def validate(self) -> None:
        if int(self.evaluation_id) < 0:
            raise StreamingDataError("training evidence has a negative evaluation ID")
        collections = (self.native_rows, self.control_rows, self.primitive_rows)
        for rows in collections:
            if any(int(row["evaluation_id"]) != self.evaluation_id for row in rows):
                raise StreamingDataError("training evidence mixes evaluation IDs")
        if int(self.episode_row["evaluation_id"]) != self.evaluation_id:
            raise StreamingDataError("episode summary has the wrong evaluation ID")
        if not self.native_rows or not self.control_rows:
            raise StreamingDataError("training evidence lacks physical trace rows")


def _column(name: str, dtype: str, *, nullable: bool = False, finite: bool = False):
    return {"name": name, "dtype": dtype, "nullable": nullable, "finite": finite}


def _contract(
    table_name: str,
    columns: list[dict[str, Any]],
    primary_key: list[str],
    *,
    step_column: str | None,
    target_rows: int,
    maximum_rows: int,
    minimum_rows: int = 1,
) -> SchemaContract:
    return SchemaContract.from_dict(
        {
            "data_contract_version": DATA_CONTRACT_VERSION,
            "table_name": table_name,
            "table_version": "1.0.0",
            "columns": columns,
            "primary_key": primary_key,
            "evaluation_id_column": "evaluation_id",
            "step_column": step_column,
            "target_rows": target_rows,
            "row_group_size": min(target_rows, 65_536),
            "compression_level": 3 if "trace" in table_name else 5,
            "data_page_size": 1_048_576,
            "minimum_rows_per_evaluation": minimum_rows,
            "maximum_rows_per_evaluation": maximum_rows,
        }
    )


def training_evidence_contracts(
    *, maximum_native_rows: int = 12_000, maximum_control_rows: int = 12_000,
    maximum_primitive_rows: int = 4,
) -> dict[str, SchemaContract]:
    native = _contract(
        "native_trace_v1",
        [
            _column("evaluation_id", "UInt64"),
            _column("step", "UInt32"),
            _column("time_ns", "UInt64"),
            _column("control_step", "UInt32"),
            _column("substep", "UInt16"),
            _column("phase_code", "UInt8"),
            _column("primitive_index", "Int16", nullable=True),
            _column("f_aud_n", "Float64", finite=True),
            _column("f_ctrl_hold_n", "Float64", nullable=True, finite=True),
            _column("f_ctrl_source_sample", "Int64", nullable=True),
            _column("f_ctrl_age_samples", "Int32", nullable=True),
            _column("progress", "Float64", finite=True),
            _column("tangential_speed_m_s", "Float64", finite=True),
            _column("contact", "Boolean"),
            _column("near_limit", "Boolean"),
            _column("force_violation", "Boolean"),
            _column("tcp_x_m", "Float64", finite=True),
            _column("tcp_y_m", "Float64", finite=True),
            _column("tcp_z_m", "Float64", finite=True),
        ],
        ["evaluation_id", "step"],
        step_column="step",
        target_rows=131_072,
        maximum_rows=maximum_native_rows,
    )
    control = _contract(
        "control_trace_v1",
        [
            _column("evaluation_id", "UInt64"),
            _column("step", "UInt32"),
            _column("time_ns", "UInt64"),
            _column("phase_code", "UInt8"),
            _column("primitive_index", "Int16", nullable=True),
            _column("f_ctrl_input_n", "Float64", finite=True),
            _column("f_trk_n", "Float64", finite=True),
            _column("force_target_n", "Float64", finite=True),
            _column("progress", "Float64", finite=True),
            _column("contact", "Boolean"),
            _column("constraint_violation", "Boolean"),
            _column("moving_obstacle_contact", "Boolean"),
            _column("controller_mode", "String"),
            _column("path_error_m", "Float64", nullable=True, finite=True),
        ],
        ["evaluation_id", "step"],
        step_column="step",
        target_rows=131_072,
        maximum_rows=maximum_control_rows,
    )
    primitive = _contract(
        "primitive_metrics_v1",
        [
            _column("evaluation_id", "UInt64"),
            _column("step", "UInt32"),
            _column("event_kind", "String"),
            _column("action_index", "Int16"),
            _column("completed", "Boolean"),
            _column("native_start", "UInt32"),
            _column("native_end", "UInt32"),
            _column("duration_s", "Float64", finite=True),
            _column("peak_force_n", "Float64", finite=True),
            _column("force_violation", "Boolean"),
            _column("spatial_constraint_violation", "Boolean"),
            _column("physical_lifecycle_complete", "Boolean"),
            _column("synthetic_coverage_success", "Boolean", nullable=True),
            _column("synthetic_residual_mass_ratio", "Float64", nullable=True, finite=True),
        ],
        ["evaluation_id", "step"],
        step_column="step",
        target_rows=16_384,
        maximum_rows=maximum_primitive_rows + 1,
    )
    episode = _contract(
        "episode_metrics_v1",
        [
            _column("evaluation_id", "UInt64"),
            _column("physical_lifecycle_complete", "Boolean"),
            _column("synthetic_coverage_success", "Boolean", nullable=True),
            _column("synthetic_residual_mass_ratio", "Float64", nullable=True, finite=True),
            _column("failure_code", "UInt16"),
            _column("first_pass_success", "Boolean"),
            _column("planned_primitives", "UInt16"),
            _column("actual_primitives", "UInt16"),
            _column("native_samples", "UInt32"),
            _column("control_samples", "UInt32"),
            _column("peak_force_n", "Float64", finite=True),
            _column("stable_window_available", "Boolean"),
            _column("tracking_mean_force_n", "Float64", nullable=True, finite=True),
            _column("tracking_rmse_n", "Float64", nullable=True, finite=True),
            _column("tracking_band_fraction", "Float64", nullable=True, finite=True),
            _column("contact_fraction", "Float64", nullable=True, finite=True),
            _column("contact_loss_count", "UInt16"),
            _column("longest_contact_loss_samples", "UInt32"),
            _column("geometric_hover_returned", "Boolean"),
            _column("terminal_normal_clearance_m", "Float64", nullable=True, finite=True),
        ],
        ["evaluation_id"],
        step_column=None,
        target_rows=16_384,
        maximum_rows=1,
    )
    return {item.table_name: item for item in (native, control, primitive, episode)}


def evidence_frames(
    evidence: TrainingEpisodeEvidence,
    contracts: Mapping[str, SchemaContract],
) -> dict[str, pl.DataFrame]:
    evidence.validate()
    source = {
        "native_trace_v1": evidence.native_rows,
        "control_trace_v1": evidence.control_rows,
        "primitive_metrics_v1": evidence.primitive_rows,
        "episode_metrics_v1": (evidence.episode_row,),
    }
    if set(source) != set(contracts):
        raise StreamingDataError("training evidence contract set differs from source tables")
    frames = {
        table: pl.DataFrame(rows, schema=contracts[table].schema)
        for table, rows in source.items()
    }
    for table, frame in frames.items():
        contracts[table].validate_frame(frame, evidence.evaluation_id)
    return frames


def tracking_summary(
    control_rows: tuple[Mapping[str, Any], ...], *, target_force_n: float,
    minimum_contact_force_n: float = 3.0,
    phase_codes: tuple[int, ...] = (int(PhysicalPhaseCode.FIRST_PASS),),
    stable_progress_minimum: float = 0.05,
) -> dict[str, Any]:
    allowed = {int(value) for value in phase_codes}
    candidates = [row for row in control_rows if int(row["phase_code"]) in allowed]
    start = next(
        (
            index
            for index, row in enumerate(candidates)
            if float(row["progress"]) >= stable_progress_minimum
            and float(row["f_trk_n"]) >= minimum_contact_force_n
        ),
        None,
    )
    stable = [] if start is None else candidates[start:]
    if not stable:
        return {
            "stable_window_available": False,
            "tracking_mean_force_n": None,
            "tracking_rmse_n": None,
            "tracking_band_fraction": None,
            "contact_fraction": None,
            "contact_loss_count": 0,
            "longest_contact_loss_samples": 0,
        }
    forces = [float(row["f_trk_n"]) for row in stable]
    mean = sum(forces) / len(forces)
    rmse = math.sqrt(sum((value - target_force_n) ** 2 for value in forces) / len(forces))
    band = sum(abs(value - target_force_n) <= 0.25 * target_force_n for value in forces) / len(forces)
    contact = [value >= minimum_contact_force_n for value in forces]
    loss_count = 0
    longest = 0
    current = 0
    for present in contact:
        if present:
            current = 0
        else:
            if current == 0:
                loss_count += 1
            current += 1
            longest = max(longest, current)
    return {
        "stable_window_available": True,
        "tracking_mean_force_n": mean,
        "tracking_rmse_n": rmse,
        "tracking_band_fraction": band,
        "contact_fraction": sum(contact) / len(contact),
        "contact_loss_count": loss_count,
        "longest_contact_loss_samples": longest,
    }
