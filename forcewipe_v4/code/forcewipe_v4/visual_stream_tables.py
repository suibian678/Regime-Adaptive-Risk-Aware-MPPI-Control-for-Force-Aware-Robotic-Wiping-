"""Sparse immutable RGB-frame table for ForceWipe V4 visual observations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import polars as pl

from .information_flow import VisualResidualMeasurement
from .streaming import DATA_CONTRACT_VERSION, SchemaContract, StreamingDataError


def simulated_rgb_frame_contract(maximum_frames_per_evaluation: int) -> SchemaContract:
    maximum = int(maximum_frames_per_evaluation)
    if maximum <= 0:
        raise StreamingDataError("visual-frame row maximum must be positive")
    columns: list[dict[str, Any]] = [
        {"name": "evaluation_id", "dtype": "UInt64", "nullable": False, "finite": False, "source": "frozen_evaluation_manifest"},
        {"name": "visual_frame_index", "dtype": "UInt32", "nullable": False, "finite": False, "source": "visual_sensor_emission_counter"},
        {"name": "source_native_sample_index", "dtype": "UInt32", "nullable": False, "finite": False, "source": "visual_measurement"},
        {"name": "time_ns", "dtype": "UInt64", "nullable": False, "finite": False, "unit": "ns", "source": "visual_measurement"},
        {"name": "height_px", "dtype": "UInt16", "nullable": False, "finite": False, "unit": "px", "source": "rgb_frame"},
        {"name": "width_px", "dtype": "UInt16", "nullable": False, "finite": False, "unit": "px", "source": "rgb_frame"},
        {"name": "cell_count", "dtype": "UInt16", "nullable": False, "finite": False, "source": "rgb_frame"},
        {"name": "pixel_format", "dtype": "String", "nullable": False, "finite": False, "source": "frozen_visual_contract"},
        {"name": "rgb8", "dtype": "Binary", "nullable": False, "finite": False, "source": "rgb_frame"},
        {"name": "estimate_f32le", "dtype": "Binary", "nullable": False, "finite": False, "source": "image_only_decoder"},
        {"name": "variance_f32le", "dtype": "Binary", "nullable": False, "finite": False, "source": "image_only_decoder"},
    ]
    return SchemaContract.from_dict(
        {
            "data_contract_version": DATA_CONTRACT_VERSION,
            "table_name": "simulated_rgb_frame_v1",
            "table_version": "1.0.0-draft",
            "columns": columns,
            "primary_key": ["evaluation_id", "visual_frame_index"],
            "evaluation_id_column": "evaluation_id",
            "step_column": "visual_frame_index",
            "target_rows": 8192,
            "row_group_size": 2048,
            "compression_level": 3,
            "data_page_size": 1048576,
            "minimum_rows_per_evaluation": 0,
            "maximum_rows_per_evaluation": maximum,
            "sparsity": "one row per emitted camera frame; held frames are not duplicated",
            "binary_arrays": {
                "rgb8": "C-order HxWx3 uint8",
                "estimate_f32le": "cell_count little-endian float32 values",
                "variance_f32le": "cell_count little-endian float32 values",
            },
        }
    )


def simulated_rgb_frame_table(
    evaluation_id: int,
    measurements: Sequence[VisualResidualMeasurement],
    *,
    contract: SchemaContract,
) -> pl.DataFrame:
    if contract.table_name != "simulated_rgb_frame_v1":
        raise StreamingDataError("visual frame builder received the wrong contract")
    rows: list[dict[str, Any]] = []
    previous_sample = -1
    previous_time = -1
    for frame_index, measurement in enumerate(measurements):
        frame = measurement.rgb_frame
        if measurement.sample_index <= previous_sample or measurement.time_ns <= previous_time:
            raise StreamingDataError("visual measurements must be strictly causal and ordered")
        estimate = np.asarray(measurement.estimate, dtype="<f4")
        variance = np.asarray(measurement.variance, dtype="<f4")
        if estimate.shape != (frame.cell_count,) or variance.shape != (frame.cell_count,):
            raise StreamingDataError("decoded visual arrays differ from RGB cell count")
        rows.append(
            {
                "evaluation_id": int(evaluation_id),
                "visual_frame_index": int(frame_index),
                "source_native_sample_index": int(measurement.sample_index),
                "time_ns": int(measurement.time_ns),
                "height_px": int(frame.height_px),
                "width_px": int(frame.width_px),
                "cell_count": int(frame.cell_count),
                "pixel_format": "RGB8_C_ORDER",
                "rgb8": frame.pixels_rgb8,
                "estimate_f32le": estimate.tobytes(order="C"),
                "variance_f32le": variance.tobytes(order="C"),
            }
        )
        previous_sample = int(measurement.sample_index)
        previous_time = int(measurement.time_ns)
    table = pl.DataFrame(rows, schema=contract.schema)
    contract.validate_frame(table, int(evaluation_id))
    return table


def decode_f32le(payload: bytes, *, expected_count: int) -> np.ndarray:
    """Decode one archived float array as an immutable float32 view."""

    try:
        values = np.frombuffer(bytes(payload), dtype="<f4")
    except ValueError as error:
        raise StreamingDataError("archived visual float payload is invalid") from error
    if values.shape != (int(expected_count),) or not np.all(np.isfinite(values)):
        raise StreamingDataError("archived visual float payload is invalid")
    values.setflags(write=False)
    return values
