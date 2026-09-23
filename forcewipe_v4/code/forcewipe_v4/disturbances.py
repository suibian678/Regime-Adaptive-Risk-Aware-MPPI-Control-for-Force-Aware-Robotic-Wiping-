"""Deterministic causal disturbance mappings for ForceWipe V4."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, replace
import math

import numpy as np

from .information_flow import TrackingForceMeasurement
from .residual import ResidualContractError
from .scenarios import ScenarioSpec


@dataclass(frozen=True)
class PhysicalDisturbanceCommand:
    tool_force_world_n: np.ndarray
    surface_height_offset_m: float

    def __post_init__(self) -> None:
        force = np.asarray(self.tool_force_world_n, dtype=np.float64).copy()
        if force.shape != (3,) or not np.all(np.isfinite(force)):
            raise ResidualContractError("disturbance force must be a finite three-vector")
        if not math.isfinite(float(self.surface_height_offset_m)):
            raise ResidualContractError("surface-height offset must be finite")
        force.setflags(write=False)
        object.__setattr__(self, "tool_force_world_n", force)


def physical_disturbance_command(
    scenario: ScenarioSpec,
    *,
    native_step_index: int,
    native_hz: float,
    outward_normal_world: np.ndarray,
) -> PhysicalDisturbanceCommand:
    """Map one declared physical disturbance to the current native step."""

    scenario.validate()
    step = int(native_step_index)
    hz = float(native_hz)
    normal = np.asarray(outward_normal_world, dtype=np.float64)
    if step < 0 or not math.isfinite(hz) or hz <= 0.0:
        raise ResidualContractError("invalid native disturbance clock")
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise ResidualContractError("disturbance normal must be a finite three-vector")
    normal /= np.linalg.norm(normal)
    scale = float(scenario.disturbance_scale)
    force = np.zeros(3, dtype=np.float64)
    height = 0.0
    impulse_steps = (220, 360)
    if scenario.disturbance_kind == "lateral_impulse" and step in impulse_steps:
        force[1] = (1.0 if step == impulse_steps[0] else -1.0) * (1.0 + 4.0 * scale)
    elif scenario.disturbance_kind == "normal_impulse" and step in impulse_steps:
        force = -normal * (2.0 + 6.0 * scale)
    elif scenario.disturbance_kind == "dynamic_height":
        phase = 2.0 * math.pi * ((scenario.scenario_seed % 997) / 997.0)
        height = 0.0025 * scale * math.sin(2.0 * math.pi * 0.75 * step / hz + phase)
    return PhysicalDisturbanceCommand(force, height)


def reference_noise_offset_world(
    scenario: ScenarioSpec,
    *,
    progress: float,
    tangent_world: np.ndarray,
    outward_normal_world: np.ndarray,
) -> np.ndarray:
    """Return a bounded, spatially correlated reference perturbation."""

    if scenario.disturbance_kind != "reference_noise":
        return np.zeros(3, dtype=np.float64)
    tangent = np.asarray(tangent_world, dtype=np.float64)
    normal = np.asarray(outward_normal_world, dtype=np.float64)
    lateral = np.cross(normal, tangent)
    norm = float(np.linalg.norm(lateral))
    if norm <= 0.0:
        raise ResidualContractError("reference-noise frame is degenerate")
    lateral /= norm
    phase = 2.0 * math.pi * ((scenario.scenario_seed % 1021) / 1021.0)
    amplitude = 0.002 * float(scenario.disturbance_scale)
    scalar = amplitude * (
        0.65 * math.sin(2.0 * math.pi * 3.0 * float(progress) + phase)
        + 0.35 * math.sin(2.0 * math.pi * 7.0 * float(progress) - 0.5 * phase)
    )
    return lateral * scalar


class CausalTrackingForceDisturbance:
    """Apply declared force-sensor noise and latency without future samples."""

    def __init__(self, scenario: ScenarioSpec):
        scenario.validate()
        self.scenario = scenario
        latency = (
            int(scenario.sensor_latency_steps)
            if scenario.disturbance_kind == "sensor_noise_latency"
            else 0
        )
        self._latency = latency
        self._history: deque[TrackingForceMeasurement] = deque(maxlen=latency + 1)
        self._rng = np.random.default_rng(int(scenario.scenario_seed) ^ 0x5EED1234)

    def __call__(self, raw: TrackingForceMeasurement) -> TrackingForceMeasurement:
        if self.scenario.disturbance_kind != "sensor_noise_latency":
            return raw
        sigma_n = 0.25 * float(self.scenario.disturbance_scale)
        noisy = replace(
            raw,
            value_n=max(0.0, float(raw.value_n + self._rng.normal(0.0, sigma_n))),
        )
        self._history.append(noisy)
        delayed = self._history[0]
        return TrackingForceMeasurement(
            value_n=delayed.value_n,
            control_step_index=raw.control_step_index,
            source_native_sample_index=delayed.source_native_sample_index,
            time_ns=delayed.time_ns,
        )
