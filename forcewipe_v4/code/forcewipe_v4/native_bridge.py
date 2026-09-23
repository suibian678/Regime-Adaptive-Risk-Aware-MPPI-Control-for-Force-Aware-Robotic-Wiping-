"""Simulator-facing native-step adapter for V4 causal residual accounting."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

import numpy as np

from .information_flow import (
    ObservationMode,
    PolicyObservation,
    ResidualInformationFlow,
    ShieldObservation,
    SimulatedVisionConfig,
    TrackingForceMeasurement,
    VisualResidualMeasurement,
)
from .simulated_vision import SimulatedRGBFrame
from .simulated_vision import ResidualStripCameraConfig
from .residual import (
    CausalContactSample,
    CausalResidualConfig,
    ResidualContractError,
    ResidualUpdateAudit,
    TruthAuditCapability,
    TruthAuditSnapshot,
    create_truth_state,
)
from .scenarios import (
    ScenarioPath,
    ScenarioSpec,
    footprint_overlap_fraction,
    make_residual_field,
    make_scenario_path,
)


@dataclass(frozen=True)
class NativeStepInput:
    sample_index: int
    time_ns: int
    audit_force_n: float
    tracking_force_measurement: TrackingForceMeasurement | None
    tcp_position_xyz: np.ndarray
    tcp_velocity_xyz: np.ndarray

    def __post_init__(self) -> None:
        if int(self.sample_index) < 0 or int(self.time_ns) < 0:
            raise ResidualContractError("native step index/time must be nonnegative")
        if not math.isfinite(float(self.audit_force_n)):
            raise ResidualContractError("native audit force must be finite")
        for name in ("tcp_position_xyz", "tcp_velocity_xyz"):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if value.shape != (3,) or not np.all(np.isfinite(value)):
                raise ResidualContractError(f"{name} must be a finite three-vector")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "sample_index", int(self.sample_index))
        object.__setattr__(self, "time_ns", int(self.time_ns))


@dataclass(frozen=True)
class NativeStepRecord:
    sample_index: int
    time_ns: int
    audit_force_n: float
    tracking_force_n: float | None
    tracking_force_measurement_sample_index: int | None
    tracking_force_measurement_time_ns: int | None
    tracking_force_age_samples: int | None
    tracking_force_age_ns: int | None
    tracking_force_is_held: bool | None
    progress: float
    path_distance_m: float
    tangential_speed_m_s: float
    tcp_position_xyz: tuple[float, float, float]
    tcp_velocity_xyz: tuple[float, float, float]
    contact: bool
    near_limit: bool
    force_violation: bool
    residual_update: ResidualUpdateAudit
    policy_observation: PolicyObservation
    control_step_index: int | None = None
    substep_index: int | None = None


class V4NativeStepBridge:
    """Consumes one complete native sample and advances all causal V4 streams."""

    def __init__(
        self,
        *,
        scenario: ScenarioSpec,
        path: ScenarioPath,
        information_flow: ResidualInformationFlow,
        audit_capability: TruthAuditCapability,
        residual_cells: int,
        force_limit_n: float = 15.0,
        near_limit_n: float = 14.5,
        contact_threshold_n: float = 3.0,
    ) -> None:
        scenario.validate()
        if int(residual_cells) < 8:
            raise ResidualContractError("native bridge requires at least eight residual cells")
        if not 0 < contact_threshold_n < near_limit_n < force_limit_n:
            raise ResidualContractError("invalid force-audit thresholds")
        self.scenario = scenario
        self.path = path
        self._flow = information_flow
        self._audit_capability = audit_capability
        self.residual_cells = int(residual_cells)
        self.force_limit_n = float(force_limit_n)
        self.near_limit_n = float(near_limit_n)
        self.contact_threshold_n = float(contact_threshold_n)

    def advance(self, native: NativeStepInput) -> NativeStepRecord:
        projection = self.path.project(native.tcp_position_xyz)
        tangential_speed = abs(
            float(np.dot(native.tcp_velocity_xyz, projection.tangent_xyz))
        )
        overlap = footprint_overlap_fraction(
            progress=projection.progress,
            path_length_m=self.path.total_length,
            footprint_length_m=self.scenario.tool_footprint_length_m,
            cells=self.residual_cells,
        )
        sample = CausalContactSample(
            sample_index=native.sample_index,
            time_ns=native.time_ns,
            audit_force_n=native.audit_force_n,
            nominal_force_n=self.scenario.target_force_n,
            tangential_speed_m_s=tangential_speed,
            overlap_fraction=overlap,
        )
        update = self._flow.advance(
            sample,
            tracking_force_measurement=native.tracking_force_measurement,
        )
        policy = self._flow.policy_observation()
        force_record = self._flow.shield_observation(residual_aware=False)
        return NativeStepRecord(
            sample_index=native.sample_index,
            time_ns=native.time_ns,
            audit_force_n=float(native.audit_force_n),
            tracking_force_n=force_record.tracking_force_n,
            tracking_force_measurement_sample_index=force_record.tracking_force_measurement_sample_index,
            tracking_force_measurement_time_ns=force_record.tracking_force_measurement_time_ns,
            tracking_force_age_samples=force_record.tracking_force_age_samples,
            tracking_force_age_ns=force_record.tracking_force_age_ns,
            tracking_force_is_held=force_record.tracking_force_is_held,
            progress=projection.progress,
            path_distance_m=projection.distance_m,
            tangential_speed_m_s=tangential_speed,
            tcp_position_xyz=(
                float(native.tcp_position_xyz[0]),
                float(native.tcp_position_xyz[1]),
                float(native.tcp_position_xyz[2]),
            ),
            tcp_velocity_xyz=(
                float(native.tcp_velocity_xyz[0]), float(native.tcp_velocity_xyz[1]), float(native.tcp_velocity_xyz[2])
            ),
            contact=bool(native.audit_force_n >= self.contact_threshold_n),
            near_limit=bool(native.audit_force_n >= self.near_limit_n),
            force_violation=bool(native.audit_force_n > self.force_limit_n),
            residual_update=update,
            policy_observation=policy,
        )

    def shield_observation(self, *, residual_aware: bool) -> ShieldObservation:
        return self._flow.shield_observation(residual_aware=residual_aware)

    def policy_visual_frame(self) -> SimulatedRGBFrame | None:
        return self._flow.policy_visual_frame()

    def truth_audit_snapshot(
        self, capability: TruthAuditCapability
    ) -> TruthAuditSnapshot:
        return self._flow.truth_audit_snapshot(capability)


def create_native_step_bridge(
    scenario: ScenarioSpec,
    *,
    mode: ObservationMode,
    residual_cells: int = 64,
    residual_config: CausalResidualConfig | None = None,
    vision_config: SimulatedVisionConfig | None = None,
    visual_measurement_sink: Callable[[VisualResidualMeasurement], None] | None = None,
    visual_decoder: Callable[
        [SimulatedRGBFrame, ResidualStripCameraConfig],
        tuple[tuple[float, ...], tuple[float, ...]],
    ]
    | None = None,
) -> tuple[V4NativeStepBridge, TruthAuditCapability]:
    scenario.validate()
    path = make_scenario_path(scenario)
    initial_truth = make_residual_field(scenario, cells=residual_cells)
    if residual_config is None:
        residual_config = CausalResidualConfig(
            kappa=scenario.residual_cleanability,
            footprint_length_m=scenario.tool_footprint_length_m
        )
    else:
        if not math.isclose(
            residual_config.footprint_length_m,
            scenario.tool_footprint_length_m,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ResidualContractError(
                "residual footprint length differs from scenario tool profile"
            )
        if not math.isclose(
            residual_config.kappa,
            scenario.residual_cleanability,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ResidualContractError(
                "residual cleanability differs from scenario material profile"
            )
    truth, capability = create_truth_state(initial_truth, residual_config)
    flow = ResidualInformationFlow(
        truth,
        capability,
        mode=mode,
        residual_config=residual_config,
        vision_config=vision_config,
        initial_estimate=np.ones(residual_cells, dtype=np.float64),
        visual_measurement_sink=visual_measurement_sink,
        visual_decoder=visual_decoder,
    )
    bridge = V4NativeStepBridge(
        scenario=scenario,
        path=path,
        information_flow=flow,
        audit_capability=capability,
        residual_cells=residual_cells,
    )
    return bridge, capability
