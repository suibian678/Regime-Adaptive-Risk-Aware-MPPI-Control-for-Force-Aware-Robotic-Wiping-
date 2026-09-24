"""Non-privileged observation adapters for ForceWipe V4 residual studies."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
import math
from typing import Callable

import numpy as np

from forcewipe.simulation.residual import (
    CausalContactSample,
    CausalResidualConfig,
    CausalResidualTruth,
    ResidualContractError,
    ResidualUpdateAudit,
    TruthAuditCapability,
    TruthAuditSnapshot,
    removal_increment_from_components,
)
from forcewipe.simulation.simulated_vision import (
    ResidualStripCameraConfig,
    SimulatedRGBFrame,
    decode_residual_strip_rgb,
    render_residual_strip_rgb,
)


class ObservationMode(str, Enum):
    ORACLE = "oracle"
    VISION_ONLY = "vision_only"
    FORCE_ONLY = "force_only"
    FUSION = "fusion"
    PROPRIOCEPTION_ONLY = "proprioception_only"


@dataclass(frozen=True)
class TrackingForceMeasurement:
    """One causal force measurement available to the controller/high level.

    The measurement is acquired before a control interval and may therefore be
    held over multiple native physics samples. ``source_native_sample_index``
    is ``None`` only for the reset-state measurement that precedes native
    sample zero.
    """

    value_n: float
    control_step_index: int
    source_native_sample_index: int | None
    time_ns: int

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.value_n)):
            raise ResidualContractError("tracking-force measurement must be finite")
        if int(self.control_step_index) < 0 or int(self.time_ns) < 0:
            raise ResidualContractError(
                "tracking-force control index/time must be nonnegative"
            )
        source = self.source_native_sample_index
        if source is not None and int(source) < 0:
            raise ResidualContractError(
                "tracking-force source sample must be nonnegative or None"
            )
        object.__setattr__(self, "value_n", float(self.value_n))
        object.__setattr__(self, "control_step_index", int(self.control_step_index))
        object.__setattr__(self, "time_ns", int(self.time_ns))
        if source is not None:
            object.__setattr__(self, "source_native_sample_index", int(source))


@dataclass(frozen=True)
class SimulatedVisionConfig:
    update_period_samples: int = 5
    noise_std: float = 0.05
    bias: float = 0.0
    dropout_probability: float = 0.0
    seed: int = 0
    camera: ResidualStripCameraConfig = ResidualStripCameraConfig()

    def validate(self) -> None:
        if int(self.update_period_samples) <= 0:
            raise ResidualContractError("vision update period must be positive")
        numeric = (self.noise_std, self.bias, self.dropout_probability)
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ResidualContractError("vision configuration must be finite")
        if self.noise_std < 0 or not 0 <= self.dropout_probability <= 1:
            raise ResidualContractError("invalid visual noise/dropout configuration")
        self.camera.validate()


@dataclass(frozen=True)
class VisualResidualMeasurement:
    sample_index: int
    time_ns: int
    estimate: tuple[float, ...]
    variance: tuple[float, ...]
    rgb_frame: SimulatedRGBFrame

    def __post_init__(self) -> None:
        if int(self.sample_index) != self.rgb_frame.sample_index or int(self.time_ns) != self.rgb_frame.time_ns:
            raise ResidualContractError("visual measurement and RGB frame timestamps differ")
        if len(self.estimate) != self.rgb_frame.cell_count or len(self.variance) != self.rgb_frame.cell_count:
            raise ResidualContractError("visual measurement and RGB frame shapes differ")
        if not all(math.isfinite(float(value)) and 0 <= value <= 1 for value in self.estimate):
            raise ResidualContractError("visual residual estimates must be finite and in [0, 1]")
        if not all(math.isfinite(float(value)) and value > 0 for value in self.variance):
            raise ResidualContractError("visual residual variances must be finite and positive")


class SimulatedVisualResidualSensor:
    """Image formation followed by an RGB-only residual decoder."""

    def __init__(
        self,
        config: SimulatedVisionConfig,
        *,
        decoder: Callable[
            [SimulatedRGBFrame, ResidualStripCameraConfig],
            tuple[tuple[float, ...], tuple[float, ...]],
        ]
        | None = None,
    ):
        config.validate()
        self.config = config
        self._rng = np.random.default_rng(int(config.seed))
        self._decoder = decoder

    def observe(
        self,
        current_truth: np.ndarray,
        *,
        sample_index: int,
        time_ns: int,
    ) -> VisualResidualMeasurement | None:
        if int(sample_index) % int(self.config.update_period_samples) != 0:
            return None
        if self._rng.random() < self.config.dropout_probability:
            return None
        frame = render_residual_strip_rgb(
            current_truth,
            sample_index=int(sample_index),
            time_ns=int(time_ns),
            config=self.config.camera,
            rng=self._rng,
            residual_bias=float(self.config.bias),
            residual_noise_std=float(self.config.noise_std),
        )
        if self._decoder is None:
            estimate, variance = decode_residual_strip_rgb(
                frame, config=self.config.camera
            )
        else:
            estimate, variance = self._decoder(frame, self.config.camera)
        return VisualResidualMeasurement(
            sample_index=int(sample_index),
            time_ns=int(time_ns),
            estimate=estimate,
            variance=variance,
            rgb_frame=frame,
        )


class CausalResidualEstimator:
    """Estimate propagated only with current measured force/motion and visual updates."""

    def __init__(
        self,
        initial_estimate: np.ndarray,
        config: CausalResidualConfig,
        initial_variance: float = 0.25,
    ) -> None:
        values = np.asarray(initial_estimate, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ResidualContractError("initial estimate must be a nonempty vector")
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ResidualContractError("initial estimate must be finite and in [0, 1]")
        if not math.isfinite(initial_variance) or initial_variance <= 0:
            raise ResidualContractError("initial variance must be finite and positive")
        self._values = values.copy()
        self._variance = np.full(values.shape, float(initial_variance), dtype=np.float64)
        self._config = config

    def predict(
        self, sample: CausalContactSample, *, tracking_force_n: float
    ) -> tuple[float, float]:
        if sample.overlap_fraction.size != self._values.size:
            raise ResidualContractError("estimator overlap shape mismatch")
        increment, quality, traversal = removal_increment_from_components(
            overlap_fraction=sample.overlap_fraction,
            force_n=tracking_force_n,
            nominal_force_n=sample.nominal_force_n,
            tangential_speed_m_s=sample.tangential_speed_m_s,
            config=self._config,
        )
        self._values = np.clip(self._values - increment, 0.0, 1.0)
        self._variance = np.minimum(self._variance + 1e-4, 1.0)
        return quality, traversal

    def fuse(self, measurement: VisualResidualMeasurement) -> None:
        observed = np.asarray(measurement.estimate, dtype=np.float64)
        variance = np.asarray(measurement.variance, dtype=np.float64)
        if observed.shape != self._values.shape or variance.shape != self._values.shape:
            raise ResidualContractError("visual measurement shape mismatch")
        gain = self._variance / (self._variance + variance)
        self._values = np.clip(
            self._values + gain * (observed - self._values), 0.0, 1.0
        )
        self._variance = np.maximum((1.0 - gain) * self._variance, 1e-12)

    def snapshot(self) -> tuple[tuple[float, ...], tuple[float, ...]]:
        return (
            tuple(float(value) for value in self._values),
            tuple(float(value) for value in self._variance),
        )


@dataclass(frozen=True)
class PolicyObservation:
    mode: ObservationMode
    sample_index: int
    time_ns: int
    tracking_force_n: float | None
    tracking_force_control_step_index: int | None
    tracking_force_measurement_sample_index: int | None
    tracking_force_measurement_time_ns: int | None
    tracking_force_age_samples: int | None
    tracking_force_age_ns: int | None
    tracking_force_is_held: bool | None
    nominal_force_n: float
    tangential_speed_m_s: float
    cumulative_effective_traversal: float | None
    residual_estimate: tuple[float, ...] | None
    residual_uncertainty: tuple[float, ...] | None
    residual_measurement_sample_index: int | None
    residual_measurement_time_ns: int | None
    privileged_oracle: bool


@dataclass(frozen=True)
class ShieldObservation:
    sample_index: int
    time_ns: int
    tracking_force_n: float | None
    tracking_force_control_step_index: int | None
    tracking_force_measurement_sample_index: int | None
    tracking_force_measurement_time_ns: int | None
    tracking_force_age_samples: int | None
    tracking_force_age_ns: int | None
    tracking_force_is_held: bool | None
    nominal_force_n: float
    residual_estimate: tuple[float, ...] | None
    residual_uncertainty: tuple[float, ...] | None
    residual_measurement_sample_index: int | None
    residual_measurement_time_ns: int | None
    residual_aware: bool


def assert_nonoracle_packet_schema() -> None:
    """Static guard against accidentally adding truth/hidden fields to policy packets."""
    prohibited = ("truth", "oracle_truth", "hidden", "future")
    for data_class in (PolicyObservation, ShieldObservation):
        names = {field.name.lower() for field in fields(data_class)}
        leaked = sorted(name for name in names if any(token in name for token in prohibited))
        if leaked:
            raise ResidualContractError(
                f"privileged field leaked into {data_class.__name__}: {leaked}"
            )


class ResidualInformationFlow:
    """Environment boundary that separates truth, sensing, policy, and shield views."""

    def __init__(
        self,
        truth: CausalResidualTruth,
        audit_capability: TruthAuditCapability,
        *,
        mode: ObservationMode,
        residual_config: CausalResidualConfig,
        vision_config: SimulatedVisionConfig | None = None,
        initial_estimate: np.ndarray | None = None,
        visual_measurement_sink: Callable[[VisualResidualMeasurement], None] | None = None,
        visual_decoder: Callable[
            [SimulatedRGBFrame, ResidualStripCameraConfig],
            tuple[tuple[float, ...], tuple[float, ...]],
        ]
        | None = None,
    ) -> None:
        assert_nonoracle_packet_schema()
        if truth.config != residual_config:
            raise ResidualContractError("truth and estimator residual configurations differ")
        self._truth = truth
        self._audit_capability = audit_capability
        self.mode = ObservationMode(mode)
        self._sensor = SimulatedVisualResidualSensor(
            vision_config if vision_config is not None else SimulatedVisionConfig(),
            decoder=visual_decoder,
        )
        estimate = (
            np.ones(truth.cell_count, dtype=np.float64)
            if initial_estimate is None
            else np.asarray(initial_estimate, dtype=np.float64)
        )
        self._estimator = CausalResidualEstimator(estimate, residual_config)
        self._last_visual: VisualResidualMeasurement | None = None
        self._visual_measurement_sink = visual_measurement_sink
        self._last_sample: CausalContactSample | None = None
        self._last_tracking_force: TrackingForceMeasurement | None = None
        self._last_update: ResidualUpdateAudit | None = None
        self._cumulative_effective_traversal = 0.0

    def advance(
        self,
        sample: CausalContactSample,
        *,
        tracking_force_measurement: TrackingForceMeasurement | None = None,
    ) -> ResidualUpdateAudit:
        if tracking_force_measurement is not None:
            measurement = tracking_force_measurement
            if measurement.time_ns > sample.time_ns:
                raise ResidualContractError(
                    "tracking-force measurement cannot come from the future"
                )
            if (
                measurement.source_native_sample_index is not None
                and measurement.source_native_sample_index > sample.sample_index
            ):
                raise ResidualContractError(
                    "tracking-force source sample cannot come from the future"
                )
            previous = self._last_tracking_force
            if previous is not None and (
                measurement.control_step_index <= previous.control_step_index
                or measurement.time_ns < previous.time_ns
            ):
                raise ResidualContractError(
                    "tracking-force measurements must be temporally ordered"
                )
            self._last_tracking_force = measurement
        update = self._truth.update(sample)
        tracking_quality = 0.0
        traversal = 0.0
        if self._last_tracking_force is not None:
            tracking_quality, traversal = self._estimator.predict(
                sample, tracking_force_n=self._last_tracking_force.value_n
            )
        visual = None
        if self.mode in {ObservationMode.VISION_ONLY, ObservationMode.FUSION}:
            visual = self._sensor.observe(
                self._truth._copy_values_for_sensor(),
                sample_index=sample.sample_index,
                time_ns=sample.time_ns,
            )
        if visual is not None:
            self._last_visual = visual
            if self._visual_measurement_sink is not None:
                self._visual_measurement_sink(visual)
            if self.mode == ObservationMode.FUSION:
                self._estimator.fuse(visual)
        self._cumulative_effective_traversal += tracking_quality * traversal
        self._last_sample = sample
        self._last_update = update
        return update

    def _require_sample(self) -> CausalContactSample:
        if self._last_sample is None:
            raise ResidualContractError("no causal sample has been processed")
        return self._last_sample

    def policy_observation(self) -> PolicyObservation:
        sample = self._require_sample()
        estimate: tuple[float, ...] | None = None
        uncertainty: tuple[float, ...] | None = None
        privileged_oracle = False
        force_measurement = self._last_tracking_force
        high_level_force = (
            float(force_measurement.value_n)
            if self.mode in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}
            and force_measurement is not None
            else None
        )
        force_history = (
            self._cumulative_effective_traversal
            if self.mode in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}
            else None
        )
        measurement_sample_index: int | None = None
        measurement_time_ns: int | None = None
        if self.mode == ObservationMode.ORACLE:
            snapshot = self._truth.audit_snapshot(self._audit_capability)
            estimate = snapshot.values
            uncertainty = tuple(0.0 for _ in snapshot.values)
            privileged_oracle = True
            measurement_sample_index = snapshot.sample_index
            measurement_time_ns = snapshot.time_ns
        elif self.mode == ObservationMode.VISION_ONLY and self._last_visual is not None:
            estimate = self._last_visual.estimate
            uncertainty = self._last_visual.variance
            measurement_sample_index = self._last_visual.sample_index
            measurement_time_ns = self._last_visual.time_ns
        elif self.mode == ObservationMode.FUSION:
            estimate, uncertainty = self._estimator.snapshot()
            if self._last_visual is not None:
                measurement_sample_index = self._last_visual.sample_index
                measurement_time_ns = self._last_visual.time_ns
        return PolicyObservation(
            mode=self.mode,
            sample_index=sample.sample_index,
            time_ns=sample.time_ns,
            tracking_force_n=high_level_force,
            tracking_force_control_step_index=(
                force_measurement.control_step_index
                if high_level_force is not None
                else None
            ),
            tracking_force_measurement_sample_index=(
                force_measurement.source_native_sample_index
                if high_level_force is not None
                else None
            ),
            tracking_force_measurement_time_ns=(
                force_measurement.time_ns if high_level_force is not None else None
            ),
            tracking_force_age_samples=(
                sample.sample_index - force_measurement.source_native_sample_index
                if high_level_force is not None
                and force_measurement.source_native_sample_index is not None
                else None
            ),
            tracking_force_age_ns=(
                sample.time_ns - force_measurement.time_ns
                if high_level_force is not None
                else None
            ),
            tracking_force_is_held=(
                sample.time_ns > force_measurement.time_ns
                if high_level_force is not None
                else None
            ),
            nominal_force_n=float(sample.nominal_force_n),
            tangential_speed_m_s=float(sample.tangential_speed_m_s),
            cumulative_effective_traversal=force_history,
            residual_estimate=estimate,
            residual_uncertainty=uncertainty,
            residual_measurement_sample_index=measurement_sample_index,
            residual_measurement_time_ns=measurement_time_ns,
            privileged_oracle=privileged_oracle,
        )

    def shield_observation(self, *, residual_aware: bool) -> ShieldObservation:
        policy = self.policy_observation()
        sample = self._require_sample()
        estimate = policy.residual_estimate if residual_aware else None
        uncertainty = policy.residual_uncertainty if residual_aware else None
        force_measurement = self._last_tracking_force
        return ShieldObservation(
            sample_index=policy.sample_index,
            time_ns=policy.time_ns,
            tracking_force_n=(
                force_measurement.value_n if force_measurement is not None else None
            ),
            tracking_force_control_step_index=(
                force_measurement.control_step_index
                if force_measurement is not None
                else None
            ),
            tracking_force_measurement_sample_index=(
                force_measurement.source_native_sample_index
                if force_measurement is not None
                else None
            ),
            tracking_force_measurement_time_ns=(
                force_measurement.time_ns if force_measurement is not None else None
            ),
            tracking_force_age_samples=(
                sample.sample_index - force_measurement.source_native_sample_index
                if force_measurement is not None
                and force_measurement.source_native_sample_index is not None
                else None
            ),
            tracking_force_age_ns=(
                sample.time_ns - force_measurement.time_ns
                if force_measurement is not None
                else None
            ),
            tracking_force_is_held=(
                sample.time_ns > force_measurement.time_ns
                if force_measurement is not None
                else None
            ),
            nominal_force_n=policy.nominal_force_n,
            residual_estimate=estimate,
            residual_uncertainty=uncertainty,
            residual_measurement_sample_index=(
                policy.residual_measurement_sample_index if residual_aware else None
            ),
            residual_measurement_time_ns=(
                policy.residual_measurement_time_ns if residual_aware else None
            ),
            residual_aware=bool(residual_aware),
        )

    def policy_visual_frame(self) -> SimulatedRGBFrame | None:
        """Return the latest causal RGB frame only to visual policy modes."""

        self._require_sample()
        if self.mode not in {ObservationMode.VISION_ONLY, ObservationMode.FUSION}:
            return None
        return self._last_visual.rgb_frame if self._last_visual is not None else None

    def truth_audit_snapshot(
        self, capability: TruthAuditCapability
    ) -> TruthAuditSnapshot:
        return self._truth.audit_snapshot(capability)
