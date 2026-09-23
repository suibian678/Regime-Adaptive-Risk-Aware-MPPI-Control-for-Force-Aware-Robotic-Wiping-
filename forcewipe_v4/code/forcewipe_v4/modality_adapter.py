"""Matched, causal observation packets for ForceWipe V4 modality studies."""

from __future__ import annotations

from dataclasses import dataclass, fields
import math

import numpy as np

from .decision_protocol import (
    DecisionContext,
    DecisionEncodingConfig,
    DecisionProtocolError,
    encode_policy_decision_observation,
)
from .information_flow import ObservationMode, PolicyObservation
from .simulated_vision import SimulatedRGBFrame


@dataclass(frozen=True)
class ForceHistoryConfig:
    history_length: int = 32
    native_rate_hz: int = 100
    sample_period_native_samples: int = 5
    force_scale_n: float = 15.0
    speed_scale_m_s: float = 0.08
    maximum_force_age_samples: int = 100
    maximum_effective_traversal: float = 100.0

    def validate(self) -> None:
        if (
            int(self.history_length) <= 0
            or int(self.native_rate_hz) <= 0
            or int(self.sample_period_native_samples) <= 0
            or int(self.maximum_force_age_samples) <= 0
        ):
            raise DecisionProtocolError("force-history lengths must be positive")
        scales = (
            self.force_scale_n,
            self.speed_scale_m_s,
            self.maximum_effective_traversal,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in scales):
            raise DecisionProtocolError("force-history scales must be finite and positive")


FORCE_HISTORY_FEATURES = (
    "tracking_force_scaled",
    "force_available",
    "force_age_scaled",
    "force_is_held",
    "nominal_force_scaled",
    "tangential_speed_scaled",
    "cumulative_effective_traversal_scaled",
    "path_progress",
    "elapsed_fraction",
)


class CausalForceHistoryBuffer:
    """Fixed-window causal force/proprio history with explicit left-padding mask."""

    def __init__(self, config: ForceHistoryConfig = ForceHistoryConfig()) -> None:
        config.validate()
        self.config = config
        self._rows: list[np.ndarray] = []
        self._last_elapsed_s: float | None = None
        self._last_observed_sample_index: int | None = None
        self._last_observed_time_ns: int | None = None

    def append(self, observation: PolicyObservation, context: DecisionContext) -> bool:
        context.validate()
        mode = ObservationMode(observation.mode)
        if mode not in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}:
            raise DecisionProtocolError("force history accepts only force-visible modes")
        if observation.tracking_force_n is None:
            raise DecisionProtocolError("force-visible mode lacks a force measurement")
        if observation.cumulative_effective_traversal is None:
            raise DecisionProtocolError("force-visible mode lacks traversal history")
        if self._last_observed_sample_index is not None and (
            int(observation.sample_index) <= self._last_observed_sample_index
            or int(observation.time_ns) <= self._last_observed_time_ns
            or float(context.elapsed_s) < self._last_elapsed_s
        ):
            raise DecisionProtocolError("force-history inputs must be strictly causal")
        if self._last_observed_sample_index is not None:
            sample_delta = int(observation.sample_index) - self._last_observed_sample_index
            time_delta = int(observation.time_ns) - self._last_observed_time_ns
            expected_time_delta = int(
                round(sample_delta * 1_000_000_000 / self.config.native_rate_hz)
            )
            if time_delta != expected_time_delta:
                raise DecisionProtocolError(
                    "force-history sample/time cadence differs from the native rate"
                )
        self._last_observed_sample_index = int(observation.sample_index)
        self._last_observed_time_ns = int(observation.time_ns)
        self._last_elapsed_s = float(context.elapsed_s)
        # The first native sample may legitimately expose the reset-time force
        # before there is a previous native-sample source.  It is causal, but
        # it is not eligible for the sampled history until an age exists.
        if observation.tracking_force_age_samples is None:
            if int(observation.sample_index) == 0:
                return False
            raise DecisionProtocolError("force-visible mode lacks force age")
        if int(observation.sample_index) % int(self.config.sample_period_native_samples) != 0:
            return False
        row = np.asarray(
            [
                float(observation.tracking_force_n) / self.config.force_scale_n,
                1.0,
                min(
                    float(observation.tracking_force_age_samples)
                    / self.config.maximum_force_age_samples,
                    1.0,
                ),
                float(bool(observation.tracking_force_is_held)),
                float(observation.nominal_force_n) / self.config.force_scale_n,
                float(observation.tangential_speed_m_s) / self.config.speed_scale_m_s,
                min(
                    float(observation.cumulative_effective_traversal)
                    / self.config.maximum_effective_traversal,
                    1.0,
                ),
                float(context.path_progress),
                float(context.elapsed_s) / float(context.episode_horizon_s),
            ],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(row)):
            raise DecisionProtocolError("force-history row contains NaN/Inf")
        self._rows.append(row)
        self._rows = self._rows[-int(self.config.history_length) :]
        return True

    def snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        length = int(self.config.history_length)
        feature_count = len(FORCE_HISTORY_FEATURES)
        values = np.zeros((length, feature_count), dtype=np.float64)
        mask = np.zeros(length, dtype=np.float64)
        count = min(len(self._rows), length)
        if count:
            values[-count:] = np.stack(self._rows[-count:], axis=0)
            mask[-count:] = 1.0
        values.setflags(write=False)
        mask.setflags(write=False)
        return values, mask


@dataclass(frozen=True)
class MatchedModalityPacket:
    mode: ObservationMode
    sample_index: int
    time_ns: int
    decision_vector: tuple[float, ...]
    force_history_flat: tuple[float, ...]
    force_history_mask: tuple[float, ...]
    force_history_length: int
    force_history_feature_count: int
    visual_frame: SimulatedRGBFrame | None
    visual_available: bool
    visual_source_sample_index: int | None
    visual_age_samples: int | None
    privileged_oracle: bool


def assert_matched_modality_packet_nonprivileged() -> None:
    prohibited = ("truth", "hidden", "future")
    names = {field.name.lower() for field in fields(MatchedModalityPacket)}
    leaked = sorted(name for name in names if any(token in name for token in prohibited))
    if leaked:
        raise DecisionProtocolError(f"privileged field leaked into modality packet: {leaked}")


class MatchedModalityAdapter:
    """Build one fixed contract while enforcing each modality's access rights."""

    def __init__(
        self,
        *,
        decision_config: DecisionEncodingConfig,
        force_history_config: ForceHistoryConfig = ForceHistoryConfig(),
    ) -> None:
        assert_matched_modality_packet_nonprivileged()
        decision_config.validate()
        force_history_config.validate()
        self.decision_config = decision_config
        self.force_history_config = force_history_config
        self._force_history = CausalForceHistoryBuffer(force_history_config)

    def observe(self, observation: PolicyObservation, context: DecisionContext) -> bool:
        """Ingest one causal native observation into the fixed-rate history stream."""

        mode = ObservationMode(observation.mode)
        if observation.privileged_oracle or mode == ObservationMode.ORACLE:
            raise DecisionProtocolError("oracle observations are excluded from matched modalities")
        if mode in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}:
            return self._force_history.append(observation, context)
        return False

    def encode(
        self,
        observation: PolicyObservation,
        context: DecisionContext,
        *,
        visual_frame: SimulatedRGBFrame | None,
    ) -> MatchedModalityPacket:
        mode = ObservationMode(observation.mode)
        if observation.privileged_oracle or mode == ObservationMode.ORACLE:
            raise DecisionProtocolError("oracle observations are excluded from matched modalities")
        visual_mode = mode in {ObservationMode.VISION_ONLY, ObservationMode.FUSION}
        force_mode = mode in {ObservationMode.FORCE_ONLY, ObservationMode.FUSION}
        if not visual_mode and visual_frame is not None:
            raise DecisionProtocolError("nonvisual modality received an RGB frame")
        if visual_frame is not None:
            if visual_frame.sample_index > observation.sample_index or visual_frame.time_ns > observation.time_ns:
                raise DecisionProtocolError("visual frame cannot come from the future")
            if visual_frame.cell_count != int(self.decision_config.residual_cell_count):
                raise DecisionProtocolError("visual frame cell count differs from decision contract")
        if force_mode:
            history, mask = self._force_history.snapshot()
            if not bool(np.any(mask)):
                raise DecisionProtocolError(
                    "force-visible modality has no causal history; call observe first"
                )
        else:
            history = np.zeros(
                (
                    int(self.force_history_config.history_length),
                    len(FORCE_HISTORY_FEATURES),
                ),
                dtype=np.float64,
            )
            mask = np.zeros(
                int(self.force_history_config.history_length), dtype=np.float64
            )
        decision = encode_policy_decision_observation(
            observation,
            context,
            config=self.decision_config,
            allow_oracle=False,
        )
        return MatchedModalityPacket(
            mode=mode,
            sample_index=int(observation.sample_index),
            time_ns=int(observation.time_ns),
            decision_vector=tuple(float(value) for value in decision),
            force_history_flat=tuple(float(value) for value in history.reshape(-1)),
            force_history_mask=tuple(float(value) for value in mask),
            force_history_length=int(self.force_history_config.history_length),
            force_history_feature_count=len(FORCE_HISTORY_FEATURES),
            visual_frame=visual_frame if visual_mode else None,
            visual_available=bool(visual_mode and visual_frame is not None),
            visual_source_sample_index=(
                visual_frame.sample_index if visual_frame is not None else None
            ),
            visual_age_samples=(
                int(observation.sample_index - visual_frame.sample_index)
                if visual_frame is not None
                else None
            ),
            privileged_oracle=False,
        )
