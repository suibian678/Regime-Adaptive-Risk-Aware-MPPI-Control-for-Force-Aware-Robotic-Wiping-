"""Causal predictive braking used only before the first stable force window.

The retained force loop already prevents additional inward motion when its
force envelope is high.  The frozen CAL showed that this is insufficient in
two 12-N contact-acquisition transients: the command was outward, but too weak
and too late to arrest the plant's stored contact momentum.  This module adds
one bounded mechanism without changing steady path tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


NormalIncrement = Callable[..., tuple[float, dict]]


@dataclass(frozen=True)
class AcquisitionPredictiveBrakeConfig:
    control_period_s: float = 0.01
    rate_clip_n_s: float = 300.0
    rate_horizon_s: float = 0.02
    envelope_margin_n: float = 0.30
    trigger_margin_above_target_n: float = 1.0
    maximum_trigger_n: float = 13.0
    outward_base_m: float = 0.00040
    outward_gain_m_per_n: float = 0.00010
    maximum_outward_m: float = 0.00080
    ready_fraction: float = 0.80
    ready_samples: int = 20

    def validate(self) -> None:
        numeric = (
            self.control_period_s,
            self.rate_clip_n_s,
            self.rate_horizon_s,
            self.envelope_margin_n,
            self.trigger_margin_above_target_n,
            self.maximum_trigger_n,
            self.outward_base_m,
            self.outward_gain_m_per_n,
            self.maximum_outward_m,
            self.ready_fraction,
        )
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("predictive-brake configuration must be finite")
        if self.control_period_s <= 0.0 or self.rate_clip_n_s <= 0.0:
            raise ValueError("predictive-brake timing/rate limits must be positive")
        if self.rate_horizon_s < 0.0 or self.envelope_margin_n < 0.0:
            raise ValueError("predictive envelope margins cannot be negative")
        if self.trigger_margin_above_target_n <= 0.0 or self.maximum_trigger_n <= 0.0:
            raise ValueError("predictive-brake trigger is invalid")
        if not 0.0 < self.outward_base_m <= self.maximum_outward_m:
            raise ValueError("predictive-brake outward base is invalid")
        if self.outward_gain_m_per_n < 0.0 or self.maximum_outward_m <= 0.0:
            raise ValueError("predictive-brake outward bounds are invalid")
        if not 0.0 < self.ready_fraction <= 1.0 or self.ready_samples < 1:
            raise ValueError("predictive-brake readiness definition is invalid")


class AcquisitionPredictiveBrake:
    """Wrap the retained normal controller with a pre-path transient brake."""

    def __init__(
        self,
        retained_controller: NormalIncrement,
        config: AcquisitionPredictiveBrakeConfig = AcquisitionPredictiveBrakeConfig(),
    ) -> None:
        config.validate()
        self._retained_controller = retained_controller
        self.config = config
        self._ready_count = 0
        self._path_started = False

    @property
    def ready_count(self) -> int:
        return self._ready_count

    @property
    def path_started(self) -> bool:
        return self._path_started

    def __call__(
        self,
        *,
        target_force_n: float,
        measured_force_n: float,
        previous_force_n: float,
        target_lead_down_m: float,
    ) -> tuple[float, dict]:
        cfg = self.config
        values = (target_force_n, measured_force_n, previous_force_n, target_lead_down_m)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("predictive-brake inputs must be finite")
        if target_force_n <= 0.0 or measured_force_n < 0.0 or target_lead_down_m < 0.0:
            raise ValueError("predictive-brake inputs lie outside their physical domain")

        retained_increment, retained_detail = self._retained_controller(
            target_force_n=target_force_n,
            measured_force_n=measured_force_n,
            previous_force_n=previous_force_n,
            target_lead_down_m=target_lead_down_m,
        )
        rate = max(
            -cfg.rate_clip_n_s,
            min(
                cfg.rate_clip_n_s,
                (measured_force_n - previous_force_n) / cfg.control_period_s,
            ),
        )
        envelope = (
            measured_force_n
            + cfg.rate_horizon_s * max(rate, 0.0)
            + cfg.envelope_margin_n
        )
        trigger = min(
            cfg.maximum_trigger_n,
            target_force_n + cfg.trigger_margin_above_target_n,
        )
        acquisition_before = not self._path_started
        brake_active = bool(acquisition_before and envelope >= trigger)

        increment = float(retained_increment)
        detail = dict(retained_detail)
        if brake_active:
            outward = min(
                cfg.maximum_outward_m,
                cfg.outward_base_m
                + cfg.outward_gain_m_per_n * max(0.0, envelope - trigger),
            )
            increment = min(increment, -outward)
            detail.update(
                {
                    "mode": "ACQUISITION_BRAKE",
                    "retained_mode": str(retained_detail["mode"]),
                    "retained_increment_m": float(retained_increment),
                    "raw_increment_m": float(retained_detail.get("raw_increment_m", retained_increment)),
                    "projected_increment_m": increment,
                    "projection_active": True,
                }
            )

        ready = bool(
            measured_force_n >= cfg.ready_fraction * target_force_n
            and detail["mode"] == "TRACK"
        )
        self._ready_count = self._ready_count + 1 if ready else 0
        if self._ready_count >= cfg.ready_samples:
            self._path_started = True

        detail.update(
            {
                "acquisition_predictive_brake_active": brake_active,
                "acquisition_active_before": acquisition_before,
                "acquisition_active_after": not self._path_started,
                "acquisition_internal_ready_count": self._ready_count,
                "acquisition_envelope_n": envelope,
                "acquisition_trigger_n": trigger,
            }
        )
        return increment, detail

