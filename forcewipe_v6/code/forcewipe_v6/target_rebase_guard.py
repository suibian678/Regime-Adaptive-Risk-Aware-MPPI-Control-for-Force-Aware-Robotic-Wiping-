"""Acquisition braking plus a path-phase target-reference rebase guard."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from forcewipe_v6.acquisition_predictive_brake import (
    AcquisitionPredictiveBrake,
    AcquisitionPredictiveBrakeConfig,
)


NormalIncrement = Callable[..., tuple[float, dict]]


@dataclass(frozen=True)
class TargetRebaseGuardConfig:
    control_period_s: float = 0.01
    rate_clip_n_s: float = 300.0
    rate_horizon_s: float = 0.02
    envelope_margin_n: float = 0.30
    trigger_margin_above_target_n: float = 1.0
    maximum_trigger_n: float = 13.0
    acquisition_outward_base_m: float = 0.00040
    acquisition_outward_gain_m_per_n: float = 0.00010
    acquisition_maximum_outward_m: float = 0.00080
    ready_fraction: float = 0.80
    ready_samples: int = 20
    collapse_previous_force_fraction: float = 0.80
    collapse_current_force_fraction: float = 0.70
    collapse_rate_threshold_n_s: float = -150.0
    target_rebase_outward_clearance_m: float = 0.00050
    maximum_target_rebase_m: float = 0.00600

    def validate(self) -> None:
        numeric = tuple(float(value) for value in self.__dict__.values())
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("target-rebase configuration must be finite")
        if self.control_period_s <= 0.0 or self.rate_clip_n_s <= 0.0:
            raise ValueError("target-rebase timing/rate limits must be positive")
        if not 0.0 < self.ready_fraction <= 1.0 or self.ready_samples < 1:
            raise ValueError("target-rebase readiness definition is invalid")
        if not 0.0 < self.collapse_current_force_fraction < self.collapse_previous_force_fraction <= 1.0:
            raise ValueError("target-rebase force fractions are not ordered")
        if self.collapse_rate_threshold_n_s >= 0.0:
            raise ValueError("target-rebase rate threshold must be negative")
        if self.target_rebase_outward_clearance_m < 0.0:
            raise ValueError("target-rebase clearance cannot be negative")
        if self.maximum_target_rebase_m <= self.target_rebase_outward_clearance_m:
            raise ValueError("target-rebase bound is too small")
        self.acquisition_config().validate()

    def acquisition_config(self) -> AcquisitionPredictiveBrakeConfig:
        return AcquisitionPredictiveBrakeConfig(
            control_period_s=self.control_period_s,
            rate_clip_n_s=self.rate_clip_n_s,
            rate_horizon_s=self.rate_horizon_s,
            envelope_margin_n=self.envelope_margin_n,
            trigger_margin_above_target_n=self.trigger_margin_above_target_n,
            maximum_trigger_n=self.maximum_trigger_n,
            outward_base_m=self.acquisition_outward_base_m,
            outward_gain_m_per_n=self.acquisition_outward_gain_m_per_n,
            maximum_outward_m=self.acquisition_maximum_outward_m,
            ready_fraction=self.ready_fraction,
            ready_samples=self.ready_samples,
        )


class TargetRebaseGuard:
    """Reset stored inward target lead when a path-phase force collapse occurs."""

    def __init__(
        self,
        retained_controller: NormalIncrement,
        config: TargetRebaseGuardConfig = TargetRebaseGuardConfig(),
    ) -> None:
        config.validate()
        self.config = config
        self._acquisition = AcquisitionPredictiveBrake(
            retained_controller,
            config.acquisition_config(),
        )

    @property
    def ready_count(self) -> int:
        return self._acquisition.ready_count

    @property
    def path_started(self) -> bool:
        return self._acquisition.path_started

    def __call__(
        self,
        *,
        target_force_n: float,
        measured_force_n: float,
        previous_force_n: float,
        target_lead_down_m: float,
    ) -> tuple[float, dict]:
        cfg = self.config
        path_started_before = self._acquisition.path_started
        increment, detail = self._acquisition(
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
        collapse = bool(
            path_started_before
            and previous_force_n >= cfg.collapse_previous_force_fraction * target_force_n
            and measured_force_n <= cfg.collapse_current_force_fraction * target_force_n
            and rate <= cfg.collapse_rate_threshold_n_s
        )
        rebase_m = 0.0
        if collapse:
            rebase_m = min(
                cfg.maximum_target_rebase_m,
                target_lead_down_m + cfg.target_rebase_outward_clearance_m,
            )
            increment = -rebase_m
            detail.update(
                {
                    "mode": "TARGET_REBASE_GUARD",
                    "retained_mode": str(detail["mode"]),
                    "retained_increment_m": float(detail["projected_increment_m"]),
                    "raw_increment_m": float(detail.get("raw_increment_m", increment)),
                    "projected_increment_m": increment,
                    "projection_active": True,
                }
            )
        detail.update(
            {
                "target_rebase_guard_active": collapse,
                "target_rebase_command_m": -rebase_m,
                "target_rebase_outward_clearance_m": cfg.target_rebase_outward_clearance_m,
                "target_rebase_bound_m": cfg.maximum_target_rebase_m,
            }
        )
        return increment, detail

