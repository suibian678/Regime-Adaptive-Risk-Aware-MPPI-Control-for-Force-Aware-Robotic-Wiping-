"""Causal acquisition braking plus a global force-collapse rebound guard."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable


NormalIncrement = Callable[..., tuple[float, dict]]


@dataclass(frozen=True)
class GlobalTransientGuardConfig:
    control_period_s: float = 0.01
    rate_clip_n_s: float = 300.0
    rate_horizon_s: float = 0.02
    envelope_margin_n: float = 0.30
    trigger_margin_above_target_n: float = 1.0
    maximum_trigger_n: float = 13.0
    acquisition_outward_base_m: float = 0.00040
    acquisition_outward_gain_m_per_n: float = 0.00010
    maximum_outward_m: float = 0.00080
    ready_fraction: float = 0.80
    ready_samples: int = 20
    collapse_previous_force_fraction: float = 0.80
    collapse_current_force_fraction: float = 0.70
    collapse_rate_threshold_n_s: float = -150.0
    collapse_outward_m: float = 0.00080

    def validate(self) -> None:
        numeric = tuple(float(value) for value in self.__dict__.values())
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("global transient-guard configuration must be finite")
        if self.control_period_s <= 0.0 or self.rate_clip_n_s <= 0.0:
            raise ValueError("transient-guard timing/rate limits must be positive")
        if self.rate_horizon_s < 0.0 or self.envelope_margin_n < 0.0:
            raise ValueError("transient-guard envelope values cannot be negative")
        if self.trigger_margin_above_target_n <= 0.0 or self.maximum_trigger_n <= 0.0:
            raise ValueError("acquisition trigger is invalid")
        if not 0.0 < self.acquisition_outward_base_m <= self.maximum_outward_m:
            raise ValueError("acquisition outward base is invalid")
        if self.acquisition_outward_gain_m_per_n < 0.0:
            raise ValueError("acquisition outward gain cannot be negative")
        if not 0.0 < self.ready_fraction <= 1.0 or self.ready_samples < 1:
            raise ValueError("readiness definition is invalid")
        if not 0.0 < self.collapse_current_force_fraction < self.collapse_previous_force_fraction <= 1.0:
            raise ValueError("collapse force fractions are not ordered")
        if self.collapse_rate_threshold_n_s >= 0.0:
            raise ValueError("collapse rate threshold must be negative")
        if not 0.0 < self.collapse_outward_m <= self.maximum_outward_m:
            raise ValueError("collapse outward command is invalid")


class GlobalTransientGuard:
    """Preserve steady tracking while intercepting two observed causal signatures."""

    def __init__(
        self,
        retained_controller: NormalIncrement,
        config: GlobalTransientGuardConfig = GlobalTransientGuardConfig(),
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
            raise ValueError("transient-guard inputs must be finite")
        if target_force_n <= 0.0 or measured_force_n < 0.0 or target_lead_down_m < 0.0:
            raise ValueError("transient-guard inputs lie outside their physical domain")

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
        envelope = measured_force_n + cfg.rate_horizon_s * max(rate, 0.0) + cfg.envelope_margin_n
        acquisition_trigger = min(
            cfg.maximum_trigger_n,
            target_force_n + cfg.trigger_margin_above_target_n,
        )
        acquisition_before = not self._path_started
        collapse_active = bool(
            previous_force_n >= cfg.collapse_previous_force_fraction * target_force_n
            and measured_force_n <= cfg.collapse_current_force_fraction * target_force_n
            and rate <= cfg.collapse_rate_threshold_n_s
        )
        acquisition_brake_active = bool(
            not collapse_active and acquisition_before and envelope >= acquisition_trigger
        )

        increment = float(retained_increment)
        detail = dict(retained_detail)
        if collapse_active:
            increment = min(increment, -cfg.collapse_outward_m)
            detail.update(
                {
                    "mode": "REBOUND_GUARD",
                    "retained_mode": str(retained_detail["mode"]),
                    "retained_increment_m": float(retained_increment),
                    "raw_increment_m": float(retained_detail.get("raw_increment_m", retained_increment)),
                    "projected_increment_m": increment,
                    "projection_active": True,
                }
            )
        elif acquisition_brake_active:
            outward = min(
                cfg.maximum_outward_m,
                cfg.acquisition_outward_base_m
                + cfg.acquisition_outward_gain_m_per_n
                * max(0.0, envelope - acquisition_trigger),
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
                "global_rebound_guard_active": collapse_active,
                "acquisition_predictive_brake_active": acquisition_brake_active,
                "acquisition_active_before": acquisition_before,
                "acquisition_active_after": not self._path_started,
                "acquisition_internal_ready_count": self._ready_count,
                "acquisition_envelope_n": envelope,
                "acquisition_trigger_n": acquisition_trigger,
                "collapse_previous_threshold_n": cfg.collapse_previous_force_fraction * target_force_n,
                "collapse_current_threshold_n": cfg.collapse_current_force_fraction * target_force_n,
                "collapse_rate_threshold_n_s": cfg.collapse_rate_threshold_n_s,
            }
        )
        return increment, detail

