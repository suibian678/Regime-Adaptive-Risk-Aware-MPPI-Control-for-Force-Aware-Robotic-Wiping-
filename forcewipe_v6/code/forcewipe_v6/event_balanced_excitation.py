"""Deterministic force-event excitation for world-model development data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


EVENT_PROGRAMS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("single_press_pulse", (1.0, -0.5, -0.5)),
    ("release_reacquire", (-1.0,) * 30 + (1.0,) * 20),
    ("alternating_rebound", (-1.0, -1.0, -0.5, 1.0, 1.0, 0.8)),
    ("tangential_press", (0.8, 0.8, 0.8, -0.8, -0.8)),
    ("double_press_pulse", (0.8, 0.8, -1.0, -1.0, 1.0, 1.0)),
)


@dataclass(frozen=True)
class EventBalancedExcitationConfig:
    target_force_n: float = 12.0
    acquisition_force_n: float = 9.0
    settle_steps: int = 30
    event_period_steps: int = 110
    proportional_gain_action_per_n: float = 0.16
    tracking_tangent_action: float = 0.35

    def validate(self) -> None:
        if self.target_force_n != 12.0:
            raise ValueError("event-balanced collection is fixed at 12 N")
        if not 0.0 < self.acquisition_force_n < self.target_force_n:
            raise ValueError("invalid acquisition force")
        if (
            self.settle_steps < 1
            or self.event_period_steps <= self.settle_steps
            or self.event_period_steps <= max(len(sequence) for _, sequence in EVENT_PROGRAMS)
        ):
            raise ValueError("invalid event timing")
        if self.proportional_gain_action_per_n <= 0.0:
            raise ValueError("force gain must be positive")
        if not 0.0 <= self.tracking_tangent_action <= 1.0:
            raise ValueError("invalid tracking tangent action")


class EventBalancedExcitationPolicy:
    """Generate repeated near-limit, release, and reacquisition transitions."""

    def __init__(self, program_index: int, config: EventBalancedExcitationConfig | None = None):
        self.config = config or EventBalancedExcitationConfig()
        self.config.validate()
        if not 0 <= int(program_index) < len(EVENT_PROGRAMS):
            raise ValueError("event program index is out of range")
        self.program_index = int(program_index)
        self.program_name, self.normal_sequence = EVENT_PROGRAMS[self.program_index]
        self.acquired = False
        self.steps_after_acquisition = 0

    def _tracking_normal(self, measured_force_n: float) -> float:
        return float(
            np.clip(
                self.config.proportional_gain_action_per_n
                * (self.config.target_force_n - float(measured_force_n)),
                -1.0,
                1.0,
            )
        )

    def act(
        self,
        measured_force_n: float,
        force_rate_n_s: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        force = float(measured_force_n)
        force_rate = float(force_rate_n_s)
        if not np.isfinite(force) or force < 0.0 or not np.isfinite(force_rate):
            raise ValueError("force inputs must be finite and measured force nonnegative")
        if not self.acquired:
            if force >= self.config.acquisition_force_n:
                self.acquired = True
                self.steps_after_acquisition = 0
            else:
                if force >= 7.0 and force_rate >= 80.0:
                    normal = -1.0
                    phase = "approach_brake"
                elif force >= 7.0:
                    normal = 0.10
                    phase = "approach_fine"
                elif force >= 3.0:
                    normal = 0.25
                    phase = "approach_coarse"
                else:
                    normal = 1.00
                    phase = "approach"
                return np.asarray((0.0, 0.0, normal), dtype=np.float32), {
                    "phase": phase,
                    "program": self.program_name,
                    "event_step": None,
                }
        elapsed = self.steps_after_acquisition
        self.steps_after_acquisition += 1
        if elapsed < self.config.settle_steps:
            normal = -1.0 if force_rate >= 80.0 else min(0.20, self._tracking_normal(force))
            return np.asarray((0.0, 0.0, normal), dtype=np.float32), {
                "phase": "settle",
                "program": self.program_name,
                "event_step": None,
            }
        cycle = (elapsed - self.config.settle_steps) % self.config.event_period_steps
        if cycle < len(self.normal_sequence):
            tangent = 1.0 if self.program_name == "tangential_press" else 0.25
            return np.asarray((tangent, 0.0, self.normal_sequence[cycle]), dtype=np.float32), {
                "phase": "event",
                "program": self.program_name,
                "event_step": int(cycle),
            }
        return np.asarray(
            (
                self.config.tracking_tangent_action,
                0.0,
                self._tracking_normal(force),
            ),
            dtype=np.float32,
        ), {
            "phase": "tracking_recovery",
            "program": self.program_name,
            "event_step": None,
        }
