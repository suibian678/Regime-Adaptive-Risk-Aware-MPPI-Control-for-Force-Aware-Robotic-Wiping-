"""Observed-progress feedback for bounded target-pose path increments."""

from __future__ import annotations

from dataclasses import dataclass
import math


class AlongTrackPathServoError(ValueError):
    pass


@dataclass(frozen=True)
class AlongTrackPathServoConfig:
    nominal_increment_m: float = 0.0002
    progress_error_gain: float = 0.05
    maximum_increment_m: float = 0.0005

    def validate(self) -> None:
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (
                self.nominal_increment_m,
                self.progress_error_gain,
                self.maximum_increment_m,
            )
        ):
            raise AlongTrackPathServoError("path-servo parameters must be finite and nonnegative")
        if self.nominal_increment_m > self.maximum_increment_m or self.maximum_increment_m <= 0.0:
            raise AlongTrackPathServoError("invalid path-servo increment bounds")


def requested_path_increment(
    *,
    observed_progress: float,
    commanded_progress: float,
    path_length_m: float,
    authority_enabled: bool,
    config: AlongTrackPathServoConfig = AlongTrackPathServoConfig(),
) -> tuple[float, dict]:
    """Return a bounded along-track increment using only prior observations."""

    config.validate()
    observed = float(observed_progress)
    commanded = float(commanded_progress)
    length = float(path_length_m)
    if not all(math.isfinite(value) for value in (observed, commanded, length)):
        raise AlongTrackPathServoError("path-servo state must be finite")
    if not 0.0 <= observed <= 1.0 or not 0.0 <= commanded <= 1.0 or length <= 0.0:
        raise AlongTrackPathServoError("path-servo progress or path length is invalid")
    error_m = (observed - commanded) * length
    raw = config.nominal_increment_m + config.progress_error_gain * error_m
    bounded = min(config.maximum_increment_m, max(0.0, raw)) if authority_enabled else 0.0
    return bounded, {
        "observed_minus_commanded_progress_m": error_m,
        "raw_path_increment_m": raw,
        "path_increment_projection_active": bool(authority_enabled and abs(bounded - raw) > 1.0e-15),
        "path_authority_enabled": bool(authority_enabled),
    }
