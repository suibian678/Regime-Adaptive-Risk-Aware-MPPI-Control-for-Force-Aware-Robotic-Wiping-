"""Bounded stateless outer-loop target bias for contact-path tracking."""

from __future__ import annotations

import math

import numpy as np


class CrossTrackCompensationError(ValueError):
    pass


def bounded_cross_track_target_bias(
    cross_track_error_world_m: np.ndarray,
    *,
    gain: float,
    maximum_bias_m: float,
) -> np.ndarray:
    """Return a bounded target bias opposite the measured cross-track error."""

    error = np.asarray(cross_track_error_world_m, dtype=np.float64)
    gain_value = float(gain)
    bound = float(maximum_bias_m)
    if error.shape != (3,) or not np.all(np.isfinite(error)):
        raise CrossTrackCompensationError("cross-track error must be a finite three-vector")
    if not math.isfinite(gain_value) or gain_value < 0.0:
        raise CrossTrackCompensationError("cross-track gain must be finite and nonnegative")
    if not math.isfinite(bound) or bound <= 0.0:
        raise CrossTrackCompensationError("cross-track bias bound must be positive and finite")
    bias = -gain_value * error
    norm = float(np.linalg.norm(bias))
    if norm > bound:
        bias *= bound / norm
    return bias


__all__ = ["CrossTrackCompensationError", "bounded_cross_track_target_bias"]
