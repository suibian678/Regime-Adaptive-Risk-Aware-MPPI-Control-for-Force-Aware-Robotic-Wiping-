"""Development and held-out motions for the revised cross-simulator port."""

from __future__ import annotations

from forcewipe_v19.crosssim_calibration import CalibrationMotion


ACTUATOR_FIT_MOTIONS = (
    CalibrationMotion(
        "fit_free_space_normal_return",
        "actuator_fit",
        (
            (20, (0.0, 0.0, -1.0)),
            (12, (0.0, 0.0, 0.0)),
            (20, (0.0, 0.0, 1.0)),
            (12, (0.0, 0.0, 0.0)),
        ),
    ),
    CalibrationMotion(
        "fit_free_space_tangent_reversal",
        "actuator_fit",
        (
            (24, (0.5, 0.0, 0.0)),
            (12, (0.0, 0.0, 0.0)),
            (24, (-0.5, 0.0, 0.0)),
            (12, (0.0, 0.0, 0.0)),
        ),
    ),
    CalibrationMotion(
        "fit_free_space_cross_reversal",
        "actuator_fit",
        (
            (24, (0.0, 0.5, 0.0)),
            (12, (0.0, 0.0, 0.0)),
            (24, (0.0, -0.5, 0.0)),
            (12, (0.0, 0.0, 0.0)),
        ),
    ),
)


# These motions are not executed during actuator/contact development.  They
# become final validation only after the port parameters and protocol hash are
# frozen.
HELD_OUT_MOTIONS = (
    CalibrationMotion(
        "validation_multilevel_normal_cycle",
        "validation",
        (
            (218, (0.0, 0.0, 1.0)),
            (20, (0.0, 0.0, 0.0)),
            (10, (0.0, 0.0, -0.4)),
            (20, (0.0, 0.0, 0.0)),
            (8, (0.0, 0.0, 0.5)),
            (20, (0.0, 0.0, 0.0)),
            (20, (0.0, 0.0, -1.0)),
        ),
    ),
    CalibrationMotion(
        "validation_bidirectional_tangent_release",
        "validation",
        (
            (220, (0.0, 0.0, 1.0)),
            (20, (0.0, 0.0, 0.0)),
            (50, (0.15, 0.0, 0.0)),
            (50, (-0.15, 0.0, 0.0)),
            (20, (0.0, 0.0, 0.0)),
            (20, (0.0, 0.0, -1.0)),
        ),
    ),
)

