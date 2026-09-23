"""Precision TRACK r3 with force-ready recovery handoff.

The completed r2 development trajectory showed five rapid
``RECOVERY_ACQUIRE -> CONTACT_VERIFY -> TRACK -> RECOVERY_ACQUIRE`` loops.
Every handoff occurred near 3 N although the requested force was 5 N.  This
revision changes only the state-machine handoff: approach/recovery must
re-establish 80% of the requested force before contact verification begins.

This threshold is a controller-state hysteresis condition.  It is not an
evaluation tolerance and does not change exact-target error, RMSE, the 15-N
force limit, or any task-success metric.
"""

from __future__ import annotations

from dataclasses import replace

from .controller import SupervisorConfig
from .controller_precision_r2 import V6PrecisionContactSupervisorR2


PRECISION_R3_CONTACT_VERIFY_TARGET_FRACTION = 0.80


class V6PrecisionContactSupervisorR3(V6PrecisionContactSupervisorR2):
    """R2 exact-target control with a single force-ready handoff change."""

    controller_revision = "V6-precision-track-r3-force-ready-handoff"

    def __init__(self, config: SupervisorConfig | None = None) -> None:
        super().__init__(config)
        self.config = replace(
            self.config,
            contact_verify_target_fraction=(
                PRECISION_R3_CONTACT_VERIFY_TARGET_FRACTION
            ),
        )
        self.config.validate()


__all__ = [
    "PRECISION_R3_CONTACT_VERIFY_TARGET_FRACTION",
    "V6PrecisionContactSupervisorR3",
]
