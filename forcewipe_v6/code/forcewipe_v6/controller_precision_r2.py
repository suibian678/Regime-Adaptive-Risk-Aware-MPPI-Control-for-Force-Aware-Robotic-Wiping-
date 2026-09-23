"""Precision TRACK r2 with continuous low-force task-speed scheduling."""

from __future__ import annotations

from dataclasses import replace

from .controller import SupervisorCommand, SupervisorInput
from .controller_precision import V6PrecisionContactSupervisor


def continuous_task_force_scale(
    *, force_n: float, target_n: float, contact_floor_n: float, force_limit_n: float
) -> float:
    """Continuous zero-to-one task scale with no acceptance band."""

    force = float(force_n)
    target = float(target_n)
    floor = float(contact_floor_n)
    limit = float(force_limit_n)
    if force <= floor:
        return 0.0
    if force <= target:
        ratio = (force - floor) / (target - floor)
    else:
        ratio = (limit - force) / (limit - target)
    ratio = max(0.0, min(1.0, ratio))
    return ratio * ratio


class V6PrecisionContactSupervisorR2(V6PrecisionContactSupervisor):
    """Preserve exact-target normal control while slowing low-force traversal."""

    controller_revision = "V6-precision-track-r2-continuous-task-scheduling"

    def command(self, observation: SupervisorInput) -> SupervisorCommand:
        candidate = super().command(observation)
        if (
            candidate.state_before == "TRACK"
            and candidate.state_after == "TRACK"
            and not candidate.transitioned
            and candidate.transition_reason.startswith("exact_target_")
            and candidate.tangential_motion_permitted
        ):
            scale = continuous_task_force_scale(
                force_n=float(observation.measured_force_n),
                target_n=float(observation.target_force_n),
                contact_floor_n=self.config.contact_loss_floor_n,
                force_limit_n=self.config.force_limit_n,
            )
            return replace(
                candidate,
                transition_reason=f"{candidate.transition_reason}_continuous_task_scale",
                executed_tangential_step_m=(
                    float(observation.requested_tangential_step_m) * scale
                ),
            )
        return candidate


__all__ = ["V6PrecisionContactSupervisorR2", "continuous_task_force_scale"]
