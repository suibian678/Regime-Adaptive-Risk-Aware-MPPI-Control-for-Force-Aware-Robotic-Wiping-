"""Raw-safety / filtered-tracking force signal split."""

from __future__ import annotations

from typing import Callable


class FilteredTrackingNormalController:
    """Use raw force for safety/contact transitions and an EWMA in TRACK."""

    def __init__(
        self,
        base_controller: Callable[..., tuple[float, dict]],
        *,
        alpha: float = 0.25,
        contact_threshold_n: float = 3.0,
        headroom_enter_n: float = 13.5,
    ):
        if not 0.0 < float(alpha) <= 1.0:
            raise ValueError("filter alpha must lie in (0, 1]")
        self.base_controller = base_controller
        self.alpha = float(alpha)
        self.contact_threshold_n = float(contact_threshold_n)
        self.headroom_enter_n = float(headroom_enter_n)
        self.reset()

    def reset(self) -> None:
        self._filtered_force_n: float | None = None

    def __call__(
        self,
        *,
        target_force_n: float,
        measured_force_n: float,
        previous_force_n: float,
        target_lead_down_m: float,
    ) -> tuple[float, dict]:
        raw = float(measured_force_n)
        if self._filtered_force_n is None:
            previous_filtered = raw
            filtered = raw
        else:
            previous_filtered = self._filtered_force_n
            filtered = self.alpha * raw + (1.0 - self.alpha) * previous_filtered
        self._filtered_force_n = filtered

        raw_transition = raw < self.contact_threshold_n or raw >= self.headroom_enter_n
        filtered_transition = filtered < self.contact_threshold_n
        if raw_transition or filtered_transition:
            control_force = raw
            control_previous = float(previous_force_n)
            source = "raw_transition"
        else:
            control_force = filtered
            control_previous = previous_filtered
            source = "filtered_track"
        increment, detail = self.base_controller(
            target_force_n=target_force_n,
            measured_force_n=control_force,
            previous_force_n=control_previous,
            target_lead_down_m=target_lead_down_m,
        )
        updated = dict(detail)
        updated.update(
            {
                "raw_measured_force_n": raw,
                "filtered_tracking_force_n": filtered,
                "tracking_force_source": source,
            }
        )
        return float(increment), updated


__all__ = ["FilteredTrackingNormalController"]
