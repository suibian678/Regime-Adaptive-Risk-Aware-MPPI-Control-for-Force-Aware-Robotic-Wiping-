"""Exact-target force servo for bounded code-only development.

The component is intentionally independent of the historical ``Band@25%``
metric.  It regulates the signed force error continuously, applies one-sided
rate damping, projects inward motion against a causal force envelope, and
uses the final executed command for anti-windup.  Tangential motion is scaled
continuously from physical force ratios instead of being enabled by a binary
tracking-tolerance band.

Positive normal motion is inward.  This module is SAPIEN-independent and does
not by itself establish physical tracking or safety performance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math


class PrecisionTrackingError(ValueError):
    """Raised when the exact-target servo receives an invalid state or input."""


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _clip(value: float, lower: float, upper: float) -> float:
    return max(float(lower), min(float(upper), float(value)))


@dataclass(frozen=True)
class PrecisionTrackingConfig:
    dt_s: float = 0.01
    force_limit_n: float = 15.0
    safety_projection_bound_n: float = 14.5
    contact_floor_n: float = 3.0
    max_inward_step_m: float = 0.010
    max_outward_step_m: float = 0.0035
    headroom_outward_step_m: float = 0.0010
    kp_m_per_n: float = 0.0008
    ki_m_per_n_s: float = 0.0012
    kd_m_s_per_n: float = 0.000006
    integral_clip_n_s: float = 15.0
    envelope_incremental_press_gain_n_per_m: float = 500.0

    def validate(self) -> None:
        names = tuple(self.__dataclass_fields__)
        for name in names:
            value = getattr(self, name)
            if not _finite(value) or float(value) <= 0.0:
                raise PrecisionTrackingError(f"{name} must be finite and positive")
        if not (
            self.contact_floor_n
            < self.safety_projection_bound_n
            < self.force_limit_n
        ):
            raise PrecisionTrackingError("force thresholds must be strictly ordered")
        if self.headroom_outward_step_m > self.max_outward_step_m:
            raise PrecisionTrackingError(
                "headroom outward step must not exceed the outward actuator limit"
            )

    def as_dict(self) -> dict[str, float]:
        self.validate()
        return asdict(self)


@dataclass(frozen=True)
class PrecisionTrackingState:
    integral_error_n_s: float = 0.0
    previous_executed_normal_step_m: float = 0.0


@dataclass(frozen=True)
class PrecisionTrackingInput:
    target_force_n: float
    measured_force_n: float
    force_rate_n_s: float
    envelope_base_upper_n: float
    contact_observed: bool
    geometric_track_ready: bool
    requested_tangential_step_m: float


@dataclass(frozen=True)
class PrecisionTrackingCommand:
    reason: str
    servo_active: bool
    target_force_n: float
    measured_force_n: float
    signed_error_n: float
    normalized_error: float
    p_step_m: float
    i_step_m: float
    one_sided_d_step_m: float
    raw_normal_step_m: float
    actuator_limited_normal_step_m: float
    safe_press_limit_m: float
    projection_reference_m: float
    safe_inward_increment_m: float
    executed_normal_step_m: float
    actuator_saturation_active: bool
    safety_projection_active: bool
    headroom_preempted: bool
    integral_before_n_s: float
    integral_candidate_n_s: float
    integral_after_n_s: float
    tangential_force_scale: float
    executed_tangential_step_m: float


class ExactTargetForceServo:
    """Continuous exact-target PI-D servo with a separate safety projection."""

    controller_revision = "exact-target-force-servo-r1-code-only"

    def __init__(self, config: PrecisionTrackingConfig | None = None):
        self.config = config or PrecisionTrackingConfig()
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        self._integral_error_n_s = 0.0
        self._previous_executed_normal_step_m = 0.0

    def snapshot(self) -> PrecisionTrackingState:
        return PrecisionTrackingState(
            integral_error_n_s=float(self._integral_error_n_s),
            previous_executed_normal_step_m=float(
                self._previous_executed_normal_step_m
            ),
        )

    def restore(self, state: PrecisionTrackingState) -> None:
        if not isinstance(state, PrecisionTrackingState):
            raise PrecisionTrackingError("restore requires PrecisionTrackingState")
        if not _finite(state.integral_error_n_s) or not _finite(
            state.previous_executed_normal_step_m
        ):
            raise PrecisionTrackingError("precision-tracking state must be finite")
        if abs(state.integral_error_n_s) > self.config.integral_clip_n_s + 1e-12:
            raise PrecisionTrackingError("integral state exceeds its configured clip")
        if not (
            -self.config.max_outward_step_m - 1e-12
            <= state.previous_executed_normal_step_m
            <= self.config.max_inward_step_m + 1e-12
        ):
            raise PrecisionTrackingError("previous command exceeds actuator bounds")
        self._integral_error_n_s = float(state.integral_error_n_s)
        self._previous_executed_normal_step_m = float(
            state.previous_executed_normal_step_m
        )

    def _validate_input(self, item: PrecisionTrackingInput) -> None:
        if not isinstance(item, PrecisionTrackingInput):
            raise PrecisionTrackingError("step requires PrecisionTrackingInput")
        for name in (
            "target_force_n",
            "measured_force_n",
            "force_rate_n_s",
            "envelope_base_upper_n",
            "requested_tangential_step_m",
        ):
            if not _finite(getattr(item, name)):
                raise PrecisionTrackingError(f"{name} must be finite")
        if item.target_force_n <= 0.0:
            raise PrecisionTrackingError("target force must be positive")
        if item.target_force_n >= self.config.force_limit_n:
            raise PrecisionTrackingError("target force must remain below the hard limit")
        if item.measured_force_n < 0.0 or item.envelope_base_upper_n < 0.0:
            raise PrecisionTrackingError("force and envelope must be nonnegative")
        if not isinstance(item.contact_observed, bool) or not isinstance(
            item.geometric_track_ready, bool
        ):
            raise PrecisionTrackingError("contact and geometry flags must be bool")

    def _tangential_scale(self, *, force: float, target: float) -> float:
        if force < self.config.contact_floor_n:
            return 0.0
        low_force_scale = _clip(force / target, 0.0, 1.0)
        high_force_scale = _clip(
            (self.config.force_limit_n - force)
            / (self.config.force_limit_n - target),
            0.0,
            1.0,
        )
        return min(low_force_scale, high_force_scale)

    def step(self, item: PrecisionTrackingInput) -> PrecisionTrackingCommand:
        self._validate_input(item)
        cfg = self.config
        target = float(item.target_force_n)
        force = float(item.measured_force_n)
        error = target - force
        normalized_error = error / target
        integral_before = float(self._integral_error_n_s)
        integral_candidate = integral_before
        p_step = i_step = d_step = raw = actuator_limited = executed = 0.0
        safe_press_limit = cfg.max_inward_step_m
        projection_reference = 0.0
        safe_inward_increment = cfg.max_inward_step_m
        actuator_saturation = False
        safety_projection = False
        headroom = False
        tangent_scale = 0.0
        reason = "inactive"

        servo_active = bool(
            item.contact_observed
            and item.geometric_track_ready
            and force >= cfg.contact_floor_n
            and force <= cfg.force_limit_n
        )

        if force > cfg.force_limit_n:
            reason = "sampled_force_limit_violation_outward"
            executed = -cfg.max_outward_step_m
            self._integral_error_n_s = 0.0
        elif item.envelope_base_upper_n >= cfg.safety_projection_bound_n:
            reason = "predicted_headroom_outward_preemption"
            headroom = True
            safety_projection = True
            executed = -cfg.headroom_outward_step_m
            self._integral_error_n_s = 0.0
        elif not item.contact_observed or force < cfg.contact_floor_n:
            reason = "contact_not_established"
            self._integral_error_n_s = 0.0
        elif not item.geometric_track_ready:
            reason = "geometry_not_ready_integrator_frozen"
        else:
            integral_candidate = _clip(
                integral_before + error * cfg.dt_s,
                -cfg.integral_clip_n_s,
                cfg.integral_clip_n_s,
            )
            p_step = cfg.kp_m_per_n * error
            i_step = cfg.ki_m_per_n_s * integral_candidate
            d_step = -cfg.kd_m_s_per_n * max(float(item.force_rate_n_s), 0.0)
            raw = p_step + i_step + d_step
            actuator_limited = _clip(
                raw,
                -cfg.max_outward_step_m,
                cfg.max_inward_step_m,
            )
            actuator_saturation = abs(actuator_limited - raw) > 1e-15
            # The causal envelope already contains the force, force-rate,
            # normal-velocity, and command-history contribution accumulated
            # before this action.  Therefore the candidate term must bound
            # only an *increase* over the previously executed inward command.
            # Charging the complete holding command again creates a structural
            # steady-state error (most visibly at 12 N).
            projection_reference = max(
                0.0,
                min(
                    cfg.max_inward_step_m,
                    float(self._previous_executed_normal_step_m),
                ),
            )
            safe_inward_increment = min(
                cfg.max_inward_step_m,
                max(
                    0.0,
                    (
                        cfg.safety_projection_bound_n
                        - float(item.envelope_base_upper_n)
                    )
                    / cfg.envelope_incremental_press_gain_n_per_m,
                ),
            )
            safe_press_limit = min(
                cfg.max_inward_step_m,
                projection_reference + safe_inward_increment,
            )
            executed = min(actuator_limited, safe_press_limit)
            safety_projection = executed < actuator_limited - 1e-15

            # Conditional integration uses the final executed command.  The
            # integrator may unwind through a limit but may not accumulate in
            # the same direction as a saturated or safety-projected request.
            limited_in_error_direction = bool(
                (error > 0.0 and executed < raw - 1e-15)
                or (error < 0.0 and executed > raw + 1e-15)
            )
            self._integral_error_n_s = (
                integral_before if limited_in_error_direction else integral_candidate
            )
            reason = (
                "exact_target_final_command_limited"
                if actuator_saturation or safety_projection
                else "exact_target_pi_d"
            )
            tangent_scale = self._tangential_scale(force=force, target=target)

        executed_tangent = (
            float(item.requested_tangential_step_m) * tangent_scale
            if servo_active and not headroom
            else 0.0
        )
        self._previous_executed_normal_step_m = float(executed)
        return PrecisionTrackingCommand(
            reason=reason,
            servo_active=servo_active and not headroom,
            target_force_n=target,
            measured_force_n=force,
            signed_error_n=error,
            normalized_error=normalized_error,
            p_step_m=p_step,
            i_step_m=i_step,
            one_sided_d_step_m=d_step,
            raw_normal_step_m=raw,
            actuator_limited_normal_step_m=actuator_limited,
            safe_press_limit_m=safe_press_limit,
            projection_reference_m=projection_reference,
            safe_inward_increment_m=safe_inward_increment,
            executed_normal_step_m=float(executed),
            actuator_saturation_active=actuator_saturation,
            safety_projection_active=safety_projection,
            headroom_preempted=headroom,
            integral_before_n_s=integral_before,
            integral_candidate_n_s=integral_candidate,
            integral_after_n_s=float(self._integral_error_n_s),
            tangential_force_scale=float(tangent_scale),
            executed_tangential_step_m=float(executed_tangent),
        )


__all__ = [
    "ExactTargetForceServo",
    "PrecisionTrackingCommand",
    "PrecisionTrackingConfig",
    "PrecisionTrackingError",
    "PrecisionTrackingInput",
    "PrecisionTrackingState",
]
