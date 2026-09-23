"""Frozen adaptive-admittance baseline for the direct ForceWipe interface.

The controller follows a conventional virtual mass--damper--spring normal
admittance and adapts damping from the measured force-rate magnitude.  Path
advance and cross-track correction use only the same causal 16-D observation
available to the learned methods.  There is no external shield or post-hoc
action projection beyond the common actuator box.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


class AdaptiveAdmittanceError(ValueError):
    pass


@dataclass(frozen=True)
class AdaptiveAdmittanceConfig:
    control_rate_hz: float = 100.0
    virtual_mass_kg: float = 1.5
    # A small centering stiffness avoids the large series-stiffness bias that
    # would arise if virtual K were comparable to the 650--1450 N/m tools.
    virtual_stiffness_n_m: float = 50.0
    damping_min_n_s_m: float = 55.0
    damping_max_n_s_m: float = 125.0
    virtual_velocity_limit_m_s: float = 0.05
    virtual_offset_outward_limit_m: float = 0.008
    virtual_offset_inward_limit_m: float = 0.025
    force_rate_scale_n_s: float = 300.0
    contact_threshold_n: float = 3.0
    headroom_threshold_n: float = 14.0
    acquisition_action: float = 0.55
    tangent_action: float = 0.85
    cross_track_gain: float = 0.50
    inward_action_scale_m: float = 0.0005

    def validate(self) -> None:
        values = (
            self.control_rate_hz, self.virtual_mass_kg, self.virtual_stiffness_n_m,
            self.damping_min_n_s_m, self.damping_max_n_s_m,
            self.virtual_velocity_limit_m_s, self.virtual_offset_outward_limit_m,
            self.virtual_offset_inward_limit_m, self.force_rate_scale_n_s,
            self.contact_threshold_n, self.headroom_threshold_n,
            self.acquisition_action,
            self.tangent_action, self.cross_track_gain, self.inward_action_scale_m,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0 for value in values):
            raise AdaptiveAdmittanceError("admittance parameters must be finite and positive")
        if self.damping_max_n_s_m < self.damping_min_n_s_m:
            raise AdaptiveAdmittanceError("adaptive damping limits are reversed")
        if self.contact_threshold_n >= self.headroom_threshold_n:
            raise AdaptiveAdmittanceError("contact threshold must be below headroom")
        if self.headroom_threshold_n >= 15.0:
            raise AdaptiveAdmittanceError("headroom threshold must be below 15 N")
        if self.tangent_action > 1.0 or self.acquisition_action > 1.0:
            raise AdaptiveAdmittanceError("declared action exceeds the common action box")


class AdaptiveAdmittanceController:
    """Stateful causal baseline with a frozen gain schedule."""

    def __init__(self, config: AdaptiveAdmittanceConfig | None = None) -> None:
        self.config = config or AdaptiveAdmittanceConfig()
        self.config.validate()
        self.reset()

    def reset(self) -> None:
        self.virtual_offset_m = 0.0
        self.virtual_velocity_m_s = 0.0
        self.contact_established = False
        self._last_control_mode = "adaptive_admittance"

    def act(self, observation) -> np.ndarray:
        obs = np.asarray(observation, dtype=np.float64).reshape(-1)
        if obs.shape != (16,) or not np.all(np.isfinite(obs)):
            raise AdaptiveAdmittanceError("observation must be one finite 16-vector")
        cfg = self.config
        dt = 1.0 / cfg.control_rate_hz
        force_n = float(obs[0] * 15.0)
        target_n = float(obs[1] * 15.0)
        force_rate_n_s = float(obs[2] * cfg.force_rate_scale_n_s)
        cross_track_normalized = float(obs[4])

        # A force-controlled admittance requires a contact establishment
        # phase when initialized from hover.  This is a fixed controller mode,
        # not a learned policy or external shield.  The admittance reference is
        # initialized at the first measured-contact sample.
        if force_n < cfg.contact_threshold_n:
            self._last_control_mode = (
                "adaptive_admittance_acquire"
                if not self.contact_established
                else "adaptive_admittance_reacquire"
            )
            self.virtual_offset_m = 0.0
            self.virtual_velocity_m_s = 0.0
            return np.asarray(
                [0.0, 0.0, cfg.acquisition_action], dtype=np.float32
            )
        if not self.contact_established:
            self.contact_established = True
            self.virtual_offset_m = 0.0
            self.virtual_velocity_m_s = 0.0
        self._last_control_mode = "adaptive_admittance_track"

        damping_fraction = min(abs(force_rate_n_s) / cfg.force_rate_scale_n_s, 1.0)
        damping = cfg.damping_min_n_s_m + damping_fraction * (
            cfg.damping_max_n_s_m - cfg.damping_min_n_s_m
        )
        force_error = target_n - force_n
        acceleration = (
            force_error
            - damping * self.virtual_velocity_m_s
            - cfg.virtual_stiffness_n_m * self.virtual_offset_m
        ) / cfg.virtual_mass_kg
        velocity = float(np.clip(
            self.virtual_velocity_m_s + acceleration * dt,
            -cfg.virtual_velocity_limit_m_s,
            cfg.virtual_velocity_limit_m_s,
        ))
        candidate_offset = float(np.clip(
            self.virtual_offset_m + velocity * dt,
            -cfg.virtual_offset_outward_limit_m,
            cfg.virtual_offset_inward_limit_m,
        ))
        normal_action = float(np.clip(
            (candidate_offset - self.virtual_offset_m) / cfg.inward_action_scale_m,
            -1.0,
            1.0,
        ))

        # A measured-headroom response is part of this declared controller,
        # not a separate safety layer.  The common actuator still executes the
        # returned normalized command without force-dependent projection.
        if force_n >= cfg.headroom_threshold_n:
            self._last_control_mode = "adaptive_admittance_headroom"
            normal_action = -1.0
            velocity = -cfg.inward_action_scale_m / dt
            candidate_offset = max(
                self.virtual_offset_m - cfg.inward_action_scale_m,
                -cfg.virtual_offset_outward_limit_m,
            )

        executed_delta = normal_action * cfg.inward_action_scale_m
        self.virtual_offset_m = float(np.clip(
            self.virtual_offset_m + executed_delta,
            -cfg.virtual_offset_outward_limit_m,
            cfg.virtual_offset_inward_limit_m,
        ))
        self.virtual_velocity_m_s = executed_delta / dt

        contact_for_motion = cfg.contact_threshold_n <= force_n < cfg.headroom_threshold_n
        tangent = cfg.tangent_action if contact_for_motion else 0.0
        cross = float(np.clip(-cfg.cross_track_gain * cross_track_normalized, -1.0, 1.0))
        if not contact_for_motion:
            cross = 0.0
        return np.asarray([tangent, cross, normal_action], dtype=np.float32)
