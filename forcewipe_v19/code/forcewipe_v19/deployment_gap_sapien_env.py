"""SAPIEN binding for one-factor learner-visible deployment-gap stresses."""

from __future__ import annotations

from typing import Any

import numpy as np

from forcewipe_v6.tdmpc2_direct_sapien_env import V6DirectFirstPassEnv, _scalar, _vector

from .deployment_gap import (
    DeploymentGapConfig,
    LearnerObservationPerturbation,
    rotated_task_frame,
)


class V19DeploymentGapEnv(V6DirectFirstPassEnv):
    """Direct environment with a traceable perturbation outside outcome metrics."""

    def __init__(self, *, gap_config: DeploymentGapConfig, **kwargs: Any) -> None:
        gap_config.validate()
        self.gap_config = gap_config
        self._observation_perturbation: LearnerObservationPerturbation | None = None
        self._last_gap_log: dict[str, Any] | None = None
        self._signed_normal_error_deg = 0.0
        super().__init__(**kwargs)

    def _geometry(self) -> dict[str, Any]:
        geometry = super()._geometry()
        magnitude = self.gap_config.surface_normal_error_magnitude_deg
        if magnitude == 0:
            self._signed_normal_error_deg = 0.0
            return geometry
        tangent, outward, signed_deg = rotated_task_frame(
            geometry["tangent_world"],
            geometry["outward_normal_world"],
            magnitude_deg=magnitude,
            seed=self.gap_config.random_seed,
        )
        cross = np.cross(outward, tangent)
        cross /= np.linalg.norm(cross)
        velocity = _vector(self._raw.v4_tool.linear_velocity, name="tool velocity")
        angle = np.deg2rad(signed_deg)
        geometry.update({
            "tangent_world": tangent,
            "outward_normal_world": outward,
            "cross_world": cross,
            "signed_normal_offset_m": (
                float(geometry["signed_normal_offset_m"]) * float(np.cos(angle))
            ),
            "tangent_velocity_m_s": float(np.dot(velocity, tangent)),
            "cross_track_velocity_m_s": float(np.dot(velocity, cross)),
            "outward_normal_velocity_m_s": float(np.dot(velocity, outward)),
        })
        self._signed_normal_error_deg = float(signed_deg)
        return geometry

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        native_observation, info = super().reset(seed=seed, options=options)
        self._observation_perturbation = LearnerObservationPerturbation(
            self.gap_config,
            control_rate_hz=self.config.control_rate_hz,
            cross_track_normalization_m=self.config.cross_track_normalization_m,
            normal_offset_normalization_m=self.config.normal_offset_normalization_m,
        )
        native_force = _scalar(self._raw._normal_force(), name="initial native force")
        visible, gap_log = self._observation_perturbation.transform(
            native_observation,
            native_force_n=native_force,
            step=0,
            initial=True,
        )
        gap_log.update({
            "factor": self.gap_config.factor,
            "signed_surface_normal_error_deg": self._signed_normal_error_deg,
        })
        self._last_gap_log = gap_log
        info.update({
            "deployment_gap_factor": self.gap_config.factor,
            "deployment_gap": gap_log,
        })
        return visible, info

    def step(self, action: object):
        if self._observation_perturbation is None:
            raise RuntimeError("reset must be called before deployment-gap step")
        native_observation, reward, terminated, truncated, info = super().step(action)
        visible, gap_log = self._observation_perturbation.transform(
            native_observation,
            native_force_n=float(info["normal_force_n"]),
            step=self._elapsed_steps,
            initial=False,
        )
        gap_log.update({
            "factor": self.gap_config.factor,
            "signed_surface_normal_error_deg": self._signed_normal_error_deg,
            "policy_action": list(info["tdmpc2_action"]),
            "issued_physical_action": list(info["physical_action"]),
        })
        self._last_gap_log = gap_log
        info.update({
            "deployment_gap_factor": self.gap_config.factor,
            "deployment_gap": gap_log,
        })
        return visible, reward, terminated, truncated, info

