"""Causal, traceable learner-visible deployment-gap perturbations."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class DeploymentGapConfig:
    force_noise_rms_n: float = 0.0
    force_bias_n: float = 0.0
    observation_delay_samples: int = 0
    tcp_pose_error_magnitude_m: float = 0.0
    surface_normal_error_magnitude_deg: float = 0.0
    random_seed: int = 0

    def validate(self) -> None:
        numeric = (
            self.force_noise_rms_n,
            self.force_bias_n,
            self.tcp_pose_error_magnitude_m,
            self.surface_normal_error_magnitude_deg,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ValueError("deployment-gap values must be finite")
        if self.force_noise_rms_n < 0 or self.tcp_pose_error_magnitude_m < 0:
            raise ValueError("noise/error magnitudes must be nonnegative")
        if self.surface_normal_error_magnitude_deg < 0:
            raise ValueError("normal-error magnitude must be nonnegative")
        if int(self.observation_delay_samples) != self.observation_delay_samples:
            raise ValueError("observation delay must be an integer")
        if not 0 <= int(self.observation_delay_samples) <= 5:
            raise ValueError("observation delay is outside the frozen range")
        active = sum((
            self.force_noise_rms_n > 0,
            self.force_bias_n != 0,
            self.observation_delay_samples > 0,
            self.tcp_pose_error_magnitude_m > 0,
            self.surface_normal_error_magnitude_deg > 0,
        ))
        if active > 1:
            raise ValueError("deployment-gap experiment is one-factor-at-a-time")

    @property
    def factor(self) -> str:
        if self.force_noise_rms_n > 0:
            return "force_white_noise"
        if self.force_bias_n != 0:
            return "episode_force_bias"
        if self.observation_delay_samples > 0:
            return "observation_delay"
        if self.tcp_pose_error_magnitude_m > 0:
            return "tcp_pose_error"
        if self.surface_normal_error_magnitude_deg > 0:
            return "surface_normal_error"
        return "nominal"


@dataclass(frozen=True)
class DeploymentGapCondition:
    condition_id: str
    config: DeploymentGapConfig


def deployment_gap_conditions(block_id: str, *, random_seed: int) -> tuple[DeploymentGapCondition, ...]:
    """Return the frozen non-nominal condition roster for one geometry block."""
    complete = str(block_id) == "B0"
    if str(block_id) not in {"B0", "S1", "S2"}:
        raise ValueError("deployment-gap block must be B0, S1, or S2")
    rows = [
        DeploymentGapCondition(
            "force_noise_rms_0p035n",
            DeploymentGapConfig(force_noise_rms_n=0.035, random_seed=random_seed),
        ),
        DeploymentGapCondition(
            "force_bias_m0p8n",
            DeploymentGapConfig(force_bias_n=-0.8, random_seed=random_seed),
        ),
        DeploymentGapCondition(
            "force_bias_p0p8n",
            DeploymentGapConfig(force_bias_n=0.8, random_seed=random_seed),
        ),
        DeploymentGapCondition(
            "delay_5samples",
            DeploymentGapConfig(observation_delay_samples=5, random_seed=random_seed),
        ),
        DeploymentGapCondition(
            "tcp_pose_error_0p1mm",
            DeploymentGapConfig(tcp_pose_error_magnitude_m=0.0001, random_seed=random_seed),
        ),
        DeploymentGapCondition(
            "normal_error_5deg",
            DeploymentGapConfig(surface_normal_error_magnitude_deg=5.0, random_seed=random_seed),
        ),
    ]
    if complete:
        rows.extend((
            DeploymentGapCondition(
                "force_bias_m0p4n",
                DeploymentGapConfig(force_bias_n=-0.4, random_seed=random_seed),
            ),
            DeploymentGapCondition(
                "force_bias_p0p4n",
                DeploymentGapConfig(force_bias_n=0.4, random_seed=random_seed),
            ),
            DeploymentGapCondition(
                "delay_1sample",
                DeploymentGapConfig(observation_delay_samples=1, random_seed=random_seed),
            ),
            DeploymentGapCondition(
                "delay_2samples",
                DeploymentGapConfig(observation_delay_samples=2, random_seed=random_seed),
            ),
            DeploymentGapCondition(
                "tcp_pose_error_0p05mm",
                DeploymentGapConfig(tcp_pose_error_magnitude_m=0.00005, random_seed=random_seed),
            ),
            DeploymentGapCondition(
                "normal_error_1deg",
                DeploymentGapConfig(surface_normal_error_magnitude_deg=1.0, random_seed=random_seed),
            ),
            DeploymentGapCondition(
                "normal_error_3deg",
                DeploymentGapConfig(surface_normal_error_magnitude_deg=3.0, random_seed=random_seed),
            ),
        ))
    result = tuple(sorted(rows, key=lambda row: row.condition_id))
    if len({row.condition_id for row in result}) != len(result):
        raise RuntimeError("deployment-gap condition ids are not unique")
    return result


def deterministic_sign(seed: int, *, stream: int) -> float:
    sequence = np.random.SeedSequence([int(seed) & 0xFFFFFFFF, int(stream)])
    value = int(np.random.default_rng(sequence).integers(0, 2))
    return -1.0 if value == 0 else 1.0


def rotated_task_frame(
    tangent: np.ndarray,
    outward: np.ndarray,
    *,
    magnitude_deg: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    tangent = np.asarray(tangent, dtype=float).reshape(3)
    outward = np.asarray(outward, dtype=float).reshape(3)
    tangent /= np.linalg.norm(tangent)
    outward /= np.linalg.norm(outward)
    sign = deterministic_sign(seed, stream=7001)
    angle = sign * math.radians(float(magnitude_deg))
    biased_outward = math.cos(angle) * outward + math.sin(angle) * tangent
    biased_tangent = math.cos(angle) * tangent - math.sin(angle) * outward
    return biased_tangent, biased_outward, math.degrees(angle)


class LearnerObservationPerturbation:
    """Transform current observations while retaining auditable intermediate values."""

    def __init__(
        self,
        gap: DeploymentGapConfig,
        *,
        control_rate_hz: float = 100.0,
        cross_track_normalization_m: float = 0.02,
        normal_offset_normalization_m: float = 0.02,
    ):
        gap.validate()
        self.gap = gap
        self.control_rate_hz = float(control_rate_hz)
        self.cross_track_normalization_m = float(cross_track_normalization_m)
        self.normal_offset_normalization_m = float(normal_offset_normalization_m)
        self._previous_measured_force: float | None = None
        self._delay_queue: list[np.ndarray] = []
        direction_rng = np.random.default_rng(np.random.SeedSequence([
            int(gap.random_seed) & 0xFFFFFFFF, 8001
        ]))
        direction = direction_rng.normal(size=2)
        direction /= np.linalg.norm(direction)
        self._tcp_bias_cross_normal_m = direction * gap.tcp_pose_error_magnitude_m

    def _force_noise(self, step: int) -> float:
        if self.gap.force_noise_rms_n == 0:
            return 0.0
        sequence = np.random.SeedSequence([
            int(self.gap.random_seed) & 0xFFFFFFFF, int(step), 9001
        ])
        return float(np.random.default_rng(sequence).normal(
            loc=0.0, scale=self.gap.force_noise_rms_n
        ))

    def transform(
        self,
        native_observation: np.ndarray,
        *,
        native_force_n: float,
        step: int,
        initial: bool = False,
    ) -> tuple[np.ndarray, dict]:
        native = np.asarray(native_observation, dtype=np.float32).reshape(-1)
        if native.shape != (16,) or not np.all(np.isfinite(native)):
            raise ValueError("native learner observation must be one finite 16-D vector")
        current = native.copy()
        sampled_force_noise = self._force_noise(step)
        measured_force = max(
            0.0,
            float(native_force_n) + self.gap.force_bias_n + sampled_force_noise,
        )
        if self.gap.force_noise_rms_n > 0 or self.gap.force_bias_n != 0:
            previous = measured_force if self._previous_measured_force is None else self._previous_measured_force
            rate = 0.0 if initial else (measured_force - previous) * self.control_rate_hz
            current[0] = np.clip(measured_force / 15.0, 0.0, 2.0)
            current[2] = np.clip(rate / 300.0, -1.0, 1.0)
            current[13] = float(measured_force >= 3.0)
            self._previous_measured_force = measured_force
        if self.gap.tcp_pose_error_magnitude_m > 0:
            cross_bias, normal_bias = self._tcp_bias_cross_normal_m
            current[4] = np.clip(
                current[4] + cross_bias / self.cross_track_normalization_m,
                -2.0,
                2.0,
            )
            current[5] = np.clip(
                current[5] + normal_bias / self.normal_offset_normalization_m,
                -2.0,
                2.0,
            )
        if initial:
            self._delay_queue = [current.copy()] * int(self.gap.observation_delay_samples)
            visible = current.copy()
        elif self.gap.observation_delay_samples:
            self._delay_queue.append(current.copy())
            visible = self._delay_queue.pop(0)
        else:
            visible = current.copy()
        return visible, {
            "native_observation": native.tolist(),
            "perturbed_current_observation": current.tolist(),
            "learner_visible_observation": visible.tolist(),
            "native_force_n": float(native_force_n),
            "learner_measured_force_n": measured_force,
            "sampled_force_noise_n": sampled_force_noise,
            "applied_force_noise_n": (
                measured_force - float(native_force_n) - self.gap.force_bias_n
            ),
            "force_bias_n": float(self.gap.force_bias_n),
            "delay_samples": int(self.gap.observation_delay_samples),
            "tcp_bias_cross_normal_m": self._tcp_bias_cross_normal_m.tolist(),
        }
