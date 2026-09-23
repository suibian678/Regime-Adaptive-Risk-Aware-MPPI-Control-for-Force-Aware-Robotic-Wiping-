"""Nonprivileged proposal and variable-budget lifecycle contracts for V4."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .information_flow import PolicyObservation
from .primitives import PrimitiveProposal


class LifecycleContractError(ValueError):
    """Invalid lifecycle state or proposal-source input."""


@dataclass(frozen=True)
class LifecycleConfig:
    maximum_primitives: int = 4
    estimated_coverage_threshold: float = 0.85
    primitive_window_width_s: float = 0.25
    primitive_travel_length_m: float = 0.04
    primitive_stiffness_n_m: float = 2000.0
    primitive_damping_n_s_m: float = 80.0
    primitive_tangential_speed_m_s: float = 0.02
    admissible_center_s: tuple[float, float] = (0.2, 0.8)

    def validate(self) -> None:
        if int(self.maximum_primitives) <= 0:
            raise LifecycleContractError("maximum primitive budget must be positive")
        if not 0.0 < float(self.estimated_coverage_threshold) <= 1.0:
            raise LifecycleContractError("coverage threshold must lie in (0, 1]")
        if (
            len(self.admissible_center_s) != 2
            or not 0.0 <= float(self.admissible_center_s[0])
            < float(self.admissible_center_s[1]) <= 1.0
        ):
            raise LifecycleContractError("admissible center interval must lie in [0, 1]")
        positive = (
            self.primitive_window_width_s,
            self.primitive_travel_length_m,
            self.primitive_stiffness_n_m,
            self.primitive_damping_n_s_m,
            self.primitive_tangential_speed_m_s,
        )
        if not all(math.isfinite(float(value)) and float(value) > 0.0 for value in positive):
            raise LifecycleContractError("primitive defaults must be finite and positive")


def estimated_coverage(observation: PolicyObservation) -> float:
    """Return coverage from the causal residual estimate, never audit truth."""

    if observation.privileged_oracle:
        raise LifecycleContractError("oracle observations are forbidden in lifecycle control")
    if observation.residual_estimate is None:
        raise LifecycleContractError("proposal source requires a residual estimate")
    residual = np.asarray(observation.residual_estimate, dtype=np.float64)
    if residual.ndim != 1 or residual.size == 0 or not np.all(np.isfinite(residual)):
        raise LifecycleContractError("residual estimate must be one finite vector")
    if np.any(residual < 0.0) or np.any(residual > 1.0):
        raise LifecycleContractError("residual estimate must lie in [0, 1]")
    return clean_fraction_from_residual(residual)


def clean_fraction_from_residual(residual_values) -> float:
    residual = np.asarray(residual_values, dtype=np.float64)
    if residual.ndim != 1 or residual.size == 0 or not np.all(np.isfinite(residual)):
        raise LifecycleContractError("residual values must be one finite vector")
    if np.any(residual < 0.0) or np.any(residual > 1.0):
        raise LifecycleContractError("residual values must lie in [0, 1]")
    return float(1.0 - np.mean(residual))


def mass_removal_fraction(*, initial_mass: float, final_mass: float) -> float:
    initial = float(initial_mass)
    final = float(final_mass)
    if not math.isfinite(initial) or not math.isfinite(final) or initial <= 0.0:
        raise LifecycleContractError("residual masses must be finite with positive initial mass")
    if final < 0.0 or final > initial + 1e-12:
        raise LifecycleContractError("final residual mass must lie in [0, initial]")
    return float(np.clip(1.0 - final / initial, 0.0, 1.0))


class ResidualGreedyProposalSource:
    """Deterministic nonlearning proposal source for executor validation."""

    def __init__(self, config: LifecycleConfig = LifecycleConfig()) -> None:
        config.validate()
        self.config = config

    def propose(
        self,
        observation: PolicyObservation,
        *,
        target_force_n: float,
        primitive_index: int,
    ) -> PrimitiveProposal:
        if observation.privileged_oracle:
            raise LifecycleContractError("proposal source cannot consume oracle state")
        if observation.residual_estimate is None:
            raise LifecycleContractError("proposal source requires residual estimate")
        residual = np.asarray(observation.residual_estimate, dtype=np.float64)
        if residual.ndim != 1 or residual.size == 0 or not np.all(np.isfinite(residual)):
            raise LifecycleContractError("invalid residual estimate")
        target = float(target_force_n)
        if not math.isfinite(target) or target <= 0.0:
            raise LifecycleContractError("target force must be finite and positive")
        centers = (np.arange(residual.size, dtype=np.float64) + 0.5) / residual.size
        lower, upper = self.config.admissible_center_s
        admissible = np.flatnonzero((centers >= lower) & (centers <= upper))
        if len(admissible) == 0:
            raise LifecycleContractError(
                "residual grid has no cell in the admissible center interval"
            )
        index = int(admissible[int(np.argmax(residual[admissible]))])
        center = float(centers[index])
        return PrimitiveProposal(
            center_s=center,
            window_width_s=self.config.primitive_window_width_s,
            travel_length_m=self.config.primitive_travel_length_m,
            direction=1,
            target_force_n=target,
            stiffness_n_m=self.config.primitive_stiffness_n_m,
            damping_n_s_m=self.config.primitive_damping_n_s_m,
            tangential_speed_m_s=self.config.primitive_tangential_speed_m_s,
        )


class LifecycleBudget:
    def __init__(self, config: LifecycleConfig = LifecycleConfig()) -> None:
        config.validate()
        self.config = config
        self.executed_primitives = 0

    def record_completed_primitive(self) -> None:
        if self.executed_primitives >= self.config.maximum_primitives:
            raise LifecycleContractError("primitive count exceeds the lifecycle budget")
        self.executed_primitives += 1

    def decision(self, observation: PolicyObservation) -> tuple[bool, str, float]:
        coverage = estimated_coverage(observation)
        if coverage >= self.config.estimated_coverage_threshold:
            return True, "estimated_coverage_reached", coverage
        if self.executed_primitives >= self.config.maximum_primitives:
            return True, "primitive_budget_exhausted", coverage
        return False, "continue", coverage
