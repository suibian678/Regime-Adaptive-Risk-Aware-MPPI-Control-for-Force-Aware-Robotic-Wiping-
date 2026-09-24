"""Causal synthetic-residual dynamics for the ForceWipe V4 simulation study.

The update is phase-agnostic: first-pass wiping and re-wiping call the same
100 Hz function.  This is a synthetic exposure construct, not a model or
measurement of physical contaminant removal.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np


class ResidualContractError(ValueError):
    """A causal-order, shape, unit, or finite-value contract violation."""


@dataclass(frozen=True)
class CausalResidualConfig:
    native_rate_hz: float = 100.0
    kappa: float = 1.0
    minimum_contact_force_n: float = 3.0
    maximum_safe_force_n: float = 15.0
    force_quality_width_fraction: float = 0.25
    footprint_length_m: float = 0.02
    maximum_traversal_per_sample: float = 1.0
    timestamp_relative_tolerance: float = 1e-6

    def validate(self) -> None:
        numeric = (
            self.native_rate_hz,
            self.kappa,
            self.minimum_contact_force_n,
            self.maximum_safe_force_n,
            self.force_quality_width_fraction,
            self.footprint_length_m,
            self.maximum_traversal_per_sample,
            self.timestamp_relative_tolerance,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ResidualContractError("residual configuration must be finite")
        if self.native_rate_hz <= 0 or self.kappa <= 0:
            raise ResidualContractError("native rate and kappa must be positive")
        if not 0 < self.minimum_contact_force_n < self.maximum_safe_force_n:
            raise ResidualContractError("invalid process-force window")
        if self.force_quality_width_fraction <= 0 or self.footprint_length_m <= 0:
            raise ResidualContractError("force width and footprint length must be positive")
        if self.maximum_traversal_per_sample <= 0:
            raise ResidualContractError("sample traversal bound must be positive")
        if self.timestamp_relative_tolerance < 0:
            raise ResidualContractError("timestamp tolerance must be nonnegative")

    @property
    def native_dt_s(self) -> float:
        self.validate()
        return 1.0 / float(self.native_rate_hz)

    @property
    def native_period_ns(self) -> int:
        return int(round(1e9 * self.native_dt_s))


@dataclass(frozen=True)
class CausalContactSample:
    sample_index: int
    time_ns: int
    audit_force_n: float
    nominal_force_n: float
    tangential_speed_m_s: float
    overlap_fraction: np.ndarray

    def __post_init__(self) -> None:
        if int(self.sample_index) < 0 or int(self.time_ns) < 0:
            raise ResidualContractError("sample index and timestamp must be nonnegative")
        scalars = (
            self.audit_force_n,
            self.nominal_force_n,
            self.tangential_speed_m_s,
        )
        if not all(math.isfinite(float(value)) for value in scalars):
            raise ResidualContractError("contact sample must be finite")
        if self.nominal_force_n <= 0 or self.tangential_speed_m_s < 0:
            raise ResidualContractError("nominal force must be positive and speed nonnegative")
        overlap = np.asarray(self.overlap_fraction, dtype=np.float64)
        if overlap.ndim != 1 or overlap.size == 0:
            raise ResidualContractError("overlap_fraction must be a nonempty vector")
        if not np.all(np.isfinite(overlap)) or np.any(overlap < 0) or np.any(overlap > 1):
            raise ResidualContractError("overlap fractions must be finite and in [0, 1]")
        overlap = overlap.copy()
        overlap.setflags(write=False)
        object.__setattr__(self, "sample_index", int(self.sample_index))
        object.__setattr__(self, "time_ns", int(self.time_ns))
        object.__setattr__(self, "overlap_fraction", overlap)


@dataclass(frozen=True)
class ResidualUpdateAudit:
    sample_index: int
    time_ns: int
    force_quality: float
    traversal: float
    removed_mass: float
    mass_before: float
    mass_after: float


@dataclass(frozen=True)
class TruthAuditSnapshot:
    sample_index: int
    time_ns: int
    values: tuple[float, ...]
    mass: float


_CAPABILITY_SECRET = object()


class TruthAuditCapability:
    """Capability held by the recorder/scorer, never by a policy or shield."""

    __slots__ = ("_secret",)

    def __init__(self, secret: Any) -> None:
        if secret is not _CAPABILITY_SECRET:
            raise ResidualContractError("truth-audit capability cannot be constructed directly")
        self._secret = secret


def force_quality(
    normal_force_n: float,
    nominal_force_n: float,
    config: CausalResidualConfig,
) -> float:
    config.validate()
    force = float(normal_force_n)
    nominal = float(nominal_force_n)
    if not math.isfinite(force) or not math.isfinite(nominal) or nominal <= 0:
        raise ResidualContractError("force inputs must be finite and nominal positive")
    if force < config.minimum_contact_force_n or force > config.maximum_safe_force_n:
        return 0.0
    width = max(config.force_quality_width_fraction * nominal, np.finfo(float).eps)
    normalized_error = (force - nominal) / width
    return float(math.exp(-0.5 * normalized_error * normalized_error))


def removal_increment(
    sample: CausalContactSample,
    config: CausalResidualConfig,
) -> tuple[np.ndarray, float, float]:
    """Return per-cell removal, force quality, and dimensionless traversal."""
    config.validate()
    return removal_increment_from_components(
        overlap_fraction=sample.overlap_fraction,
        force_n=sample.audit_force_n,
        nominal_force_n=sample.nominal_force_n,
        tangential_speed_m_s=sample.tangential_speed_m_s,
        config=config,
    )


def removal_increment_from_components(
    *,
    overlap_fraction: np.ndarray,
    force_n: float,
    nominal_force_n: float,
    tangential_speed_m_s: float,
    config: CausalResidualConfig,
) -> tuple[np.ndarray, float, float]:
    """Shared update kernel with an explicitly named force stream."""
    overlap = np.asarray(overlap_fraction, dtype=np.float64)
    if overlap.ndim != 1 or overlap.size == 0:
        raise ResidualContractError("overlap_fraction must be a nonempty vector")
    if not np.all(np.isfinite(overlap)) or np.any(overlap < 0) or np.any(overlap > 1):
        raise ResidualContractError("overlap fractions must be finite and in [0, 1]")
    if not math.isfinite(float(tangential_speed_m_s)) or tangential_speed_m_s < 0:
        raise ResidualContractError("tangential speed must be finite and nonnegative")
    quality = force_quality(force_n, nominal_force_n, config)
    traversal = (
        float(tangential_speed_m_s)
        * config.native_dt_s
        / float(config.footprint_length_m)
    )
    traversal = float(np.clip(traversal, 0.0, config.maximum_traversal_per_sample))
    increment = config.kappa * overlap * quality * traversal
    return increment, quality, traversal


class CausalResidualTruth:
    """Environment-owned mutable truth; policies receive observations, not this object."""

    def __init__(self, initial_values: np.ndarray, config: CausalResidualConfig):
        config.validate()
        values = np.asarray(initial_values, dtype=np.float64)
        if values.ndim != 1 or values.size == 0:
            raise ResidualContractError("initial residual must be a nonempty vector")
        if not np.all(np.isfinite(values)) or np.any(values < 0) or np.any(values > 1):
            raise ResidualContractError("initial residual must be finite and in [0, 1]")
        self._values = values.copy()
        self._config = config
        self._next_sample_index = 0
        self._last_time_ns: int | None = None
        self._last_audit: ResidualUpdateAudit | None = None

    @property
    def cell_count(self) -> int:
        return int(self._values.size)

    @property
    def config(self) -> CausalResidualConfig:
        return self._config

    @property
    def next_sample_index(self) -> int:
        return self._next_sample_index

    def update(self, sample: CausalContactSample) -> ResidualUpdateAudit:
        if sample.sample_index != self._next_sample_index:
            raise ResidualContractError(
                f"noncausal sample order: expected {self._next_sample_index}, got {sample.sample_index}"
            )
        if sample.overlap_fraction.size != self.cell_count:
            raise ResidualContractError("overlap shape differs from residual field")
        if self._last_time_ns is not None:
            delta_ns = sample.time_ns - self._last_time_ns
            expected_ns = self._config.native_period_ns
            tolerance_ns = max(
                1,
                int(round(expected_ns * self._config.timestamp_relative_tolerance)),
            )
            if abs(delta_ns - expected_ns) > tolerance_ns:
                raise ResidualContractError(
                    f"native timestamp discontinuity: expected {expected_ns} ns, got {delta_ns} ns"
                )
        increment, quality, traversal = removal_increment(sample, self._config)
        before_mass = float(self._values.sum())
        before = self._values.copy()
        self._values = np.clip(before - increment, 0.0, 1.0)
        after_mass = float(self._values.sum())
        audit = ResidualUpdateAudit(
            sample_index=sample.sample_index,
            time_ns=sample.time_ns,
            force_quality=quality,
            traversal=traversal,
            removed_mass=before_mass - after_mass,
            mass_before=before_mass,
            mass_after=after_mass,
        )
        self._last_time_ns = sample.time_ns
        self._next_sample_index += 1
        self._last_audit = audit
        return audit

    def _copy_values_for_sensor(self) -> np.ndarray:
        """Environment-internal current-state copy for simulated sensing."""
        return self._values.copy()

    def audit_snapshot(self, capability: TruthAuditCapability) -> TruthAuditSnapshot:
        if not isinstance(capability, TruthAuditCapability) or capability._secret is not _CAPABILITY_SECRET:
            raise ResidualContractError("valid truth-audit capability required")
        if self._last_audit is None:
            return TruthAuditSnapshot(
                sample_index=-1,
                time_ns=-1,
                values=tuple(float(value) for value in self._values),
                mass=float(self._values.sum()),
            )
        return TruthAuditSnapshot(
            sample_index=self._last_audit.sample_index,
            time_ns=self._last_audit.time_ns,
            values=tuple(float(value) for value in self._values),
            mass=float(self._values.sum()),
        )


def create_truth_state(
    initial_values: np.ndarray,
    config: CausalResidualConfig,
) -> tuple[CausalResidualTruth, TruthAuditCapability]:
    """Create environment truth and a separately held audit capability."""
    return CausalResidualTruth(initial_values, config), TruthAuditCapability(_CAPABILITY_SECRET)
