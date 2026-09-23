"""Bounded parameterized local-wiping primitives for ForceWipe V4."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math
from typing import Iterable, Protocol

import numpy as np


class PrimitiveContractError(ValueError):
    """Invalid primitive proposal, bound, path, or compiled reference."""


class ArcLengthPathProtocol(Protocol):
    total_length: float

    def at(self, progress: float) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class PrimitiveProposal:
    center_s: float
    window_width_s: float
    travel_length_m: float
    direction: int
    target_force_n: float
    stiffness_n_m: float
    damping_n_s_m: float
    tangential_speed_m_s: float

    def validate_finite(self) -> None:
        numeric = (
            self.center_s,
            self.window_width_s,
            self.travel_length_m,
            self.target_force_n,
            self.stiffness_n_m,
            self.damping_n_s_m,
            self.tangential_speed_m_s,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise PrimitiveContractError("primitive proposal must be finite")
        if int(self.direction) not in {-1, 1}:
            raise PrimitiveContractError("primitive direction must be -1 or +1")


@dataclass(frozen=True)
class PrimitiveBounds:
    center_s: tuple[float, float] = (0.0, 1.0)
    window_width_s: tuple[float, float] = (0.05, 0.50)
    travel_length_m: tuple[float, float] = (0.01, 0.32)
    target_force_n: tuple[float, float] = (3.0, 12.5)
    stiffness_n_m: tuple[float, float] = (100.0, 2000.0)
    damping_n_s_m: tuple[float, float] = (2.0, 150.0)
    tangential_speed_m_s: tuple[float, float] = (0.01, 0.08)

    def validate(self) -> None:
        for name in (
            "center_s",
            "window_width_s",
            "travel_length_m",
            "target_force_n",
            "stiffness_n_m",
            "damping_n_s_m",
            "tangential_speed_m_s",
        ):
            interval = getattr(self, name)
            if len(interval) != 2:
                raise PrimitiveContractError(f"{name} bound must have two values")
            lower, upper = (float(interval[0]), float(interval[1]))
            if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
                raise PrimitiveContractError(f"invalid primitive bound: {name}")
        if self.center_s[0] < 0 or self.center_s[1] > 1:
            raise PrimitiveContractError("center bounds must remain in normalized path")
        if self.window_width_s[0] <= 0 or self.window_width_s[1] > 1:
            raise PrimitiveContractError("window-width bounds must lie in (0, 1]")
        for name in (
            "travel_length_m",
            "target_force_n",
            "stiffness_n_m",
            "damping_n_s_m",
            "tangential_speed_m_s",
        ):
            if getattr(self, name)[0] <= 0:
                raise PrimitiveContractError(f"{name} lower bound must be positive")


@dataclass(frozen=True)
class PrimitiveProjection:
    proposed: PrimitiveProposal
    executed: PrimitiveProposal
    intervened: bool
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class PrimitiveReference:
    progress: np.ndarray
    position_xyz: np.ndarray
    tangent_xyz: np.ndarray
    motion_sign: np.ndarray
    target_force_n: np.ndarray
    stiffness_n_m: np.ndarray
    damping_n_s_m: np.ndarray
    dt_s: float
    lower_s: float
    upper_s: float
    requested_travel_length_m: float

    def __post_init__(self) -> None:
        arrays = (
            self.progress,
            self.position_xyz,
            self.tangent_xyz,
            self.motion_sign,
            self.target_force_n,
            self.stiffness_n_m,
            self.damping_n_s_m,
        )
        count = len(self.progress)
        if count < 2 or any(len(value) != count for value in arrays):
            raise PrimitiveContractError("compiled primitive arrays have inconsistent length")
        for value in arrays:
            if not np.all(np.isfinite(value)):
                raise PrimitiveContractError("compiled primitive contains NaN/Inf")
            value.setflags(write=False)


def project_proposal(
    proposal: PrimitiveProposal,
    bounds: PrimitiveBounds,
) -> PrimitiveProjection:
    proposal.validate_finite()
    bounds.validate()
    reasons: list[str] = []

    def bounded(name: str) -> float:
        value = float(getattr(proposal, name))
        lower, upper = getattr(bounds, name)
        executed = float(np.clip(value, lower, upper))
        if executed != value:
            reasons.append(f"clip_{name}")
        return executed

    executed = PrimitiveProposal(
        center_s=bounded("center_s"),
        window_width_s=bounded("window_width_s"),
        travel_length_m=bounded("travel_length_m"),
        direction=int(proposal.direction),
        target_force_n=bounded("target_force_n"),
        stiffness_n_m=bounded("stiffness_n_m"),
        damping_n_s_m=bounded("damping_n_s_m"),
        tangential_speed_m_s=bounded("tangential_speed_m_s"),
    )
    return PrimitiveProjection(
        proposed=proposal,
        executed=executed,
        intervened=bool(reasons),
        reason_codes=tuple(reasons),
    )


def _reflected_position(
    distance_m: np.ndarray,
    span_m: float,
    direction: int,
) -> np.ndarray:
    start = 0.0 if direction == 1 else span_m
    phase = np.mod(start + float(direction) * distance_m, 2.0 * span_m)
    return np.where(phase <= span_m, phase, 2.0 * span_m - phase)


def compile_primitive(
    executed: PrimitiveProposal,
    path: ArcLengthPathProtocol,
    *,
    bounds: PrimitiveBounds = PrimitiveBounds(),
    native_rate_hz: float = 100.0,
    maximum_reference_samples: int = 100_000,
) -> PrimitiveReference:
    executed.validate_finite()
    projection = project_proposal(executed, bounds)
    if projection.intervened:
        raise PrimitiveContractError(
            "compile_primitive requires an already projected in-bounds action"
        )
    path_length = float(path.total_length)
    if not math.isfinite(path_length) or path_length <= 0:
        raise PrimitiveContractError("path must have finite positive arc length")
    if not math.isfinite(native_rate_hz) or native_rate_hz <= 0:
        raise PrimitiveContractError("native reference rate must be positive")
    if int(maximum_reference_samples) < 2:
        raise PrimitiveContractError("maximum_reference_samples must be at least two")
    lower_s = max(0.0, executed.center_s - 0.5 * executed.window_width_s)
    upper_s = min(1.0, executed.center_s + 0.5 * executed.window_width_s)
    span_m = (upper_s - lower_s) * path_length
    if span_m <= np.finfo(float).eps:
        raise PrimitiveContractError("clipped primitive window has zero arc length")
    dt_s = 1.0 / float(native_rate_hz)
    intervals = max(
        1,
        int(math.ceil(executed.travel_length_m / (executed.tangential_speed_m_s * dt_s))),
    )
    count = intervals + 1
    if count > int(maximum_reference_samples):
        raise PrimitiveContractError("compiled primitive exceeds reference-sample limit")
    distance = np.linspace(0.0, executed.travel_length_m, count, dtype=np.float64)
    local_m = _reflected_position(distance, span_m, executed.direction)
    progress = lower_s + local_m / path_length
    positions = np.empty((count, 3), dtype=np.float64)
    tangents = np.empty((count, 3), dtype=np.float64)
    for index, value in enumerate(progress):
        point, tangent = path.at(float(value))
        point = np.asarray(point, dtype=np.float64)
        tangent = np.asarray(tangent, dtype=np.float64)
        if point.shape != (3,) or tangent.shape != (3,):
            raise PrimitiveContractError("path.at must return two three-vectors")
        tangent_norm = float(np.linalg.norm(tangent))
        if not np.all(np.isfinite(point)) or not np.all(np.isfinite(tangent)) or tangent_norm <= 0:
            raise PrimitiveContractError("path reference is nonfinite or has zero tangent")
        positions[index] = point
        tangents[index] = tangent / tangent_norm
    delta_s = np.diff(progress)
    signs = np.sign(delta_s)
    if np.any(signs == 0):
        raise PrimitiveContractError("compiled primitive contains a stationary reference interval")
    motion_sign = np.concatenate((signs, signs[-1:])).astype(np.int8)
    tangent_motion = tangents * motion_sign[:, None]
    return PrimitiveReference(
        progress=progress,
        position_xyz=positions,
        tangent_xyz=tangent_motion,
        motion_sign=motion_sign,
        target_force_n=np.full(count, executed.target_force_n, dtype=np.float64),
        stiffness_n_m=np.full(count, executed.stiffness_n_m, dtype=np.float64),
        damping_n_s_m=np.full(count, executed.damping_n_s_m, dtype=np.float64),
        dt_s=dt_s,
        lower_s=lower_s,
        upper_s=upper_s,
        requested_travel_length_m=executed.travel_length_m,
    )


def generate_discrete_library(
    *,
    centers_s: Iterable[float],
    window_widths_s: Iterable[float],
    travel_lengths_m: Iterable[float],
    directions: Iterable[int],
    target_forces_n: Iterable[float],
    stiffness_values_n_m: Iterable[float],
    damping_values_n_s_m: Iterable[float],
    speeds_m_s: Iterable[float],
    bounds: PrimitiveBounds,
) -> tuple[PrimitiveProposal, ...]:
    """Create a deterministic enlarged library without a fixed action count."""
    library: list[PrimitiveProposal] = []
    seen: set[PrimitiveProposal] = set()
    axes = (
        tuple(centers_s),
        tuple(window_widths_s),
        tuple(travel_lengths_m),
        tuple(directions),
        tuple(target_forces_n),
        tuple(stiffness_values_n_m),
        tuple(damping_values_n_s_m),
        tuple(speeds_m_s),
    )
    if any(len(axis) == 0 for axis in axes):
        raise PrimitiveContractError("library axes must be nonempty")
    for values in itertools.product(*axes):
        proposal = PrimitiveProposal(*values)
        projection = project_proposal(proposal, bounds)
        if projection.intervened:
            raise PrimitiveContractError("library contains an out-of-bounds proposal")
        if proposal in seen:
            raise PrimitiveContractError("library contains duplicate proposals")
        seen.add(proposal)
        library.append(proposal)
    return tuple(library)
