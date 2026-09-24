"""Deterministic joint TRAIN/DEV/CAL/STRESS scenario generation for V4.

Formal TEST families are deliberately unavailable in this module while the
V4 protocol gates and execution permissions remain open.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import re
from typing import Iterable

import numpy as np


class ScenarioContractError(ValueError):
    """Invalid role, seed namespace, factor, geometry, or residual field."""


PATH_KINDS = ("line", "diagonal", "arc", "s_curve", "polyline")
SURFACE_KINDS = ("flat", "incline_5", "incline_10", "cylinder_low")
TOOL_KINDS = ("narrow_soft", "narrow_hard", "wide_soft", "wide_hard")
RESIDUAL_FAMILIES = (
    "single_blob",
    "islands",
    "edge_dense",
    "stripe",
    "multimodal",
    "gaussian_random_field",
)
NOMINAL_DISTURBANCES = (
    "none",
    "lateral_impulse",
    "normal_impulse",
    "reference_noise",
    "sensor_noise_latency",
)
STRESS_DISTURBANCES = NOMINAL_DISTURBANCES + (
    "dynamic_height",
    "moving_obstacle",
)
CONSTRAINT_KINDS = ("none", "static_obstacle", "keep_out_zone")
ALLOWED_ROLES = {"TRAIN", "DEV", "CAL", "STRESS"}
ROLE_BASE = {"TRAIN": 1_000_000, "DEV": 2_000_000, "CAL": 3_000_000, "STRESS": 4_000_000}
SAFE_NAMESPACE = re.compile(r"^v[45]_(train|dev|cal|stress)_[a-z0-9_.-]+$")


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: int
    role: str
    seed_namespace: str
    scenario_seed: int
    path_kind: str
    path_length_m: float
    path_lateral_scale_m: float
    surface_kind: str
    cylinder_radius_m: float
    tool_kind: str
    tool_width_m: float
    tool_footprint_length_m: float
    tool_normal_stiffness_n_m: float
    friction_coefficient: float
    support_stiffness_n_m: float
    effective_mass_kg: float
    damping_ratio: float
    restitution: float
    residual_family: str
    residual_seed: int
    residual_severity: float
    disturbance_kind: str
    disturbance_scale: float
    sensor_latency_steps: int
    constraint_kind: str
    obstacle_center_s: float
    obstacle_half_width_s: float
    target_force_n: float
    residual_cleanability: float = 1.0

    @property
    def support_damping_n_s_m(self) -> float:
        return float(
            2.0
            * self.damping_ratio
            * math.sqrt(self.support_stiffness_n_m * self.effective_mass_kg)
        )

    def validate(self) -> None:
        if self.role not in ALLOWED_ROLES:
            raise ScenarioContractError(f"unsupported scenario role: {self.role}")
        if not SAFE_NAMESPACE.fullmatch(self.seed_namespace):
            raise ScenarioContractError("invalid or role-ambiguous seed namespace")
        if not self.seed_namespace.startswith(
            (f"v4_{self.role.lower()}_", f"v5_{self.role.lower()}_")
        ):
            raise ScenarioContractError("seed namespace does not match scenario role")
        if self.path_kind not in PATH_KINDS or self.surface_kind not in SURFACE_KINDS:
            raise ScenarioContractError("unknown path or surface kind")
        if self.tool_kind not in TOOL_KINDS or self.residual_family not in RESIDUAL_FAMILIES:
            raise ScenarioContractError("unknown tool or residual family")
        allowed_disturbances = STRESS_DISTURBANCES if self.role == "STRESS" else NOMINAL_DISTURBANCES
        if self.disturbance_kind not in allowed_disturbances:
            raise ScenarioContractError("disturbance is not permitted for this role")
        if self.constraint_kind not in CONSTRAINT_KINDS:
            raise ScenarioContractError("unknown spatial-constraint kind")
        numeric = (
            self.path_length_m,
            self.path_lateral_scale_m,
            self.cylinder_radius_m,
            self.tool_width_m,
            self.tool_footprint_length_m,
            self.tool_normal_stiffness_n_m,
            self.friction_coefficient,
            self.support_stiffness_n_m,
            self.effective_mass_kg,
            self.damping_ratio,
            self.restitution,
            self.residual_severity,
            self.disturbance_scale,
            self.obstacle_center_s,
            self.obstacle_half_width_s,
            self.target_force_n,
            self.residual_cleanability,
            self.support_damping_n_s_m,
        )
        if not all(math.isfinite(float(value)) for value in numeric):
            raise ScenarioContractError("scenario contains NaN/Inf")
        if not 0.10 <= self.path_length_m <= 0.24:
            raise ScenarioContractError("path length is outside the simulated family")
        if self.path_lateral_scale_m < 0 or self.cylinder_radius_m <= 0:
            raise ScenarioContractError("invalid path/surface scale")
        if self.tool_width_m <= 0 or self.tool_footprint_length_m <= 0:
            raise ScenarioContractError("tool dimensions must be positive")
        if self.tool_normal_stiffness_n_m <= 0 or self.support_stiffness_n_m <= 0:
            raise ScenarioContractError("tool/support stiffness must be positive")
        if not 0 < self.friction_coefficient <= 1.5:
            raise ScenarioContractError("friction coefficient is outside the simulated family")
        if self.effective_mass_kg <= 0 or not 0.1 <= self.damping_ratio <= 2.0:
            raise ScenarioContractError("invalid support mass/damping ratio")
        if not 0 <= self.restitution <= 0.5:
            raise ScenarioContractError("invalid restitution")
        if not 0 < self.residual_severity <= 1:
            raise ScenarioContractError("residual severity must lie in (0, 1]")
        if not 0 < self.residual_cleanability <= 1:
            raise ScenarioContractError("residual cleanability must lie in (0, 1]")
        if self.disturbance_scale < 0 or self.sensor_latency_steps < 0:
            raise ScenarioContractError("invalid disturbance or latency")
        if not 0 <= self.obstacle_center_s <= 1 or not 0 <= self.obstacle_half_width_s <= 0.25:
            raise ScenarioContractError("invalid obstacle interval")
        if self.target_force_n not in {5.0, 8.0, 12.0}:
            raise ScenarioContractError("target force must be 5, 8, or 12 N")


def scenario_digest(spec: ScenarioSpec) -> str:
    spec.validate()
    payload = json.dumps(
        asdict(spec), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def latin_hypercube(count: int, dimensions: int, rng: np.random.Generator) -> np.ndarray:
    if int(count) <= 0 or int(dimensions) <= 0:
        raise ScenarioContractError("Latin-hypercube shape must be positive")
    output = np.empty((count, dimensions), dtype=np.float64)
    for dimension in range(dimensions):
        strata = (np.arange(count, dtype=np.float64) + rng.random(count)) / count
        output[:, dimension] = strata[rng.permutation(count)]
    return output


def _category(unit_value: float, values: tuple[str, ...]) -> str:
    index = min(int(float(unit_value) * len(values)), len(values) - 1)
    return values[index]


def _tool_profile(kind: str, unit_value: float) -> tuple[float, float, float]:
    narrow = kind.startswith("narrow")
    soft = kind.endswith("soft")
    width = (0.026 if narrow else 0.050) * (0.95 + 0.10 * unit_value)
    footprint = (0.018 if narrow else 0.030) * (0.95 + 0.10 * (1.0 - unit_value))
    stiffness = (650.0 if soft else 1450.0) * (0.9 + 0.2 * unit_value)
    return float(width), float(footprint), float(stiffness)


def generate_joint_scenarios(
    *,
    role: str,
    count: int,
    master_seed: int,
    seed_namespace: str,
    targets_n: Iterable[float] = (5.0, 8.0, 12.0),
) -> tuple[ScenarioSpec, ...]:
    role = str(role).upper()
    if role not in ALLOWED_ROLES:
        raise ScenarioContractError(
            "formal TEST/ID/COMB/OOD scenarios are unavailable before protocol authorization"
        )
    if int(count) <= 0 or int(master_seed) < 0:
        raise ScenarioContractError("count must be positive and master seed nonnegative")
    if not SAFE_NAMESPACE.fullmatch(str(seed_namespace)) or not str(seed_namespace).startswith(
        (f"v4_{role.lower()}_", f"v5_{role.lower()}_")
    ):
        raise ScenarioContractError("seed namespace must be explicit and role-specific")
    targets = tuple(float(value) for value in targets_n)
    if not targets or not set(targets).issubset({5.0, 8.0, 12.0}):
        raise ScenarioContractError("invalid force-target roster")
    rng = np.random.default_rng(int(master_seed))
    design = latin_hypercube(int(count), 21, rng)
    target_roster = np.resize(np.asarray(targets, dtype=np.float64), int(count))
    rng.shuffle(target_roster)
    scenario_seeds = rng.integers(0, np.iinfo(np.uint32).max, count, dtype=np.uint64)
    residual_seeds = rng.integers(0, np.iinfo(np.uint32).max, count, dtype=np.uint64)
    disturbances = STRESS_DISTURBANCES if role == "STRESS" else NOMINAL_DISTURBANCES
    scenarios: list[ScenarioSpec] = []
    for index in range(int(count)):
        row = design[index]
        path_kind = _category(row[0], PATH_KINDS)
        surface_kind = _category(row[1], SURFACE_KINDS)
        tool_kind = _category(row[2], TOOL_KINDS)
        residual_family = _category(row[3], RESIDUAL_FAMILIES)
        disturbance_kind = _category(row[4], disturbances)
        constraint_kind = _category(row[5], CONSTRAINT_KINDS)
        tool_width, footprint_length, tool_stiffness = _tool_profile(tool_kind, row[6])
        spec = ScenarioSpec(
            scenario_id=ROLE_BASE[role] + index,
            role=role,
            seed_namespace=str(seed_namespace),
            scenario_seed=int(scenario_seeds[index]),
            path_kind=path_kind,
            path_length_m=float(0.12 + 0.08 * row[7]),
            path_lateral_scale_m=float(0.008 + 0.016 * row[8]),
            surface_kind=surface_kind,
            cylinder_radius_m=float(0.30 + 0.30 * row[9]),
            tool_kind=tool_kind,
            tool_width_m=tool_width,
            tool_footprint_length_m=footprint_length,
            tool_normal_stiffness_n_m=tool_stiffness,
            friction_coefficient=float(0.20 + 0.60 * row[10]),
            support_stiffness_n_m=float(1000.0 + 3000.0 * row[11]),
            effective_mass_kg=float(0.20 + 0.40 * row[12]),
            damping_ratio=float(0.35 + 0.90 * row[13]),
            restitution=float(0.02 + 0.16 * row[14]),
            residual_family=residual_family,
            residual_seed=int(residual_seeds[index]),
            residual_severity=float(0.55 + 0.45 * row[15]),
            disturbance_kind=disturbance_kind,
            disturbance_scale=(0.0 if disturbance_kind == "none" else float(0.1 + 0.9 * row[16])),
            sensor_latency_steps=(
                int(math.floor(1 + 4 * row[17]))
                if disturbance_kind == "sensor_noise_latency"
                else 0
            ),
            constraint_kind=constraint_kind,
            obstacle_center_s=float(0.20 + 0.60 * row[18]),
            obstacle_half_width_s=(
                float(0.025 + 0.075 * row[19]) if constraint_kind != "none" else 0.0
            ),
            target_force_n=float(target_roster[index]),
            residual_cleanability=float(0.25 + 0.75 * row[20]),
        )
        spec.validate()
        scenarios.append(spec)
    digests = [scenario_digest(spec) for spec in scenarios]
    if len(digests) != len(set(digests)):
        raise ScenarioContractError("joint generator produced duplicate scenarios")
    return tuple(scenarios)


class ScenarioPath:
    """Immutable polyline with normalized arc-length interpolation."""

    def __init__(self, points_xyz: np.ndarray):
        points = np.asarray(points_xyz, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ScenarioContractError("path points must have shape (N>=2, 3)")
        if not np.all(np.isfinite(points)):
            raise ScenarioContractError("path points must be finite")
        delta = np.diff(points, axis=0)
        lengths = np.linalg.norm(delta, axis=1)
        if np.any(lengths <= np.finfo(float).eps):
            raise ScenarioContractError("path contains a zero-length segment")
        cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
        self._points = points.copy()
        self._points.setflags(write=False)
        self._delta = delta
        self._lengths = lengths
        self._cumulative = cumulative
        self.total_length = float(cumulative[-1])

    @property
    def points_xyz(self) -> np.ndarray:
        return self._points

    def at(self, progress: float) -> tuple[np.ndarray, np.ndarray]:
        s = float(np.clip(progress, 0.0, 1.0)) * self.total_length
        index = int(np.searchsorted(self._cumulative, s, side="right") - 1)
        index = min(max(index, 0), len(self._lengths) - 1)
        alpha = (s - self._cumulative[index]) / self._lengths[index]
        point = self._points[index] + alpha * self._delta[index]
        tangent = self._delta[index] / self._lengths[index]
        return point.copy(), tangent.copy()

    def project(self, point_xyz: np.ndarray) -> "ScenarioPathProjection":
        query = np.asarray(point_xyz, dtype=np.float64)
        if query.shape != (3,) or not np.all(np.isfinite(query)):
            raise ScenarioContractError("projection query must be a finite three-vector")
        starts = self._points[:-1]
        denominator = np.sum(self._delta * self._delta, axis=1)
        alpha = np.sum((query - starts) * self._delta, axis=1) / denominator
        alpha = np.clip(alpha, 0.0, 1.0)
        candidates = starts + alpha[:, None] * self._delta
        distances = np.linalg.norm(candidates - query, axis=1)
        index = int(np.argmin(distances))
        arc_length = self._cumulative[index] + alpha[index] * self._lengths[index]
        tangent = self._delta[index] / self._lengths[index]
        return ScenarioPathProjection(
            progress=float(arc_length / self.total_length),
            point_xyz=candidates[index].copy(),
            tangent_xyz=tangent.copy(),
            distance_m=float(distances[index]),
            segment_index=index,
        )


@dataclass(frozen=True)
class ScenarioPathProjection:
    progress: float
    point_xyz: np.ndarray
    tangent_xyz: np.ndarray
    distance_m: float
    segment_index: int

    def __post_init__(self) -> None:
        point = np.asarray(self.point_xyz, dtype=np.float64).copy()
        tangent = np.asarray(self.tangent_xyz, dtype=np.float64).copy()
        if point.shape != (3,) or tangent.shape != (3,):
            raise ScenarioContractError("path projection vectors must have shape (3,)")
        point.setflags(write=False)
        tangent.setflags(write=False)
        object.__setattr__(self, "point_xyz", point)
        object.__setattr__(self, "tangent_xyz", tangent)


def _surface_height(spec: ScenarioSpec, x_m: np.ndarray) -> np.ndarray:
    if spec.surface_kind == "flat":
        return np.zeros_like(x_m)
    if spec.surface_kind in {"incline_5", "incline_10"}:
        angle = 5.0 if spec.surface_kind == "incline_5" else 10.0
        return np.tan(np.deg2rad(angle)) * x_m
    if spec.surface_kind == "cylinder_low":
        centered = x_m - 0.5 * spec.path_length_m
        radius = spec.cylinder_radius_m
        if np.any(np.abs(centered) >= radius):
            raise ScenarioContractError("path exceeds cylinder domain")
        return np.sqrt(radius * radius - centered * centered) - radius
    raise ScenarioContractError("unknown surface kind")


def surface_normal(spec: ScenarioSpec, x_m: np.ndarray | float) -> np.ndarray:
    """Return the outward unit normal of the physical surface at path x."""

    spec.validate()
    x = np.asarray(x_m, dtype=np.float64)
    normal = np.zeros(x.shape + (3,), dtype=np.float64)
    if spec.surface_kind == "flat":
        normal[..., 2] = 1.0
    elif spec.surface_kind in {"incline_5", "incline_10"}:
        angle_deg = 5.0 if spec.surface_kind == "incline_5" else 10.0
        angle = np.deg2rad(angle_deg)
        normal[..., 0] = -np.sin(angle)
        normal[..., 2] = np.cos(angle)
    elif spec.surface_kind == "cylinder_low":
        centered = x - 0.5 * spec.path_length_m
        radius = spec.cylinder_radius_m
        if np.any(np.abs(centered) >= radius):
            raise ScenarioContractError("normal query exceeds cylinder domain")
        normal[..., 0] = centered / radius
        normal[..., 2] = np.sqrt(radius * radius - centered * centered) / radius
    else:
        raise ScenarioContractError("unknown surface kind")
    return normal


def make_scenario_path(spec: ScenarioSpec, *, points: int = 401) -> ScenarioPath:
    spec.validate()
    if int(points) < 11:
        raise ScenarioContractError("scenario path requires at least 11 points")
    dense = max(4001, int(points) * 10)
    u = np.linspace(0.0, 1.0, dense)
    x = spec.path_length_m * u
    if spec.path_kind == "line":
        y = np.zeros_like(u)
    elif spec.path_kind == "diagonal":
        y = spec.path_lateral_scale_m * (u - 0.5)
    elif spec.path_kind == "arc":
        y = spec.path_lateral_scale_m * 4.0 * u * (1.0 - u)
    elif spec.path_kind == "s_curve":
        y = spec.path_lateral_scale_m * np.sin(2.0 * np.pi * u)
    elif spec.path_kind == "polyline":
        knots_u = np.array([0.0, 0.25, 0.50, 0.75, 1.0])
        knots_y = spec.path_lateral_scale_m * np.array([0.0, 0.8, -0.6, 0.5, 0.0])
        y = np.interp(u, knots_u, knots_y)
    else:
        raise ScenarioContractError("unknown path kind")
    z = _surface_height(spec, x)
    raw = np.column_stack((x, y, z))
    delta = np.diff(raw, axis=0)
    cumulative = np.concatenate(([0.0], np.cumsum(np.linalg.norm(delta, axis=1))))
    desired = np.linspace(0.0, cumulative[-1], int(points))
    resampled = np.column_stack(
        [np.interp(desired, cumulative, raw[:, axis]) for axis in range(3)]
    )
    return ScenarioPath(resampled)


def make_residual_field(spec: ScenarioSpec, *, cells: int = 64) -> np.ndarray:
    spec.validate()
    if int(cells) < 8:
        raise ScenarioContractError("residual field requires at least eight cells")
    rng = np.random.default_rng(spec.residual_seed)
    s = (np.arange(cells, dtype=np.float64) + 0.5) / cells
    if spec.residual_family == "single_blob":
        center = rng.uniform(0.2, 0.8)
        width = rng.uniform(0.06, 0.16)
        field = np.exp(-0.5 * ((s - center) / width) ** 2)
    elif spec.residual_family == "islands":
        field = np.zeros(cells, dtype=np.float64)
        for center, width, amplitude in zip(
            rng.uniform(0.05, 0.95, 4),
            rng.uniform(0.025, 0.07, 4),
            rng.uniform(0.5, 1.0, 4),
        ):
            field += amplitude * np.exp(-0.5 * ((s - center) / width) ** 2)
    elif spec.residual_family == "edge_dense":
        field = np.exp(-s / 0.10) + np.exp(-(1.0 - s) / 0.10)
    elif spec.residual_family == "stripe":
        phase = rng.uniform(0.0, 2.0 * np.pi)
        field = 0.5 + 0.5 * np.sin(8.0 * np.pi * s + phase) ** 2
    elif spec.residual_family == "multimodal":
        field = np.zeros(cells, dtype=np.float64)
        for center in (0.22, 0.50, 0.78):
            jitter = rng.normal(0.0, 0.025)
            field += np.exp(-0.5 * ((s - center - jitter) / 0.07) ** 2)
    elif spec.residual_family == "gaussian_random_field":
        white = rng.normal(0.0, 1.0, cells)
        radius = 5
        x = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(-0.5 * (x / 2.0) ** 2)
        kernel /= kernel.sum()
        field = np.convolve(np.pad(white, radius, mode="reflect"), kernel, mode="valid")
        field -= field.min()
    else:
        raise ScenarioContractError("unknown residual family")
    maximum = float(field.max())
    if not math.isfinite(maximum) or maximum <= 0:
        raise ScenarioContractError("residual generator produced an empty field")
    field = np.clip(field / maximum * spec.residual_severity, 0.0, 1.0)
    field.setflags(write=False)
    return field


def footprint_overlap_fraction(
    *,
    progress: float,
    path_length_m: float,
    footprint_length_m: float,
    cells: int,
) -> np.ndarray:
    if not 0 <= float(progress) <= 1:
        raise ScenarioContractError("progress must lie in [0, 1]")
    if path_length_m <= 0 or footprint_length_m <= 0 or int(cells) <= 0:
        raise ScenarioContractError("invalid path/footprint discretization")
    half_s = 0.5 * footprint_length_m / path_length_m
    lower = max(0.0, float(progress) - half_s)
    upper = min(1.0, float(progress) + half_s)
    edges = np.linspace(0.0, 1.0, int(cells) + 1)
    overlap = np.maximum(
        0.0,
        np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower),
    )
    overlap /= np.diff(edges)
    overlap = np.clip(overlap, 0.0, 1.0)
    overlap.setflags(write=False)
    return overlap
