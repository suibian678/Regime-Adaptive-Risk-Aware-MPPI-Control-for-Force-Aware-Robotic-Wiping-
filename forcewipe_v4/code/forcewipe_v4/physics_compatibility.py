"""Fail-closed mapping audit from V4 scenario factors to a physics backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from .plant_numerics import VerticalSupportAudit, audit_vertical_support
from .scene_geometry import V4_SURFACE_HALF_X_M, V4_SURFACE_HALF_Y_M
from .scenarios import (
    CONSTRAINT_KINDS,
    PATH_KINDS,
    STRESS_DISTURBANCES,
    SURFACE_KINDS,
    TOOL_KINDS,
    ScenarioSpec,
    make_scenario_path,
)


@dataclass(frozen=True)
class PhysicsBackendCapabilities:
    backend_name: str
    native_rate_hz: float
    pad_half_extent_x_m: float
    pad_half_extent_y_m: float
    supported_path_kinds: tuple[str, ...]
    physical_surface_kinds: tuple[str, ...]
    physical_tool_kinds: tuple[str, ...]
    implemented_disturbance_kinds: tuple[str, ...]
    implemented_constraint_kinds: tuple[str, ...]
    maps_friction: bool
    maps_restitution: bool
    maps_support_mass: bool
    maps_support_stiffness: bool
    maps_support_damping: bool

    def validate(self) -> None:
        if self.native_rate_hz <= 0.0:
            raise ValueError("native_rate_hz must be positive")
        if self.pad_half_extent_x_m <= 0.0 or self.pad_half_extent_y_m <= 0.0:
            raise ValueError("pad half extents must be positive")
        if not set(self.supported_path_kinds).issubset(PATH_KINDS):
            raise ValueError("backend declares an unknown path kind")


@dataclass(frozen=True)
class PhysicsCompatibilityAudit:
    scenario_id: int
    backend_name: str
    supported: bool
    mapped_factors: tuple[str, ...]
    unsupported_factors: tuple[str, ...]
    path_within_pad: bool
    path_x_range_m: tuple[float, float]
    path_y_range_m: tuple[float, float]
    support_numerics: VerticalSupportAudit

    def as_record(self) -> dict[str, object]:
        output = asdict(self)
        output["support_numerics"] = self.support_numerics.as_record()
        return output


def current_forcewipe_v1_capabilities() -> PhysicsBackendCapabilities:
    """Describe only mappings implemented by the current ForceWipe-v1 scene.

    The existing contact geometry is the Panda finger pair against one flat
    dynamic box. Therefore the four proposed V4 tool variants and non-flat
    surfaces are intentionally unsupported until a dedicated V4 scene exists.
    """

    return PhysicsBackendCapabilities(
        backend_name="ForceWipe-v1-flat-pad-panda-fingers",
        native_rate_hz=100.0,
        pad_half_extent_x_m=0.18,
        pad_half_extent_y_m=0.06,
        supported_path_kinds=tuple(PATH_KINDS),
        physical_surface_kinds=("flat",),
        physical_tool_kinds=(),
        implemented_disturbance_kinds=("none",),
        implemented_constraint_kinds=("none",),
        maps_friction=True,
        maps_restitution=True,
        maps_support_mass=True,
        maps_support_stiffness=True,
        maps_support_damping=True,
    )


def current_forcewipe_v4_capabilities() -> PhysicsBackendCapabilities:
    """Describe the implemented dedicated V4 SAPIEN scene."""

    return PhysicsBackendCapabilities(
        backend_name="ForceWipeV4-v1-dedicated-tool-surface",
        native_rate_hz=100.0,
        pad_half_extent_x_m=V4_SURFACE_HALF_X_M,
        pad_half_extent_y_m=V4_SURFACE_HALF_Y_M,
        supported_path_kinds=tuple(PATH_KINDS),
        physical_surface_kinds=tuple(SURFACE_KINDS),
        physical_tool_kinds=tuple(TOOL_KINDS),
        implemented_disturbance_kinds=tuple(STRESS_DISTURBANCES),
        implemented_constraint_kinds=tuple(CONSTRAINT_KINDS),
        maps_friction=True,
        maps_restitution=True,
        maps_support_mass=True,
        maps_support_stiffness=True,
        maps_support_damping=True,
    )


def assess_physics_compatibility(
    spec: ScenarioSpec,
    capabilities: PhysicsBackendCapabilities,
) -> PhysicsCompatibilityAudit:
    """Return a complete mapping audit; unsupported factors never fall back."""

    spec.validate()
    capabilities.validate()
    mapped: list[str] = []
    unsupported: list[str] = []

    if spec.path_kind in capabilities.supported_path_kinds:
        mapped.append("path_reference")
    else:
        unsupported.append(f"path_kind:{spec.path_kind}")
    if spec.surface_kind in capabilities.physical_surface_kinds:
        mapped.append("surface_geometry")
    else:
        unsupported.append(f"surface_geometry:{spec.surface_kind}")
    if spec.tool_kind in capabilities.physical_tool_kinds:
        mapped.extend(("tool_geometry", "tool_compliance"))
    else:
        unsupported.append(f"physical_tool:{spec.tool_kind}")
    if spec.disturbance_kind in capabilities.implemented_disturbance_kinds:
        mapped.append("disturbance")
    else:
        unsupported.append(f"disturbance:{spec.disturbance_kind}")
    if spec.constraint_kind in capabilities.implemented_constraint_kinds:
        mapped.append("spatial_constraint")
    else:
        unsupported.append(f"spatial_constraint:{spec.constraint_kind}")

    switches = (
        (capabilities.maps_friction, "friction"),
        (capabilities.maps_restitution, "restitution"),
        (capabilities.maps_support_mass, "support_mass"),
        (capabilities.maps_support_stiffness, "support_stiffness"),
        (capabilities.maps_support_damping, "support_damping"),
    )
    for available, name in switches:
        (mapped if available else unsupported).append(name)

    path = make_scenario_path(spec, points=401).points_xyz
    half_length = 0.5 * spec.tool_footprint_length_m
    half_width = 0.5 * spec.tool_width_m
    x_range = (
        float(np.min(path[:, 0]) - half_length),
        float(np.max(path[:, 0]) + half_length),
    )
    y_range = (
        float(np.min(path[:, 1]) - half_width),
        float(np.max(path[:, 1]) + half_width),
    )
    within_pad = bool(
        x_range[0] >= -capabilities.pad_half_extent_x_m
        and x_range[1] <= capabilities.pad_half_extent_x_m
        and y_range[0] >= -capabilities.pad_half_extent_y_m
        and y_range[1] <= capabilities.pad_half_extent_y_m
    )
    if within_pad:
        mapped.append("tool_footprint_within_pad")
    else:
        unsupported.append("tool_footprint_exceeds_pad")

    numerics = audit_vertical_support(
        mass_kg=spec.effective_mass_kg,
        support_stiffness_n_m=spec.support_stiffness_n_m,
        damping_ratio=spec.damping_ratio,
        dt_s=1.0 / capabilities.native_rate_hz,
    )
    if numerics.discrete_stable:
        mapped.append("support_discretization")
    else:
        unsupported.append("support_discretization_unstable")

    return PhysicsCompatibilityAudit(
        scenario_id=spec.scenario_id,
        backend_name=capabilities.backend_name,
        supported=not unsupported,
        mapped_factors=tuple(sorted(mapped)),
        unsupported_factors=tuple(sorted(unsupported)),
        path_within_pad=within_pad,
        path_x_range_m=x_range,
        path_y_range_m=y_range,
        support_numerics=numerics,
    )


def summarize_compatibility(
    scenarios: Iterable[ScenarioSpec],
    capabilities: PhysicsBackendCapabilities,
) -> dict[str, object]:
    audits = tuple(assess_physics_compatibility(item, capabilities) for item in scenarios)
    reason_counts: dict[str, int] = {}
    for audit in audits:
        for reason in audit.unsupported_factors:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return {
        "backend_name": capabilities.backend_name,
        "scenario_count": len(audits),
        "supported_count": sum(item.supported for item in audits),
        "discrete_stable_count": sum(
            item.support_numerics.discrete_stable for item in audits
        ),
        "path_within_pad_count": sum(item.path_within_pad for item in audits),
        "unsupported_reason_counts": dict(sorted(reason_counts.items())),
    }
