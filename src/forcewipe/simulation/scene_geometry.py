"""Deterministic geometry contracts for the dedicated ForceWipe V4 scene."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.spatial.transform import Rotation

from forcewipe.simulation.scenarios import ScenarioPath, ScenarioSpec, make_scenario_path, surface_normal


V4_SURFACE_HALF_X_M = 0.24
V4_SURFACE_HALF_Y_M = 0.09
V4_SURFACE_THICKNESS_M = 0.02
V4_SURFACE_TOP_ORIGIN_Z_M = 0.04
V4_CYLINDER_CAP_THICKNESS_M = 0.005
V4_TOOL_THICKNESS_M = 0.008
V4_TOOL_TCP_CENTER_OFFSET_M = 0.025


class SceneGeometryError(ValueError):
    """Invalid or internally inconsistent V4 physical geometry."""


@dataclass(frozen=True)
class SurfaceGeometry:
    geometry_kind: str
    position_xyz_m: np.ndarray
    quaternion_wxyz: np.ndarray
    box_half_sizes_m: np.ndarray | None
    mesh_vertices_m: np.ndarray | None
    mesh_faces: np.ndarray | None
    top_origin_z_m: float

    def __post_init__(self) -> None:
        position = np.asarray(self.position_xyz_m, dtype=np.float64).copy()
        quaternion = np.asarray(self.quaternion_wxyz, dtype=np.float64).copy()
        if position.shape != (3,) or quaternion.shape != (4,):
            raise SceneGeometryError("surface pose has an invalid shape")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(quaternion)):
            raise SceneGeometryError("surface pose contains NaN/Inf")
        norm = float(np.linalg.norm(quaternion))
        if not np.isclose(norm, 1.0, atol=1e-12):
            raise SceneGeometryError("surface quaternion must be unit length")
        object.__setattr__(self, "position_xyz_m", position)
        object.__setattr__(self, "quaternion_wxyz", quaternion)
        for name in ("box_half_sizes_m", "mesh_vertices_m", "mesh_faces"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value).copy()
                array.setflags(write=False)
                object.__setattr__(self, name, array)
        position.setflags(write=False)
        quaternion.setflags(write=False)


@dataclass(frozen=True)
class ToolGeometry:
    half_sizes_xyz_m: np.ndarray
    normal_stiffness_n_m: float

    def __post_init__(self) -> None:
        half_sizes = np.asarray(self.half_sizes_xyz_m, dtype=np.float64).copy()
        if half_sizes.shape != (3,) or not np.all(np.isfinite(half_sizes)):
            raise SceneGeometryError("tool half sizes must be a finite three-vector")
        if np.any(half_sizes <= 0.0) or self.normal_stiffness_n_m <= 0.0:
            raise SceneGeometryError("tool geometry and stiffness must be positive")
        half_sizes.setflags(write=False)
        object.__setattr__(self, "half_sizes_xyz_m", half_sizes)


@dataclass(frozen=True)
class ObstacleGeometry:
    position_xyz_m: np.ndarray
    quaternion_wxyz: np.ndarray
    half_sizes_xyz_m: np.ndarray
    center_progress: float
    half_width_progress: float

    def __post_init__(self) -> None:
        for name, shape in (
            ("position_xyz_m", (3,)),
            ("quaternion_wxyz", (4,)),
            ("half_sizes_xyz_m", (3,)),
        ):
            value = np.asarray(getattr(self, name), dtype=np.float64).copy()
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise SceneGeometryError(f"{name} has an invalid value")
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if np.any(self.half_sizes_xyz_m <= 0.0):
            raise SceneGeometryError("obstacle half sizes must be positive")


def _quat_about_y(angle_rad: float) -> np.ndarray:
    half = 0.5 * float(angle_rad)
    return np.array([math.cos(half), 0.0, math.sin(half), 0.0], dtype=np.float64)


def convex_cylinder_cap_mesh(
    *,
    radius_m: float,
    half_extent_x_m: float = V4_SURFACE_HALF_X_M,
    half_extent_y_m: float = V4_SURFACE_HALF_Y_M,
    thickness_m: float = V4_SURFACE_THICKNESS_M,
    arc_segments: int = 48,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a convex, y-extruded outer-cylinder cap in local coordinates."""

    radius = float(radius_m)
    half_x = float(half_extent_x_m)
    half_y = float(half_extent_y_m)
    thickness = float(thickness_m)
    segments = int(arc_segments)
    if not 0.0 < half_x < radius or half_y <= 0.0 or thickness <= 0.0:
        raise SceneGeometryError("invalid cylinder-cap dimensions")
    if segments < 8:
        raise SceneGeometryError("cylinder cap requires at least eight arc segments")

    x = np.linspace(-half_x, half_x, segments + 1, dtype=np.float64)
    top_z = np.sqrt(radius * radius - x * x) - radius
    bottom_z = float(top_z.min() - thickness)
    cross_section = np.column_stack(
        (
            np.concatenate((x, [half_x, -half_x])),
            np.concatenate((top_z, [bottom_z, bottom_z])),
        )
    )
    count = len(cross_section)
    vertices = np.empty((2 * count, 3), dtype=np.float64)
    vertices[:count, 0] = cross_section[:, 0]
    vertices[:count, 1] = -half_y
    vertices[:count, 2] = cross_section[:, 1]
    vertices[count:, 0] = cross_section[:, 0]
    vertices[count:, 1] = half_y
    vertices[count:, 2] = cross_section[:, 1]

    faces: list[tuple[int, int, int]] = []
    for index in range(1, count - 1):
        faces.append((0, index + 1, index))
        faces.append((count, count + index, count + index + 1))
    for index in range(count):
        nxt = (index + 1) % count
        faces.append((index, nxt, count + nxt))
        faces.append((index, count + nxt, count + index))
    output_faces = np.asarray(faces, dtype=np.int32)
    vertices.setflags(write=False)
    output_faces.setflags(write=False)
    return vertices, output_faces


def wavefront_obj_text(vertices_m: np.ndarray, faces: np.ndarray) -> str:
    """Serialize one triangular mesh for SAPIEN's convex-file loader."""

    vertices = np.asarray(vertices_m, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4:
        raise SceneGeometryError("OBJ vertices must have shape (N>=4, 3)")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or len(triangles) < 4:
        raise SceneGeometryError("OBJ faces must have shape (M>=4, 3)")
    if not np.all(np.isfinite(vertices)):
        raise SceneGeometryError("OBJ vertices contain NaN/Inf")
    if np.any(triangles < 0) or np.any(triangles >= len(vertices)):
        raise SceneGeometryError("OBJ face index is outside the vertex array")
    lines = ["# ForceWipe V4 deterministic convex surface"]
    lines.extend(
        f"v {float(x):.17g} {float(y):.17g} {float(z):.17g}"
        for x, y, z in vertices
    )
    lines.extend(
        f"f {int(a) + 1} {int(b) + 1} {int(c) + 1}"
        for a, b, c in triangles
    )
    return "\n".join(lines) + "\n"


def make_surface_geometry(
    spec: ScenarioSpec,
    *,
    top_origin_z_m: float = V4_SURFACE_TOP_ORIGIN_Z_M,
) -> SurfaceGeometry:
    """Compile one scenario surface into a box or convex-cap actor contract."""

    spec.validate()
    origin_z = float(top_origin_z_m)
    if not math.isfinite(origin_z):
        raise SceneGeometryError("top_origin_z_m must be finite")
    center_x = 0.5 * spec.path_length_m
    if spec.surface_kind == "cylinder_low":
        half_extent_x = (
            0.5 * spec.path_length_m
            + 0.5 * spec.tool_footprint_length_m
            + 0.005
        )
        half_extent_y = max(
            0.06,
            spec.path_lateral_scale_m + 0.5 * spec.tool_width_m + 0.01,
        )
        vertices, faces = convex_cylinder_cap_mesh(
            radius_m=spec.cylinder_radius_m,
            half_extent_x_m=half_extent_x,
            half_extent_y_m=half_extent_y,
            thickness_m=V4_CYLINDER_CAP_THICKNESS_M,
        )
        if origin_z + float(vertices[:, 2].min()) <= 0.0:
            raise SceneGeometryError("cylinder cap would initially penetrate the table")
        return SurfaceGeometry(
            geometry_kind="convex_cylinder_cap",
            position_xyz_m=np.array([center_x, 0.0, origin_z]),
            quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            box_half_sizes_m=None,
            mesh_vertices_m=vertices,
            mesh_faces=faces,
            top_origin_z_m=origin_z,
        )

    angle_deg = 0.0
    if spec.surface_kind == "incline_5":
        angle_deg = 5.0
    elif spec.surface_kind == "incline_10":
        angle_deg = 10.0
    elif spec.surface_kind != "flat":
        raise SceneGeometryError("unsupported surface kind")
    angle = math.radians(angle_deg)
    thickness = V4_SURFACE_THICKNESS_M
    top_center = np.array(
        [center_x, 0.0, origin_z + math.tan(angle) * center_x],
        dtype=np.float64,
    )
    rotated_top_offset = np.array(
        [-math.sin(angle) * thickness, 0.0, math.cos(angle) * thickness],
        dtype=np.float64,
    )
    return SurfaceGeometry(
        geometry_kind="box",
        position_xyz_m=top_center - rotated_top_offset,
        quaternion_wxyz=_quat_about_y(-angle),
        box_half_sizes_m=np.array(
            [V4_SURFACE_HALF_X_M, V4_SURFACE_HALF_Y_M, thickness],
            dtype=np.float64,
        ),
        mesh_vertices_m=None,
        mesh_faces=None,
        top_origin_z_m=origin_z,
    )


def make_tool_geometry(spec: ScenarioSpec) -> ToolGeometry:
    spec.validate()
    return ToolGeometry(
        half_sizes_xyz_m=np.array(
            [
                0.5 * spec.tool_footprint_length_m,
                0.5 * spec.tool_width_m,
                0.5 * V4_TOOL_THICKNESS_M,
            ],
            dtype=np.float64,
        ),
        normal_stiffness_n_m=spec.tool_normal_stiffness_n_m,
    )


def make_obstacle_geometry(
    spec: ScenarioSpec,
    *,
    path: ScenarioPath | None = None,
    moving: bool = False,
) -> ObstacleGeometry:
    """Place one path-aligned obstacle or keep-out marker on the surface."""

    spec.validate()
    path = make_scenario_path(spec) if path is None else path
    center = float(spec.obstacle_center_s)
    point, tangent = path.at(center)
    normal = np.asarray(surface_normal(spec, point[0]), dtype=np.float64)
    normal /= np.linalg.norm(normal)
    tangent -= float(np.dot(tangent, normal)) * normal
    tangent /= np.linalg.norm(tangent)
    lateral = np.cross(normal, tangent)
    lateral /= np.linalg.norm(lateral)
    rotation = np.column_stack((tangent, lateral, normal))
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    quaternion_wxyz = np.array(
        [
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        ],
        dtype=np.float64,
    )
    half_progress = max(float(spec.obstacle_half_width_s), 0.04 if moving else 0.025)
    half_length = max(0.006, half_progress * path.total_length)
    half_height = 0.014 if moving else 0.012
    half_lateral = 0.008
    world_point = point.copy()
    world_point[2] += V4_SURFACE_TOP_ORIGIN_Z_M
    position = world_point + normal * half_height
    return ObstacleGeometry(
        position_xyz_m=position,
        quaternion_wxyz=quaternion_wxyz,
        half_sizes_xyz_m=np.array(
            [half_length, half_lateral, half_height], dtype=np.float64
        ),
        center_progress=center,
        half_width_progress=half_progress,
    )


def surface_world_height(
    spec: ScenarioSpec,
    x_m: np.ndarray | float,
    *,
    top_origin_z_m: float = V4_SURFACE_TOP_ORIGIN_Z_M,
) -> np.ndarray:
    x = np.asarray(x_m, dtype=np.float64)
    if spec.surface_kind == "flat":
        relative = np.zeros_like(x)
    elif spec.surface_kind in {"incline_5", "incline_10"}:
        angle = 5.0 if spec.surface_kind == "incline_5" else 10.0
        relative = np.tan(np.deg2rad(angle)) * x
    elif spec.surface_kind == "cylinder_low":
        centered = x - 0.5 * spec.path_length_m
        radius = spec.cylinder_radius_m
        if np.any(np.abs(centered) >= radius):
            raise SceneGeometryError("height query exceeds cylinder domain")
        relative = np.sqrt(radius * radius - centered * centered) - radius
    else:
        raise SceneGeometryError("unsupported surface kind")
    return relative + float(top_origin_z_m)


def surface_frame(spec: ScenarioSpec, x_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Return a surface point and outward normal for pose-reference generation."""

    point = np.array(
        [float(x_m), 0.0, float(surface_world_height(spec, float(x_m)))],
        dtype=np.float64,
    )
    normal = np.asarray(surface_normal(spec, float(x_m)), dtype=np.float64)
    return point, normal
