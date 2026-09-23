"""Development-only S5 path-frame binding.

S5 uses the instantaneous surface projection frame for TRACK/acquisition normal
and tangent commands, while retaining an explicitly separate committed-progress
normal for the full-recovery hover target.  A controller-authorized cross-track
projection is a distinct request type whose bounded vector is recomputed from
causal pre-step geometry before execution.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from .controller import SupervisorState
from .controller_s5 import (
    S5_CROSS_TRACK_CORRECTION_ENTER_M,
    S5_CROSS_TRACK_CORRECTION_MAX_STEP_M,
)
from .sapien_sandbox_binding import (
    SapienSandboxBinding,
    SurfacePathGeometry,
    _Geometry,
    _bounded,
)


def bounded_cross_track_correction_xyz_m(
    *,
    contact_tool_position_xyz_m: Any,
    task_projection_point_xyz_m: Any,
    projection_outward_normal_xyz: Any,
    maximum_step_m: float = S5_CROSS_TRACK_CORRECTION_MAX_STEP_M,
) -> np.ndarray:
    """Return the bounded opposite of the surface-tangential residual."""

    tool = np.asarray(contact_tool_position_xyz_m, dtype=np.float64)
    projected = np.asarray(task_projection_point_xyz_m, dtype=np.float64)
    normal = np.asarray(projection_outward_normal_xyz, dtype=np.float64)
    residual = tool - projected
    lateral = residual - float(np.dot(residual, normal)) * normal
    return _bounded(-lateral, float(maximum_step_m))


class S5SurfacePathGeometry(SurfacePathGeometry):
    """Use the current projection frame for normal and tangent actuation."""

    def evaluate(
        self,
        contact_tool_position_xyz_m: Any,
        *,
        committed_progress: float,
    ) -> _Geometry:
        geometry = super().evaluate(
            contact_tool_position_xyz_m,
            committed_progress=committed_progress,
        )
        return replace(
            geometry,
            command_outward_normal_xyz=geometry.projection_outward_normal_xyz,
            command_tangent_unit_xyz=geometry.projection_tangent_unit_xyz,
        )


class S5SapienSandboxBinding(SapienSandboxBinding):
    """Supply a causal cross-track vector only when S5 may request it."""

    def _build_pre_step(self):
        pre, physical = super()._build_pre_step()
        geometry = self._geometry_by_key[pre.key]
        snapshot = self.supervisor.snapshot()
        if (
            snapshot.state == SupervisorState.TRACK.value
            and geometry.geometric_track_error_m
            >= S5_CROSS_TRACK_CORRECTION_ENTER_M
        ):
            correction = bounded_cross_track_correction_xyz_m(
                contact_tool_position_xyz_m=(
                    physical.state.contact_tool_position_xyz_m
                ),
                task_projection_point_xyz_m=(
                    geometry.task_projection_point_xyz_m
                ),
                projection_outward_normal_xyz=(
                    geometry.projection_outward_normal_xyz
                ),
            )
            pre = replace(
                pre,
                requested_cross_track_correction_delta_xyz_m=tuple(
                    float(value) for value in correction
                ),
            )
            physical = replace(
                physical,
                requested_cross_track_correction_delta_xyz_m=tuple(
                    float(value) for value in correction
                ),
            )
            self._pre_by_key[pre.key] = physical
        return pre, physical


__all__ = [
    "S5SapienSandboxBinding",
    "S5SurfacePathGeometry",
    "bounded_cross_track_correction_xyz_m",
]
