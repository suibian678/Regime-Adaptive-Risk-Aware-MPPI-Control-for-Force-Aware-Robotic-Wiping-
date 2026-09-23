"""Exact physical binding identity for precision TRACK r4."""

from __future__ import annotations

from typing import Any

from .controller_precision_r4 import V6PrecisionContactSupervisorR4
from .sapien_sandbox_binding import V6SapienSandboxContractError
from .sapien_sandbox_binding_s5 import S5SapienSandboxBinding, S5SurfacePathGeometry


class PrecisionR4SurfacePathGeometry(S5SurfacePathGeometry):
    """Versioned identity for the unchanged S5 path geometry."""


class PrecisionR4SapienSandboxBinding(S5SapienSandboxBinding):
    """Require the exact r4 controller/geometry pair."""

    def __init__(
        self,
        env: Any,
        raw_env: Any,
        supervisor: V6PrecisionContactSupervisorR4,
        geometry: PrecisionR4SurfacePathGeometry,
        **kwargs: Any,
    ) -> None:
        if type(supervisor) is not V6PrecisionContactSupervisorR4:
            raise V6SapienSandboxContractError(
                "precision-r4 binding requires the exact r4 supervisor"
            )
        if type(geometry) is not PrecisionR4SurfacePathGeometry:
            raise V6SapienSandboxContractError(
                "precision-r4 binding requires the exact r4 geometry"
            )
        super().__init__(env, raw_env, supervisor, geometry, **kwargs)


__all__ = ["PrecisionR4SapienSandboxBinding", "PrecisionR4SurfacePathGeometry"]
