"""Exact physical binding identity for precision TRACK r2."""

from __future__ import annotations

from typing import Any

from .controller_precision_r2 import V6PrecisionContactSupervisorR2
from .sapien_sandbox_binding import V6SapienSandboxContractError
from .sapien_sandbox_binding_s5 import S5SapienSandboxBinding, S5SurfacePathGeometry


class PrecisionR2SurfacePathGeometry(S5SurfacePathGeometry):
    """Versioned identity for unchanged S5 path geometry."""


class PrecisionR2SapienSandboxBinding(S5SapienSandboxBinding):
    """Require the exact r2 controller/geometry pair."""

    def __init__(
        self,
        env: Any,
        raw_env: Any,
        supervisor: V6PrecisionContactSupervisorR2,
        geometry: PrecisionR2SurfacePathGeometry,
        **kwargs: Any,
    ) -> None:
        if type(supervisor) is not V6PrecisionContactSupervisorR2:
            raise V6SapienSandboxContractError(
                "precision-r2 binding requires the exact r2 supervisor"
            )
        if type(geometry) is not PrecisionR2SurfacePathGeometry:
            raise V6SapienSandboxContractError(
                "precision-r2 binding requires the exact r2 geometry"
            )
        super().__init__(env, raw_env, supervisor, geometry, **kwargs)


__all__ = ["PrecisionR2SapienSandboxBinding", "PrecisionR2SurfacePathGeometry"]
