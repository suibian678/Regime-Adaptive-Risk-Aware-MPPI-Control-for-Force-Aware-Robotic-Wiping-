"""Exact physical binding identity for precision TRACK r3."""

from __future__ import annotations

from typing import Any

from .controller_precision_r3 import V6PrecisionContactSupervisorR3
from .sapien_sandbox_binding import V6SapienSandboxContractError
from .sapien_sandbox_binding_s5 import S5SapienSandboxBinding, S5SurfacePathGeometry


class PrecisionR3SurfacePathGeometry(S5SurfacePathGeometry):
    """Versioned identity for the unchanged S5 path geometry."""


class PrecisionR3SapienSandboxBinding(S5SapienSandboxBinding):
    """Require the exact r3 controller/geometry pair."""

    def __init__(
        self,
        env: Any,
        raw_env: Any,
        supervisor: V6PrecisionContactSupervisorR3,
        geometry: PrecisionR3SurfacePathGeometry,
        **kwargs: Any,
    ) -> None:
        if type(supervisor) is not V6PrecisionContactSupervisorR3:
            raise V6SapienSandboxContractError(
                "precision-r3 binding requires the exact r3 supervisor"
            )
        if type(geometry) is not PrecisionR3SurfacePathGeometry:
            raise V6SapienSandboxContractError(
                "precision-r3 binding requires the exact r3 geometry"
            )
        super().__init__(env, raw_env, supervisor, geometry, **kwargs)


__all__ = ["PrecisionR3SapienSandboxBinding", "PrecisionR3SurfacePathGeometry"]
