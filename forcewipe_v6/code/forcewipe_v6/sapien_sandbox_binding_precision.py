"""Physical binding identity for the exact-target precision SANDBOX."""

from __future__ import annotations

from typing import Any

from .controller_precision import V6PrecisionContactSupervisor
from .sapien_sandbox_binding import V6SapienSandboxContractError
from .sapien_sandbox_binding_s5 import (
    S5SapienSandboxBinding,
    S5SurfacePathGeometry,
)


class PrecisionSurfacePathGeometry(S5SurfacePathGeometry):
    """Versioned identity for unchanged path-frame geometry."""


class PrecisionSapienSandboxBinding(S5SapienSandboxBinding):
    """Reject any controller or geometry other than the precision pair."""

    def __init__(
        self,
        env: Any,
        raw_env: Any,
        supervisor: V6PrecisionContactSupervisor,
        geometry: PrecisionSurfacePathGeometry,
        **kwargs: Any,
    ) -> None:
        if type(supervisor) is not V6PrecisionContactSupervisor:
            raise V6SapienSandboxContractError(
                "precision binding requires the exact precision supervisor"
            )
        if type(geometry) is not PrecisionSurfacePathGeometry:
            raise V6SapienSandboxContractError(
                "precision binding requires the exact precision geometry"
            )
        super().__init__(env, raw_env, supervisor, geometry, **kwargs)


__all__ = ["PrecisionSapienSandboxBinding", "PrecisionSurfacePathGeometry"]
