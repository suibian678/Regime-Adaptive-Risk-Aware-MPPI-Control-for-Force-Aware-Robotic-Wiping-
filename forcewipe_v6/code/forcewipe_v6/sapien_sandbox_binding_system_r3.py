"""Exact physical-binding identity for the System R3 rev2 SANDBOX candidate.

The geometric and Cartesian contracts are intentionally unchanged from S5.
This wrapper exists so the runtime cannot silently bind an R2/S5 supervisor
under the R3 protocol identity.  Importing this module does not import SAPIEN.
"""

from __future__ import annotations

from typing import Any

from .controller_system_r3 import (
    R3_CONTACT_LOSS_TARGET_FRACTION,
)
from .controller_system_r3_rev2 import V6SystemContactSupervisorR3Rev2
from .sapien_sandbox_binding import V6SapienSandboxContractError
from .sapien_sandbox_binding_s5 import (
    S5SapienSandboxBinding,
    S5SurfacePathGeometry,
)


class R3Rev2SurfacePathGeometry(S5SurfacePathGeometry):
    """Versioned identity for the unchanged local projection geometry."""


class R3Rev2SapienSandboxBinding(S5SapienSandboxBinding):
    """Require the exact R3 controller and R3 geometry before binding."""

    def __init__(
        self,
        env: Any,
        raw_env: Any,
        supervisor: V6SystemContactSupervisorR3Rev2,
        geometry: R3Rev2SurfacePathGeometry,
        **kwargs: Any,
    ) -> None:
        if type(supervisor) is not V6SystemContactSupervisorR3Rev2:
            raise V6SapienSandboxContractError(
                "System R3 rev2 binding requires the exact rev2 supervisor"
            )
        if type(geometry) is not R3Rev2SurfacePathGeometry:
            raise V6SapienSandboxContractError(
                "System R3 rev2 binding requires the exact rev2 geometry"
            )
        if (
            abs(
                float(supervisor.config.contact_loss_target_fraction)
                - R3_CONTACT_LOSS_TARGET_FRACTION
            )
            > 1e-12
        ):
            raise V6SapienSandboxContractError(
                "System R3 rev2 local-acquire threshold is not frozen"
            )
        super().__init__(env, raw_env, supervisor, geometry, **kwargs)


__all__ = ["R3Rev2SapienSandboxBinding", "R3Rev2SurfacePathGeometry"]
