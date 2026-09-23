"""Versioned physical binding for the System R4 development candidate.

The causal geometry and executor are unchanged from S5.  R4's two internal
subattempt ledgers are deliberately kept in the controller snapshot and are
reconstructible from versioned transition reasons.  The binding closes the
existing immutable diagnostic schema by returning the base command fields to
the storage layer while retaining the full R4 command in ``last_r4_command``.
No simulator package is imported by this module.
"""

from __future__ import annotations

from dataclasses import asdict, fields, replace
from typing import Any

from .controller import SupervisorCommand
from .controller_system_r3 import R3_CONTACT_LOSS_TARGET_FRACTION
from .controller_system_r4 import (
    R4SupervisorCommand,
    V6SystemContactSupervisorR4,
)
from .sapien_sandbox_binding import V6SapienSandboxContractError
from .sapien_sandbox_binding_s5 import (
    S5SapienSandboxBinding,
    S5SurfacePathGeometry,
)


class R4SurfacePathGeometry(S5SurfacePathGeometry):
    """Versioned identity for the unchanged dual-frame S5 geometry."""


class R4SapienSandboxBinding(S5SapienSandboxBinding):
    """Require the exact R4 controller and expose schema-closed evidence."""

    def __init__(
        self,
        env: Any,
        raw_env: Any,
        supervisor: V6SystemContactSupervisorR4,
        geometry: R4SurfacePathGeometry,
        **kwargs: Any,
    ) -> None:
        if type(supervisor) is not V6SystemContactSupervisorR4:
            raise V6SapienSandboxContractError(
                "System R4 binding requires the exact R4 supervisor"
            )
        if type(geometry) is not R4SurfacePathGeometry:
            raise V6SapienSandboxContractError(
                "System R4 binding requires the exact R4 geometry"
            )
        if (
            abs(
                float(supervisor.config.contact_loss_target_fraction)
                - R3_CONTACT_LOSS_TARGET_FRACTION
            )
            > 1e-12
        ):
            raise V6SapienSandboxContractError(
                "System R4 local-reacquire threshold is not frozen"
            )
        self.last_r4_command: R4SupervisorCommand | None = None
        super().__init__(env, raw_env, supervisor, geometry, **kwargs)

    def step(self):
        bundle = super().step()
        command = bundle.supervisor_command
        if type(command) is not R4SupervisorCommand:
            raise V6SapienSandboxContractError(
                "System R4 supervisor did not return an R4 command"
            )
        self.last_r4_command = command
        payload = asdict(command)
        base_command = SupervisorCommand(
            **{field.name: payload[field.name] for field in fields(SupervisorCommand)}
        )
        closed = replace(bundle, supervisor_command=base_command)
        self._last_bundle = closed
        return closed


__all__ = ["R4SapienSandboxBinding", "R4SurfacePathGeometry"]
