"""Physical binding for precision-r5 orthogonal low-level composition."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from .causal_adapter import CausalControlAdapter, IssuedCartesianCommand
from .causal_adapter_precision_r5 import PrecisionR5EvidenceLedger
from .controller_precision_r5 import (
    PRECISION_R5_COMBINED_REASON,
    V6PrecisionContactSupervisorR5,
)
from .sapien_sandbox_binding import V6SapienSandboxContractError
from .sapien_sandbox_binding_s5 import S5SapienSandboxBinding, S5SurfacePathGeometry


class PrecisionR5SurfacePathGeometry(S5SurfacePathGeometry):
    """Exact type identity for the r5 development profile."""


class PrecisionR5SapienSandboxBinding(S5SapienSandboxBinding):
    """Issue and read back normal plus bounded cross-track components."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if type(self.supervisor) is not V6PrecisionContactSupervisorR5:
            raise V6SapienSandboxContractError(
                "precision-r5 binding requires the exact r5 supervisor"
            )
        if type(self.geometry) is not PrecisionR5SurfacePathGeometry:
            raise V6SapienSandboxContractError(
                "precision-r5 binding requires the exact r5 geometry"
            )
        adapter_config = self.adapter.config
        self.adapter = CausalControlAdapter(
            self.supervisor,
            self._build_issued_command,
            config=adapter_config,
            ledger=PrecisionR5EvidenceLedger(adapter_config),
        )

    def _build_issued_command(self, command, pre) -> IssuedCartesianCommand:
        if command.transition_reason != PRECISION_R5_COMBINED_REASON:
            return super()._build_issued_command(command, pre)
        geometry = self._geometry_by_key.get(pre.key)
        if geometry is None or self._pre_by_key.get(pre.key) is None:
            raise V6SapienSandboxContractError(
                "precision-r5 issued command lacks pre-step geometry"
            )
        if (
            command.recovery_reposition_permitted
            or not command.cross_track_correction_permitted
            or abs(float(command.executed_tangential_step_m)) > 1e-15
        ):
            raise V6SapienSandboxContractError(
                "precision-r5 combined authority flags are inconsistent"
            )
        normal_step = float(command.executed_normal_step_m)
        normal_delta = -geometry.command_outward_normal_xyz * normal_step
        cross_delta = np.asarray(
            pre.requested_cross_track_correction_delta_xyz_m,
            dtype=np.float64,
        )
        total = normal_delta + cross_delta
        normalized = total / self.config.position_action_scale_m
        if np.any(np.abs(normalized) > 1.0 + 1e-12):
            raise V6SapienSandboxContractError(
                "precision-r5 combined Cartesian command would be clipped downstream"
            )
        command_id = ":".join(
            (
                pre.key.run_id,
                str(pre.key.scenario_id),
                str(pre.key.evaluation_id),
                str(pre.key.segment_index),
                str(pre.key.control_step_index),
            )
        )
        issued = IssuedCartesianCommand(
            key=pre.key,
            command_id=command_id,
            issued_time_ns=int(pre.pre_time_ns),
            command_frame_id=self.config.command_frame_id,
            command_mode=self.config.command_mode,
            normal_step_m=normal_step,
            task_tangential_step_m=0.0,
            recovery_reposition_permitted=False,
            cross_track_correction_permitted=True,
            normal_delta_xyz_m=tuple(float(value) for value in normal_delta),
            task_delta_xyz_m=(0.0, 0.0, 0.0),
            recovery_delta_xyz_m=(0.0, 0.0, 0.0),
            cross_track_correction_delta_xyz_m=tuple(
                float(value) for value in cross_delta
            ),
            cartesian_delta_xyz_m=tuple(float(value) for value in total),
            recovery_rotation_delta_euler_xyz_rad=(0.0, 0.0, 0.0),
            rotation_delta_euler_xyz_rad=(0.0, 0.0, 0.0),
        )
        issued.validate()
        self._pending_command_id = command_id
        return issued

    def _make_readback(self, issued, *args, **kwargs):
        readback, evidence = super()._make_readback(issued, *args, **kwargs)
        if not (
            issued.cross_track_correction_permitted
            and abs(float(issued.normal_step_m)) > 0.0
        ):
            return readback, evidence
        # The base physical check has already established that the actuator's
        # final target equals the full issued vector.  Restore the registered
        # orthogonal decomposition for ledger-level component readback.
        readback = replace(
            readback,
            normal_step_m=issued.normal_step_m,
            normal_delta_xyz_m=issued.normal_delta_xyz_m,
            cross_track_correction_delta_xyz_m=(
                issued.cross_track_correction_delta_xyz_m
            ),
            cartesian_delta_xyz_m=issued.cartesian_delta_xyz_m,
        )
        return readback, evidence


__all__ = [
    "PrecisionR5SapienSandboxBinding",
    "PrecisionR5SurfacePathGeometry",
]
