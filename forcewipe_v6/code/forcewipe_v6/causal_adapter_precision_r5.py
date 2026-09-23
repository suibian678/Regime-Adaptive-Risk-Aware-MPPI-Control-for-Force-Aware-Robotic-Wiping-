"""Versioned causal-ledger contract for precision-r5 orthogonal composition."""

from __future__ import annotations

from dataclasses import replace

from .causal_adapter import (
    CausalEvidenceLedger,
    CausalPreStepSample,
    IssuedCartesianCommand,
    V6AdapterContractError,
    _add,
    _close,
    _scale,
    _vector_close,
)
from .controller import SupervisorCommand, SupervisorState
from .controller_precision_r5 import PRECISION_R5_COMBINED_REASON


class PrecisionR5EvidenceLedger(CausalEvidenceLedger):
    """Accept one explicitly bounded normal-plus-cross-track TRACK command."""

    def _validate_issued(
        self,
        pre: CausalPreStepSample,
        command: SupervisorCommand,
        issued: IssuedCartesianCommand,
    ) -> None:
        if command.transition_reason != PRECISION_R5_COMBINED_REASON:
            super()._validate_issued(pre, command, issued)
            return

        tolerance = self.config.command_readback_tolerance_m
        valid_state = bool(
            command.state_before == SupervisorState.TRACK.value
            and command.state_after == SupervisorState.TRACK.value
            and not command.transitioned
            and not command.stable_track
            and not command.tangential_motion_permitted
            and not command.recovery_reposition_permitted
            and command.cross_track_correction_permitted
            and command.projection_active
            and abs(command.executed_tangential_step_m) <= tolerance
            and abs(
                command.committed_progress_after
                - command.committed_progress_before
            )
            <= tolerance
            and pre.geometric_track_error_m
            >= self.config.cross_track_correction_enter_m - tolerance
        )
        if not valid_state:
            raise V6AdapterContractError(
                "precision-r5 composition is outside registered TRACK hold authority"
            )

        expected_normal = _scale(
            pre.command_outward_normal_xyz,
            -float(command.executed_normal_step_m),
        )
        expected_cross = pre.requested_cross_track_correction_delta_xyz_m
        expected_total = _add(expected_normal, expected_cross)
        if not _close(issued.normal_step_m, command.executed_normal_step_m, tolerance):
            raise V6AdapterContractError("precision-r5 normal scalar is inconsistent")
        for actual, expected, name in (
            (issued.normal_delta_xyz_m, expected_normal, "normal component"),
            (
                issued.cross_track_correction_delta_xyz_m,
                expected_cross,
                "cross-track component",
            ),
            (issued.cartesian_delta_xyz_m, expected_total, "total component"),
        ):
            if not _vector_close(actual, expected, tolerance):
                raise V6AdapterContractError(
                    f"precision-r5 issued {name} is inconsistent"
                )

        # Reuse the complete historical cross-track contract after removing the
        # newly registered orthogonal normal component.  History fields and the
        # real command identity remain unchanged.
        surrogate_command = replace(
            command,
            transition_reason="track_cross_track_correction_projection",
            executed_normal_step_m=0.0,
            integral_after_n_s=command.integral_before_n_s,
        )
        surrogate_issued = replace(
            issued,
            normal_step_m=0.0,
            normal_delta_xyz_m=(0.0, 0.0, 0.0),
            cartesian_delta_xyz_m=tuple(float(value) for value in expected_cross),
        )
        super()._validate_issued(pre, surrogate_command, surrogate_issued)


__all__ = ["PrecisionR5EvidenceLedger"]
