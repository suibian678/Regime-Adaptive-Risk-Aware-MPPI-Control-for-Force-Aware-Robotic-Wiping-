"""Nonphysical replay utilities for the frozen V5.7 counterexample.

The recorded post-command force is retained only as an audit label.  It is
never passed into the V6 controller command that precedes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from .controller import (
    SupervisorCommand,
    SupervisorInput,
    SupervisorSnapshot,
    SupervisorState,
    UnifiedCausalForceRecoverySupervisor,
    V6ControlError,
)


DEFAULT_V57_RESULT = Path(
    "/tmp/forcewipe_v5_diagnostics/v5_7_r3_rev2_dev_once"
)
EXPECTED_EVALUATION_ID = 557_310_011
EXPECTED_DECISION_STEP = 1032


@dataclass(frozen=True)
class FrozenCounterexample:
    evaluation_id: int
    decision_step: int
    snapshot: SupervisorSnapshot
    causal_input: SupervisorInput
    future_observed_force_n: float
    future_force_used_by_controller: bool = False


@dataclass(frozen=True)
class CounterfactualReplayResult:
    source: FrozenCounterexample
    command: SupervisorCommand

    @property
    def causal_gate_passed(self) -> bool:
        return bool(
            not self.source.future_force_used_by_controller
            and self.command.state_before == SupervisorState.TRACK.value
            and self.command.state_after == SupervisorState.REBOUND_GUARD.value
            and self.command.rebound_guard_triggered
            and self.command.executed_normal_step_m < 0.0
            and not self.command.tangential_motion_permitted
            and self.command.committed_progress_after
            == self.command.committed_progress_before
        )


def _one(frame: pl.DataFrame, *, evaluation_id: int, step: int) -> dict:
    rows = frame.filter(
        (pl.col("evaluation_id") == evaluation_id) & (pl.col("step") == step)
    )
    if rows.height != 1:
        raise V6ControlError(
            f"expected exactly one frozen row for evaluation={evaluation_id}, step={step}"
        )
    return rows.row(0, named=True)


def load_v57_rebound_counterexample(
    root: Path = DEFAULT_V57_RESULT,
) -> FrozenCounterexample:
    native = pl.read_parquet(root / "first_pass_native_trace.parquet")
    diagnostic = pl.read_parquet(root / "controller_diagnostics.parquet")

    d1030 = _one(diagnostic, evaluation_id=EXPECTED_EVALUATION_ID, step=1030)
    d1031 = _one(diagnostic, evaluation_id=EXPECTED_EVALUATION_ID, step=1031)
    d1032 = _one(diagnostic, evaluation_id=EXPECTED_EVALUATION_ID, step=1032)
    n1029 = _one(native, evaluation_id=EXPECTED_EVALUATION_ID, step=1029)
    n1030 = _one(native, evaluation_id=EXPECTED_EVALUATION_ID, step=1030)
    n1031 = _one(native, evaluation_id=EXPECTED_EVALUATION_ID, step=1031)
    n1032 = _one(native, evaluation_id=EXPECTED_EVALUATION_ID, step=1032)

    expected = {
        "n1030": 10.056784629821777,
        "n1031": 3.5712366104125977,
        "n1032": 17.31221580505371,
        "previous_command": 0.00392049299812317,
    }
    observed = {
        "n1030": float(n1030["f_aud_n"]),
        "n1031": float(n1031["f_aud_n"]),
        "n1032": float(n1032["f_aud_n"]),
        "previous_command": float(d1031["clipped_normal_step_m"]),
    }
    if any(abs(observed[name] - value) > 1e-12 for name, value in expected.items()):
        raise V6ControlError("frozen V5.7 counterexample identity mismatch")

    snapshot = SupervisorSnapshot(
        state=SupervisorState.TRACK.value,
        state_dwell_samples=100,
        contact_verify_count=0,
        headroom_exit_count=0,
        previous_force_n=float(n1030["f_aud_n"]),
        previous_previous_force_n=float(n1029["f_aud_n"]),
        previous_normal_command_m=float(d1031["clipped_normal_step_m"]),
        previous_previous_normal_command_m=float(d1030["clipped_normal_step_m"]),
        integral_error_n_s=float(d1031["integral_error_n_s"]),
        committed_progress=float(d1031["committed_progress"]),
        episode_total_recovery_cycles=int(d1031["v57_total_recovery_cycles"]),
        episode_total_recovery_samples=int(
            d1031["v57_total_recovery_cycles"]
        ),
        pre_recovery_highwater_progress=None,
        verified_return_count=0,
        in_recovery_episode=False,
    )
    causal_input = SupervisorInput(
        measured_force_n=float(n1031["f_aud_n"]),
        target_force_n=12.0,
        normal_velocity_outward_m_s=float(d1032["normal_velocity_m_s"]),
        contact_observed=bool(n1031["contact"]),
        instantaneous_progress=float(n1031["progress"]),
        geometric_track_error_m=0.0,
        requested_tangential_step_m=0.001,
        recovery_hover_pose_error_m=0.0,
        recovery_clearance_m=0.0,
    )
    return FrozenCounterexample(
        evaluation_id=EXPECTED_EVALUATION_ID,
        decision_step=EXPECTED_DECISION_STEP,
        snapshot=snapshot,
        causal_input=causal_input,
        future_observed_force_n=float(n1032["f_aud_n"]),
    )


def replay_v57_rebound_counterexample(
    supervisor: UnifiedCausalForceRecoverySupervisor | None = None,
    root: Path = DEFAULT_V57_RESULT,
) -> CounterfactualReplayResult:
    source = load_v57_rebound_counterexample(root)
    controller = supervisor or UnifiedCausalForceRecoverySupervisor()
    controller.restore(source.snapshot)
    command = controller.command(source.causal_input)
    return CounterfactualReplayResult(source=source, command=command)
