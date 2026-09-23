"""Causal SAPIEN/ManiSkill adapter for ForceWipe V4 native-rate logging."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
import math
from typing import Any

import numpy as np

from .information_flow import TrackingForceMeasurement
from .native_bridge import NativeStepInput, NativeStepRecord, V4NativeStepBridge
from .residual import ResidualContractError


@dataclass(frozen=True)
class ControlStepForceAudit:
    """Force-stream provenance for one completed control interval."""

    control_step_index: int
    tracking_input_force_n: float
    tracking_input_source_native_sample_index: int | None
    tracking_input_time_ns: int
    first_native_sample_index: int
    last_native_sample_index: int
    native_sample_count: int
    end_tracking_force_n: float
    last_audit_force_n: float
    end_tracking_minus_last_audit_n: float


class SapienNativeRateAdapter:
    """Owns the native-rate hook and the causal tracking-force hold.

    The wrapped ForceWipe task computes both force values through
    ``raw_env._normal_force()``.  Their semantics nevertheless differ:

    * the audit stream is sampled after every PhysX substep;
    * the control input is sampled immediately before ``env.step`` and held
      over that control interval;
    * ``info['normal_force']`` is sampled after the final substep and is the
      control-rate tracking/metric observation for the completed interval.

    The end-of-step value is never fed backward into native samples that have
    already occurred.
    """

    def __init__(
        self,
        raw_env: Any,
        bridge: V4NativeStepBridge,
        *,
        native_hz: int = 100,
        expected_native_steps_per_control: int | None = None,
        record_sink: Callable[[NativeStepRecord], None] | None = None,
        tracking_measurement_transform: Callable[
            [TrackingForceMeasurement], TrackingForceMeasurement
        ]
        | None = None,
    ) -> None:
        if int(native_hz) <= 0 or 1_000_000_000 % int(native_hz) != 0:
            raise ResidualContractError("native_hz must exactly divide one second")
        if not callable(getattr(raw_env, "_normal_force", None)):
            raise ResidualContractError("raw environment does not expose _normal_force")
        if not callable(getattr(raw_env, "_after_simulation_step", None)):
            raise ResidualContractError(
                "raw environment does not expose _after_simulation_step"
            )
        inferred = getattr(raw_env, "_sim_steps_per_control", None)
        expected = (
            int(inferred)
            if expected_native_steps_per_control is None and inferred is not None
            else expected_native_steps_per_control
        )
        if expected is None or int(expected) <= 0:
            raise ResidualContractError(
                "expected native steps per control interval must be positive"
            )
        self.raw_env = raw_env
        self.bridge = bridge
        self.native_hz = int(native_hz)
        self.period_ns = 1_000_000_000 // self.native_hz
        self.expected_native_steps_per_control = int(expected)
        self.record_sink = record_sink
        self.tracking_measurement_transform = tracking_measurement_transform
        self._original_after_simulation_step = raw_env._after_simulation_step
        self._installed_wrapper: Callable[..., Any] | None = None
        self._installed = False
        self._active_control_step: int | None = None
        self._next_control_step_index = 0
        self._next_native_sample_index = 0
        self._time_ns = 0
        self._control_first_native_index: int | None = None
        self._control_native_count = 0
        self._pending_tracking_measurement: TrackingForceMeasurement | None = None
        self._tracking_input: TrackingForceMeasurement | None = None
        self._last_tcp_position: np.ndarray | None = None
        self._last_record: NativeStepRecord | None = None

    @staticmethod
    def _scalar(value: Any, *, name: str) -> float:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=np.float64).reshape(-1)
        if array.size != 1 or not math.isfinite(float(array[0])):
            raise ResidualContractError(f"{name} must contain one finite scalar")
        return float(array[0])

    @staticmethod
    def _position(value: Any) -> np.ndarray:
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        array = np.asarray(value, dtype=np.float64)
        if array.shape == (1, 3):
            array = array[0]
        if array.shape != (3,) or not np.all(np.isfinite(array)):
            raise ResidualContractError("TCP position must be one finite three-vector")
        return array.copy()

    def _read_normal_force(self) -> float:
        return self._scalar(self.raw_env._normal_force(), name="normal force")

    def _read_tcp_position(self) -> np.ndarray:
        try:
            position = self.raw_env.agent.tcp.pose.p
        except AttributeError as exc:
            raise ResidualContractError(
                "raw environment does not expose agent.tcp.pose.p"
            ) from exc
        return self._position(position)

    def install(self) -> None:
        if self._installed:
            raise ResidualContractError("native-rate adapter is already installed")
        if self.raw_env._after_simulation_step != self._original_after_simulation_step:
            raise ResidualContractError("native hook changed before adapter installation")

        original = self._original_after_simulation_step

        def wrapped_after_simulation_step(*args: Any, **kwargs: Any) -> Any:
            if self._active_control_step is None:
                raise ResidualContractError(
                    "native physics advanced outside SapienNativeRateAdapter.step"
                )
            result = original(*args, **kwargs)
            self._capture_native_sample()
            return result

        self.raw_env._after_simulation_step = wrapped_after_simulation_step
        self._installed_wrapper = wrapped_after_simulation_step
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._active_control_step is not None:
            raise ResidualContractError("cannot uninstall during an active control step")
        if self.raw_env._after_simulation_step is not self._installed_wrapper:
            raise ResidualContractError("native hook was replaced by another component")
        self.raw_env._after_simulation_step = self._original_after_simulation_step
        self._installed_wrapper = None
        self._installed = False

    def begin_control_step(self, control_step_index: int) -> TrackingForceMeasurement:
        if not self._installed:
            raise ResidualContractError("native-rate adapter is not installed")
        if self._active_control_step is not None:
            raise ResidualContractError("a control step is already active")
        if int(control_step_index) != self._next_control_step_index:
            raise ResidualContractError(
                "control steps must be contiguous and start at zero"
            )
        raw_measurement = TrackingForceMeasurement(
            value_n=self._read_normal_force(),
            control_step_index=int(control_step_index),
            source_native_sample_index=(
                self._next_native_sample_index - 1
                if self._next_native_sample_index > 0
                else None
            ),
            time_ns=self._time_ns,
        )
        measurement = (
            raw_measurement
            if self.tracking_measurement_transform is None
            else self.tracking_measurement_transform(raw_measurement)
        )
        if not isinstance(measurement, TrackingForceMeasurement):
            raise ResidualContractError(
                "tracking measurement transform returned an invalid packet"
            )
        self._active_control_step = int(control_step_index)
        self._control_first_native_index = self._next_native_sample_index
        self._control_native_count = 0
        self._pending_tracking_measurement = measurement
        self._tracking_input = measurement
        self._last_tcp_position = self._read_tcp_position()
        return measurement

    def _capture_native_sample(self) -> None:
        if self._active_control_step is None or self._last_tcp_position is None:
            raise ResidualContractError("native sample arrived without a control context")
        position = self._read_tcp_position()
        velocity = (position - self._last_tcp_position) * float(self.native_hz)
        self._time_ns += self.period_ns
        native = NativeStepInput(
            sample_index=self._next_native_sample_index,
            time_ns=self._time_ns,
            audit_force_n=self._read_normal_force(),
            tracking_force_measurement=self._pending_tracking_measurement,
            tcp_position_xyz=position,
            tcp_velocity_xyz=velocity,
        )
        record = replace(
            self.bridge.advance(native),
            control_step_index=self._active_control_step,
            substep_index=self._control_native_count,
        )
        self._pending_tracking_measurement = None
        self._last_tcp_position = position
        self._last_record = record
        self._next_native_sample_index += 1
        self._control_native_count += 1
        if self.record_sink is not None:
            self.record_sink(record)

    @staticmethod
    def _extract_info(step_result: Any) -> Mapping[str, Any]:
        if not isinstance(step_result, tuple) or len(step_result) not in {4, 5}:
            raise ResidualContractError("environment step returned an unsupported tuple")
        info = step_result[-1]
        if not isinstance(info, Mapping):
            raise ResidualContractError("environment step did not return an info mapping")
        return info

    def finish_control_step(self, step_result: Any) -> ControlStepForceAudit:
        if self._active_control_step is None or self._tracking_input is None:
            raise ResidualContractError("no control step is active")
        if self._control_native_count != self.expected_native_steps_per_control:
            self._active_control_step = None
            raise ResidualContractError(
                "native sample count differs from the frozen control ratio"
            )
        if self._last_record is None or self._control_first_native_index is None:
            self._active_control_step = None
            raise ResidualContractError("control interval produced no native record")
        info = self._extract_info(step_result)
        if "normal_force" not in info:
            self._active_control_step = None
            raise ResidualContractError("control-rate info lacks normal_force")
        end_tracking_force = self._scalar(
            info["normal_force"], name="end tracking force"
        )
        audit = ControlStepForceAudit(
            control_step_index=self._active_control_step,
            tracking_input_force_n=self._tracking_input.value_n,
            tracking_input_source_native_sample_index=(
                self._tracking_input.source_native_sample_index
            ),
            tracking_input_time_ns=self._tracking_input.time_ns,
            first_native_sample_index=self._control_first_native_index,
            last_native_sample_index=self._last_record.sample_index,
            native_sample_count=self._control_native_count,
            end_tracking_force_n=end_tracking_force,
            last_audit_force_n=self._last_record.audit_force_n,
            end_tracking_minus_last_audit_n=(
                end_tracking_force - self._last_record.audit_force_n
            ),
        )
        self._active_control_step = None
        self._next_control_step_index += 1
        self._tracking_input = None
        self._control_first_native_index = None
        self._control_native_count = 0
        return audit

    def abort_control_step(self) -> None:
        """Clear active bookkeeping after an infrastructure exception."""
        self._active_control_step = None
        self._tracking_input = None
        self._pending_tracking_measurement = None
        self._control_first_native_index = None
        self._control_native_count = 0

    def step_with_action_factory(
        self,
        env: Any,
        action_factory: Callable[[TrackingForceMeasurement], Any],
        *,
        control_step_index: int,
    ) -> tuple[Any, ControlStepForceAudit]:
        measurement = self.begin_control_step(control_step_index)
        try:
            action = action_factory(measurement)
            result = env.step(action)
            audit = self.finish_control_step(result)
        except BaseException:
            self.abort_control_step()
            raise
        return result, audit

    def step(
        self, env: Any, action: Any, *, control_step_index: int
    ) -> tuple[Any, ControlStepForceAudit]:
        return self.step_with_action_factory(
            env,
            lambda _measurement: action,
            control_step_index=control_step_index,
        )

    @property
    def next_native_sample_index(self) -> int:
        return self._next_native_sample_index

    @property
    def time_ns(self) -> int:
        return self._time_ns
