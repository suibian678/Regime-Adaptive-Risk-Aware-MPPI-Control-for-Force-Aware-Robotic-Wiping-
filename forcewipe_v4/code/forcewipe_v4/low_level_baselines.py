"""Auditable local-normal candidates for the V4 Study-C comparison.

The controllers share one measured-force input and one bounded normal-position
step output.  Their local force model is explicit and must be calibrated only
on DEV/CAL.  Passing unit tests is not a robustness or real-time claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import time
import warnings

import numpy as np
from scipy.optimize import minimize

from .low_level_control import ForceControlCommand, LowLevelControlError


class LowLevelBaseline(str, Enum):
    HYBRID_FORCE_POSITION = "hybrid_force_position"
    NORMAL_IMPEDANCE = "model_based_normal_impedance"
    ADAPTIVE_ADMITTANCE = "adaptive_normal_admittance_candidate"
    FORCE_CBF_QP = "force_cbf_qp"
    FORCE_CONSTRAINED_MPC = "force_constrained_mpc"


@dataclass(frozen=True)
class LocalNormalModel:
    force_retention: float = 0.0
    force_sensitivity_n_per_m: float = 2500.0
    disturbance_bound_n: float = 0.20
    force_limit_n: float = 15.0
    headroom_limit_n: float = 14.5
    minimum_contact_force_n: float = 3.0
    max_press_step_m: float = 0.0100
    max_lift_step_m: float = 0.0035
    action_position_scale_m: float = 0.1
    dt_s: float = 0.01

    def validate(self) -> None:
        numeric = tuple(
            float(getattr(self, name))
            for name in self.__dataclass_fields__
            if name != "force_retention"
        )
        if not all(math.isfinite(value) and value > 0 for value in numeric):
            raise LowLevelControlError("local-normal model parameters must be positive")
        if not math.isfinite(self.force_retention) or not 0 <= self.force_retention < 1:
            raise LowLevelControlError("force retention must lie in [0, 1)")
        if not self.minimum_contact_force_n < self.headroom_limit_n < self.force_limit_n:
            raise LowLevelControlError("local-normal force thresholds are not ordered")


@dataclass(frozen=True)
class SolverAudit:
    method: LowLevelBaseline
    solver: str
    solved: bool
    feasible_without_slack: bool
    slack_n: float
    solve_time_ms: float
    deadline_ms: float
    deadline_missed: bool
    predicted_next_force_n: float
    status: str
    solver_warning_count: int = 0


@dataclass(frozen=True)
class BaselineControlOutput:
    command: ForceControlCommand
    audit: SolverAudit


def _normal_force_rate(previous: float | None, current: float, dt_s: float) -> float:
    return 0.0 if previous is None else (float(current) - float(previous)) / float(dt_s)


def _command_from_step(
    *,
    step_m: float,
    force_n: float,
    target_force_n: float,
    force_rate_n_s: float,
    outward_normal_xyz: np.ndarray,
    model: LocalNormalModel,
    mode: str,
    integral_error_n_s: float = 0.0,
) -> ForceControlCommand:
    normal = np.asarray(outward_normal_xyz, dtype=np.float64)
    if normal.shape != (3,) or not np.all(np.isfinite(normal)):
        raise LowLevelControlError("surface normal must be a finite three-vector")
    norm = float(np.linalg.norm(normal))
    if norm <= 0:
        raise LowLevelControlError("surface normal must be nonzero")
    bounded = float(np.clip(step_m, -model.max_lift_step_m, model.max_press_step_m))
    delta = -(normal / norm) * bounded
    normalized = np.clip(delta / model.action_position_scale_m, -1.0, 1.0)
    return ForceControlCommand(
        normalized_position_action=normalized,
        normal_step_m=bounded,
        force_error_n=float(target_force_n) - float(force_n),
        force_rate_n_s=float(force_rate_n_s),
        integral_error_n_s=float(integral_error_n_s),
        mode=mode,
    )


class ModelBasedNormalImpedanceController:
    """Quasi-static normal impedance candidate using an explicit contact model."""

    def __init__(
        self,
        model: LocalNormalModel = LocalNormalModel(),
        *,
        position_gain: float = 0.35,
        velocity_damping_s: float = 0.004,
    ) -> None:
        model.validate()
        if position_gain <= 0 or velocity_damping_s < 0:
            raise LowLevelControlError("impedance gains are invalid")
        self.model = model
        self.position_gain = float(position_gain)
        self.velocity_damping_s = float(velocity_damping_s)
        self._previous_force_n: float | None = None

    def reset(self) -> None:
        self._previous_force_n = None

    def command(self, *, measured_force_n, target_force_n, outward_normal_xyz):
        force = float(measured_force_n)
        target = float(target_force_n)
        rate = _normal_force_rate(self._previous_force_n, force, self.model.dt_s)
        displacement_error = (target - force) / self.model.force_sensitivity_n_per_m
        force_feedforward = target / self.model.force_sensitivity_n_per_m
        step = force_feedforward + self.position_gain * displacement_error
        step -= self.velocity_damping_s * rate / self.model.force_sensitivity_n_per_m
        command = _command_from_step(
            step_m=step,
            force_n=force,
            target_force_n=target,
            force_rate_n_s=rate,
            outward_normal_xyz=outward_normal_xyz,
            model=self.model,
            mode=LowLevelBaseline.NORMAL_IMPEDANCE.value,
        )
        self._previous_force_n = force
        predicted = (
            self.model.force_retention * force
            + self.model.force_sensitivity_n_per_m * command.normal_step_m
        )
        return BaselineControlOutput(
            command,
            SolverAudit(
                LowLevelBaseline.NORMAL_IMPEDANCE,
                "closed_form",
                True,
                predicted <= self.model.headroom_limit_n,
                max(0.0, predicted - self.model.headroom_limit_n),
                0.0,
                10.0,
                False,
                predicted,
                "closed_form",
            ),
        )


@dataclass(frozen=True)
class AdaptiveAdmittanceConfig:
    virtual_mass_kg: float = 1.0
    virtual_damping_n_s_m: float = 80.0
    virtual_stiffness_n_m: float = 200.0
    adaptation_rate: float = 0.15
    compliance_scale_bounds: tuple[float, float] = (0.5, 2.0)

    def validate(self) -> None:
        if not all(
            math.isfinite(float(value)) and float(value) > 0
            for value in (
                self.virtual_mass_kg,
                self.virtual_damping_n_s_m,
                self.virtual_stiffness_n_m,
                self.adaptation_rate,
            )
        ):
            raise LowLevelControlError("admittance parameters must be positive")
        lower, upper = self.compliance_scale_bounds
        if not 0 < lower <= 1.0 <= upper:
            raise LowLevelControlError("admittance adaptation bounds must contain one")


class AdaptiveNormalAdmittanceController:
    def __init__(
        self,
        model: LocalNormalModel = LocalNormalModel(),
        config: AdaptiveAdmittanceConfig = AdaptiveAdmittanceConfig(),
    ) -> None:
        model.validate()
        config.validate()
        self.model = model
        self.config = config
        self.reset()

    def reset(self) -> None:
        self._virtual_position_m = 0.0
        self._virtual_velocity_m_s = 0.0
        self._compliance_scale = 1.0
        self._previous_force_n: float | None = None

    @property
    def compliance_scale(self) -> float:
        return self._compliance_scale

    def command(self, *, measured_force_n, target_force_n, outward_normal_xyz):
        force = float(measured_force_n)
        target = float(target_force_n)
        error = target - force
        cfg = self.config
        direction = 1.0 if abs(error) > 0.5 else -1.0
        self._compliance_scale = float(
            np.clip(
                self._compliance_scale + direction * cfg.adaptation_rate * self.model.dt_s,
                *cfg.compliance_scale_bounds,
            )
        )
        acceleration = (
            self._compliance_scale * error
            - cfg.virtual_damping_n_s_m * self._virtual_velocity_m_s
            - cfg.virtual_stiffness_n_m * self._virtual_position_m
        ) / cfg.virtual_mass_kg
        self._virtual_velocity_m_s += acceleration * self.model.dt_s
        correction_step = self._virtual_velocity_m_s * self.model.dt_s
        step = target / self.model.force_sensitivity_n_per_m + correction_step
        self._virtual_position_m += correction_step
        rate = _normal_force_rate(self._previous_force_n, force, self.model.dt_s)
        command = _command_from_step(
            step_m=step,
            force_n=force,
            target_force_n=target,
            force_rate_n_s=rate,
            outward_normal_xyz=outward_normal_xyz,
            model=self.model,
            mode=LowLevelBaseline.ADAPTIVE_ADMITTANCE.value,
        )
        self._previous_force_n = force
        predicted = (
            self.model.force_retention * force
            + self.model.force_sensitivity_n_per_m * command.normal_step_m
        )
        return BaselineControlOutput(
            command,
            SolverAudit(
                LowLevelBaseline.ADAPTIVE_ADMITTANCE,
                "explicit_virtual_dynamics",
                True,
                predicted <= self.model.headroom_limit_n,
                max(0.0, predicted - self.model.headroom_limit_n),
                0.0,
                10.0,
                False,
                predicted,
                f"compliance_scale={self._compliance_scale:.6f}",
            ),
        )


@dataclass(frozen=True)
class ForceCBFConfig:
    alpha: float = 0.35
    slack_penalty: float = 100.0
    deadline_ms: float = 10.0
    nominal_gain_m_per_n: float = 0.0008
    nominal_integral_gain_m_per_n_s: float = 0.0002
    integral_clip_n_s: float = 15.0

    def validate(self) -> None:
        if not 0 < self.alpha <= 1:
            raise LowLevelControlError("CBF alpha must lie in (0, 1]")
        if (
            self.slack_penalty <= 0
            or self.deadline_ms <= 0
            or self.nominal_gain_m_per_n <= 0
            or self.nominal_integral_gain_m_per_n_s <= 0
            or self.integral_clip_n_s <= 0
        ):
            raise LowLevelControlError("CBF penalties/deadline/gain must be positive")


class ForceCBFQPController:
    """Exact one-dimensional QP for a discrete local-normal force barrier."""

    def __init__(
        self,
        model: LocalNormalModel = LocalNormalModel(),
        config: ForceCBFConfig = ForceCBFConfig(),
    ) -> None:
        model.validate()
        config.validate()
        self.model = model
        self.config = config
        self._previous_force_n: float | None = None
        self._integral_error_n_s = 0.0

    def reset(self) -> None:
        self._previous_force_n = None
        self._integral_error_n_s = 0.0

    def command(self, *, measured_force_n, target_force_n, outward_normal_xyz):
        started = time.perf_counter_ns()
        force = float(measured_force_n)
        target = float(target_force_n)
        cfg = self.config
        model = self.model
        error = target - force
        self._integral_error_n_s = float(
            np.clip(
                self._integral_error_n_s + error * model.dt_s,
                -cfg.integral_clip_n_s,
                cfg.integral_clip_n_s,
            )
        )
        nominal = float(
            np.clip(
                target / model.force_sensitivity_n_per_m
                + cfg.nominal_gain_m_per_n * error
                + cfg.nominal_integral_gain_m_per_n_s * self._integral_error_n_s,
                -model.max_lift_step_m,
                model.max_press_step_m,
            )
        )
        cbf_force_bound = force + cfg.alpha * (model.force_limit_n - force)
        force_bound = min(cbf_force_bound, model.headroom_limit_n)
        rhs = (
            force_bound
            - model.disturbance_bound_n
            - model.force_retention * force
        )
        sensitivity = model.force_sensitivity_n_per_m
        threshold = rhs / sensitivity

        candidates = {
            float(np.clip(nominal, -model.max_lift_step_m, model.max_press_step_m)),
            float(np.clip(threshold, -model.max_lift_step_m, model.max_press_step_m)),
            -model.max_lift_step_m,
            model.max_press_step_m,
        }
        active_optimum = (
            nominal + cfg.slack_penalty * sensitivity * rhs
        ) / (1.0 + cfg.slack_penalty * sensitivity * sensitivity)
        candidates.add(
            float(np.clip(active_optimum, -model.max_lift_step_m, model.max_press_step_m))
        )

        def objective(step):
            slack = max(0.0, sensitivity * step - rhs)
            return 0.5 * (step - nominal) ** 2 + 0.5 * cfg.slack_penalty * slack**2

        step = min(candidates, key=objective)
        slack = max(0.0, sensitivity * step - rhs)
        predicted = (
            model.force_retention * force
            + sensitivity * step
            + model.disturbance_bound_n
        )
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        rate = _normal_force_rate(self._previous_force_n, force, model.dt_s)
        command = _command_from_step(
            step_m=step,
            force_n=force,
            target_force_n=target,
            force_rate_n_s=rate,
            outward_normal_xyz=outward_normal_xyz,
            model=model,
            mode=LowLevelBaseline.FORCE_CBF_QP.value,
        )
        self._previous_force_n = force
        return BaselineControlOutput(
            command,
            SolverAudit(
                LowLevelBaseline.FORCE_CBF_QP,
                "analytic_convex_1d_qp",
                True,
                slack <= 1e-12,
                slack,
                elapsed_ms,
                cfg.deadline_ms,
                elapsed_ms > cfg.deadline_ms,
                predicted,
                "optimal_with_slack" if slack > 1e-12 else "optimal",
            ),
        )


@dataclass(frozen=True)
class ForceMPCConfig:
    horizon_steps: int = 8
    force_tracking_weight: float = 1.0
    control_weight: float = 1e4
    deadline_ms: float = 10.0
    maximum_iterations: int = 50

    def validate(self) -> None:
        if int(self.horizon_steps) < 2 or int(self.maximum_iterations) <= 0:
            raise LowLevelControlError("MPC horizon/iteration count is invalid")
        if self.force_tracking_weight <= 0 or self.control_weight <= 0 or self.deadline_ms <= 0:
            raise LowLevelControlError("MPC weights/deadline must be positive")


class ForceConstrainedMPCController:
    """Receding-horizon local-force MPC with explicit hard force/input constraints."""

    def __init__(
        self,
        model: LocalNormalModel = LocalNormalModel(),
        config: ForceMPCConfig = ForceMPCConfig(),
    ) -> None:
        model.validate()
        config.validate()
        self.model = model
        self.config = config
        self._previous_force_n: float | None = None

    def reset(self) -> None:
        self._previous_force_n = None

    def command(self, *, measured_force_n, target_force_n, outward_normal_xyz):
        started = time.perf_counter_ns()
        force = float(measured_force_n)
        target = float(target_force_n)
        model = self.model
        cfg = self.config
        horizon = int(cfg.horizon_steps)
        sensitivity = model.force_sensitivity_n_per_m

        def rollout(sequence, disturbance_n=0.0):
            predicted_forces = []
            state = force
            for step in np.asarray(sequence, dtype=np.float64):
                state = model.force_retention * state + sensitivity * step + disturbance_n
                predicted_forces.append(state)
            return np.asarray(predicted_forces, dtype=np.float64)

        def predicted(sequence):
            return rollout(sequence)

        def predicted_upper(sequence):
            return rollout(sequence, model.disturbance_bound_n)

        def predicted_lower(sequence):
            return rollout(sequence, -model.disturbance_bound_n)

        def objective(sequence):
            forces = predicted(sequence)
            return float(
                cfg.force_tracking_weight * np.sum((forces - target) ** 2)
                + cfg.control_weight * np.sum(np.asarray(sequence) ** 2)
            )

        initial_step = float(
            np.clip(
                target / sensitivity,
                -model.max_lift_step_m,
                model.max_press_step_m,
            )
        )
        initial = np.full(horizon, initial_step, dtype=np.float64)
        constraints = (
            {"type": "ineq", "fun": lambda u: model.headroom_limit_n - predicted_upper(u)},
            {"type": "ineq", "fun": lambda u: predicted_lower(u) - model.minimum_contact_force_n},
        )
        with warnings.catch_warnings(record=True) as solver_warnings:
            warnings.simplefilter("always", RuntimeWarning)
            result = minimize(
                objective,
                initial,
                method="SLSQP",
                bounds=[(-model.max_lift_step_m, model.max_press_step_m)] * horizon,
                constraints=constraints,
                options={"maxiter": int(cfg.maximum_iterations), "ftol": 1e-9, "disp": False},
            )
        elapsed_ms = (time.perf_counter_ns() - started) / 1e6
        solved = bool(result.success) and np.all(np.isfinite(result.x))
        if solved:
            sequence = np.asarray(result.x, dtype=np.float64)
            force_upper = predicted_upper(sequence)
            force_lower = predicted_lower(sequence)
            feasible = bool(
                np.all(force_upper <= model.headroom_limit_n + 1e-8)
                and np.all(force_lower >= model.minimum_contact_force_n - 1e-8)
            )
            solved = solved and feasible
        else:
            feasible = False
        step = float(result.x[0]) if solved else -model.max_lift_step_m
        predicted_next = (
            model.force_retention * force
            + sensitivity * step
            + model.disturbance_bound_n
        )
        rate = _normal_force_rate(self._previous_force_n, force, model.dt_s)
        command = _command_from_step(
            step_m=step,
            force_n=force,
            target_force_n=target,
            force_rate_n_s=rate,
            outward_normal_xyz=outward_normal_xyz,
            model=model,
            mode=LowLevelBaseline.FORCE_CONSTRAINED_MPC.value,
        )
        self._previous_force_n = force
        return BaselineControlOutput(
            command,
            SolverAudit(
                LowLevelBaseline.FORCE_CONSTRAINED_MPC,
                "scipy_slsqp",
                solved,
                feasible,
                0.0 if feasible else math.nan,
                elapsed_ms,
                cfg.deadline_ms,
                elapsed_ms > cfg.deadline_ms,
                predicted_next,
                str(result.message),
                len(solver_warnings),
            ),
        )
