"""Transactional first-order normal admittance for the target-pose interface."""

from __future__ import annotations

from dataclasses import dataclass
import math


class NormalAdmittanceError(ValueError):
    pass


@dataclass(frozen=True)
class NormalAdmittanceConfig:
    dt_s: float = 0.01
    force_error_acceleration_gain_m_per_n_s2: float = 0.1
    velocity_damping_per_s: float = 20.0
    maximum_inward_acceleration_m_s2: float = 0.4
    maximum_outward_acceleration_m_s2: float = 1.5
    maximum_inward_velocity_m_s: float = 0.03
    maximum_outward_velocity_m_s: float = 0.05

    def validate(self) -> None:
        values = (
            self.dt_s,
            self.force_error_acceleration_gain_m_per_n_s2,
            self.velocity_damping_per_s,
            self.maximum_inward_acceleration_m_s2,
            self.maximum_outward_acceleration_m_s2,
            self.maximum_inward_velocity_m_s,
            self.maximum_outward_velocity_m_s,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise NormalAdmittanceError("normal-admittance parameters must be positive and finite")
        if self.dt_s * self.velocity_damping_per_s >= 1.0:
            raise NormalAdmittanceError("explicit admittance damping step must remain below one")


@dataclass(frozen=True)
class NormalAdmittanceTransaction:
    sequence_id: int
    velocity_before_m_s: float
    raw_acceleration_m_s2: float
    clipped_acceleration_m_s2: float
    unconstrained_velocity_after_m_s: float
    velocity_limited_after_m_s: float
    unconstrained_increment_m: float
    base_safety_increment_m: float
    executed_increment_m: float
    base_safety_projection_active: bool


class TransactionalNormalAdmittance:
    """Discretize Mv_dot+Bv=K(F*-F) and commit state after execution."""

    def __init__(self, config: NormalAdmittanceConfig = NormalAdmittanceConfig()):
        config.validate()
        self.config = config
        self._velocity_m_s = 0.0
        self._next_sequence_id = 0
        self._pending: NormalAdmittanceTransaction | None = None

    @property
    def velocity_m_s(self) -> float:
        return self._velocity_m_s

    @property
    def pending(self) -> NormalAdmittanceTransaction | None:
        return self._pending

    def reset(self, *, velocity_m_s: float = 0.0) -> None:
        if self._pending is not None:
            raise NormalAdmittanceError("cannot reset with a pending admittance transaction")
        velocity = float(velocity_m_s)
        if not math.isfinite(velocity):
            raise NormalAdmittanceError("reset velocity must be finite")
        if not -self.config.maximum_outward_velocity_m_s <= velocity <= self.config.maximum_inward_velocity_m_s:
            raise NormalAdmittanceError("reset velocity exceeds the registered bounds")
        self._velocity_m_s = velocity
        self._next_sequence_id = 0

    def propose(
        self,
        *,
        target_force_n: float,
        measured_force_n: float,
        base_safety_increment_m: float,
    ) -> NormalAdmittanceTransaction:
        if self._pending is not None:
            raise NormalAdmittanceError("an admittance transaction is already pending")
        target = float(target_force_n)
        measured = float(measured_force_n)
        base_increment = float(base_safety_increment_m)
        if not all(math.isfinite(value) for value in (target, measured, base_increment)):
            raise NormalAdmittanceError("admittance inputs must be finite")
        if target <= 0.0 or measured < 0.0:
            raise NormalAdmittanceError("admittance force inputs are invalid")
        force_error = target - measured
        raw_acceleration = (
            self.config.force_error_acceleration_gain_m_per_n_s2 * force_error
            - self.config.velocity_damping_per_s * self._velocity_m_s
        )
        clipped_acceleration = max(
            -self.config.maximum_outward_acceleration_m_s2,
            min(self.config.maximum_inward_acceleration_m_s2, raw_acceleration),
        )
        unconstrained_velocity = self._velocity_m_s + clipped_acceleration * self.config.dt_s
        velocity_limited = max(
            -self.config.maximum_outward_velocity_m_s,
            min(self.config.maximum_inward_velocity_m_s, unconstrained_velocity),
        )
        unconstrained_increment = velocity_limited * self.config.dt_s
        # The inherited projected command is a one-sided safety authority.  The
        # admittance may request less inward or more outward motion, never more
        # inward motion than that already-projected command.
        executed_increment = min(unconstrained_increment, base_increment)
        executed_increment = max(
            -self.config.maximum_outward_velocity_m_s * self.config.dt_s,
            min(self.config.maximum_inward_velocity_m_s * self.config.dt_s, executed_increment),
        )
        transaction = NormalAdmittanceTransaction(
            sequence_id=self._next_sequence_id,
            velocity_before_m_s=self._velocity_m_s,
            raw_acceleration_m_s2=raw_acceleration,
            clipped_acceleration_m_s2=clipped_acceleration,
            unconstrained_velocity_after_m_s=unconstrained_velocity,
            velocity_limited_after_m_s=velocity_limited,
            unconstrained_increment_m=unconstrained_increment,
            base_safety_increment_m=base_increment,
            executed_increment_m=executed_increment,
            base_safety_projection_active=executed_increment < base_increment - 1.0e-15,
        )
        self._pending = transaction
        return transaction

    def commit(self, transaction: NormalAdmittanceTransaction) -> None:
        if self._pending is None or transaction is not self._pending:
            raise NormalAdmittanceError("only the current admittance transaction may be committed")
        self._velocity_m_s = transaction.executed_increment_m / self.config.dt_s
        self._next_sequence_id += 1
        self._pending = None

    def discard(self, transaction: NormalAdmittanceTransaction) -> None:
        if self._pending is None or transaction is not self._pending:
            raise NormalAdmittanceError("only the current admittance transaction may be discarded")
        self._pending = None
