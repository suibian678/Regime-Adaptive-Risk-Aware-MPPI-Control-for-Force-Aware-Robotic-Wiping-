"""Auditable A0--A4 control-authority interface contracts for V4."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np


class AuthorityContractError(ValueError):
    pass


class AuthorityInterface(str, Enum):
    A0_CLASSICAL = "A0_classical"
    A1_PRIMITIVE = "A1_primitive_id"
    A2_FORCE_PARAMETER = "A2_primitive_force"
    A3_IMPEDANCE_PARAMETER = "A3_primitive_force_impedance"
    A4_CONTINUOUS = "A4_bounded_cartesian_impedance"


@dataclass(frozen=True)
class AuthorityDefaults:
    target_force_n: float
    stiffness_n_m: float = 650.0
    damping_n_s_m: float = 15.0


@dataclass(frozen=True)
class AuthorityBounds:
    target_force_n: tuple[float, float] = (3.0, 12.5)
    stiffness_n_m: tuple[float, float] = (100.0, 2000.0)
    damping_n_s_m: tuple[float, float] = (2.0, 150.0)
    cartesian_delta_m: tuple[float, float] = (-0.0035, 0.0035)

    def validate(self) -> None:
        for name in self.__dataclass_fields__:
            lower, upper = getattr(self, name)
            if not all(math.isfinite(float(value)) for value in (lower, upper)):
                raise AuthorityContractError(f"{name} bounds must be finite")
            if float(lower) >= float(upper):
                raise AuthorityContractError(f"{name} bounds are not ordered")


@dataclass(frozen=True)
class AuthorityProposal:
    interface: AuthorityInterface
    primitive_id: int | None = None
    target_force_n: float | None = None
    stiffness_n_m: float | None = None
    damping_n_s_m: float | None = None
    cartesian_delta_xyz_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class ExecutableAuthorityAction:
    interface: AuthorityInterface
    primitive_id: int | None
    target_force_n: float
    stiffness_n_m: float
    damping_n_s_m: float
    cartesian_delta_xyz_m: tuple[float, float, float] | None
    projected: bool
    projection_reasons: tuple[str, ...]


def policy_controlled_fields(interface: AuthorityInterface) -> tuple[str, ...]:
    interface = AuthorityInterface(interface)
    if interface == AuthorityInterface.A0_CLASSICAL:
        return ()
    if interface == AuthorityInterface.A1_PRIMITIVE:
        return ("primitive_id",)
    if interface == AuthorityInterface.A2_FORCE_PARAMETER:
        return ("primitive_id", "target_force_n")
    if interface == AuthorityInterface.A3_IMPEDANCE_PARAMETER:
        return (
            "primitive_id",
            "target_force_n",
            "stiffness_n_m",
            "damping_n_s_m",
        )
    return (
        "cartesian_delta_xyz_m",
        "target_force_n",
        "stiffness_n_m",
        "damping_n_s_m",
    )


def _bounded(value: float, interval: tuple[float, float], name: str, reasons: list[str]) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AuthorityContractError(f"{name} must be finite")
    projected = float(np.clip(numeric, interval[0], interval[1]))
    if projected != numeric:
        reasons.append(f"clip_{name}")
    return projected


def compile_authority_action(
    proposal: AuthorityProposal,
    *,
    defaults: AuthorityDefaults,
    primitive_count: int,
    bounds: AuthorityBounds = AuthorityBounds(),
) -> ExecutableAuthorityAction:
    """Validate authority first, then project only fields that interface controls."""

    interface = AuthorityInterface(proposal.interface)
    bounds.validate()
    if int(primitive_count) <= 0:
        raise AuthorityContractError("primitive_count must be positive")
    allowed = set(policy_controlled_fields(interface))
    provided = {
        name
        for name in (
            "primitive_id",
            "target_force_n",
            "stiffness_n_m",
            "damping_n_s_m",
            "cartesian_delta_xyz_m",
        )
        if getattr(proposal, name) is not None
    }
    if interface == AuthorityInterface.A0_CLASSICAL:
        allowed = {
            "primitive_id",
            "target_force_n",
            "stiffness_n_m",
            "damping_n_s_m",
        }
    unauthorized = sorted(provided - allowed)
    if unauthorized:
        raise AuthorityContractError(
            f"{interface.value} proposal contains unauthorized fields: {unauthorized}"
        )
    required = set(allowed)
    if interface == AuthorityInterface.A0_CLASSICAL:
        required = {"primitive_id"}
    missing = sorted(required - provided)
    if missing:
        raise AuthorityContractError(
            f"{interface.value} proposal lacks controlled fields: {missing}"
        )

    primitive_id = proposal.primitive_id
    if primitive_id is not None:
        if int(primitive_id) != primitive_id or not 0 <= int(primitive_id) < int(primitive_count):
            raise AuthorityContractError("primitive_id is outside the shared library")
        primitive_id = int(primitive_id)

    reasons: list[str] = []
    force = _bounded(
        defaults.target_force_n if proposal.target_force_n is None else proposal.target_force_n,
        bounds.target_force_n,
        "target_force_n",
        reasons,
    )
    stiffness = _bounded(
        defaults.stiffness_n_m if proposal.stiffness_n_m is None else proposal.stiffness_n_m,
        bounds.stiffness_n_m,
        "stiffness_n_m",
        reasons,
    )
    damping = _bounded(
        defaults.damping_n_s_m if proposal.damping_n_s_m is None else proposal.damping_n_s_m,
        bounds.damping_n_s_m,
        "damping_n_s_m",
        reasons,
    )
    cartesian = None
    if proposal.cartesian_delta_xyz_m is not None:
        vector = np.asarray(proposal.cartesian_delta_xyz_m, dtype=np.float64)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise AuthorityContractError("Cartesian delta must be one finite three-vector")
        clipped = np.clip(vector, *bounds.cartesian_delta_m)
        if not np.array_equal(clipped, vector):
            reasons.append("clip_cartesian_delta_xyz_m")
        cartesian = tuple(float(value) for value in clipped)
    return ExecutableAuthorityAction(
        interface=interface,
        primitive_id=primitive_id,
        target_force_n=force,
        stiffness_n_m=stiffness,
        damping_n_s_m=damping,
        cartesian_delta_xyz_m=cartesian,
        projected=bool(reasons),
        projection_reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class EmergencySupervisorDecision:
    terminate: bool
    minimum_risk_retract: bool
    reason: str | None


def emergency_supervisor(
    *,
    audit_force_n: float,
    finite_state: bool,
    force_limit_n: float = 15.0,
) -> EmergencySupervisorDecision:
    """Common terminal supervisor; it never selects or substitutes a task action."""

    force = float(audit_force_n)
    if not finite_state or not math.isfinite(force):
        return EmergencySupervisorDecision(True, True, "nonfinite_state_or_force")
    if force > float(force_limit_n):
        return EmergencySupervisorDecision(True, True, "native_force_limit")
    return EmergencySupervisorDecision(False, False, None)
