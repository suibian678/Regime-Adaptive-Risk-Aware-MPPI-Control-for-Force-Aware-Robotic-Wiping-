"""Discrete-time numerical audit for the V4 compliant pad support model."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np


class PlantNumericsError(ValueError):
    """Invalid physical parameter or an unauditable discrete support model."""


@dataclass(frozen=True)
class VerticalSupportAudit:
    mass_kg: float
    support_stiffness_n_m: float
    damping_ratio: float
    requested_damping_n_s_m: float
    explicit_damping_n_s_m: float
    dt_s: float
    natural_frequency_rad_s: float
    natural_frequency_times_dt: float
    velocity_decay_per_step: float
    spring_number: float
    spectral_radius: float
    stability_margin: float
    discrete_stable: bool

    def as_record(self) -> dict[str, float | bool]:
        return asdict(self)


def _finite_positive(name: str, value: float) -> float:
    output = float(value)
    if not math.isfinite(output) or output <= 0.0:
        raise PlantNumericsError(f"{name} must be finite and positive")
    return output


def equivalent_explicit_damping(
    mass_kg: float,
    damping_n_s_m: float,
    dt_s: float,
) -> float:
    """Map continuous viscous decay to an explicit per-step force coefficient.

    The mapping equates ``exp(-c dt / m)`` with the velocity multiplier
    ``1 - c_eff dt / m``. It prevents a large damping coefficient from
    flipping the velocity sign solely because of forward-Euler damping.
    """

    mass = _finite_positive("mass_kg", mass_kg)
    damping = float(damping_n_s_m)
    dt = _finite_positive("dt_s", dt_s)
    if not math.isfinite(damping) or damping < 0.0:
        raise PlantNumericsError("damping_n_s_m must be finite and nonnegative")
    return mass / dt * (-math.expm1(-damping * dt / mass))


def audit_vertical_support(
    *,
    mass_kg: float,
    support_stiffness_n_m: float,
    damping_ratio: float,
    dt_s: float = 0.01,
    tolerance: float = 1e-12,
) -> VerticalSupportAudit:
    """Audit the semi-implicit discrete spring--damper update used by V4.

    The current ForceWipe plant applies a virtual support force before each
    PhysX step. This audit models the vertical scalar update as

    ``v[k+1] = d v[k] - (k dt / m) x[k]`` and
    ``x[k+1] = x[k] + dt v[k+1]``,

    where ``d = exp(-c dt / m)`` after the exact-decay damping conversion.
    It is a numerical compatibility test for the configured plant, not a
    proof of closed-loop robot safety or of the full nonlinear simulator.
    """

    mass = _finite_positive("mass_kg", mass_kg)
    stiffness = _finite_positive("support_stiffness_n_m", support_stiffness_n_m)
    ratio = _finite_positive("damping_ratio", damping_ratio)
    dt = _finite_positive("dt_s", dt_s)
    tol = float(tolerance)
    if not math.isfinite(tol) or tol < 0.0:
        raise PlantNumericsError("tolerance must be finite and nonnegative")

    natural_frequency = math.sqrt(stiffness / mass)
    requested_damping = 2.0 * ratio * math.sqrt(stiffness * mass)
    explicit_damping = equivalent_explicit_damping(mass, requested_damping, dt)
    velocity_decay = math.exp(-requested_damping * dt / mass)
    spring_number = stiffness * dt * dt / mass

    transition = np.array(
        [
            [1.0 - spring_number, dt * velocity_decay],
            [-stiffness * dt / mass, velocity_decay],
        ],
        dtype=np.float64,
    )
    eigenvalues = np.linalg.eigvals(transition)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    stable = bool(math.isfinite(spectral_radius) and spectral_radius <= 1.0 + tol)
    return VerticalSupportAudit(
        mass_kg=mass,
        support_stiffness_n_m=stiffness,
        damping_ratio=ratio,
        requested_damping_n_s_m=requested_damping,
        explicit_damping_n_s_m=explicit_damping,
        dt_s=dt,
        natural_frequency_rad_s=natural_frequency,
        natural_frequency_times_dt=natural_frequency * dt,
        velocity_decay_per_step=velocity_decay,
        spring_number=spring_number,
        spectral_radius=spectral_radius,
        stability_margin=1.0 - spectral_radius,
        discrete_stable=stable,
    )


def require_discrete_stability(audit: VerticalSupportAudit) -> None:
    if not audit.discrete_stable:
        raise PlantNumericsError(
            "vertical support parameters fail the 100 Hz discrete stability screen: "
            f"spectral_radius={audit.spectral_radius:.6g}"
        )
