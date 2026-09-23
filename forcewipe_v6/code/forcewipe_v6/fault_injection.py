"""Deterministic code-only fault injection for V6 input validation."""

from __future__ import annotations

from dataclasses import replace
from enum import Enum
import math

from .controller import SupervisorInput


class InputFault(str, Enum):
    NAN_FORCE = "NAN_FORCE"
    NEGATIVE_FORCE = "NEGATIVE_FORCE"
    ZERO_TARGET = "ZERO_TARGET"
    INFINITE_VELOCITY = "INFINITE_VELOCITY"
    NAN_PROGRESS = "NAN_PROGRESS"
    NEGATIVE_TANGENTIAL_REQUEST = "NEGATIVE_TANGENTIAL_REQUEST"
    CONTACT_DROPOUT = "CONTACT_DROPOUT"


def inject_input_fault(observation: SupervisorInput, fault: InputFault) -> SupervisorInput:
    if fault is InputFault.NAN_FORCE:
        return replace(observation, measured_force_n=math.nan)
    if fault is InputFault.NEGATIVE_FORCE:
        return replace(observation, measured_force_n=-1.0)
    if fault is InputFault.ZERO_TARGET:
        return replace(observation, target_force_n=0.0)
    if fault is InputFault.INFINITE_VELOCITY:
        return replace(observation, normal_velocity_outward_m_s=math.inf)
    if fault is InputFault.NAN_PROGRESS:
        return replace(observation, instantaneous_progress=math.nan)
    if fault is InputFault.NEGATIVE_TANGENTIAL_REQUEST:
        return replace(observation, requested_tangential_step_m=-0.001)
    if fault is InputFault.CONTACT_DROPOUT:
        return replace(observation, contact_observed=False, measured_force_n=0.0)
    raise ValueError(f"unknown fault: {fault}")
