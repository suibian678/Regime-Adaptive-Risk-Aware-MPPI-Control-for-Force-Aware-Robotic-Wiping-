"""Independent V4 shield and low-level protection ablation roster."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

from .low_level_control import ForceControllerConfig


@dataclass(frozen=True)
class ProtectionAblationSpec:
    name: str
    shield_enabled: bool = True
    preload_ramp_enabled: bool = True
    impact_brake_enabled: bool = True
    headroom_enabled: bool = True
    taper_enabled: bool = True
    anti_windup_enabled: bool = True
    reacquisition_enabled: bool = True
    lift_guard_enabled: bool = True


def canonical_low_level_ablation_roster() -> tuple[ProtectionAblationSpec, ...]:
    return (
        ProtectionAblationSpec("full"),
        ProtectionAblationSpec("shield_off", shield_enabled=False),
        ProtectionAblationSpec("preload_ramp_off", preload_ramp_enabled=False),
        ProtectionAblationSpec("impact_brake_off", impact_brake_enabled=False),
        ProtectionAblationSpec(
            "preload_and_impact_off",
            preload_ramp_enabled=False,
            impact_brake_enabled=False,
        ),
    )


def changed_protection_fields(
    reference: ProtectionAblationSpec,
    candidate: ProtectionAblationSpec,
) -> tuple[str, ...]:
    return tuple(
        field.name
        for field in fields(reference)
        if field.name != "name" and getattr(reference, field.name) != getattr(candidate, field.name)
    )


def validate_canonical_ablation_roster(
    roster: tuple[ProtectionAblationSpec, ...] | None = None,
) -> None:
    roster = canonical_low_level_ablation_roster() if roster is None else roster
    by_name = {item.name: item for item in roster}
    required = {
        "full",
        "shield_off",
        "preload_ramp_off",
        "impact_brake_off",
        "preload_and_impact_off",
    }
    if set(by_name) != required:
        raise ValueError("ablation roster names differ from the frozen minimum set")
    full = by_name["full"]
    expected = {
        "shield_off": ("shield_enabled",),
        "preload_ramp_off": ("preload_ramp_enabled",),
        "impact_brake_off": ("impact_brake_enabled",),
        "preload_and_impact_off": (
            "preload_ramp_enabled",
            "impact_brake_enabled",
        ),
    }
    for name, changed in expected.items():
        if changed_protection_fields(full, by_name[name]) != changed:
            raise ValueError(f"{name} changes fields outside its declared ablation")


def force_controller_config_for_ablation(
    spec: ProtectionAblationSpec,
    *,
    base: ForceControllerConfig = ForceControllerConfig(),
) -> ForceControllerConfig:
    """Apply only independently implemented low-level factors to one base config."""

    return replace(
        base,
        preload_ramp_enabled=spec.preload_ramp_enabled,
        impact_brake_enabled=spec.impact_brake_enabled,
    )
