"""Versioned V6-S4 profile over the frozen shared SANDBOX runner."""

from __future__ import annotations

import json
from pathlib import Path

from .controller_s4 import (
    S4_MAXIMUM_EPISODE_RECOVERY_CYCLES,
    V6S4BoundedPhaseSupervisor,
)
from .sandbox_runner import (
    SandboxProfile,
    SandboxRunResult,
    V6SandboxPreflightError,
    _early_permission_check,
    run_sandbox,
)
from .sandbox_storage import (
    AUTHORITY_METRIC_SPLIT_V3,
    PERMISSION_SCOPE_S3,
    RUN_DEFINITION_FORMAT_S3_R2,
)


PROTOCOL_FORMAT_S4 = "forcewipe_v6_s4_sandbox_protocol_v1"
PREAUTH_MANIFEST_FORMAT_S4 = "forcewipe_v6_s4_sandbox_preauth_manifest_v1"

S4_PROFILE = SandboxProfile(
    label="V6-S4",
    protocol_format=PROTOCOL_FORMAT_S4,
    preauth_manifest_format=PREAUTH_MANIFEST_FORMAT_S4,
    permission_scope=PERMISSION_SCOPE_S3,
    protocol_filename="V6_S4_SANDBOX_PHYSICS_PROTOCOL.json",
    permission_filename="V6_S4_SANDBOX_PHYSICS_PERMISSION.json",
    source_manifest_filename="V6_S4_SANDBOX_PHYSICS_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_S4_SANDBOX_PHYSICS_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6s4_",
    evaluation_namespace="forcewipe-v6-s4",
    run_definition_format=RUN_DEFINITION_FORMAT_S3_R2,
    authority_metric_version=AUTHORITY_METRIC_SPLIT_V3,
    controller_type=V6S4BoundedPhaseSupervisor,
    manual_approval_required=True,
    external_runtime_required=True,
)


def _verify_s4_identity(package_root: Path) -> None:
    path = package_root / "config" / S4_PROFILE.protocol_filename
    try:
        protocol = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V6SandboxPreflightError("cannot read the V6-S4 protocol") from exc
    if protocol.get("controller_revision") != (
        V6S4BoundedPhaseSupervisor.controller_revision
    ):
        raise V6SandboxPreflightError("V6-S4 controller revision is not frozen")
    if protocol.get("controller_parameter_overrides") != {
        "maximum_episode_recovery_cycles": (
            S4_MAXIMUM_EPISODE_RECOVERY_CYCLES
        )
    }:
        raise V6SandboxPreflightError("V6-S4 controller override is not frozen")
    if protocol.get("maximum_episode_recovery_samples") != 1500:
        raise V6SandboxPreflightError("V6-S4 recovery-sample bound changed")


def run_sandbox_s4(
    *,
    package_root: Path,
    source_root: Path,
    run_id: str | None = None,
) -> SandboxRunResult:
    package = Path(package_root).resolve(strict=True)
    _early_permission_check(
        package / "config" / S4_PROFILE.permission_filename,
        profile=S4_PROFILE,
    )
    _verify_s4_identity(package)
    return run_sandbox(
        package_root=package,
        source_root=source_root,
        run_id=run_id,
        profile=S4_PROFILE,
    )


__all__ = [
    "PREAUTH_MANIFEST_FORMAT_S4",
    "PROTOCOL_FORMAT_S4",
    "S4_PROFILE",
    "run_sandbox_s4",
]
