"""Repeatable-development SANDBOX profile for System R4."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .authority_metrics_system_r4 import AUTHORITY_METRIC_SYSTEM_R4
from .controller_system_r4 import V6SystemContactSupervisorR4
from .sandbox_runner import (
    SandboxProfile,
    SandboxRunResult,
    V6SandboxPreflightError,
    _early_permission_check,
    _supervisor_config,
    _verify_preauth_manifest,
    run_sandbox,
)
from .sandbox_storage import (
    PERMISSION_SCOPE_SYSTEM_R4,
    RUN_DEFINITION_FORMAT_SYSTEM_R4,
)
from .sapien_sandbox_binding_system_r4 import (
    R4SapienSandboxBinding,
    R4SurfacePathGeometry,
)


PROTOCOL_FORMAT = "forcewipe_v6_system_r4_sandbox_protocol_v1"
PREAUTH_MANIFEST_FORMAT = "forcewipe_v6_system_r4_sandbox_preauth_manifest_v1"

SYSTEM_R4_PROFILE = SandboxProfile(
    label="V6-system-r4",
    protocol_format=PROTOCOL_FORMAT,
    preauth_manifest_format=PREAUTH_MANIFEST_FORMAT,
    permission_scope=PERMISSION_SCOPE_SYSTEM_R4,
    protocol_filename="V6_SYSTEM_R4_SANDBOX_PROTOCOL.json",
    permission_filename="V6_SYSTEM_R4_SANDBOX_PERMISSION.json",
    source_manifest_filename="V6_SYSTEM_R4_SANDBOX_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_SYSTEM_R4_SANDBOX_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6systemr4_",
    evaluation_namespace="forcewipe-v6-system-r4-sandbox-v1",
    run_definition_format=RUN_DEFINITION_FORMAT_SYSTEM_R4,
    authority_metric_version=AUTHORITY_METRIC_SYSTEM_R4,
    controller_type=V6SystemContactSupervisorR4,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=R4SurfacePathGeometry,
    binding_type=R4SapienSandboxBinding,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V6SandboxPreflightError(f"cannot read the System R4 {label}") from exc
    if not isinstance(value, dict):
        raise V6SandboxPreflightError(f"System R4 {label} is not an object")
    return value


def _verify_identity(package_root: Path) -> None:
    config_root = package_root / "config"
    protocol = _load_json(
        config_root / SYSTEM_R4_PROFILE.protocol_filename,
        label="protocol",
    )
    controller = V6SystemContactSupervisorR4(
        _supervisor_config(config_root / "V6_SANDBOX_CONFIG.json")
    )
    expected = {
        "scope": SYSTEM_R4_PROFILE.permission_scope,
        "controller_revision": controller.controller_revision,
        "controller_type": (
            "forcewipe_v6.controller_system_r4.V6SystemContactSupervisorR4"
        ),
        "binding_type": (
            "forcewipe_v6.sapien_sandbox_binding_system_r4."
            "R4SapienSandboxBinding"
        ),
        "geometry_type": (
            "forcewipe_v6.sapien_sandbox_binding_system_r4."
            "R4SurfacePathGeometry"
        ),
        "evaluation_id_namespace": SYSTEM_R4_PROFILE.evaluation_namespace,
        "run_prefix": SYSTEM_R4_PROFILE.run_prefix,
        "run_definition_format": SYSTEM_R4_PROFILE.run_definition_format,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise V6SandboxPreflightError(f"System R4 {key} is not frozen")
    if protocol.get("controller_configuration") != controller.config.as_dict():
        raise V6SandboxPreflightError(
            "System R4 effective controller configuration is not frozen"
        )
    controller_path = (
        package_root / "code" / "forcewipe_v6" / "controller_system_r4.py"
    )
    if protocol.get("controller_sha256") != _sha256(controller_path):
        raise V6SandboxPreflightError("System R4 controller identity changed")


def _verify_live_source_manifest(package_root: Path, source_root: Path) -> None:
    path = (
        package_root
        / "config"
        / SYSTEM_R4_PROFILE.source_manifest_filename
    )
    try:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise V6SandboxPreflightError("cannot read the System R4 source manifest") from exc
    if not rows:
        raise V6SandboxPreflightError("System R4 source manifest is empty")
    root = Path(source_root).resolve(strict=True)
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"relative_path", "size_bytes", "sha256", "role"}:
            raise V6SandboxPreflightError("System R4 source manifest row is malformed")
        relative = Path(str(row["relative_path"]))
        key = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or key in seen:
            raise V6SandboxPreflightError("System R4 source path is unsafe or duplicated")
        seen.add(key)
        source = (root / relative).resolve(strict=True)
        if not source.is_relative_to(root) or not source.is_file():
            raise V6SandboxPreflightError(f"System R4 source escapes its root: {key}")
        if (
            source.stat().st_size != int(row["size_bytes"])
            or _sha256(source) != row["sha256"]
        ):
            raise V6SandboxPreflightError(
                f"System R4 source identity mismatch: {key}"
            )


def run_sandbox_system_r4(
    *, package_root: Path, source_root: Path, run_id: str | None = None
) -> SandboxRunResult:
    package = Path(package_root).resolve(strict=True)
    source = Path(source_root).resolve(strict=True)
    permission = _early_permission_check(
        package / "config" / SYSTEM_R4_PROFILE.permission_filename,
        profile=SYSTEM_R4_PROFILE,
    )
    _verify_preauth_manifest(package, permission, profile=SYSTEM_R4_PROFILE)
    _verify_identity(package)
    _verify_live_source_manifest(package, source)
    return run_sandbox(
        package_root=package,
        source_root=source,
        run_id=run_id,
        profile=SYSTEM_R4_PROFILE,
    )


__all__ = ["SYSTEM_R4_PROFILE", "run_sandbox_system_r4"]
