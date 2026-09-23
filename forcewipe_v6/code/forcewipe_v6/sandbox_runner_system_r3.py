"""Fresh repeatable-development SANDBOX profile for System R3 rev2."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from .authority_metrics_s5 import AUTHORITY_METRIC_PATH_FRAME_V4
from .authorization_system_r3 import (
    validate_approval_identity,
    validate_review_basis_reconstruction,
    validate_review_binding,
)
from .controller_system_r3_rev2 import V6SystemContactSupervisorR3Rev2
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
    PERMISSION_SCOPE_SYSTEM_R3_REV2,
    RUN_DEFINITION_FORMAT_SYSTEM_R3_REV2,
)
from .sapien_sandbox_binding_system_r3 import (
    R3Rev2SapienSandboxBinding,
    R3Rev2SurfacePathGeometry,
)


PROTOCOL_FORMAT = "forcewipe_v6_system_r3_rev2_sandbox_protocol_v1"
PREAUTH_MANIFEST_FORMAT = "forcewipe_v6_system_r3_rev2_sandbox_preauth_manifest_v1"

SYSTEM_R3_REV2_PROFILE = SandboxProfile(
    label="V6-system-r3-rev2",
    protocol_format=PROTOCOL_FORMAT,
    preauth_manifest_format=PREAUTH_MANIFEST_FORMAT,
    permission_scope=PERMISSION_SCOPE_SYSTEM_R3_REV2,
    protocol_filename="V6_SYSTEM_R3_REV2_SANDBOX_PROTOCOL.json",
    permission_filename="V6_SYSTEM_R3_REV2_SANDBOX_PERMISSION.json",
    source_manifest_filename="V6_SYSTEM_R3_REV2_SANDBOX_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_SYSTEM_R3_REV2_SANDBOX_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6systemr3r2_",
    evaluation_namespace="forcewipe-v6-system-r3-rev2-sandbox-v1",
    run_definition_format=RUN_DEFINITION_FORMAT_SYSTEM_R3_REV2,
    authority_metric_version=AUTHORITY_METRIC_PATH_FRAME_V4,
    controller_type=V6SystemContactSupervisorR3Rev2,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=R3Rev2SurfacePathGeometry,
    binding_type=R3Rev2SapienSandboxBinding,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V6SandboxPreflightError(f"cannot read the System R3 {label}") from exc
    if not isinstance(value, dict):
        raise V6SandboxPreflightError(f"System R3 {label} is not an object")
    return value


def _verify_identity(package_root: Path) -> None:
    config_root = package_root / "config"
    protocol_path = config_root / SYSTEM_R3_REV2_PROFILE.protocol_filename
    permission_path = config_root / SYSTEM_R3_REV2_PROFILE.permission_filename
    protocol = _load_json(protocol_path, label="protocol")
    permission = _load_json(permission_path, label="permission")
    effective = V6SystemContactSupervisorR3Rev2(
        _supervisor_config(config_root / "V6_SANDBOX_CONFIG.json")
    )
    expected = {
        "scope": SYSTEM_R3_REV2_PROFILE.permission_scope,
        "controller_revision": V6SystemContactSupervisorR3Rev2.controller_revision,
        "controller_type": (
            "forcewipe_v6.controller_system_r3_rev2."
            "V6SystemContactSupervisorR3Rev2"
        ),
        "binding_type": (
            "forcewipe_v6.sapien_sandbox_binding_system_r3."
            "R3Rev2SapienSandboxBinding"
        ),
        "geometry_type": (
            "forcewipe_v6.sapien_sandbox_binding_system_r3."
            "R3Rev2SurfacePathGeometry"
        ),
        "evaluation_id_namespace": SYSTEM_R3_REV2_PROFILE.evaluation_namespace,
        "run_prefix": SYSTEM_R3_REV2_PROFILE.run_prefix,
        "run_definition_format": SYSTEM_R3_REV2_PROFILE.run_definition_format,
        "r2_physical_failure_superseded": False,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise V6SandboxPreflightError(f"System R3 {key} is not frozen")
    if protocol.get("controller_configuration") != effective.config.as_dict():
        raise V6SandboxPreflightError(
            "System R3 effective controller configuration is not frozen"
        )
    controller_path = (
        package_root / "code" / "forcewipe_v6" / "controller_system_r3_rev2.py"
    )
    if protocol.get("controller_sha256") != _sha256_file(controller_path):
        raise V6SandboxPreflightError("System R3 controller identity changed")
    try:
        validate_approval_identity(
            permission.get("reviewer"),
            permission.get("approved_utc"),
            permission.get("approval_id"),
        )
        validate_review_binding(permission)
    except ValueError as exc:
        raise V6SandboxPreflightError(str(exc)) from exc


def _verify_live_source_manifest(package_root: Path, source_root: Path) -> None:
    """Verify every transitive payload before importing simulator packages."""

    path = (
        package_root
        / "config"
        / SYSTEM_R3_REV2_PROFILE.source_manifest_filename
    )
    try:
        rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    except (OSError, UnicodeDecodeError, csv.Error) as exc:
        raise V6SandboxPreflightError("cannot read the System R3 source manifest") from exc
    if not rows:
        raise V6SandboxPreflightError("System R3 source manifest is empty")
    root = Path(source_root).resolve(strict=True)
    seen: set[str] = set()
    for row in rows:
        if set(row) != {"relative_path", "size_bytes", "sha256", "role"}:
            raise V6SandboxPreflightError("System R3 source manifest row is malformed")
        relative = Path(str(row["relative_path"]))
        key = relative.as_posix()
        if relative.is_absolute() or ".." in relative.parts or key in seen:
            raise V6SandboxPreflightError("System R3 source path is unsafe or duplicated")
        seen.add(key)
        try:
            source = (root / relative).resolve(strict=True)
        except OSError as exc:
            raise V6SandboxPreflightError(
                f"System R3 source payload is missing: {key}"
            ) from exc
        if not source.is_relative_to(root) or not source.is_file():
            raise V6SandboxPreflightError(f"System R3 source escapes its root: {key}")
        try:
            expected_size = int(row["size_bytes"])
        except (TypeError, ValueError) as exc:
            raise V6SandboxPreflightError(
                f"System R3 source size is malformed: {key}"
            ) from exc
        if source.stat().st_size != expected_size or _sha256_file(source) != row["sha256"]:
            raise V6SandboxPreflightError(
                f"System R3 source identity mismatch before runtime probe: {key}"
            )


def _verify_reviewed_disabled_basis(
    package_root: Path, permission: dict[str, Any]
) -> None:
    preauth = _load_json(
        package_root / "config" / SYSTEM_R3_REV2_PROFILE.preauth_manifest_filename,
        label="preauthorization manifest",
    )
    permission_key = (
        Path("config") / SYSTEM_R3_REV2_PROFILE.permission_filename
    ).as_posix()
    try:
        validate_review_basis_reconstruction(
            permission,
            preauth,
            permission_relative_path=permission_key,
        )
    except ValueError as exc:
        raise V6SandboxPreflightError(str(exc)) from exc


def run_sandbox_system_r3_rev2(
    *, package_root: Path, source_root: Path, run_id: str | None = None
) -> SandboxRunResult:
    package = Path(package_root).resolve(strict=True)
    source = Path(source_root).resolve(strict=True)
    permission = _early_permission_check(
        package / "config" / SYSTEM_R3_REV2_PROFILE.permission_filename,
        profile=SYSTEM_R3_REV2_PROFILE,
    )
    _verify_preauth_manifest(package, permission, profile=SYSTEM_R3_REV2_PROFILE)
    _verify_reviewed_disabled_basis(package, permission)
    _verify_identity(package)
    _verify_live_source_manifest(package, source)
    return run_sandbox(
        package_root=package,
        source_root=source,
        run_id=run_id,
        profile=SYSTEM_R3_REV2_PROFILE,
    )


__all__ = [
    "SYSTEM_R3_REV2_PROFILE",
    "_verify_live_source_manifest",
    "_verify_reviewed_disabled_basis",
    "run_sandbox_system_r3_rev2",
]
