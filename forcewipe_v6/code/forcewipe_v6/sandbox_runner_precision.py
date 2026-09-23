"""Repeatable four-case physical SANDBOX for exact-target TRACK development."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .authority_metrics_s5 import AUTHORITY_METRIC_PATH_FRAME_V4
from .controller_precision import V6PrecisionContactSupervisor
from .sandbox_runner import (
    SandboxProfile,
    SandboxRunResult,
    V6SandboxPreflightError,
    _early_permission_check,
    _verify_preauth_manifest,
    run_sandbox,
)
from .sandbox_storage import (
    PERMISSION_SCOPE_S5,
    RUN_DEFINITION_FORMAT_S5,
)
from .sapien_sandbox_binding_precision import (
    PrecisionSapienSandboxBinding,
    PrecisionSurfacePathGeometry,
)


PROTOCOL_FORMAT = "forcewipe_v6_precision_sandbox_protocol_v1"
PREAUTH_MANIFEST_FORMAT = "forcewipe_v6_precision_sandbox_preauth_manifest_v1"

PRECISION_PROFILE = SandboxProfile(
    label="V6-precision-track",
    protocol_format=PROTOCOL_FORMAT,
    preauth_manifest_format=PREAUTH_MANIFEST_FORMAT,
    permission_scope=PERMISSION_SCOPE_S5,
    protocol_filename="V6_PRECISION_SANDBOX_PROTOCOL.json",
    permission_filename="V6_PRECISION_SANDBOX_PERMISSION.json",
    source_manifest_filename="V6_PRECISION_SANDBOX_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_PRECISION_SANDBOX_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6precision_",
    evaluation_namespace="forcewipe-v6-precision-repeatable-sandbox-v1",
    run_definition_format=RUN_DEFINITION_FORMAT_S5,
    authority_metric_version=AUTHORITY_METRIC_PATH_FRAME_V4,
    controller_type=V6PrecisionContactSupervisor,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=PrecisionSurfacePathGeometry,
    binding_type=PrecisionSapienSandboxBinding,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V6SandboxPreflightError(f"cannot read precision JSON: {path}") from exc
    if not isinstance(value, dict):
        raise V6SandboxPreflightError("precision JSON root is not an object")
    return value


def _verify_identity(package_root: Path) -> None:
    config = package_root / "config"
    protocol_path = config / PRECISION_PROFILE.protocol_filename
    permission_path = config / PRECISION_PROFILE.permission_filename
    protocol = _load_json(protocol_path)
    permission = _load_json(permission_path)
    expected = {
        "controller_revision": V6PrecisionContactSupervisor.controller_revision,
        "controller_type": (
            "forcewipe_v6.controller_precision.V6PrecisionContactSupervisor"
        ),
        "binding_type": (
            "forcewipe_v6.sapien_sandbox_binding_precision."
            "PrecisionSapienSandboxBinding"
        ),
        "geometry_type": (
            "forcewipe_v6.sapien_sandbox_binding_precision."
            "PrecisionSurfacePathGeometry"
        ),
        "evaluation_id_namespace": PRECISION_PROFILE.evaluation_namespace,
        "run_prefix": PRECISION_PROFILE.run_prefix,
        "run_definition_format": PRECISION_PROFILE.run_definition_format,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise V6SandboxPreflightError(f"precision protocol changed: {key}")
        if key in {
            "controller_revision",
            "evaluation_id_namespace",
            "run_prefix",
            "run_definition_format",
        } and permission.get(key) != value:
            raise V6SandboxPreflightError(f"precision permission changed: {key}")
    controller_path = package_root / "code" / "forcewipe_v6" / "controller_precision.py"
    if protocol.get("controller_sha256") != _sha256_file(controller_path):
        raise V6SandboxPreflightError("precision controller identity changed")


def run_precision_sandbox(
    *, package_root: Path, source_root: Path, run_id: str | None = None
) -> SandboxRunResult:
    package = Path(package_root).resolve(strict=True)
    permission = _early_permission_check(
        package / "config" / PRECISION_PROFILE.permission_filename,
        profile=PRECISION_PROFILE,
    )
    _verify_preauth_manifest(package, permission, profile=PRECISION_PROFILE)
    _verify_identity(package)
    return run_sandbox(
        package_root=package,
        source_root=Path(source_root).resolve(strict=True),
        run_id=run_id,
        profile=PRECISION_PROFILE,
    )


__all__ = ["PRECISION_PROFILE", "run_precision_sandbox"]
