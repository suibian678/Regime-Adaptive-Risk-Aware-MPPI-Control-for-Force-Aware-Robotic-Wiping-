"""Fresh repeatable-development SANDBOX profile for the V6 system R2 controller."""

from __future__ import annotations

import json
from pathlib import Path

from .authority_metrics_s5 import AUTHORITY_METRIC_PATH_FRAME_V4
from .controller_system_r2 import V6SystemContactSupervisorR2
from .sandbox_runner import (
    SandboxProfile,
    SandboxRunResult,
    V6SandboxPreflightError,
    _early_permission_check,
    run_sandbox,
)
from .sandbox_storage import PERMISSION_SCOPE_S5, RUN_DEFINITION_FORMAT_S5
from .sapien_sandbox_binding_s5 import S5SapienSandboxBinding, S5SurfacePathGeometry


PROTOCOL_FORMAT = "forcewipe_v6_system_r2_sandbox_protocol_v1"
PREAUTH_MANIFEST_FORMAT = "forcewipe_v6_system_r2_sandbox_preauth_manifest_v1"

SYSTEM_R2_PROFILE = SandboxProfile(
    label="V6-system-r2",
    protocol_format=PROTOCOL_FORMAT,
    preauth_manifest_format=PREAUTH_MANIFEST_FORMAT,
    permission_scope=PERMISSION_SCOPE_S5,
    protocol_filename="V6_SYSTEM_R2_SANDBOX_PROTOCOL.json",
    permission_filename="V6_SYSTEM_R2_SANDBOX_PERMISSION.json",
    source_manifest_filename="V6_SYSTEM_R2_SANDBOX_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_SYSTEM_R2_SANDBOX_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6systemr2_",
    evaluation_namespace="forcewipe-v6-system-r2-sandbox-v1",
    run_definition_format=RUN_DEFINITION_FORMAT_S5,
    authority_metric_version=AUTHORITY_METRIC_PATH_FRAME_V4,
    controller_type=V6SystemContactSupervisorR2,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=S5SurfacePathGeometry,
    binding_type=S5SapienSandboxBinding,
)


def _verify_identity(package_root: Path) -> None:
    path = package_root / "config" / SYSTEM_R2_PROFILE.protocol_filename
    try:
        protocol = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V6SandboxPreflightError("cannot read the V6 system R2 protocol") from exc
    expected = {
        "controller_revision": V6SystemContactSupervisorR2.controller_revision,
        "controller_type": "forcewipe_v6.controller_system_r2.V6SystemContactSupervisorR2",
        "binding_type": "forcewipe_v6.sapien_sandbox_binding_s5.S5SapienSandboxBinding",
        "geometry_type": "forcewipe_v6.sapien_sandbox_binding_s5.S5SurfacePathGeometry",
        "evaluation_id_namespace": SYSTEM_R2_PROFILE.evaluation_namespace,
        "historical_provenance_exception_accepted": True,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise V6SandboxPreflightError(f"V6 system R2 {key} is not frozen")


def run_sandbox_system_r2(
    *, package_root: Path, source_root: Path, run_id: str | None = None
) -> SandboxRunResult:
    package = Path(package_root).resolve(strict=True)
    _early_permission_check(
        package / "config" / SYSTEM_R2_PROFILE.permission_filename,
        profile=SYSTEM_R2_PROFILE,
    )
    _verify_identity(package)
    return run_sandbox(
        package_root=package,
        source_root=source_root,
        run_id=run_id,
        profile=SYSTEM_R2_PROFILE,
    )


__all__ = ["SYSTEM_R2_PROFILE", "run_sandbox_system_r2"]
