"""Bounded execution-chain revision of the V6-S5-r2 SANDBOX profile."""

from __future__ import annotations

import json
from pathlib import Path

from .authority_metrics_s5 import AUTHORITY_METRIC_PATH_FRAME_V4
from .controller_s5 import V6S5PathFrameSupervisor
from .sandbox_runner import (
    SandboxProfile,
    SandboxRunResult,
    V6SandboxPreflightError,
    _early_permission_check,
    run_sandbox,
)
from .sandbox_storage import PERMISSION_SCOPE_S5, RUN_DEFINITION_FORMAT_S5
from .sapien_sandbox_binding_s5 import S5SapienSandboxBinding, S5SurfacePathGeometry


PROTOCOL_FORMAT = "forcewipe_v6_s5_r2_sandbox_protocol_rev2_v1"
PREAUTH_MANIFEST_FORMAT = "forcewipe_v6_s5_r2_sandbox_preauth_manifest_rev2_v1"

S5_R2_REV2_PROFILE = SandboxProfile(
    label="V6-S5-r2-preauth-rev2",
    protocol_format=PROTOCOL_FORMAT,
    preauth_manifest_format=PREAUTH_MANIFEST_FORMAT,
    permission_scope=PERMISSION_SCOPE_S5,
    protocol_filename="V6_S5_R2_SANDBOX_PHYSICS_PROTOCOL_REV2.json",
    permission_filename="V6_S5_R2_SANDBOX_PHYSICS_PERMISSION_REV2.json",
    source_manifest_filename="V6_S5_R2_SANDBOX_PHYSICS_SOURCE_MANIFEST_REV2.csv",
    preauth_manifest_filename="V6_S5_R2_SANDBOX_PHYSICS_PREAUTH_MANIFEST_REV2.json",
    run_prefix="sbx_v6s5r2rev2_",
    evaluation_namespace="forcewipe-v6-s5-r2-preauth-rev2",
    run_definition_format=RUN_DEFINITION_FORMAT_S5,
    authority_metric_version=AUTHORITY_METRIC_PATH_FRAME_V4,
    controller_type=V6S5PathFrameSupervisor,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=S5SurfacePathGeometry,
    binding_type=S5SapienSandboxBinding,
)


def _verify_identity(package_root: Path) -> None:
    path = package_root / "config" / S5_R2_REV2_PROFILE.protocol_filename
    try:
        protocol = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise V6SandboxPreflightError("cannot read the S5-r2 rev2 protocol") from exc
    expected = {
        "controller_revision": V6S5PathFrameSupervisor.controller_revision,
        "binding_type": "forcewipe_v6.sapien_sandbox_binding_s5.S5SapienSandboxBinding",
        "geometry_type": "forcewipe_v6.sapien_sandbox_binding_s5.S5SurfacePathGeometry",
        "execution_chain_revision": "pre_request_only_post_projection_v2",
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise V6SandboxPreflightError(f"S5-r2 rev2 {key} is not frozen")


def run_sandbox_s5_r2_rev2(
    *, package_root: Path, source_root: Path, run_id: str | None = None
) -> SandboxRunResult:
    package = Path(package_root).resolve(strict=True)
    _early_permission_check(
        package / "config" / S5_R2_REV2_PROFILE.permission_filename,
        profile=S5_R2_REV2_PROFILE,
    )
    _verify_identity(package)
    return run_sandbox(
        package_root=package,
        source_root=source_root,
        run_id=run_id,
        profile=S5_R2_REV2_PROFILE,
    )


__all__ = ["S5_R2_REV2_PROFILE", "run_sandbox_s5_r2_rev2"]
