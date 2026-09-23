"""Repeatable four-case physical SANDBOX profile for precision TRACK r5."""

from __future__ import annotations

from .authority_metrics_precision_r5 import AUTHORITY_METRIC_PRECISION_R5
from .controller_precision_r5 import V6PrecisionContactSupervisorR5
from .sandbox_runner import SandboxProfile, SandboxRunResult, run_sandbox
from .sandbox_storage import PERMISSION_SCOPE_S5, RUN_DEFINITION_FORMAT_S5
from .sapien_sandbox_binding_precision_r5 import (
    PrecisionR5SapienSandboxBinding,
    PrecisionR5SurfacePathGeometry,
)


PRECISION_R5_PROFILE = SandboxProfile(
    label="V6-precision-track-r5",
    protocol_format="forcewipe_v6_precision_r5_sandbox_protocol_v1",
    preauth_manifest_format="forcewipe_v6_precision_r5_sandbox_preauth_manifest_v1",
    permission_scope=PERMISSION_SCOPE_S5,
    protocol_filename="V6_PRECISION_R5_SANDBOX_PROTOCOL.json",
    permission_filename="V6_PRECISION_R5_SANDBOX_PERMISSION.json",
    source_manifest_filename="V6_PRECISION_R5_SANDBOX_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_PRECISION_R5_SANDBOX_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6precisionr5_",
    evaluation_namespace="forcewipe-v6-precision-r5-repeatable-sandbox-v1",
    run_definition_format=RUN_DEFINITION_FORMAT_S5,
    authority_metric_version=AUTHORITY_METRIC_PRECISION_R5,
    controller_type=V6PrecisionContactSupervisorR5,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=PrecisionR5SurfacePathGeometry,
    binding_type=PrecisionR5SapienSandboxBinding,
)


def run_precision_r5_sandbox(
    *, package_root, source_root, run_id: str | None = None
) -> SandboxRunResult:
    return run_sandbox(
        package_root=package_root,
        source_root=source_root,
        run_id=run_id,
        profile=PRECISION_R5_PROFILE,
    )


__all__ = ["PRECISION_R5_PROFILE", "run_precision_r5_sandbox"]
