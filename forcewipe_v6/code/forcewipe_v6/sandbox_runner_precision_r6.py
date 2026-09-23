"""Repeatable four-case physical SANDBOX profile for precision TRACK r6."""

from __future__ import annotations

from .authority_metrics_precision_r5 import AUTHORITY_METRIC_PRECISION_R5
from .controller_precision_r6 import V6PrecisionContactSupervisorR6
from .sandbox_runner import SandboxProfile, SandboxRunResult, run_sandbox
from .sandbox_storage import PERMISSION_SCOPE_S5, RUN_DEFINITION_FORMAT_S5
from .sapien_sandbox_binding_precision_r6 import (
    PrecisionR6SapienSandboxBinding,
    PrecisionR6SurfacePathGeometry,
)


PRECISION_R6_PROFILE = SandboxProfile(
    label="V6-precision-track-r6",
    protocol_format="forcewipe_v6_precision_r6_sandbox_protocol_v1",
    preauth_manifest_format="forcewipe_v6_precision_r6_sandbox_preauth_manifest_v1",
    permission_scope=PERMISSION_SCOPE_S5,
    protocol_filename="V6_PRECISION_R6_SANDBOX_PROTOCOL.json",
    permission_filename="V6_PRECISION_R6_SANDBOX_PERMISSION.json",
    source_manifest_filename="V6_PRECISION_R6_SANDBOX_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_PRECISION_R6_SANDBOX_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6precisionr6_",
    evaluation_namespace="forcewipe-v6-precision-r6-repeatable-sandbox-v1",
    run_definition_format=RUN_DEFINITION_FORMAT_S5,
    authority_metric_version=AUTHORITY_METRIC_PRECISION_R5,
    controller_type=V6PrecisionContactSupervisorR6,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=PrecisionR6SurfacePathGeometry,
    binding_type=PrecisionR6SapienSandboxBinding,
)


def run_precision_r6_sandbox(
    *, package_root, source_root, run_id: str | None = None
) -> SandboxRunResult:
    return run_sandbox(
        package_root=package_root,
        source_root=source_root,
        run_id=run_id,
        profile=PRECISION_R6_PROFILE,
    )


__all__ = ["PRECISION_R6_PROFILE", "run_precision_r6_sandbox"]
