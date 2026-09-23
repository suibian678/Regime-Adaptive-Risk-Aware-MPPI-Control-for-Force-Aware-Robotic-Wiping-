"""Repeatable four-case physical SANDBOX profile for precision TRACK r7."""

from __future__ import annotations

from .authority_metrics_precision_r5 import AUTHORITY_METRIC_PRECISION_R5
from .controller_precision_r7 import V6PrecisionContactSupervisorR7
from .sandbox_runner import SandboxProfile, SandboxRunResult, run_sandbox
from .sandbox_storage import PERMISSION_SCOPE_S5, RUN_DEFINITION_FORMAT_S5
from .sapien_sandbox_binding_precision_r7 import (
    PrecisionR7SapienSandboxBinding,
    PrecisionR7SurfacePathGeometry,
)


PRECISION_R7_PROFILE = SandboxProfile(
    label="V6-precision-track-r7",
    protocol_format="forcewipe_v6_precision_r7_sandbox_protocol_v1",
    preauth_manifest_format="forcewipe_v6_precision_r7_sandbox_preauth_manifest_v1",
    permission_scope=PERMISSION_SCOPE_S5,
    protocol_filename="V6_PRECISION_R7_SANDBOX_PROTOCOL.json",
    permission_filename="V6_PRECISION_R7_SANDBOX_PERMISSION.json",
    source_manifest_filename="V6_PRECISION_R7_SANDBOX_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_PRECISION_R7_SANDBOX_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6precisionr7_",
    evaluation_namespace="forcewipe-v6-precision-r7-repeatable-sandbox-v1",
    run_definition_format=RUN_DEFINITION_FORMAT_S5,
    authority_metric_version=AUTHORITY_METRIC_PRECISION_R5,
    controller_type=V6PrecisionContactSupervisorR7,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=PrecisionR7SurfacePathGeometry,
    binding_type=PrecisionR7SapienSandboxBinding,
)


def run_precision_r7_sandbox(
    *, package_root, source_root, run_id: str | None = None
) -> SandboxRunResult:
    return run_sandbox(
        package_root=package_root,
        source_root=source_root,
        run_id=run_id,
        profile=PRECISION_R7_PROFILE,
    )


__all__ = ["PRECISION_R7_PROFILE", "run_precision_r7_sandbox"]
