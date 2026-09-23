"""Repeatable four-case physical SANDBOX profile for precision TRACK r4."""

from __future__ import annotations

from .authority_metrics_s5 import AUTHORITY_METRIC_PATH_FRAME_V4
from .controller_precision_r4 import V6PrecisionContactSupervisorR4
from .sandbox_runner import SandboxProfile, SandboxRunResult, run_sandbox
from .sandbox_storage import PERMISSION_SCOPE_S5, RUN_DEFINITION_FORMAT_S5
from .sapien_sandbox_binding_precision_r4 import (
    PrecisionR4SapienSandboxBinding,
    PrecisionR4SurfacePathGeometry,
)


PRECISION_R4_PROFILE = SandboxProfile(
    label="V6-precision-track-r4",
    protocol_format="forcewipe_v6_precision_r4_sandbox_protocol_v1",
    preauth_manifest_format="forcewipe_v6_precision_r4_sandbox_preauth_manifest_v1",
    permission_scope=PERMISSION_SCOPE_S5,
    protocol_filename="V6_PRECISION_R4_SANDBOX_PROTOCOL.json",
    permission_filename="V6_PRECISION_R4_SANDBOX_PERMISSION.json",
    source_manifest_filename="V6_PRECISION_R4_SANDBOX_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_PRECISION_R4_SANDBOX_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6precisionr4_",
    evaluation_namespace="forcewipe-v6-precision-r4-repeatable-sandbox-v1",
    run_definition_format=RUN_DEFINITION_FORMAT_S5,
    authority_metric_version=AUTHORITY_METRIC_PATH_FRAME_V4,
    controller_type=V6PrecisionContactSupervisorR4,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=PrecisionR4SurfacePathGeometry,
    binding_type=PrecisionR4SapienSandboxBinding,
)


def run_precision_r4_sandbox(
    *, package_root, source_root, run_id: str | None = None
) -> SandboxRunResult:
    return run_sandbox(
        package_root=package_root,
        source_root=source_root,
        run_id=run_id,
        profile=PRECISION_R4_PROFILE,
    )


__all__ = ["PRECISION_R4_PROFILE", "run_precision_r4_sandbox"]
