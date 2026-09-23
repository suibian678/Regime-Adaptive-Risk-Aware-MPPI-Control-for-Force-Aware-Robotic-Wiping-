"""Repeatable four-case physical SANDBOX profile for precision TRACK r3."""

from __future__ import annotations

from .authority_metrics_s5 import AUTHORITY_METRIC_PATH_FRAME_V4
from .controller_precision_r3 import V6PrecisionContactSupervisorR3
from .sandbox_runner import SandboxProfile, SandboxRunResult, run_sandbox
from .sandbox_storage import PERMISSION_SCOPE_S5, RUN_DEFINITION_FORMAT_S5
from .sapien_sandbox_binding_precision_r3 import (
    PrecisionR3SapienSandboxBinding,
    PrecisionR3SurfacePathGeometry,
)


PRECISION_R3_PROFILE = SandboxProfile(
    label="V6-precision-track-r3",
    protocol_format="forcewipe_v6_precision_r3_sandbox_protocol_v1",
    preauth_manifest_format="forcewipe_v6_precision_r3_sandbox_preauth_manifest_v1",
    permission_scope=PERMISSION_SCOPE_S5,
    protocol_filename="V6_PRECISION_R3_SANDBOX_PROTOCOL.json",
    permission_filename="V6_PRECISION_R3_SANDBOX_PERMISSION.json",
    source_manifest_filename="V6_PRECISION_R3_SANDBOX_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_PRECISION_R3_SANDBOX_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6precisionr3_",
    evaluation_namespace="forcewipe-v6-precision-r3-repeatable-sandbox-v1",
    run_definition_format=RUN_DEFINITION_FORMAT_S5,
    authority_metric_version=AUTHORITY_METRIC_PATH_FRAME_V4,
    controller_type=V6PrecisionContactSupervisorR3,
    manual_approval_required=True,
    external_runtime_required=True,
    geometry_type=PrecisionR3SurfacePathGeometry,
    binding_type=PrecisionR3SapienSandboxBinding,
)


def run_precision_r3_sandbox(
    *, package_root, source_root, run_id: str | None = None
) -> SandboxRunResult:
    return run_sandbox(
        package_root=package_root,
        source_root=source_root,
        run_id=run_id,
        profile=PRECISION_R3_PROFILE,
    )


__all__ = ["PRECISION_R3_PROFILE", "run_precision_r3_sandbox"]
