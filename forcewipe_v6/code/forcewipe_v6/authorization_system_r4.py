"""Named repeatable-development authorization identity for System R4."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Mapping


APPROVAL_ID_RE = re.compile(
    r"V6-SYSTEM-R4-SANDBOX-[0-9a-f]{8}-"
    r"[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def validate_approval_identity(
    reviewer: Any,
    approved_utc: Any,
    approval_id: Any,
) -> tuple[str, str, str]:
    if not isinstance(reviewer, str) or not 2 <= len(reviewer.strip()) <= 128:
        raise ValueError("reviewer must be a named identity")
    if not isinstance(approved_utc, str) or UTC_RE.fullmatch(approved_utc) is None:
        raise ValueError("approved_utc must be second-resolution RFC3339 UTC")
    try:
        datetime.strptime(approved_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("approved_utc is not a valid UTC timestamp") from exc
    if not isinstance(approval_id, str) or APPROVAL_ID_RE.fullmatch(approval_id) is None:
        raise ValueError("approval_id must be a System R4 SANDBOX UUID identity")
    return reviewer.strip(), approved_utc, approval_id


def validate_review_binding(permission: Mapping[str, Any]) -> None:
    if permission.get("approval_reviewed_protocol_sha256") != permission.get(
        "protocol_sha256"
    ):
        raise ValueError("approval does not bind the active R4 protocol")
    if permission.get("approval_reviewed_source_manifest_sha256") != permission.get(
        "source_manifest_sha256"
    ):
        raise ValueError("approval does not bind the active R4 source manifest")
    for name in (
        "approval_reviewed_disabled_permission_sha256",
        "approval_reviewed_disabled_preauth_manifest_sha256",
    ):
        value = permission.get(name)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"approval lacks reviewed basis: {name}")


__all__ = ["validate_approval_identity", "validate_review_binding"]
