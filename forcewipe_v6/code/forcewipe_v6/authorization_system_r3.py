"""Strict procedural-approval validation for System R3 SANDBOX runs."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
import hashlib
import json
import re
from typing import Any, Mapping


APPROVAL_ID_RE = re.compile(
    r"V6-SYSTEM-R3-REV2-SANDBOX-[0-9a-f]{8}-"
    r"[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}"
)
UTC_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def validate_approval_identity(
    reviewer: Any,
    approved_utc: Any,
    approval_id: Any,
) -> tuple[str, str, str]:
    """Validate the explicit human approval fields used by the local gate."""

    if not isinstance(reviewer, str) or not 2 <= len(reviewer.strip()) <= 128:
        raise ValueError("reviewer must be a named identity")
    if not isinstance(approved_utc, str) or UTC_RE.fullmatch(approved_utc) is None:
        raise ValueError("approved_utc must be second-resolution RFC3339 UTC")
    try:
        datetime.strptime(approved_utc, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError as exc:
        raise ValueError("approved_utc is not a valid UTC timestamp") from exc
    if not isinstance(approval_id, str) or APPROVAL_ID_RE.fullmatch(approval_id) is None:
        raise ValueError("approval_id must be a System R3 rev2 SANDBOX UUID identity")
    return reviewer.strip(), approved_utc, approval_id


def validate_review_binding(permission: Mapping[str, Any]) -> None:
    """Require the human-reviewed identities to equal the executable ones."""

    if permission.get("approval_reviewed_protocol_sha256") != permission.get(
        "protocol_sha256"
    ):
        raise ValueError("approval does not bind the active R3 protocol")
    if permission.get("approval_reviewed_source_manifest_sha256") != permission.get(
        "source_manifest_sha256"
    ):
        raise ValueError("approval does not bind the active R3 source manifest")
    for name in (
        "approval_reviewed_disabled_permission_sha256",
        "approval_reviewed_disabled_preauth_manifest_sha256",
    ):
        value = permission.get(name)
        if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"approval lacks reviewed basis: {name}")


def validate_review_basis_reconstruction(
    permission: Mapping[str, Any],
    preauth: Mapping[str, Any],
    *,
    permission_relative_path: str,
) -> None:
    """Rebuild the reviewed disabled bytes from the enabled package.

    This makes the two disabled-basis hashes independently checkable after
    authorization; the executable package cannot satisfy the gate by merely
    copying arbitrary SHA strings into its permission fields.
    """

    validate_approval_identity(
        permission.get("reviewer"),
        permission.get("approved_utc"),
        permission.get("approval_id"),
    )
    validate_review_binding(permission)
    if permission.get("sandbox_execution_permitted") is not True:
        raise ValueError("review-basis reconstruction requires enabled permission")
    if preauth.get("sandbox_execution_permitted") is not True:
        raise ValueError("review-basis reconstruction requires enabled preauth")

    field_pairs = (
        ("reviewed_protocol_sha256", "approval_reviewed_protocol_sha256"),
        (
            "reviewed_source_manifest_sha256",
            "approval_reviewed_source_manifest_sha256",
        ),
        (
            "reviewed_disabled_permission_sha256",
            "approval_reviewed_disabled_permission_sha256",
        ),
        (
            "reviewed_disabled_preauth_manifest_sha256",
            "approval_reviewed_disabled_preauth_manifest_sha256",
        ),
    )
    for preauth_name, permission_name in field_pairs:
        if preauth.get(preauth_name) != permission.get(permission_name):
            raise ValueError(f"enabled preauth review binding differs: {preauth_name}")

    disabled_permission = deepcopy(dict(permission))
    disabled_permission.update(
        {
            "sandbox_execution_permitted": False,
            "reviewer": None,
            "approved_utc": None,
            "approval_id": None,
            "approval_reviewed_protocol_sha256": None,
            "approval_reviewed_source_manifest_sha256": None,
            "approval_reviewed_disabled_permission_sha256": None,
            "approval_reviewed_disabled_preauth_manifest_sha256": None,
        }
    )
    disabled_permission_bytes = canonical_json_bytes(disabled_permission)
    disabled_permission_sha = hashlib.sha256(disabled_permission_bytes).hexdigest()
    if disabled_permission_sha != permission.get(
        "approval_reviewed_disabled_permission_sha256"
    ):
        raise ValueError("enabled permission cannot reconstruct reviewed disabled bytes")

    disabled_preauth = deepcopy(dict(preauth))
    disabled_preauth.update(
        {
            "sandbox_execution_permitted": False,
            "reviewed_protocol_sha256": None,
            "reviewed_source_manifest_sha256": None,
            "reviewed_disabled_permission_sha256": None,
            "reviewed_disabled_preauth_manifest_sha256": None,
        }
    )
    rows = disabled_preauth.get("files")
    if not isinstance(rows, list):
        raise ValueError("enabled preauth lacks its file rows")
    changed = 0
    for row in rows:
        if isinstance(row, dict) and row.get("relative_path") == permission_relative_path:
            row["size_bytes"] = len(disabled_permission_bytes)
            row["sha256"] = disabled_permission_sha
            changed += 1
    if changed != 1:
        raise ValueError("enabled preauth lacks exactly one permission row")
    disabled_preauth_sha = hashlib.sha256(
        canonical_json_bytes(disabled_preauth)
    ).hexdigest()
    if disabled_preauth_sha != permission.get(
        "approval_reviewed_disabled_preauth_manifest_sha256"
    ):
        raise ValueError("enabled preauth cannot reconstruct reviewed disabled bytes")


__all__ = [
    "APPROVAL_ID_RE",
    "UTC_RE",
    "SHA256_RE",
    "canonical_json_bytes",
    "validate_approval_identity",
    "validate_review_binding",
    "validate_review_basis_reconstruction",
]
