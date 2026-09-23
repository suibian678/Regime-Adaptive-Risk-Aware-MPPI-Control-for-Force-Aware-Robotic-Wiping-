"""Bounded repeatable runner for the V6 S2 physical development SANDBOX.

The runner deliberately has a narrower authority than every qualification
launcher in the repository.  It can execute only the four pre-declared,
already-observed development scenarios.  It cannot create DEV, CAL, TRAIN, or
TEST output.  A permission-off invocation returns before importing a simulator,
probing a runtime, creating an environment, or opening a run directory.

Every physical interval is synchronously journalled before the causal adapter
can commit its logical bundle.  A malformed interval is additionally copied to
the storage quarantine and closes the run as ``aborted_infrastructure``.
Physical outcomes such as SAFE_HOLD, a sampled force above 15 N, an environment
termination, or the 4,500-control timeout are preserved as scientific failures.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, fields, is_dataclass
from datetime import datetime, timezone
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import re
import sys
import time
from typing import Any, Protocol
from uuid import uuid4

import numpy as np

from .causal_adapter import AdapterStepBundle
from .controller import (
    SupervisorConfig,
    SupervisorState,
    UnifiedCausalForceRecoverySupervisor,
)
from .controller_s3 import V6S3LivenessSupervisor
from .sandbox_storage import (
    AUTHORITY_METRIC_SPLIT_V3,
    AUTHORITY_METRIC_STATE_AWARE,
    CONTROL_SCHEMA,
    DIAGNOSTIC_SCHEMA,
    EPISODE_SCHEMA,
    NATIVE_SCHEMA,
    PERMISSION_FORMAT,
    PERMISSION_SCOPE,
    PERMISSION_SCOPE_S3,
    RUN_DEFINITION_FORMAT,
    RUN_DEFINITION_FORMAT_S3,
    RUN_DEFINITION_FORMAT_S3_R2,
    SandboxInfrastructureAbort,
    SandboxIntegrityError,
    SandboxPermissionError,
    SandboxRunStore,
    schema_sha256,
    verify_committed_run,
    _authority_violations as _storage_authority_violations,
)
from .sapien_sandbox_binding import (
    PhysicalIntervalEvidence,
    PhysicalIntervalFault,
    RawIntervalEvent,
    SapienSandboxBinding,
    SapienSandboxBindingConfig,
    SurfacePathGeometry,
)


PROTOCOL_FORMAT = "forcewipe_v6_s2_sandbox_protocol_v1"
PREAUTH_MANIFEST_FORMAT = "forcewipe_v6_s2_sandbox_preauth_manifest_v1"
PROTOCOL_FORMAT_S3 = "forcewipe_v6_s3_sandbox_protocol_v1"
PREAUTH_MANIFEST_FORMAT_S3 = "forcewipe_v6_s3_sandbox_preauth_manifest_v1"
PROTOCOL_FORMAT_S3_R2 = "forcewipe_v6_s3_sandbox_protocol_v2"
PREAUTH_MANIFEST_FORMAT_S3_R2 = "forcewipe_v6_s3_sandbox_preauth_manifest_v2"
SCENARIO_USE = "DEVELOPMENT_REUSED_NOT_HELDOUT"
ALLOWED_SCENARIO_IDS = (3_000_000, 3_000_013, 3_000_005, 3_000_011)
EXPECTED_TARGETS_N = {
    3_000_000: 5.0,
    3_000_013: 8.0,
    3_000_005: 12.0,
    3_000_011: 12.0,
}
MAX_CONTROL_STEPS = 4_500
TANGENTIAL_REQUEST_M = 0.001
FORCE_LIMIT_N = 15.0
COMPLETION_PROGRESS = 0.99
READBACK_STAGE = "ik_and_joint_drive_target_after_limits"
EVALUATION_NAMESPACE_BASE = 8_000_000_000_000_000_000
EVALUATION_NAMESPACE_WIDTH = 1_000_000_000_000_000_000


class V6SandboxRunnerError(RuntimeError):
    """Base error for the S2-only launcher."""


class V6SandboxPreflightError(V6SandboxRunnerError):
    """Frozen identities or permission scope do not close."""


class V6SandboxRunAborted(V6SandboxRunnerError):
    """The run closed as an infrastructure abort."""

    def __init__(self, message: str, *, final_path: Path | None = None) -> None:
        super().__init__(message)
        self.final_path = final_path


@dataclass(frozen=True)
class SandboxProfile:
    """Versioned identities for one bounded SANDBOX controller revision."""

    label: str
    protocol_format: str
    preauth_manifest_format: str
    permission_scope: str
    protocol_filename: str
    permission_filename: str
    source_manifest_filename: str
    preauth_manifest_filename: str
    run_prefix: str
    evaluation_namespace: str
    run_definition_format: str
    authority_metric_version: str
    controller_type: type[UnifiedCausalForceRecoverySupervisor]
    manual_approval_required: bool
    external_runtime_required: bool
    geometry_type: type[SurfacePathGeometry] = SurfacePathGeometry
    binding_type: type[SapienSandboxBinding] = SapienSandboxBinding


S2_PROFILE = SandboxProfile(
    label="V6-S2",
    protocol_format=PROTOCOL_FORMAT,
    preauth_manifest_format=PREAUTH_MANIFEST_FORMAT,
    permission_scope=PERMISSION_SCOPE,
    protocol_filename="V6_SANDBOX_PHYSICS_PROTOCOL.json",
    permission_filename="V6_SANDBOX_PHYSICS_PERMISSION.json",
    source_manifest_filename="V6_SANDBOX_PHYSICS_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_SANDBOX_PHYSICS_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6s2_",
    evaluation_namespace="forcewipe-v6-s2",
    run_definition_format=RUN_DEFINITION_FORMAT,
    authority_metric_version=AUTHORITY_METRIC_STATE_AWARE,
    controller_type=UnifiedCausalForceRecoverySupervisor,
    manual_approval_required=False,
    external_runtime_required=False,
)

S3_PROFILE = SandboxProfile(
    label="V6-S3",
    protocol_format=PROTOCOL_FORMAT_S3,
    preauth_manifest_format=PREAUTH_MANIFEST_FORMAT_S3,
    permission_scope=PERMISSION_SCOPE_S3,
    protocol_filename="V6_S3_SANDBOX_PHYSICS_PROTOCOL.json",
    permission_filename="V6_S3_SANDBOX_PHYSICS_PERMISSION.json",
    source_manifest_filename="V6_S3_SANDBOX_PHYSICS_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_S3_SANDBOX_PHYSICS_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6s3_",
    evaluation_namespace="forcewipe-v6-s3",
    run_definition_format=RUN_DEFINITION_FORMAT_S3,
    authority_metric_version=AUTHORITY_METRIC_SPLIT_V3,
    controller_type=V6S3LivenessSupervisor,
    manual_approval_required=True,
    external_runtime_required=False,
)

S3_R2_PROFILE = SandboxProfile(
    label="V6-S3-r2",
    protocol_format=PROTOCOL_FORMAT_S3_R2,
    preauth_manifest_format=PREAUTH_MANIFEST_FORMAT_S3_R2,
    permission_scope=PERMISSION_SCOPE_S3,
    protocol_filename="V6_S3_R2_SANDBOX_PHYSICS_PROTOCOL.json",
    permission_filename="V6_S3_R2_SANDBOX_PHYSICS_PERMISSION.json",
    source_manifest_filename="V6_S3_R2_SANDBOX_PHYSICS_SOURCE_MANIFEST.csv",
    preauth_manifest_filename="V6_S3_R2_SANDBOX_PHYSICS_PREAUTH_MANIFEST.json",
    run_prefix="sbx_v6s3r2_",
    evaluation_namespace="forcewipe-v6-s3-r2",
    run_definition_format=RUN_DEFINITION_FORMAT_S3_R2,
    authority_metric_version=AUTHORITY_METRIC_SPLIT_V3,
    controller_type=V6S3LivenessSupervisor,
    manual_approval_required=True,
    external_runtime_required=True,
)


@dataclass(frozen=True)
class ScenarioExecution:
    scenario: dict[str, Any]
    scenario_manifest_row: dict[str, Any]


@dataclass(frozen=True)
class EvaluationEvidence:
    native_rows: tuple[dict[str, Any], ...]
    control_rows: tuple[dict[str, Any], ...]
    diagnostic_rows: tuple[dict[str, Any], ...]
    episode_metrics: dict[str, Any]


@dataclass(frozen=True)
class SandboxRunResult:
    run_id: str
    final_path: Path
    status: str
    evaluations: int
    scientific_failures: int


class EvaluationExecutor(Protocol):
    def __call__(
        self,
        *,
        run_id: str,
        execution: ScenarioExecution,
        store: SandboxRunStore,
        supervisor_config: SupervisorConfig,
        protocol: Mapping[str, Any],
    ) -> EvaluationEvidence: ...


RuntimeProbe = Callable[[], Mapping[str, Any]]


def _canonical_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise V6SandboxPreflightError("value is not finite canonical JSON") from exc
    return encoded + b"\n"


def _canonical_text(value: Any) -> str:
    return _canonical_bytes(value).decode("utf-8").rstrip("\n")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise V6SandboxPreflightError(f"cannot load JSON: {path}") from exc
    if not isinstance(value, dict):
        raise V6SandboxPreflightError(f"JSON root is not an object: {path}")
    return value


def _early_permission_check(
    permission_path: Path, *, profile: SandboxProfile = S2_PROFILE
) -> dict[str, Any]:
    """Fail before runtime probing or any output when SANDBOX is closed."""

    permission = _load_json(permission_path)
    if permission.get("format") != PERMISSION_FORMAT:
        raise SandboxPermissionError("unsupported SANDBOX permission format")
    if permission.get("scope") != profile.permission_scope:
        raise SandboxPermissionError("permission scope is not SANDBOX-only")
    if permission.get("sandbox_execution_permitted") is not True:
        raise SandboxPermissionError("SANDBOX execution permission is closed")
    for key in (
        "qualification_dev_execution_permitted",
        "cal_execution_permitted",
        "train_execution_permitted",
        "test_execution_permitted",
    ):
        if permission.get(key) is not False:
            raise SandboxPermissionError(f"downstream permission {key} must remain false")
    if permission.get("allowed_output_root") != "/tmp/forcewipe_v6_sandbox":
        raise SandboxPermissionError("SANDBOX output root is not the frozen exact path")
    if permission.get("max_evaluations_per_run") != 4:
        raise SandboxPermissionError("SANDBOX evaluation bound changed")
    if permission.get("max_control_steps_per_evaluation") != MAX_CONTROL_STEPS:
        raise SandboxPermissionError("SANDBOX control-step bound changed")
    if permission.get("schema_sha256") != schema_sha256():
        raise SandboxPermissionError("SANDBOX permission schema identity is stale")
    if profile.manual_approval_required:
        for key in ("reviewer", "approved_utc", "approval_id"):
            value = permission.get(key)
            if not isinstance(value, str) or not value.strip():
                raise SandboxPermissionError(
                    f"S3 SANDBOX permission requires nonempty {key}"
                )
    return permission


def _verify_preauth_manifest(
    package_root: Path,
    permission: Mapping[str, Any],
    *,
    profile: SandboxProfile = S2_PROFILE,
) -> None:
    path = package_root / "config" / profile.preauth_manifest_filename
    manifest = _load_json(path)
    if manifest.get("format") != profile.preauth_manifest_format:
        raise V6SandboxPreflightError("unsupported preauthorization manifest")
    if manifest.get("scope") != profile.permission_scope:
        raise V6SandboxPreflightError("preauthorization manifest scope changed")
    if manifest.get("sandbox_execution_permitted") is not permission.get(
        "sandbox_execution_permitted"
    ):
        raise V6SandboxPreflightError("preauthorization permission state is stale")
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise V6SandboxPreflightError("preauthorization manifest is empty")
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise V6SandboxPreflightError("preauthorization manifest row is malformed")
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise V6SandboxPreflightError("preauthorization path escapes package root")
        key = relative.as_posix()
        if key in seen:
            raise V6SandboxPreflightError("preauthorization path is duplicated")
        seen.add(key)
        source = (package_root / relative).resolve(strict=True)
        if not source.is_relative_to(package_root):
            raise V6SandboxPreflightError("preauthorization source escapes package root")
        if (
            not source.is_file()
            or source.stat().st_size != int(row["size_bytes"])
            or _sha256_file(source) != row["sha256"]
        ):
            raise V6SandboxPreflightError(
                f"preauthorization file identity mismatch: {key}"
            )


def new_run_id(
    *,
    now: datetime | None = None,
    nonce: str | None = None,
    profile: SandboxProfile = S2_PROFILE,
) -> str:
    current = datetime.now(timezone.utc) if now is None else now.astimezone(timezone.utc)
    token = uuid4().hex[:12] if nonce is None else str(nonce).lower()
    if not token or any(character not in "0123456789abcdef" for character in token):
        raise V6SandboxPreflightError("run nonce must be nonempty lowercase hexadecimal")
    return f"{profile.run_prefix}{current.strftime('%Y%m%dT%H%M%SZ')}_{token}"


def evaluation_id_for(
    run_id: str, scenario_id: int, *, profile: SandboxProfile = S2_PROFILE
) -> int:
    payload = (
        f"{profile.evaluation_namespace}\0{run_id}\0{int(scenario_id)}"
    ).encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
    return EVALUATION_NAMESPACE_BASE + offset % EVALUATION_NAMESPACE_WIDTH


def _existing_evaluation_ids(output_root: Path) -> set[int]:
    observed: set[int] = set()
    manifests = []
    for branch in ("runs", "staging"):
        root = output_root / branch
        if root.is_dir():
            manifests.extend(root.glob("*/SCENARIO_MANIFEST.csv"))
    for manifest in manifests:
        try:
            with manifest.open("r", encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    observed.add(int(row["evaluation_id"]))
        except (OSError, KeyError, TypeError, ValueError) as exc:
            raise V6SandboxPreflightError(
                f"cannot audit prior evaluation namespace: {manifest}"
            ) from exc
    return observed


def _load_source_manifest(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    except OSError as exc:
        raise V6SandboxPreflightError("cannot load frozen source manifest") from exc
    required = {"relative_path", "size_bytes", "sha256", "role"}
    if not rows or any(set(row) != required for row in rows):
        raise V6SandboxPreflightError("frozen source manifest has invalid fields")
    return [
        {
            "relative_path": row["relative_path"],
            "size_bytes": int(row["size_bytes"]),
            "sha256": row["sha256"],
            "role": row["role"],
        }
        for row in rows
    ]


def _validate_external_python_runtime(protocol: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    """Validate and activate the exact external ForceWipe base environment."""

    runtime = protocol.get("external_python_runtime")
    if not isinstance(runtime, Mapping):
        raise V6SandboxPreflightError("protocol lacks external Python runtime")
    if set(runtime) != {"root", "files", "sys_path_prepend"}:
        raise V6SandboxPreflightError("external Python runtime fields changed")
    if runtime.get("sys_path_prepend") is not True:
        raise V6SandboxPreflightError("external Python runtime is not a frozen prepend")
    root = Path(str(runtime.get("root", "")))
    if not root.is_absolute() or root != Path("/tmp/forcewipe_legacy_workspace"):
        raise V6SandboxPreflightError("external Python runtime root changed")
    try:
        resolved_root = root.resolve(strict=True)
    except OSError as exc:
        raise V6SandboxPreflightError("external Python runtime root is absent") from exc
    files = runtime.get("files")
    expected = {
        "force_press/__init__.py",
        "force_press/press_env.py",
        "force_press/wipe_env.py",
    }
    if not isinstance(files, list) or {
        str(row.get("relative_path", ""))
        for row in files
        if isinstance(row, Mapping)
    } != expected:
        raise V6SandboxPreflightError("external Python runtime roster changed")
    seen: set[str] = set()
    for row in files:
        if not isinstance(row, Mapping) or set(row) != {
            "relative_path",
            "size_bytes",
            "sha256",
        }:
            raise V6SandboxPreflightError("external runtime row is malformed")
        relative = Path(str(row["relative_path"]))
        if relative.is_absolute() or ".." in relative.parts:
            raise V6SandboxPreflightError("external runtime path escapes its root")
        key = relative.as_posix()
        if key in seen:
            raise V6SandboxPreflightError("external runtime path is duplicated")
        seen.add(key)
        path = (resolved_root / relative).resolve(strict=True)
        if not path.is_relative_to(resolved_root):
            raise V6SandboxPreflightError("external runtime resolved outside its root")
        if (
            not path.is_file()
            or path.stat().st_size != int(row["size_bytes"])
            or _sha256_file(path) != row["sha256"]
        ):
            raise V6SandboxPreflightError(
                f"external runtime identity mismatch: {key}"
            )
    normalized = {
        "root": str(root),
        "files": [dict(row) for row in files],
        "sys_path_prepend": True,
    }
    digest = _sha256_bytes(_canonical_bytes(normalized))
    if digest != protocol.get("external_runtime_sha256"):
        raise V6SandboxPreflightError("external runtime aggregate hash changed")
    root_text = str(resolved_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    return normalized, digest


def _scenario_sha(value: Mapping[str, Any]) -> str:
    return _sha256_bytes(_canonical_text(dict(value)).encode("utf-8"))


def _load_scenarios(
    *,
    protocol: Mapping[str, Any],
    source_root: Path,
    run_id: str,
    output_root: Path,
    profile: SandboxProfile = S2_PROFILE,
) -> tuple[ScenarioExecution, ...]:
    relative = protocol.get("scenario_source_relative_path")
    expected_file_sha = protocol.get("scenario_source_sha256")
    if not isinstance(relative, str) or not isinstance(expected_file_sha, str):
        raise V6SandboxPreflightError("protocol lacks scenario-source identity")
    source = source_root / Path(relative)
    if not source.is_file() or _sha256_file(source) != expected_file_sha:
        raise V6SandboxPreflightError("development scenario source hash mismatch")
    rows: dict[int, dict[str, Any]] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or "scenario_id" not in value:
            raise V6SandboxPreflightError("malformed source scenario row")
        rows[int(value["scenario_id"])] = value

    roster = protocol.get("scenarios")
    if not isinstance(roster, list) or tuple(item.get("scenario_id") for item in roster) != ALLOWED_SCENARIO_IDS:
        raise V6SandboxPreflightError("protocol scenario roster is not the frozen four-case set")
    prior_ids = _existing_evaluation_ids(output_root)
    current_ids: set[int] = set()
    executions: list[ScenarioExecution] = []
    for registered in roster:
        scenario_id = int(registered["scenario_id"])
        scenario = rows.get(scenario_id)
        if scenario is None:
            raise V6SandboxPreflightError(f"source scenario {scenario_id} is missing")
        target = float(scenario.get("target_force_n", math.nan))
        if target != EXPECTED_TARGETS_N[scenario_id]:
            raise V6SandboxPreflightError("source scenario target changed")
        if scenario.get("role") != "CAL":
            raise V6SandboxPreflightError("reused scenario no longer has its source role")
        if registered.get("usage") != SCENARIO_USE:
            raise V6SandboxPreflightError("scenario contamination label is missing")
        if _scenario_sha(scenario) != registered.get("scenario_sha256"):
            raise V6SandboxPreflightError("selected scenario row hash mismatch")
        evaluation_id = evaluation_id_for(run_id, scenario_id, profile=profile)
        if evaluation_id in prior_ids or evaluation_id in current_ids:
            raise V6SandboxPreflightError("sandbox evaluation ID collision")
        current_ids.add(evaluation_id)
        manifest = {
            "scenario_id": scenario_id,
            "evaluation_id": evaluation_id,
            "family": SCENARIO_USE,
            "target_force_n": target,
            "seed": int(scenario["scenario_seed"]),
            "max_control_steps": MAX_CONTROL_STEPS,
            "expected_trigger": str(registered.get("expected_trigger", "")),
            "parameters_json": _canonical_text(scenario),
        }
        executions.append(ScenarioExecution(dict(scenario), manifest))
    return tuple(executions)


def _module_identity(module: Any) -> dict[str, Any]:
    path_value = getattr(module, "__file__", None)
    path = None if path_value is None else Path(path_value).resolve()
    return {
        "name": str(getattr(module, "__name__", "")),
        "version": str(getattr(module, "__version__", "unknown")),
        "file": None if path is None else str(path),
        "file_sha256": None if path is None or not path.is_file() else _sha256_file(path),
    }


def _distribution_identity(name: str) -> dict[str, Any]:
    try:
        distribution = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"name": name, "installed": False}
    record = distribution.locate_file("RECORD")
    if not record.is_file():
        candidates = tuple(
            path for path in (distribution.files or ()) if str(path).endswith(".dist-info/RECORD")
        )
        record = distribution.locate_file(candidates[0]) if candidates else record
    return {
        "name": str(distribution.metadata.get("Name", name)),
        "version": str(distribution.version),
        "installed": True,
        "record_file": str(record.resolve()) if record.is_file() else None,
        "record_sha256": _sha256_file(record) if record.is_file() else None,
    }


def default_runtime_probe() -> Mapping[str, Any]:
    """Capture runtime identities without constructing a simulator environment."""

    import gymnasium
    import mani_skill
    import polars
    import sapien

    modules = [gymnasium, mani_skill, np, polars, sapien]
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None:
        modules.append(torch)
    return {
        "format": "forcewipe_v6_sandbox_runtime_snapshot_v1",
        "python": sys.version,
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "modules": [_module_identity(module) for module in modules],
        "distributions": [
            _distribution_identity(name)
            for name in ("gymnasium", "mani-skill", "numpy", "polars", "sapien", "torch")
        ],
        "simulator_environment_created": False,
        "captured_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


class DurableRawIntervalJournal:
    """Synchronous append-only evidence written before adapter bundle commit."""

    _ORDER = {
        "pre_step": 0,
        "command_readback": 1,
        "native_sample": 2,
        "environment_outcome": 3,
        "post_step": 4,
    }

    def __init__(self, store: SandboxRunStore, evaluation_id: int) -> None:
        self.store = store
        self.evaluation_id = int(evaluation_id)
        self.sequence = 0
        self._step = -1
        self._stage = -1
        directory = store.staging_path / "raw_intervals"
        directory.mkdir(exist_ok=True)
        self.path = directory / f"evaluation_{self.evaluation_id}.jsonl"

    def __call__(self, event: RawIntervalEvent) -> None:
        stage = self._ORDER.get(event.event)
        if stage is None:
            raise V6SandboxRunnerError("unknown raw interval event")
        if int(event.key.evaluation_id) != self.evaluation_id:
            raise V6SandboxRunnerError("raw event evaluation identity mismatch")
        step = int(event.key.control_step_index)
        if stage == 0:
            if step != self._step + 1 or self._stage not in (-1, 4):
                raise V6SandboxRunnerError("raw interval step order is not contiguous")
            self._step, self._stage = step, stage
        else:
            if step != self._step or stage != self._stage + 1:
                raise V6SandboxRunnerError("raw interval event order is not closed")
            self._stage = stage
        record = {
            "format": "forcewipe_v6_raw_physical_event_v1",
            "sequence": self.sequence,
            "event": _jsonable(event),
        }
        payload = _canonical_bytes(record)
        with self.path.open("ab", buffering=0) as stream:
            stream.write(payload)
            os.fsync(stream.fileno())
        self.sequence += 1

    @property
    def closed_intervals(self) -> int:
        if self.sequence == 0:
            return 0
        if self._stage != 4:
            raise V6SandboxRunnerError("raw physical journal ends mid-interval")
        return self._step + 1


def _put_vector(row: dict[str, Any], names: Sequence[str], value: Sequence[float]) -> None:
    if len(names) != len(value):
        raise V6SandboxRunnerError("vector width does not match evidence schema")
    row.update({name: float(component) for name, component in zip(names, value)})


def _exact_row(schema: Any, values: Mapping[str, Any], *, table: str) -> dict[str, Any]:
    expected = set(schema.names())
    if set(values) != expected:
        missing = sorted(expected - set(values))
        extra = sorted(set(values) - expected)
        raise V6SandboxRunnerError(
            f"{table} conversion does not close: missing={missing}, extra={extra}"
        )
    return dict(values)


def _native_row(bundle: AdapterStepBundle, interval: PhysicalIntervalEvidence) -> dict[str, Any]:
    if len(bundle.native_samples) != 1 or len(interval.native) != 1:
        raise V6SandboxRunnerError("physical SANDBOX requires exactly one native sample")
    native = bundle.native_samples[0]
    physical = interval.native[0].state
    pre = bundle.pre
    row: dict[str, Any] = {
        **native.key.as_dict(),
        "command_id": native.command_id,
        "native_sample_index": native.native_sample_index,
        "substep_index": native.substep_index,
        "time_ns": native.time_ns,
        "audit_force_n": native.audit_force_n,
        "contact_observed": native.contact_observed,
        "normal_velocity_outward_m_s": native.normal_velocity_outward_m_s,
        "instantaneous_progress": native.instantaneous_progress,
        "geometric_track_error_m": native.geometric_track_error_m,
        "recovery_hover_pose_error_m": native.recovery_hover_pose_error_m,
        "recovery_clearance_m": native.recovery_clearance_m,
        "signed_recovery_clearance_m": physical.signed_recovery_clearance_m,
        "geometry_frame_id": pre.geometry_frame_id,
        "contact_tool_frame_id": pre.contact_tool_frame_id,
        "robot_tcp_frame_id": pre.robot_tcp_frame_id,
        "task_path_id": pre.task_path_id,
        "surface_id": pre.surface_id,
        "task_projection_progress": native.task_projection_progress,
        "task_projection_arc_length_m": native.task_projection_arc_length_m,
        "task_path_length_m": pre.task_path_length_m,
    }
    for names, value in (
        (("surface_reference_x_m", "surface_reference_y_m", "surface_reference_z_m"), native.surface_reference_point_xyz_m),
        (("projection_outward_normal_x", "projection_outward_normal_y", "projection_outward_normal_z"), native.projection_outward_normal_xyz),
        (("command_outward_normal_x", "command_outward_normal_y", "command_outward_normal_z"), native.command_outward_normal_xyz),
        (("committed_recovery_outward_normal_x", "committed_recovery_outward_normal_y", "committed_recovery_outward_normal_z"), native.committed_recovery_outward_normal_xyz),
        (("task_projection_point_x_m", "task_projection_point_y_m", "task_projection_point_z_m"), native.task_projection_point_xyz_m),
        (("projection_tangent_x", "projection_tangent_y", "projection_tangent_z"), native.projection_tangent_unit_xyz),
        (("command_tangent_x", "command_tangent_y", "command_tangent_z"), native.command_tangent_unit_xyz),
        (("recovery_hover_target_x_m", "recovery_hover_target_y_m", "recovery_hover_target_z_m"), native.recovery_hover_target_position_xyz_m),
        (("contact_tool_x_m", "contact_tool_y_m", "contact_tool_z_m"), native.contact_tool_position_xyz_m),
        (("contact_tool_qw", "contact_tool_qx", "contact_tool_qy", "contact_tool_qz"), native.contact_tool_quaternion_wxyz),
        (("contact_tool_vx_m_s", "contact_tool_vy_m_s", "contact_tool_vz_m_s"), native.contact_tool_linear_velocity_xyz_m_s),
        (("contact_tool_wx_rad_s", "contact_tool_wy_rad_s", "contact_tool_wz_rad_s"), native.contact_tool_angular_velocity_xyz_rad_s),
        (("robot_tcp_x_m", "robot_tcp_y_m", "robot_tcp_z_m"), native.robot_tcp_position_xyz_m),
        (("robot_tcp_qw", "robot_tcp_qx", "robot_tcp_qy", "robot_tcp_qz"), native.robot_tcp_quaternion_wxyz),
        (("robot_tcp_vx_m_s", "robot_tcp_vy_m_s", "robot_tcp_vz_m_s"), native.robot_tcp_linear_velocity_xyz_m_s),
        (("robot_tcp_wx_rad_s", "robot_tcp_wy_rad_s", "robot_tcp_wz_rad_s"), native.robot_tcp_angular_velocity_xyz_rad_s),
    ):
        _put_vector(row, names, value)
    return _exact_row(NATIVE_SCHEMA, row, table="native")


def _control_row(bundle: AdapterStepBundle, interval: PhysicalIntervalEvidence) -> dict[str, Any]:
    pre, issued, readback, post = bundle.pre, bundle.issued, bundle.readback, bundle.post
    physical_pre = interval.pre.state
    row: dict[str, Any] = {
        **pre.key.as_dict(),
        "command_id": issued.command_id,
        "pre_time_ns": pre.pre_time_ns,
        "state_source_kind": pre.state_source_kind,
        "force_source_native_sample_index": pre.force_source_native_sample_index,
        "force_source_time_ns": pre.force_source_time_ns,
        "bootstrap_id": pre.bootstrap_id,
        "pre_force_n": pre.measured_force_n,
        "target_force_n": pre.target_force_n,
        "pre_contact_observed": pre.contact_observed,
        "normal_velocity_outward_m_s": pre.normal_velocity_outward_m_s,
        "instantaneous_progress": pre.instantaneous_progress,
        "geometric_track_error_m": pre.geometric_track_error_m,
        "requested_tangential_step_m": pre.requested_tangential_step_m,
        "recovery_hover_pose_error_m": pre.recovery_hover_pose_error_m,
        "recovery_clearance_m": pre.recovery_clearance_m,
        "geometry_frame_id": pre.geometry_frame_id,
        "command_frame_id": pre.command_frame_id,
        "contact_tool_frame_id": pre.contact_tool_frame_id,
        "robot_tcp_frame_id": pre.robot_tcp_frame_id,
        "task_path_id": pre.task_path_id,
        "surface_id": pre.surface_id,
        "task_projection_progress": pre.task_projection_progress,
        "task_projection_arc_length_m": pre.task_projection_arc_length_m,
        "task_path_length_m": pre.task_path_length_m,
        "signed_recovery_clearance_m": physical_pre.signed_recovery_clearance_m,
        "issued_time_ns": issued.issued_time_ns,
        "issued_command_frame_id": issued.command_frame_id,
        "issued_command_mode": issued.command_mode,
        "issued_normal_step_m": issued.normal_step_m,
        "issued_tangential_step_m": issued.task_tangential_step_m,
        "issued_recovery_reposition_permitted": issued.recovery_reposition_permitted,
        "issued_cross_track_correction_permitted": (
            issued.cross_track_correction_permitted
        ),
        "readback_time_ns": readback.readback_time_ns,
        "readback_accepted": readback.accepted,
        "readback_stage": readback.readback_stage,
        "applied_command_frame_id": readback.command_frame_id,
        "applied_command_mode": readback.command_mode,
        "applied_normal_step_m": readback.normal_step_m,
        "applied_tangential_step_m": readback.task_tangential_step_m,
        "applied_recovery_reposition_permitted": readback.recovery_reposition_permitted,
        "applied_cross_track_correction_permitted": (
            readback.cross_track_correction_permitted
        ),
        "post_time_ns": post.post_time_ns,
        "last_native_sample_index": post.last_native_sample_index,
        "last_native_time_ns": post.last_native_time_ns,
        "post_force_n": post.measured_force_n,
        "post_contact_observed": post.contact_observed,
        "post_normal_velocity_outward_m_s": post.normal_velocity_outward_m_s,
        "post_instantaneous_progress": post.instantaneous_progress,
        "post_geometric_track_error_m": post.geometric_track_error_m,
        "post_recovery_hover_pose_error_m": post.recovery_hover_pose_error_m,
        "post_recovery_clearance_m": post.recovery_clearance_m,
    }
    vector_mappings = (
        (("surface_reference_x_m", "surface_reference_y_m", "surface_reference_z_m"), pre.surface_reference_point_xyz_m),
        (("task_projection_point_x_m", "task_projection_point_y_m", "task_projection_point_z_m"), pre.task_projection_point_xyz_m),
        (("pre_contact_tool_x_m", "pre_contact_tool_y_m", "pre_contact_tool_z_m"), pre.contact_tool_position_xyz_m),
        (("pre_contact_tool_qw", "pre_contact_tool_qx", "pre_contact_tool_qy", "pre_contact_tool_qz"), pre.contact_tool_quaternion_wxyz),
        (("pre_contact_tool_vx_m_s", "pre_contact_tool_vy_m_s", "pre_contact_tool_vz_m_s"), pre.contact_tool_linear_velocity_xyz_m_s),
        (("pre_contact_tool_wx_rad_s", "pre_contact_tool_wy_rad_s", "pre_contact_tool_wz_rad_s"), pre.contact_tool_angular_velocity_xyz_rad_s),
        (("pre_robot_tcp_x_m", "pre_robot_tcp_y_m", "pre_robot_tcp_z_m"), pre.robot_tcp_position_xyz_m),
        (("pre_robot_tcp_qw", "pre_robot_tcp_qx", "pre_robot_tcp_qy", "pre_robot_tcp_qz"), pre.robot_tcp_quaternion_wxyz),
        (("pre_robot_tcp_vx_m_s", "pre_robot_tcp_vy_m_s", "pre_robot_tcp_vz_m_s"), pre.robot_tcp_linear_velocity_xyz_m_s),
        (("pre_robot_tcp_wx_rad_s", "pre_robot_tcp_wy_rad_s", "pre_robot_tcp_wz_rad_s"), pre.robot_tcp_angular_velocity_xyz_rad_s),
        (("projection_outward_normal_x", "projection_outward_normal_y", "projection_outward_normal_z"), pre.projection_outward_normal_xyz),
        (("command_outward_normal_x", "command_outward_normal_y", "command_outward_normal_z"), pre.command_outward_normal_xyz),
        (("committed_recovery_outward_normal_x", "committed_recovery_outward_normal_y", "committed_recovery_outward_normal_z"), pre.committed_recovery_outward_normal_xyz),
        (("projection_tangent_x", "projection_tangent_y", "projection_tangent_z"), pre.projection_tangent_unit_xyz),
        (("command_tangent_x", "command_tangent_y", "command_tangent_z"), pre.command_tangent_unit_xyz),
        (("recovery_hover_target_x_m", "recovery_hover_target_y_m", "recovery_hover_target_z_m"), pre.recovery_hover_target_position_xyz_m),
        (("requested_recovery_dx_m", "requested_recovery_dy_m", "requested_recovery_dz_m"), pre.requested_recovery_delta_xyz_m),
        (("requested_cross_track_dx_m", "requested_cross_track_dy_m", "requested_cross_track_dz_m"), pre.requested_cross_track_correction_delta_xyz_m),
        (("requested_recovery_rx_rad", "requested_recovery_ry_rad", "requested_recovery_rz_rad"), pre.requested_recovery_rotation_delta_euler_xyz_rad),
        (("issued_normal_dx_m", "issued_normal_dy_m", "issued_normal_dz_m"), issued.normal_delta_xyz_m),
        (("issued_task_dx_m", "issued_task_dy_m", "issued_task_dz_m"), issued.task_delta_xyz_m),
        (("issued_recovery_dx_m", "issued_recovery_dy_m", "issued_recovery_dz_m"), issued.recovery_delta_xyz_m),
        (("issued_cross_track_dx_m", "issued_cross_track_dy_m", "issued_cross_track_dz_m"), issued.cross_track_correction_delta_xyz_m),
        (("issued_dx_m", "issued_dy_m", "issued_dz_m"), issued.cartesian_delta_xyz_m),
        (("issued_recovery_rx_rad", "issued_recovery_ry_rad", "issued_recovery_rz_rad"), issued.recovery_rotation_delta_euler_xyz_rad),
        (("issued_rotation_rx_rad", "issued_rotation_ry_rad", "issued_rotation_rz_rad"), issued.rotation_delta_euler_xyz_rad),
        (("applied_normal_dx_m", "applied_normal_dy_m", "applied_normal_dz_m"), readback.normal_delta_xyz_m),
        (("applied_task_dx_m", "applied_task_dy_m", "applied_task_dz_m"), readback.task_delta_xyz_m),
        (("applied_recovery_dx_m", "applied_recovery_dy_m", "applied_recovery_dz_m"), readback.recovery_delta_xyz_m),
        (("applied_cross_track_dx_m", "applied_cross_track_dy_m", "applied_cross_track_dz_m"), readback.cross_track_correction_delta_xyz_m),
        (("applied_dx_m", "applied_dy_m", "applied_dz_m"), readback.cartesian_delta_xyz_m),
        (("applied_recovery_rx_rad", "applied_recovery_ry_rad", "applied_recovery_rz_rad"), readback.recovery_rotation_delta_euler_xyz_rad),
        (("applied_rotation_rx_rad", "applied_rotation_ry_rad", "applied_rotation_rz_rad"), readback.rotation_delta_euler_xyz_rad),
        (("post_contact_tool_x_m", "post_contact_tool_y_m", "post_contact_tool_z_m"), post.contact_tool_position_xyz_m),
        (("post_contact_tool_qw", "post_contact_tool_qx", "post_contact_tool_qy", "post_contact_tool_qz"), post.contact_tool_quaternion_wxyz),
        (("post_contact_tool_vx_m_s", "post_contact_tool_vy_m_s", "post_contact_tool_vz_m_s"), post.contact_tool_linear_velocity_xyz_m_s),
        (("post_contact_tool_wx_rad_s", "post_contact_tool_wy_rad_s", "post_contact_tool_wz_rad_s"), post.contact_tool_angular_velocity_xyz_rad_s),
        (("post_robot_tcp_x_m", "post_robot_tcp_y_m", "post_robot_tcp_z_m"), post.robot_tcp_position_xyz_m),
        (("post_robot_tcp_qw", "post_robot_tcp_qx", "post_robot_tcp_qy", "post_robot_tcp_qz"), post.robot_tcp_quaternion_wxyz),
        (("post_robot_tcp_vx_m_s", "post_robot_tcp_vy_m_s", "post_robot_tcp_vz_m_s"), post.robot_tcp_linear_velocity_xyz_m_s),
        (("post_robot_tcp_wx_rad_s", "post_robot_tcp_wy_rad_s", "post_robot_tcp_wz_rad_s"), post.robot_tcp_angular_velocity_xyz_rad_s),
    )
    for names, value in vector_mappings:
        _put_vector(row, names, value)
    return _exact_row(CONTROL_SCHEMA, row, table="control")


def _diagnostic_row(bundle: AdapterStepBundle) -> dict[str, Any]:
    row = {**bundle.key.as_dict(), **asdict(bundle.supervisor_command)}
    return _exact_row(DIAGNOSTIC_SCHEMA, row, table="diagnostic")


def _authority_violations(
    rows: Sequence[Mapping[str, Any]],
    *,
    metric_version: str = AUTHORITY_METRIC_STATE_AWARE,
) -> int:
    return _storage_authority_violations(
        rows, metric_version=metric_version
    )


def _command_readback_violations(
    rows: Sequence[Mapping[str, Any]],
    *,
    translation_tolerance_m: float = 2e-7,
    rotation_tolerance_rad: float = 2e-7,
) -> int:
    translation_pairs = (
        ("issued_normal_step_m", "applied_normal_step_m"),
        ("issued_tangential_step_m", "applied_tangential_step_m"),
        *(
            (f"issued_{component}_d{axis}_m", f"applied_{component}_d{axis}_m")
            for component in ("normal", "task", "recovery", "cross_track")
            for axis in "xyz"
        ),
        ("issued_dx_m", "applied_dx_m"),
        ("issued_dy_m", "applied_dy_m"),
        ("issued_dz_m", "applied_dz_m"),
    )
    rotation_pairs = (
        *(
            (f"issued_recovery_r{axis}_rad", f"applied_recovery_r{axis}_rad")
            for axis in "xyz"
        ),
        *(
            (f"issued_rotation_r{axis}_rad", f"applied_rotation_r{axis}_rad")
            for axis in "xyz"
        ),
    )
    count = 0
    for row in rows:
        mismatch = not bool(row["readback_accepted"])
        mismatch |= any(
            abs(float(row[left]) - float(row[right])) > translation_tolerance_m
            for left, right in translation_pairs
        )
        mismatch |= any(
            abs(float(row[left]) - float(row[right])) > rotation_tolerance_rad
            for left, right in rotation_pairs
        )
        mismatch |= bool(row["issued_recovery_reposition_permitted"]) != bool(
            row["applied_recovery_reposition_permitted"]
        )
        mismatch |= bool(row["issued_cross_track_correction_permitted"]) != bool(
            row["applied_cross_track_correction_permitted"]
        )
        mismatch |= (
            row["issued_command_frame_id"] != row["applied_command_frame_id"]
            or row["issued_command_mode"] != row["applied_command_mode"]
            or row["readback_stage"] != READBACK_STAGE
        )
        count += int(mismatch)
    return count


def _episode_metrics(
    *,
    run_id: str,
    execution: ScenarioExecution,
    native_rows: Sequence[Mapping[str, Any]],
    control_rows: Sequence[Mapping[str, Any]],
    diagnostic_rows: Sequence[Mapping[str, Any]],
    terminal_status: str,
    failure_reasons: Sequence[str],
    lifecycle_complete: bool,
    wall_time_s: float,
    authority_metric_version: str = AUTHORITY_METRIC_STATE_AWARE,
) -> dict[str, Any]:
    target = float(execution.scenario_manifest_row["target_force_n"])
    forces = [float(row["audit_force_n"]) for row in native_rows]
    initial_force = float(control_rows[0]["pre_force_n"])
    all_forces = [initial_force, *forces]
    violations = sum(force > FORCE_LIMIT_N for force in all_forces)
    authority = _authority_violations(
        diagnostic_rows, metric_version=authority_metric_version
    )
    readback = _command_readback_violations(control_rows)
    stable_forces = [
        float(row["measured_force_n"])
        for row in diagnostic_rows
        if bool(row["stable_track"])
    ]
    tracking_rmse = (
        None
        if not stable_forces
        else math.sqrt(sum((force - target) ** 2 for force in stable_forces) / len(stable_forces))
    )
    band_fraction = (
        None
        if not stable_forces
        else sum(abs(force - target) <= 0.25 * target for force in stable_forces)
        / len(stable_forces)
    )
    reasons = list(dict.fromkeys(str(reason) for reason in failure_reasons if reason))
    if violations:
        reasons.append("native_force_limit_violation")
    if authority:
        reasons.append("authority_invariant_violation")
    if readback:
        reasons.append("command_readback_violation")
    scientific_failure = bool(reasons or not lifecycle_complete)
    if scientific_failure and not reasons:
        reasons.append("lifecycle_incomplete")
    row = {
        "run_id": run_id,
        "scenario_id": int(execution.scenario_manifest_row["scenario_id"]),
        "evaluation_id": int(execution.scenario_manifest_row["evaluation_id"]),
        "target_force_n": target,
        "terminal_status": terminal_status,
        "scientific_failure": scientific_failure,
        "scientific_failure_reason": ";".join(dict.fromkeys(reasons)),
        "lifecycle_complete": bool(lifecycle_complete),
        "control_steps": len(control_rows),
        "native_samples": len(native_rows),
        "peak_force_n": max(all_forces),
        "force_limit_violation_count": violations,
        "authority_violation_count": authority,
        "command_readback_violation_count": readback,
        "tracking_rmse_n": tracking_rmse,
        "band_fraction": band_fraction,
        "contact_fraction": sum(bool(row["contact_observed"]) for row in native_rows)
        / len(native_rows),
        "final_committed_progress": float(diagnostic_rows[-1]["committed_progress_after"]),
        "final_supervisor_state": str(diagnostic_rows[-1]["state_after"]),
        "wall_time_s": float(wall_time_s),
    }
    return _exact_row(EPISODE_SCHEMA, row, table="episode")


def _fault_payload(fault: PhysicalIntervalFault) -> dict[str, Any]:
    return {
        "format": "forcewipe_v6_physical_interval_fault_v1",
        "fault": _jsonable(fault),
    }


class PhysicalSapienEvaluationExecutor:
    """Create and close one ForceWipeV4-v1 environment per scenario."""

    def __init__(
        self,
        *,
        controller_type: type[UnifiedCausalForceRecoverySupervisor] = UnifiedCausalForceRecoverySupervisor,
        authority_metric_version: str = AUTHORITY_METRIC_STATE_AWARE,
        geometry_type: type[SurfacePathGeometry] = SurfacePathGeometry,
        binding_type: type[SapienSandboxBinding] = SapienSandboxBinding,
    ) -> None:
        self._controller_type = controller_type
        self._authority_metric_version = authority_metric_version
        self._geometry_type = geometry_type
        self._binding_type = binding_type

    def __call__(
        self,
        *,
        run_id: str,
        execution: ScenarioExecution,
        store: SandboxRunStore,
        supervisor_config: SupervisorConfig,
        protocol: Mapping[str, Any],
    ) -> EvaluationEvidence:
        # Delayed imports ensure a closed permission never imports or initializes
        # the SAPIEN environment stack.
        from dataclasses import asdict as dataclass_dict

        import gymnasium as gym
        import forcewipe_v4.sapien_v4_env  # noqa: F401 - registers ForceWipeV4-v1
        from forcewipe_v4.scenarios import (
            ScenarioSpec,
            make_scenario_path,
            surface_normal,
        )

        manifest = execution.scenario_manifest_row
        scenario_id = int(manifest["scenario_id"])
        evaluation_id = int(manifest["evaluation_id"])
        spec = ScenarioSpec(**execution.scenario)
        spec.validate()
        path = make_scenario_path(spec, points=801)
        env = None
        started = time.perf_counter()
        try:
            env = gym.make(
                "ForceWipeV4-v1",
                scenario_spec=dataclass_dict(spec),
                lifecycle_mode=True,
                num_envs=1,
                obs_mode="state_dict",
                reward_mode="dense",
                control_mode="pd_ee_delta_pose",
                render_mode=None,
                sim_backend="physx_cpu",
                render_backend="none",
                sim_config=dict(control_freq=100),
                max_episode_steps=MAX_CONTROL_STEPS + 10,
            )
            env.reset(seed=int(spec.scenario_seed))
            raw = env.unwrapped
            supervisor = self._controller_type(supervisor_config)
            geometry = self._geometry_type(
                path,
                lambda x_value: surface_normal(spec, x_value),
                surface_top_origin_z_m=float(raw.v4_surface_geometry.top_origin_z_m),
                recovery_hover_clearance_m=supervisor_config.recovery_hover_clearance_m,
            )
            journal = DurableRawIntervalJournal(store, evaluation_id)
            fault_seen: list[PhysicalIntervalFault] = []

            def fault_sink(fault: PhysicalIntervalFault) -> None:
                fault_seen.append(fault)
                safe_stage = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(fault.stage)).strip("_")
                store.record_quarantine_interval(
                    evaluation_id=evaluation_id,
                    control_step_index=int(fault.key.control_step_index),
                    stage=safe_stage or "physical_binding",
                    raw_interval=_fault_payload(fault),
                )

            binding = self._binding_type(
                env,
                raw,
                supervisor,
                geometry,
                run_id=run_id,
                scenario_id=scenario_id,
                evaluation_id=evaluation_id,
                target_force_n=float(spec.target_force_n),
                bootstrap_id=f"{run_id}:{evaluation_id}:bootstrap",
                task_path_id=f"v4-scenario-{scenario_id}-path",
                surface_id=f"v4-scenario-{scenario_id}-{spec.surface_kind}",
                tangential_request_source=lambda _key, _snapshot: TANGENTIAL_REQUEST_M,
                raw_interval_sink=journal,
                interval_fault_sink=fault_sink,
                config=SapienSandboxBindingConfig(),
            )
            bundles: list[AdapterStepBundle] = []
            intervals: list[PhysicalIntervalEvidence] = []
            terminal_status = "tracking_timeout"
            reasons: list[str] = []
            lifecycle_complete = False
            for _ in range(MAX_CONTROL_STEPS):
                bundle = binding.step()
                interval = binding.physical_intervals[-1]
                bundles.append(bundle)
                intervals.append(interval)
                command = bundle.supervisor_command
                if any(float(sample.audit_force_n) > FORCE_LIMIT_N for sample in bundle.native_samples):
                    terminal_status = "force_limit_violation"
                    reasons.append("native_force_limit_violation")
                    break
                if command.state_after == SupervisorState.SAFE_HOLD.value:
                    terminal_status = "safe_hold"
                    reasons.append("safe_hold")
                    break
                outcome = interval.environment_outcome
                if outcome.terminated or outcome.truncated:
                    terminal_status = "environment_termination"
                    reasons.append("environment_termination")
                    break
                if (
                    float(command.committed_progress_after) >= COMPLETION_PROGRESS
                    and bool(command.stable_track)
                ):
                    terminal_status = "sandbox_path_complete"
                    lifecycle_complete = True
                    break
            else:
                reasons.append("tracking_timeout")
            if fault_seen:
                raise V6SandboxRunnerError("binding returned after emitting a physical fault")
            if journal.closed_intervals != len(bundles):
                raise V6SandboxRunnerError("raw journal and logical bundle counts differ")
            if not bundles:
                raise V6SandboxRunnerError("physical evaluation produced no control interval")
            native_rows = tuple(
                _native_row(bundle, interval)
                for bundle, interval in zip(bundles, intervals)
            )
            control_rows = tuple(
                _control_row(bundle, interval)
                for bundle, interval in zip(bundles, intervals)
            )
            diagnostic_rows = tuple(_diagnostic_row(bundle) for bundle in bundles)
            episode = _episode_metrics(
                run_id=run_id,
                execution=execution,
                native_rows=native_rows,
                control_rows=control_rows,
                diagnostic_rows=diagnostic_rows,
                terminal_status=terminal_status,
                failure_reasons=reasons,
                lifecycle_complete=lifecycle_complete,
                wall_time_s=time.perf_counter() - started,
                authority_metric_version=self._authority_metric_version,
            )
            return EvaluationEvidence(
                native_rows=native_rows,
                control_rows=control_rows,
                diagnostic_rows=diagnostic_rows,
                episode_metrics=episode,
            )
        finally:
            if env is not None:
                env.close()


def _supervisor_config(path: Path) -> SupervisorConfig:
    payload = _load_json(path)
    allowed = {item.name for item in fields(SupervisorConfig)}
    unknown = set(payload) - allowed - {"status"}
    if unknown:
        raise V6SandboxPreflightError(f"supervisor config has unknown fields: {sorted(unknown)}")
    config = SupervisorConfig(**{key: value for key, value in payload.items() if key in allowed})
    config.validate()
    return config


def run_sandbox(
    *,
    package_root: Path,
    source_root: Path,
    run_id: str | None = None,
    runtime_probe: RuntimeProbe = default_runtime_probe,
    evaluation_executor: EvaluationExecutor | None = None,
    profile: SandboxProfile = S2_PROFILE,
) -> SandboxRunResult:
    package_root = Path(package_root).resolve(strict=True)
    source_root = Path(source_root).resolve(strict=True)
    protocol_path = package_root / "config" / profile.protocol_filename
    permission_path = package_root / "config" / profile.permission_filename
    source_manifest_path = package_root / "config" / profile.source_manifest_filename
    supervisor_config_path = package_root / "config" / "V6_SANDBOX_CONFIG.json"

    # This is intentionally first.  Nothing below it may import the simulator,
    # probe packages, construct an environment, or create output.
    permission = _early_permission_check(permission_path, profile=profile)
    _verify_preauth_manifest(package_root, permission, profile=profile)
    protocol = _load_json(protocol_path)
    if protocol.get("format") != profile.protocol_format:
        raise V6SandboxPreflightError(f"unsupported {profile.label} protocol")
    if protocol.get("scope") != profile.permission_scope:
        raise V6SandboxPreflightError("protocol is not SANDBOX-only")
    if permission.get("protocol_sha256") != _sha256_file(protocol_path):
        raise V6SandboxPreflightError("permission does not bind the active protocol")
    if permission.get("source_manifest_sha256") != _sha256_file(source_manifest_path):
        raise V6SandboxPreflightError("permission does not bind the source manifest")
    if protocol.get("source_manifest_sha256") != _sha256_file(source_manifest_path):
        raise V6SandboxPreflightError("protocol does not bind the source manifest")
    if protocol.get("schema_sha256") != schema_sha256():
        raise V6SandboxPreflightError("protocol schema identity is stale")
    if protocol.get("authority_metric_version") != profile.authority_metric_version:
        raise V6SandboxPreflightError("protocol authority metric does not match profile")
    if (
        profile in {S3_PROFILE, S3_R2_PROFILE}
        and protocol.get("controller_revision")
        != profile.controller_type.controller_revision
    ):
        raise V6SandboxPreflightError("protocol controller revision does not match profile")
    if protocol.get("trusted_infrastructure_sha256") != permission.get(
        "trusted_infrastructure_sha256"
    ):
        raise V6SandboxPreflightError("protocol and permission infrastructure differ")
    if protocol.get("scenario_usage") != SCENARIO_USE:
        raise V6SandboxPreflightError("protocol does not mark reused development scenarios")
    if int(protocol.get("max_control_steps_per_evaluation", -1)) != MAX_CONTROL_STEPS:
        raise V6SandboxPreflightError("protocol step bound changed")
    if float(protocol.get("tangential_request_m", math.nan)) != TANGENTIAL_REQUEST_M:
        raise V6SandboxPreflightError("protocol task command changed")
    if any(protocol.get("downstream_permissions", {}).get(name) is not False for name in ("qualification_dev", "cal", "train", "test")):
        raise V6SandboxPreflightError("protocol downstream permission is open")
    external_runtime: dict[str, Any] | None = None
    external_runtime_sha256: str | None = None
    if profile.external_runtime_required:
        external_runtime, external_runtime_sha256 = _validate_external_python_runtime(
            protocol
        )
    output_root = Path(str(permission["allowed_output_root"]))
    actual_run_id = (
        new_run_id(profile=profile) if run_id is None else str(run_id)
    )
    if not actual_run_id.startswith(profile.run_prefix):
        raise V6SandboxPreflightError("run id is outside the active profile namespace")
    executions = _load_scenarios(
        protocol=protocol,
        source_root=source_root,
        run_id=actual_run_id,
        output_root=output_root,
        profile=profile,
    )
    if len(executions) != 4:
        raise V6SandboxPreflightError("exactly four bounded scenarios are required")
    source_manifest = _load_source_manifest(source_manifest_path)
    config = _supervisor_config(supervisor_config_path)
    runtime_error: Exception | None = None
    try:
        runtime_snapshot = dict(runtime_probe())
        runtime_snapshot["simulator_environment_created"] = False
    except Exception as exc:
        runtime_error = exc
        runtime_snapshot = {
            "format": "forcewipe_v6_sandbox_runtime_snapshot_v1",
            "runtime_probe_completed": False,
            "simulator_environment_created": False,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "captured_utc": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
        }

    infrastructure = permission["trusted_infrastructure_sha256"]
    definition = {
        "format": profile.run_definition_format,
        "run_id": actual_run_id,
        "parent_run_id": None,
        "controller_revision": protocol["controller_revision"],
        "evaluation_id_namespace": profile.evaluation_namespace,
        "run_prefix": profile.run_prefix,
        "controller_sha256": protocol["controller_sha256"],
        "adapter_sha256": infrastructure["adapter"],
        "schema_sha256": schema_sha256(),
        "authority_metric_version": profile.authority_metric_version,
        "trusted_infrastructure_sha256": infrastructure,
        "force_limit_n": FORCE_LIMIT_N,
        "command_readback_tolerance_m": 2e-7,
        "pose_alignment_tolerance_m": 1e-7,
        "basis_tolerance": 1e-9,
        "quaternion_tolerance": 1e-8,
        "rotation_readback_tolerance_rad": 2e-7,
        "maximum_recovery_reposition_step_m": 0.006,
        "maximum_recovery_rotation_step_rad": 0.0,
        "maximum_task_tangential_step_m": 0.0035,
        "recovery_hover_clearance_m": config.recovery_hover_clearance_m,
        "cross_track_correction_enter_m": 0.003,
        "maximum_cross_track_correction_step_m": 0.0005,
        "expected_command_frame_id": "world",
        "expected_command_mode": "delta_pose_euler_xyz",
        "required_readback_stage": READBACK_STAGE,
        "expected_native_steps_per_control": 1,
        "native_period_ns": 10_000_000,
        "control_period_ns": 10_000_000,
    }
    if external_runtime is not None:
        definition["external_python_runtime"] = external_runtime
        definition["external_runtime_sha256"] = external_runtime_sha256
    store = SandboxRunStore.begin(
        permission_path=permission_path,
        run_id=actual_run_id,
        run_definition=definition,
        source_root=source_root,
        source_manifest=source_manifest,
        scenario_manifest=[execution.scenario_manifest_row for execution in executions],
        runtime_snapshot=runtime_snapshot,
        change_rationale={
            "parent_run_id": None,
            "source_failure_ids": protocol.get(
                "source_failure_ids",
                ["V5.7-R3-REV2-SCENARIO-3000011-TAIL-REBOUND"],
            ),
            "hypothesis": protocol.get(
                "development_hypothesis",
                "V6 unified causal supervisor closes tail rebound and recovery authority in a repeatable development sandbox.",
            ),
            "scenario_usage": SCENARIO_USE,
            "changed_hashes": protocol.get("lineage_changed_hashes", {}),
        },
    )
    if runtime_error is not None:
        final_path = store.abort_infrastructure(
            reason=f"{type(runtime_error).__name__}: {runtime_error}",
            stage="runtime_probe",
        )
        raise V6SandboxRunAborted(str(runtime_error), final_path=final_path)
    executor = (
        PhysicalSapienEvaluationExecutor(
            controller_type=profile.controller_type,
            authority_metric_version=profile.authority_metric_version,
            geometry_type=profile.geometry_type,
            binding_type=profile.binding_type,
        )
        if evaluation_executor is None
        else evaluation_executor
    )
    episode_summaries: list[dict[str, Any]] = []
    try:
        for execution in executions:
            evidence = executor(
                run_id=actual_run_id,
                execution=execution,
                store=store,
                supervisor_config=config,
                protocol=protocol,
            )
            store.append_evaluation(
                evaluation_id=int(execution.scenario_manifest_row["evaluation_id"]),
                native_rows=evidence.native_rows,
                control_rows=evidence.control_rows,
                diagnostic_rows=evidence.diagnostic_rows,
                episode_metrics=evidence.episode_metrics,
            )
            episode_summaries.append(
                {
                    "scenario_id": evidence.episode_metrics["scenario_id"],
                    "evaluation_id": evidence.episode_metrics["evaluation_id"],
                    "target_force_n": evidence.episode_metrics["target_force_n"],
                    "terminal_status": evidence.episode_metrics["terminal_status"],
                    "scientific_failure": evidence.episode_metrics["scientific_failure"],
                    "control_steps": evidence.episode_metrics["control_steps"],
                    "native_samples": evidence.episode_metrics["native_samples"],
                    "peak_force_n": evidence.episode_metrics["peak_force_n"],
                    "force_limit_violation_count": evidence.episode_metrics[
                        "force_limit_violation_count"
                    ],
                    "final_committed_progress": evidence.episode_metrics[
                        "final_committed_progress"
                    ],
                    "final_supervisor_state": evidence.episode_metrics[
                        "final_supervisor_state"
                    ],
                }
            )
        final_path = store.complete(
            summary={
                "scope": profile.permission_scope,
                "scenario_usage": SCENARIO_USE,
                "qualification_claim": False,
                "cal_claim": False,
                "train_claim": False,
                "test_claim": False,
                "total_control_steps": sum(
                    int(row["control_steps"]) for row in episode_summaries
                ),
                "total_native_samples": sum(
                    int(row["native_samples"]) for row in episode_summaries
                ),
                "peak_force_n": max(
                    float(row["peak_force_n"]) for row in episode_summaries
                ),
                "force_limit_violation_count": sum(
                    int(row["force_limit_violation_count"])
                    for row in episode_summaries
                ),
                "scenario_results": episode_summaries,
            }
        )
    except SandboxInfrastructureAbort:
        raise
    except Exception as exc:
        try:
            final_path = store.abort_infrastructure(
                reason=f"{type(exc).__name__}: {exc}",
                stage="sandbox_runner",
            )
        except Exception:
            final_path = None
        raise V6SandboxRunAborted(str(exc), final_path=final_path) from exc
    verified = verify_committed_run(final_path)
    summary = _load_json(final_path / "RUN_SUMMARY.json")
    return SandboxRunResult(
        run_id=actual_run_id,
        final_path=final_path,
        status=verified.status,
        evaluations=verified.evaluations_verified,
        scientific_failures=int(summary["scientific_failures"]),
    )


def run_sandbox_s3(
    *,
    package_root: Path,
    source_root: Path,
    run_id: str | None = None,
    runtime_probe: RuntimeProbe = default_runtime_probe,
    evaluation_executor: EvaluationExecutor | None = None,
) -> SandboxRunResult:
    """Run the bounded V6-S3 development SANDBOX under its own identities."""

    return run_sandbox(
        package_root=package_root,
        source_root=source_root,
        run_id=run_id,
        runtime_probe=runtime_probe,
        evaluation_executor=evaluation_executor,
        profile=S3_PROFILE,
    )


def run_sandbox_s3_r2(
    *,
    package_root: Path,
    source_root: Path,
    run_id: str | None = None,
    runtime_probe: RuntimeProbe = default_runtime_probe,
    evaluation_executor: EvaluationExecutor | None = None,
) -> SandboxRunResult:
    """Run the external-runtime-pinned V6-S3-r2 development SANDBOX."""

    return run_sandbox(
        package_root=package_root,
        source_root=source_root,
        run_id=run_id,
        runtime_probe=runtime_probe,
        evaluation_executor=evaluation_executor,
        profile=S3_R2_PROFILE,
    )


__all__ = [
    "ALLOWED_SCENARIO_IDS",
    "COMPLETION_PROGRESS",
    "DurableRawIntervalJournal",
    "EvaluationEvidence",
    "FORCE_LIMIT_N",
    "MAX_CONTROL_STEPS",
    "PhysicalSapienEvaluationExecutor",
    "PROTOCOL_FORMAT",
    "READBACK_STAGE",
    "PREAUTH_MANIFEST_FORMAT",
    "PREAUTH_MANIFEST_FORMAT_S3",
    "PREAUTH_MANIFEST_FORMAT_S3_R2",
    "SCENARIO_USE",
    "S2_PROFILE",
    "S3_PROFILE",
    "S3_R2_PROFILE",
    "SandboxProfile",
    "SandboxRunResult",
    "ScenarioExecution",
    "TANGENTIAL_REQUEST_M",
    "V6SandboxPreflightError",
    "V6SandboxRunAborted",
    "V6SandboxRunnerError",
    "default_runtime_probe",
    "evaluation_id_for",
    "new_run_id",
    "run_sandbox",
    "run_sandbox_s3",
    "run_sandbox_s3_r2",
]
