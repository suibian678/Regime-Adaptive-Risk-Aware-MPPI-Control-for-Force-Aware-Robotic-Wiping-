"""Bounded Polars-only Parquet/Zstd transaction writer for ForceWipe V4.

The implementation deliberately separates scientific table schemas from the
storage mechanism.  A table is admitted only through a canonical JSON schema
contract.  Readers see a bundle only after its commit marker exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import csv
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
from typing import Any, Callable, Iterable, Mapping
import uuid

import polars as pl


DATA_CONTRACT_VERSION = "4.0.0"
SAFE_CODE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
MANIFEST_COLUMNS = (
    "relative_path",
    "table",
    "bundle_id",
    "bytes",
    "rows",
    "min_evaluation_id",
    "max_evaluation_id",
    "min_step",
    "max_step",
    "schema_sha256",
    "file_sha256",
)


class StreamingDataError(RuntimeError):
    """A fail-closed data-contract or transaction error."""


class FaultPoint(str, Enum):
    BEFORE_TABLE_WRITE = "before_table_write"
    AFTER_TABLE_WRITE = "after_table_write"
    AFTER_PARTIAL_MOVE = "after_partial_move"
    BEFORE_COMMIT = "before_commit"


FaultInjector = Callable[[FaultPoint, Mapping[str, Any]], None]


def _no_fault(_point: FaultPoint, _context: Mapping[str, Any]) -> None:
    return None


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe_code(value: str, field: str) -> str:
    value = str(value)
    if not SAFE_CODE.fullmatch(value) or value in {".", ".."}:
        raise StreamingDataError(f"unsafe {field}: {value!r}")
    return value


def _filesystem_type(path: Path) -> str | None:
    """Return the most specific Linux mount type without an external command."""
    if os.name != "posix" or not Path("/proc/self/mountinfo").is_file():
        return None
    resolved = Path(path).resolve()
    best: tuple[int, str] | None = None
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        left, separator, right = line.partition(" - ")
        if not separator:
            continue
        fields = left.split()
        if len(fields) < 5:
            continue
        mount = Path(fields[4].replace("\\040", " ")).resolve()
        if resolved != mount and mount not in resolved.parents:
            continue
        filesystem = right.split()[0]
        candidate = (len(mount.parts), filesystem)
        if best is None or candidate[0] > best[0]:
            best = candidate
    return None if best is None else best[1]


def _fsync_file(path: Path) -> None:
    with Path(path).open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(Path(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("xb") as stream:
        stream.write(canonical_json_bytes(payload))
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError as error:
        raise StreamingDataError(f"immutable marker already exists: {path.name}") from error
    finally:
        temporary.unlink(missing_ok=True)
    _fsync_directory(path.parent)


POLARS_TYPES: dict[str, pl.DataType] = {
    "UInt8": pl.UInt8,
    "UInt16": pl.UInt16,
    "UInt32": pl.UInt32,
    "UInt64": pl.UInt64,
    "Int8": pl.Int8,
    "Int16": pl.Int16,
    "Int32": pl.Int32,
    "Int64": pl.Int64,
    "Float32": pl.Float32,
    "Float64": pl.Float64,
    "Boolean": pl.Boolean,
    "String": pl.String,
    "Binary": pl.Binary,
}

EVALUATION_MANIFEST_SCHEMA: dict[str, pl.DataType] = {
    "evaluation_id": pl.UInt64,
    "split_code": pl.String,
    "study_code": pl.String,
    "authority_code": pl.String,
    "method_code": pl.String,
    "training_seed": pl.UInt64,
    "seed_code": pl.String,
    "scenario_id": pl.UInt64,
    "target_code": pl.UInt16,
    "repeat": pl.UInt16,
    "scenario_seed": pl.UInt64,
    "checkpoint_sha256": pl.String,
    "config_sha256": pl.String,
    "code_sha256": pl.String,
    "planned_native_max": pl.UInt32,
    "planned_control_max": pl.UInt32,
    "planned_primitive_max": pl.UInt16,
    "role": pl.String,
}
ROW_LIMIT_COLUMNS = {
    "native_trace_v1": "planned_native_max",
    "control_trace_v1": "planned_control_max",
    "primitive_metrics_v1": "planned_primitive_max",
    "episode_metrics_v1": None,
    "simulated_rgb_frame_v1": "planned_control_max",
}


@dataclass(frozen=True)
class SchemaContract:
    payload: Mapping[str, Any]
    canonical_bytes: bytes
    sha256: str
    schema: Mapping[str, pl.DataType]

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "SchemaContract":
        normalized = json.loads(canonical_json_bytes(payload))
        required = {
            "data_contract_version",
            "table_name",
            "table_version",
            "columns",
            "primary_key",
            "evaluation_id_column",
            "target_rows",
            "row_group_size",
            "compression_level",
            "data_page_size",
            "maximum_rows_per_evaluation",
        }
        missing = required - set(normalized)
        if missing:
            raise StreamingDataError(f"schema contract missing fields: {sorted(missing)}")
        if normalized["data_contract_version"] != DATA_CONTRACT_VERSION:
            raise StreamingDataError("unsupported data contract version")
        _safe_code(normalized["table_name"], "table_name")
        columns = normalized["columns"]
        names = [column["name"] for column in columns]
        if len(names) != len(set(names)) or not names:
            raise StreamingDataError("schema columns must be nonempty and unique")
        try:
            schema = {column["name"]: POLARS_TYPES[column["dtype"]] for column in columns}
        except KeyError as error:
            raise StreamingDataError(f"unsupported Polars dtype: {error.args[0]}") from error
        primary_key = list(normalized["primary_key"])
        if not primary_key or not set(primary_key).issubset(schema):
            raise StreamingDataError("invalid primary key")
        if normalized["evaluation_id_column"] not in schema:
            raise StreamingDataError("evaluation ID column is absent")
        for field in (
            "target_rows",
            "row_group_size",
            "compression_level",
            "data_page_size",
            "maximum_rows_per_evaluation",
        ):
            if int(normalized[field]) <= 0:
                raise StreamingDataError(f"schema field must be positive: {field}")
        minimum_rows = int(normalized.get("minimum_rows_per_evaluation", 0))
        if minimum_rows < 0 or minimum_rows > int(normalized["maximum_rows_per_evaluation"]):
            raise StreamingDataError("invalid per-evaluation row interval")
        canonical = canonical_json_bytes(normalized)
        return cls(
            payload=normalized,
            canonical_bytes=canonical,
            sha256=sha256_bytes(canonical),
            schema=schema,
        )

    @classmethod
    def from_path(cls, path: Path) -> "SchemaContract":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @property
    def table_name(self) -> str:
        return str(self.payload["table_name"])

    @property
    def primary_key(self) -> list[str]:
        return list(self.payload["primary_key"])

    @property
    def evaluation_id_column(self) -> str:
        return str(self.payload["evaluation_id_column"])

    @property
    def step_column(self) -> str | None:
        value = self.payload.get("step_column")
        return None if value is None else str(value)

    def validate_frame(self, frame: pl.DataFrame, evaluation_id: int | None = None) -> None:
        if list(frame.columns) != list(self.schema):
            raise StreamingDataError(f"{self.table_name}: column order/schema drift")
        if dict(frame.schema) != dict(self.schema):
            raise StreamingDataError(f"{self.table_name}: Polars dtype mismatch")
        maximum = int(self.payload["maximum_rows_per_evaluation"])
        minimum = int(self.payload.get("minimum_rows_per_evaluation", 0))
        if evaluation_id is not None and frame.height < minimum:
            raise StreamingDataError(
                f"{self.table_name}: evaluation has fewer than required rows"
            )
        if evaluation_id is not None and frame.height > maximum:
            raise StreamingDataError(
                f"{self.table_name}: evaluation exceeds pre-specified row maximum"
            )
        nullable = {column["name"]: bool(column.get("nullable", False)) for column in self.payload["columns"]}
        finite = {column["name"]: bool(column.get("finite", False)) for column in self.payload["columns"]}
        for name in frame.columns:
            series = frame.get_column(name)
            if not nullable[name] and series.null_count() != 0:
                raise StreamingDataError(f"{self.table_name}.{name}: null is prohibited")
            if finite[name] and series.dtype in {pl.Float32, pl.Float64}:
                if not bool(series.is_finite().all()):
                    raise StreamingDataError(f"{self.table_name}.{name}: NaN/Inf is prohibited")
        if frame.height and bool(frame.select(pl.struct(self.primary_key).is_duplicated().any()).item()):
            raise StreamingDataError(f"{self.table_name}: duplicate primary key")
        if evaluation_id is not None:
            observed = frame.get_column(self.evaluation_id_column).unique().to_list()
            if frame.height and observed != [evaluation_id]:
                raise StreamingDataError(
                    f"{self.table_name}: rows do not belong to evaluation {evaluation_id}"
                )
            keys = frame.select(self.primary_key)
            if not keys.equals(keys.sort(self.primary_key)):
                raise StreamingDataError(
                    f"{self.table_name}: primary key is not in canonical order"
                )
            if self.step_column is not None:
                steps = frame.get_column(self.step_column).to_list()
                if steps != list(range(frame.height)):
                    raise StreamingDataError(
                        f"{self.table_name}: step sequence is not contiguous from zero"
                    )


@dataclass(frozen=True)
class EvaluationRegistry:
    path: Path
    sha256: str
    frame: pl.DataFrame
    assignments: Mapping[int, Mapping[str, Any]]

    @classmethod
    def from_path(cls, path: Path) -> "EvaluationRegistry":
        path = Path(path).resolve()
        if not path.is_file():
            raise StreamingDataError("frozen evaluation_manifest.parquet is absent")
        try:
            frame = pl.read_parquet(path)
        except Exception as error:
            raise StreamingDataError("invalid evaluation_manifest.parquet") from error
        if list(frame.columns) != list(EVALUATION_MANIFEST_SCHEMA):
            raise StreamingDataError("evaluation manifest column order/schema drift")
        if dict(frame.schema) != EVALUATION_MANIFEST_SCHEMA:
            raise StreamingDataError("evaluation manifest dtype mismatch")
        if frame.height == 0 or sum(frame.null_count().row(0)) != 0:
            raise StreamingDataError("evaluation manifest must be nonempty and non-null")
        if bool(frame.get_column("evaluation_id").is_duplicated().any()):
            raise StreamingDataError("duplicate evaluation ID in frozen manifest")
        for column in (
            "split_code",
            "study_code",
            "authority_code",
            "method_code",
            "seed_code",
            "role",
        ):
            for value in frame.get_column(column).unique().to_list():
                _safe_code(str(value), f"evaluation_manifest.{column}")
        if not set(frame.get_column("role").unique().to_list()).issubset(
            {"TRAIN", "DEV", "CAL", "TEST", "STRESS"}
        ):
            raise StreamingDataError("evaluation manifest contains an invalid role")
        digest_columns = ("checkpoint_sha256", "config_sha256", "code_sha256")
        for column in digest_columns:
            if any(
                re.fullmatch(r"[0-9a-f]{64}", str(value)) is None
                for value in frame.get_column(column).unique().to_list()
            ):
                raise StreamingDataError(f"evaluation manifest has invalid {column}")
        for column in (
            "planned_native_max",
            "planned_control_max",
            "planned_primitive_max",
        ):
            if bool((frame.get_column(column) <= 0).any()):
                raise StreamingDataError(f"evaluation manifest has nonpositive {column}")
        assignments = {
            int(row["evaluation_id"]): row
            for row in frame.iter_rows(named=True)
        }
        return cls(path=path, sha256=sha256_file(path), frame=frame, assignments=assignments)

    @property
    def evaluation_ids(self) -> set[int]:
        return set(self.assignments)

    def validate_assignment(
        self,
        evaluation_id: int,
        *,
        stage: str,
        method_code: str,
        seed_code: str,
        tables: Mapping[str, pl.DataFrame] | None = None,
    ) -> Mapping[str, Any]:
        if evaluation_id not in self.assignments:
            raise StreamingDataError(
                f"evaluation ID is not preallocated: {evaluation_id}"
            )
        row = self.assignments[evaluation_id]
        observed = (stage, method_code, seed_code)
        expected = (str(row["role"]), str(row["method_code"]), str(row["seed_code"]))
        if observed != expected:
            raise StreamingDataError(
                f"evaluation assignment mismatch: observed={observed}, expected={expected}"
            )
        if tables is not None:
            for table, frame in tables.items():
                limit_column = ROW_LIMIT_COLUMNS.get(table)
                limit = 1 if limit_column is None else int(row[limit_column])
                if frame.height > limit:
                    raise StreamingDataError(
                        f"{table}: rows exceed evaluation-manifest plan for {evaluation_id}"
                    )
        return row


@dataclass(frozen=True)
class BudgetPolicy:
    run_quota_bytes: int
    minimum_free_bytes: int

    def check(self, run_root: Path, anticipated_bytes: int) -> None:
        usage = shutil.disk_usage(run_root)
        if usage.free - anticipated_bytes < self.minimum_free_bytes:
            raise OSError(errno.ENOSPC, "minimum free-space reserve would be violated")
        current = sum(
            path.stat().st_size
            for path in Path(run_root).rglob("*")
            if path.is_file()
        )
        if current + anticipated_bytes > self.run_quota_bytes:
            raise OSError(errno.ENOSPC, "run storage quota would be exceeded")


class BundleWriter:
    """Synchronously buffer whole evaluations and commit immutable bundles."""

    def __init__(
        self,
        run_root: Path,
        *,
        stage: str,
        method_code: str,
        seed_code: str,
        worker_code: str,
        contracts: Iterable[SchemaContract],
        required_tables: Iterable[str],
        evaluation_manifest: Path,
        soft_buffer_bytes: int = 192 * 1024**2,
        hard_buffer_bytes: int = 256 * 1024**2,
        budget_policy: BudgetPolicy | None = None,
        fault_injector: FaultInjector = _no_fault,
        require_ext4: bool = True,
        resume: bool = False,
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.stage = _safe_code(stage.upper(), "stage")
        if self.stage not in {"TRAIN", "DEV", "CAL", "TEST", "STRESS"}:
            raise StreamingDataError(f"unsupported stage: {self.stage}")
        if resume and self.stage == "TEST":
            raise StreamingDataError("TEST writers are never resumable")
        self.method_code = _safe_code(method_code, "method_code")
        self.seed_code = _safe_code(seed_code, "seed_code")
        self.worker_code = _safe_code(worker_code, "worker_code")
        self.contracts = {contract.table_name: contract for contract in contracts}
        self.required_tables = set(required_tables)
        if not self.required_tables or self.required_tables != set(self.contracts):
            raise StreamingDataError("required table set must exactly match contracts")
        if not 0 < soft_buffer_bytes <= hard_buffer_bytes:
            raise StreamingDataError("invalid buffer limits")
        self.soft_buffer_bytes = int(soft_buffer_bytes)
        self.hard_buffer_bytes = int(hard_buffer_bytes)
        self.budget_policy = budget_policy
        self.fault_injector = fault_injector
        self.sequence = 0
        self.closed = False
        self._frames: dict[str, list[pl.DataFrame]] = {
            table: [] for table in self.required_tables
        }
        self._rows = {table: 0 for table in self.required_tables}
        self._buffer_bytes = 0
        self._evaluation_ids: list[int] = []
        self._seen_evaluations: set[int] = set()
        self.run_root.mkdir(parents=True, exist_ok=True)
        if require_ext4:
            filesystem = _filesystem_type(self.run_root)
            if filesystem is not None and filesystem != "ext4":
                raise StreamingDataError(
                    f"active writer root must be WSL ext4, observed {filesystem}"
                )
        for directory in ("schemas", "data", "commits", "staging", "manifests"):
            (self.run_root / directory).mkdir(exist_ok=True)
        manifest_path = Path(evaluation_manifest).resolve()
        canonical_manifest = (self.run_root / "evaluation_manifest.parquet").resolve()
        if manifest_path != canonical_manifest:
            raise StreamingDataError(
                "writer must use run_root/evaluation_manifest.parquet"
            )
        self.evaluation_registry = EvaluationRegistry.from_path(manifest_path)
        self._install_contracts()
        self._initialize_history(resume=resume)

    def _install_contracts(self) -> None:
        for table, contract in self.contracts.items():
            path = self.run_root / "schemas" / f"{table}.schema.json"
            if path.exists():
                if path.read_bytes() != contract.canonical_bytes:
                    raise StreamingDataError(f"installed schema differs: {table}")
                continue
            temporary = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
            try:
                with temporary.open("xb") as stream:
                    stream.write(contract.canonical_bytes)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != contract.canonical_bytes:
                    raise StreamingDataError(f"concurrent schema conflict: {table}")
            finally:
                temporary.unlink(missing_ok=True)
        _fsync_directory(self.run_root / "schemas")

    def _initialize_history(self, *, resume: bool) -> None:
        commits = _read_commits(self.run_root)
        if not commits:
            return
        verify_committed_run(self.run_root, require_complete=False)
        for commit in commits:
            self._seen_evaluations.update(
                int(value) for value in commit["evaluation_ids"]
            )
        worker_commits = [
            commit
            for commit in commits
            if str(commit.get("worker_code")) == self.worker_code
        ]
        if not worker_commits:
            return
        if not resume:
            raise StreamingDataError(
                f"worker {self.worker_code} already has committed bundles; explicit resume required"
            )
        for commit in worker_commits:
            identity = (
                str(commit.get("stage")),
                str(commit.get("method_code")),
                str(commit.get("seed_code")),
            )
            if identity != (self.stage, self.method_code, self.seed_code):
                raise StreamingDataError("resume identity differs from committed worker")
        sequences = sorted(int(commit["sequence"]) for commit in worker_commits)
        if sequences != list(range(len(sequences))):
            raise StreamingDataError("worker bundle sequence is not contiguous from zero")
        self.sequence = len(sequences)

    def append_evaluation(
        self, evaluation_id: int, tables: Mapping[str, pl.DataFrame]
    ) -> list[str]:
        if self.closed:
            raise StreamingDataError("writer is closed")
        evaluation_id = int(evaluation_id)
        if evaluation_id < 0 or evaluation_id in self._seen_evaluations:
            raise StreamingDataError(f"duplicate/invalid evaluation ID: {evaluation_id}")
        if set(tables) != self.required_tables:
            missing = sorted(self.required_tables - set(tables))
            extra = sorted(set(tables) - self.required_tables)
            raise StreamingDataError(f"table set mismatch: missing={missing}, extra={extra}")
        self.evaluation_registry.validate_assignment(
            evaluation_id,
            stage=self.stage,
            method_code=self.method_code,
            seed_code=self.seed_code,
            tables=tables,
        )
        incoming_bytes = 0
        for table, frame in tables.items():
            self.contracts[table].validate_frame(frame, evaluation_id)
            incoming_bytes += int(frame.estimated_size())
        if incoming_bytes > self.hard_buffer_bytes:
            raise StreamingDataError("one evaluation exceeds the hard trace-buffer limit")
        committed: list[str] = []
        if self._evaluation_ids and self._buffer_bytes + incoming_bytes > self.soft_buffer_bytes:
            committed.append(self.flush())
        for table, frame in tables.items():
            self._frames[table].append(frame)
            self._rows[table] += frame.height
        self._buffer_bytes += incoming_bytes
        self._evaluation_ids.append(evaluation_id)
        self._seen_evaluations.add(evaluation_id)
        if self._buffer_bytes > self.hard_buffer_bytes:
            raise StreamingDataError("trace buffer exceeded its hard limit")
        if any(
            self._rows[table] >= int(self.contracts[table].payload["target_rows"])
            for table in self.required_tables
        ):
            committed.append(self.flush())
        return committed

    def close(self) -> list[str]:
        if self.closed:
            return []
        committed = [self.flush()] if self._evaluation_ids else []
        self.closed = True
        return committed

    def flush_if_pending(self) -> list[str]:
        """Durably commit the current whole-evaluation buffer, if nonempty."""

        if self.closed:
            raise StreamingDataError("writer is closed")
        return [self.flush()] if self._evaluation_ids else []

    def _bundle_id(self) -> str:
        return f"{self.worker_code}-{self.sequence:06d}"

    def flush(self) -> str:
        if not self._evaluation_ids:
            raise StreamingDataError("cannot flush an empty bundle")
        bundle_id = self._bundle_id()
        commit_path = self.run_root / "commits" / f"{bundle_id}.json"
        staging = self.run_root / "staging" / bundle_id
        if commit_path.exists() or staging.exists():
            raise StreamingDataError(f"bundle sequence is already occupied: {bundle_id}")
        frames = {
            table: pl.concat(parts, how="vertical", rechunk=False)
            for table, parts in self._frames.items()
        }
        anticipated = max(self._buffer_bytes, 1)
        if self.budget_policy is not None:
            self.budget_policy.check(self.run_root, anticipated)
        staging.mkdir(parents=False, exist_ok=False)
        file_plan: dict[str, tuple[Path, Path]] = {}
        for table in sorted(self.required_tables):
            relative = Path("data") / self.stage / self.method_code / self.seed_code / table / f"part-{bundle_id}.parquet"
            final_path = self.run_root / relative
            if final_path.exists():
                raise StreamingDataError(f"immutable data file already exists: {relative}")
            final_path.parent.mkdir(parents=True, exist_ok=True)
            file_plan[table] = (staging / f"{table}.parquet.tmp", final_path)
        transaction = {
            "bundle_id": bundle_id,
            "stage": self.stage,
            "final_paths": {
                table: final.relative_to(self.run_root).as_posix()
                for table, (_temporary, final) in file_plan.items()
            },
        }
        transaction_path = staging / "transaction.json"
        transaction_path.write_bytes(canonical_json_bytes(transaction))
        _fsync_file(transaction_path)
        _fsync_directory(staging)

        file_rows: list[dict[str, Any]] = []
        for table in sorted(self.required_tables):
            contract = self.contracts[table]
            temporary, final_path = file_plan[table]
            context = {
                "bundle_id": bundle_id,
                "table": table,
                "temporary_path": temporary,
                "final_path": final_path,
            }
            self.fault_injector(FaultPoint.BEFORE_TABLE_WRITE, context)
            frame = frames[table]
            frame.write_parquet(
                temporary,
                compression="zstd",
                compression_level=int(contract.payload["compression_level"]),
                statistics=True,
                row_group_size=int(contract.payload["row_group_size"]),
                data_page_size=int(contract.payload["data_page_size"]),
                use_pyarrow=False,
                metadata={
                    "data_contract_version": DATA_CONTRACT_VERSION,
                    "table_version": str(contract.payload["table_version"]),
                    "schema_sha256": contract.sha256,
                    "bundle_id": bundle_id,
                    "table_name": table,
                },
            )
            self.fault_injector(FaultPoint.AFTER_TABLE_WRITE, context)
            _validate_parquet_file(temporary, contract, bundle_id)
            _fsync_file(temporary)
            evaluations = frame.get_column(contract.evaluation_id_column)
            step = (
                frame.get_column(contract.step_column)
                if contract.step_column is not None and frame.height
                else None
            )
            file_rows.append(
                {
                    "relative_path": final_path.relative_to(self.run_root).as_posix(),
                    "table": table,
                    "bundle_id": bundle_id,
                    "bytes": temporary.stat().st_size,
                    "rows": frame.height,
                    "min_evaluation_id": int(evaluations.min()) if frame.height else None,
                    "max_evaluation_id": int(evaluations.max()) if frame.height else None,
                    "min_step": int(step.min()) if step is not None else None,
                    "max_step": int(step.max()) if step is not None else None,
                    "schema_sha256": contract.sha256,
                    "file_sha256": sha256_file(temporary),
                }
            )

        for index, table in enumerate(sorted(self.required_tables)):
            temporary, final_path = file_plan[table]
            os.replace(temporary, final_path)
            _fsync_directory(final_path.parent)
            if index == 0:
                self.fault_injector(
                    FaultPoint.AFTER_PARTIAL_MOVE,
                    {"bundle_id": bundle_id, "moved_path": final_path},
                )
        self.fault_injector(
            FaultPoint.BEFORE_COMMIT,
            {"bundle_id": bundle_id, "staging_path": staging},
        )
        commit = {
            "data_contract_version": DATA_CONTRACT_VERSION,
            "bundle_id": bundle_id,
            "stage": self.stage,
            "method_code": self.method_code,
            "seed_code": self.seed_code,
            "worker_code": self.worker_code,
            "sequence": self.sequence,
            "evaluation_manifest_sha256": self.evaluation_registry.sha256,
            "evaluation_ids": list(self._evaluation_ids),
            "files": file_rows,
        }
        _atomic_exclusive_json(commit_path, commit)
        shutil.rmtree(staging)
        _fsync_directory(self.run_root / "staging")
        self.sequence += 1
        self._frames = {table: [] for table in self.required_tables}
        self._rows = {table: 0 for table in self.required_tables}
        self._buffer_bytes = 0
        self._evaluation_ids = []
        return bundle_id


def _validate_parquet_file(
    path: Path,
    contract: SchemaContract,
    bundle_id: str,
    *,
    expected_evaluation_ids: set[int] | None = None,
    registry: EvaluationRegistry | None = None,
) -> dict[str, Any]:
    try:
        schema = pl.read_parquet_schema(path)
        metadata = pl.read_parquet_metadata(path)
        frame = pl.read_parquet(path)
    except Exception as error:
        raise StreamingDataError(f"invalid Parquet/footer: {path.name}") from error
    if dict(schema) != dict(contract.schema):
        raise StreamingDataError(f"Parquet schema mismatch: {path.name}")
    required_metadata = {
        "data_contract_version": DATA_CONTRACT_VERSION,
        "table_version": str(contract.payload["table_version"]),
        "schema_sha256": contract.sha256,
        "bundle_id": bundle_id,
        "table_name": contract.table_name,
    }
    for key, expected in required_metadata.items():
        if metadata.get(key) != expected:
            raise StreamingDataError(f"Parquet metadata mismatch: {path.name}:{key}")
    contract.validate_frame(frame)
    evaluation_column = contract.evaluation_id_column
    evaluation_ids = {
        int(value) for value in frame.get_column(evaluation_column).unique().to_list()
    }
    if expected_evaluation_ids is not None and evaluation_ids != expected_evaluation_ids:
        raise StreamingDataError(
            f"Parquet evaluation set mismatch: {path.name}"
        )
    for evaluation_id in sorted(evaluation_ids):
        subset = frame.filter(pl.col(evaluation_column) == evaluation_id)
        contract.validate_frame(subset, evaluation_id)
        if registry is not None:
            registry.validate_assignment(
                evaluation_id,
                stage=str(registry.assignments[evaluation_id]["role"]),
                method_code=str(registry.assignments[evaluation_id]["method_code"]),
                seed_code=str(registry.assignments[evaluation_id]["seed_code"]),
                tables={contract.table_name: subset},
            )
    step = (
        frame.get_column(contract.step_column)
        if contract.step_column is not None and frame.height
        else None
    )
    evaluations = frame.get_column(evaluation_column)
    return {
        "rows": frame.height,
        "min_evaluation_id": int(evaluations.min()) if frame.height else None,
        "max_evaluation_id": int(evaluations.max()) if frame.height else None,
        "min_step": int(step.min()) if step is not None else None,
        "max_step": int(step.max()) if step is not None else None,
    }


def _load_contracts(run_root: Path) -> dict[str, SchemaContract]:
    schema_root = Path(run_root) / "schemas"
    unexpected = [
        path.name
        for path in schema_root.iterdir()
        if not path.is_file() or not path.name.endswith(".schema.json")
    ]
    if unexpected:
        raise StreamingDataError(f"unexpected schema-directory entries: {sorted(unexpected)}")
    paths = sorted(schema_root.glob("*.schema.json"))
    if not paths:
        raise StreamingDataError("no installed schema contracts")
    contracts = {SchemaContract.from_path(path).table_name: SchemaContract.from_path(path) for path in paths}
    if len(contracts) != len(paths):
        raise StreamingDataError("duplicate installed table contract")
    return contracts


def _read_commits(run_root: Path) -> list[dict[str, Any]]:
    unexpected = [
        path.name
        for path in (Path(run_root) / "commits").iterdir()
        if path.is_file() and (path.suffix != ".json" or path.name.startswith("."))
    ]
    if unexpected:
        raise StreamingDataError(f"unexpected commit-directory files: {sorted(unexpected)}")
    commits = []
    for path in sorted((Path(run_root) / "commits").glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as error:
            raise StreamingDataError(f"invalid commit marker: {path.name}") from error
        if payload.get("bundle_id") != path.stem:
            raise StreamingDataError(f"commit marker/bundle mismatch: {path.name}")
        commits.append(payload)
    return commits


def verify_committed_run(
    run_root: Path,
    *,
    require_manifest: bool = False,
    require_complete: bool = True,
) -> dict[str, int]:
    run_root = Path(run_root).resolve()
    contracts = _load_contracts(run_root)
    registry = EvaluationRegistry.from_path(run_root / "evaluation_manifest.parquet")
    staging = run_root / "staging"
    leftovers = [path for path in staging.rglob("*") if path.is_file() or path.is_dir()]
    if leftovers:
        raise StreamingDataError("staging/orphan transaction blocks finalization")
    commits = _read_commits(run_root)
    if not commits:
        raise StreamingDataError("no committed bundles")
    committed_paths: set[str] = set()
    observed_evaluations: set[int] = set()
    rows = 0
    for commit in commits:
        required_commit_fields = {
            "bundle_id",
            "stage",
            "method_code",
            "seed_code",
            "worker_code",
            "sequence",
            "evaluation_manifest_sha256",
            "evaluation_ids",
            "files",
        }
        if not required_commit_fields.issubset(commit):
            raise StreamingDataError("commit marker lacks required fields")
        bundle_id = str(commit["bundle_id"])
        stage = _safe_code(str(commit["stage"]), "commit.stage")
        method_code = _safe_code(str(commit["method_code"]), "commit.method_code")
        seed_code = _safe_code(str(commit["seed_code"]), "commit.seed_code")
        worker_code = _safe_code(str(commit["worker_code"]), "commit.worker_code")
        sequence = int(commit["sequence"])
        if bundle_id != f"{worker_code}-{sequence:06d}":
            raise StreamingDataError(f"noncanonical bundle identity: {bundle_id}")
        if commit.get("evaluation_manifest_sha256") != registry.sha256:
            raise StreamingDataError(
                f"evaluation-manifest digest mismatch: {bundle_id}"
            )
        evaluation_ids = [int(value) for value in commit["evaluation_ids"]]
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise StreamingDataError(f"duplicate evaluation within bundle: {bundle_id}")
        overlap = observed_evaluations.intersection(evaluation_ids)
        if overlap:
            raise StreamingDataError(f"evaluation belongs to multiple bundles: {sorted(overlap)}")
        observed_evaluations.update(evaluation_ids)
        for evaluation_id in evaluation_ids:
            registry.validate_assignment(
                evaluation_id,
                stage=stage,
                method_code=method_code,
                seed_code=seed_code,
            )
        file_tables = {str(row["table"]) for row in commit["files"]}
        if file_tables != set(contracts) or len(commit["files"]) != len(contracts):
            raise StreamingDataError(f"commit table set mismatch: {bundle_id}")
        for row in commit["files"]:
            relative = str(row["relative_path"])
            table = str(row["table"])
            expected_relative = (
                Path("data")
                / stage
                / method_code
                / seed_code
                / table
                / f"part-{bundle_id}.parquet"
            ).as_posix()
            if relative != expected_relative or str(row.get("bundle_id")) != bundle_id:
                raise StreamingDataError(f"noncanonical committed path: {relative}")
            if relative in committed_paths:
                raise StreamingDataError(f"file belongs to multiple commits: {relative}")
            committed_paths.add(relative)
            path = (run_root / relative).resolve()
            if run_root not in path.parents or not path.is_file():
                raise StreamingDataError(f"missing/unsafe committed file: {relative}")
            if path.stat().st_size != int(row["bytes"]):
                raise StreamingDataError(f"byte-size mismatch: {relative}")
            if sha256_file(path) != row["file_sha256"]:
                raise StreamingDataError(f"SHA-256 mismatch: {relative}")
            contract = contracts[table]
            if contract.sha256 != row["schema_sha256"]:
                raise StreamingDataError(f"schema digest mismatch: {relative}")
            validation = _validate_parquet_file(
                path,
                contract,
                bundle_id,
                expected_evaluation_ids=set(evaluation_ids),
                registry=registry,
            )
            if validation["rows"] != int(row["rows"]):
                raise StreamingDataError(f"row-count mismatch: {relative}")
            for field in (
                "min_evaluation_id",
                "max_evaluation_id",
                "min_step",
                "max_step",
            ):
                observed = validation[field]
                recorded = row.get(field)
                recorded = None if recorded is None else int(recorded)
                if observed != recorded:
                    raise StreamingDataError(
                        f"commit extrema mismatch: {relative}:{field}"
                    )
            rows += int(row["rows"])
    actual_data = {
        path.relative_to(run_root).as_posix()
        for path in (run_root / "data").rglob("*")
        if path.is_file()
    }
    if actual_data != committed_paths:
        missing = sorted(committed_paths - actual_data)
        extra = sorted(actual_data - committed_paths)
        raise StreamingDataError(f"data file set mismatch: missing={missing}, extra={extra}")
    if require_complete and observed_evaluations != registry.evaluation_ids:
        missing = sorted(registry.evaluation_ids - observed_evaluations)
        extra = sorted(observed_evaluations - registry.evaluation_ids)
        raise StreamingDataError(
            f"evaluation set differs from frozen manifest: missing={missing}, extra={extra}"
        )
    manifest = run_root / "manifests" / "FILE_MANIFEST.csv"
    if require_manifest and not manifest.is_file():
        raise StreamingDataError("frozen FILE_MANIFEST.csv is absent")
    if manifest.is_file():
        with manifest.open("r", encoding="utf-8", newline="") as stream:
            manifest_rows = list(csv.DictReader(stream))
        expected = sorted(
            (row for commit in commits for row in commit["files"]),
            key=lambda row: row["relative_path"],
        )
        comparable = [
            {column: "" if row.get(column) is None else str(row.get(column)) for column in MANIFEST_COLUMNS}
            for row in expected
        ]
        if manifest_rows != comparable:
            raise StreamingDataError("FILE_MANIFEST.csv differs from commit markers")
        digest_path = run_root / "manifests" / "FILE_MANIFEST.sha256"
        if not digest_path.is_file() or digest_path.read_text(encoding="utf-8").split()[0] != sha256_file(manifest):
            raise StreamingDataError("FILE_MANIFEST.sha256 mismatch")
    return {
        "bundles": len(commits),
        "files": len(committed_paths),
        "evaluations": len(observed_evaluations),
        "rows": rows,
    }


def finalize_file_manifest(
    run_root: Path, *, require_complete: bool = True
) -> dict[str, Any]:
    run_root = Path(run_root).resolve()
    manifest = run_root / "manifests" / "FILE_MANIFEST.csv"
    digest_path = run_root / "manifests" / "FILE_MANIFEST.sha256"
    if manifest.exists() != digest_path.exists():
        raise StreamingDataError("partial immutable file manifest blocks recovery")
    summary = verify_committed_run(run_root, require_complete=require_complete)
    commits = _read_commits(run_root)
    rows = sorted(
        (row for commit in commits for row in commit["files"]),
        key=lambda row: row["relative_path"],
    )
    if manifest.exists() and digest_path.exists():
        verified = verify_committed_run(
            run_root, require_manifest=True, require_complete=require_complete
        )
        return {**verified, "manifest_sha256": sha256_file(manifest)}
    temporary = manifest.parent / f".{manifest.name}.{uuid.uuid4().hex}.tmp"
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in MANIFEST_COLUMNS})
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary, manifest)
    temporary.unlink()
    _fsync_directory(manifest.parent)
    digest = sha256_file(manifest)
    temporary_digest = digest_path.parent / f".{digest_path.name}.{uuid.uuid4().hex}.tmp"
    with temporary_digest.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(f"{digest}  {manifest.name}\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.link(temporary_digest, digest_path)
    temporary_digest.unlink()
    _fsync_directory(digest_path.parent)
    verify_committed_run(
        run_root, require_manifest=True, require_complete=require_complete
    )
    return {**summary, "manifest_sha256": digest}


def recover_uncommitted_bundle(
    run_root: Path, bundle_id: str, *, stage: str
) -> dict[str, Any]:
    """Remove an uncommitted DEV/CAL/TRAIN/STRESS transaction for full replay."""
    run_root = Path(run_root).resolve()
    bundle_id = _safe_code(bundle_id, "bundle_id")
    stage = stage.upper()
    if stage == "TEST":
        raise StreamingDataError("TEST transactions are never resumable")
    commit = run_root / "commits" / f"{bundle_id}.json"
    staging = run_root / "staging" / bundle_id
    if commit.exists():
        if staging.exists():
            shutil.rmtree(staging)
        return {"bundle_id": bundle_id, "status": "already_committed"}
    removed: list[str] = []
    if staging.exists():
        transaction = staging / "transaction.json"
        if transaction.is_file():
            payload = json.loads(transaction.read_text(encoding="utf-8"))
            if payload.get("bundle_id") != bundle_id:
                raise StreamingDataError("staging transaction bundle mismatch")
            for relative in payload.get("final_paths", {}).values():
                path = (run_root / relative).resolve()
                if run_root not in path.parents:
                    raise StreamingDataError("recovery path escapes run root")
                if path.is_file():
                    path.unlink()
                    removed.append(path.relative_to(run_root).as_posix())
        shutil.rmtree(staging)
    return {"bundle_id": bundle_id, "status": "recovered", "removed": sorted(removed)}
