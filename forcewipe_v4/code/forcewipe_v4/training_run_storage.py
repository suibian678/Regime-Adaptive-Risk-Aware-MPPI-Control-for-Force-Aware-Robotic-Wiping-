"""Transactional journal for formal V4/V5 policy-training runs.

An update is considered committed only after its immutable checkpoint and all
episode-evidence bundles exist and one final update marker has been created.
The marker is written last.  Committed evidence without a marker is verified
against a deterministic replay instead of being silently duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping
import uuid

import polars as pl

from .streaming import (
    BudgetPolicy,
    BundleWriter,
    EvaluationRegistry,
    StreamingDataError,
    canonical_json_bytes,
    sha256_file,
    verify_committed_run,
)
from .training_evidence import (
    TrainingEpisodeEvidence,
    evidence_frames,
    training_evidence_contracts,
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
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


class TrainingEvidenceBuffer:
    def __init__(self) -> None:
        self._items: list[TrainingEpisodeEvidence] = []
        self._ids: set[int] = set()

    def append(self, evidence: TrainingEpisodeEvidence) -> None:
        evidence.validate()
        if evidence.evaluation_id in self._ids:
            raise StreamingDataError("duplicate evaluation in pending training evidence")
        self._ids.add(evidence.evaluation_id)
        self._items.append(evidence)

    def drain(self) -> tuple[TrainingEpisodeEvidence, ...]:
        items = tuple(self._items)
        self._items.clear()
        self._ids.clear()
        return items

    def __len__(self) -> int:
        return len(self._items)

    @property
    def evaluation_ids(self) -> tuple[int, ...]:
        return tuple(item.evaluation_id for item in self._items)


@dataclass(frozen=True)
class UpdateCommit:
    update_index: int
    evaluation_ids: tuple[int, ...]
    bundle_ids: tuple[str, ...]
    checkpoint_relative_path: str
    checkpoint_sha256: str
    payload: Mapping[str, Any]


@dataclass(frozen=True)
class TerminalEvidenceCommit:
    status: str
    evaluation_ids: tuple[int, ...]
    bundle_ids: tuple[str, ...]
    checkpoint_relative_path: str
    checkpoint_sha256: str
    payload: Mapping[str, Any]


class TrainingRunJournal:
    """Evidence writer plus last-marker update transaction."""

    def __init__(
        self,
        run_root: Path,
        *,
        method_code: str,
        seed_code: str,
        resume: bool,
        require_ext4: bool = True,
        budget_policy: BudgetPolicy | None = None,
    ) -> None:
        self.run_root = Path(run_root).resolve()
        self.run_root.mkdir(parents=True, exist_ok=True)
        for name in ("checkpoints", "update_commits", "terminal_commits"):
            (self.run_root / name).mkdir(exist_ok=True)
        self.contracts = training_evidence_contracts()
        self.registry = EvaluationRegistry.from_path(
            self.run_root / "evaluation_manifest.parquet"
        )
        self.writer = BundleWriter(
            self.run_root,
            stage="TRAIN",
            method_code=method_code,
            seed_code=seed_code,
            worker_code="ppo",
            contracts=self.contracts.values(),
            required_tables=self.contracts,
            evaluation_manifest=self.run_root / "evaluation_manifest.parquet",
            budget_policy=budget_policy,
            require_ext4=require_ext4,
            resume=resume,
        )
        if any((self.run_root / "commits").glob("*.json")):
            verify_committed_run(self.run_root, require_complete=False)
        self._committed_frames = self._index_committed_frames()
        self._bundle_evaluations = self._index_bundle_evaluations()
        self.update_commits = self._load_update_commits()
        self.terminal_commit = self._load_terminal_commit()

    def _index_bundle_evaluations(self) -> dict[str, set[int]]:
        indexed: dict[str, set[int]] = {}
        for marker in sorted((self.run_root / "commits").glob("*.json")):
            payload = json.loads(marker.read_text(encoding="utf-8"))
            bundle_id = str(payload["bundle_id"])
            if bundle_id in indexed:
                raise StreamingDataError("duplicate episode-evidence bundle identity")
            indexed[bundle_id] = {
                int(value) for value in payload.get("evaluation_ids", ())
            }
        return indexed

    def _verify_marker_evidence(
        self, *, evaluation_ids: tuple[int, ...], bundle_ids: tuple[str, ...]
    ) -> None:
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise StreamingDataError("training marker repeats an evaluation ID")
        if not set(evaluation_ids).issubset(self._committed_frames):
            raise StreamingDataError("training marker episode evidence is absent")
        if len(bundle_ids) != len(set(bundle_ids)):
            raise StreamingDataError("training marker repeats an evidence bundle")
        missing = set(bundle_ids) - set(self._bundle_evaluations)
        if missing:
            raise StreamingDataError(
                f"training marker references missing bundle commits: {sorted(missing)}"
            )
        if bundle_ids:
            bundled = set().union(
                *(self._bundle_evaluations[bundle_id] for bundle_id in bundle_ids)
            )
            if bundled != set(evaluation_ids):
                raise StreamingDataError(
                    "training marker evaluation IDs differ from its bundle commits"
                )

    def _load_terminal_commit(self) -> dict[str, Any] | None:
        markers = sorted((self.run_root / "terminal_commits").glob("*.json"))
        if len(markers) > 1:
            raise StreamingDataError("multiple immutable training terminal markers exist")
        if not markers:
            return None
        payload = json.loads(markers[0].read_text(encoding="utf-8"))
        if payload.get("format") not in {
            "forcewipe_v4_training_terminal_commit_v1",
            "forcewipe_v5_training_terminal_commit_v1",
        }:
            raise StreamingDataError("unsupported training terminal marker")
        checkpoint = self.run_root / str(payload["checkpoint_relative_path"])
        if not checkpoint.is_file() or sha256_file(checkpoint) != payload["checkpoint_sha256"]:
            raise StreamingDataError("training terminal checkpoint hash mismatch")
        ids = tuple(int(value) for value in payload.get("evaluation_ids", ()))
        bundles = tuple(str(value) for value in payload.get("bundle_ids", ()))
        self._verify_marker_evidence(evaluation_ids=ids, bundle_ids=bundles)
        return payload

    def _load_update_commits(self) -> tuple[dict[str, Any], ...]:
        commits = []
        observed_evaluations: set[int] = set()
        observed_bundles: set[str] = set()
        for path in sorted((self.run_root / "update_commits").glob("update-*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") not in {
                "forcewipe_v4_training_update_commit_v1",
                "forcewipe_v5_training_update_commit_v1",
            }:
                raise StreamingDataError("unsupported training update marker")
            if int(payload.get("update_index", -1)) != len(commits) + 1:
                raise StreamingDataError("training update markers are not contiguous")
            checkpoint = self.run_root / str(payload["checkpoint_relative_path"])
            if not checkpoint.is_file() or sha256_file(checkpoint) != payload["checkpoint_sha256"]:
                raise StreamingDataError("training update checkpoint hash mismatch")
            ids = tuple(int(value) for value in payload.get("evaluation_ids", ()))
            bundles = tuple(str(value) for value in payload.get("bundle_ids", ()))
            if not ids:
                raise StreamingDataError("training update marker has no episode evidence")
            self._verify_marker_evidence(evaluation_ids=ids, bundle_ids=bundles)
            if observed_evaluations.intersection(ids):
                raise StreamingDataError("episode evidence is referenced by multiple updates")
            if observed_bundles.intersection(bundles):
                raise StreamingDataError("bundle commit is referenced by multiple updates")
            observed_evaluations.update(ids)
            observed_bundles.update(bundles)
            commits.append(payload)
        return tuple(commits)

    def _index_committed_frames(self) -> dict[int, dict[str, pl.DataFrame]]:
        indexed: dict[int, dict[str, pl.DataFrame]] = {}
        commits_root = self.run_root / "commits"
        if not commits_root.is_dir():
            return indexed
        for marker in sorted(commits_root.glob("*.json")):
            payload = json.loads(marker.read_text(encoding="utf-8"))
            for file_row in payload["files"]:
                table = str(file_row["table"])
                frame = pl.read_parquet(self.run_root / str(file_row["relative_path"]))
                for evaluation_id in payload["evaluation_ids"]:
                    subset = frame.filter(pl.col("evaluation_id") == int(evaluation_id))
                    indexed.setdefault(int(evaluation_id), {})[table] = subset
        return indexed

    @property
    def committed_update_count(self) -> int:
        return len(self.update_commits)

    @property
    def committed_evaluation_ids(self) -> set[int]:
        return set(self._committed_frames)

    def append_or_verify_evidence(
        self, evidences: Iterable[TrainingEpisodeEvidence]
    ) -> tuple[tuple[int, ...], tuple[str, ...]]:
        evaluation_ids: list[int] = []
        bundle_ids: list[str] = []
        for evidence in evidences:
            frames = evidence_frames(evidence, self.contracts)
            evaluation_id = int(evidence.evaluation_id)
            evaluation_ids.append(evaluation_id)
            existing = self._committed_frames.get(evaluation_id)
            if existing is not None:
                if set(existing) != set(frames) or any(
                    not existing[name].equals(frame)
                    for name, frame in frames.items()
                ):
                    raise StreamingDataError(
                        "deterministic replay differs from orphan committed evidence"
                    )
                continue
            bundle_ids.extend(self.writer.append_evaluation(evaluation_id, frames))
        bundle_ids.extend(self.writer.flush_if_pending())
        self._committed_frames = self._index_committed_frames()
        self._bundle_evaluations = self._index_bundle_evaluations()
        return tuple(evaluation_ids), tuple(bundle_ids)

    def commit_update(
        self,
        *,
        update_index: int,
        checkpoint_path: Path,
        evaluation_ids: Iterable[int],
        bundle_ids: Iterable[str],
        payload: Mapping[str, Any],
    ) -> UpdateCommit:
        if int(update_index) != self.committed_update_count + 1:
            raise StreamingDataError("update index is not the next immutable sequence")
        checkpoint = Path(checkpoint_path).resolve()
        if self.run_root not in checkpoint.parents or not checkpoint.is_file():
            raise StreamingDataError("checkpoint is absent or outside the run root")
        ids = tuple(int(value) for value in evaluation_ids)
        if len(ids) != len(set(ids)) or not ids:
            raise StreamingDataError("update evidence IDs must be unique and nonempty")
        if not set(ids).issubset(self._committed_frames):
            raise StreamingDataError("update marker precedes durable episode evidence")
        relative = checkpoint.relative_to(self.run_root).as_posix()
        marker_payload = {
            "format": "forcewipe_v5_training_update_commit_v1",
            "update_index": int(update_index),
            "evaluation_ids": list(ids),
            "bundle_ids": list(bundle_ids),
            "checkpoint_relative_path": relative,
            "checkpoint_sha256": sha256_file(checkpoint),
            "payload": dict(payload),
        }
        marker = self.run_root / "update_commits" / f"update-{update_index:06d}.json"
        atomic_exclusive_json(marker, marker_payload)
        self.update_commits = self._load_update_commits()
        return UpdateCommit(
            update_index=int(update_index),
            evaluation_ids=ids,
            bundle_ids=tuple(str(value) for value in bundle_ids),
            checkpoint_relative_path=relative,
            checkpoint_sha256=marker_payload["checkpoint_sha256"],
            payload=dict(payload),
        )

    def commit_terminal_evidence(
        self,
        *,
        status: str,
        checkpoint_path: Path,
        evaluation_ids: Iterable[int],
        bundle_ids: Iterable[str],
        payload: Mapping[str, Any],
    ) -> TerminalEvidenceCommit:
        """Commit a scientific terminal after its evidence and checkpoint exist."""

        if status != "completed_scientific_fail":
            raise StreamingDataError("terminal evidence commit requires scientific-fail status")
        if self.terminal_commit is not None:
            raise StreamingDataError("immutable training terminal marker already exists")
        checkpoint = Path(checkpoint_path).resolve()
        if self.run_root not in checkpoint.parents or not checkpoint.is_file():
            raise StreamingDataError("terminal checkpoint is absent or outside the run root")
        ids = tuple(int(value) for value in evaluation_ids)
        if len(ids) != len(set(ids)):
            raise StreamingDataError("terminal evidence IDs must be unique")
        if not set(ids).issubset(self._committed_frames):
            raise StreamingDataError("terminal marker precedes durable episode evidence")
        relative = checkpoint.relative_to(self.run_root).as_posix()
        marker_payload = {
            "format": "forcewipe_v5_training_terminal_commit_v1",
            "status": status,
            "evaluation_ids": list(ids),
            "bundle_ids": [str(value) for value in bundle_ids],
            "checkpoint_relative_path": relative,
            "checkpoint_sha256": sha256_file(checkpoint),
            "payload": dict(payload),
        }
        marker = self.run_root / "terminal_commits" / "scientific-fail.json"
        atomic_exclusive_json(marker, marker_payload)
        self.terminal_commit = self._load_terminal_commit()
        return TerminalEvidenceCommit(
            status=status,
            evaluation_ids=ids,
            bundle_ids=tuple(str(value) for value in bundle_ids),
            checkpoint_relative_path=relative,
            checkpoint_sha256=marker_payload["checkpoint_sha256"],
            payload=dict(payload),
        )

    def close(self) -> None:
        self.writer.close()


def commit_scientific_terminal_transaction(
    *,
    journal: TrainingRunJournal,
    evidence_buffer: TrainingEvidenceBuffer,
    checkpoint_path: Path,
    checkpoint_writer: Callable[[Path], None],
    payload: Mapping[str, Any],
) -> TerminalEvidenceCommit:
    """Durably bind pending scientific evidence, boundary checkpoint, and marker.

    The episode evidence is committed first, the immutable checkpoint second,
    and the terminal marker last.  The caller may write RUN_STATE only after
    this function returns.
    """

    evidences = evidence_buffer.drain()
    evaluation_ids, bundle_ids = journal.append_or_verify_evidence(evidences)
    checkpoint = Path(checkpoint_path)
    checkpoint_writer(checkpoint)
    return journal.commit_terminal_evidence(
        status="completed_scientific_fail",
        checkpoint_path=checkpoint,
        evaluation_ids=evaluation_ids,
        bundle_ids=bundle_ids,
        payload=payload,
    )
