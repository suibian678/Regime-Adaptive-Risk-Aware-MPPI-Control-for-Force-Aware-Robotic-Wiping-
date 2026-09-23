"""Fail-closed definition and manifest readers for V4 training."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .information_flow import ObservationMode
from .ppo_training import PPOConfig
from .decision_protocol import SharedRewardConfig
from .sapien_training_backend import SapienTrainingBackendConfig
from .scenarios import ScenarioSpec
from .information_flow import SimulatedVisionConfig
from .simulated_vision import ResidualStripCameraConfig


class TrainingProtocolError(RuntimeError):
    pass


MATCHED_MODES = (
    ObservationMode.VISION_ONLY.value,
    ObservationMode.FORCE_ONLY.value,
    ObservationMode.FUSION.value,
)
ALLOWED_ROLES = {"TRAIN", "DEV", "CAL"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TrainingProtocolError("protocol JSON root must be an object")
    return payload


def _exact_dataclass_payload(payload: dict[str, Any], cls) -> None:
    expected = {field.name for field in fields(cls)}
    if set(payload) != expected:
        raise TrainingProtocolError(
            f"{cls.__name__} fields differ: expected {sorted(expected)}"
        )
    cls(**payload).validate()


def _validate_training_semantics(payload: dict[str, Any]) -> None:
    if payload.get("status") != "prepared_not_authorized":
        raise TrainingProtocolError("training definition has an unexpected status")
    if payload.get("algorithm") != "PPO" or payload.get("authority_interface") != "A1":
        raise TrainingProtocolError("this definition must be PPO under A1 authority")
    if tuple(payload.get("observation_modes", ())) != MATCHED_MODES:
        raise TrainingProtocolError("matched observation modes differ from the frozen triplet")
    seeds = tuple(int(value) for value in payload.get("training_seeds", ()))
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise TrainingProtocolError("observation study requires exactly five unique seeds")
    positive = (
        payload.get("minimum_decisions_per_run"),
        payload.get("maximum_decisions_per_run"),
        payload.get("rollout_steps"),
        payload.get("episode_slots_per_run"),
        payload.get("expected_run_count"),
    )
    if not all(isinstance(value, int) and value > 0 for value in positive):
        raise TrainingProtocolError("training budgets/counts must be positive integers")
    if payload["expected_run_count"] != len(MATCHED_MODES) * len(seeds):
        raise TrainingProtocolError("training run count does not match modes times seeds")
    minimum = int(payload["minimum_decisions_per_run"])
    maximum = int(payload["maximum_decisions_per_run"])
    if maximum < minimum or maximum - minimum > 3:
        raise TrainingProtocolError("episode-boundary decision interval is invalid")
    if int(payload["episode_slots_per_run"]) <= maximum:
        raise TrainingProtocolError("episode schedule lacks scientific-failure headroom")
    _exact_dataclass_payload(dict(payload["ppo_config"]), PPOConfig)
    reward_payload = dict(payload["reward_config"])
    if str(payload.get("format", "")).startswith("forcewipe_v4_"):
        legacy_fields = {
            "removed_mass_fraction_gain_weight",
            "duration_cost_per_s",
            "primitive_cost",
            "shield_intervention_cost",
            "force_violation_cost",
            "spatial_violation_cost",
            "incomplete_lifecycle_cost",
        }
        if set(reward_payload) != legacy_fields:
            raise TrainingProtocolError("legacy V4 SharedRewardConfig fields differ")
        if not all(
            math.isfinite(float(value)) and float(value) >= 0.0
            for value in reward_payload.values()
        ):
            raise TrainingProtocolError("legacy V4 reward weights are invalid")
    else:
        _exact_dataclass_payload(reward_payload, SharedRewardConfig)
    _exact_dataclass_payload(
        dict(payload["backend_config"]), SapienTrainingBackendConfig
    )
    if not str(payload.get("format", "")).startswith("forcewipe_v4_") and not math.isclose(
        float(payload["reward_config"]["terminal_residual_target_ratio"]),
        float(payload["backend_config"]["terminal_residual_mass_ratio"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise TrainingProtocolError("reward and terminal scorer residual thresholds differ")
    expected_actions = 18 + int(bool(payload.get("allow_stop_action")))
    if payload["ppo_config"]["action_count"] != expected_actions:
        raise TrainingProtocolError("PPO logits differ from the frozen action interface")
    vision = dict(payload["vision_config"])
    camera = ResidualStripCameraConfig(**dict(vision.pop("camera")))
    SimulatedVisionConfig(camera=camera, **vision).validate()
    output_root = Path(str(payload.get("runtime_output_root", "")))
    if not output_root.is_absolute() or not str(output_root).startswith("/home/"):
        raise TrainingProtocolError("formal training root must be an absolute WSL home path")
    manifests = payload.get("scenario_manifests", {})
    if set(manifests) != ALLOWED_ROLES:
        raise TrainingProtocolError("scenario manifests must be exactly TRAIN/DEV/CAL")
    if payload.get("simulation_test_permitted") is not False:
        raise TrainingProtocolError("training definition may not authorize simulation TEST")
    if payload.get("formal_training_started") is not False:
        raise TrainingProtocolError("definition builder may not mark training as started")


def validate_protocol_core(payload: dict[str, Any]) -> None:
    if payload.get("format") not in {
        "forcewipe_v4_training_protocol_core_v1",
        "forcewipe_v5_training_protocol_core_v1",
    }:
        raise TrainingProtocolError("unsupported training protocol-core format")
    _validate_training_semantics(payload)


def validate_training_definition(payload: dict[str, Any]) -> None:
    if payload.get("format") not in {
        "forcewipe_v4_training_definition_v3",
        "forcewipe_v5_training_definition_v1",
    }:
        raise TrainingProtocolError("unsupported training-definition format")
    core = payload.get("protocol_core")
    if not isinstance(core, dict) or set(core) != {
        "relative_path", "sha256", "size_bytes"
    }:
        raise TrainingProtocolError("final definition lacks one protocol-core file entry")
    _validate_training_semantics(payload)


def load_training_definition(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    validate_training_definition(payload)
    return payload


def load_protocol_core(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    validate_protocol_core(payload)
    return payload


def load_scenario_manifest(path: Path, *, expected_role: str) -> tuple[ScenarioSpec, ...]:
    role = str(expected_role)
    if role not in ALLOWED_ROLES:
        raise TrainingProtocolError("formal TEST scenarios are forbidden in training reader")
    rows = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            spec = ScenarioSpec(**payload)
            spec.validate()
        except Exception as exc:
            raise TrainingProtocolError(
                f"invalid scenario manifest row {line_number}"
            ) from exc
        if spec.role != role:
            raise TrainingProtocolError("scenario manifest mixes split roles")
        rows.append(spec)
    if not rows:
        raise TrainingProtocolError("scenario manifest is empty")
    if len({item.scenario_seed for item in rows}) != len(rows):
        raise TrainingProtocolError("scenario manifest contains repeated scenario seeds")
    return tuple(rows)


def verify_file_manifest(root: Path, manifest_path: Path) -> dict[str, int]:
    import csv

    rows = list(csv.DictReader(Path(manifest_path).open("r", encoding="utf-8", newline="")))
    if not rows or set(rows[0]) != {"relative_path", "size_bytes", "sha256"}:
        raise TrainingProtocolError("file manifest schema is invalid")
    seen = set()
    for row in rows:
        relative = row["relative_path"]
        if relative in seen or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise TrainingProtocolError("file manifest path is duplicated or unsafe")
        seen.add(relative)
        target = Path(root) / relative
        if not target.is_file():
            raise TrainingProtocolError(f"manifest file is missing: {relative}")
        if target.stat().st_size != int(row["size_bytes"]):
            raise TrainingProtocolError(f"manifest size mismatch: {relative}")
        if sha256_file(target) != row["sha256"]:
            raise TrainingProtocolError(f"manifest hash mismatch: {relative}")
    return {"files": len(rows), "mismatches": 0}
