"""Transactional geometry-aware target-pose sequencing for ForceWipe.

The sequencer does not implement force control. It combines one already
bounded scalar normal increment with one bounded path-arc increment. Positive
normal increments point inward. Path progress is committed only after the
caller confirms that the target-pose command was issued successfully.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from forcewipe_v4.path_pose_control import desired_tool_rotation
from forcewipe_v4.path_pose_control import quaternion_wxyz_to_matrix
from forcewipe_v4.scenarios import ScenarioPath, ScenarioSpec, surface_normal
from scipy.spatial.transform import Rotation


class GeometryTargetPoseError(ValueError):
    pass


def _unit(value: np.ndarray, *, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise GeometryTargetPoseError(f"{name} must be a finite three-vector")
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        raise GeometryTargetPoseError(f"{name} must be nonzero")
    return vector / norm


@dataclass(frozen=True)
class GeometryTargetPoseConfig:
    maximum_path_increment_m: float = 0.0002
    maximum_normal_increment_m: float = 0.0008
    surface_top_origin_z_m: float = 0.04
    orthogonality_tolerance: float = 2.0e-5
    action_position_scale_m: float = 0.1
    action_rotation_scale_rad: float = 0.1
    maximum_rotation_step_rad: float = 0.05

    def validate(self) -> None:
        values = (
            self.maximum_path_increment_m,
            self.maximum_normal_increment_m,
            self.surface_top_origin_z_m,
            self.orthogonality_tolerance,
            self.action_position_scale_m,
            self.action_rotation_scale_rad,
            self.maximum_rotation_step_rad,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise GeometryTargetPoseError("geometry target-pose parameters must be positive and finite")
        if self.maximum_rotation_step_rad > self.action_rotation_scale_rad:
            raise GeometryTargetPoseError("rotation step exceeds the action scale")


@dataclass(frozen=True)
class GeometryTargetPoseProposal:
    sequence_id: int
    committed_progress_before: float
    proposed_progress_after: float
    requested_path_increment_m: float
    executed_path_increment_m: float
    normal_increment_m: float
    path_point_before_world_m: tuple[float, float, float]
    path_point_after_world_m: tuple[float, float, float]
    path_delta_world_m: tuple[float, float, float]
    normal_delta_world_m: tuple[float, float, float]
    total_position_delta_world_m: tuple[float, float, float]
    actuation_outward_normal_world: tuple[float, float, float]
    committed_outward_normal_after_world: tuple[float, float, float]
    committed_tangent_after_world: tuple[float, float, float]
    desired_world_from_tool_after: tuple[tuple[float, float, float], ...]
    path_complete_after: bool


def proposal_to_target_delta_action(
    proposal: GeometryTargetPoseProposal,
    *,
    root_quaternion_world_wxyz: np.ndarray,
    target_quaternion_root_wxyz: np.ndarray,
    config: GeometryTargetPoseConfig = GeometryTargetPoseConfig(),
) -> np.ndarray:
    """Map one proposal into ManiSkill's root-aligned target-delta action."""

    config.validate()
    world_from_root = quaternion_wxyz_to_matrix(root_quaternion_world_wxyz)
    root_from_target = quaternion_wxyz_to_matrix(target_quaternion_root_wxyz)
    desired_world_from_tool = np.asarray(
        proposal.desired_world_from_tool_after,
        dtype=np.float64,
    )
    if desired_world_from_tool.shape != (3, 3) or not np.all(np.isfinite(desired_world_from_tool)):
        raise GeometryTargetPoseError("desired tool rotation is invalid")
    desired_root_from_tool = world_from_root.T @ desired_world_from_tool
    root_aligned_delta = desired_root_from_tool @ root_from_target.T
    rotation_delta = Rotation.from_matrix(root_aligned_delta).as_euler("XYZ", degrees=False)
    rotation_norm = float(np.linalg.norm(rotation_delta))
    if rotation_norm > config.maximum_rotation_step_rad:
        rotation_delta *= config.maximum_rotation_step_rad / rotation_norm

    world_position_delta = np.asarray(proposal.total_position_delta_world_m, dtype=np.float64)
    root_position_delta = world_from_root.T @ world_position_delta
    normalized_position = root_position_delta / config.action_position_scale_m
    normalized_rotation = rotation_delta / config.action_rotation_scale_rad
    action = np.concatenate((normalized_position, normalized_rotation, np.array([-1.0])))
    if action.shape != (7,) or not np.all(np.isfinite(action)):
        raise GeometryTargetPoseError("target-delta action is invalid")
    if np.any(np.abs(action[:6]) > 1.0 + 1e-12):
        raise GeometryTargetPoseError("target-delta action exceeds normalized bounds")
    return action.astype(np.float32)


class GeometryTargetPoseSequencer:
    def __init__(
        self,
        *,
        scenario: ScenarioSpec,
        path: ScenarioPath,
        config: GeometryTargetPoseConfig = GeometryTargetPoseConfig(),
    ):
        scenario.validate()
        config.validate()
        if not math.isfinite(path.total_length) or path.total_length <= 0.0:
            raise GeometryTargetPoseError("path total length must be positive and finite")
        self.scenario = scenario
        self.path = path
        self.config = config
        self._committed_progress = 0.0
        self._next_sequence_id = 0
        self._pending: GeometryTargetPoseProposal | None = None

    @property
    def committed_progress(self) -> float:
        return self._committed_progress

    @property
    def pending(self) -> GeometryTargetPoseProposal | None:
        return self._pending

    def reset(self, *, progress: float = 0.0) -> None:
        value = float(progress)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise GeometryTargetPoseError("reset progress must lie in [0, 1]")
        self._committed_progress = value
        self._next_sequence_id = 0
        self._pending = None

    def _world_point(self, relative: np.ndarray) -> np.ndarray:
        point = np.asarray(relative, dtype=np.float64).copy()
        point[2] += self.config.surface_top_origin_z_m
        return point

    def propose(
        self,
        *,
        requested_path_increment_m: float,
        normal_increment_m: float,
    ) -> GeometryTargetPoseProposal:
        if self._pending is not None:
            raise GeometryTargetPoseError("a target-pose proposal is already pending")
        path_increment = float(requested_path_increment_m)
        normal_increment = float(normal_increment_m)
        if not math.isfinite(path_increment) or not 0.0 <= path_increment <= self.config.maximum_path_increment_m:
            raise GeometryTargetPoseError("path increment is outside the registered bound")
        if not math.isfinite(normal_increment) or abs(normal_increment) > self.config.maximum_normal_increment_m:
            raise GeometryTargetPoseError("normal increment is outside the registered bound")

        before_progress = self._committed_progress
        remaining = max(0.0, (1.0 - before_progress) * self.path.total_length)
        executed_path_increment = min(path_increment, remaining)
        after_progress = min(1.0, before_progress + executed_path_increment / self.path.total_length)
        point_before_relative, _tangent_before = self.path.at(before_progress)
        point_after_relative, tangent_after = self.path.at(after_progress)
        midpoint_progress = 0.5 * (before_progress + after_progress)
        midpoint_relative, _midpoint_tangent = self.path.at(midpoint_progress)

        actuation_normal = _unit(
            np.asarray(surface_normal(self.scenario, float(midpoint_relative[0])), dtype=np.float64),
            name="actuation outward normal",
        )
        committed_normal_after = _unit(
            np.asarray(surface_normal(self.scenario, float(point_after_relative[0])), dtype=np.float64),
            name="committed outward normal",
        )
        tangent_after = _unit(np.asarray(tangent_after, dtype=np.float64), name="committed tangent")
        tangent_after = _unit(
            tangent_after - float(np.dot(tangent_after, committed_normal_after)) * committed_normal_after,
            name="surface-projected committed tangent",
        )

        point_before_world = self._world_point(point_before_relative)
        point_after_world = self._world_point(point_after_relative)
        path_delta = point_after_world - point_before_world
        if executed_path_increment > 0.0:
            chord_tangency = abs(float(np.dot(path_delta, actuation_normal)))
            if chord_tangency > self.config.orthogonality_tolerance:
                raise GeometryTargetPoseError("path chord is inconsistent with the instantaneous surface normal")
        normal_delta = -actuation_normal * normal_increment
        total_delta = path_delta + normal_delta
        desired_rotation = desired_tool_rotation(tangent_after, committed_normal_after)

        proposal = GeometryTargetPoseProposal(
            sequence_id=self._next_sequence_id,
            committed_progress_before=before_progress,
            proposed_progress_after=after_progress,
            requested_path_increment_m=path_increment,
            executed_path_increment_m=executed_path_increment,
            normal_increment_m=normal_increment,
            path_point_before_world_m=tuple(float(value) for value in point_before_world),
            path_point_after_world_m=tuple(float(value) for value in point_after_world),
            path_delta_world_m=tuple(float(value) for value in path_delta),
            normal_delta_world_m=tuple(float(value) for value in normal_delta),
            total_position_delta_world_m=tuple(float(value) for value in total_delta),
            actuation_outward_normal_world=tuple(float(value) for value in actuation_normal),
            committed_outward_normal_after_world=tuple(float(value) for value in committed_normal_after),
            committed_tangent_after_world=tuple(float(value) for value in tangent_after),
            desired_world_from_tool_after=tuple(tuple(float(value) for value in row) for row in desired_rotation),
            path_complete_after=after_progress >= 1.0,
        )
        self._pending = proposal
        return proposal

    def commit(self, proposal: GeometryTargetPoseProposal) -> None:
        if self._pending is None or proposal != self._pending:
            raise GeometryTargetPoseError("only the current pending proposal may be committed")
        self._committed_progress = proposal.proposed_progress_after
        self._next_sequence_id += 1
        self._pending = None

    def discard(self, proposal: GeometryTargetPoseProposal) -> None:
        if self._pending is None or proposal != self._pending:
            raise GeometryTargetPoseError("only the current pending proposal may be discarded")
        self._pending = None
