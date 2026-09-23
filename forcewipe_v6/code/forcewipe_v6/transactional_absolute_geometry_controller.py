"""Transactional absolute-reference geometry and normal-offset controller.

The controller owns every command-side state used by the passing V6 geometry
mechanism: path progress, scalar normal offset, the initial target/tool-center
reference, and the pending action.  Neither path progress nor normal offset is
advanced until the caller commits the exact transaction after a successful
environment step.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from forcewipe_v4.path_pose_control import quaternion_wxyz_to_matrix
from forcewipe_v4.scene_geometry import V4_TOOL_TCP_CENTER_OFFSET_M
from forcewipe_v4.scenarios import ScenarioPath, ScenarioSpec
from forcewipe_v6.absolute_geometry_target_pose import (
    absolute_geometry_target_delta_action,
    proposed_normal_offset,
)
from forcewipe_v6.geometry_target_pose import (
    GeometryTargetPoseConfig,
    GeometryTargetPoseError,
    GeometryTargetPoseProposal,
    GeometryTargetPoseSequencer,
)


def _vector(value: np.ndarray, *, name: str, length: int = 3) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise GeometryTargetPoseError(f"{name} must be a finite {length}-vector")
    return vector.copy()


@dataclass(frozen=True)
class AbsoluteGeometryFrames:
    root_position_world_m: tuple[float, float, float]
    root_quaternion_world_wxyz: tuple[float, float, float, float]
    target_position_root_m: tuple[float, float, float]
    target_quaternion_root_wxyz: tuple[float, float, float, float]

    @classmethod
    def from_mapping(cls, frames: dict[str, np.ndarray]) -> "AbsoluteGeometryFrames":
        return cls(
            root_position_world_m=tuple(
                float(value) for value in _vector(frames["root_position_world"], name="root position")
            ),
            root_quaternion_world_wxyz=tuple(
                float(value)
                for value in _vector(
                    frames["root_quaternion_world"], name="root quaternion", length=4
                )
            ),
            target_position_root_m=tuple(
                float(value) for value in _vector(frames["target_position_root"], name="target position")
            ),
            target_quaternion_root_wxyz=tuple(
                float(value)
                for value in _vector(
                    frames["target_quaternion_root"], name="target quaternion", length=4
                )
            ),
        )


@dataclass(frozen=True)
class AbsoluteGeometryTransaction:
    geometry_proposal: GeometryTargetPoseProposal
    normal_offset_before_m: float
    normal_offset_after_m: float
    normal_offset_projection_active: bool
    desired_tool_center_world_m: tuple[float, float, float]
    normalized_action: tuple[float, float, float, float, float, float, float]

    @property
    def sequence_id(self) -> int:
        return self.geometry_proposal.sequence_id

    def action_array(self) -> np.ndarray:
        return np.asarray(self.normalized_action, dtype=np.float32)


class TransactionalAbsoluteGeometryController:
    """Own and atomically update geometry and scalar normal-offset state."""

    def __init__(
        self,
        *,
        scenario: ScenarioSpec,
        path: ScenarioPath,
        geometry_config: GeometryTargetPoseConfig = GeometryTargetPoseConfig(),
        minimum_normal_offset_m: float = -0.01,
        maximum_normal_offset_m: float = 0.15,
    ):
        scenario.validate()
        geometry_config.validate()
        minimum = float(minimum_normal_offset_m)
        maximum = float(maximum_normal_offset_m)
        if not all(math.isfinite(value) for value in (minimum, maximum)) or minimum >= maximum:
            raise GeometryTargetPoseError("normal-offset bounds are invalid")
        self.scenario = scenario
        self.path = path
        self.geometry_config = geometry_config
        self.minimum_normal_offset_m = minimum
        self.maximum_normal_offset_m = maximum
        self._sequencer = GeometryTargetPoseSequencer(
            scenario=scenario,
            path=path,
            config=geometry_config,
        )
        self._normal_offset_m = 0.0
        self._initial_target_tool_center_world_m: np.ndarray | None = None
        self._path_start_world_m: np.ndarray | None = None
        self._pending: AbsoluteGeometryTransaction | None = None

    @property
    def committed_progress(self) -> float:
        return self._sequencer.committed_progress

    @property
    def normal_offset_m(self) -> float:
        return self._normal_offset_m

    @property
    def pending(self) -> AbsoluteGeometryTransaction | None:
        return self._pending

    @property
    def initialized(self) -> bool:
        return self._initial_target_tool_center_world_m is not None

    def reset(
        self,
        *,
        frames: AbsoluteGeometryFrames,
        progress: float = 0.0,
        normal_offset_m: float = 0.0,
    ) -> None:
        if self._pending is not None:
            raise GeometryTargetPoseError("cannot reset with a pending transaction")
        offset = float(normal_offset_m)
        if not math.isfinite(offset) or not self.minimum_normal_offset_m <= offset <= self.maximum_normal_offset_m:
            raise GeometryTargetPoseError("reset normal offset lies outside the registered bounds")
        root_position = _vector(np.asarray(frames.root_position_world_m), name="root position")
        target_position = _vector(np.asarray(frames.target_position_root_m), name="target position")
        world_from_root = quaternion_wxyz_to_matrix(
            np.asarray(frames.root_quaternion_world_wxyz, dtype=np.float64)
        )
        root_from_target = quaternion_wxyz_to_matrix(
            np.asarray(frames.target_quaternion_root_wxyz, dtype=np.float64)
        )
        local_offset = np.array([0.0, 0.0, V4_TOOL_TCP_CENTER_OFFSET_M], dtype=np.float64)
        initial_center = root_position + world_from_root @ (
            target_position + root_from_target @ local_offset
        )
        path_start_relative, _ = self.path.at(0.0)
        path_start = np.asarray(path_start_relative, dtype=np.float64).copy()
        path_start[2] += self.geometry_config.surface_top_origin_z_m
        if not np.all(np.isfinite(initial_center)) or not np.all(np.isfinite(path_start)):
            raise GeometryTargetPoseError("initial absolute geometry reference is invalid")
        self._sequencer.reset(progress=progress)
        self._normal_offset_m = offset
        self._initial_target_tool_center_world_m = initial_center
        self._path_start_world_m = path_start

    def propose(
        self,
        *,
        requested_path_increment_m: float,
        requested_normal_increment_m: float,
        frames: AbsoluteGeometryFrames,
    ) -> AbsoluteGeometryTransaction:
        if not self.initialized:
            raise GeometryTargetPoseError("controller must be reset from causal frames before proposing")
        if self._pending is not None:
            raise GeometryTargetPoseError("an absolute-geometry transaction is already pending")
        proposal = self._sequencer.propose(
            requested_path_increment_m=requested_path_increment_m,
            normal_increment_m=requested_normal_increment_m,
        )
        try:
            offset_after, projection_active = proposed_normal_offset(
                self._normal_offset_m,
                proposal.normal_increment_m,
                minimum_offset_m=self.minimum_normal_offset_m,
                maximum_offset_m=self.maximum_normal_offset_m,
            )
            action, desired_center = absolute_geometry_target_delta_action(
                proposal,
                initial_target_tool_center_world_m=self._initial_target_tool_center_world_m,
                path_start_world_m=self._path_start_world_m,
                normal_offset_after_m=offset_after,
                root_position_world_m=np.asarray(frames.root_position_world_m, dtype=np.float64),
                root_quaternion_world_wxyz=np.asarray(
                    frames.root_quaternion_world_wxyz, dtype=np.float64
                ),
                target_position_root_m=np.asarray(frames.target_position_root_m, dtype=np.float64),
                target_quaternion_root_wxyz=np.asarray(
                    frames.target_quaternion_root_wxyz, dtype=np.float64
                ),
                config=self.geometry_config,
            )
            transaction = AbsoluteGeometryTransaction(
                geometry_proposal=proposal,
                normal_offset_before_m=self._normal_offset_m,
                normal_offset_after_m=offset_after,
                normal_offset_projection_active=projection_active,
                desired_tool_center_world_m=tuple(float(value) for value in desired_center),
                normalized_action=tuple(float(value) for value in action),
            )
        except Exception:
            self._sequencer.discard(proposal)
            raise
        self._pending = transaction
        return transaction

    def commit(self, transaction: AbsoluteGeometryTransaction) -> None:
        if self._pending is None or transaction is not self._pending:
            raise GeometryTargetPoseError("only the current absolute-geometry transaction may be committed")
        self._sequencer.commit(transaction.geometry_proposal)
        self._normal_offset_m = transaction.normal_offset_after_m
        self._pending = None

    def discard(self, transaction: AbsoluteGeometryTransaction) -> None:
        if self._pending is None or transaction is not self._pending:
            raise GeometryTargetPoseError("only the current absolute-geometry transaction may be discarded")
        self._sequencer.discard(transaction.geometry_proposal)
        self._pending = None
