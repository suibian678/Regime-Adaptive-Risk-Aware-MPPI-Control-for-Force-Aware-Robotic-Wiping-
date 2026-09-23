"""Bounded DAgger contracts for the direct target-12 development screen."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class DirectDaggerConfig:
    rounds: int = 3
    scenarios_per_round: int = 3
    query_force_n: float = 8.0
    planning_horizon_steps: int = 50
    execution_chunk_steps: int = 20
    maximum_queries_per_episode: int = 50
    updates_per_round: int = 1000
    batch_size: int = 256
    learning_rate: float = 1.0e-4
    retention_coefficient: float = 0.25
    hard_limit_n: float = 15.0
    internal_headroom_n: float = 14.5
    maximum_mae_n: float = 1.0
    maximum_rmse_n: float = 2.0

    def validate(self) -> None:
        if self.rounds != 3 or self.scenarios_per_round != 3:
            raise ValueError("the bounded DAgger screen requires exactly three 3-cell rounds")
        if not 0.0 < self.query_force_n < self.internal_headroom_n < self.hard_limit_n:
            raise ValueError("invalid force ordering")
        if self.planning_horizon_steps != 50:
            raise ValueError("the frozen planning horizon is 50 steps")
        if not 0 < self.execution_chunk_steps <= self.planning_horizon_steps:
            raise ValueError("invalid execution chunk")
        if self.maximum_queries_per_episode < 1:
            raise ValueError("maximum queries must be positive")
        if self.updates_per_round < 1 or self.batch_size < 1:
            raise ValueError("training budget must be positive")
        if self.learning_rate <= 0.0 or self.retention_coefficient < 0.0:
            raise ValueError("invalid optimizer configuration")
        if self.maximum_mae_n <= 0.0 or self.maximum_rmse_n <= 0.0:
            raise ValueError("tracking gates must be positive")


def should_query_oracle(
    *,
    current_force_n: float,
    queued_actions: int,
    completed_queries: int,
    config: DirectDaggerConfig,
) -> bool:
    """Return whether a new causal expert query is permitted at this state."""

    config.validate()
    if queued_actions < 0 or completed_queries < 0:
        raise ValueError("query counters cannot be negative")
    return bool(
        queued_actions == 0
        and completed_queries < config.maximum_queries_per_episode
        and float(current_force_n) >= config.query_force_n
    )


def strict_direct_cell_pass(metrics: Mapping[str, object], config: DirectDaggerConfig) -> bool:
    """Apply the frozen task, safety, headroom, and tracking gates."""

    config.validate()
    mae = metrics.get("contact_force_mae_n")
    rmse = metrics.get("contact_force_rmse_n")
    return bool(
        metrics.get("success") is True
        and int(metrics.get("force_limit_violation_samples", -1)) == 0
        and float(metrics.get("peak_force_n", float("inf"))) <= config.internal_headroom_n
        and mae is not None
        and float(mae) <= config.maximum_mae_n
        and rmse is not None
        and float(rmse) <= config.maximum_rmse_n
    )


def validate_round_schedule(round_cells: tuple[tuple[str, ...], ...]) -> None:
    """Require three disjoint rounds with one cell from each geometry."""

    if len(round_cells) != 3 or any(len(row) != 3 for row in round_cells):
        raise ValueError("expected a three-by-three DAgger schedule")
    flattened = [cell for row in round_cells for cell in row]
    if len(set(flattened)) != 9:
        raise ValueError("DAgger aggregation cells must be unique")
    for row in round_cells:
        geometries = {
            cell.removeprefix("12n_").rsplit("_r", 1)[0]
            for cell in row
        }
        if geometries != {"flat_arc", "flat_s_curve", "incline5_diagonal"}:
            raise ValueError("each DAgger round requires the three supported geometries")
