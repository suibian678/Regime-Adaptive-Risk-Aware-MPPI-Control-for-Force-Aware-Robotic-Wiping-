"""Stage-A split and recovery-sequence contracts for direct target-12 data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


@dataclass(frozen=True)
class StageARecoveryConfig:
    activation_force_n: float = 11.5
    release_force_n: float = 10.5
    sequence_horizon_steps: int = 20
    internal_headroom_n: float = 14.5
    hard_limit_n: float = 15.0

    def validate(self) -> None:
        if not 0.0 < self.release_force_n < self.activation_force_n:
            raise ValueError("release force must be below activation force")
        if not self.activation_force_n < self.internal_headroom_n < self.hard_limit_n:
            raise ValueError("invalid Stage-A force ordering")
        if self.sequence_horizon_steps < 1:
            raise ValueError("recovery horizon must be positive")


def split_target12_scenarios(
    cells: Iterable[Mapping[str, object]],
    scenarios: Iterable[Mapping[str, object]],
) -> dict[str, tuple[dict, ...]]:
    """Split the 15 physical cells by replicate before fitting Stage A.

    Replicates 0--2 are development-training cells; replicates 3--4 are a
    predictor-validation family.  Every geometry must contribute to both.
    """

    scenario_by_id = {int(row["scenario_id"]): dict(row) for row in scenarios}
    selected = [dict(row) for row in cells if float(row["target_force_n"]) == 12.0]
    if len(selected) != 15:
        raise ValueError("expected exactly 15 target-12 physical cells")
    training, validation = [], []
    geometry_counts = {"training": {}, "validation": {}}
    digests = set()
    for cell in selected:
        scenario_id = int(cell["scenario_id"])
        scenario = scenario_by_id.get(scenario_id)
        if scenario is None:
            raise ValueError(f"missing scenario {scenario_id}")
        digest = str(cell["scenario_digest"])
        if digest in digests:
            raise ValueError("duplicate physical scenario digest")
        digests.add(digest)
        payload = {"cell": cell, "scenario": scenario}
        destination = training if int(cell["replicate"]) <= 2 else validation
        split_name = "training" if destination is training else "validation"
        destination.append(payload)
        key = str(cell["geometry_key"])
        geometry_counts[split_name][key] = geometry_counts[split_name].get(key, 0) + 1
    if len(training) != 9 or len(validation) != 6:
        raise ValueError("Stage-A split must contain 9 training and 6 validation cells")
    if set(geometry_counts["training"]) != set(geometry_counts["validation"]):
        raise ValueError("training and validation geometry families do not align")
    if any(count != 3 for count in geometry_counts["training"].values()):
        raise ValueError("each training geometry requires three replicates")
    if any(count != 2 for count in geometry_counts["validation"].values()):
        raise ValueError("each validation geometry requires two replicates")
    return {"training": tuple(training), "validation": tuple(validation)}


def recovery_query_active(
    current_force_n: float,
    remaining_sequence_steps: int,
    config: StageARecoveryConfig,
) -> bool:
    """Return whether the current action belongs to a recovery sequence."""

    config.validate()
    if remaining_sequence_steps < 0:
        raise ValueError("remaining recovery steps cannot be negative")
    return bool(
        remaining_sequence_steps > 0
        or float(current_force_n) >= config.activation_force_n
    )

