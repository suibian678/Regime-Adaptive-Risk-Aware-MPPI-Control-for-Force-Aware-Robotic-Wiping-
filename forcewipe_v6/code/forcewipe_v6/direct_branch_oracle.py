"""Training-only replay-prefix branching for direct TD-MPC2 aggregation.

ManiSkill/SAPIEN public state dictionaries do not include every contact-solver
cache needed for an exact next-step branch. This oracle deliberately uses the
slower scientific path: reset the same scenario and replay the complete action
prefix independently for every candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

from .tdmpc2_boundary_risk import select_counterfactual_candidate


@dataclass(frozen=True)
class BranchOutcome:
    candidate_index: int
    action: tuple[float, float, float]
    future_peak_force_n: float
    progress_gain: float
    force_limit_violation: bool
    executed_steps: int


@dataclass(frozen=True)
class BranchSelection:
    selected_index: int
    selected_action: tuple[float, float, float]
    selected_is_headroom_safe: bool
    outcomes: tuple[BranchOutcome, ...]


@dataclass(frozen=True)
class SequenceTemplate:
    name: str
    constrained_steps: int
    normal_upper_bound: float | None
    tangent_scale: float = 1.0

    def validate(self) -> None:
        if self.constrained_steps < 0:
            raise ValueError("constrained_steps cannot be negative")
        if self.normal_upper_bound is not None and not -1.0 <= self.normal_upper_bound <= 1.0:
            raise ValueError("normal upper bound must lie in the actuator box")
        if not 0.0 < self.tangent_scale <= 1.0:
            raise ValueError("tangent scale must lie in (0, 1]")


@dataclass(frozen=True)
class SequenceBranchOutcome:
    template_index: int
    template_name: str
    actions: tuple[tuple[float, float, float], ...]
    future_peak_force_n: float
    progress_gain: float
    force_limit_violation: bool
    executed_steps: int


@dataclass(frozen=True)
class SequenceBranchSelection:
    selected_index: int
    selected_template_name: str
    selected_is_headroom_safe: bool
    selected_actions: tuple[tuple[float, float, float], ...]
    outcomes: tuple[SequenceBranchOutcome, ...]


def candidate_action_lattice(behavior_action: object) -> np.ndarray:
    """Return the fixed eight-action training-only branch lattice."""

    behavior = np.clip(np.asarray(behavior_action, dtype=np.float32).reshape(3), -1, 1)
    rows = [behavior.copy()]
    for normal in (-0.25, -0.50, -0.75, -1.00):
        rows.append(np.asarray([behavior[0], behavior[1], normal], dtype=np.float32))
    for normal in (-0.50, -0.75, -1.00):
        rows.append(
            np.asarray([0.5 * behavior[0], behavior[1], normal], dtype=np.float32)
        )
    result = np.asarray(rows, dtype=np.float32)
    if result.shape != (8, 3):
        raise RuntimeError("counterfactual candidate lattice must contain eight actions")
    return result


def recovery_sequence_templates(horizon_steps: int) -> tuple[SequenceTemplate, ...]:
    """Return the fixed, training-only sequence lattice."""

    if horizon_steps < 10:
        raise ValueError("recovery sequence horizon must be at least ten steps")
    return (
        SequenceTemplate("actor_baseline", 0, None),
        SequenceTemplate("no_inward_full", horizon_steps, 0.0),
        SequenceTemplate("outward025_first5", 5, -0.25),
        SequenceTemplate("outward025_first10", 10, -0.25),
        SequenceTemplate("outward050_first5", 5, -0.50),
        SequenceTemplate("outward050_first10", 10, -0.50),
        SequenceTemplate("outward075_first5", 5, -0.75),
        SequenceTemplate("half_tangent_outward050_first10", 10, -0.50, 0.5),
    )


def recovery_sequence_templates_v2(horizon_steps: int) -> tuple[SequenceTemplate, ...]:
    """Extend the original lattice with diagnosed long outward phases.

    The v1 function remains unchanged so completed development artifacts keep
    an unambiguous template definition.
    """

    if horizon_steps < 20:
        raise ValueError("v2 recovery sequence horizon must be at least twenty steps")
    return recovery_sequence_templates(horizon_steps) + (
        SequenceTemplate("outward025_first20", 20, -0.25),
        SequenceTemplate("outward050_first20", 20, -0.50),
        SequenceTemplate("outward075_first20", 20, -0.75),
        SequenceTemplate("outward100_first10", 10, -1.00),
        SequenceTemplate("outward100_first20", 20, -1.00),
    )


def apply_sequence_template(
    actor_action: object,
    branch_step: int,
    template: SequenceTemplate,
) -> np.ndarray:
    """Apply one causal template to the current actor proposal."""

    template.validate()
    if branch_step < 0:
        raise ValueError("branch_step cannot be negative")
    action = np.clip(np.asarray(actor_action, dtype=np.float32).reshape(3), -1.0, 1.0)
    if branch_step < template.constrained_steps:
        action = action.copy()
        action[0] *= float(template.tangent_scale)
        if template.normal_upper_bound is not None:
            action[2] = min(float(action[2]), float(template.normal_upper_bound))
    return action


def evaluate_counterfactual_sequence_templates_by_replay(
    env_factory: Callable[[], object],
    agent,
    query_observation: np.ndarray,
    prefix_actions: object,
    templates: tuple[SequenceTemplate, ...],
    *,
    reset_seed: int,
    horizon_steps: int,
    internal_headroom_n: float,
    replay_atol: float = 1e-6,
) -> SequenceBranchSelection:
    """Evaluate complete causal action sequences after the same replayed prefix."""

    if horizon_steps < 1 or not templates:
        raise ValueError("positive horizon and nonempty templates are required")
    prefix = np.asarray(prefix_actions, dtype=np.float32)
    query = np.asarray(query_observation, dtype=np.float32).reshape(-1)
    if prefix.ndim != 2 or prefix.shape[1] != 3 or len(prefix) < 1:
        raise ValueError("prefix_actions must be a nonempty T-by-3 array")
    outcomes = []
    for index, template in enumerate(templates):
        template.validate()
        env = env_factory()
        try:
            observation, _ = env.reset(seed=int(reset_seed))
            replay_info = None
            for prefix_index, action in enumerate(prefix):
                observation, _reward, terminated, truncated, replay_info = env.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        "recorded prefix terminated before sequence branch at "
                        f"index {prefix_index}"
                    )
            if not np.allclose(observation, query, atol=replay_atol, rtol=0.0):
                difference = float(np.max(np.abs(observation - query)))
                raise RuntimeError(
                    f"replayed sequence observation mismatch: max_abs={difference}"
                )
            if replay_info is None:
                raise RuntimeError("sequence replay prefix produced no physical info")
            initial_progress = float(replay_info["progress"])
            progress = initial_progress
            peak = float(replay_info["normal_force_n"])
            violation = False
            executed_actions = []
            for branch_step in range(horizon_steps):
                proposal = agent.act(
                    torch.from_numpy(observation),
                    t0=False,
                    eval_mode=True,
                ).numpy()
                action = apply_sequence_template(proposal, branch_step, template)
                executed_actions.append(tuple(float(value) for value in action))
                observation, _reward, terminated, truncated, info = env.step(action)
                peak = max(peak, float(info["normal_force_n"]))
                progress = float(info["progress"])
                violation = violation or bool(info["force_limit_violation"])
                if terminated or truncated:
                    break
            outcomes.append(
                SequenceBranchOutcome(
                    template_index=index,
                    template_name=template.name,
                    actions=tuple(executed_actions),
                    future_peak_force_n=peak,
                    progress_gain=progress - initial_progress,
                    force_limit_violation=violation,
                    executed_steps=len(executed_actions),
                )
            )
        finally:
            env.close()
    safe = [
        index
        for index, row in enumerate(outcomes)
        if row.future_peak_force_n <= float(internal_headroom_n)
        and not row.force_limit_violation
    ]
    if safe:
        selected = sorted(
            safe,
            key=lambda i: (-outcomes[i].progress_gain, outcomes[i].future_peak_force_n, i),
        )[0]
        is_safe = True
    else:
        selected = min(
            range(len(outcomes)),
            key=lambda i: (outcomes[i].future_peak_force_n, -outcomes[i].progress_gain, i),
        )
        is_safe = False
    chosen = outcomes[selected]
    return SequenceBranchSelection(
        selected_index=int(selected),
        selected_template_name=chosen.template_name,
        selected_is_headroom_safe=is_safe,
        selected_actions=chosen.actions,
        outcomes=tuple(outcomes),
    )


def evaluate_counterfactual_actions_by_replay(
    env_factory: Callable[[], object],
    agent,
    query_observation: np.ndarray,
    prefix_actions: object,
    candidate_actions: np.ndarray,
    *,
    reset_seed: int,
    horizon_steps: int,
    internal_headroom_n: float,
    replay_atol: float = 1e-6,
) -> BranchSelection:
    """Evaluate candidates after independently replaying the same full prefix."""

    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    candidates = np.asarray(candidate_actions, dtype=np.float32)
    prefix = np.asarray(prefix_actions, dtype=np.float32)
    query = np.asarray(query_observation, dtype=np.float32).reshape(-1)
    if candidates.ndim != 2 or candidates.shape[1] != 3:
        raise ValueError("candidate_actions must be N-by-3")
    if prefix.ndim != 2 or prefix.shape[1] != 3:
        raise ValueError("prefix_actions must be T-by-3")
    if len(prefix) < 1:
        raise ValueError("at least one prefix action is required")
    outcomes = []
    for index, candidate in enumerate(candidates):
        env = env_factory()
        try:
            replay_observation, _ = env.reset(seed=int(reset_seed))
            replay_info = None
            for prefix_index, action in enumerate(prefix):
                (
                    replay_observation,
                    _reward,
                    terminated,
                    truncated,
                    replay_info,
                ) = env.step(action)
                if terminated or truncated:
                    raise RuntimeError(
                        "recorded prefix terminated before the branch at "
                        f"index {prefix_index}"
                    )
            if not np.allclose(replay_observation, query, atol=replay_atol, rtol=0.0):
                difference = float(np.max(np.abs(replay_observation - query)))
                raise RuntimeError(
                    f"replayed causal observation mismatch: max_abs={difference}"
                )
            if replay_info is None:
                raise RuntimeError("replay prefix did not produce physical information")
            initial_progress = float(replay_info["progress"])
            peak = float(replay_info["normal_force_n"])
            progress = initial_progress
            violation = False
            executed = 0
            branch_observation = replay_observation
            for branch_step in range(horizon_steps):
                if branch_step == 0:
                    action = candidate
                else:
                    action = agent.act(
                        torch.from_numpy(branch_observation),
                        t0=False,
                        eval_mode=True,
                    ).numpy()
                (
                    branch_observation,
                    _reward,
                    terminated,
                    truncated,
                    info,
                ) = env.step(action)
                executed += 1
                peak = max(peak, float(info["normal_force_n"]))
                progress = float(info["progress"])
                violation = violation or bool(info["force_limit_violation"])
                if terminated or truncated:
                    break
            outcomes.append(
                BranchOutcome(
                    candidate_index=index,
                    action=tuple(float(value) for value in candidate),
                    future_peak_force_n=peak,
                    progress_gain=progress - initial_progress,
                    force_limit_violation=violation,
                    executed_steps=executed,
                )
            )
        finally:
            env.close()
    selected, safe = select_counterfactual_candidate(
        candidates,
        [row.future_peak_force_n for row in outcomes],
        [row.progress_gain for row in outcomes],
        internal_headroom_n=internal_headroom_n,
    )
    return BranchSelection(
        selected_index=selected,
        selected_action=outcomes[selected].action,
        selected_is_headroom_safe=safe,
        outcomes=tuple(outcomes),
    )
