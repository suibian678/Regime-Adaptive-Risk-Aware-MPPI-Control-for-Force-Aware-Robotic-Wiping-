"""Bounded PPO budget and hyperparameter study for the ForceWipe baseline.

The frozen V16 PPO implementation deliberately required a single matched
transition budget.  This module leaves that historical implementation intact
and defines the small, prespecified extension used to test whether its null
result was caused by budget or by a conventional PPO setting.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Iterable

from forcewipe.learning.direct_ppo_baseline import (
    DirectPPOConfig,
    DirectPPOError,
    TDMPC2_TRAINING_SOURCE_TRANSITIONS,
)


BASE_TRANSITIONS = TDMPC2_TRAINING_SOURCE_TRANSITIONS
TRAINING_SEEDS = (301, 302, 303, 304, 305)


@dataclass(frozen=True)
class ExtendedDirectPPOConfig(DirectPPOConfig):
    """The historical PPO configuration with an explicitly bounded budget."""

    def validate(self) -> None:
        positive = (
            self.observation_dimension,
            self.action_dimension,
            self.hidden_dimension,
            self.total_environment_transitions,
            self.rollout_transitions,
            self.update_epochs,
            self.minibatches_per_epoch,
            self.learning_rate,
            self.adam_epsilon,
            self.maximum_gradient_norm,
        )
        if not all(float(value) > 0 for value in positive):
            raise DirectPPOError("PPO dimensions/budgets must be positive")
        if self.observation_dimension != 16 or self.action_dimension != 3:
            raise DirectPPOError("direct PPO must use the common 16-D/3-D interface")
        if self.total_environment_transitions not in {
            BASE_TRANSITIONS,
            2 * BASE_TRANSITIONS,
            4 * BASE_TRANSITIONS,
        }:
            raise DirectPPOError("extended PPO budget must be exactly 1x, 2x, or 4x")
        if not 0 < self.discount <= 1 or not 0 < self.gae_lambda <= 1:
            raise DirectPPOError("invalid discount/GAE")
        if not 0 < self.clip_ratio < 1:
            raise DirectPPOError("invalid PPO clip ratio")
        if self.entropy_coefficient < 0:
            raise DirectPPOError("entropy coefficient must be non-negative")


@dataclass(frozen=True)
class PPOStudyProfile:
    profile_id: str
    changed_parameter: str
    changed_value: float
    maximum_budget_multiplier: int
    checkpoint_multipliers: tuple[int, ...]

    @property
    def maximum_transitions(self) -> int:
        return self.maximum_budget_multiplier * BASE_TRANSITIONS

    @property
    def checkpoint_transitions(self) -> tuple[int, ...]:
        return tuple(value * BASE_TRANSITIONS for value in self.checkpoint_multipliers)


# One-factor-at-a-time brackets around the historical implementation.
# Clip 0.1--0.3 is the usual range documented by OpenAI Spinning Up; the
# learning-rate bracket is approximately one decade around 3e-4.  Entropy
# brackets the historical 0.005 coefficient with no bonus and a twofold bonus.
PROFILES = (
    PPOStudyProfile("default", "none", 0.0, 4, (1, 2, 4)),
    PPOStudyProfile("lr_1e-4", "learning_rate", 1e-4, 2, (2,)),
    PPOStudyProfile("lr_1e-3", "learning_rate", 1e-3, 2, (2,)),
    PPOStudyProfile("clip_0p1", "clip_ratio", 0.1, 2, (2,)),
    PPOStudyProfile("clip_0p3", "clip_ratio", 0.3, 2, (2,)),
    PPOStudyProfile("entropy_0", "entropy_coefficient", 0.0, 2, (2,)),
    PPOStudyProfile("entropy_0p01", "entropy_coefficient", 0.01, 2, (2,)),
)
PROFILE_BY_ID = {profile.profile_id: profile for profile in PROFILES}


def config_for_profile(profile_id: str) -> ExtendedDirectPPOConfig:
    try:
        profile = PROFILE_BY_ID[str(profile_id)]
    except KeyError as exc:
        raise DirectPPOError(f"unknown PPO sensitivity profile: {profile_id}") from exc
    config = ExtendedDirectPPOConfig(
        total_environment_transitions=profile.maximum_transitions
    )
    if profile.changed_parameter != "none":
        config = replace(config, **{profile.changed_parameter: profile.changed_value})
    config.validate()
    return config


def configuration_endpoints() -> tuple[tuple[str, int], ...]:
    """The nine prespecified profile-by-budget endpoints."""

    return tuple(
        (profile.profile_id, multiplier)
        for profile in PROFILES
        for multiplier in profile.checkpoint_multipliers
    )


def checkpoint_payload(
    *, model, optimizer, profile_id: str, seed: int, transitions: int
) -> dict:
    profile = PROFILE_BY_ID[profile_id]
    if int(seed) not in TRAINING_SEEDS:
        raise DirectPPOError("unexpected PPO training seed")
    if int(transitions) not in profile.checkpoint_transitions:
        raise DirectPPOError("checkpoint is not a prespecified endpoint")
    return {
        "format": "forcewipe_v19_ppo_budget_sensitivity_checkpoint_v1",
        "profile_id": profile_id,
        "seed": int(seed),
        "environment_transitions": int(transitions),
        "budget_multiplier": int(transitions) // BASE_TRANSITIONS,
        "config": asdict(model.config),
        "model": model.state_dict(),
        "optimizer": optimizer.optimizer.state_dict(),
    }


def selection_key(rows: Iterable[dict], *, profile_id: str, budget_multiplier: int) -> tuple:
    """Return the frozen lexicographic DEV-selection key (larger is better)."""

    selected = [
        row
        for row in rows
        if row["profile_id"] == profile_id
        and int(row["budget_multiplier"]) == int(budget_multiplier)
    ]
    if len(selected) != 90:
        raise DirectPPOError("each DEV endpoint must contain 6 blocks x 3 targets x 5 seeds")
    compound = sum(
        bool(row["task_success"])
        and bool(row["tracking_pass"])
        and bool(row["safety_pass"])
        and bool(row["authority_pass"])
        for row in selected
    )
    task = sum(bool(row["task_success"]) for row in selected)
    bins = sum(int(row["completed_dose_bins"]) for row in selected)
    episode_return = sum(float(row["episode_return"]) for row in selected)
    # The final two fields make a residual exact tie deterministic without
    # pretending that profile names contain scientific information.
    endpoint_order = configuration_endpoints().index((profile_id, int(budget_multiplier)))
    return compound, task, bins, episode_return, -endpoint_order

