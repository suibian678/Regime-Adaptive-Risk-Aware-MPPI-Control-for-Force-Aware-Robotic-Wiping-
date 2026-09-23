"""Method-identity and comparison-budget contracts for V4 baselines.

These contracts do not implement or certify an algorithm.  They prevent a
partial implementation from being reported under a stronger published method
name and keep the eventual comparison budget auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from .authority import AuthorityInterface


class BaselineContractError(ValueError):
    pass


class BaselineMethod(str, Enum):
    CLASSICAL_OPTIMIZER = "classical_optimizer"
    RANDOM = "residual_free_random"
    GREEDY = "constrained_greedy"
    EXPLICIT_ORACLE = "explicit_optimizer_oracle"
    SUPERVISED_CLASSIFIER = "supervised_classifier"
    TD_MPC2_INITIALIZED_CLASSIFIER = "tdmpc2_initialized_categorical_classifier"
    PPO = "PPO"
    DQN = "DQN"
    SAC = "SAC"
    TD_MPC2 = "TD-MPC2"
    MAPLE = "MAPLE"
    IMP_HRL = "IMP-HRL"


class ActionTopology(str, Enum):
    DISCRETE = "discrete"
    DISCRETE_LATTICE = "discrete_lattice"
    PARAMETERIZED = "parameterized"
    CONTINUOUS = "continuous"


_REQUIRED_COMPONENTS: dict[BaselineMethod, frozenset[str]] = {
    BaselineMethod.SUPERVISED_CLASSIFIER: frozenset(
        {"encoder", "categorical_head", "supervised_objective"}
    ),
    BaselineMethod.TD_MPC2_INITIALIZED_CLASSIFIER: frozenset(
        {"tdmpc2_encoder_initialization", "categorical_head", "supervised_objective"}
    ),
    BaselineMethod.PPO: frozenset(
        {"actor", "critic", "clipped_surrogate", "on_policy_rollouts"}
    ),
    BaselineMethod.DQN: frozenset(
        {"q_network", "target_network", "replay_buffer", "epsilon_greedy"}
    ),
    BaselineMethod.SAC: frozenset(
        {"stochastic_actor", "twin_q", "replay_buffer", "entropy_temperature"}
    ),
    BaselineMethod.TD_MPC2: frozenset(
        {
            "encoder",
            "latent_dynamics",
            "reward_head",
            "value_head",
            "policy_head",
            "online_mppi",
        }
    ),
    BaselineMethod.MAPLE: frozenset(
        {
            "high_level_policy",
            "parameterized_skill_library",
            "skill_parameter_policy",
        }
    ),
    BaselineMethod.IMP_HRL: frozenset(
        {
            "high_level_primitive_policy",
            "model_based_low_level_controller",
            "learned_force_or_stiffness_parameter",
        }
    ),
}

_NONLEARNING_METHODS = frozenset(
    {
        BaselineMethod.CLASSICAL_OPTIMIZER,
        BaselineMethod.RANDOM,
        BaselineMethod.GREEDY,
        BaselineMethod.EXPLICIT_ORACLE,
    }
)


@dataclass(frozen=True)
class BaselineIdentityEvidence:
    method: BaselineMethod
    authority_interface: AuthorityInterface
    action_topology: ActionTopology
    components: frozenset[str]
    upstream_mechanism_checklist_complete: bool = False


@dataclass(frozen=True)
class BaselineIdentityAudit:
    eligible_for_canonical_name: bool
    reporting_label: str
    missing_components: tuple[str, ...]
    reasons: tuple[str, ...]


def _reporting_label(
    method: BaselineMethod,
    *,
    canonical: bool,
    missing: tuple[str, ...],
) -> str:
    if canonical:
        return method.value
    if method == BaselineMethod.MAPLE:
        return "MAPLE-style"
    if method == BaselineMethod.IMP_HRL:
        return "IMP-HRL-style"
    if method == BaselineMethod.TD_MPC2:
        return "incomplete world-model baseline (not TD-MPC2)"
    suffix = "_".join(missing) if missing else "contract"
    return f"unverified_{method.value}_{suffix}"


def audit_baseline_identity(evidence: BaselineIdentityEvidence) -> BaselineIdentityAudit:
    """Audit method identity without claiming that training or evaluation passed."""

    method = BaselineMethod(evidence.method)
    topology = ActionTopology(evidence.action_topology)
    components = frozenset(evidence.components)
    missing = tuple(sorted(_REQUIRED_COMPONENTS.get(method, frozenset()) - components))
    reasons: list[str] = []

    if method == BaselineMethod.DQN and topology not in {
        ActionTopology.DISCRETE,
        ActionTopology.DISCRETE_LATTICE,
    }:
        reasons.append("DQN_requires_an_explicit_discrete_action_set")
    if method == BaselineMethod.SAC and topology not in {
        ActionTopology.PARAMETERIZED,
        ActionTopology.CONTINUOUS,
    }:
        reasons.append("SAC_requires_a_parameterized_or_continuous_action")
    if method == BaselineMethod.TD_MPC2 and "online_mppi" not in components:
        reasons.append("TD-MPC2_deployment_requires_online_MPPI")
    if method in {BaselineMethod.MAPLE, BaselineMethod.IMP_HRL}:
        if not evidence.upstream_mechanism_checklist_complete:
            reasons.append("published_method_fidelity_checklist_incomplete")
    if method == BaselineMethod.TD_MPC2_INITIALIZED_CLASSIFIER:
        forbidden = {
            "latent_dynamics",
            "reward_head",
            "value_head",
            "online_mppi",
        }
        if components & forbidden:
            reasons.append("classifier_identity_conflicts_with_online_world_model_components")

    if missing:
        reasons.append("missing_required_components")
    canonical = not missing and not reasons
    return BaselineIdentityAudit(
        eligible_for_canonical_name=canonical,
        reporting_label=_reporting_label(method, canonical=canonical, missing=missing),
        missing_components=missing,
        reasons=tuple(reasons),
    )


@dataclass(frozen=True)
class TrainingComparisonBudget:
    method: BaselineMethod
    training_seeds: tuple[int, ...]
    environment_transition_budget: int
    physical_horizon_s: float
    observation_contract_id: str
    reward_contract_id: str
    dev_tuning_run_cap: int

    @property
    def learned(self) -> bool:
        return BaselineMethod(self.method) not in _NONLEARNING_METHODS


def validate_matched_training_budgets(
    budgets: Iterable[TrainingComparisonBudget],
    *,
    minimum_learned_seeds: int = 5,
) -> None:
    """Require matched information/reward/transition budgets across learned methods."""

    items = tuple(budgets)
    if len(items) < 2:
        raise BaselineContractError("at least two methods are required for a comparison")
    if int(minimum_learned_seeds) < 1:
        raise BaselineContractError("minimum_learned_seeds must be positive")

    reference = items[0]
    matched_fields = (
        "environment_transition_budget",
        "physical_horizon_s",
        "observation_contract_id",
        "reward_contract_id",
        "dev_tuning_run_cap",
    )
    for item in items:
        if item.environment_transition_budget <= 0 or item.physical_horizon_s <= 0:
            raise BaselineContractError("training budget and physical horizon must be positive")
        if item.learned:
            unique_seeds = set(item.training_seeds)
            if len(unique_seeds) != len(item.training_seeds):
                raise BaselineContractError(f"{item.method.value} contains repeated training seeds")
            if len(unique_seeds) < int(minimum_learned_seeds):
                raise BaselineContractError(
                    f"{item.method.value} has fewer than {minimum_learned_seeds} training seeds"
                )
        mismatched = [
            field
            for field in matched_fields
            if getattr(item, field) != getattr(reference, field)
        ]
        if mismatched:
            raise BaselineContractError(
                f"{item.method.value} has unmatched comparison fields: {mismatched}"
            )

