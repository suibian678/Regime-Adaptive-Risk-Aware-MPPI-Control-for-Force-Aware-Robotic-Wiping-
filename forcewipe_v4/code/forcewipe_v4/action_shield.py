"""Explicit V4 action-shield modes with information-use and intervention logs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from .authority import AuthorityProposal


class ShieldContractError(ValueError):
    pass


class ShieldMode(str, Enum):
    OFF = "off"
    SUBSTITUTE_ONLY = "substitute_only"
    VETO_TERMINATE = "veto_terminate"


class ShieldResidualAccess(str, Enum):
    RESIDUAL_FREE = "residual_free"
    ESTIMATE_ONLY = "residual_estimate_only"


class RiskResidualSource(str, Enum):
    NONE = "none"
    ESTIMATE = "estimate"
    TRUTH = "truth"


@dataclass(frozen=True)
class ShieldRiskAssessment:
    predicted_peak_force_n: float
    spatial_constraint_violation: bool = False
    model_valid: bool = True
    residual_source: RiskResidualSource = RiskResidualSource.NONE


@dataclass(frozen=True)
class ShieldDecision:
    mode: ShieldMode
    proposal: AuthorityProposal
    executed: AuthorityProposal | None
    intervention: str
    reason: str | None
    proposed_predicted_peak_force_n: float
    executed_predicted_peak_force_n: float | None
    terminate: bool


def _validate_information_access(
    assessment: ShieldRiskAssessment,
    residual_access: ShieldResidualAccess,
) -> None:
    source = RiskResidualSource(assessment.residual_source)
    if source == RiskResidualSource.TRUTH:
        raise ShieldContractError("the action shield may not read truth residual")
    if residual_access == ShieldResidualAccess.RESIDUAL_FREE and source != RiskResidualSource.NONE:
        raise ShieldContractError("residual-free shield received a residual-dependent risk")


def _unsafe_reason(
    assessment: ShieldRiskAssessment,
    *,
    predicted_force_limit_n: float,
) -> str | None:
    force = float(assessment.predicted_peak_force_n)
    if not assessment.model_valid or not math.isfinite(force):
        return "invalid_risk_model_output"
    if assessment.spatial_constraint_violation:
        return "predicted_spatial_constraint_violation"
    if force > float(predicted_force_limit_n):
        return "predicted_force_headroom_violation"
    return None


def apply_action_shield(
    proposal: AuthorityProposal,
    proposal_assessment: ShieldRiskAssessment,
    *,
    mode: ShieldMode,
    residual_access: ShieldResidualAccess,
    predicted_force_limit_n: float = 14.5,
    substitute: AuthorityProposal | None = None,
    substitute_assessment: ShieldRiskAssessment | None = None,
) -> ShieldDecision:
    """Pass, substitute, or terminate without altering the authority interface."""

    mode = ShieldMode(mode)
    residual_access = ShieldResidualAccess(residual_access)
    if predicted_force_limit_n <= 0 or not math.isfinite(float(predicted_force_limit_n)):
        raise ShieldContractError("predicted force limit must be finite and positive")
    _validate_information_access(proposal_assessment, residual_access)
    reason = _unsafe_reason(
        proposal_assessment,
        predicted_force_limit_n=predicted_force_limit_n,
    )

    if mode == ShieldMode.OFF or reason is None:
        return ShieldDecision(
            mode=mode,
            proposal=proposal,
            executed=proposal,
            intervention="pass_through",
            reason=reason if mode == ShieldMode.OFF else None,
            proposed_predicted_peak_force_n=float(
                proposal_assessment.predicted_peak_force_n
            ),
            executed_predicted_peak_force_n=float(
                proposal_assessment.predicted_peak_force_n
            ),
            terminate=False,
        )

    if mode == ShieldMode.VETO_TERMINATE:
        return ShieldDecision(
            mode=mode,
            proposal=proposal,
            executed=None,
            intervention="veto_terminate",
            reason=reason,
            proposed_predicted_peak_force_n=float(
                proposal_assessment.predicted_peak_force_n
            ),
            executed_predicted_peak_force_n=None,
            terminate=True,
        )

    if substitute is None or substitute_assessment is None:
        raise ShieldContractError("substitute-only mode requires an assessed substitute")
    if substitute.interface != proposal.interface:
        raise ShieldContractError("shield substitution may not change the authority interface")
    _validate_information_access(substitute_assessment, residual_access)
    substitute_reason = _unsafe_reason(
        substitute_assessment,
        predicted_force_limit_n=predicted_force_limit_n,
    )
    if substitute_reason is not None:
        raise ShieldContractError(f"substitute remains inadmissible: {substitute_reason}")
    return ShieldDecision(
        mode=mode,
        proposal=proposal,
        executed=substitute,
        intervention="substitute",
        reason=reason,
        proposed_predicted_peak_force_n=float(proposal_assessment.predicted_peak_force_n),
        executed_predicted_peak_force_n=float(
            substitute_assessment.predicted_peak_force_n
        ),
        terminate=False,
    )
