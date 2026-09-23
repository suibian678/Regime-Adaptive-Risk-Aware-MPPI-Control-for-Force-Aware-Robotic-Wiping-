"""Regime-adaptive risk-aware MPPI for the V19 ForceWipe method study.

The EMA actor is used as the MPPI proposal distribution in every regime.  It
is not selected by the historical 0.85 target-force switch.  A single planner
changes its objective, sampling budget, and exploration scale using causal
force-regime memberships and validation-calibrated prediction radii.
"""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn.functional as F

from tdmpc2.tdmpc2 import TDMPC2
from forcewipe_v15.actor_anchored_mppi import (
    ActorAnchoredTDMPC2,
    configure_actor_anchored_mppi,
)
from forcewipe_v19.calibration import select_target_radius
from forcewipe_v19.force_conditioned_world_model import _install_force_conditioned_model
from forcewipe_v19.risk_conditioning import (
    PlannerBudget,
    RiskConditionerConfig,
    RiskWeights,
    choose_planner_budget,
    compute_regime_membership,
    compute_risk_weights,
)


SUPPORTED_TARGETS_N = (5.0, 8.0, 12.0)


def compose_upper_prediction(
    predicted_force,
    predicted_envelope,
    residual_radius,
    *,
    mode: str,
):
    """Compose the three predeclared transient-risk representations."""

    if mode == "force_only":
        return predicted_force
    upper = (
        torch.maximum(predicted_force, predicted_envelope)
        if torch.is_tensor(predicted_force)
        else max(predicted_force, predicted_envelope)
    )
    if mode == "point_envelope":
        return upper
    if mode == "calibrated":
        return upper + residual_radius
    raise ValueError("unknown risk_envelope_mode")


def configure_regime_adaptive_risk_mppi(
    cfg,
    *,
    calibrated_radii_n: dict[float, float],
    adaptive_objective: bool = True,
    adaptive_compute: bool = True,
    risk_envelope_mode: str = "calibrated",
):
    if risk_envelope_mode not in {"force_only", "point_envelope", "calibrated"}:
        raise ValueError("unknown risk_envelope_mode")
    cfg = configure_actor_anchored_mppi(cfg)
    cfg.force_plan = True
    cfg.force_plan_deadband = 0.0
    # Disable the fixed base penalty.  V19 applies one regime-conditioned
    # tracking term exactly once in _estimate_value.
    cfg.force_plan_coef = 0.0
    cfg.ra_force_obs_idx = int(getattr(cfg, "force_obs_idx", 0))
    cfg.ra_target_obs_idx = int(getattr(cfg, "force_plan_target_obs_idx", 1))
    cfg.ra_force_rate_obs_idx = 2
    cfg.ra_force_scale_n = 15.0
    cfg.ra_force_rate_scale_n_s = 300.0
    cfg.ra_contact_floor_n = 3.0
    cfg.ra_soft_force_cap_n = 14.5
    cfg.ra_hard_force_cap_n = 15.0
    cfg.ra_unsafe_penalty = 1_000.0
    cfg.ra_base_contact_coef = 8.0
    cfg.ra_base_tracking_coef = 6.0
    cfg.ra_base_transient_coef = 100.0
    cfg.ra_base_anchor_coef = 1.0
    cfg.ra_calibration_coverage = 0.90
    cfg.ra_uncertainty_targets_n = list(SUPPORTED_TARGETS_N)
    cfg.ra_uncertainty_radii_n = [
        select_target_radius(target, calibrated_radii_n)
        for target in SUPPORTED_TARGETS_N
    ]
    cfg.ra_uncertainty_calibrated = True
    cfg.ra_adaptive_objective = bool(adaptive_objective)
    cfg.ra_adaptive_compute = bool(adaptive_compute)
    cfg.ra_risk_envelope_mode = str(risk_envelope_mode)
    # These frozen settings define the matched non-adaptive ablation.  The
    # four method arms therefore differ by two explicit switches rather than
    # by separate planner implementations.
    cfg.ra_fixed_contact_coef = 8.0
    cfg.ra_fixed_tracking_coef = 6.0
    cfg.ra_fixed_transient_coef = 100.0
    cfg.ra_fixed_anchor_coef = 1.0
    cfg.ra_fixed_num_samples = 128
    cfg.ra_fixed_num_elites = 16
    cfg.ra_fixed_iterations = 4
    cfg.ra_fixed_max_std = 0.25
    cfg.mppi_actor_deviation_coef = 0.0
    cfg.exp_name = "v19-regime-adaptive-risk-aware-mppi"
    return cfg


@contextmanager
def _temporary_config(cfg, **updates):
    previous = {name: getattr(cfg, name) for name in updates}
    try:
        for name, value in updates.items():
            setattr(cfg, name, value)
        yield
    finally:
        for name, value in previous.items():
            setattr(cfg, name, value)


class RegimeAdaptiveRiskAwareTDMPC2(ActorAnchoredTDMPC2):
    """One MPPI controller with causal regime-conditioned cost and compute."""

    def __init__(self, cfg):
        if not bool(getattr(cfg, "ra_uncertainty_calibrated", False)):
            raise ValueError("V19 requires training/validation-calibrated radii")
        if not bool(getattr(cfg, "force_pred", False)):
            raise ValueError("V19 requires the learned next-force head")
        if not bool(getattr(cfg, "envelope_pred", False)):
            raise ValueError("V19 requires the learned transient-envelope head")
        super().__init__(cfg)
        self._had_contact = False
        self._last_regime_membership = None
        self._last_risk_weights = None
        self._last_planner_budget = None
        self._last_calibrated_upper_force_n = None

    def _radii_by_target(self) -> dict[float, float]:
        return {
            float(target): float(radius)
            for target, radius in zip(
                self.cfg.ra_uncertainty_targets_n,
                self.cfg.ra_uncertainty_radii_n,
            )
        }

    def _upper_prediction(self, predicted_force, predicted_envelope, target_force_n: float):
        mode = str(self.cfg.ra_risk_envelope_mode)
        radius = select_target_radius(target_force_n, self._radii_by_target())
        if torch.is_tensor(predicted_force):
            radius = radius / float(self.cfg.ra_force_scale_n)
        return compose_upper_prediction(
            predicted_force, predicted_envelope, radius, mode=mode
        )

    @torch.no_grad()
    def _actor_root_upper_force_n(self, obs, task, target_force_n: float) -> float:
        model = self._eval_ema_model if self._eval_ema_model is not None else self.model
        z = model.encode(obs, task)
        _, info = model.pi(z, task)
        action = info["mean"]
        force_scale = float(self.cfg.ra_force_scale_n)
        force_n = float(model.force(z, action, task).reshape(-1)[0].detach().cpu()) * force_scale
        envelope_n = (
            float(model.envelope(z, action, task).reshape(-1)[0].detach().cpu())
            * force_scale
        )
        return float(self._upper_prediction(force_n, envelope_n, target_force_n))

    @torch.no_grad()
    def _plan(self, obs, t0=False, eval_mode=False, task=None):
        if t0:
            self._had_contact = False
        force_scale = float(self.cfg.ra_force_scale_n)
        rate_scale = float(self.cfg.ra_force_rate_scale_n_s)
        force_n = float(obs[0, int(self.cfg.ra_force_obs_idx)].detach().cpu()) * force_scale
        target_force_n = (
            float(obs[0, int(self.cfg.ra_target_obs_idx)].detach().cpu()) * force_scale
        )
        force_rate_n_s = (
            float(obs[0, int(self.cfg.ra_force_rate_obs_idx)].detach().cpu()) * rate_scale
        )
        predicted_upper_n = self._actor_root_upper_force_n(obs, task, target_force_n)
        conditioner = RiskConditionerConfig(
            contact_floor_n=float(self.cfg.ra_contact_floor_n),
            soft_force_cap_n=float(self.cfg.ra_soft_force_cap_n),
            hard_force_cap_n=float(self.cfg.ra_hard_force_cap_n),
            base_contact_coef=float(self.cfg.ra_base_contact_coef),
            base_tracking_coef=float(self.cfg.ra_base_tracking_coef),
            base_transient_coef=float(self.cfg.ra_base_transient_coef),
            base_anchor_coef=float(self.cfg.ra_base_anchor_coef),
        )
        membership = compute_regime_membership(
            force_n=force_n,
            target_force_n=target_force_n,
            force_rate_n_s=force_rate_n_s,
            predicted_upper_force_n=predicted_upper_n,
            had_contact=self._had_contact,
            cfg=conditioner,
        )
        if bool(self.cfg.ra_adaptive_objective):
            weights = compute_risk_weights(membership, cfg=conditioner)
        else:
            weights = RiskWeights(
                contact=float(self.cfg.ra_fixed_contact_coef),
                tracking=float(self.cfg.ra_fixed_tracking_coef),
                transient=float(self.cfg.ra_fixed_transient_coef),
                actor_anchor=float(self.cfg.ra_fixed_anchor_coef),
                risk_score=float(membership.transient),
            )
        if bool(self.cfg.ra_adaptive_compute):
            budget = choose_planner_budget(membership)
        else:
            budget = PlannerBudget(
                "fixed",
                int(self.cfg.ra_fixed_num_samples),
                int(self.cfg.ra_fixed_num_elites),
                int(self.cfg.ra_fixed_iterations),
                float(self.cfg.ra_fixed_max_std),
            )
        self._last_regime_membership = membership
        self._last_risk_weights = weights
        self._last_planner_budget = budget
        self._last_calibrated_upper_force_n = predicted_upper_n

        with _temporary_config(
            self.cfg,
            num_samples=budget.num_samples,
            num_elites=budget.num_elites,
            iterations=budget.iterations,
            max_std=budget.max_std,
        ):
            action = super()._plan(obs, t0=t0, eval_mode=eval_mode, task=task)
        if force_n >= float(self.cfg.ra_contact_floor_n):
            self._had_contact = True
        return action

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
        if force_target is None or self._last_risk_weights is None:
            raise RuntimeError("V19 risk state must be prepared before trajectory scoring")
        model = self.model if model is None else model
        with _temporary_config(self.cfg, force_plan=False, mppi_force_hard_cap=-1.0):
            value = TDMPC2._estimate_value(
                self,
                z,
                actions,
                task,
                force_target=force_target,
                model=model,
            )

        force_scale = float(self.cfg.ra_force_scale_n)
        target_force_n = float(force_target[0, 0].detach().cpu()) * force_scale
        contact_floor = float(self.cfg.ra_contact_floor_n) / force_scale
        soft_cap = float(self.cfg.ra_soft_force_cap_n) / force_scale
        hard_cap = float(self.cfg.ra_hard_force_cap_n) / force_scale
        discount = torch.as_tensor(float(self.discount), dtype=z.dtype, device=z.device)
        weight = torch.ones((), dtype=z.dtype, device=z.device)
        rollout_z = z
        tracking_cost = torch.zeros(actions.shape[1], 1, dtype=z.dtype, device=z.device)
        contact_cost = torch.zeros_like(tracking_cost)
        transient_cost = torch.zeros_like(tracking_cost)
        unsafe = torch.zeros(actions.shape[1], 1, dtype=torch.bool, device=z.device)
        for step in range(actions.shape[0]):
            predicted_force = model.force(rollout_z, actions[step], task)
            predicted_envelope = model.envelope(rollout_z, actions[step], task)
            calibrated_upper = self._upper_prediction(
                predicted_force, predicted_envelope, target_force_n
            )
            tracking_cost = tracking_cost + weight * F.smooth_l1_loss(
                predicted_force,
                force_target,
                reduction="none",
            )
            contact_cost = contact_cost + weight * torch.relu(
                contact_floor - predicted_force
            ).square()
            transient_cost = transient_cost + weight * torch.relu(
                calibrated_upper - soft_cap
            ).square()
            unsafe = unsafe | (calibrated_upper >= hard_cap)
            rollout_z = model.next(rollout_z, actions[step], task)
            weight = weight * discount

        anchor_cost = torch.zeros_like(value)
        if self._anchor_actor_centers is not None:
            deviation = actions - self._anchor_actor_centers.unsqueeze(1)
            time_weights = torch.pow(
                discount,
                torch.arange(actions.shape[0], device=actions.device, dtype=actions.dtype),
            ).view(-1, 1)
            anchor_cost = (
                deviation.square().sum(dim=-1) * time_weights
            ).sum(dim=0).unsqueeze(-1)

        weights = self._last_risk_weights
        return (
            value
            - float(weights.tracking) * tracking_cost
            - float(weights.contact) * contact_cost
            - float(weights.transient) * transient_cost
            - float(weights.actor_anchor) * anchor_cost
            - float(self.cfg.ra_unsafe_penalty) * unsafe.float()
        )

    def risk_diagnostics(self) -> dict:
        """Return the latest causal regime, objective and compute decision."""

        return {
            "membership": (
                None
                if self._last_regime_membership is None
                else self._last_regime_membership.__dict__.copy()
            ),
            "weights": (
                None if self._last_risk_weights is None else self._last_risk_weights.__dict__.copy()
            ),
            "budget": (
                None if self._last_planner_budget is None else self._last_planner_budget.__dict__.copy()
            ),
            "calibrated_upper_force_n": self._last_calibrated_upper_force_n,
            "risk_envelope_mode": str(self.cfg.ra_risk_envelope_mode),
            "historical_contact_established": bool(self._had_contact),
        }


class ForceConditionedRegimeAdaptiveRiskAwareTDMPC2(
    RegimeAdaptiveRiskAwareTDMPC2
):
    """Full V19 agent: explicit force conditioning plus adaptive RA-MPPI."""

    def __init__(self, cfg):
        if not bool(getattr(cfg, "force_conditioned_dynamics", False)):
            raise ValueError("full V19 agent requires force-conditioned dynamics")
        super().__init__(cfg)
        _install_force_conditioned_model(self)
