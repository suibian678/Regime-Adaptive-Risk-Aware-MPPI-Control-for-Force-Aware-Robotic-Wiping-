"""Acquisition-aware risk-gated MPPI for direct TD-MPC2 contact control."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tdmpc2 import TDMPC2

from forcewipe_v6.contact_risk_mppi import (
    ContactRiskMPPIConfig,
    ContactRiskMPPIEvaluator,
    candidate_gate,
)


@dataclass(frozen=True)
class AcquisitionAwareMPPIConfig(ContactRiskMPPIConfig):
    acquisition_force_n: float = 3.0
    no_acquisition_penalty: float = 100_000.0

    def validate(self) -> None:
        super().validate()
        if self.acquisition_force_n <= 0 or self.no_acquisition_penalty <= 0:
            raise ValueError("acquisition threshold and penalty must be positive")
        if self.no_acquisition_penalty >= self.unsafe_penalty:
            raise ValueError("no-acquisition penalty must remain below the unsafe penalty")


class AcquisitionAwareContactRiskMPPIEvaluator(ContactRiskMPPIEvaluator):
    """Expose predicted contact acquisition in addition to the frozen risk rule."""

    @torch.no_grad()
    def evaluate_candidates(self, observation, actions_nha):
        actions = torch.as_tensor(actions_nha, dtype=torch.float32, device=self.device)
        if actions.ndim != 3 or actions.shape[-1] != self.model.config.action_dim:
            raise ValueError("candidate actions must be N-by-H-by-action_dim")
        if actions.shape[1] < 1:
            raise ValueError("candidate horizon must be nonempty")
        obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
        if obs.ndim != 2 or obs.shape[-1] != self.model.config.observation_dim:
            raise ValueError("observation dimension does not match the contact model")
        if obs.shape[0] == 1:
            obs = obs.expand(actions.shape[0], -1)
        elif obs.shape[0] != actions.shape[0]:
            raise ValueError("observation batch must be one or match the candidate count")
        if not torch.isfinite(obs).all() or not torch.isfinite(actions).all():
            raise ValueError("candidate model inputs must be finite")
        prediction = self.model.rollout(obs, actions)
        risk = prediction["window_risk_probability"]
        maximum_upper_n = prediction["force_upper"].max(dim=1).values * self.model.config.force_scale_n
        maximum_mean_n = prediction["force_mean"].max(dim=1).values * self.model.config.force_scale_n
        unsafe = candidate_gate(
            risk,
            maximum_upper_n,
            risk_threshold=self.risk_threshold,
            force_upper_score_threshold_n=self.config.force_upper_score_threshold_n,
        )
        severity = risk + torch.relu(
            (maximum_upper_n - self.config.force_upper_score_threshold_n)
            / self.model.config.force_scale_n
        )
        return {
            "unsafe": unsafe,
            "risk_probability": risk,
            "maximum_force_upper_n": maximum_upper_n,
            "maximum_force_mean_n": maximum_mean_n,
            "unsafe_severity": severity,
        }

    def penalize_values(self, values, evaluation, *, require_acquisition: bool = False):
        adjusted = super().penalize_values(values, evaluation)
        if not require_acquisition:
            return adjusted
        no_acquisition = (
            evaluation["maximum_force_mean_n"]
            < float(self.config.acquisition_force_n)
        ).to(values.device)
        penalty = no_acquisition.float() * float(self.config.no_acquisition_penalty)
        return adjusted - (penalty.unsqueeze(1) if values.ndim == 2 else penalty)


class AcquisitionAwareContactRiskTDMPC2(TDMPC2):
    """TD-MPC2 with learned safety and learned contact-acquisition constraints."""

    def __init__(self, cfg, evaluator: AcquisitionAwareContactRiskMPPIEvaluator):
        super().__init__(cfg)
        self.contact_risk_evaluator = evaluator
        self._contact_risk_observation: torch.Tensor | None = None
        self._contact_risk_plan_rows: list[dict] = []
        self._contact_risk_episode_rows: list[dict] = []

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        self._contact_risk_observation = torch.as_tensor(obs, dtype=torch.float32).detach().clone()
        self._contact_risk_plan_rows = []
        try:
            return super().act(obs, t0=t0, eval_mode=eval_mode, task=task)
        finally:
            self._contact_risk_observation = None

    @torch.no_grad()
    def _estimate_value(self, z, actions, task, force_target=None, model=None):
        value = super()._estimate_value(z, actions, task, force_target=force_target, model=model)
        if self._contact_risk_observation is None:
            raise RuntimeError("contact-risk MPPI evaluation lacks the causal current observation")
        evaluation = self.contact_risk_evaluator.evaluate_candidates(
            self._contact_risk_observation,
            actions.transpose(0, 1).contiguous(),
        )
        force_index = int(self.cfg.force_obs_idx)
        current_force_n = float(self._contact_risk_observation[force_index]) * self.contact_risk_evaluator.model.config.force_scale_n
        require_acquisition = current_force_n < self.contact_risk_evaluator.config.acquisition_force_n
        predicted_acquisition = (
            evaluation["maximum_force_mean_n"]
            >= self.contact_risk_evaluator.config.acquisition_force_n
        )
        row = {
            "candidate_count": int(actions.shape[1]),
            "unsafe_candidate_count": int(evaluation["unsafe"].sum()),
            "acquisition_required": bool(require_acquisition),
            "predicted_acquisition_candidate_count": int(predicted_acquisition.sum()),
            "safe_acquisition_candidate_count": int((predicted_acquisition & ~evaluation["unsafe"]).sum()),
            "minimum_risk_probability": float(evaluation["risk_probability"].min()),
            "maximum_risk_probability": float(evaluation["risk_probability"].max()),
            "minimum_force_upper_n": float(evaluation["maximum_force_upper_n"].min()),
            "maximum_force_upper_n": float(evaluation["maximum_force_upper_n"].max()),
            "minimum_force_mean_n": float(evaluation["maximum_force_mean_n"].min()),
            "maximum_force_mean_n": float(evaluation["maximum_force_mean_n"].max()),
        }
        self._contact_risk_plan_rows.append(row)
        self._contact_risk_episode_rows.append(dict(row))
        return self.contact_risk_evaluator.penalize_values(
            value,
            evaluation,
            require_acquisition=require_acquisition,
        )

    def contact_risk_plan_diagnostics(self):
        return tuple(dict(row) for row in self._contact_risk_plan_rows)

    def reset_contact_risk_episode_diagnostics(self):
        self._contact_risk_episode_rows = []

    def contact_risk_episode_diagnostics(self):
        return tuple(dict(row) for row in self._contact_risk_episode_rows)
