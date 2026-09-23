"""Risk-gated MPPI integration for direct TD-MPC2 contact control."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from tdmpc2 import TDMPC2

from forcewipe_v6.contact_sequence_risk_world_model import ContactSequenceRiskWorldModelEnsemble
from forcewipe_v6.contact_sequence_world_model import ContactSequenceWorldModelConfig


@dataclass(frozen=True)
class ContactRiskMPPIConfig:
    force_upper_score_threshold_n: float = 16.0
    unsafe_penalty: float = 1_000_000.0
    severity_penalty: float = 1_000.0

    def validate(self) -> None:
        if self.force_upper_score_threshold_n <= 0:
            raise ValueError("force-upper score threshold must be positive")
        if self.unsafe_penalty <= 0 or self.severity_penalty < 0:
            raise ValueError("candidate penalties must be nonnegative and the unsafe penalty positive")


def candidate_gate(
    risk_probability: torch.Tensor,
    maximum_force_upper_n: torch.Tensor,
    *,
    risk_threshold: float,
    force_upper_score_threshold_n: float,
) -> torch.Tensor:
    """Return the frozen disjunctive unsafe-candidate decision."""
    if risk_probability.ndim != 1 or maximum_force_upper_n.shape != risk_probability.shape:
        raise ValueError("candidate scores must be matching one-dimensional tensors")
    if not torch.isfinite(risk_probability).all() or not torch.isfinite(maximum_force_upper_n).all():
        raise ValueError("candidate scores must be finite")
    return (risk_probability >= float(risk_threshold)) | (
        maximum_force_upper_n > float(force_upper_score_threshold_n)
    )


class ContactRiskMPPIEvaluator:
    """Evaluate complete MPPI action sequences using the frozen contact model."""

    def __init__(
        self,
        model: ContactSequenceRiskWorldModelEnsemble,
        *,
        risk_threshold: float,
        config: ContactRiskMPPIConfig,
        device: torch.device | str,
    ):
        config.validate()
        if not 0.0 < risk_threshold < 1.0:
            raise ValueError("risk threshold must be strictly between zero and one")
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.risk_threshold = float(risk_threshold)
        self.config = config
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Path | str,
        *,
        config: ContactRiskMPPIConfig,
        device: torch.device | str,
    ) -> "ContactRiskMPPIEvaluator":
        checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
        model_config = ContactSequenceWorldModelConfig(**checkpoint["model_config"])
        model = ContactSequenceRiskWorldModelEnsemble(
            model_config,
            members=int(checkpoint["ensemble_members"]),
        )
        model.load_state_dict(checkpoint["model"])
        return cls(
            model,
            risk_threshold=float(checkpoint["window_risk_threshold"]),
            config=config,
            device=device,
        )

    @torch.no_grad()
    def evaluate_candidates(
        self,
        observation: torch.Tensor,
        actions_nha: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
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
            "unsafe_severity": severity,
        }

    def penalize_values(
        self,
        values: torch.Tensor,
        evaluation: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if values.ndim not in (1, 2) or values.shape[0] != len(evaluation["unsafe"]):
            raise ValueError("candidate value shape does not match risk evaluation")
        column = values.ndim == 2
        unsafe = evaluation["unsafe"].to(values.device)
        severity = evaluation["unsafe_severity"].to(values.device)
        penalty = unsafe.float() * (
            self.config.unsafe_penalty + self.config.severity_penalty * severity
        )
        return values - (penalty.unsqueeze(1) if column else penalty)


class ContactRiskTDMPC2(TDMPC2):
    """TD-MPC2 whose MPPI candidates are scored by the contact-risk model."""

    def __init__(self, cfg, evaluator: ContactRiskMPPIEvaluator):
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
        value = super()._estimate_value(
            z,
            actions,
            task,
            force_target=force_target,
            model=model,
        )
        if self._contact_risk_observation is None:
            raise RuntimeError("contact-risk MPPI evaluation lacks the causal current observation")
        evaluation = self.contact_risk_evaluator.evaluate_candidates(
            self._contact_risk_observation,
            actions.transpose(0, 1).contiguous(),
        )
        row = {
                "candidate_count": int(actions.shape[1]),
                "unsafe_candidate_count": int(evaluation["unsafe"].sum()),
                "minimum_risk_probability": float(evaluation["risk_probability"].min()),
                "maximum_risk_probability": float(evaluation["risk_probability"].max()),
                "minimum_force_upper_n": float(evaluation["maximum_force_upper_n"].min()),
                "maximum_force_upper_n": float(evaluation["maximum_force_upper_n"].max()),
            }
        self._contact_risk_plan_rows.append(row)
        self._contact_risk_episode_rows.append(dict(row))
        return self.contact_risk_evaluator.penalize_values(value, evaluation)

    def contact_risk_plan_diagnostics(self) -> tuple[dict, ...]:
        return tuple(dict(row) for row in self._contact_risk_plan_rows)

    def reset_contact_risk_episode_diagnostics(self) -> None:
        self._contact_risk_episode_rows = []

    def contact_risk_episode_diagnostics(self) -> tuple[dict, ...]:
        return tuple(dict(row) for row in self._contact_risk_episode_rows)
