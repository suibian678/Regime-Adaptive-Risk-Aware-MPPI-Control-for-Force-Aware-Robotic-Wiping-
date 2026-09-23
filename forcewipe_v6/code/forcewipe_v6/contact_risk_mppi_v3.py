"""TD-MPC2 actor acquisition followed by risk-gated TD-MPC2 MPPI contact control."""

from __future__ import annotations

import torch

from forcewipe_v6.contact_risk_mppi import ContactRiskTDMPC2


class ActorAcquireContactRiskTDMPC2(ContactRiskTDMPC2):
    """Use the learned actor below contact threshold and learned MPPI in contact."""

    def __init__(self, cfg, evaluator, *, contact_threshold_n: float = 3.0):
        super().__init__(cfg, evaluator)
        if contact_threshold_n <= 0:
            raise ValueError("contact threshold must be positive")
        self.contact_threshold_n = float(contact_threshold_n)
        self._contact_mppi_started = False
        self._last_action_source = "uninitialized"

    @torch.no_grad()
    def act(self, obs, t0=False, eval_mode=False, task=None):
        observation = torch.as_tensor(obs, dtype=torch.float32)
        force_index = int(self.cfg.force_obs_idx)
        current_force_n = float(observation[force_index]) * self.contact_risk_evaluator.model.config.force_scale_n
        if current_force_n < self.contact_threshold_n:
            self._contact_mppi_started = False
            self._contact_risk_plan_rows = []
            model = (
                self._eval_ema_model
                if eval_mode and self.cfg.eval_ema and self._eval_ema_model is not None
                else self.model
            )
            encoded = model.encode(observation.to(self.device).unsqueeze(0), task=None)
            action, info = model.pi(encoded, task=None)
            selected = info["mean"] if eval_mode else action
            self._last_action_source = "learned_actor_acquisition"
            return selected[0].cpu().clamp(-1, 1)
        first_contact_plan = not self._contact_mppi_started
        self._contact_mppi_started = True
        self._last_action_source = "risk_gated_mppi_contact"
        return super().act(
            observation,
            t0=bool(t0 or first_contact_plan),
            eval_mode=eval_mode,
            task=task,
        )

    @property
    def last_action_source(self) -> str:
        return self._last_action_source
