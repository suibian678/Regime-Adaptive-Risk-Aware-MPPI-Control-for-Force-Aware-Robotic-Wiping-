"""Boundary-risk utilities for the direct TD-MPC2 target-12 curriculum.

All functions in this module are training-only.  They do not alter the direct
environment action contract or apply a runtime action projection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class BoundaryRiskConfig:
    prediction_horizon_steps: int = 20
    precursor_window_steps: int = 50
    boundary_force_n: float = 11.5
    internal_headroom_n: float = 14.5
    hard_limit_n: float = 15.0
    quantile: float = 0.95

    def validate(self) -> None:
        if self.prediction_horizon_steps < 1:
            raise ValueError("prediction horizon must be positive")
        if self.precursor_window_steps < self.prediction_horizon_steps:
            raise ValueError("precursor window must cover the prediction horizon")
        if not 0.5 < self.quantile < 1.0:
            raise ValueError("quantile must lie strictly between 0.5 and 1")
        if not 0.0 < self.boundary_force_n < self.internal_headroom_n:
            raise ValueError("boundary force must be below the internal headroom")
        if not self.internal_headroom_n < self.hard_limit_n:
            raise ValueError("internal headroom must be below the hard limit")


def future_max_force_labels(
    post_action_forces_n: object,
    horizon_steps: int,
) -> np.ndarray:
    """Return the observed future maximum force for every behavior action.

    Entry ``t`` is the maximum over post-action forces ``t`` through
    ``t+horizon_steps-1``.  Near an episode end the available suffix is used;
    there is no padding with fabricated measurements.
    """

    forces = np.asarray(post_action_forces_n, dtype=np.float32).reshape(-1)
    if forces.size < 1 or not np.all(np.isfinite(forces)):
        raise ValueError("forces must be one nonempty finite vector")
    if horizon_steps < 1:
        raise ValueError("horizon_steps must be positive")
    labels = np.empty_like(forces)
    for index in range(len(forces)):
        labels[index] = np.max(forces[index : index + horizon_steps])
    return labels


def precursor_slice(length: int, window_steps: int) -> slice:
    """Return the final behavior window including one leading observation."""

    if length < 2:
        raise ValueError("episode TensorDict must contain at least one transition")
    if window_steps < 1:
        raise ValueError("window_steps must be positive")
    transitions = length - 1
    start_transition = max(0, transitions - window_steps)
    return slice(start_transition, length)


def boundary_mask(
    current_forces_n: object,
    future_max_forces_n: object,
    config: BoundaryRiskConfig,
) -> np.ndarray:
    """Identify current high-force or future near-limit precursor states."""

    config.validate()
    current = np.asarray(current_forces_n, dtype=np.float32).reshape(-1)
    future = np.asarray(future_max_forces_n, dtype=np.float32).reshape(-1)
    if current.shape != future.shape or current.size < 1:
        raise ValueError("current and future force vectors must align")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(future)):
        raise ValueError("force vectors must be finite")
    return (current >= config.boundary_force_n) | (
        future >= config.internal_headroom_n
    )


def quantile_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    quantile: float,
) -> torch.Tensor:
    """Pinball loss for a conservative upper force-envelope estimate."""

    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if not 0.5 < float(quantile) < 1.0:
        raise ValueError("quantile must lie strictly between 0.5 and 1")
    error = target - prediction
    return torch.maximum(quantile * error, (quantile - 1.0) * error).mean()


def actor_risk_loss(
    predicted_envelope_normalized: torch.Tensor,
    *,
    internal_headroom_n: float,
    force_limit_n: float,
) -> torch.Tensor:
    """Squared upper-envelope hinge used only during actor training."""

    if not 0.0 < internal_headroom_n < force_limit_n:
        raise ValueError("headroom must lie inside the force limit")
    threshold = float(internal_headroom_n) / float(force_limit_n)
    return F.relu(predicted_envelope_normalized - threshold).square().mean()


def select_counterfactual_candidate(
    candidate_actions: object,
    future_peak_forces_n: object,
    progress_gains: object,
    *,
    internal_headroom_n: float,
) -> tuple[int, bool]:
    """Select the fastest safe branch, or the lowest-peak branch if none is safe."""

    actions = np.asarray(candidate_actions, dtype=np.float32)
    peaks = np.asarray(future_peak_forces_n, dtype=np.float32).reshape(-1)
    gains = np.asarray(progress_gains, dtype=np.float32).reshape(-1)
    if actions.ndim != 2 or actions.shape[1] != 3:
        raise ValueError("candidate actions must be N-by-3")
    if len(actions) != len(peaks) or len(actions) != len(gains) or len(actions) < 1:
        raise ValueError("candidate outcome arrays must align")
    if not np.all(np.isfinite(actions)) or not np.all(np.isfinite(peaks)):
        raise ValueError("candidate actions and peaks must be finite")
    safe = np.flatnonzero(peaks <= float(internal_headroom_n))
    if len(safe):
        # Lexicographic: maximum progress, then lower peak, then stable index.
        ranked = sorted(safe.tolist(), key=lambda i: (-gains[i], peaks[i], i))
        return int(ranked[0]), True
    return int(np.argmin(peaks)), False


class BoundaryRiskOptimizer:
    """Separate envelope-quantile and actor-risk optimizers.

    The encoder is treated as a fixed feature map for these auxiliary updates.
    Envelope parameters are frozen during the actor update, so gradients reach
    only the policy through its proposed action.
    """

    def __init__(
        self,
        agent,
        config: BoundaryRiskConfig,
        *,
        envelope_lr: float = 3e-4,
        risk_coefficient: float = 25.0,
        oracle_coefficient: float = 1.0,
    ) -> None:
        config.validate()
        if getattr(agent.model, "_envelope", None) is None:
            raise ValueError("agent must enable the transient-envelope head")
        self.agent = agent
        self.config = config
        self.risk_coefficient = float(risk_coefficient)
        self.oracle_coefficient = float(oracle_coefficient)
        self.envelope_optimizer = torch.optim.Adam(
            agent.model._envelope.parameters(),
            lr=float(envelope_lr),
        )

    def update_envelope(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        future_max_force_n: torch.Tensor,
    ) -> dict[str, float]:
        device = self.agent.device
        obs = observations.to(device)
        action = actions.to(device)
        target = future_max_force_n.to(device).reshape(-1, 1)
        target = target / float(self.config.hard_limit_n)
        with torch.no_grad():
            z = self.agent.model.encode(obs, task=None)
        prediction = self.agent.model.envelope(z, action, task=None)
        loss = quantile_loss(prediction, target, self.config.quantile)
        self.envelope_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.agent.model._envelope.parameters(),
            10.0,
        )
        self.envelope_optimizer.step()
        # The same parameters also appear in the agent's model optimizer.
        # Clear auxiliary gradients immediately so the next TD-MPC2 update
        # cannot apply them a second time.
        self.envelope_optimizer.zero_grad(set_to_none=True)
        under_rate = (prediction.detach() < target).float().mean()
        return {
            "envelope_quantile_loss": float(loss.detach().cpu()),
            "envelope_underprediction_rate": float(under_rate.cpu()),
        }

    def update_actor(
        self,
        observations: torch.Tensor,
        oracle_actions: torch.Tensor,
        oracle_safe: torch.Tensor,
    ) -> dict[str, float]:
        device = self.agent.device
        obs = observations.to(device)
        labels = oracle_actions.to(device)
        safe = oracle_safe.to(device).reshape(-1, 1).float()
        if obs.shape[0] != labels.shape[0] or obs.shape[0] != safe.shape[0]:
            raise ValueError("actor-risk batch columns must align")
        with torch.no_grad():
            z = self.agent.model.encode(obs, task=None)
        _, policy_info = self.agent.model.pi(z, task=None)
        action = policy_info["mean"]
        envelope_parameters = list(self.agent.model._envelope.parameters())
        requires_grad = [parameter.requires_grad for parameter in envelope_parameters]
        for parameter in envelope_parameters:
            parameter.requires_grad_(False)
        prediction = self.agent.model.envelope(z, action, task=None)
        for parameter, original in zip(envelope_parameters, requires_grad):
            parameter.requires_grad_(original)
        risk = actor_risk_loss(
            prediction,
            internal_headroom_n=self.config.internal_headroom_n,
            force_limit_n=self.config.hard_limit_n,
        )
        oracle_squared = (action - labels).square().mean(dim=-1, keepdim=True)
        oracle_loss = (oracle_squared * safe).sum() / safe.sum().clamp(min=1.0)
        loss = self.risk_coefficient * risk + self.oracle_coefficient * oracle_loss
        self.agent.pi_optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.agent.model._pi.parameters(), 10.0)
        self.agent.pi_optim.step()
        self.agent.pi_optim.zero_grad(set_to_none=True)
        return {
            "actor_risk_loss": float(risk.detach().cpu()),
            "actor_oracle_loss": float(oracle_loss.detach().cpu()),
            "actor_risk_total_loss": float(loss.detach().cpu()),
            "oracle_safe_fraction": float(safe.mean().detach().cpu()),
            "predicted_envelope_mean_n": float(
                prediction.detach().mean().cpu() * self.config.hard_limit_n
            ),
        }
