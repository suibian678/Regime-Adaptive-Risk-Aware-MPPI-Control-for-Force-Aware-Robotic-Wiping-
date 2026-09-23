"""Contact-conditioned MPPI search support for direct TD-MPC2."""

from __future__ import annotations

import torch

from tdmpc2.tdmpc2 import TDMPC2
from forcewipe_v15.actor_anchored_mppi import clip_to_actor_anchor
from forcewipe_v16.force_regime_tracking_mppi import (
    ForceRegimeTrackingMPPITDMPC2,
    configure_force_regime_tracking_mppi,
)


CONTACT_THRESHOLD_FRACTION = 3.0 / 15.0
RECOVERY_LOWER_DEVIATION = (-0.25, -0.10, -0.06)
RECOVERY_UPPER_DEVIATION = (0.02, 0.10, 0.40)


def configure_contact_recovery_search_mppi(cfg):
    cfg = configure_force_regime_tracking_mppi(cfg)
    cfg.mppi_contact_recovery_threshold = CONTACT_THRESHOLD_FRACTION
    cfg.mppi_contact_recovery_lower_deviation = list(RECOVERY_LOWER_DEVIATION)
    cfg.mppi_contact_recovery_upper_deviation = list(RECOVERY_UPPER_DEVIATION)
    return cfg


def contact_recovery_search_active(
    current_force_fraction: float, target_force_fraction: float
) -> bool:
    return bool(
        target_force_fraction < 0.70
        and current_force_fraction < CONTACT_THRESHOLD_FRACTION
    )


class ContactRecoverySearchMPPITDMPC2(ForceRegimeTrackingMPPITDMPC2):
    """Expand only MPPI's normal candidate support during measured contact loss."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._contact_recovery_lower_deviation = torch.as_tensor(
            cfg.mppi_contact_recovery_lower_deviation,
            dtype=torch.float32,
            device=self.device,
        )
        self._contact_recovery_upper_deviation = torch.as_tensor(
            cfg.mppi_contact_recovery_upper_deviation,
            dtype=torch.float32,
            device=self.device,
        )

    @torch.no_grad()
    def _apply_mppi_action_safety(self, actions, obs=None):
        if obs is None:
            return super()._apply_mppi_action_safety(actions, obs)
        current_force = float(obs[0, int(self.cfg.force_obs_idx)].detach().cpu())
        target_force = float(
            obs[0, int(self.cfg.force_plan_target_obs_idx)].detach().cpu()
        )
        if not contact_recovery_search_active(current_force, target_force):
            return super()._apply_mppi_action_safety(actions, obs)

        actions = TDMPC2._apply_mppi_action_safety(self, actions, obs)
        if self._anchor_actor_centers is None:
            self._anchor_actor_centers = self._actor_centers(obs, task=None)
        anchored = clip_to_actor_anchor(
            actions,
            self._anchor_actor_centers,
            self._contact_recovery_lower_deviation,
            self._contact_recovery_upper_deviation,
        )
        if actions.ndim == 1:
            self._last_selected_preanchor_action = actions.detach().clone()
            self._last_selected_postanchor_action = anchored.detach().clone()
            self._last_selected_actor_center = self._anchor_actor_centers[0].detach().clone()
        return anchored

