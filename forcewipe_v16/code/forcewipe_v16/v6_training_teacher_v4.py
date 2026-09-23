"""Steady-force refinement of the TRAIN-only V6 teacher.

This class is used only to collect TD-MPC2 training trajectories.  It is not
available to the deployed actor or MPPI planner.
"""

from __future__ import annotations

import numpy as np

from forcewipe_v14.v6_training_teacher_v3 import V6TrainingTeacherV3


class V6TrainingTeacherV4(V6TrainingTeacherV3):
    """Preserve V3 tail handling while reducing its 12 N steady-state bias."""

    def act(self, observation: object) -> np.ndarray:
        frame = self._latest(observation)
        action = super().act(frame)
        cfg = self.config
        force_n = float(frame[0] * cfg.force_limit_n)
        target_n = float(frame[1] * cfg.force_limit_n)
        force_rate_n_s = float(frame[2] * 300.0)
        previous_inward = float(frame[11])

        # Lower tiers remain byte-for-byte behaviorally identical to V3.
        if target_n < 10.0:
            return action

        tail_envelope_n = (
            force_n
            + 0.035 * max(force_rate_n_s, 0.0)
            + 2.2 * max(previous_inward, 0.0)
        )
        # V3's tail taper safely completed all observed training episodes, but
        # left most steady samples between 10 and 11 N.  Add a bounded support
        # request only when the measured rate and causal tail estimate are low.
        support_ready = (
            9.0 <= force_n < target_n
            and abs(force_rate_n_s) <= 45.0
            and tail_envelope_n < 11.9
            and float(action[2]) > -0.25
        )
        if support_ready:
            force_error_n = target_n - force_n
            support = float(np.clip(
                0.07 + 0.04 * force_error_n + 0.04 * max(float(action[0]), 0.0),
                0.07,
                0.18,
            ))
            action[2] = max(float(action[2]), support)
        return action.astype(np.float32, copy=False)

