"""Tail-aware TRAIN-only teacher for 12 N V6 demonstrations.

The additional logic is deliberately absent from deployment.  It creates
safer demonstrations for the rare high-stiffness contact-onset tail that the
first V14 actor did not learn from fifteen trajectories.
"""

from __future__ import annotations

import numpy as np

from forcewipe_v14.v6_training_teacher import V6TrainingTeacher


class V6TrainingTeacherV2(V6TrainingTeacher):
    """V1 behavior plus a causal 12 N approach taper."""

    def act(self, observation: object) -> np.ndarray:
        frame = self._latest(observation)
        action = super().act(frame)
        cfg = self.config
        force_n = float(frame[0] * cfg.force_limit_n)
        target_n = float(frame[1] * cfg.force_limit_n)
        force_rate_n_s = float(frame[2] * 300.0)
        previous_inward = float(frame[11])

        # Preserve the original teacher exactly at the lower two force tiers.
        if target_n < 10.0 or force_n < cfg.contact_threshold_n:
            return action

        positive_rate = max(force_rate_n_s, 0.0)
        positive_previous_inward = max(previous_inward, 0.0)
        tail_envelope_n = (
            force_n
            + 0.035 * positive_rate
            + 2.2 * positive_previous_inward
        )

        # An estimated high-force tail removes tangential authority and commands
        # outward motion.  All terms are available in the current observation.
        if tail_envelope_n >= 12.4:
            outward = float(np.clip(
                0.25 + 0.20 * (tail_envelope_n - 12.4),
                0.25,
                1.0,
            ))
            action[0] = 0.0
            action[2] = min(float(action[2]), -outward)
            self._integral_n_s = min(self._integral_n_s, 0.0)
            return action

        # Before the tail gate, progressively reduce inward authority.  This
        # preserves slow steady-state force correction without repeating the
        # 0.4-scale inward command observed immediately before V14 failures.
        cap = 0.08 + 0.055 * max(target_n - force_n, 0.0)
        cap -= 0.0015 * max(positive_rate - 20.0, 0.0)
        cap = float(np.clip(cap, 0.03, 0.35))
        action[2] = min(float(action[2]), cap)
        if positive_rate >= 80.0:
            action[0] = 0.0
        return action.astype(np.float32, copy=False)
