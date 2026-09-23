"""Task-progress refinement of the TRAIN-only V2 tail-aware teacher."""

from __future__ import annotations

import numpy as np

from forcewipe_v14.v6_training_teacher_v2 import V6TrainingTeacherV2


class V6TrainingTeacherV3(V6TrainingTeacherV2):
    """Restore nominal tangential speed after stable 12 N contact is formed."""

    def act(self, observation: object) -> np.ndarray:
        frame = self._latest(observation)
        action = super().act(frame)
        cfg = self.config
        force_n = float(frame[0] * cfg.force_limit_n)
        target_n = float(frame[1] * cfg.force_limit_n)
        force_rate_n_s = float(frame[2] * 300.0)
        previous_inward = float(frame[11])
        tail_envelope_n = (
            force_n
            + 0.035 * max(force_rate_n_s, 0.0)
            + 2.2 * max(previous_inward, 0.0)
        )
        if (
            target_n >= 10.0
            and force_n >= 9.0
            and abs(force_rate_n_s) <= 60.0
            and tail_envelope_n < 12.4
            and float(action[2]) > -0.25
            and float(frame[3]) < 0.995
        ):
            action[0] = cfg.tangential_track_action
        return action.astype(np.float32, copy=False)
