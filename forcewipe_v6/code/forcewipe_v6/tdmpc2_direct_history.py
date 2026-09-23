"""Causal observation-history utilities for direct TD-MPC2 control."""

from __future__ import annotations

from collections import deque

import numpy as np


def stack_observation_sequence(observations: object, history_length: int) -> np.ndarray:
    """Return oldest-to-newest causal history for every row in one episode."""

    array = np.asarray(observations, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("observations must be one nonempty T-by-D array")
    if history_length < 1:
        raise ValueError("history_length must be positive")
    if not np.all(np.isfinite(array)):
        raise ValueError("observations must be finite")
    rows = []
    for index in range(len(array)):
        start = max(0, index - history_length + 1)
        window = array[start : index + 1]
        if len(window) < history_length:
            padding = np.repeat(
                array[[0]],
                history_length - len(window),
                axis=0,
            )
            window = np.concatenate((padding, window), axis=0)
        rows.append(window.reshape(-1))
    return np.asarray(rows, dtype=np.float32)


class CausalHistoryEnv:
    """Minimal wrapper that stacks only observations available before action."""

    def __init__(self, env, history_length: int):
        if history_length < 1:
            raise ValueError("history_length must be positive")
        self.env = env
        self.history_length = int(history_length)
        self._history: deque[np.ndarray] = deque(maxlen=self.history_length)

    def _stack(self) -> np.ndarray:
        if len(self._history) != self.history_length:
            raise RuntimeError("history is not initialized")
        return np.concatenate(tuple(self._history), axis=0).astype(np.float32)

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        current = np.asarray(observation, dtype=np.float32).reshape(-1)
        self._history.clear()
        for _ in range(self.history_length):
            self._history.append(current.copy())
        return self._stack(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._history.append(np.asarray(observation, dtype=np.float32).reshape(-1))
        return self._stack(), reward, terminated, truncated, info

    def close(self) -> None:
        self.env.close()
