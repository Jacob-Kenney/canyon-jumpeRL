"""The reference heuristic policy: jump when a hazard is about to reach the ship. Works on the Box(33) observation.

Same logic as harbor/canyon-racer-task/solution/play.py (which must stay standalone for the agent container). It is stateful:
once triggered it holds A for ``hold`` steps.
"""
from __future__ import annotations

import numpy as np

from .env import BUTTON_ORDER, N_BUTTONS, OBS_LAYOUT

A_IDX = BUTTON_ORDER.index("a")


class HeuristicPolicy:
    def __init__(self, th: float = 30.0, hold: int = 2):
        self.th, self.hold_steps, self._hold = th, hold, 0

    def reset(self):
        self._hold = 0

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        act = np.zeros(N_BUTTONS, np.float32)
        if self._hold > 0:
            self._hold -= 1
            act[A_IDX] = 1.0
            return act
        z, crashing = obs[2], obs[OBS_LAYOUT["crashing"]]
        if z < 1e-6 and crashing == 0:  # on the ground and not crashing
            for o in obs[OBS_LAYOUT["obstacles"]].reshape(-1, 9):
                dy = float(o[8]) * 100  # ship_y - obstacle_y in pixels
                if o[0] and o[5] == 0 and self.th - 1e-3 <= dy <= self.th + 8 + 1e-3:  # o[5] = +1 item: ignore
                    self._hold = self.hold_steps - 1
                    act[A_IDX] = 1.0
                    return act
        return act
