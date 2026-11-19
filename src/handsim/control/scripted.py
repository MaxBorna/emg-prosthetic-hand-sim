"""Phase 0: open-loop scripted grasp (approach -> close -> lift -> hold)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ramp(t: float, t0: float, t1: float) -> float:
    """0 before t0, 1 after t1, smooth (cosine) in between."""
    if t <= t0:
        return 0.0
    if t >= t1:
        return 1.0
    return 0.5 - 0.5 * np.cos(np.pi * (t - t0) / (t1 - t0))


@dataclass
class ScriptedGrasp:
    grip_closed: float = 5.2  # tendon target (sum of 4 joint angles); beyond contact -> squeeze
    lift_height: float = 0.10  # [m]
    t_close: tuple[float, float] = (0.3, 1.3)
    t_lift: tuple[float, float] = (1.6, 2.6)
    duration: float = 4.0

    def __call__(self, t: float) -> tuple[float, float]:
        grip = self.grip_closed * _ramp(t, *self.t_close)
        lift = self.lift_height * _ramp(t, *self.t_lift)
        return grip, lift
