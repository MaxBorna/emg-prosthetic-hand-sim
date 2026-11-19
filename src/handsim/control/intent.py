"""The interface between EMG decoding and control.

Every decoder (LDA, SVM, CNN, ...) outputs an `Intent`; every controller
consumes one. Keeping this contract fixed lets either side be swapped
independently, including replacing the simulation with real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass

GRASPS = ("rest", "power", "pinch")


@dataclass(frozen=True)
class Intent:
    grasp: str = "rest"
    level: float = 1.0  # 0..1, grip intensity; fixed at 1.0 until Phase 4

    def __post_init__(self) -> None:
        if self.grasp not in GRASPS:
            raise ValueError(f"unknown grasp {self.grasp!r}, expected one of {GRASPS}")
        if not 0.0 <= self.level <= 1.0:
            raise ValueError(f"level must be in [0, 1], got {self.level}")
