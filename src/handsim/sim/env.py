"""Thin wrapper around a MuJoCo model of the hand/gripper.

Keeps all name lookups in one place so the controller only talks in terms of
"grip command" and "lift command", not actuator indices.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from handsim.paths import MODELS_DIR

DEFAULT_MODEL = MODELS_DIR / "gripper_v0" / "gripper.xml"


class GripperSim:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._act_grip = self.model.actuator("flexor").id
        self._act_lift = self.model.actuator("lift").id
        self._obj_body = self.model.body("object").id
        self.reset()

    @property
    def dt(self) -> float:
        return self.model.opt.timestep

    @property
    def time(self) -> float:
        return self.data.time

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_command(self, grip: float, lift: float) -> None:
        """grip: tendon length target [rad-sum]; lift: palm height offset [m]."""
        self.data.ctrl[self._act_grip] = grip
        self.data.ctrl[self._act_lift] = lift

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    def object_pos(self) -> np.ndarray:
        return self.data.xpos[self._obj_body].copy()

    def grip_force(self) -> float:
        """Actuator force on the flexor tendon [N·m equivalent]."""
        return float(self.data.actuator_force[self._act_grip])
