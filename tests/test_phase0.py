"""Smoke tests: the model compiles and the scripted grasp lifts the object."""

import numpy as np

from handsim.control import Intent, ScriptedGrasp
from handsim.sim import GripperSim


def test_model_loads():
    sim = GripperSim()
    assert sim.model.nu == 2
    assert np.isfinite(sim.object_pos()).all()


def test_scripted_grasp_lifts_object():
    sim = GripperSim()
    policy = ScriptedGrasp()
    z0 = sim.object_pos()[2]
    while sim.time < policy.duration:
        sim.set_command(*policy(sim.time))
        sim.step()
    assert sim.object_pos()[2] - z0 > 0.05


def test_open_gripper_does_not_lift():
    """Negative control: lifting without closing must leave the object behind."""
    sim = GripperSim()
    policy = ScriptedGrasp(grip_closed=0.0)
    z0 = sim.object_pos()[2]
    while sim.time < policy.duration:
        sim.set_command(*policy(sim.time))
        sim.step()
    assert abs(sim.object_pos()[2] - z0) < 0.01


def test_intent_validation():
    Intent("power", 0.5)
    for bad in ({"grasp": "fist"}, {"grasp": "power", "level": 1.5}):
        try:
            Intent(**bad)
        except ValueError:
            continue
        raise AssertionError(f"Intent accepted {bad}")
