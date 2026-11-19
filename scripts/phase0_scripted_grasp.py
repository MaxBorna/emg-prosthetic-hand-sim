"""Phase 0: load the gripper, run a scripted grasp, and report if it lifted.

Headless (works anywhere):
    python scripts/phase0_scripted_grasp.py

With the interactive viewer (macOS needs `mjpython` instead of `python`):
    mjpython scripts/phase0_scripted_grasp.py --viewer
"""

from __future__ import annotations

import argparse
import time

from handsim.control import ScriptedGrasp
from handsim.sim import GripperSim

LIFT_SUCCESS_M = 0.05  # object must rise at least this much


def run(viewer: bool = False) -> bool:
    sim = GripperSim()
    policy = ScriptedGrasp()
    z0 = sim.object_pos()[2]

    def loop(sync=None):
        while sim.time < policy.duration:
            sim.set_command(*policy(sim.time))
            sim.step()
            if sync is not None:
                sync()
                time.sleep(sim.dt)  # roughly real time

    if viewer:
        import mujoco.viewer

        with mujoco.viewer.launch_passive(sim.model, sim.data) as v:
            loop(v.sync)
    else:
        loop()

    dz = sim.object_pos()[2] - z0
    ok = dz > LIFT_SUCCESS_M
    print(f"object lifted {dz * 100:.1f} cm, grip force {sim.grip_force():.2f} -> "
          f"{'SUCCESS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--viewer", action="store_true", help="open the MuJoCo viewer")
    args = ap.parse_args()
    raise SystemExit(0 if run(args.viewer) else 1)
