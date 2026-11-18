"""Open the UR5e MuJoCo simulator with python -m vai_robotics.ur5e.sim."""

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np


SCENE = Path(__file__).resolve().parent / "assets" / "scene.xml"


def load_simulation():
    """Load the bundled scene in the same MuJoCo parser used by the SG2."""
    vendor = Path(__file__).resolve().parents[2] / "VR_teleoperation" / "vendor"
    sys.path.insert(0, str(vendor))
    try:
        from ri_motion_v5_package.mujoco_sim import MuJoCoParser
    finally:
        sys.path.remove(str(vendor))

    sim = MuJoCoParser(rel_xml_path=str(SCENE))
    mujoco.mj_resetDataKeyframe(sim.model, sim.data, sim.model.key("home").id)
    mujoco.mj_forward(sim.model, sim.data)
    return sim


def check_motion(sim):
    """Exercise a position actuator for one simulated second without opening a window."""
    joint = sim.data.joint("shoulder_pan_joint")
    start = float(joint.qpos[0])
    target = start + np.deg2rad(15)
    sim.data.actuator("shoulder_pan").ctrl[0] = target
    sim.step(nstep=round(1 / sim.dt))
    measured = float(joint.qpos[0])
    if not np.isfinite(sim.data.qpos).all() or abs(target - measured) > np.deg2rad(1):
        raise RuntimeError(f"Shoulder pan failed to reach {target:.3f} rad (measured {measured:.3f})")
    print(f"UR5e OK | {sim.model.nu} actuators | shoulder pan "
          f"{np.rad2deg(start):.1f} → {np.rad2deg(measured):.1f}° | {sim.data.time:.2f} s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="test joint motion without a viewer")
    args = parser.parse_args()
    sim = load_simulation()
    if args.check:
        check_motion(sim)
    else:
        import glfw
        from mujoco import viewer

        # The macOS viewer queries the primary monitor while constructing its UI.
        if not glfw.init() or not glfw.get_primary_monitor():
            parser.exit(1, "No display available for MuJoCo. Use --check for a headless run.\n")
        viewer.launch(sim.model, sim.data)


if __name__ == "__main__":
    main()
