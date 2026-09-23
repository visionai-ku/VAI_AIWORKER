"""Manually drag both wrist targets in SIM to verify the inference-time safety box.

The real robot's autonomous-policy path (package/policy_deploy.py's step()) FKs
every predicted joint target and checks it with package/safety_box.py's
wrist_safety_box_violations() before sending it to hardware -- on a violation
it holds the last accepted pose instead of applying the new one. This tool
runs the exact same check function, driven by hand instead of by a policy, so
you can watch it hold before ever pointing it at the real robot.

(This is NOT the VR teleop path -- VR teleop's IK does not clip anything.)

No VR headset needed. Opens a MuJoCo window with one draggable gizmo per
wrist, seeded at that wrist's current TCP. Drag either gizmo's arrow as far
as you like -- each tick, a small per-arm IK solves toward it, and the
resulting candidate joint target is only committed to the visible arm if
wrist_safety_box_violations() reports nothing; otherwise the arm freezes at
its last accepted pose while the gizmo keeps moving.

    python scripts/check_safety_box_inference.py
"""
from __future__ import annotations

from pathlib import Path
import runpy
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / "package/init_project.py"))

from ri_motion_v5_package.mujoco_sim import BodyPositionIKSolverRevPriBase, InteractiveMarker, MuJoCoParser
from ri_motion_v5_package.utility import SimpleTimer
from utils import (
    SG2_GRIPPER_DRIVER_JOINT_NAMES, SG2_HAND_TYPES, SG2_REAL_COMMAND_JOINT_NAMES,
    add_sg2_wrist_ik_coordinate_target, create_sg2_arm_ik_solvers, get_sg2_wrist_control_T,
)
from package.fixed_quest_scene import build_fixed_quest_scene
from package.fixed_quest_sim import reset_sg2_sim
from package.safety_box import SAFETY_BOX, wrist_safety_box_violations

VIS_EXTENT = 1.2  # meters; only widens an unbounded side for drawing -- never enforced
HAND_COLORS = {"left": (0.20, 0.55, 1.00, 0.35), "right": (1.00, 0.45, 0.15, 0.35)}


def safety_box_draw_ranges(hand):
    ranges = {}
    for axis in ("x", "y", "z"):
        low, high = SAFETY_BOX[hand][axis]
        ranges[axis] = (-VIS_EXTENT if not np.isfinite(low) else low,
                        VIS_EXTENT if not np.isfinite(high) else high)
    return ranges


def build_sim(cache_dir):
    xml = build_fixed_quest_scene(ROOT / "xml/ffw_sg2.xml", ROOT / "xml/floor_simple_white.xml",
                                  Path(cache_dir) / "scene.xml")
    env, env_ik = [MuJoCoParser(rel_xml_path=xml, verbose=False) for _ in range(2)]
    for model_env in (env, env_ik):
        reset_sg2_sim(model_env)
    names = list(SG2_REAL_COMMAND_JOINT_NAMES) + list(SG2_GRIPPER_DRIVER_JOINT_NAMES.values())
    ik_solvers, _ = create_sg2_arm_ik_solvers(env_ik, MuJoCoParser, BodyPositionIKSolverRevPriBase)
    return env, env_ik, ik_solvers, names


def solve_candidate(env_ik, env, ik_solvers, targets, names):
    """IK env_ik toward `targets` from env's current pose. Returns the candidate q; does not touch env."""
    env_ik.data.qpos[:] = env.data.qpos
    env_ik.forward()
    for hand, T_target in targets.items():
        solver = ik_solvers[hand]
        solver.reset_buffers(env=env_ik)
        add_sg2_wrist_ik_coordinate_target(solver, env_ik, hand, T_target, weight=1.0, axis_width=.10)
        _, ik_info = solver.solve_ik(env=env_ik, joint_limit_handle_flag=True, nullspace_control_flag=True)
        env_ik.forward(q=ik_info["qpos_used_best"], joint_names=solver.joints_use)
    return env_ik.get_qpos(joint_names=names)


def main():
    (ROOT / ".cache").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / ".cache") as cache_dir:
        env, env_ik, ik_solvers, names = build_sim(cache_dir)

        env.init_viewer(title="SG2 Inference Safety Box Check (SIM)", x_offset=0.24, y_offset=0.0,
                        width=0.76, height=1.0, fontscale=180)
        env.viewer.set_cam_info(180.0, 2.8, -20.0, [0.0, 0.0, 0.95])
        env.viewer.set_transparency(transparent=True)

        markers = {
            hand: InteractiveMarker(env=env, T=get_sg2_wrist_control_T(env, hand),
                                    axis_len=0.12, ring_r=0.12, control_mode="p")
            for hand in SG2_HAND_TYPES
        }

        timer = SimpleTimer(name="Inference Safety Box Check", Hz=30, verbose=False)
        timer.start()
        env.reset_wall_time()
        env.reset_sim_time()

        try:
            while env.is_viewer_alive():
                if not timer.do_run():
                    continue

                targets = {hand: marker.get_T(update=True, draw=False) for hand, marker in markers.items()}
                candidate_q = solve_candidate(env_ik, env, ik_solvers, targets, names)
                violations = wrist_safety_box_violations(env_ik, candidate_q, names)
                if not violations:
                    env.forward(q=candidate_q, joint_names=names)

                # A HOLD can strand a marker past the box while the arm stays put.
                # Snap any marker you are not actively dragging back onto the real
                # (committed) wrist, so the next grab always starts from the EEF --
                # same idea as SG2JointMarkerController.reanchor_markers().
                for hand, marker in markers.items():
                    if marker.drag_mode is None:
                        marker.set_T(get_sg2_wrist_control_T(env, hand), clear_state=False)

                lines = []
                for hand in SG2_HAND_TYPES:
                    requested = targets[hand][:3, 3]
                    committed = get_sg2_wrist_control_T(env, hand)[:3, 3]
                    status = "HOLD (safety box)" if hand in violations else "ok"
                    lines.append(
                        "%-5s gizmo x=%+.3f y=%+.3f z=%+.3f | committed x=%+.3f y=%+.3f z=%+.3f | %s"
                        % (hand.upper(), requested[0], requested[1], requested[2],
                           committed[0], committed[1], committed[2], status)
                    )

                env.plot_global_coordinate_axes()
                for hand in SG2_HAND_TYPES:
                    ranges = safety_box_draw_ranges(hand)
                    env.plot_box_from_ranges(x_range=ranges["x"], y_range=ranges["y"], z_range=ranges["z"],
                                             rgba=HAND_COLORS[hand], label="%s safety box" % hand)
                for marker in markers.values():
                    marker.draw()
                env.viewer_text_overlay("Inference Safety Box Check", "\n".join(lines), loc="top left")
                env.render()
                timer.end()
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            env.close_viewer()


if __name__ == "__main__":
    main()
