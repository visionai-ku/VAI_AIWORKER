"""Exercise the actual notebook transmission blocks without GUI or sockets."""
import ast
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from package.worker.fixed_quest_protocol import JOINT_NAMES, make_joint_groups


def transmission_block(kind):
    path = (ROOT.parent / "ri_motion_v5_VR/project/ffw_sg2_vr_teleoperation/real_notebook"
            / ("real_" + kind + "_controller_sg2_teleop.ipynb"))
    cells = json.loads(path.read_text())["cells"]
    source = next("".join(c["source"]) for c in cells if c["cell_type"] == "code"
                  and "prev_on_flag = False" in "".join(c["source"]))
    node = next(node for node in ast.walk(ast.parse(source)) if isinstance(node, ast.If)
                and ast.unparse(node.test) == "tmr_robot.do_run()")
    return compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])), str(path), "exec")


class ControllerNotebookTest(unittest.TestCase):
    def namespace(self, kind, synchronize, enabled):
        measured = np.zeros(19)
        measured[1:3] = [.69, -.12]
        measured[0] = -.1
        measured[-2:] = [.35, .4]
        names = list(reversed(JOINT_NAMES))  # Exercise named ordering, not a lucky index match.
        result = dict(state="ARMING" if synchronize else "OFF", reason="test",
                      synchronize=synchronize, sent=enabled, q_command=measured)
        namespace = dict(
            np=np, ROBOT_JOINT_NAMES=JOINT_NAMES, command_joint_names=names,
            q_use=np.zeros(19), q_msg_layout={}, on_flag=enabled,
            pub_robot=SimpleNamespace(step=Mock(return_value=result)),
            tmr_robot=SimpleNamespace(do_run=lambda: True, sec_elps_diff=.01, end=Mock()),
            get_qmsg_from_qpos=lambda **kw: make_joint_groups(kw["qpos"][::-1]),
            env=SimpleNamespace(forward=Mock()), controller=SimpleNamespace(sync_from_env=Mock()),
            SG2_HAND_TYPES=("left", "right"), hand_types=("left", "right"),
            SG2_GRIPPER_DRIVER_JOINT_NAMES={"left": "gripper_l_joint1", "right": "gripper_r_joint1"},
            set_sg2_gripper_qpos=Mock(), last_robot_status=None,
            markers={hand: SimpleNamespace(set_T=Mock(), get_T=lambda **kw: np.eye(4))
                     for hand in ("left", "right")},
            get_sg2_wrist_control_T=lambda env, hand: np.eye(4),
            gripper_slider=SimpleNamespace(set_values=Mock()),
            gripper_driver_joint_names=["gripper_l_joint1", "gripper_r_joint1"],
        )
        return namespace, measured

    def test_off_still_polls_client_and_releases_session(self):
        for kind in ("joint", "eef"):
            with self.subTest(kind=kind):
                ns, _ = self.namespace(kind, False, False)
                exec(transmission_block(kind), ns)
                self.assertFalse(ns["pub_robot"].step.call_args.kwargs["enabled"])
                ns["env"].forward.assert_not_called()
                self.assertEqual(ns["robot_pub_hz_actual"], 0.)

    def test_on_synchronizes_actual_joint_order_head_and_grippers(self):
        for kind in ("joint", "eef"):
            with self.subTest(kind=kind):
                ns, measured = self.namespace(kind, True, True)
                exec(transmission_block(kind), ns)
                np.testing.assert_array_equal(ns["q_use"], measured[::-1])
                np.testing.assert_array_equal(ns["env"].forward.call_args.kwargs["q"], measured[::-1])
                self.assertEqual(ns["set_sg2_gripper_qpos"].call_count, 2)
                if kind == "joint":
                    ns["controller"].sync_from_env.assert_called_once()
                else:
                    ns["gripper_slider"].set_values.assert_called_once_with([.35, .4])
                    for marker in ns["markers"].values():
                        marker.set_T.assert_called_once()

    def test_active_preview_is_not_overwritten_without_sync_request(self):
        for kind in ("joint", "eef"):
            with self.subTest(kind=kind):
                ns, _ = self.namespace(kind, False, True)
                before = ns["q_use"].copy()
                exec(transmission_block(kind), ns)
                np.testing.assert_array_equal(ns["q_use"], before)
                # Rendering receives the transport's final paced command, while
                # the controller keeps its independently editable preview.
                np.testing.assert_array_equal(ns["q_robot"], ns["pub_robot"].step.return_value["q_command"][::-1])
                ns["env"].forward.assert_not_called()


if __name__ == "__main__":
    unittest.main()
