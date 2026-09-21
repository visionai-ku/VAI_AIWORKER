"""Hardware-free SG2 pose IK checks with independent FK and elbow geometry."""

from pathlib import Path
import runpy
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

PROJECT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_DIR / "package/init_project.py"))

from ri_motion_v5_package.mujoco_sim import MuJoCoParser
from utils import SG2_ARM_JOINT_NAMES, get_sg2_wrist_control_T
from package.fixed_quest_scene import build_fixed_quest_scene
from package.fixed_quest_sim import reset_sg2_sim
from package.posture_ik import SG2PostureIK


# Two independently generated configurations of the default wrist pose. The
# second bends the physical elbow to the other side despite keeping q4 < 0.
# A right arm configuration is the reflected left arm joint vector below.
SAME_POSE_LEFT = np.array([
    -.033486115991477, .261113741052260, -.243845780787566,
    -1.558516507198731, .261662620405631, -.041670508147588,
    .232349706495362,
])
OPPOSITE_ELBOW_LEFT = np.array([
    -1.394832703968667, .291274905361460, -2.992665057675578,
    -1.371197526018215, 1.874270466403243, -1.469638042304203,
    .953804558427514,
])
RIGHT_MIRROR = np.array([1., -1., -1., 1., -1., 1., -1.])


class PostureIKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (PROJECT_DIR / ".cache").mkdir(exist_ok=True)
        cls.cache = tempfile.TemporaryDirectory(dir=PROJECT_DIR / ".cache")
        xml = build_fixed_quest_scene(
            PROJECT_DIR / "xml/ffw_sg2.xml",
            PROJECT_DIR / "xml/floor_simple_white.xml",
            Path(cls.cache.name) / "scene.xml",
        )
        cls.env = MuJoCoParser(rel_xml_path=xml, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.cache.cleanup()

    def setUp(self):
        reset_sg2_sim(self.env)
        self.q_reference = self.env.data.qpos.copy()

    def solver(self, hand):
        solver = SG2PostureIK(self.env, hand)
        solver.capture_reference(self.q_reference)
        return solver

    @staticmethod
    def mirrored(left, hand):
        return left.copy() if hand == "left" else left * RIGHT_MIRROR

    def q_arm(self, hand):
        return self.env.get_qpos(joint_names=SG2_ARM_JOINT_NAMES[hand]).copy()

    def set_arm(self, hand, q):
        self.env.forward(q=q, joint_names=SG2_ARM_JOINT_NAMES[hand])

    def elbow_direction(self, hand):
        """Elbow's radial direction about the shoulder-to-controlled-wrist line."""
        side = "l" if hand == "left" else "r"
        shoulder = self.env.get_T("arm_" + side + "_link2", "body")[:3, 3]
        elbow = self.env.get_T("arm_" + side + "_link4", "body")[:3, 3]
        wrist = get_sg2_wrist_control_T(self.env, hand)[:3, 3]
        reach = wrist - shoulder
        reach /= np.linalg.norm(reach)
        radial = elbow - shoulder - reach * np.dot(elbow - shoulder, reach)
        self.assertGreater(np.linalg.norm(radial), .01)
        return radial / np.linalg.norm(radial)

    def assert_pose(self, hand, target, position=.001, rotation=np.deg2rad(.5)):
        actual = get_sg2_wrist_control_T(self.env, hand)
        self.assertLessEqual(np.linalg.norm(actual[:3, 3] - target[:3, 3]), position)
        angle = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).magnitude()
        self.assertLessEqual(angle, rotation)

    def assert_solution(self, hand, target, result):
        self.assertTrue(result["success"], result)
        q = np.asarray(result["qpos_used_best"])
        self.assertEqual(q.shape, (7,))
        self.assertTrue(np.isfinite(q).all())
        ranges = np.array([self.env.model.joint(name).range
                           for name in SG2_ARM_JOINT_NAMES[hand]])
        self.assertTrue(np.all(q >= ranges[:, 0] - 1e-10))
        self.assertTrue(np.all(q <= ranges[:, 1] + 1e-10))
        self.assertLessEqual(result["position_error_m"], .001)
        self.assertLessEqual(result["rotation_error_rad"], np.deg2rad(.5))
        self.assertLessEqual(abs(result["elbow_angle_rad"]), np.deg2rad(85.) + 1e-8)
        self.assertTrue(np.isfinite(result["posture_cost"]))
        self.assertGreaterEqual(result["posture_cost"], 0.)
        self.assertGreaterEqual(result["iterations"], 0)
        self.assertLessEqual(result["iterations"], 24)
        self.assertGreaterEqual(result["elapsed_s"], 0.)
        self.set_arm(hand, q)
        self.assert_pose(hand, target)

    def test_both_arms_reach_full_position_and_orientation_targets(self):
        offsets = (
            [.07, .14, -.09, .11, .12, -.06, .03],
            [-.06, .10, -.05, -.08, -.06, .05, -.03],
        )
        for hand in ("left", "right"):
            for offset in offsets:
                with self.subTest(hand=hand, offset=offset):
                    reset_sg2_sim(self.env)
                    solver = self.solver(hand)
                    q_start = self.q_arm(hand)
                    initial = get_sg2_wrist_control_T(self.env, hand)
                    self.set_arm(hand, q_start + self.mirrored(np.array(offset), hand))
                    target = get_sg2_wrist_control_T(self.env, hand)
                    self.assertGreater(np.linalg.norm(target[:3, 3] - initial[:3, 3]), .001)
                    self.assertGreater(Rotation.from_matrix(target[:3, :3] @ initial[:3, :3].T).magnitude(), .01)
                    self.set_arm(hand, q_start)
                    self.assert_solution(hand, target, solver.solve(self.env, target))

    def test_exact_pose_with_twisted_joints_still_improves_captured_posture(self):
        for hand in ("left", "right"):
            with self.subTest(hand=hand):
                reset_sg2_sim(self.env)
                solver = self.solver(hand)
                q_ref = self.q_arm(hand)
                target = get_sg2_wrist_control_T(self.env, hand)
                q_twisted = self.mirrored(SAME_POSE_LEFT, hand)
                self.set_arm(hand, q_twisted)
                self.assert_pose(hand, target, position=1e-10, rotation=1e-10)
                before = solver.configuration_info(self.env.data.qpos)
                result = solver.solve(self.env, target)
                self.assert_solution(hand, target, result)
                # The wrist was already exact: an early "pose reached" return
                # would fail this check and leave accumulated arm twist intact.
                self.assertLess(np.linalg.norm(self.q_arm(hand) - q_ref),
                                .5 * np.linalg.norm(q_twisted - q_ref))
                self.assertLess(result["posture_cost"], before["posture_cost"])
                np.testing.assert_array_equal(solver.reference_qpos, self.q_reference)

    def test_opposite_physical_elbow_is_not_accepted_by_joint4_sign(self):
        for hand in ("left", "right"):
            with self.subTest(hand=hand):
                reset_sg2_sim(self.env)
                solver = self.solver(hand)
                target = get_sg2_wrist_control_T(self.env, hand)
                reference_direction = self.elbow_direction(hand)
                q_ref = self.q_arm(hand)
                wrong = self.mirrored(OPPOSITE_ELBOW_LEFT, hand)
                self.set_arm(hand, wrong)
                self.assert_pose(hand, target, position=1e-10, rotation=1e-10)
                self.assertEqual(np.sign(wrong[3]), np.sign(q_ref[3]))
                wrong_angle = np.arccos(np.clip(reference_direction @ self.elbow_direction(hand), -1., 1.))
                self.assertGreater(wrong_angle, np.deg2rad(130.))
                diagnostic = solver.configuration_info(self.env.data.qpos)
                self.assertFalse(diagnostic["valid"], diagnostic)
                self.assertAlmostEqual(abs(diagnostic["elbow_angle_rad"]), wrong_angle, places=8)
                result = solver.solve(self.env, target)
                if result["success"]:
                    self.assert_solution(hand, target, result)
                    repaired_angle = np.arccos(np.clip(reference_direction @ self.elbow_direction(hand), -1., 1.))
                    self.assertLessEqual(repaired_angle, np.deg2rad(85.) + 1e-6)
                else:
                    self.assertTrue(result["reason"])

    def test_capture_and_solve_preserve_entire_callers_dynamic_state(self):
        for hand in ("left", "right"):
            with self.subTest(hand=hand):
                reset_sg2_sim(self.env)
                self.env.data.qvel[:] = np.linspace(-.02, .02, self.env.model.nv)
                self.env.data.ctrl[:] = np.linspace(-.01, .01, self.env.model.nu)
                self.env.data.time = 12.5
                arrays = ("qpos", "qvel", "qacc", "act", "ctrl", "xpos", "xmat")
                before = {name: getattr(self.env.data, name).copy() for name in arrays}
                solver = self.solver(hand)
                target = get_sg2_wrist_control_T(self.env, hand)
                target[2, 3] += .01
                solver.solve(self.env, target)
                for name in arrays:
                    np.testing.assert_array_equal(getattr(self.env.data, name), before[name], err_msg=name)
                self.assertEqual(self.env.data.time, 12.5)

    def test_reference_lifecycle_and_invalid_capture_are_atomic(self):
        for hand in ("left", "right"):
            with self.subTest(hand=hand):
                solver = SG2PostureIK(self.env, hand)
                self.assertFalse(solver.has_reference)
                solver.capture_reference(self.q_reference)
                self.assertTrue(solver.has_reference)
                valid = solver.reference_joint_positions.copy()
                bad_nan = self.q_reference.copy()
                bad_nan[0] = np.nan
                bad_bound = self.q_reference.copy()
                joint = self.env.model.joint(SG2_ARM_JOINT_NAMES[hand][0])
                bad_bound[int(joint.qposadr[0])] = float(joint.range[1]) + .1
                for bad in (self.q_reference[:-1], bad_nan, bad_bound):
                    with self.assertRaises(ValueError):
                        solver.capture_reference(bad)
                    self.assertTrue(solver.has_reference)
                    np.testing.assert_array_equal(solver.reference_joint_positions, valid)
                solver.reset_reference()
                self.assertFalse(solver.has_reference)

    def test_invalid_live_joints_and_target_are_rejected_without_state_changes(self):
        for hand in ("left", "right"):
            with self.subTest(hand=hand):
                reset_sg2_sim(self.env)
                solver = self.solver(hand)
                target = get_sg2_wrist_control_T(self.env, hand)
                invalid_q = []
                bad_nan = self.q_reference.copy()
                bad_nan[0] = np.nan
                invalid_q.append(bad_nan)
                bad_bound = self.q_reference.copy()
                joint = self.env.model.joint(SG2_ARM_JOINT_NAMES[hand][0])
                bad_bound[int(joint.qposadr[0])] = float(joint.range[1]) + .1
                invalid_q.append(bad_bound)
                for q in invalid_q:
                    self.env.data.qpos[:] = q
                    with self.assertRaises(ValueError):
                        solver.solve(self.env, target)
                    np.testing.assert_array_equal(self.env.data.qpos, q)
                self.env.data.qpos[:] = self.q_reference
                invalid_target = target.copy()
                invalid_target[0, 3] = np.nan
                reflection = target.copy()
                reflection[:3, 0] *= -1.
                for transform in (target[:3], invalid_target, reflection):
                    with self.assertRaises(ValueError):
                        solver.solve(self.env, transform)
                    np.testing.assert_array_equal(self.env.data.qpos, self.q_reference)

    def test_unreachable_pose_is_not_accepted_as_best_effort(self):
        for hand in ("left", "right"):
            with self.subTest(hand=hand):
                solver = self.solver(hand)
                target = get_sg2_wrist_control_T(self.env, hand)
                target[:3, 3] += [2., 0., 0.]
                before = self.env.data.qpos.copy()
                result = solver.solve(self.env, target)
                self.assertFalse(result["success"], result)
                self.assertTrue(result["reason"])
                self.assertLessEqual(result["iterations"], 24)
                self.assertTrue(np.isfinite(result["qpos_used_best"]).all())
                np.testing.assert_array_equal(self.env.data.qpos, before)


if __name__ == "__main__":
    unittest.main()
