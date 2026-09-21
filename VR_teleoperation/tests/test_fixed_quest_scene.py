"""Headless checks against the original SG2 meshes and MuJoCo parser."""

import hashlib
from pathlib import Path
import runpy
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
runpy.run_path(str(PROJECT_DIR / "package/init_project.py"))

from package.fixed_quest_scene import SG2CommandGuard,build_fixed_quest_scene
from ri_motion_v5_package.mujoco_sim.parser import MuJoCoParser


class FixedQuestSceneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (PROJECT_DIR / ".cache").mkdir(exist_ok=True)
        cls.cache_dir = tempfile.TemporaryDirectory(dir=PROJECT_DIR / ".cache")
        cls.robot_xml = PROJECT_DIR / "xml/ffw_sg2.xml"
        cls.floor_xml = PROJECT_DIR / "xml/floor_simple_white.xml"
        cls.source_hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in (cls.robot_xml,cls.floor_xml)]
        cls.scene_xml = build_fixed_quest_scene(cls.robot_xml,cls.floor_xml,Path(cls.cache_dir.name) / "scene.xml")
        cls.env = MuJoCoParser(rel_xml_path=cls.scene_xml,verbose=False)
        cls.env_check = MuJoCoParser(rel_xml_path=cls.scene_xml,verbose=False)
        cls.joint_names = ["arm_%s_joint%d" % (side,j) for side in ("l","r") for j in range(1,8)]
        cls.q_init = np.zeros(14)
        cls.q_init[[3,10]] = -np.pi/2
        cls.guard = SG2CommandGuard(cls.env_check,cls.joint_names)

    @classmethod
    def tearDownClass(cls):
        cls.cache_dir.cleanup()

    def setUp(self):
        self.env.reset()
        self.env_check.reset()
        self.env.forward(q=self.q_init,joint_names=self.joint_names)
        self.env_check.forward(q=self.q_init,joint_names=self.joint_names)

    def test_scene_preserves_source_assets_and_physics(self):
        self.assertEqual(self.source_hashes,[hashlib.sha256(path.read_bytes()).hexdigest() for path in (self.robot_xml,self.floor_xml)])
        original = ET.parse(self.robot_xml).getroot()
        generated = ET.parse(Path(self.scene_xml).with_name("scene.robot.xml")).getroot()
        self.assertIsNone(generated.find(".//freejoint"))
        self.assertEqual(generated.find("./worldbody/body[@name='base_link']").get("pos"),"0 0 0.01")
        for section in ("actuator","equality","contact","default"):
            self.assertEqual(ET.tostring(original.find(section)),ET.tostring(generated.find(section)))
        self.assertEqual([node.attrib for node in original.iter("inertial")],[node.attrib for node in generated.iter("inertial")])
        self.assertEqual(len(list(original.iter("mesh"))),len(list(generated.iter("mesh"))))
        self.assertEqual(self.env.model.nu,25)
        self.assertFalse(np.any(self.env.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE))

    def test_small_motion_and_checker_state_isolation(self):
        q_target = self.q_init.copy()
        q_target[0] += np.deg2rad(0.5)
        q_main_before = self.env.data.qpos.copy()
        q_check_before = self.env_check.data.qpos.copy()
        check_info = self.guard.check(self.q_init,q_target)
        self.assertTrue(check_info["accepted"],check_info)
        self.assertGreater(check_info["min_distance_m"],0.005)
        np.testing.assert_array_equal(q_main_before,self.env.data.qpos)
        np.testing.assert_array_equal(q_check_before,self.env_check.data.qpos)

    def test_nonfinite_malformed_and_joint_limit_rejected(self):
        invalid_vectors = [None,[0.0],np.zeros((2,7)),["0"]*14,[False]*14]
        for value in (np.nan,np.inf):
            q_bad = self.q_init.copy()
            q_bad[0] = value
            invalid_vectors.append(q_bad)
        for q_bad in invalid_vectors:
            with self.subTest(command=q_bad):
                self.assertFalse(self.guard.check(self.q_init,q_bad)["accepted"])
        q_bad = self.q_init.copy()
        q_bad[1] = -0.01
        self.assertEqual(self.guard.check(self.q_init,q_bad)["reason"],"joint_limit_end:arm_l_joint2")

    def test_real_mesh_collision_and_swept_path_blocked(self):
        q_collision = self.q_init.copy()
        q_collision[2] = -2.0
        q_main_before = self.env.data.qpos.copy()
        endpoint_info = self.guard.check(q_collision,q_collision)
        self.assertFalse(endpoint_info["accepted"])
        self.assertEqual(endpoint_info["reason"],"collision_clearance")
        self.assertLess(endpoint_info["min_distance_m"],0.0)
        sweep_info = self.guard.check(self.q_init,q_collision)
        self.assertFalse(sweep_info["accepted"])
        self.assertEqual(sweep_info["reason"],"collision_clearance")
        self.assertGreater(sweep_info["samples_checked"],1)
        self.assertLess(sweep_info["sweep_fraction"],1.0)
        np.testing.assert_array_equal(q_main_before,self.env.data.qpos)
        np.testing.assert_array_equal(self.q_init,self.env_check.get_qpos(joint_names=self.joint_names))

    def test_collision_can_be_skipped_without_changing_check_environment(self):
        q_collision = self.q_init.copy()
        q_collision[2] = -2.0
        q_check_before = self.env_check.data.qpos.copy()
        strict = self.guard.check(self.q_init,q_collision)
        self.assertFalse(strict["accepted"])
        self.assertTrue(strict["collision_check_enabled"])
        with patch.object(self.guard,"_minimum_distance",side_effect=AssertionError("collision query ran")):
            unchecked = self.guard.check(self.q_init,q_collision,check_collision=False)
        self.assertTrue(unchecked["accepted"],unchecked)
        self.assertFalse(unchecked["collision_check_enabled"])
        self.assertEqual(unchecked["samples_checked"],0)
        self.assertTrue(np.isnan(unchecked["min_distance_m"]))
        np.testing.assert_array_equal(q_check_before,self.env_check.data.qpos)

    def test_collision_skip_preserves_command_and_environment_validation(self):
        invalid_vectors = [None,[0.0],np.zeros((2,7)),["0"]*14,[False]*14]
        for value in (np.nan,np.inf,-.01):
            q_bad = self.q_init.copy()
            q_bad[1] = value
            invalid_vectors.append(q_bad)
        for q_bad in invalid_vectors:
            for q_start,q_end in ((self.q_init,q_bad),(q_bad,self.q_init)):
                with self.subTest(start=q_start,end=q_end):
                    check_info = self.guard.check(q_start,q_end,check_collision=False)
                    self.assertFalse(check_info["accepted"],check_info)
                    self.assertFalse(check_info["collision_check_enabled"])
        self.env_check.data.qpos[0] = np.nan
        check_info = self.guard.check(self.q_init,self.q_init,check_collision=False)
        self.assertEqual(check_info["reason"],"invalid_check_environment")
        self.assertFalse(check_info["accepted"])

    def test_collision_check_option_requires_a_boolean(self):
        for invalid in (None,0,1,"false"):
            with self.subTest(value=invalid), self.assertRaises(ValueError):
                self.guard.check(self.q_init,self.q_init,check_collision=invalid)

    def test_tiny_ik_motion_does_not_accept_unbounded_zero_distance(self):
        # Recorded from the first 0.08 s of a 2 cm controller translation.
        # lift_link/link4 can return d=0 with nearest points 0.398 m apart.
        q_target = self.q_init.copy()
        q_target[:7] = [
            3.047782287074554e-5,3.441154515796083e-9,-2.957916526250121e-9,
            -1.5710499179054065,-6.716132081234605e-9,
            0.00022308211502606548,-4.008326195809007e-10,
        ]
        check_info = self.guard.check(self.q_init,q_target)
        self.assertTrue(check_info["accepted"],check_info)
        self.assertGreater(check_info["min_distance_m"],0.005)

    def test_zero_endpoint_buffer_retries_at_clearance_threshold(self):
        # Recorded dynamic measured-to-command interpolation, with real meshes.
        # link4/link6 returns an all-zero result at .05 m but certifies >.005 m
        # with a finite margin query. Keep every gripper mimic position intact.
        self.env_check.data.qpos[:] = [
            4.0177382100889187e-35,0,-1.7625822606693896e-36,0,2.8494977702388835e-37,0,
            -0.010635635599757878,0.002172697347656333,-1.1384095068059617e-8,
            0.027357687448955087,0.00018002844406239524,-0.00021477580947629188,
            -1.5576623561479912,0.0010877143953877306,0.011290174524418084,0.0002622873958686931,
            0.3282053415992914,-0.3282053415992914,-0.3282053415992914,0.3282053415992914,
            0.01705369887752153,-0.000343140076074937,0.000485943313530372,
            -1.4921187909461762,-0.001411466383429318,-0.043867644866707925,-0.00044742513097640886,
            0.24952287338071472,-0.24952287338071472,-0.24952287338071472,0.24952287338071472,
        ]
        self.env_check.forward(clip_position=False)
        q_sample = self.env_check.get_qpos(joint_names=self.joint_names)
        check_info = self.guard.check(q_sample,q_sample)
        self.assertTrue(check_info["accepted"],check_info)
        self.assertGreater(check_info["min_distance_m"],0.005)
        self.assertTrue(check_info["distance_is_lower_bound"])

    def test_missing_collision_query_results_fail_closed(self):
        with patch.object(self.env_check,"get_geom_pair_signed_distance_infos",return_value=[]):
            check_info = self.guard.check(self.q_init,self.q_init)
        self.assertFalse(check_info["accepted"])
        self.assertTrue(check_info["reason"].startswith("collision_query_error"))

    def test_candidates_keep_real_contacts_and_document_exclusions(self):
        pairs = self.guard.candidate_geom_pairs.copy()
        self.assertGreater(len(pairs),400)
        self.assertEqual(len([entry for entry in self.guard.explicit_exclusions if entry["reason"] == "same_gripper_mechanical_linkage"]),4)
        q_collision = self.q_init.copy()
        q_collision[2] = -2.0
        self.env_check.forward(q=q_collision,joint_names=self.joint_names)
        guard_in_contact = SG2CommandGuard(self.env_check,self.joint_names)
        np.testing.assert_array_equal(pairs,guard_in_contact.candidate_geom_pairs)
        self.assertFalse(guard_in_contact.check(q_collision,q_collision)["accepted"])

    def test_generation_refuses_source_directory(self):
        with self.assertRaises(ValueError):
            build_fixed_quest_scene(self.robot_xml,self.floor_xml,PROJECT_DIR / "xml/do_not_write.xml")


if __name__ == "__main__":
    unittest.main()
