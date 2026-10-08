"""IK joint-limit margin setting and stale-feedback guidance, with no robot/Quest I/O."""
from pathlib import Path
import runpy
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
runpy.run_path(str(ROOT / 'package/init_project.py'))

from package.intent_teleop import IntentTeleopSettings, feedback_diagnostic_text


class LimitMarginSettingsTest(unittest.TestCase):
    def test_settings_validate_margin(self):
        self.assertEqual(IntentTeleopSettings().validate().IK_JOINT_LIMIT_MARGIN_RAD, .005)
        for value in (-.001, .06, np.nan, True, "0.005"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                IntentTeleopSettings(IK_JOINT_LIMIT_MARGIN_RAD=value).validate()


class FeedbackGuidanceTest(unittest.TestCase):
    REJECTION = {"at_s": 1., "reason": "joint_limit:arm_l_joint6: position=1.57000536795 rad, "
                                       "allowed=[-1.57, 1.57] rad, excess=5.36795114292e-06 rad"}

    def feedback(self, result, stale, rejection):
        worker = {"last_callback_result": result, "last_rejection": rejection, "stale_joints": stale}
        return {"ready": False, "diagnostics": {"packet_age_s": .001, "outbound_joint_state": worker,
                                                "inbound_joint_state": dict(worker)}}

    def test_rejected_joint_limit_explains_the_way_out(self):
        text = feedback_diagnostic_text(self.feedback("rejected", ["arm_l_joint6"], self.REJECTION))
        self.assertIn("arm_l_joint6가 관절 한계를 미세하게 넘어", text)
        self.assertEqual(text.count("!!"), 1)

    def test_stale_joint_with_earlier_limit_rejection_also_explains(self):
        lift = {"at_s": 1., "reason": "joint_limit:lift_joint: position=4e-05 m, allowed=[-0.5, 0] m, excess=4e-05 m"}
        text = feedback_diagnostic_text(self.feedback("updated", ["lift_joint"], lift))
        self.assertIn("갱신 지연: lift_joint", text)
        self.assertIn("마지막 거부: joint_limit:lift_joint", text)
        self.assertIn("lift_joint가 관절 한계를", text)

    def test_old_rejection_without_staleness_is_not_flagged(self):
        text = feedback_diagnostic_text(self.feedback("updated", [], self.REJECTION))
        self.assertNotIn("!!", text)

    def test_ready_feedback_has_no_text(self):
        self.assertEqual(feedback_diagnostic_text({"ready": True, "diagnostics": {}}), "")


if __name__ == "__main__":
    unittest.main()
