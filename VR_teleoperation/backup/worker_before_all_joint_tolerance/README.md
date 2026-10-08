# Rollback: worker feedback tolerance (2026-10-07)

`fixed_quest_protocol.py` here is the version before every joint got a feedback
boundary tolerance (0.002 rad, lift 0.001 m). It differs from
`package/worker/fixed_quest_protocol.py` only in `FEEDBACK_BOUNDARY_TOLERANCES`:
the original allows only `arm_l_joint2`, `arm_r_joint2`, `gripper_l_joint1` and
`gripper_r_joint1`.

To roll back:

1. Stop the notebook and the robot's inbound/outbound.
2. Copy this file over `package/worker/fixed_quest_protocol.py` and over
   `/root/ros2_ws/src/ai_worker/zmq/fixed_quest_protocol.py` on the robot.
3. Restart both workers with `SG2_FIXED_QUEST=1` (see docs/WORKER_SETUP.md).
4. Remove `test_every_joint_boundary_sample_is_normalized` from
   tests/test_feedback_gripper_upgrade.py.

The PC-side IK margin (`IK_JOINT_LIMIT_MARGIN_RAD`) is independent and can stay.
