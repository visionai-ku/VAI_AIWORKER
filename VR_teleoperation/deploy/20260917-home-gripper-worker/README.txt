2026-09-17 HOME/gripper worker bundle

1. Stop robot teleoperation and existing inbound/outbound processes.
2. Back up the robot container's existing ai_worker/zmq/inbound.py, outbound.py and fixed_quest_protocol.py outside that directory.
3. Replace those three files with this bundle. Do not change bringup or ROS topics.
4. Restart BOTH outbound and inbound with SG2_FIXED_QUEST=1. For inbound also set SG2_ZMQ_SUB_IP to the notebook PC LAN IP.
5. Confirm inbound reports arm 120 deg/s, gripper 120 deg/s, home 10 deg/s.
6. Restart the notebook kernel. Test SIM/DRY RUN before live operation.

Gripper default remains 60 deg/s. UI range is 10..120 with the new bridge. HOME and manual joint adjustment remain at most 10 deg/s.
The outbound.py source is unchanged, but it must restart to import the new shared protocol.
Small J2 feedback boundary noise up to 0.0005 rad is normalized with raw-value diagnostics. Command limits are unchanged.

Rollback: stop control, restore the robot's own backed-up three files and matching PC backup, then restart workers and notebook.
