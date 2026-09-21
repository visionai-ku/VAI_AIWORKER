"""In-process ideal joint follower: existing strict gate, no sockets or ROS.

Accepted joint commands become measured positions on the next feedback sample.
This validates protocol/state/IK wiring, NOT motor dynamics or network timing.
"""
import copy
import numpy as np
from .worker.fixed_quest_protocol import (JointStateTracker, StrictCommandGate,
    JOINT_NAMES, FIXED_QUEST_VR_GATE_OPTIONS, parse_joint_groups)


class IntentSimBridge:
    def __init__(self, q):
        self.q = np.asarray(q, float).copy()
        self.tracker = JointStateTracker(stream_id='v2-sim-state')
        self.gate = StrictCommandGate(self.tracker, bridge_id='v2-sim-bridge', **FIXED_QUEST_VR_GATE_OPTIONS)
        self.enabled, self.closed, self.now = True, False, None
        self.last_result = None
        self._stamp = 0

    def latest(self, now):
        if self.closed: raise RuntimeError('SIM bridge closed')
        self.now = float(now)
        self._stamp = max(self._stamp + 1, int(now * 1e9), 1)
        self.tracker.update(list(JOINT_NAMES), self.q.tolist(), self._stamp, now)
        expired = self.gate.expire(now)
        if expired['publish_joints'] is not None:
            self.q = np.asarray(parse_joint_groups(expired['publish_joints']))
        status = self.gate.status(now)
        return dict(ready=True, reason='SIM ideal feedback', q=self.q.copy(), age_s=0.,
            bridge_age_s=0., generation=0, requires_resync=False, stream_id='v2-sim-state',
            bridge_id=status['bridge_id'], bridge_state=status['state'],
            owner_session_id=status['owner_session_id'], last_command_seq=status['last_command_seq'],
            state=self.tracker.snapshot(now), bridge=status)

    def send(self, packet):
        if self.closed or self.now is None: raise RuntimeError('SIM feedback is not running')
        self.last_result = self.gate.receive(copy.deepcopy(packet), self.now)
        if self.last_result['publish_joints'] is not None:
            self.q = np.asarray(parse_joint_groups(self.last_result['publish_joints']))
        # Same transport meaning as ZMQ: delivered, not necessarily accepted.
        return True

    def acknowledge_resync(self, generation):
        return generation == 0

    def close(self):
        self.closed = True
