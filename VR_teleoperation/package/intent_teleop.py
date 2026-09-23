"""Shared SIM/REAL notebook runtime using existing FK/IK, guard and grip adapter.

SIM substitutes only the transport with an in-process ideal follower. REAL starts
in DRY RUN by default. Importing/building models never opens any transport.
"""
import time
import sys
import traceback
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from numbers import Real

import numpy as np
from PyQt5.QtWidgets import (QApplication, QPushButton, QLabel, QLineEdit, QWidget, QHBoxLayout, QGridLayout, QComboBox,
                             QVBoxLayout, QScrollArea, QButtonGroup)
from ri_motion_v5_package.mujoco_sim import MuJoCoParser, BodyPositionIKSolverRevPriBase, MuJoCoRenderProcess
from ri_motion_v5_package.qt import MultiRadioQtWidget, use_pyqt_platform_plugins
from ri_motion_v5_package.utility import get_ip_address
from utils import (PROJECT_DIR, FLOOR_XML_PATH, SG2_XML_PATH, SG2_HAND_TYPES,
                   SG2_NULL_REFERENCE_MODES,
                   create_sg2_arm_ik_solvers, get_sg2_wrist_control_T, build_q_msg_layout)
from package.controller_mapping import ControllerMapper
from package.quest_controller_server import QuestControllerServer, create_self_signed_certificate
from package.fixed_quest_scene import build_fixed_quest_scene, SG2CommandGuard
from package.fixed_quest_sim import reset_sg2_sim, SG2FixedQuestSimulation, ControllerRecorder
from package.fixed_quest_transport import SG2RobotStateReceiver, SG2RobotCommandPublisher, sample_control_inputs
from package.fixed_quest_real import (SG2FixedQuestReal, FIXED_QUEST_MIN_GRIPPER_SPEED,
                                      FIXED_QUEST_MAX_GRIPPER_SPEED, FIXED_QUEST_MAX_TRACKING_PAUSE_S)
from package.worker.fixed_quest_protocol import (JOINT_NAMES, JOINT_LIMITS, FIXED_QUEST_DEFAULT_ARM_SPEED, FIXED_QUEST_MAX_ARM_SPEED,
    OFFICIAL_VR_CONTROL_HZ, VR_ARM_TRACKING_LAG, FIXED_QUEST_VR_BRIDGE_LIMITS)
from dataclasses import dataclass
from .intent_calibration import load_intent_profile
from .intent_sim_bridge import IntentSimBridge
from .home_path import HOME_PATH_STRATEGIES
from .posture_ik import SG2PostureIK
from .runtime_diagnostics import GCPauseMonitor

@dataclass
class IntentTeleopSettings:
    backend: str = "sim"
    PROFILE_PATH: object = None
    SIM_CANDIDATE_CAPTURE_PATH: object = None
    ALLOW_FAILED_PROFILE_IN_SIM: bool = False
    REAL_CANDIDATE_REVIEW_PATH: object = None
    REAL_CANDIDATE_CAPTURE_PATH: object = None
    CONFIRM_CALIBRATION_SETUP: bool = False
    ROBOT_IP: str = "192.168.6.2"
    ROBOT_STATE_PORT: int = 5560
    ROBOT_COMMAND_PORT: int = 5561
    DRY_RUN: bool = True
    QUEST_HOST: object = None
    QUEST_PORT: int = 8443
    CERTFILE: object = None
    KEYFILE: object = None
    VIEWER_MODE: str = "process" if sys.platform == "darwin" else "direct"
    COORDINATED_ARM_RATE_LIMIT: bool = True
    TRACKING_PAUSE_S: float = .10
    ACCEPT_EMULATED_POSE: bool = False  # use Quest estimated poses instead of pausing
    TRACKING_MODE_BUTTONS: tuple = ()  # TRACKING_MODES keys shown as buttons; () shows none
    RETRY_IK_NEAR_TARGET: bool = True
    IK_RECOVERY_TIMEOUT_S: float = 1.0
    NULL_REFERENCE: str = "self_motion"
    # True: re-anchor the self-motion (J3/J5 twist) reference to the arm pose at every
    # both-grip start, instead of the SIM startup pose captured when the solvers were built.
    NULL_ANCHOR_ON_GRIP: bool = False
    # Full-pose constrained IK; posture reference is captured once at first
    # active grip from feedback, including DRY RUN, and survives re-gripping.
    POSTURE_IK_ENABLED: bool = True
    POSTURE_POSITION_TOLERANCE_M: float = .001
    POSTURE_ROTATION_TOLERANCE_DEG: float = .5
    POSTURE_ELBOW_LIMIT_DEG: float = 85.
    # Ignore controller rotations up to this angle per clutch (soft deadband); 0 = off.
    ROTATION_DEADBAND_DEG: float = 0.0
    CONTROL_HZ: float = OFFICIAL_VR_CONTROL_HZ
    IK_HZ: float = 50.
    RENDER_HZ: float = 25.
    POSITION_SCALE: float = .5
    POSITION_SCALE_Y: float = .5
    POSITION_SCALE_Z: float = .5
    MAX_JOINT_SPEED: float = FIXED_QUEST_DEFAULT_ARM_SPEED
    MAX_AUX_JOINT_SPEED: float = np.deg2rad(60.)
    MAX_GRIPPER_SPEED: float = np.deg2rad(60.)
    TELEOP_RAMP_S: float = 3.0
    MAX_ARM_JOINT_LAG: float = VR_ARM_TRACKING_LAG
    MAX_HOME_JOINT_SPEED: float = np.deg2rad(10.)
    MAX_LIFT_SPEED: float = .01
    MAX_JOINT_LAG: float = np.deg2rad(5.)
    MAX_LIFT_LAG: float = .01
    INPUT_TIMEOUT_S: float = .2
    FEEDBACK_TIMEOUT_S: float = .2
    COMMAND_TTL_S: float = .2
    MAX_LOOP_DT_S: float = .10
    MAX_COMMAND_DT_S: float = .02
    # Operator-controlled motion may bring both grippers together. Automatic
    # Initial pose / Reset pose / manual joint moves always check clearance.
    TELEOP_COLLISION_CHECK: bool = False
    COLLISION_MARGIN_M: float = .005
    # Allow commands that move OUT of the margin (clearance never decreasing) when the measured
    # pose drifted inside it, down to this distance [m]; None = off (original strict guard).
    COLLISION_ESCAPE_MIN_M: object = None
    R_BQ: object = None

    @property
    def POSITION_SCALES(self):
        return {hand: [self.POSITION_SCALE, self.POSITION_SCALE_Y, self.POSITION_SCALE_Z]
                for hand in SG2_HAND_TYPES}

    def validate(self):
        if self.backend not in ("sim", "real"): raise ValueError("backend must be sim or real")
        if type(self.ALLOW_FAILED_PROFILE_IN_SIM) is not bool:
            raise ValueError('ALLOW_FAILED_PROFILE_IN_SIM must be bool')
        if self.ALLOW_FAILED_PROFILE_IN_SIM and self.backend != 'sim':
            raise ValueError('Failed calibration candidate is allowed only in SIM')
        if self.REAL_CANDIDATE_REVIEW_PATH is not None and self.backend != 'real':
            raise ValueError('REAL candidate review requires real backend')
        if type(self.DRY_RUN) is not bool: raise ValueError("DRY_RUN must be bool")
        if type(self.TELEOP_COLLISION_CHECK) is not bool:
            raise ValueError("TELEOP_COLLISION_CHECK must be bool")
        if self.backend == "sim": self.DRY_RUN = False  # in-process gate only; no robot connection
        if not np.deg2rad(10) <= self.MAX_JOINT_SPEED <= FIXED_QUEST_MAX_ARM_SPEED:
            raise ValueError("arm speed must be 10..120 deg/s")
        if not 0 < self.MAX_HOME_JOINT_SPEED <= np.deg2rad(10): raise ValueError("home speed exceeds 10 deg/s")
        if (isinstance(self.MAX_GRIPPER_SPEED, (bool, np.bool_)) or not isinstance(self.MAX_GRIPPER_SPEED, Real)
                or not FIXED_QUEST_MIN_GRIPPER_SPEED <= self.MAX_GRIPPER_SPEED <= FIXED_QUEST_MAX_GRIPPER_SPEED):
            raise ValueError("gripper speed must be 10..120 deg/s")
        if not all(np.isfinite(x) and .05 <= x <= 3 for x in
                   (self.POSITION_SCALE, self.POSITION_SCALE_Y, self.POSITION_SCALE_Z)):
            raise ValueError("v2 intent gains must be positive: .05..3; do not invert intent Z")
        for name in ("CONTROL_HZ", "IK_HZ", "RENDER_HZ"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0: raise ValueError(name)
        recovery_timeout = self.IK_RECOVERY_TIMEOUT_S
        if (isinstance(recovery_timeout, (bool, np.bool_)) or not isinstance(recovery_timeout, Real)
                or not np.isfinite(recovery_timeout) or not 0 <= recovery_timeout <= 5):
            raise ValueError("IK_RECOVERY_TIMEOUT_S must be a finite number in 0..5 seconds; 0 disables recovery")
        if self.NULL_REFERENCE not in SG2_NULL_REFERENCE_MODES:
            raise ValueError("NULL_REFERENCE must be one of %s" % (SG2_NULL_REFERENCE_MODES,))
        if type(self.NULL_ANCHOR_ON_GRIP) is not bool:
            raise ValueError("NULL_ANCHOR_ON_GRIP must be a bool")
        if type(self.POSTURE_IK_ENABLED) is not bool:
            raise ValueError("POSTURE_IK_ENABLED must be a bool")
        for name, maximum in (("POSTURE_POSITION_TOLERANCE_M", .01),
                              ("POSTURE_ROTATION_TOLERANCE_DEG", 5.),
                              ("POSTURE_ELBOW_LIMIT_DEG", 89.)):
            value = getattr(self, name)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
                    or not np.isfinite(value) or not 0 < value <= maximum):
                raise ValueError(name + " must be finite and positive, at most " + str(maximum))
        deadband = self.ROTATION_DEADBAND_DEG
        if (isinstance(deadband, (bool, np.bool_)) or not isinstance(deadband, Real)
                or not np.isfinite(deadband) or not 0 <= deadband <= 30):
            raise ValueError("ROTATION_DEADBAND_DEG must be a number in 0..30")
        pause = self.TRACKING_PAUSE_S
        if (isinstance(pause, (bool, np.bool_)) or not isinstance(pause, Real)
                or not np.isfinite(pause) or not 0 <= pause <= FIXED_QUEST_MAX_TRACKING_PAUSE_S):
            raise ValueError("TRACKING_PAUSE_S must be a finite number in 0..%g seconds" % FIXED_QUEST_MAX_TRACKING_PAUSE_S)
        if type(self.ACCEPT_EMULATED_POSE) is not bool:
            raise ValueError("ACCEPT_EMULATED_POSE must be bool")
        buttons = self.TRACKING_MODE_BUTTONS
        known = [key for key, _, _, _ in TRACKING_MODES]
        if (not isinstance(buttons, (tuple, list)) or not all(isinstance(key, str) for key in buttons)
                or len(set(buttons)) != len(buttons) or any(key not in known for key in buttons)):
            raise ValueError("TRACKING_MODE_BUTTONS must list distinct keys from %s" % known)
        start = tracking_mode_key({"tracking_pause_s": pause, "accept_emulated": self.ACCEPT_EMULATED_POSE})
        if buttons and start not in buttons:
            raise ValueError("TRACKING_MODE_BUTTONS must include the start mode set by TRACKING_PAUSE_S/ACCEPT_EMULATED_POSE")
        return self


def selected_profile(cfg):
    # Check here too: setup notebooks can call selection without validate().
    if cfg.REAL_CANDIDATE_REVIEW_PATH is not None:
        if cfg.backend != 'real' or cfg.ALLOW_FAILED_PROFILE_IN_SIM:
            raise ValueError('REAL candidate review requires real backend without SIM override')
        if cfg.PROFILE_PATH is None:
            raise ValueError('REAL candidate review requires PROFILE_PATH')
        from .intent_sim_candidate import load_reviewed_real_candidate
        return load_reviewed_real_candidate(cfg.PROFILE_PATH, cfg.REAL_CANDIDATE_CAPTURE_PATH,
            cfg.REAL_CANDIDATE_REVIEW_PATH, setup_confirmed=cfg.CONFIRM_CALIBRATION_SETUP)
    if cfg.ALLOW_FAILED_PROFILE_IN_SIM:
        if cfg.backend != 'sim':
            raise ValueError('Failed calibration candidate is allowed only in SIM')
        if cfg.PROFILE_PATH is None:
            raise ValueError('SIM candidate mode requires PROFILE_PATH')
        from .intent_sim_candidate import load_sim_candidate
        return load_sim_candidate(cfg.PROFILE_PATH, cfg.SIM_CANDIDATE_CAPTURE_PATH,
                                  setup_confirmed=cfg.CONFIRM_CALIBRATION_SETUP)
    if cfg.PROFILE_PATH is None: return None, None
    return load_intent_profile(cfg.PROFILE_PATH, real=cfg.backend == "real",
                              setup_confirmed=cfg.CONFIRM_CALIBRATION_SETUP)


def feedback_diagnostic_text(feedback):
    """Explain stopped feedback without interpreting old poses as fresh."""
    if feedback.get("ready") or not feedback.get("diagnostics"):
        return ""
    detail = feedback["diagnostics"]
    age = detail.get("packet_age_s")
    lines = [] if age is None else ["ZMQ 마지막 수신: %.0fms 전" % (1000 * age)]
    for key, label in (("outbound_joint_state", "outbound"), ("inbound_joint_state", "inbound")):
        worker = detail.get(key)
        if not isinstance(worker, dict):
            continue
        lines.append(label + " ROS: " + str(worker.get("last_callback_result")))
        rejection = worker.get("last_rejection")
        stale = worker.get("stale_joints")
        if worker.get("last_callback_result") == "rejected" and isinstance(rejection, dict):
            lines.append(str(rejection.get("reason", "")))
        elif isinstance(stale, list) and stale:
            lines.append("갱신 지연: " + ", ".join(str(name) for name in stale))
    return "\n" + "\n".join(lines) if lines else ""


def ik_recovery_status_text(result):
    """Show recovery separately from a transport loss or a grip rearm request."""
    recovery = result.get("ik_recovery") or {}
    status = recovery.get("status")
    if status == "waiting" and result.get("state") == "IK_WAIT":
        label = "IK 재시도 중 · 현재 명령 유지"
    elif status == "recovered" and result.get("state") == "ACTIVE":
        label = "IK 복구 완료 · 조작 재개"
    elif status == "timeout":
        label = "IK 재시도 시간 초과"
    elif status == "cancelled":
        label = "IK 재시도 중단"
    else:
        return ""
    text = "%s · %.2f/%.2f초 · %d회" % (
        label, recovery.get("elapsed_s", 0.), recovery.get("timeout_s", 0.),
        recovery.get("attempts", 0))
    if recovery.get("reason"):
        text += "\n원인: " + str(recovery["reason"])
    return text


# Tracking-loss modes: (key, button label, pause seconds, accept emulated).
# The emulated mode keeps the 0.1 s pause for controllers that vanish entirely.
TRACKING_MODES = (
    ("pause_0.1", "정지 0.1초", .1, False),
    ("pause_0.5", "정지 0.5초", .5, False),
    ("pause_1.0", "정지 1.0초", 1., False),
    ("emulated", "추정 위치 사용", .1, True),
)


def tracking_mode_key(mode):
    """Button key for an applied adapter mode, or None for a custom setting."""
    for key, _, pause, emulated in TRACKING_MODES:
        if np.isclose(mode["tracking_pause_s"], pause) and mode["accept_emulated"] == emulated:
            return key
    return None


def tracking_mode_text(mode):
    if mode["accept_emulated"]:
        return "추정 위치 사용 (완전 소실 시 정지 %.1f초)" % mode["tracking_pause_s"]
    return "정지 %.1f초" % mode["tracking_pause_s"]


HOME_STRATEGY_LABELS = {
    "direct": "직선 경로",
    "unwind_j3j5_first": "J3·J5 먼저 풀기",
    "unwind_forearm_wrist_first": "전완·손목 먼저 풀기",
    "right_arm_first": "오른팔 먼저",
    "left_arm_first": "왼팔 먼저",
}


def home_status_text(result):
    """Initial pose path check and progress; empty when no return is involved."""
    if result.get("manual_adjustment"):
        return "수동 관절 조정 완료" if result.get("state") == "HOME" else "수동 관절 조정 · 최대 10°/s · Cancel return으로 중단"
    plan = result.get("home_plan")
    if plan is not None and not plan["ok"]:
        hint = ("현재 자세가 이미 충돌 여유 기준 안쪽입니다" if "current pose already" in plan["reason"]
                else "수동 관절 미리보기·조정 또는 VR 조작 후 다시 누르세요")
        return "초기 자세 경로 차단 · 명령 전송 없음\n%s\n원인: %s" % (hint, plan["reason"])
    path = result.get("home_path")
    if result.get("state") == "HOMING" and path:
        return "초기 자세 복귀 · %s · 단계 %d/%d · 전체 예상 %.0f초" % (
            HOME_STRATEGY_LABELS.get(path["strategy"], path["strategy"]),
            path["stage"], path["stages"], path["duration_s"])
    if plan is not None:
        return "초기 자세 경로 확인 · %s · 예상 %.0f초" % (
            HOME_STRATEGY_LABELS.get(plan["strategy"], plan["strategy"]), plan["duration_s"])
    return ""


def execution_diagnostics(result, env, env_measured):
    """Current commanded FK and separately measured FK; never alias q_measured."""
    result["execution_stages"] = {}
    for hand in SG2_HAND_TYPES:
        stages = result.get("mapping_stages", {}).get(hand)
        command = get_sg2_wrist_control_T(env, hand)[:3, 3].copy()
        measured = None if result.get("robot_q") is None else get_sg2_wrist_control_T(env_measured, hand)[:3, 3].copy()
        result["execution_stages"][hand] = {
            "command_eef_m": command, "measured_eef_m": measured,
            "bounded_to_command_error_m": None if stages is None else np.asarray(stages["bounded_eef_m"]) - command,
            "command_to_measured_error_m": None if measured is None else command - measured,
        }
    return result

def build_models(cfg):
    cfg.validate()
    xml_path = build_fixed_quest_scene(
        robot_xml=SG2_XML_PATH, floor_xml=FLOOR_XML_PATH,
        output_xml=PROJECT_DIR / ".cache" / "fixed_quest" / ("intent_" + cfg.backend + "_scene.xml"),
    )
    # Parser/IK가 범위를 캐시하므로 원본이 아닌 생성 XML에서 먼저 제한한다.
    robot_copy_path = Path(xml_path).with_name(Path(xml_path).stem + ".robot.xml")
    scene_tree = ET.parse(robot_copy_path)
    assert scene_tree.getroot().find("compiler").get("angle") == "radian"
    for name, (lower, upper) in JOINT_LIMITS.items():
        joint = scene_tree.getroot().find(".//worldbody//joint[@name='%s']" % name)
        assert joint is not None, name
        defaults = scene_tree.getroot().find(".//default[@class='%s']/joint" % joint.get("class"))
        joint_range = joint.get("range") if joint.get("range") is not None else defaults.get("range")
        local_lower, local_upper = map(float, joint_range.split())
        lower, upper = max(lower, local_lower), min(upper, local_upper)
        assert lower < upper, name
        joint.set("range", "%.12g %.12g" % (lower, upper))
    scene_tree.write(robot_copy_path, encoding="utf-8", xml_declaration=True)
    
    env = MuJoCoParser(rel_xml_path=xml_path, verbose=False)
    env_ik = MuJoCoParser(rel_xml_path=xml_path, verbose=False)
    env_check = MuJoCoParser(rel_xml_path=xml_path, verbose=False)
    env_measured = MuJoCoParser(rel_xml_path=xml_path, verbose=False)
    for model_env in (env, env_ik, env_check, env_measured):
        reset_sg2_sim(model_env)
    command_joint_names = list(JOINT_NAMES)
    q_msg_format_dict = {
        "lift": {"prefixes": ["lift_joint"]}, "head": {"prefixes": ["head_joint"]},
        "arm_l": {"prefixes": ["arm_l_joint"]}, "arm_r": {"prefixes": ["arm_r_joint"]},
        "gripper_l": {"prefixes": ["gripper_l_joint1"]}, "gripper_r": {"prefixes": ["gripper_r_joint1"]},
    }
    q_msg_layout = build_q_msg_layout(env, q_msg_format_dict, use_joint_names=command_joint_names)
    ik_solvers, arm_joint_names = create_sg2_arm_ik_solvers(env_ik, MuJoCoParser, BodyPositionIKSolverRevPriBase)
    guard = SG2CommandGuard(env_check, command_joint_names, safe_margin=cfg.COLLISION_MARGIN_M,
                            escape_min_distance=cfg.COLLISION_ESCAPE_MIN_M)
    print("Command joints:", len(command_joint_names), "| Collision pairs:", len(guard.candidate_geom_pairs))
    print("No robot connection or commands in this setup cell.")
    return {'xml_path': xml_path, 'env': env, 'env_ik': env_ik, 'env_check': env_check, 'env_measured': env_measured, 'command_joint_names': command_joint_names, 'q_msg_layout': q_msg_layout, 'ik_solvers': ik_solvers, 'arm_joint_names': arm_joint_names, 'guard': guard}


def build_planner(models, cfg, profile=None, intent_matrices=None):
    env = models['env']
    env_ik = models['env_ik']
    ik_solvers = models['ik_solvers']
    arm_joint_names = models['arm_joint_names']
    command_joint_names = models['command_joint_names']
    guard = models['guard']
    posture_solvers = ({hand: SG2PostureIK(
        env_ik, hand, position_tolerance=cfg.POSTURE_POSITION_TOLERANCE_M,
        rotation_tolerance=np.deg2rad(cfg.POSTURE_ROTATION_TOLERANCE_DEG),
        elbow_limit_rad=np.deg2rad(cfg.POSTURE_ELBOW_LIMIT_DEG)) for hand in SG2_HAND_TYPES}
        if cfg.POSTURE_IK_ENABLED else None)
    mapper = ControllerMapper(scale=cfg.POSITION_SCALE, timeout_s=cfg.INPUT_TIMEOUT_S, R_BQ=cfg.R_BQ,
                              position_scales=cfg.POSITION_SCALES, intent_matrices=intent_matrices,
                              intent_profile_id=None if profile is None else profile["profile_id"],
                              accept_emulated=cfg.ACCEPT_EMULATED_POSE,
                              rotation_deadband_rad=np.deg2rad(cfg.ROTATION_DEADBAND_DEG))
    planner = SG2FixedQuestSimulation(
        env, env_ik, ik_solvers, arm_joint_names, command_joint_names, mapper, guard,
        mode="kinematic", ik_hz=cfg.IK_HZ, max_joint_speed=cfg.MAX_JOINT_SPEED,
        max_loop_dt=cfg.MAX_LOOP_DT_S, max_command_dt=cfg.MAX_COMMAND_DT_S,
        coordinated_arm_rate_limit=cfg.COORDINATED_ARM_RATE_LIMIT, retry_ik_near_target=cfg.RETRY_IK_NEAR_TARGET,
        ik_recovery_timeout_s=cfg.IK_RECOVERY_TIMEOUT_S, null_reference=cfg.NULL_REFERENCE,
        null_anchor_on_grip=cfg.NULL_ANCHOR_ON_GRIP,
        posture_solvers=posture_solvers,
        teleop_collision_check=cfg.TELEOP_COLLISION_CHECK,
    )
    for index, name in enumerate(command_joint_names):
        if name == "lift_joint":
            planner.speed_limits[index] = cfg.MAX_LIFT_SPEED
        elif name.startswith("gripper_"):
            planner.speed_limits[index] = cfg.MAX_GRIPPER_SPEED
        elif not name.startswith("arm_"):
            planner.speed_limits[index] = cfg.MAX_AUX_JOINT_SPEED
    return mapper, planner


def run_intent_teleop(cfg=None, on_tick=None):
    """on_tick (optional): object with __call__(now, frame, result, real, ui_actions),
    and attributes lock_input (bool: ignore Quest input for teleop) and status (str: shown
    in the control window). Called once per control tick after the result is logged."""
    cfg = (cfg or IntentTeleopSettings()).validate()
    profile, intent_matrices = selected_profile(cfg)
    if cfg.QUEST_HOST is None: cfg.QUEST_HOST = get_ip_address()
    models = build_models(cfg)
    xml_path = models['xml_path']
    env = models['env']
    env_ik = models['env_ik']
    env_check = models['env_check']
    env_measured = models['env_measured']
    command_joint_names = models['command_joint_names']
    q_msg_layout = models['q_msg_layout']
    ik_solvers = models['ik_solvers']
    arm_joint_names = models['arm_joint_names']
    guard = models['guard']
    for model_env in (env, env_ik, env_check, env_measured):
        reset_sg2_sim(model_env)
    mapper, planner = build_planner(models, cfg, profile, intent_matrices)
    receiver = publisher = real = server = rp = controls = recorder = None
    ui_actions = []
    speed_requests = []
    gripper_speed_requests = []
    position_gain_requests = []
    joint_preview_requests = []
    tracking_mode_requests = []
    pending_tracking_mode = None
    speed_error = None
    gripper_speed_error = None
    last_stop_reason = ""
    last_home_text = ""
    recent_ik_recovery = None
    last_ik_warning = None   # (time, state, reason, {hand: target xyz}) shown in the viewer for 3 s
    last_targets = {}
    gc_monitor = GCPauseMonitor()
    mode_label = "SIM" if cfg.backend == "sim" else ("SHADOW" if cfg.DRY_RUN else "REAL")
    speed_label = "팔 최대 %.0f°/s · 복귀 %.0f°/s · 시작 ramp %g초" % (
        np.rad2deg(cfg.MAX_JOINT_SPEED), np.rad2deg(cfg.MAX_HOME_JOINT_SPEED), cfg.TELEOP_RAMP_S)
    log_path = PROJECT_DIR / ".cache" / "fixed_quest" / "recordings" / (datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-intent-" + cfg.backend + ".jsonl")
    exit_reason = "startup_incomplete"
    if not isinstance(cfg.QUEST_HOST, str) or not cfg.QUEST_HOST:
        raise ValueError("QUEST_HOST를 Quest와 로봇이 접근할 수 있는 PC LAN IP로 지정하세요.")
    try:
        gc_monitor.start()
        use_pyqt_platform_plugins()
        app = QApplication.instance() or QApplication([])
        if cfg.VIEWER_MODE == "process":
            rp = MuJoCoRenderProcess(xml_path=xml_path, title="SG2 Fixed Quest",
                x_offset=.28, y_offset=0, window_width=.72, window_height=1,
                render_hz=cfg.RENDER_HZ, max_sec=float("inf"),
                log_path=str(PROJECT_DIR / ".cache" / "fixed_quest" / "real_viewer.log"), profile=False)
            rp.start()
        else:
            env.init_viewer(title="SG2 Fixed Quest", x_offset=.28, y_offset=0, width=.72, height=1, fontscale=180)
            env.viewer.set_cam_info(180., 2.8, -20., [0, 0, .95])
        controls = MultiRadioQtWidget(
            title="Fixed Quest coordinated " + mode_label, x_offset=0, y_offset=0, window_width=.28, window_height=.70,
            label_texts=["Mode", "View", "Run"],
            option_texts=[(["SIM"] if cfg.backend == "sim" else ["DRY RUN", "REAL RUN"]), ["Measured", "Preview", "Joint preview"], ["RUN", "STOP"]],
            initial_texts=["SIM" if cfg.backend == "sim" else ("DRY RUN" if cfg.DRY_RUN else "REAL RUN"), "Preview" if cfg.DRY_RUN else "Measured", "RUN"],
        )
        mode_widget = QLabel(("SIM · 이상적 관절 추종" if cfg.backend == "sim" else ("DRY RUN · 미리보기" if cfg.DRY_RUN else "REAL RUN")) + " · " + speed_label, controls)
        mode_widget.setWordWrap(True)
        controls.layout().insertWidget(0, mode_widget)
        speed_row = QWidget(controls)
        speed_row_layout = QHBoxLayout(speed_row)
        speed_row_layout.setContentsMargins(0, 0, 0, 0)
        speed_input = QLineEdit("%g" % np.rad2deg(cfg.MAX_JOINT_SPEED), speed_row)
        speed_input.setObjectName("teleop_speed_input")
        speed_input.setMaxLength(32)
        speed_input.setPlaceholderText("10~120")
        speed_input.setToolTip("10~120°/s 숫자를 입력하고 Enter 또는 적용을 누르세요.")
        speed_apply = QPushButton("적용", speed_row)
        speed_row_layout.addWidget(QLabel("팔 속도 상한 (°/s)", speed_row))
        speed_row_layout.addWidget(speed_input)
        speed_row_layout.addWidget(speed_apply)
        speed_apply.clicked.connect(lambda: speed_requests.append(speed_input.text()))
        speed_input.returnPressed.connect(lambda: speed_requests.append(speed_input.text()))
        speed_status = QLabel("적용: %g°/s · 입력 범위 10~120" % np.rad2deg(cfg.MAX_JOINT_SPEED), controls)
        speed_status.setWordWrap(True)
        controls.layout().insertWidget(1, speed_row)
        controls.layout().insertWidget(2, speed_status)
        gripper_row = QWidget(controls)
        gripper_row_layout = QHBoxLayout(gripper_row)
        gripper_row_layout.setContentsMargins(0, 0, 0, 0)
        gripper_input = QLineEdit("%g" % np.rad2deg(cfg.MAX_GRIPPER_SPEED), gripper_row)
        gripper_input.setObjectName("gripper_speed_input")
        gripper_input.setMaxLength(32)
        gripper_input.setPlaceholderText("10~120")
        gripper_input.setToolTip("10~120°/s 숫자를 입력하고 Enter 또는 적용을 누르세요. 복귀 중에는 복귀 속도를 따릅니다.")
        gripper_apply = QPushButton("적용", gripper_row)
        gripper_apply.setObjectName("gripper_speed_apply")
        gripper_row_layout.addWidget(QLabel("그리퍼 속도 상한 (°/s)", gripper_row))
        gripper_row_layout.addWidget(gripper_input)
        gripper_row_layout.addWidget(gripper_apply)
        gripper_apply.clicked.connect(lambda: gripper_speed_requests.append(gripper_input.text()))
        gripper_input.returnPressed.connect(lambda: gripper_speed_requests.append(gripper_input.text()))
        gripper_status = QLabel("적용: %g°/s · 입력 범위 10~120" % np.rad2deg(cfg.MAX_GRIPPER_SPEED), controls)
        gripper_status.setObjectName("gripper_speed_status")
        gripper_status.setWordWrap(True)
        controls.layout().insertWidget(3, gripper_row)
        controls.layout().insertWidget(4, gripper_status)
        position_row = QWidget(controls)
        position_layout = QGridLayout(position_row)
        position_layout.setContentsMargins(0, 0, 0, 0)
        position_x_input = QLineEdit("%g" % cfg.POSITION_SCALE, position_row)
        position_y_input = QLineEdit("%g" % cfg.POSITION_SCALE_Y, position_row)
        position_z_input = QLineEdit("%g" % cfg.POSITION_SCALE_Z, position_row)
        position_x_input.setObjectName("position_x_input")
        position_y_input.setObjectName("position_y_input")
        position_z_input.setObjectName("position_z_input")
        position_apply = QPushButton("배율 적용", position_row)
        def queue_position_gains():
            position_gain_requests.append((position_x_input.text(), position_y_input.text(), position_z_input.text()))
        for column, (label, field) in enumerate((("전후 X", position_x_input), ("좌우 Y", position_y_input), ("상하 Z", position_z_input))):
            field.setMaxLength(16)
            field.setMaximumWidth(75)
            field.setToolTip("0.05~3.0 · 보정 후 의도적인 상하 이동 배율" if column == 2
                             else "0.05~3.0 · 컨트롤러 이동에 대한 로봇 목표 이동 배율")
            position_layout.addWidget(QLabel(label, position_row), 0, column)
            position_layout.addWidget(field, 1, column)
            field.returnPressed.connect(queue_position_gains)
        position_layout.addWidget(position_apply, 0, 3, 2, 1)
        position_apply.clicked.connect(queue_position_gains)
        position_status = QLabel("적용 배율 X %.2f / Y %.2f / Z %.2f · 양손 grip을 놓고 변경" % (
            cfg.POSITION_SCALE, cfg.POSITION_SCALE_Y, cfg.POSITION_SCALE_Z), controls)
        position_status.setWordWrap(True)
        controls.layout().addWidget(position_row)
        controls.layout().addWidget(position_status)
        collision_status = QLabel("충돌 검사: Teleoperation %s · Initial pose / Reset pose / 수동 조정 ON" % (
            "ON" if cfg.TELEOP_COLLISION_CHECK else "OFF"), controls)
        collision_status.setWordWrap(True)
        controls.layout().addWidget(collision_status)
        tracking_mode_status = None
        if cfg.TRACKING_MODE_BUTTONS:
            tracking_row = QWidget(controls)
            tracking_layout = QGridLayout(tracking_row)
            tracking_layout.setContentsMargins(0, 0, 0, 0)
            tracking_group = QButtonGroup(tracking_row)
            tracking_group.setExclusive(True)
            initial_tracking_mode = {"tracking_pause_s": cfg.TRACKING_PAUSE_S, "accept_emulated": cfg.ACCEPT_EMULATED_POSE}
            shown = [mode for key in cfg.TRACKING_MODE_BUTTONS for mode in TRACKING_MODES if mode[0] == key]
            for index, (key, label, _, _) in enumerate(shown):
                button = QPushButton(label, tracking_row)
                button.setObjectName("tracking_mode_" + key)
                button.setCheckable(True)
                button.setChecked(key == tracking_mode_key(initial_tracking_mode))
                tracking_group.addButton(button)
                tracking_layout.addWidget(button, index // 2, index % 2)
                button.clicked.connect(lambda checked=False, key=key: tracking_mode_requests.append(key))
            tracking_mode_status = QLabel("적용: " + tracking_mode_text(initial_tracking_mode), controls)
            tracking_mode_status.setObjectName("tracking_mode_status")
            tracking_mode_status.setWordWrap(True)
            controls.layout().addWidget(QLabel("추적 손실 모드 · 양손 grip을 놓으면 전환", controls))
            controls.layout().addWidget(tracking_row)
            controls.layout().addWidget(tracking_mode_status)
        home_button = QPushButton("Initial pose", controls)
        home_button.setToolTip("고정된 worker_bringup 자세까지 전체 경로를 검사한 뒤 복귀합니다. head_joint1 목표는 bringup과 같은 0.69 rad로 유지합니다.")
        cancel_button = QPushButton("Cancel return", controls)
        status_label = QLabel("로봇 피드백 대기", controls)
        status_label.setWordWrap(True)
        home_button.clicked.connect(lambda: ui_actions.append("home"))
        cancel_button.clicked.connect(lambda: ui_actions.append("cancel_home"))
        home_button.setEnabled(False)
        cancel_button.setEnabled(False)
        for widget in (home_button, cancel_button, status_label):
            controls.layout().addWidget(widget)
        joint_row = QWidget(controls)
        joint_layout = QGridLayout(joint_row)
        joint_layout.setContentsMargins(0, 0, 0, 0)
        joint_selector = QComboBox(joint_row)
        joint_selector.setObjectName("manual_joint_selector")
        joint_selector.addItems([name for name in command_joint_names if name.startswith("arm_")])
        joint_angle = QLineEdit("0", joint_row)
        joint_angle.setObjectName("manual_joint_angle")
        joint_angle.setPlaceholderText("절대 목표 각도 (°)")
        joint_preview_button = QPushButton("관절 미리보기", joint_row)
        joint_preview_button.setObjectName("manual_joint_preview")
        joint_apply_button = QPushButton("조정 실행 (≤10°/s)", joint_row)
        joint_apply_button.setObjectName("manual_joint_apply")
        joint_apply_button.setEnabled(False)
        joint_current = QLabel("관절 피드백 대기", joint_row)
        joint_status = QLabel("VR 조작을 멈춘 뒤 관절 선택 → 목표 각도 → 미리보기 → 조정 실행 → Initial pose", controls)
        joint_status.setWordWrap(True)
        joint_status.setObjectName("manual_joint_status")
        joint_layout.addWidget(joint_selector, 0, 0)
        joint_layout.addWidget(joint_angle, 0, 1)
        joint_layout.addWidget(joint_current, 1, 0, 1, 2)
        joint_layout.addWidget(joint_preview_button, 2, 0)
        joint_layout.addWidget(joint_apply_button, 2, 1)
        joint_preview_button.clicked.connect(lambda: joint_preview_requests.append((joint_selector.currentText(), joint_angle.text())))
        joint_apply_button.clicked.connect(lambda: ui_actions.append("joint_adjust"))
        def invalidate_joint_preview(*args):
            joint_apply_button.setEnabled(False)
            if real is not None:
                real._joint_preview = None
        joint_selector.currentTextChanged.connect(invalidate_joint_preview)
        joint_angle.textChanged.connect(invalidate_joint_preview)
        controls.layout().addWidget(joint_row)
        controls.layout().addWidget(joint_status)
        if cfg.backend == "sim":
            receiver = publisher = IntentSimBridge(env.get_qpos(joint_names=command_joint_names))
        else:
            receiver = SG2RobotStateReceiver(cfg.ROBOT_IP, cfg.ROBOT_STATE_PORT, cfg.FEEDBACK_TIMEOUT_S).start()
            publisher = SG2RobotCommandPublisher(enabled=not cfg.DRY_RUN, port=cfg.ROBOT_COMMAND_PORT)
        real = SG2FixedQuestReal(planner, env_measured, q_msg_layout, publisher, dry_run=cfg.DRY_RUN,
            max_joint_lag=cfg.MAX_JOINT_LAG, max_lift_lag=cfg.MAX_LIFT_LAG, command_ttl_s=cfg.COMMAND_TTL_S,
            max_home_joint_speed=cfg.MAX_HOME_JOINT_SPEED, max_arm_joint_lag=cfg.MAX_ARM_JOINT_LAG,
            teleop_ramp_s=cfg.TELEOP_RAMP_S, required_bridge_limits=FIXED_QUEST_VR_BRIDGE_LIMITS,
            command_arm_speed_limit=True, tracking_pause_s=cfg.TRACKING_PAUSE_S)
        certfile, keyfile = (cfg.CERTFILE, cfg.KEYFILE) if cfg.CERTFILE else create_self_signed_certificate(PROJECT_DIR / ".cache" / "fixed_quest" / "tls", cfg.QUEST_HOST)
        server = QuestControllerServer(host="0.0.0.0", port=cfg.QUEST_PORT, certfile=certfile, keyfile=keyfile).start()
        print("Quest Browser: https://%s:%d/" % (cfg.QUEST_HOST, server.port))
        if cfg.backend == "real":
            print("Robot workers: SG2_FIXED_QUEST=1 | inbound SG2_ZMQ_SUB_IP=" + cfg.QUEST_HOST)
        print(mode_label, "| 양손 grip으로 시작 · 한쪽이라도 놓으면 양팔 정지")
        print("IK:", "초기 팔 자세 유지 · 첫 조작 자세를 기준으로 고정" if cfg.POSTURE_IK_ENABLED
              else "기존 IK", "| 위치/방향 허용오차:",
              ("%g mm / %g°" % (1000 * cfg.POSTURE_POSITION_TOLERANCE_M,
                                 cfg.POSTURE_ROTATION_TOLERANCE_DEG)) if cfg.POSTURE_IK_ENABLED else "기존 설정")
        profile_widget = QLabel("개인 의도 보정: " + ("미적용 (identity)" if profile is None else profile["profile_id"]), controls)
        profile_widget.setWordWrap(True)
        if cfg.ALLOW_FAILED_PROFILE_IN_SIM:
            profile_widget.setText(profile_widget.text() + '\nSIM 후보 시험 · 궤적 검증 미통과 · REAL 적용 불가')
            profile_widget.setStyleSheet('color: #9a5500; font-weight: bold;')
        elif cfg.REAL_CANDIDATE_REVIEW_PATH is not None:
            profile_widget.setText(profile_widget.text() + '\n사용자 SIM 확인 후 REAL 후보 시험 · 자동 궤적 검증 미통과')
            profile_widget.setStyleSheet('color: #9a5500; font-weight: bold;')
        controls.layout().addWidget(profile_widget)
        mapping_widget = QLabel("입력 → 보정 → 목표 → FK (mm)", controls)
        mapping_widget.setWordWrap(True)
        controls.layout().addWidget(mapping_widget)
        # Keep all controls reachable on smaller laptop screens.
        content = QWidget()
        content.setLayout(controls.layout())
        scroll = QScrollArea(controls)
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)
        outer_layout = QVBoxLayout(controls)
        outer_layout.addWidget(scroll)
        recorder = ControllerRecorder(log_path, command_joint_names, "intent_" + cfg.backend, configuration={
            "pipeline_version": 2, "backend": cfg.backend, "intent_profile": profile,
            "failed_profile_sim_trial": cfg.ALLOW_FAILED_PROFILE_IN_SIM,
            "operator_reviewed_real_candidate": cfg.REAL_CANDIDATE_REVIEW_PATH is not None,
            "real_candidate_review_path": None if cfg.REAL_CANDIDATE_REVIEW_PATH is None else str(cfg.REAL_CANDIDATE_REVIEW_PATH),
            "candidate_capture_path": None if cfg.SIM_CANDIDATE_CAPTURE_PATH is None else str(cfg.SIM_CANDIDATE_CAPTURE_PATH),
            "initial_dry_run": cfg.DRY_RUN, "robot_ip": cfg.ROBOT_IP, "input_mode": "quest", "scale": cfg.POSITION_SCALE,
            "control_source": "both_grips",
            "coordinated_arm_rate_limit": cfg.COORDINATED_ARM_RATE_LIMIT,
            "tracking_pause_s": cfg.TRACKING_PAUSE_S, "retry_ik_near_target": cfg.RETRY_IK_NEAR_TARGET,
            "accept_emulated_pose": cfg.ACCEPT_EMULATED_POSE,
            "tracking_mode_buttons": [{"key": key, "tracking_pause_s": pause, "accept_emulated": emulated}
                                      for button in cfg.TRACKING_MODE_BUTTONS
                                      for key, _, pause, emulated in TRACKING_MODES if key == button],
            "ik_recovery_timeout_s": cfg.IK_RECOVERY_TIMEOUT_S, "null_reference": cfg.NULL_REFERENCE,
            "posture_ik_enabled": cfg.POSTURE_IK_ENABLED,
            "posture_reference": "first_active_measured_pose_including_dry_run",
            "posture_position_tolerance_m": cfg.POSTURE_POSITION_TOLERANCE_M,
            "posture_rotation_tolerance_deg": cfg.POSTURE_ROTATION_TOLERANCE_DEG,
            "posture_elbow_limit_deg": cfg.POSTURE_ELBOW_LIMIT_DEG,
            "position_scales": cfg.POSITION_SCALES, "position_gain_mode": "intent_then_positive_xyz",
            "R_BQ": cfg.R_BQ, "control_hz": cfg.CONTROL_HZ, "ik_hz": cfg.IK_HZ, "max_joint_speed_rad_s": cfg.MAX_JOINT_SPEED,
            "max_aux_joint_speed_rad_s": cfg.MAX_AUX_JOINT_SPEED, "max_gripper_speed_rad_s": cfg.MAX_GRIPPER_SPEED,
            "teleop_ramp_s": cfg.TELEOP_RAMP_S,
            "speed_input_range_deg_s": [10, 120], "command_arm_speed_limit": True,
            "max_arm_joint_lag_rad": cfg.MAX_ARM_JOINT_LAG,
            "max_home_joint_speed_rad_s": cfg.MAX_HOME_JOINT_SPEED,
            "home_target": "first_measured_pose_of_run",
            "home_path_strategies": [name for name, _ in HOME_PATH_STRATEGIES],
            "feedback_timeout_s": cfg.FEEDBACK_TIMEOUT_S, "command_ttl_s": cfg.COMMAND_TTL_S, "collision_margin_m": cfg.COLLISION_MARGIN_M,
            "teleop_collision_check": cfg.TELEOP_COLLISION_CHECK,
            "home_collision_check": True,
            "collision_escape_min_m": cfg.COLLISION_ESCAPE_MIN_M,
        })
        next_control = next_render = time.monotonic()
        while True:
            app.processEvents()
            if not controls.isVisible() or controls.get_button_option("Run") == "STOP":
                exit_reason = "PC_STOP_or_control_window_closed"
                break
            if not (rp.is_running() if rp is not None else env.is_viewer_alive()):
                exit_reason = "viewer_closed_or_exited"
                break
            now = time.monotonic()
            if now < next_control:
                time.sleep(min(.001, next_control - now))
                continue
            mode_changed = False
            mode_change_error = None
            selected_dry_run = cfg.backend == "real" and controls.get_button_option("Mode") == "DRY RUN"
            if selected_dry_run != real.dry_run:
                ui_actions.clear()
                joint_preview_requests.clear()
                invalidate_joint_preview()
                try:
                    mode_changed = real.set_dry_run(selected_dry_run,
                        lambda enabled: SG2RobotCommandPublisher(enabled=enabled, port=cfg.ROBOT_COMMAND_PORT))
                    last_stop_reason = "모드 전환 완료 · 양손 grip을 놓았다가 다시 잡으면 시작합니다."
                except Exception as mode_error:
                    mode_change_error = str(mode_error)
                    last_stop_reason = "모드 전환 실패 · DRY RUN 유지: " + mode_change_error
                    print(last_stop_reason)
                publisher = real.publisher
                controls.set_button_option("Mode", "DRY RUN" if real.dry_run else "REAL RUN")
                controls.set_button_option("View", "Preview" if real.dry_run else "Measured")
                mode_label = "SHADOW" if real.dry_run else "REAL"
                mode_widget.setText(("SIM · 이상적 관절 추종" if cfg.backend == "sim" else ("DRY RUN · 미리보기" if real.dry_run else "REAL RUN")) + " · " + speed_label)
                controls.setWindowTitle("Fixed Quest coordinated " + mode_label)
            # Sample after mode switching and feedback polling; validate both
            # inputs at the same observation time after copying Quest input.
            feedback, frame, now = sample_control_inputs(server, receiver, simulated=cfg.backend == "sim")
            action = ui_actions[-1] if ui_actions else None
            ui_actions.clear()
            speed_change = None
            if speed_requests:
                requested_text = speed_requests[-1].strip()
                speed_requests.clear()
                try:
                    requested_speed = float(requested_text)
                    changed = real.set_teleop_speed(requested_speed)
                    speed_error = None
                    speed_change = {"requested_deg_s": requested_speed, "accepted": True, "changed": changed}
                except (ValueError, OverflowError):
                    speed_error = "10~120°/s의 유효한 숫자를 입력하세요. 기존 속도를 유지합니다."
                    speed_change = {"requested_text": requested_text, "accepted": False, "reason": speed_error}
            gripper_speed_change = None
            if gripper_speed_requests:
                requested_text = gripper_speed_requests[-1].strip()
                gripper_speed_requests.clear()
                try:
                    requested_speed = float(requested_text)
                    changed = real.set_gripper_speed(requested_speed)
                    gripper_speed_error = None
                    gripper_speed_change = {"requested_deg_s": requested_speed, "accepted": True, "changed": changed}
                except (ValueError, OverflowError) as exc:
                    gripper_speed_error = str(exc)
                    gripper_speed_change = {"requested_text": requested_text, "accepted": False,
                                            "reason": gripper_speed_error}
            tick_started = time.perf_counter()
            tick_cpu_started = time.thread_time()
            control_frame = None if getattr(on_tick, "lock_input", False) else frame
            result = real.tick_grip(control_frame, feedback, now=now, action=action)
            if action == "joint_adjust":
                joint_status.setText(result["reason"])
                controls.set_button_option("View", "Preview")
            elif action == "cancel_home":
                joint_status.setText("수동 조정/복귀 중단 · 현재 자세에서 다시 미리보기 하세요.")
            if result.get("manual_adjustment"):
                joint_status.setText("수동 관절 조정 완료 · Initial pose로 복귀 가능" if result["state"] == "HOME"
                                     else "수동 관절 조정 중 · Cancel return으로 중단")
            if joint_preview_requests:
                joint, angle = joint_preview_requests[-1]
                joint_preview_requests.clear()
                try:
                    preview = real.preview_joint_adjustment(joint, float(angle), feedback)
                    result["joint_preview"] = dict(preview, accepted=True)
                    joint_status.setText("%s → %.2f° · 경로 확인 · 약 %.1f초. 조정 실행을 누르면 움직입니다." %
                                         (joint, preview["degrees"], preview["duration_s"]))
                    controls.set_button_option("View", "Joint preview")
                except (ValueError, OverflowError) as exc:
                    result["joint_preview"] = {"accepted": False, "reason": str(exc)}
                    joint_status.setText(str(exc))
            position_gain_change = None
            if position_gain_requests:
                requested_gains = position_gain_requests[-1]
                position_gain_requests.clear()
                try:
                    if not all(np.isfinite(float(x)) and .05 <= float(x) <= 3 for x in requested_gains):
                        raise ValueError("XYZ 배율은 0.05~3.0 양수입니다. 높이 고정/반전은 의도 보정에 사용하지 않습니다.")
                    applied_gains = real.set_position_gains(
                        float(requested_gains[1]), float(requested_gains[2]), now=time.monotonic(),
                        forward=float(requested_gains[0]))
                    position_gain_change = {"accepted": True, "position_scales": applied_gains}
                    gx, gy, gz = applied_gains["left"]
                    position_status.setText("적용 배율 X %g / Y %g / Z %g · 편한 자세에서 양손 grip" % (gx, gy, gz))
                except (ValueError, OverflowError) as gain_error:
                    position_gain_change = {"accepted": False, "requested": requested_gains, "reason": str(gain_error)}
                    current_gains = mapper.position_scales["left"]
                    position_status.setText("배율 유지 X %g / Y %g / Z %g · %s" % (*current_gains, str(gain_error)))
            result["position_gain_change"] = position_gain_change
            tracking_mode_change = None
            if tracking_mode_requests:
                pending_tracking_mode = tracking_mode_requests[-1]
                tracking_mode_requests.clear()
            if pending_tracking_mode is not None:
                if tracking_mode_key(real.tracking_loss_mode) == pending_tracking_mode:
                    pending_tracking_mode = None
                else:
                    _, _, pause, emulated = next(mode for mode in TRACKING_MODES if mode[0] == pending_tracking_mode)
                    try:
                        # Retried every tick; applies only once both grips are released and idle.
                        applied = real.set_tracking_loss_mode(pause, emulated, now=time.monotonic())
                        tracking_mode_change = dict(applied, mode=pending_tracking_mode, accepted=True)
                        pending_tracking_mode = None
                    except ValueError:
                        pass
            result["tracking_mode_change"] = tracking_mode_change
            result["tracking_mode_pending"] = pending_tracking_mode
            result["control_compute_ms"] = (time.perf_counter() - tick_started) * 1000
            result["control_thread_cpu_ms"] = (time.thread_time() - tick_cpu_started) * 1000
            result["gc_events_since_previous_tick"] = gc_monitor.take_events()
            result["speed_change"] = speed_change
            result["gripper_speed_change"] = gripper_speed_change
            speed_label = "팔 상한 %g°/s · 그리퍼 %g°/s · 복귀 %g°/s · 시작 ramp %g초" % (
                result["teleop_speed_deg_s"], result["gripper_speed_deg_s"],
                np.rad2deg(cfg.MAX_HOME_JOINT_SPEED), cfg.TELEOP_RAMP_S)
            mode_widget.setText(("SIM · 이상적 관절 추종" if cfg.backend == "sim" else ("DRY RUN · 미리보기" if real.dry_run else "REAL RUN")) + " · " + speed_label)
            if speed_error:
                speed_status.setText(speed_error)
            elif result["speed_change_pending"]:
                speed_status.setText("적용 중: %g → %g°/s · 이전 명령 수신 확인 대기" % (
                    result["teleop_speed_deg_s"], result["requested_teleop_speed_deg_s"]))
            else:
                speed_status.setText("적용: %g°/s · 입력 범위 10~120" % result["teleop_speed_deg_s"])
            if gripper_speed_error:
                gripper_status.setText(gripper_speed_error)
            elif result["gripper_speed_change_pending"]:
                gripper_status.setText("적용 중: %g → %g°/s · 이전 명령 수신 확인 대기" % (
                    result["gripper_speed_deg_s"], result["requested_gripper_speed_deg_s"]))
            else:
                gripper_status.setText("적용: %g°/s · bridge 상한 %.0f°/s" %
                                       (result["gripper_speed_deg_s"], np.rad2deg(real.gripper_bridge_cap(feedback))))
            if real.initialized and feedback["ready"] and real.generation == feedback["generation"]:
                receiver.acknowledge_resync(real.generation)
            home_text = home_status_text(result)
            if result.get("home_plan") is not None:
                # Keep the checked path visible while arming and returning.
                last_home_text = home_text
            if result["needs_disable"]:
                last_stop_reason = result["reason"]
                if result.get("home_plan") is not None and not result["home_plan"]["ok"]:
                    last_stop_reason = home_text
            elif result["state"] in ("ARMING", "ACTIVE", "HOMING", "IK_WAIT"):
                last_stop_reason = ""
            recovery_text = ik_recovery_status_text(result)
            if result["state"] != "ACTIVE":
                recent_ik_recovery = None
            elif (result.get("ik_recovery") or {}).get("status") == "recovered":
                # A recovery event may last one control tick; leave it visible
                # briefly without carrying it into a later pause or stop.
                recent_ik_recovery = (now, recovery_text)
            elif recent_ik_recovery is not None and now - recent_ik_recovery[0] >= 3.:
                recent_ik_recovery = None
            result["mode_changed"] = mode_changed
            result["mode_change_error"] = mode_change_error
            result["feedback_reason"] = feedback["reason"]
            result["feedback_diagnostics"] = feedback.get("diagnostics")
            result["bridge_state"] = feedback["bridge_state"]
            result["bridge_reason"] = (feedback["bridge"] or {}).get("reason", "")
            result["last_command_seq"] = feedback["last_command_seq"]
            result["bridge_limits"] = (feedback["bridge"] or {}).get("limits")
            result["bridge_rejection_detail"] = (feedback["bridge"] or {}).get("rejection_detail")
            execution_diagnostics(result, env, env_measured)
            for hand, T in (result.get("targets") or {}).items():
                last_targets[hand] = np.asarray(T, dtype=float)[:3, 3].copy()
            if "IK" in result["reason"] and result["state"] in ("IK_WAIT", "HOLD", "FAULT"):
                ik_hands = [hand for hand in SG2_HAND_TYPES if hand in result["reason"]] or list(SG2_HAND_TYPES)
                last_ik_warning = (now, result["state"], result["reason"],
                                   {hand: last_targets[hand] for hand in ik_hands if hand in last_targets})
            result["backend"] = cfg.backend
            recorder.write(now, frame, result)
            if on_tick is not None:
                on_tick(now, frame, result, real, ui_actions)
            next_control = now + 1 / cfg.CONTROL_HZ
            if now >= next_render:
                rows = []
                for hand, stage in result.get("mapping_stages", {}).items():
                    fmt = lambda p: np.array2string(1000 * np.asarray(p), precision=1, suppress_small=True)
                    execution = result["execution_stages"][hand]
                    rows.append(hand + " 입력 " + fmt(stage["operator_delta_m"])
                        + "\n의도 " + fmt(stage["intent_delta_m"]) + " / 배율 후 " + fmt(stage["requested_delta_m"])
                        + "\n작업범위 배율 %.2f / 목표–명령 " % stage["workspace_scale"]
                        + fmt(execution["bounded_to_command_error_m"]))
                mapping_widget.setText("\n".join(rows) or "양손 grip 대기 · 높이 고정 없음")
                view = controls.get_button_option("View")
                display_env = env_measured if view == "Measured" else env
                if view == "Joint preview" and real._joint_preview is not None:
                    display_env = env_ik
                measured_frames = [get_sg2_wrist_control_T(env_measured, hand) for hand in SG2_HAND_TYPES] if real.initialized else []
                frames = measured_frames + list(result["targets"].values())
                age_text = "--" if feedback["age_s"] is None else "%.0f ms" % (1000 * feedback["age_s"])
                lag_text = "--" if result["robot_q"] is None else "%.2f deg / lift %.1f mm" % (
                    np.rad2deg(np.max(np.abs(result["q_command"][1:] - result["robot_q"][1:]))),
                    1000 * abs(result["q_command"][0] - result["robot_q"][0]))
                returning = result["homing"] or result.get("home_pending", False)
                home_available = real.initialized and feedback["ready"] and not returning
                manual_available = home_available and not real.armed and not real._grip_engaged
                joint_preview_button.setEnabled(manual_available)
                joint_apply_button.setEnabled(manual_available and real._joint_preview is not None)
                if feedback.get("q") is not None:
                    joint_index = command_joint_names.index(joint_selector.currentText())
                    joint_current.setText("현재 %.2f° · Initial %.2f°" %
                                          (np.rad2deg(feedback["q"][joint_index]), np.rad2deg(planner.q_home[joint_index])))
                home_button.setEnabled(home_available)
                if returning:
                    home_button.setToolTip("시작 자세로 복귀 중입니다. Cancel return 또는 Run STOP으로 중단합니다.")
                elif not feedback["ready"] or not real.initialized:
                    home_button.setToolTip("복귀하려면 로봇의 정상 실측 피드백이 필요합니다: " + feedback["reason"])
                else:
                    home_button.setToolTip(
                        "그립이나 Quest 추적 없이 이번 실행의 첫 실측 자세로 복귀합니다. 누르는 순간 전체 경로를 충돌 검사하고, "
                        "직선 경로가 막히면 J3·J5 먼저 풀기 등 단계 경로를 사용합니다. 모든 경로가 막히면 움직이지 않습니다. "
                        "REAL RUN은 로봇 승인 후 10°/s로 복귀하고, DRY RUN은 미리보기만 복귀합니다.")
                cancel_button.setEnabled(returning)
                if tracking_mode_status is not None:
                    tracking_text = "적용: " + tracking_mode_text(real.tracking_loss_mode)
                    if pending_tracking_mode is not None:
                        pending_label = next(mode[1] for mode in TRACKING_MODES if mode[0] == pending_tracking_mode)
                        tracking_text += "\n대기: %s · 양손 grip을 놓고 추적이 유효하면 전환" % pending_label
                    tracking_mode_status.setText(tracking_text)
                tracking_status = {}
                for hand in SG2_HAND_TYPES:
                    if frame is None:
                        tracking_status[hand] = "Quest 메시지 대기"
                    elif now - frame["received_at"] > cfg.INPUT_TIMEOUT_S:
                        tracking_status[hand] = "Quest 입력 지연"
                    else:
                        hand_input = frame["hands"][hand]
                        diagnostics = frame.get("diagnostics", {})
                        detail = diagnostics.get("hands", {}).get(hand, {})
                        # Explain the actual gate, including packets from an older
                        # Quest page whose hand reason says only "추정 위치 · 사용 중지".
                        if frame.get("visibility") != "visible":
                            tracking_status[hand] = "XR 화면 비활성 · 입력 차단"
                        elif detail.get("blocked_by") == "viewer_pose":
                            tracking_status[hand] = "헤드셋 위치 추적 손실 · 양손 입력 차단 · Quest XR 재시작/추적 확인"
                        elif hand_input["valid"] and not hand_input["emulated"]:
                            tracking_status[hand] = "추적 유효 · grip %.2f" % hand_input["grip"]
                        elif hand_input["emulated"] and mapper.accept_emulated:
                            tracking_status[hand] = "추정 위치 사용 중 · grip %.2f" % hand_input["grip"]
                        elif hand_input["emulated"]:
                            tracking_status[hand] = "컨트롤러 추정 위치 · PC 설정에서 사용 중지"
                        else:
                            tracking_status[hand] = detail.get("reason", "컨트롤러 추적 없음")
                            if detail.get("blocked_by"):
                                tracking_status[hand] += " · " + detail["blocked_by"]
                if result["state"] == "IK_WAIT":
                    state_text = recovery_text or "IK 재시도 중 · 현재 명령 유지"
                elif result["state"] == "TRACKING_PAUSED":
                    state_text = "추적 일시정지 · 현재 명령 유지 · 최대 %.0fms 복구 대기" % (1000 * real.tracking_pause_s)
                elif result["state"] in ("DISABLED", "WAIT_GRIP"):
                    state_text = "WAIT_GRIP · 양손 grip을 모두 잡으면 시작"
                elif result["state"] == "WAIT_RELEASE":
                    state_text = "WAIT_RELEASE · 양손 grip을 놓았다가 다시 잡으세요"
                elif result["state"] == "WAIT_INPUT":
                    state_text = "WAIT_INPUT · 양손 추적 대기 (명령 전송 없음)"
                else:
                    state_text = result["state"] + " · " + result["reason"]
                if result["state"] != "IK_WAIT":
                    if recovery_text:
                        state_text += "\n" + recovery_text
                    elif recent_ik_recovery is not None:
                        state_text += "\n" + recent_ik_recovery[1]
                if returning and (result.get("home_path") or last_home_text):
                    state_text += "\n" + (home_text if result.get("home_path") else last_home_text)
                if result["state"] == "ACTIVE" and result.get("teleop_ramp_scale", 1.) < 1.:
                    state_text += " · 시작 응답 %.0f%%" % (100 * result["teleop_ramp_scale"])
                if result.get("tracking_paced"):
                    state_text += " · 로봇이 따라오는 속도에 맞춰 조절 중"
                if result.get("ack_paced"):
                    state_text += " · 로봇 수신 확인 대기"
                status_label.setText("%s\nRobot: %s (%s)\n왼손: %s\n오른손: %s\nCommand–measured: %s" % (
                    state_text, feedback["reason"], age_text, tracking_status["left"], tracking_status["right"], lag_text)
                    + ("\n최근 중단: " + last_stop_reason if last_stop_reason else "")
                    + "\nBridge: " + result["bridge_reason"]
                    + ("\n버튼: " + on_tick.status if getattr(on_tick, "status", "") else "")
                    + feedback_diagnostic_text(feedback))
                overlays = [(mode_label, "%s | %s | %s" % (view, result["state"], result["reason"])),
                            ("Feedback", "%s | age %s | bridge %s" % (feedback["reason"], age_text, feedback["bridge_state"])),
                            ("Command / measured", lag_text)]
                ik_spheres = []
                if last_ik_warning is not None and now - last_ik_warning[0] < 3.:
                    _, ik_state, _, ik_targets = last_ik_warning
                    ik_side = "/".join(ik_targets) or "arm"
                    overlays.insert(0, ("!! IK FAILED (%s)" % ik_side,
                        "target unreachable - retrying, move hand back" if ik_state == "IK_WAIT"
                        else "stopped - move hand back, release both grips, grab again"))
                    ik_spheres = list(ik_targets.values())
                if rp is not None:
                    rp.render(display_env, mode="snapshot", overlays=[rp.overlay(label, text, loc="top left") for label, text in overlays],
                        draws=[rp.draw("plot_global_coordinate_axes")] + [rp.draw("plot_T", T=T, axis_len=.1) for T in frames]
                              + [rp.draw("plot_sphere", p=p, r=.04, rgba=[1., 0., 0., .6], label="IK FAIL") for p in ik_spheres])
                else:
                    # The direct viewer belongs to env; measured q is passed only for drawing.
                    preview_qpos = env.data.qpos.copy()
                    try:
                        env.data.qpos[:] = display_env.data.qpos
                        env.forward()
                        env.plot_global_coordinate_axes()
                        for T in frames:
                            env.plot_T(T=T, axis_len=.1)
                        for p in ik_spheres:
                            env.plot_sphere(p=p, r=.04, rgba=[1., 0., 0., .6], label="IK FAIL")
                        for label, text in overlays:
                            env.viewer_text_overlay(label, text, loc="top left")
                        env.render()
                    finally:
                        env.data.qpos[:] = preview_qpos
                        env.forward()
                next_render = now + 1 / cfg.RENDER_HZ
    except KeyboardInterrupt:
        exit_reason = "keyboard_interrupt"
    except Exception:
        exit_reason = "runtime_exception"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.with_suffix(".error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        # Release the robot lease before closing transports, even if another cleanup fails.
        for obj in (real, publisher, receiver, server, rp, controls, recorder, gc_monitor):
            if obj is not None:
                try:
                    obj.stop() if obj is real else obj.close()
                except Exception as cleanup_error:
                    print("Cleanup:", type(obj).__name__, cleanup_error)
        if getattr(env, "viewer", None) is not None:
            env.close_viewer()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.with_suffix(".status.txt").write_text(exit_reason + "\n", encoding="utf-8")
        print("종료:", exit_reason, "| Log:", log_path)
        print("마지막 셀을 다시 실행할 수 있습니다.")
    return log_path
