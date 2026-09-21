// WebXR native axes: +X right, +Y up, -Z forward. Mapping belongs to Python.
// Pose source: gripSpace, never targetRaySpace or a body/hand skeleton.
const diagnosticNames = ["secure", "xr", "support", "socket", "session", "visibility", "viewer", "sources", "left", "right", "sent", "event"];
const ui = Object.fromEntries(["status", "start", "stop", "connect", "left", "right", "error",
  ...diagnosticNames.map((name) => `diag-${name}`)]
  .map((name) => [name, document.getElementById(name)]));
let socket = null;
let session = null;
let referenceSpace = null;
let sessionId = "";
let originId = 0;
let sequence = 0;
let lastSentTime = -Infinity;
let lastUIUpdate = -Infinity;
let gl = null;
let supportsImmersiveXR = false;
let reconnectTimer = null;
let reconnectAttempt = 0;
let reconnectAllowed = true;
let hasConnected = false;
let startingSession = false;
let transportError = false;
let head = {position_m: [0, 0, 0], quaternion_xyzw: [0, 0, 0, 1]};
let trackingDiagnostics = {};
// Face buttons for PC-side shortcuts (xr-standard: 3 thumbstick, 4 A/X, 5 B/Y).
// Read even without a pose; the Meta/menu buttons are reserved by the system.
function emptyButtons() {
  return {left: {x: false, y: false, stick: false}, right: {a: false, b: false, stick: false}};
}
let buttons = emptyButtons();
const diagnosticText = {secure: window.isSecureContext ? "예" : "아니요",
  xr: navigator.xr ? "있음" : "없음", support: "확인 중", socket: "연결 전",
  session: "시작 전", visibility: "—", viewer: "프레임 대기", sources: "0개",
  left: "프레임 대기", right: "프레임 대기", sent: "0개", event: "페이지 열림"};

function showDiagnostics() {
  diagnosticText.sent = `${sequence}개 · 전송 대기 ${socket?.bufferedAmount ?? 0} byte`;
  for (const name of diagnosticNames) ui[`diag-${name}`].textContent = diagnosticText[name];
}

function updateStartButton() {
  ui.start.disabled = Boolean(session) || startingSession || !supportsImmersiveXR
    || socket?.readyState !== WebSocket.OPEN;
}

function cancelReconnect() {
  if (reconnectTimer !== null) clearTimeout(reconnectTimer);
  reconnectTimer = null;
}

function scheduleReconnect() {
  if (!reconnectAllowed || reconnectTimer !== null) return;
  const delay = Math.min(500 * 2 ** Math.min(reconnectAttempt++, 4), 5000);
  diagnosticText.socket = `${delay / 1000}초 후 PC 재연결 · ${reconnectAttempt}차`;
  ui.status.textContent = session ? "XR 유지 중 · PC 재연결 대기" : "PC 재연결 대기 · XR 시작 전";
  showDiagnostics();
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, delay);
}

function showError(stage, error) {
  const detail = `${error?.name ?? "Error"}: ${error?.message ?? String(error)}`;
  ui.error.textContent = `${stage} — ${detail}`;
  diagnosticText.event = ui.error.textContent;
  showDiagnostics();
}

function emptyHand() {
  return {position_m: [0, 0, 0], quaternion_xyzw: [0, 0, 0, 1],
    valid: false, emulated: false, grip: 0, trigger: 0};
}

function poseData(pose) {
  const {position: p, orientation: q} = pose.transform;
  return {position_m: [p.x, p.y, p.z], quaternion_xyzw: [q.x, q.y, q.z, q.w]};
}

function sendFrame(timestamp, hands, visibility = session?.visibilityState ?? "hidden") {
  // A queued packet will already be stale by the time another pose is ready.
  if (!socket || socket.readyState !== WebSocket.OPEN || socket.bufferedAmount !== 0) return;
  socket.send(JSON.stringify({version: 1, session_id: sessionId, origin_id: originId,
    seq: sequence++, sample_time_ms: timestamp, visibility, head, hands, buttons,
    diagnostics: {...trackingDiagnostics, visibility, sent_frames: sequence}}));
}

function invalidate(visibility = "hidden") {
  buttons = emptyButtons();
  trackingDiagnostics = {...trackingDiagnostics, input_invalidated: true,
    invalidated_reason: diagnosticText.event.slice(0, 300)};
  sendFrame(performance.now(), {left: emptyHand(), right: emptyHand()}, visibility);
  ui.left.textContent = "입력 중지";
  ui.right.textContent = "입력 중지";
}

async function checkXRSupport() {
  try {
    supportsImmersiveXR = Boolean(navigator.xr && await navigator.xr.isSessionSupported("immersive-vr"));
    diagnosticText.support = supportsImmersiveXR ? "지원됨" : "지원되지 않음";
    updateStartButton();
    if (!supportsImmersiveXR) {
      ui.status.textContent = "XR 미지원 브라우저 · 입력 연결 안 함";
      diagnosticText.socket = "연결하지 않음 · XR 미지원";
      ui.connect.hidden = true;
      ui.error.textContent = "이 입력 페이지는 Quest Browser에서 여세요. PC의 SIM 제어는 노트북에서 합니다. Quest에서 이 안내가 보이면 아래 보안 컨텍스트와 WebXR 상태를 확인하세요.";
    }
    showDiagnostics();
    return supportsImmersiveXR;
  } catch (error) {
    diagnosticText.support = "확인 실패";
    diagnosticText.socket = "연결하지 않음 · VR 지원 확인 실패";
    showError("VR 지원 확인", error);
    ui.start.disabled = true;
    return false;
  }
}

function connect() {
  cancelReconnect();
  if (!reconnectAllowed || !supportsImmersiveXR) return;
  if (socket && [WebSocket.OPEN, WebSocket.CONNECTING].includes(socket.readyState)) return;
  ui.status.textContent = session ? "XR 유지 중 · PC 연결 중" : "PC 연결 중…";
  ui.connect.hidden = true;
  diagnosticText.socket = "연결 중";
  const connection = new WebSocket(`${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`);
  socket = connection;
  showDiagnostics();
  connection.addEventListener("open", () => {
    if (socket !== connection) return;
    cancelReconnect();
    reconnectAttempt = 0;
    if (hasConnected && session) {
      // Also invalidate calibration after very short network outages that the
      // notebook's polling loop may not have observed as a disconnect.
      originId += 1;
      diagnosticText.event = "PC 재연결 · 양손 grip을 놓은 뒤 다시 시작";
      invalidate(session.visibilityState);
    }
    hasConnected = true;
    if (transportError) ui.error.textContent = "";
    transportError = false;
    ui.status.textContent = session ? "XR 실행 중 · PC 연결됨" : "PC 수신기 연결됨 · Quest에서 XR 시작";
    diagnosticText.socket = `${location.protocol === "https:" ? "WSS" : "WS"} 연결됨`;
    ui.connect.hidden = true;
    updateStartButton();
    showDiagnostics();
  });
  connection.addEventListener("close", (event) => {
    if (socket !== connection) return;
    ui.status.textContent = session ? "PC 연결 끊김 · XR 유지 중" : "PC 연결 끊김 · XR 시작 전";
    updateStartButton();
    ui.connect.hidden = false;
    diagnosticText.socket = `연결 종료 · 코드 ${event.code ?? "—"}`;
    if (event.reason) {
      ui.error.textContent = event.reason;
      transportError = true;
    }
    showDiagnostics();
    // Reconnecting a transport needs no new immersive-session request. Keep
    // the existing XR reference space alive while the notebook restarts.
    if (![1003, 1008].includes(event.code)) scheduleReconnect();
    else {
      ui.status.textContent = "PC에서 연결 거절 · 오류 확인 후 다시 연결";
      if (event.code === 1008 && event.reason === "Quest connection resumed in another socket") {
        ui.status.textContent = "새 Quest 연결로 전환됨 · 이전 연결 종료";
        ui.error.textContent = "같은 XR 세션이 새 연결에서 이어지고 있습니다. 이 이전 탭은 닫고, 새 연결을 사용하세요.";
      } else {
        ui.error.textContent = event.code === 1008
          ? "다른 Quest 탭이 이미 입력을 보내고 있습니다. 다른 Quest 탭을 닫거나 그 탭의 XR을 종료한 뒤, 여기서 ‘PC에 지금 다시 연결’을 누르세요."
          : "수신기가 입력 형식을 거절했습니다. 최신 수신기를 실행하고 이 페이지를 새로고침한 뒤 다시 시작하세요.";
      }
      transportError = true;
      diagnosticText.socket += " · 자동 재시도 중지";
      showDiagnostics();
    }
  });
  connection.addEventListener("error", () => {
    if (socket !== connection) return;
    ui.error.textContent = "PC 수신기에 연결할 수 없습니다. 노트북 실행 상태와 HTTPS 연결을 확인하세요.";
    transportError = true;
    diagnosticText.socket = "연결 오류";
    showDiagnostics();
  });
}

async function start() {
  if (session || startingSession) return;
  startingSession = true;
  ui.error.textContent = "";
  ui.start.disabled = true;
  diagnosticText.session = "시작 요청 중";
  showDiagnostics();
  try {
    // requestSession must run directly within the user's button activation.
    session = await navigator.xr.requestSession("immersive-vr", {requiredFeatures: ["local"]});
    sessionId = crypto.randomUUID();
    originId = 0;
    sequence = 0;
    lastSentTime = -Infinity;
    lastUIUpdate = -Infinity;
    trackingDiagnostics = {};
    diagnosticText.session = "실행 중";
    diagnosticText.visibility = session.visibilityState;
    diagnosticText.viewer = "첫 프레임 대기";
    diagnosticText.event = "XR 세션 시작";
    head = {position_m: [0, 0, 0], quaternion_xyzw: [0, 0, 0, 1]};
    session.addEventListener("end", () => {
      trackingDiagnostics.session_running = false;
      try {
        invalidate();
      } catch (error) {
        showError("입력 종료 알림 전송", error);
      }
      session = null;
      referenceSpace = null;
      // Release sender ownership immediately. An idle reconnect can prepare
      // the next Start button, but never sends poses or starts XR by itself.
      socket?.close(1000, "XR session ended");
      ui.stop.disabled = true;
      updateStartButton();
      ui.status.textContent = "XR 종료됨 · Quest에서 다시 시작 필요";
      diagnosticText.session = "종료됨";
      showDiagnostics();
    });
    session.addEventListener("visibilitychange", () => {
      diagnosticText.visibility = session.visibilityState;
      diagnosticText.event = `표시 상태 변경: ${session.visibilityState}`;
      if (session.visibilityState !== "visible") invalidate(session.visibilityState);
      showDiagnostics();
    });
    session.addEventListener("inputsourceschange", () => {
      diagnosticText.event = "입력 장치 구성 변경";
      invalidate();
      showDiagnostics();
    });
    referenceSpace = await session.requestReferenceSpace("local");
    referenceSpace.addEventListener("reset", () => {
      originId += 1;
      diagnosticText.event = `좌표계 초기화: ${originId}`;
      invalidate();
      showDiagnostics();
    });
    const canvas = document.createElement("canvas");
    gl = canvas.getContext("webgl", {xrCompatible: true});
    if (!gl) throw new Error("XR 표시용 WebGL을 초기화할 수 없습니다.");
    await gl.makeXRCompatible();
    session.updateRenderState({baseLayer: new XRWebGLLayer(session, gl)});
    ui.stop.disabled = false;
    ui.status.textContent = socket?.readyState === WebSocket.OPEN
      ? "XR 실행 중 · PC 연결됨" : "XR 유지 중 · PC 재연결 대기";
    showDiagnostics();
    session.requestAnimationFrame(onFrame);
  } catch (error) {
    diagnosticText.session = "시작 실패";
    showError("VR 세션 시작", error);
    if (session) await session.end().catch((endError) => showError("세션 종료", endError));
  } finally {
    startingSession = false;
    updateStartButton();
  }
}

function onFrame(timestamp, frame) {
  const activeSession = frame.session;
  if (session !== activeSession) return;
  try {
    updateFrame(timestamp, frame, activeSession);
    activeSession.requestAnimationFrame(onFrame);
  } catch (error) {
    // A failed frame must not leave the previous valid pose streaming.
    diagnosticText.session = "프레임 오류 · 종료 중";
    trackingDiagnostics.frame_error = `${error?.name ?? "Error"}: ${error?.message ?? String(error)}`.slice(0, 300);
    showError("XR 프레임 처리", error);
    try {
      invalidate();
    } catch (sendError) {
      showError("입력 중단 알림 전송", sendError);
      socket?.close(1011, "frame error");
    }
    activeSession.end().catch((endError) => showError("세션 종료", endError));
  }
}

function sourceSummary(source) {
  return {handedness: source.handedness, has_hand: Boolean(source.hand),
    grip_space: Boolean(source.gripSpace), gamepad_mapping: source.gamepad?.mapping ?? null,
    button_count: source.gamepad?.buttons?.length ?? 0,
    profiles: Array.from(source.profiles ?? []).slice(0, 4).map((profile) => String(profile).slice(0, 80))};
}

function updateFrame(timestamp, frame, activeSession) {
  const hands = {left: emptyHand(), right: emptyHand()};
  const nextButtons = emptyButtons();
  const viewerPose = frame.getViewerPose(referenceSpace);
  if (viewerPose) head = poseData(viewerPose);
  const inputSources = Array.from(activeSession.inputSources);
  trackingDiagnostics = {secure_context: Boolean(window.isSecureContext), xr_api_available: Boolean(navigator.xr),
    session_running: true, visibility: activeSession.visibilityState,
    observation_time_ms: timestamp, input_invalidated: false,
    viewer_pose_available: Boolean(viewerPose), viewer_pose_emulated: viewerPose?.emulatedPosition ?? null,
    source_count: inputSources.length, sources: inputSources.slice(0, 4).map(sourceSummary), hands: {}};
  diagnosticText.visibility = activeSession.visibilityState;
  diagnosticText.viewer = !viewerPose ? "위치 없음" : viewerPose.emulatedPosition ? "추정 위치 · 사용 중지" : "위치 유효";
  diagnosticText.sources = `${inputSources.length}개`;
  for (const handedness of ["left", "right"]) {
    const matching = inputSources.filter((source) => source.handedness === handedness);
    const sources = matching.filter((source) => !source.hand);
    const detail = {source_count: matching.length, controller_source_count: sources.length,
      hand_source_count: matching.filter((source) => source.hand).length,
      grip_space: false, gamepad_mapping: null, button_count: 0, pose_available: false, pose_emulated: null};
    let reason = sources.length === 0 ? "컨트롤러 장치 없음" : sources.length > 1 ? "같은 손 장치가 여러 개" : "";
    if (sources.length === 1) {
      const source = sources[0];
      Object.assign(detail, sourceSummary(source));
      const pressed = (index) => Boolean(source.gamepad?.buttons?.[index]?.pressed);
      if (source.gamepad?.mapping === "xr-standard") {
        nextButtons[handedness] = handedness === "left"
          ? {x: pressed(4), y: pressed(5), stick: pressed(3)}
          : {a: pressed(4), b: pressed(5), stick: pressed(3)};
      }
      const pose = source.gripSpace && activeSession.visibilityState === "visible"
        ? frame.getPose(source.gripSpace, referenceSpace) : null;
      detail.pose_available = Boolean(pose);
      detail.pose_emulated = pose?.emulatedPosition ?? null;
      const trigger = source.gamepad?.buttons?.[0]?.value;
      const grip = source.gamepad?.buttons?.[1]?.value;
      if (!source.gripSpace) reason = "컨트롤러 좌표계 없음";
      else if (source.gamepad?.mapping !== "xr-standard") reason = "버튼 매핑 미지원";
      else if (!pose) reason = "컨트롤러 위치 없음";
      else if (![trigger, grip].every((value) => Number.isFinite(value) && value >= 0 && value <= 1)) reason = "버튼 값 없음/범위 오류";
      else {
        reason = pose.emulatedPosition ? "추정 위치 · 사용 중지" : "추적 유효";
        // Diagnostics observe every source, but the original tracking gates stay.
        if (activeSession.visibilityState === "visible" && viewerPose && !viewerPose.emulatedPosition) {
          hands[handedness] = {...poseData(pose), valid: !pose.emulatedPosition,
            emulated: pose.emulatedPosition, grip, trigger};
        }
      }
    }
    detail.reason = reason;
    if (activeSession.visibilityState !== "visible") detail.blocked_by = "session_visibility";
    else if (!viewerPose || viewerPose.emulatedPosition) detail.blocked_by = "viewer_pose";
    else detail.blocked_by = null;
    trackingDiagnostics.hands[handedness] = detail;
    diagnosticText[handedness] = `${reason} · 장치 ${detail.controller_source_count}개 · 매핑 ${detail.gamepad_mapping ?? "없음"}${detail.blocked_by ? ` · 입력 차단: ${detail.blocked_by}` : ""}`;
  }
  buttons = activeSession.visibilityState === "visible" ? nextButtons : emptyButtons();
  // A 60 Hz cap limits Wi-Fi load; latest-only transport skips busy sockets.
  if (timestamp - lastSentTime >= 1000 / 60) {
    sendFrame(timestamp, hands);
    lastSentTime = timestamp;
  }
  if (timestamp - lastUIUpdate > 200) {
    for (const hand of ["left", "right"]) {
      const value = hands[hand];
      ui[hand].textContent = value.valid
        ? `추적 중 · grip ${value.grip.toFixed(2)} · trigger ${value.trigger.toFixed(2)}` : "추적 대기";
    }
    lastUIUpdate = timestamp;
    showDiagnostics();
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, activeSession.renderState.baseLayer.framebuffer);
  gl.clearColor(0.035, 0.065, 0.085, 1);
  gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
}

ui.connect.addEventListener("click", () => {
  reconnectAttempt = 0;
  connect();
});
ui.start.addEventListener("click", start);
ui.stop.addEventListener("click", () => session?.end().catch((error) => showError("세션 종료", error)));
window.addEventListener("pagehide", () => {
  reconnectAllowed = false;
  cancelReconnect();
  invalidate();
  socket?.close(1000, "page hidden");
});
// Opening this page on an ordinary PC browser must not reserve an input socket.
checkXRSupport().then((supported) => { if (supported) connect(); });
