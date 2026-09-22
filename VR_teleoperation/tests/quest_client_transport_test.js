// Hardware-free browser stubs. Run with: node quest_client_transport_test.js
// The function also accepts source directly in an ES2020+ JS host.
function testQuestClient(source) {
  const elements = new Map();
  const document = {getElementById(id) {
    if (!elements.has(id)) elements.set(id, {addEventListener() {}});
    return elements.get(id);
  }};
  const run = new Function("document", "window", "navigator", "WebSocket", source + `
    const assert = (ok, message) => { if (!ok) throw new Error(message); };
    let now = 0;
    const performance = {now: () => now};
    const packets = [];
    socket = {readyState: WebSocket.OPEN, bufferedAmount: 0,
      send(raw) { packets.push(JSON.parse(raw)); }};
    session = {visibilityState: "visible", inputSources: [], renderState: {baseLayer: {framebuffer: {}}}};
    gl = {bindFramebuffer() {}, clearColor() {}, clear() {}};
    const frame = {getViewerPose: () => null};
    function tick(timestamp) { now = timestamp; updateFrame(timestamp, frame, session); }

    tick(0);
    assert(packets.length === 1 && packets[0].seq === 0, "first frame missing");
    socket.bufferedAmount = 512;
    tick(20);
    assert(packets.length === 1 && lastSentTime === 0, "busy socket must not consume send period");
    socket.bufferedAmount = 0;
    tick(30);  // Only 10 ms since the skipped attempt; retry this XR frame.
    assert(packets.length === 2 && packets[1].sample_time_ms === 30, "retry delayed or old frame queued");
    assert(packets[1].seq === 1, "skipped attempt consumed a sequence");
    let diag = packets[1].diagnostics.transport;
    assert(diag.busy_socket_skips === 1 && diag.max_buffered_bytes === 512, "missing backlog diagnostics");
    assert(diag.busy_wait_ms === 10 && diag.send_gap_ms === 30, "incorrect send gap");
    assert(diag.xr_callback_gap_ms === 10, "incorrect XR callback gap");
    tick(40);
    assert(packets.length === 2, "successful sends must retain the 60 Hz cap");
    socket.readyState = 3;
    tick(50);
    socket.readyState = WebSocket.OPEN;
    tick(60);
    assert(packets.length === 3 && packets[2].sample_time_ms === 60, "reconnect retry delayed");
    assert(packets[2].diagnostics.transport.disconnected_skips === 1, "missing disconnect count");
    tick(300);
    assert(packets[3].diagnostics.transport.xr_callback_gap_ms === 240, "XR stall not observed");
    assert(packets[3].diagnostics.transport.busy_wait_ms === 0, "old backlog duration leaked");
    socket.bufferedAmount = 512;
    tick(600);
    socket.bufferedAmount = 0;
    tick(610);
    assert(packets[4].diagnostics.transport.max_xr_callback_gap_ms === 300,
      "busy socket hid the earlier XR stall");
    assert(packets[4].diagnostics.transport.xr_frames_since_send === 2, "XR frame count missing");
    const previousSequence = sequence;
    socket.send = () => { throw new Error("closed during send"); };
    try { sendFrame(320, {left: emptyHand(), right: emptyHand()}); } catch (_) {}
    assert(sequence === previousSequence, "failed send consumed a sequence");
    return "Quest transport: retry, cap, freshness, sequence and timing checks passed";
  `);
  return run(document, {isSecureContext: true, addEventListener() {}}, {}, {OPEN: 1, CONNECTING: 0});
}

if (typeof require === "function" && typeof module !== "undefined" && require.main === module) {
  const fs = require("fs"), path = require("path");
  console.log(testQuestClient(fs.readFileSync(path.join(__dirname, "../quest_client/client.js"), "utf8")));
}
