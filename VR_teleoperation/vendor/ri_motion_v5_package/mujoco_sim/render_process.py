"""
Process-separated MuJoCo rendering helpers.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Union

import glfw
import numpy as np
import zmq

from ri_motion_v5_package.mujoco_sim.parser import MuJoCoParser
from ri_motion_v5_package.mujoco_sim.interactive_marker import InteractiveMarker
from ri_motion_v5_package.utility.utils import SimpleTimer


# Draw-command kwargs that should be restored as NumPy arrays on the render side.
ARRAY_KWARG_NAMES = {
    "T",
    "p",
    "R",
    "p_offset",
    "p_fr",
    "p_to",
    "xyz_len",
    "rgba",
    "contact_rgba",
    "threshold_rgba",
    "axis_rgba",
    "sphere_rgba",
    "x_range",
    "y_range",
    "z_range",
}

# Render-loop profiling keys shown in the optional overlay.
PROFILE_STEP_NAMES = (
    "recv",
    "state",
    "fk",
    "marker",
    "draw",
    "overlay",
    "render",
)

# Already-forwarded MuJoCo data fields needed to render without render-side FK.
RENDER_SNAPSHOT_FIELD_NAMES = (
    "qpos",
    "xpos",
    "xquat",
    "xmat",
    "xipos",
    "ximat",
    "geom_xpos",
    "geom_xmat",
    "site_xpos",
    "site_xmat",
    "cam_xpos",
    "cam_xmat",
    "light_xpos",
    "light_xdir",
    "subtree_com",
)



def _normalize_initial_camera(azimuth=None,distance=None,elevation=None,lookat=None):
    """Validate optional camera overrides while preserving the original view."""
    try:
        camera = {
            "azimuth":float(173 if azimuth is None else azimuth),
            "distance":float(3.56 if distance is None else distance),
            "elevation":float(-22.56 if elevation is None else elevation),
        }
        point = np.asarray([0,0,0.59] if lookat is None else lookat,dtype=float)
    except (TypeError,ValueError,OverflowError) as exc:
        raise ValueError("Initial camera must contain finite numeric values.") from exc
    if not all(np.isfinite(value) for value in camera.values()):
        raise ValueError("Initial camera angles and distance must be finite.")
    if camera["distance"] <= 0:
        raise ValueError("Initial camera distance must be positive.")
    if point.shape != (3,) or not np.all(np.isfinite(point)):
        raise ValueError("Initial camera lookat must contain three finite coordinates.")
    camera["lookat"] = point.tolist()
    return camera


def parse_server_args(argv=None):
    """
    Parse command-line arguments for the render subprocess.

    Parameters:
        argv (list[str] | None): Optional argument list. If None, argparse reads
            from `sys.argv`.

    Returns:
        argparse.Namespace: Parsed render-server settings.
    """
    # Keep the subprocess interface flat so `start()` can launch it with `python -c`.
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",action="store_true")
    parser.add_argument("--xml",required=True)
    parser.add_argument("--port",type=int,required=True)
    parser.add_argument("--feedback-port",type=int,default=-1)
    parser.add_argument("--title",default="MuJoCo render server")
    parser.add_argument("--x-offset",type=float,default=0)
    parser.add_argument("--y-offset",type=float,default=0)
    parser.add_argument("--window-width",type=float,default=500)
    parser.add_argument("--window-height",type=float,default=300)
    parser.add_argument("--render-hz",type=float,default=60.0)
    parser.add_argument("--max-sec",type=float,default=np.inf)
    parser.add_argument("--profile",type=int,default=1)
    parser.add_argument("--transparent",type=int,default=0)
    parser.add_argument("--camera-azimuth",type=float,default=None)
    parser.add_argument("--camera-distance",type=float,default=None)
    parser.add_argument("--camera-elevation",type=float,default=None)
    parser.add_argument("--camera-lookat",type=float,nargs=3,default=None)
    args = parser.parse_args(argv)
    try:
        camera = _normalize_initial_camera(args.camera_azimuth,args.camera_distance,
                                           args.camera_elevation,args.camera_lookat)
    except ValueError as exc:
        parser.error(str(exc))
    for name,value in camera.items():
        setattr(args,"camera_"+name,value)
    return args


def render_server_main(argv=None):
    """
    Run the render-process server loop.

    Parameters:
        argv (list[str] | None): Optional command-line arguments for tests or
            direct calls.

    Returns:
        None.
    """
    # Parse the subprocess-only command line and bind a local ZMQ receiver.
    args = parse_server_args(argv)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.PULL)
    sock.setsockopt(zmq.RCVHWM,1)
    sock.bind("tcp://127.0.0.1:%d"%args.port)
    feedback_sock = None
    if int(args.feedback_port) >= 0:
        # Feedback carries marker poses during runtime and one final RGB image
        # when the viewer exits. Keep only the newest message for notebook use.
        feedback_sock = ctx.socket(zmq.PUSH)
        feedback_sock.setsockopt(zmq.SNDHWM,32)
        feedback_sock.setsockopt(zmq.CONFLATE,1)
        feedback_sock.setsockopt(zmq.LINGER,0)
        feedback_sock.connect("tcp://127.0.0.1:%d"%args.feedback_port)

    # The render process owns its own model, data, and viewer.
    env = MuJoCoParser(rel_xml_path=args.xml,verbose=False)
    env.reset()
    env.init_viewer(
        title    = args.title,
        width    = args.window_width,
        height   = args.window_height,
        x_offset = args.x_offset,
        y_offset = args.y_offset,
        fontscale=200,
    )
    env.viewer.set_transparency(transparent=bool(args.transparent))
    env.viewer.set_cam_info(
        azimuth   = args.camera_azimuth,
        distance  = args.camera_distance,
        elevation = args.camera_elevation,
        lookat    = args.camera_lookat,
    )
    env.viewer.set_geomgroup(group_0=True,group_1=True,group_2=True,group_3=False)
    env.viewer.set_sitegroup(group_0=False,group_1=False,group_2=False)
    env.viewer.set_convexhull(convexhull=False)

    # Runtime state is cached so rendering can continue when no new message arrives.
    render_hz_target     = max(float(args.render_hz),1.0)
    profile_flag         = bool(args.profile)
    last_msg             = None
    hz_update_sec        = 0.25
    hz_ema_alpha         = 0.35
    recv_count           = 0
    render_count         = 0
    recv_hz              = 0.0
    render_hz            = 0.0
    render_factor        = 0.0
    render_period_ms     = 1000.0/render_hz_target
    hz_ready             = False
    profile_ms           = {}
    profile_alpha        = 0.2
    cached_profile_flag  = profile_flag
    cached_overlays      = []
    cached_draws         = []
    cached_markers       = []
    # The render-side env captures final_rgb_img when ESC closes the viewer.
    # Send it once so notebook code can call rp.get_final_rgb_img() in finally.
    final_rgb_sent       = False
    marker_objs          = {}
    
    # One timer throttles rendering; the other updates visible Hz statistics.
    render_timer = SimpleTimer(
        name      = "render_timer",
        Hz        = render_hz_target,
        max_sec   = float(args.max_sec),
        verbose   = False,
        sleep_sec = 0.001,
    )
    hz_timer = SimpleTimer(
        name      = "hz_timer",
        Hz        = 1.0/hz_update_sec,
        max_sec   = np.inf,
        verbose   = False,
        sleep_sec = 0.0,
    )
    render_timer.start()
    hz_timer.start()

    try:
        while env.is_viewer_alive():
            # Enforce the target render rate before doing per-frame work.
            if render_timer.is_finished():
                break
            if not render_timer.do_run():
                continue

            # Drain pending ZMQ messages and keep only the newest state.
            time_loop_start = time.perf_counter()
            time_profile = time_loop_start
            while True:
                try:
                    last_msg = sock.recv_json(flags=zmq.NOBLOCK)
                    recv_count += 1
                except zmq.Again:
                    break
            MuJoCoRenderProcess._update_profile_ms(
                profile_ms,
                "recv",
                time.perf_counter()-time_profile,
                profile_alpha,
            )

            # Periodically update receive/render Hz with EMA smoothing.
            time_profile = time.perf_counter()
            if hz_timer.do_run():
                dt_hz = max(hz_timer.sec_elps_diff,1e-12)
                recv_hz_inst = recv_count/dt_hz
                render_hz_inst = render_count/dt_hz
                if hz_ready:
                    recv_hz = (1.0-hz_ema_alpha)*recv_hz + hz_ema_alpha*recv_hz_inst
                    render_hz = (1.0-hz_ema_alpha)*render_hz + hz_ema_alpha*render_hz_inst
                else:
                    recv_hz = recv_hz_inst
                    render_hz = render_hz_inst
                render_factor = render_hz/render_hz_target
                recv_count = 0
                render_count = 0
                hz_ready = True
                hz_timer.end()
            MuJoCoRenderProcess._update_profile_ms(
                profile_ms,
                "hz",
                time.perf_counter()-time_profile,
                profile_alpha,
            )

            # Apply the newest state message, measuring copy and FK separately.
            time_profile = time.perf_counter()
            state_ms_updated = False
            fk_ms_updated = False
            if last_msg is not None:
                msg_type = last_msg.get("type","state")
                if msg_type == "shutdown":
                    break
                cached_profile_flag = bool(last_msg.get("profile",cached_profile_flag))
                if msg_type == "state":
                    qpos = np.asarray(last_msg["qpos"],dtype=np.float64)
                    if qpos.shape[0] == env.model.nq:
                        time_state = time.perf_counter()
                        env.data.qpos[:] = qpos
                        MuJoCoRenderProcess._update_profile_ms(
                            profile_ms,
                            "state",
                            time.perf_counter()-time_state,
                            profile_alpha,
                        )
                        state_ms_updated = True
                        time_fk = time.perf_counter()
                        env.forward()
                        MuJoCoRenderProcess._update_profile_ms(
                            profile_ms,
                            "fk",
                            time.perf_counter()-time_fk,
                            profile_alpha,
                        )
                        fk_ms_updated = True
                    cached_overlays = last_msg.get("overlays",[])
                    cached_draws = last_msg.get("draws",[])
                    cached_markers = last_msg.get("markers",[])
                if msg_type == "snapshot":
                    time_state = time.perf_counter()
                    MuJoCoRenderProcess._copy_render_snapshot_to_data(env.data,last_msg.get("snapshot",{}))
                    MuJoCoRenderProcess._update_profile_ms(
                        profile_ms,
                        "state",
                        time.perf_counter()-time_state,
                        profile_alpha,
                    )
                    state_ms_updated = True
                    MuJoCoRenderProcess._update_profile_ms(profile_ms,"fk",0.0,profile_alpha)
                    fk_ms_updated = True
                    cached_overlays = last_msg.get("overlays",[])
                    cached_draws = last_msg.get("draws",[])
                    cached_markers = last_msg.get("markers",[])
                last_msg = None
            if not state_ms_updated:
                MuJoCoRenderProcess._update_profile_ms(
                    profile_ms,
                    "state",
                    time.perf_counter()-time_profile,
                    profile_alpha,
                )
            if not fk_ms_updated:
                MuJoCoRenderProcess._update_profile_ms(profile_ms,"fk",0.0,profile_alpha)

            # Poll and draw render-process InteractiveMarkers, then send marker poses back.
            time_profile = time.perf_counter()
            marker_states = MuJoCoRenderProcess._apply_marker_commands(env,cached_markers,marker_objs)
            if feedback_sock is not None and marker_states:
                try:
                    feedback_sock.send_pyobj(
                        {"type":"markers","markers":marker_states},
                        flags=zmq.NOBLOCK,
                    )
                except Exception:
                    pass
            MuJoCoRenderProcess._update_profile_ms(
                profile_ms,
                "marker",
                time.perf_counter()-time_profile,
                profile_alpha,
            )

            # Replay cached draw commands against the render-process env.
            time_profile = time.perf_counter()
            MuJoCoRenderProcess._apply_draw_commands(env,cached_draws)
            MuJoCoRenderProcess._update_profile_ms(
                profile_ms,
                "draw",
                time.perf_counter()-time_profile,
                profile_alpha,
            )

            # Draw user overlays first, then append render-process diagnostics.
            time_profile = time.perf_counter()
            for overlay_cmd in cached_overlays:
                env.viewer_text_overlay(
                    overlay_cmd.get("text1",""),
                    overlay_cmd.get("text2",""),
                    overlay_cmd.get("loc","top right"),
                )
            env.viewer_text_overlay("recv Hz","[%.1f]Hz"%(recv_hz),loc="top left")
            env.viewer_text_overlay("render target","[%.1f]Hz"%(render_hz_target),loc="top left")
            env.viewer_text_overlay("render actual","[%.1f]Hz"%(render_hz),loc="top left")
            if hz_ready:
                render_status = "OK" if render_factor >= 1.0 else "LOW"
                render_factor_text = "[%.2f]x %s"%(render_factor,render_status)
            else:
                render_factor_text = "measuring"
            env.viewer_text_overlay("render factor",render_factor_text,loc="top left")
            env.viewer_text_overlay("draw commands","[%d]"%(len(cached_draws)),loc="top left")
            if cached_profile_flag:
                profile_sum_ms = sum(profile_ms.get(key,0.0) for key in PROFILE_STEP_NAMES)
                loop_ms = profile_ms.get("loop",0.0)
                loop_load = loop_ms/render_period_ms if render_period_ms > 0.0 else np.nan
                bottleneck_name,bottleneck_ms = MuJoCoRenderProcess._get_profile_bottleneck(profile_ms)
                bottleneck_pct = 100.0*bottleneck_ms/max(loop_ms,1e-12)
                other_ms = max(0.0,loop_ms-profile_sum_ms)
                env.viewer_text_overlay(
                    "budget target",
                    "[%.1f]ms"%(render_period_ms),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "budget used",
                    "[%.2f]ms"%(loop_ms),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "budget load",
                    "[%.2f]x"%(loop_load),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "time bottleneck",
                    "%s:[%.2f]ms [%.0f%%]"%(
                        bottleneck_name,bottleneck_ms,bottleneck_pct),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "  used render",
                    "[%.2f]ms [%.0f%%]"%(
                        profile_ms.get("render",0.0),
                        100.0*profile_ms.get("render",0.0)/max(loop_ms,1e-12),
                    ),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "  used draw",
                    "[%.2f]ms [%.0f%%]"%(
                        profile_ms.get("draw",0.0),
                        100.0*profile_ms.get("draw",0.0)/max(loop_ms,1e-12),
                    ),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "  used marker",
                    "[%.2f]ms [%.0f%%]"%(
                        profile_ms.get("marker",0.0),
                        100.0*profile_ms.get("marker",0.0)/max(loop_ms,1e-12),
                    ),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "  used state",
                    "[%.2f]ms [%.0f%%]"%(
                        profile_ms.get("state",0.0),
                        100.0*profile_ms.get("state",0.0)/max(loop_ms,1e-12),
                    ),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "  used fk",
                    "[%.2f]ms [%.0f%%]"%(
                        profile_ms.get("fk",0.0),
                        100.0*profile_ms.get("fk",0.0)/max(loop_ms,1e-12),
                    ),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "  used recv",
                    "[%.2f]ms [%.0f%%]"%(
                        profile_ms.get("recv",0.0),
                        100.0*profile_ms.get("recv",0.0)/max(loop_ms,1e-12),
                    ),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "  used overlay",
                    "[%.2f]ms [%.0f%%]"%(
                        profile_ms.get("overlay",0.0),
                        100.0*profile_ms.get("overlay",0.0)/max(loop_ms,1e-12),
                    ),
                    loc="top left",
                )
                env.viewer_text_overlay(
                    "  used other",
                    "[%.2f]ms [%.0f%%]"%(
                        other_ms,
                        100.0*other_ms/max(loop_ms,1e-12),
                    ),
                    loc="top left",
                )
            MuJoCoRenderProcess._update_profile_ms(
                profile_ms,
                "overlay",
                time.perf_counter()-time_profile,
                profile_alpha,
            )
            # Render one frame and close the profiling budget for this loop.
            time_profile = time.perf_counter()
            env.render()
            # MuJoCoParser.render() also checks ESC, but the render process owns
            # this viewer, so keep the final-image capture explicit here.
            if glfw.KEY_ESCAPE in env.get_key_pressed_list():
                if env.final_rgb_img is None:
                    try:
                        print("Final RGB image captured.")
                        env.final_rgb_img,_ = env.capture_clean_viewer_rgbd_imgs()
                    except Exception:
                        pass
            if (not final_rgb_sent) and (env.final_rgb_img is not None) and (feedback_sock is not None):
                try:
                    feedback_sock.send_pyobj(
                        {"type":"rgb","rgb":env.final_rgb_img},
                        flags=zmq.NOBLOCK,
                    )
                    final_rgb_sent = True
                except Exception:
                    pass
            MuJoCoRenderProcess._update_profile_ms(
                profile_ms,
                "render",
                time.perf_counter()-time_profile,
                profile_alpha,
            )
            render_count += 1
            render_timer.end()
            MuJoCoRenderProcess._update_profile_ms(
                profile_ms,
                "loop",
                time.perf_counter()-time_loop_start,
                profile_alpha,
            )
    finally:
        # Always release viewer and ZMQ resources when the server loop exits.
        # If the viewer closed without the ESC branch above sending an image,
        # attempt one last clean capture before destroying the viewer context.
        if feedback_sock is not None and not final_rgb_sent:
            try:
                rgb_img = env.final_rgb_img
                if rgb_img is None:
                    rgb_img,_ = env.capture_clean_viewer_rgbd_imgs()
                feedback_sock.send_pyobj(
                    {"type":"rgb","rgb":rgb_img},
                    flags=zmq.NOBLOCK,
                )
            except Exception:
                pass
        try:
            env.close_viewer()
        except Exception:
            pass
        if feedback_sock is not None:
            feedback_sock.close(linger=0)
        sock.close(linger=0)
        ctx.term()


class MuJoCoRenderProcess:
    """
    Manage a separate MuJoCo viewer process from a notebook or script.

    Lifecycle:
        1. Construct the object with the XML path and render settings.
        2. Call `start()` to spawn a normal Python subprocess.
        3. The subprocess creates its own `MuJoCoParser`, `MjModel`, `MjData`,
           viewer, and ZMQ PULL socket.
        4. The main process sends state messages with `send(...)` or
           `render(...)`.
        5. The render process copies received state into its own `MjData`,
           replays cached overlays and `env.plot_*` draw commands, and renders
           at `render_hz`.
        6. Call `close()` in a `finally` block to send shutdown, terminate the
           process if needed, and close ZMQ resources.

    Important behavior:
        - The main process and render process do not share Python objects.
          They communicate only through JSON-safe messages.
        - `render(env,...,mode="qpos",profile=True)` sends `env.data.qpos`,
          overlays, and draw commands. The render process then calls
          `env.forward()` and reports that time as `used fk`.
        - `render(env,...,mode="snapshot")` sends already-forwarded render
          fields and lets the render process skip its own `env.forward()`.
        - Draw commands use `rp.draw("plot_...",...)`; the render process only
          calls callable `env.plot_*` methods.
        - Interactive marker commands are normally registered with
          `rp.add_interactive_marker(...)`. The render process owns mouse input, and the
          notebook updates local marker state with `rp.update_markers()`.
        - If the main loop is slower than `render_hz`, the render process keeps
          rendering the most recently received state.
        - `max_sec` is a safety limit for the render subprocess. It prevents a
          forgotten subprocess from running forever, but normal notebooks should
          still call `close()`.
        - `startup_sec` is only a short guard that checks whether the subprocess
          exits immediately after launch. It is not a readiness handshake.

    Typical notebook usage:
        rp = MuJoCoRenderProcess(
            0.25, 0.00, 0.75, 1.00,
            xml_path  = xml_path,
            render_hz = 30.0,
            max_sec   = 300.0,
        )
        rp.add_interactive_marker("target", T_target, axis_len=0.22, ring_r=0.22)
        rp.start()
        try:
            while rp.is_running():
                rp.update_markers()
                T_target = rp.get_marker_T("target")
                rp.render(
                    env,
                    overlays = overlay_cmds,
                    draws    = draw_cmds,
                    mode     = "snapshot",
                    profile  = True,
                )
                sleep(0.001)
        finally:
            rp.close()
    """

    # JSON/protocol helpers used by both the notebook sender and render server.
    @staticmethod
    def _to_jsonable(value):
        """
        Convert common NumPy containers into JSON-safe Python values.

        Parameters:
            value (Any): Value that may contain NumPy arrays, NumPy scalar
                values, tuples, lists, or dictionaries.

        Returns:
            Any: JSON-safe value for `socket.send_json(...)`.
        """
        if isinstance(value,np.ndarray):
            return value.tolist()
        if isinstance(value,np.generic):
            return value.item()
        if isinstance(value,tuple):
            return [MuJoCoRenderProcess._to_jsonable(v) for v in value]
        if isinstance(value,list):
            return [MuJoCoRenderProcess._to_jsonable(v) for v in value]
        if isinstance(value,dict):
            return {str(k):MuJoCoRenderProcess._to_jsonable(v) for k,v in value.items()}
        return value

    @staticmethod
    def _overlay_cmd(text1,text2="",loc="top right"):
        """
        Create a text-overlay command for the render process.

        Parameters:
            text1 (str): Overlay label text.
            text2 (str): Overlay value text.
            loc (str): Viewer overlay location, such as `'top right'`.

        Returns:
            dict: JSON-safe overlay command.
        """
        return {
            "loc"   : str(loc),
            "text1" : str(text1),
            "text2" : str(text2),
        }

    @staticmethod
    def _draw_cmd(fn,**kwargs):
        """
        Create a draw command for a render-process `env.plot_*` call.

        Parameters:
            fn (str): Name of the render-process plotting function. It must
                start with `'plot_'`.
            **kwargs: Keyword arguments passed after JSON serialization.

        Returns:
            dict: JSON-safe draw command.
        """
        fn = str(fn)
        if not fn.startswith("plot_"):
            raise ValueError("draw function must start with 'plot_': %s"%(fn))
        return {
            "fn"     : fn,
            "kwargs" : MuJoCoRenderProcess._to_jsonable(kwargs),
        }

    @staticmethod
    def _marker_cmd(name,T,alpha=0.5,**kwargs):
        """
        Create an interactive-marker command for the render process.

        Parameters:
            name (str): Stable marker name used to preserve render-side marker
                state and route feedback back to the notebook.
            T (array-like): 4x4 marker transform.
            alpha (float): Marker draw alpha.
            **kwargs: Constructor keyword arguments for `InteractiveMarker`.

        Returns:
            dict: JSON-safe marker command.
        """
        return {
            "name"   : str(name),
            "T"      : MuJoCoRenderProcess._to_jsonable(np.asarray(T,dtype=np.float64)),
            "alpha"  : float(alpha),
            "kwargs" : MuJoCoRenderProcess._to_jsonable(kwargs),
        }

    @staticmethod
    def _state_msg(qpos,overlays=None,draws=None,markers=None,profile=True):
        """
        Create a qpos-state message for the render process.

        Parameters:
            qpos (array-like): MuJoCo generalized position vector.
            overlays (list[dict] | None): Overlay commands.
            draws (list[dict] | None): Draw commands.
            markers (list[dict] | None): Interactive marker commands.
            profile (bool): If True, show render-loop profiling overlay.

        Returns:
            dict: JSON-safe state message.
        """
        return {
            "type"     : "state",
            "qpos"     : MuJoCoRenderProcess._to_jsonable(qpos),
            "overlays" : [] if overlays is None else MuJoCoRenderProcess._to_jsonable(overlays),
            "draws"    : [] if draws is None else MuJoCoRenderProcess._to_jsonable(draws),
            "markers"  : [] if markers is None else MuJoCoRenderProcess._to_jsonable(markers),
            "profile"  : bool(profile),
        }

    @staticmethod
    def _render_snapshot_msg(env,overlays=None,draws=None,markers=None,profile=True):
        """
        Create a render-state snapshot message from an already-forwarded env.

        Parameters:
            env (MuJoCoParser): Notebook-side environment after `env.forward()`.
            overlays (list[dict] | None): Overlay commands.
            draws (list[dict] | None): Draw commands.
            markers (list[dict] | None): Interactive marker commands.
            profile (bool): If True, show render-loop profiling overlay.

        Returns:
            dict: JSON-safe snapshot message.
        """
        snapshot = {}
        data = env.data
        for field_name in RENDER_SNAPSHOT_FIELD_NAMES:
            if hasattr(data,field_name):
                snapshot[field_name] = np.asarray(getattr(data,field_name)).copy()
        return {
            "type"     : "snapshot",
            "snapshot" : MuJoCoRenderProcess._to_jsonable(snapshot),
            "overlays" : [] if overlays is None else MuJoCoRenderProcess._to_jsonable(overlays),
            "draws"    : [] if draws is None else MuJoCoRenderProcess._to_jsonable(draws),
            "markers"  : [] if markers is None else MuJoCoRenderProcess._to_jsonable(markers),
            "profile"  : bool(profile),
        }

    @staticmethod
    def _shutdown_msg():
        """
        Create a shutdown message for the render process.

        Parameters:
            None.

        Returns:
            dict: JSON-safe shutdown message.
        """
        return {"type":"shutdown"}

    @staticmethod
    def _update_profile_ms(profile,key,elapsed_sec,alpha=0.2):
        """
        Update an exponential moving average profiling entry in milliseconds.

        Parameters:
            profile (dict): Mutable profiling dictionary.
            key (str): Entry name to update.
            elapsed_sec (float): Measured elapsed time in seconds.
            alpha (float): EMA update ratio.

        Returns:
            None.
        """
        value_ms = 1000.0*float(elapsed_sec)
        if key not in profile:
            profile[key] = value_ms
        else:
            profile[key] = (1.0-alpha)*profile[key] + alpha*value_ms

    @staticmethod
    def _get_profile_bottleneck(profile,step_names=PROFILE_STEP_NAMES):
        """
        Return the largest measured profiling step.

        Parameters:
            profile (dict): Profiling dictionary with millisecond values.
            step_names (tuple[str, ...]): Profiling keys to compare.

        Returns:
            tuple[str, float]: Name and millisecond value of the largest step.
        """
        if len(step_names) == 0:
            return "none",0.0
        name = max(step_names,key=lambda key:profile.get(key,0.0))
        return name,float(profile.get(name,0.0))

    @staticmethod
    def _normalize_draw_kwargs(kwargs):
        """
        Convert selected draw-command keyword values back to NumPy arrays.

        Parameters:
            kwargs (dict): JSON-decoded keyword arguments from a draw command.

        Returns:
            dict: Keyword arguments suitable for `env.plot_*` functions.
        """
        out = {}
        for key,val in kwargs.items():
            if key in ARRAY_KWARG_NAMES and val is not None:
                out[key] = np.asarray(val,dtype=np.float64)
            else:
                out[key] = val
        return out

    @staticmethod
    def _copy_render_snapshot_to_data(data,snapshot):
        """
        Copy render snapshot arrays into a render-process `MjData`.

        Parameters:
            data (mujoco.MjData): Render-process MuJoCo data object.
            snapshot (dict): JSON-decoded snapshot created by
                `_render_snapshot_msg(...)`.

        Returns:
            None.
        """
        for field_name in RENDER_SNAPSHOT_FIELD_NAMES:
            if field_name not in snapshot or not hasattr(data,field_name):
                continue
            dst = getattr(data,field_name)
            src = np.asarray(snapshot[field_name],dtype=dst.dtype)
            if src.shape == dst.shape:
                dst[...] = src

    @staticmethod
    def _is_allowed_draw_fn(env,fn_name):
        """
        Check whether a draw command may call a render-process plotting function.

        Parameters:
            env (MuJoCoParser): Render-process environment.
            fn_name (str): Requested function name.

        Returns:
            bool: True when `fn_name` is a callable `env.plot_*` method.
        """
        return (
            isinstance(fn_name,str)
            and fn_name.startswith("plot_")
            and hasattr(env,fn_name)
            and callable(getattr(env,fn_name))
        )

    @staticmethod
    def _apply_draw_commands(env,draws):
        """
        Apply draw commands to the render-process environment.

        Parameters:
            env (MuJoCoParser): Render-process environment with a live viewer.
            draws (list[dict]): Draw commands created by `MuJoCoRenderProcess.draw(...)`.

        Returns:
            None.
        """
        for draw_cmd in draws:
            fn_name = draw_cmd.get("fn","")
            if not MuJoCoRenderProcess._is_allowed_draw_fn(env,fn_name):
                continue
            kwargs = MuJoCoRenderProcess._normalize_draw_kwargs(draw_cmd.get("kwargs",{}))
            getattr(env,fn_name)(**kwargs)

    @staticmethod
    def _apply_marker_commands(env,markers,marker_objs):
        """
        Apply interactive-marker commands and return their latest render-side poses.

        Parameters:
            env (MuJoCoParser): Render-process environment with a live viewer.
            markers (list[dict]): Marker commands created by
                `MuJoCoRenderProcess.marker(...)`.
            marker_objs (dict[str, InteractiveMarker]): Persistent render-side
                marker objects keyed by stable marker name.

        Returns:
            list[dict]: Marker feedback states for the notebook process.
        """
        marker_names = {str(marker_cmd.get("name","")) for marker_cmd in markers}
        marker_names.discard("")
        for stale_name in list(marker_objs.keys()):
            if stale_name not in marker_names:
                marker_objs.pop(stale_name,None)

        marker_states = []
        for marker_cmd in markers:
            name = str(marker_cmd.get("name",""))
            if not name:
                continue
            T = np.asarray(marker_cmd.get("T",np.eye(4)),dtype=np.float64)
            if T.shape != (4,4):
                continue
            alpha = float(marker_cmd.get("alpha",0.5))
            kwargs = marker_cmd.get("kwargs",{})
            if name not in marker_objs:
                marker_objs[name] = InteractiveMarker(env=env,T=T,**kwargs)
            marker = marker_objs[name]
            if marker.drag_mode is None:
                marker.set_T(T,clear_state=False)
            T_marker = marker.get_T(update=True,draw=False)
            marker.draw(alpha=alpha)
            marker_states.append({
                "name"   : name,
                "T"      : T_marker,
                "status" : marker.get_status(),
            })
        return marker_states

    @staticmethod
    def _pick_free_local_port():
        """
        Pick an available localhost TCP port.

        Parameters:
            None.

        Returns:
            int: Port number that was free at selection time.
        """
        port_sock = socket.socket(socket.AF_INET,socket.SOCK_STREAM)
        port_sock.bind(("127.0.0.1",0))
        port = int(port_sock.getsockname()[1])
        port_sock.close()
        return port

    # Notebook-facing lifecycle and scene-sending API.
    def __init__(
            self,
            x_offset: Union[int, float] = 0,
            y_offset: Union[int, float] = 0,
            window_width: Union[int, float] = 500,
            window_height: Union[int, float] = 300,
            xml_path = None,
            title = "MuJoCo render process",
            render_hz = 30.0,
            max_sec = np.inf,
            startup_sec = 1.0,
            log_path = None,
            profile = True,
            transparent = False,
            verbose = True,
            camera_azimuth = None,
            camera_distance = None,
            camera_elevation = None,
            camera_lookat = None,
        ):
        """
        Store render-process settings without starting the subprocess.

        Parameters:
            x_offset (Union[int, float]): Viewer x offset. Defaults to `0`.
            y_offset (Union[int, float]): Viewer y offset. Defaults to `0`.
            window_width (Union[int, float]): Viewer window width. Values at or
                below `1.0` follow `MuJoCoParser.init_viewer(...)` ratio
                convention; larger values are pixels.
            window_height (Union[int, float]): Viewer window height. Values at
                or below `1.0` follow `MuJoCoParser.init_viewer(...)` ratio
                convention; larger values are pixels.
            xml_path (str | Path): XML path loaded by the render process.
            title (str): Viewer window title.
            render_hz (float): Target render-process frame rate.
            max_sec (float): Safety timeout for the render subprocess.
            startup_sec (float): Short wait after launch used to detect immediate
                subprocess failure.
            log_path (str | Path | None): Optional log file path. If None, a
                temporary log file is created.
            profile (bool): Default render-loop profiling overlay setting used
                when `render(...,profile=None)`.
            transparent (bool): If True, render the subprocess viewer with
                transparent MuJoCo geoms enabled.
            verbose (bool): If True, print process id, port, and log path.
            camera_azimuth (float | None): Optional initial azimuth in degrees.
            camera_distance (float | None): Optional positive initial distance.
            camera_elevation (float | None): Optional initial elevation in degrees.
            camera_lookat (array-like | None): Optional three-coordinate lookat.
                Omitted camera values retain the original render-process view.

        Returns:
            None.
        """
        # Store launch settings; the subprocess is created later by `start()`.
        if xml_path is None:
            raise ValueError("xml_path should be provided.")
        self.x_offset = float(x_offset)
        self.y_offset = float(y_offset)
        self.window_width = float(window_width)
        self.window_height = float(window_height)
        self.xml_path = str(xml_path)
        self.title = str(title)
        self.render_hz = float(render_hz)
        self.max_sec = float(max_sec)
        self.startup_sec = float(startup_sec)
        self.log_path = log_path
        self.profile = bool(profile)
        self.transparent = bool(transparent)
        self.verbose = bool(verbose)
        self.initial_camera = _normalize_initial_camera(
            camera_azimuth,camera_distance,camera_elevation,camera_lookat,
        )

        self.port = None
        self.feedback_port = None
        self.proc = None
        self.ctx = None
        self.socket = None
        self.feedback_socket = None
        self._log_file = None
        self.time_start = None
        self.marker_specs = {}
        self.marker_states = {}
        # Last RGB frame sent by the render subprocess when its viewer exits.
        self.final_rgb_img = None

    def start(self):
        """
        Start the render subprocess and connect the ZMQ PUSH socket.

        Parameters:
            None.

        Returns:
            MuJoCoRenderProcess: This object, so notebook code may write
                `rp = MuJoCoRenderProcess(...).start()` if desired.
        """
        # Restart cleanly if this object already owns a previous subprocess.
        self.close()
        self.port = self._pick_free_local_port()
        self.feedback_port = self._pick_free_local_port()

        # Capture subprocess stdout/stderr in a log file for startup debugging.
        if self.log_path is None:
            log_file = tempfile.NamedTemporaryFile(
                mode   = "w",
                prefix = "ri_motion_render_",
                suffix = ".log",
                delete = False,
            )
            self.log_path = log_file.name
        else:
            log_file = open(self.log_path,"w")
        self._log_file = log_file

        # Launch the server through the current Python executable and this module.
        cmd = [
            sys.executable,
            "-c",
            "from ri_motion_v5_package.mujoco_sim.render_process import main; main()",
            "--server",
            "--xml", self.xml_path,
            "--port", str(self.port),
            "--feedback-port", str(self.feedback_port),
            "--title", self.title,
            "--x-offset", "%.6f"%(self.x_offset),
            "--y-offset", "%.6f"%(self.y_offset),
            "--window-width", "%.6f"%(self.window_width),
            "--window-height", "%.6f"%(self.window_height),
            "--render-hz", "%.6f"%(self.render_hz),
            "--max-sec", "%.6f"%(self.max_sec),
            "--profile", "%d"%(1 if self.profile else 0),
            "--transparent", "%d"%(1 if self.transparent else 0),
            "--camera-azimuth="+repr(self.initial_camera["azimuth"]),
            "--camera-distance="+repr(self.initial_camera["distance"]),
            "--camera-elevation="+repr(self.initial_camera["elevation"]),
            "--camera-lookat",
            # Positional decimal strings avoid argparse treating a negative
            # scientific-notation coordinate as a new command-line option.
            *[np.format_float_positional(value,unique=True,trim="-")
              for value in self.initial_camera["lookat"]],
        ]

        env = os.environ.copy()
        workspace_root = str(Path(__file__).resolve().parents[2])
        old_pythonpath = env.get("PYTHONPATH","")
        env["PYTHONPATH"] = workspace_root if not old_pythonpath else workspace_root + os.pathsep + old_pythonpath

        # Bind the feedback receiver before the subprocess attempts to connect.
        self.ctx = zmq.Context()
        self.feedback_socket = self.ctx.socket(zmq.PULL)
        self.feedback_socket.setsockopt(zmq.RCVHWM,32)
        # Marker feedback is high-rate, while final RGB is only needed at the
        # end. Keeping the newest feedback prevents stale marker messages from
        # hiding the final image.
        self.feedback_socket.setsockopt(zmq.CONFLATE,1)
        self.feedback_socket.setsockopt(zmq.LINGER,0)
        self.feedback_socket.bind("tcp://127.0.0.1:%d"%self.feedback_port)

        # Start the render process before connecting the notebook-side sender.
        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd    = str(Path.cwd()),
                env    = env,
                stdout = self._log_file,
                stderr = subprocess.STDOUT,
                text   = True,
            )
        except Exception:
            self._close_feedback_socket()
            self._close_context()
            self._close_log_file()
            self.proc = None
            raise

        time.sleep(self.startup_sec)
        if self.proc.poll() is not None:
            self._raise_startup_error("render process exited early")

        # Connect after startup so `render()` can use nonblocking ZMQ sends.
        self.socket = self.ctx.socket(zmq.PUSH)
        self.socket.setsockopt(zmq.SNDHWM,1)
        self.socket.setsockopt(zmq.LINGER,0)
        self.socket.connect("tcp://127.0.0.1:%d"%self.port)
        self.time_start = time.perf_counter()

        if self.verbose:
            print(
                "render process pid:[%d] port:[%d] target_hz:[%.1f]"
                % (self.proc.pid,self.port,self.render_hz)
            )
            print("render process log:[%s]"%(self.log_path))
        return self

    def is_running(self):
        """
        Check whether the render subprocess is still alive.

        Parameters:
            None.

        Returns:
            bool: True when the subprocess exists and has not exited.
        """
        return self.proc is not None and self.proc.poll() is None

    def send(self,msg):
        """
        Send one JSON-safe message to the render process.

        Parameters:
            msg (dict): Message created by `self._state_msg(...)`,
                `self._shutdown_msg(...)`, or another compatible protocol
                helper.

        Returns:
            bool: True when the message was queued, False when the ZMQ high-water
                mark rejected the nonblocking send.
        """
        # Use nonblocking send so a slow renderer cannot stall the notebook loop.
        if self.socket is None:
            raise RuntimeError("Render process is not started.")
        try:
            self.socket.send_json(self._to_jsonable(msg),flags=zmq.NOBLOCK)
            return True
        except zmq.Again:
            return False

    def overlay(self,text1,text2="",loc="top right"):
        """
        Create a text-overlay command for this render process.

        Parameters:
            text1 (str): Overlay label text.
            text2 (str): Overlay value text.
            loc (str): Viewer overlay location, such as `'top right'`.

        Returns:
            dict: JSON-safe overlay command for `render(...)`.
        """
        return self._overlay_cmd(text1=text1,text2=text2,loc=loc)

    def draw(self,fn,**kwargs):
        """
        Create a draw command for this render process.

        Parameters:
            fn (str): Name of the render-process plotting function. It must
                start with `'plot_'`.
            **kwargs: Keyword arguments passed to the render-process plotting
                function after JSON serialization.

        Returns:
            dict: JSON-safe draw command for `render(...)`.
        """
        return self._draw_cmd(fn,**kwargs)

    def marker(self,name,T,alpha=0.5,**kwargs):
        """
        Create an interactive-marker command for this render process.

        Parameters:
            name (str): Stable marker name used to keep render-side state and
                route feedback to the notebook.
            T (array-like): 4x4 marker transform sent from the notebook.
            alpha (float): Marker draw alpha.
            **kwargs: Constructor keyword arguments for `InteractiveMarker`.

        Returns:
            dict: JSON-safe marker command for `render(...)`.
        """
        return self._marker_cmd(name=name,T=T,alpha=alpha,**kwargs)

    def add_interactive_marker(self,name,T,alpha=0.5,**kwargs):
        """
        Register an interactive marker owned by this render process.

        The notebook process stores the marker specification here, and
        `render(...)` forwards that specification to the render process.
        The render process then creates or updates an `InteractiveMarker` with
        its own viewer, mouse input, and marker state. The notebook can later
        call `update_markers()`, `get_marker_T(...)`, and
        `get_marker_status(...)` to read the latest render-side feedback.

        Parameters:
            name (str): Stable marker name. Reusing the same name updates the
                existing registered marker specification.
            T (array-like): Initial 4x4 marker transform. This is also the
                pose sent to the render process until mouse feedback updates
                the marker state.
            alpha (float): Marker draw alpha passed to the render-side marker
                command. Defaults to `0.5`.
            **kwargs: Constructor keyword arguments for `InteractiveMarker` in
                the render process. Common options are:
                - `axis_len` (float): Length of translation axes. Defaults to
                  `0.20`.
                - `ring_r` (float): Rotation-ring radius. Defaults to `0.20`.
                - `ring_segs` (int): Number of line segments for each rotation
                  ring. Defaults to `50`.
                - `translation_frame` (str): Translation handle frame,
                  `"global"` or `"local"`. Defaults to `"global"`.
                - `control_mode` (str): Enabled controls: `"p"` for
                  translation, `"R"` for rotation, or `"pR"` for both. Defaults
                  to `"pR"`.
                - `show_frame_toggle` (bool): Show the top handle that toggles
                  local/global translation axes. Defaults to `True`.
                - `show_rotation_reset` (bool): Show the top handle that resets
                  marker rotation to its initial orientation. Defaults to
                  `True`.
                - `plane` (tuple | None): Translation-plane geometry
                  `(offset, size, thickness)`. If `None`, a default is derived
                  from `axis_len`.
                - `hover_scale` (float): Visual scale multiplier for hovered
                  handles. Defaults to `1.2`.
                - `hover_px` (tuple): Pixel hit thresholds for axis, plane, and
                  ring handles. Defaults to `(28.0,22.0,18.0)`.
                - `mouse_view_offset` (tuple): Small normalized view offset used
                  for mouse ray calculations. Defaults to `(0.03,0.0)`.
                - `top_handle_px` (float): Pixel threshold for top utility
                  handles. Defaults to `18.0`.
                - `top_handle_r` (float): Radius of top utility handle spheres.
                  Defaults to `0.014`.
                - `top_handle_offset` (float): Vertical offset for top utility
                  handles. Defaults to `0.07`.
                - `view_q_min` (float): Minimum view-direction quality used by
                  hover selection. Defaults to `0.10`.
                - `ring_thick`, `arrow_r`, `click_sphere_r`,
                  `mouse_sphere_r`, `label_offset` (float): Marker drawing
                  sizes for rings, arrows, click markers, mouse debug markers,
                  and labels.
                - `hover_score_bias`, `top_score_bias`, `rotate_edge_min`
                  (float): Hover scoring and rotation-edge tuning constants.
                - `axis_colors`, `plane_colors`, `ring_colors`, `hi_color`,
                  `top_handle_color`, `click_color`, `mouse_color`: RGBA colors
                  used for marker handles and debug points.

        Returns:
            MuJoCoRenderProcess: This object, so calls may be chained.
        """
        name = str(name)
        T = np.asarray(T,dtype=np.float64).copy()
        if T.shape != (4,4):
            raise ValueError("marker T must have shape (4,4).")
        self.marker_specs[name] = {
            "name"   : name,
            "T"      : T,
            "alpha"  : float(alpha),
            "kwargs" : dict(kwargs),
            "status" : "waiting",
        }
        self.marker_states[name] = {
            "name"   : name,
            "T"      : T.copy(),
            "status" : "waiting",
        }
        return self

    def remove_marker(self,name):
        """
        Remove a registered interactive marker.

        Parameters:
            name (str): Marker name to remove.

        Returns:
            MuJoCoRenderProcess: This object.
        """
        name = str(name)
        self.marker_specs.pop(name,None)
        self.marker_states.pop(name,None)
        return self

    def clear_markers(self):
        """
        Remove all registered interactive markers.

        Parameters:
            None.

        Returns:
            MuJoCoRenderProcess: This object.
        """
        self.marker_specs.clear()
        self.marker_states.clear()
        return self

    def set_marker_T(self,name,T):
        """
        Set a registered marker transform from the notebook process.

        Parameters:
            name (str): Marker name.
            T (array-like): 4x4 transform.

        Returns:
            MuJoCoRenderProcess: This object.
        """
        name = str(name)
        if name not in self.marker_specs:
            raise KeyError("unknown marker: %s"%(name))
        T = np.asarray(T,dtype=np.float64).copy()
        if T.shape != (4,4):
            raise ValueError("marker T must have shape (4,4).")
        self.marker_specs[name]["T"] = T
        self.marker_states.setdefault(name,{"name":name})["T"] = T.copy()
        return self

    def get_marker_T(self,name):
        """
        Return the latest known marker transform.

        Parameters:
            name (str): Marker name.

        Returns:
            np.ndarray: 4x4 marker transform.
        """
        name = str(name)
        if name in self.marker_states and "T" in self.marker_states[name]:
            return np.asarray(self.marker_states[name]["T"],dtype=np.float64).copy()
        if name in self.marker_specs:
            return np.asarray(self.marker_specs[name]["T"],dtype=np.float64).copy()
        raise KeyError("unknown marker: %s"%(name))

    def get_marker_status(self,name,default="waiting"):
        """
        Return the latest known marker status text.

        Parameters:
            name (str): Marker name.
            default (str): Value returned before feedback is received.

        Returns:
            str: Marker status text.
        """
        name = str(name)
        return str(self.marker_states.get(name,{}).get("status",default))

    def get_marker_states(self):
        """
        Return a copy of latest marker feedback states.

        Parameters:
            None.

        Returns:
            dict[str, dict]: Marker states keyed by marker name.
        """
        out = {}
        for name,state in self.marker_states.items():
            out[name] = dict(state)
            if "T" in out[name]:
                out[name]["T"] = np.asarray(out[name]["T"],dtype=np.float64).copy()
        return out

    def get_marker_cmds(self):
        """
        Create render-process marker commands for all registered markers.

        Parameters:
            None.

        Returns:
            list[dict]: JSON-safe marker commands.
        """
        marker_cmds = []
        for name,spec in self.marker_specs.items():
            marker_cmds.append(
                self._marker_cmd(
                    name  = name,
                    T     = spec["T"],
                    alpha = spec["alpha"],
                    **spec["kwargs"],
                )
            )
        return marker_cmds

    def update_markers(self):
        """
        Update registered marker state from render-process feedback.

        Parameters:
            None.

        Returns:
            dict[str, dict]: Latest marker states keyed by marker name. Each
                state contains at least `name`, `T`, and `status`.
        """
        self._drain_feedback()
        return self.get_marker_states()

    def recv_markers(self):
        """
        Compatibility alias for `update_markers()`.

        Parameters:
            None.

        Returns:
            dict[str, dict]: Latest marker states keyed by marker name.
        """
        return self.update_markers()

    def get_final_rgb_img(self):
        """
        Return the final RGB image captured by the render subprocess.

        The render subprocess captures this image when its viewer exits,
        normally through the ESC-key shutdown path, and sends it through the
        feedback socket before closing. This getter only drains already-sent
        feedback; it does not render or request a new frame.

        Parameters:
            None.

        Returns:
            np.ndarray | None: A copy of the final RGB image, or None if the
                render process has not sent one yet.
        """
        self._drain_feedback()
        if self.final_rgb_img is None:
            return None
        return self.final_rgb_img.copy()

    def _drain_feedback(self):
        """
        Drain pending render-process feedback messages.

        Feedback currently carries marker states while the viewer is running
        and one final RGB image as the viewer exits.

        Parameters:
            None.

        Returns:
            None.
        """
        if self.feedback_socket is None:
            return
        while True:
            try:
                msg = self.feedback_socket.recv_pyobj(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception:
                break
            self._handle_feedback_msg(msg)

    def _handle_feedback_msg(self,msg):
        """
        Apply one feedback message from the render process.

        Parameters:
            msg (dict): Feedback message. Supported types are `"markers"` and
                `"rgb"`.

        Returns:
            None.
        """
        if not isinstance(msg,dict):
            return
        msg_type = msg.get("type","")
        if msg_type == "markers":
            for marker_state in msg.get("markers",[]):
                name = str(marker_state.get("name",""))
                if not name:
                    continue
                if "T" in marker_state:
                    marker_state["T"] = np.asarray(marker_state["T"],dtype=np.float64)
                self.marker_states[name] = marker_state
                if name in self.marker_specs and "T" in marker_state:
                    self.marker_specs[name]["T"] = np.asarray(marker_state["T"],dtype=np.float64).copy()
                if name in self.marker_specs and "status" in marker_state:
                    self.marker_specs[name]["status"] = str(marker_state["status"])
        elif msg_type == "rgb" and "rgb" in msg:
            self.final_rgb_img = np.asarray(msg["rgb"],dtype=np.uint8).copy()

    def render(
            self,
            env,
            overlays = None,
            draws    = None,
            markers  = None,
            mode     = "snapshot",
            profile  = None,
        ):
        """
        Send one render scene update to the render process.

        Parameters:
            env (MuJoCoParser): Notebook-side environment. The full Python
                object is never sent to the render process.
            overlays (list[dict] | None): Overlay commands created by
                `rp.overlay(...)`.
            draws (list[dict] | None): Draw commands created by `rp.draw(...)`.
            markers (list[dict] | None): Optional explicit interactive marker
                commands. If None, registered markers from `add_interactive_marker(...)`
                are sent automatically.
            mode (str): State transfer mode.
                - `"qpos"` sends only `env.data.qpos`. The render process
                  copies `qpos` and calls `env.forward()`.
                - `"snapshot"` sends render-relevant `MjData` fields from an
                  already-forwarded main-process env. The render process copies
                  those fields and skips `env.forward()`. 
            profile (bool | None): If True, show render-loop profiling overlay
                for this scene. If False, hide it. If None, use `self.profile`.

        Returns:
            bool: True when the message was queued, False when the nonblocking
                send was dropped by ZMQ backpressure.
        """
        # Build either a qpos-only message or an already-forwarded snapshot message.
        profile_flag = self.profile if profile is None else bool(profile)
        if mode not in ("qpos","snapshot"):
            raise ValueError("mode must be 'qpos' or 'snapshot'.")
        marker_cmds = self.get_marker_cmds() if markers is None else markers
        try:
            if mode == "qpos":
                return self.send(
                    self._state_msg(
                        qpos     = env.data.qpos,
                        overlays = overlays,
                        draws    = draws,
                        markers  = marker_cmds,
                        profile  = profile_flag,
                    )
                )
            return self.send(
                self._render_snapshot_msg(
                    env      = env,
                    overlays = overlays,
                    draws    = draws,
                    markers  = marker_cmds,
                    profile  = profile_flag,
                )
            )
        except Exception as err:
            if self.verbose:
                print("render send failed:",err)
            return False

    def get_elapsed_sec(self):
        """
        Return elapsed time since the render process successfully started.

        Parameters:
            None.

        Returns:
            float: Elapsed seconds. Returns 0.0 before `start()`.
        """
        if self.time_start is None:
            return 0.0
        return time.perf_counter() - self.time_start

    def close(self):
        """
        Close the render subprocess and local ZMQ resources.

        Parameters:
            None.

        Returns:
            None.
        """
        # Ask the server to exit, then terminate if it does not shut down quickly.
        if self.socket is not None:
            try:
                self.socket.send_json(self._shutdown_msg(),flags=zmq.NOBLOCK)
            except Exception:
                pass
        time.sleep(0.2)
        self._terminate_process()
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self._close_feedback_socket()
        self._close_context()
        self._close_log_file()
        self.proc = None
        self.time_start = None

    # Internal cleanup and error-reporting helpers.
    def _terminate_process(self):
        """
        Terminate the subprocess, escalating to kill if it does not exit.

        Parameters:
            None.

        Returns:
            None.
        """
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def _read_log(self):
        """
        Read the render-process log file for error reporting.

        Parameters:
            None.

        Returns:
            str: Log text, or an empty string if the log cannot be read.
        """
        try:
            if self.log_path is not None:
                return Path(self.log_path).read_text()
        except Exception:
            pass
        return ""

    def _raise_startup_error(self,msg):
        """
        Raise a startup error that includes the render-process log contents.

        Parameters:
            msg (str): Short error message.

        Returns:
            None.
        """
        self._close_feedback_socket()
        self._close_context()
        self._close_log_file()
        self.proc = None
        raise RuntimeError(
            str(msg) + "\n"
            + "log path: " + str(self.log_path) + "\n"
            + "log:\n" + str(self._read_log())
        )

    def _close_feedback_socket(self):
        """
        Close the marker-feedback socket if it is open.

        Parameters:
            None.

        Returns:
            None.
        """
        if self.feedback_socket is not None:
            self.feedback_socket.close(linger=0)
            self.feedback_socket = None

    def _close_context(self):
        """
        Terminate the ZMQ context if it exists.

        Parameters:
            None.

        Returns:
            None.
        """
        if self.ctx is not None:
            self.ctx.term()
            self.ctx = None

    def _close_log_file(self):
        """
        Close the local log file handle if it is open.

        Parameters:
            None.

        Returns:
            None.
        """
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def __enter__(self):
        """
        Start the render process when entering a context manager.

        Parameters:
            None.

        Returns:
            MuJoCoRenderProcess: Started render-process manager.
        """
        return self.start()

    def __exit__(self,exc_type,exc_value,traceback):
        """
        Close the render process when leaving a context manager.

        Parameters:
            exc_type (type | None): Exception type from the context body.
            exc_value (BaseException | None): Exception instance from the
                context body.
            traceback (traceback | None): Traceback from the context body.

        Returns:
            bool: False so exceptions from the context body are not suppressed.
        """
        self.close()
        return False


def main():
    """
    Entry point used by `python -c ...` when launching the render subprocess.

    Parameters:
        None.

    Returns:
        None.
    """
    render_server_main()


if __name__ == "__main__":
    main()
