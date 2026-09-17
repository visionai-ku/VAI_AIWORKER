# `ri_motion_v5_package.mujoco_sim`

MuJoCo parsing, simulation, drawing, XML utility, IK, optional native bind, and
process-separated rendering helpers.

## Modules

- `parser.py`
  - `MuJoCoParser`, lightweight viewer utilities, model/data accessors, stepping, rendering, and plotting hooks
- `mujoco_utils.py`
  - MJCF merge helpers, hand-frame helpers, and MuJoCo utility functions
- `mujoco_ik.py`
  - Site and body-position IK solvers using revolute/prismatic joints and optional base control
- `mujoco_draw.py`
  - Low-level drawing helpers for markers, arrows, frames, and overlays
- `interactive_marker.py`
  - `InteractiveMarker`, an ImGuizmo-style polling marker for editing 4x4 pose matrices in the MuJoCo viewer
- `mujoco_replay.py`
  - `MuJoCoReplay`, a small recorder/replayer for MuJoCo control trajectories on a shared env initialization
- `render_process.py`
  - `MuJoCoRenderProcess` for notebook-driven subprocess rendering over ZMQ
- `_native.py`
  - Optional native backend import helpers and Python wrappers around C++
    signed-distance, IK, and echo functions
- `_native_build.py`
  - Notebook-facing nanobind build and cleanup helpers for selected `.cpp` and
    `.hpp` files
- `cpp/`
  - C++ source, headers, and nanobind binding entry points
- `so/`
  - Python subpackage for compiled `_mujoco_*.so` extension modules; generated
    `.so` files are ignored by git

## Public API Snapshot

- `parser.py`: `MuJoCoParser`, `MuJoCoMinimalViewer`, `MinimalCallbacks`
- `mujoco_utils.py`: `normalize_mujoco_joint_types(...)`, `merge_mjcfs(...)`, `get_label_colors_from_joint_axis(...)`, `get_T_hand_from_sites(...)`
- `mujoco_ik.py`: `SitePositionIKSolverRevPriBase`, `BodyPositionIKSolverRevPriBase`, `normalize_lam_vec(...)`, `solve_dls(...)`, `project_dq_null_damped(...)`
- `mujoco_draw.py`: `draw_rotation_plane(...)`, `draw_rotation_arc(...)`, `draw_numpy_chain(...)`
- `interactive_marker.py`: `InteractiveMarker`
- `mujoco_replay.py`: `MuJoCoReplay`
- `render_process.py`: `MuJoCoRenderProcess`
- `_native.py`: `get_native_backend(...)`, `echo_model_data_native(...)`
- `_native_build.py`: `build_native(...)`, `clean_native(...)`

## Import

```python
from ri_motion_v5_package.mujoco_sim import *
```

## Notes

- Most interactive notebooks rely on this package together with `kinematics`, `utility`, and `qt`
- Keep viewer cleanup in notebook `finally` blocks when using `env.render()`
- Use `merge_mjcfs(...)` before creating `MuJoCoParser` when composing scenes from shared assets
- Use `env.plot_contact_info(...,max_contact_plot=N)` when a scene can produce many contacts; the overlay can still report total contacts while drawing only the first `N` markers/arrows
- Use `InteractiveMarker.get_T(update=True, draw=False)` in an update timer and `InteractiveMarker.draw(alpha=...)` in a render timer when input polling and rendering are separated
- Use `env.plot_tmr_lag(tmr)` for timer overlays; it expects a timer object such as `SimpleTimer` that implements `get_lag_info()` and shows both lag and recent run time

## Native Bind Layout

- C++ source and header files belong in `mujoco_sim/cpp/`.
- Compiled extension modules belong in `mujoco_sim/so/`.
- Notebook examples should keep `cpp_source_names`, `hpp_header_names`,
  `define_names`, and `module_name` visible in the build cell.
- Use `build_native(...)` to build only the selected C++ files for that notebook.
- Use `clean_native(module_name=module_name, verbose=True)` before rebuilding a
  notebook-local extension. It removes the selected file under `mujoco_sim/so/`,
  legacy `_mujoco_*.so` files under `mujoco_sim/`, and object files under
  `build/native/`.
- Import the compiled backend explicitly:

```python
be = get_native_backend(required=True, module_name=module_name)
```

- Production-style accelerated calls should keep the normal Python object as
  the caller and pass the compiled backend explicitly, for example
  `env.get_geom_pair_signed_distance_infos_bind(..., be=be)` or
  `ik_body_solver.solve_ik_bind(..., be=be)`.
- Native extension builds are currently macOS-only. On Linux, build/import
  helpers warn and skip rather than raising during normal notebook execution.

## Render Process Layout

- `MuJoCoRenderProcess` is a small lifecycle wrapper around a subprocess renderer.
- The notebook process owns sliders, UI events, and its own `MuJoCoParser`.
- The render process owns a separate `MuJoCoParser`, `MjModel`, `MjData`, viewer,
  and render loop.
- `MuJoCoRenderProcess(x_offset, y_offset, window_width, window_height, ...)`
  follows the same first-four-argument window layout convention as
  `MultiSliderQtWidget(...)`.
- `render(env, overlays=..., draws=..., mode="qpos", profile=True)` sends
  only `qpos`. The render process then runs `env.forward()` before drawing, and
  the profiling overlay reports that time as `used fk`.
- `render(env, overlays=..., draws=..., mode="snapshot")` sends
  already-forwarded render fields. The render process skips its own
  `env.forward()`.
- If markers are registered with `rp.add_interactive_marker(...)`, `render(...)` sends
  those marker commands automatically.
- `render(...)` sends only JSON-safe state and command payloads. It does
  not send the full Python environment object.
- `rp.overlay(...)` creates a text-overlay command for the render process.
- `rp.draw("plot_...", ...)` asks the render process to call matching
  `env.plot_*` helpers there.
- `rp.add_interactive_marker(...)` registers a render-process `InteractiveMarker`.
  `rp.update_markers()` updates registered marker state from the subprocess
  viewer, and `rp.get_marker_T(...)` reads the latest pose.
- `rp.get_final_rgb_img()` returns the final RGB image captured by the render
  subprocess when its viewer exits. Keep display calls such as
  `imshow(rp.get_final_rgb_img())` in `finally`, not inside the render loop.
- Set `profile=False` in `render(...)` to hide the render-loop time
  profiling overlay for that scene message.
- Notebook sender loops should use `update_qt()` for local Qt events and
  `sleep(sec)` for a short main-process delay.
- Always close the process in a `finally` block with `rp.close()`. `max_sec` is
  only a safety limit against forgotten subprocesses.

## Interactive Marker Convention

- `InteractiveMarker` is the package-backed ImGuizmo-style pose editor for 4x4 transforms.
- It uses polling, not notebook-installed viewer callbacks, so marker behavior stays local to the marker object.
- Create the marker once and reuse it; when the selected body changes, call `set_T(...)` and optionally `set_T0(...)` instead of constructing a new marker every loop.
- In timer-split notebooks, use `get_T(update=True, draw=False)` in the update block and `draw(alpha=...)` in the render block.
- For multiple markers, call `get_T(update=True, draw=False)` on all markers in the update block before applying object poses or solving IK, then call `draw(alpha=...)` on all markers in the render block.
- Viewer drag suppression is press-owned: a marker suppresses the viewer only if the current left-button press started on a marker handle. This allows camera drags that pass over a marker to keep working.
- `translation_frame` controls only translation handles: `"local"` follows the marker frame and `"global"` follows world axes. Rotation handles remain marker-local.
- The top handles are reserved for marker-level actions such as reset rotation and local/global translation-frame toggle.
- `hover_scale` should thicken hovered handles only; it should not change translation-axis length.
- `plane=None` lets the class derive plane-handle offset, size, and thickness from `axis_len`.
- `mouse_view_offset=(screen_right, screen_up)` is a view-space compensation knob for local pointer bias. Keep it as a visible notebook parameter when used.
- Avoid pixel-space hover offsets because they become depth/zoom dependent after projection.

## Object PCD Convention

- Public object point-cloud helpers default to `pcd_mode="mesh"`.
- `"mesh"` samples the original visual asset mesh when possible and falls back to compiled-mesh surface sampling when alignment is unreliable.
- `"face"` and `"vertex"` are explicit diagnostic modes for compiled MuJoCo mesh sources.
- Use `get_obj_body_name_left_double_click(...)` for double-click object selection; it compares the clicked world point to each candidate body's current world-frame point cloud.
- Double-click picking reconstructs the clicked point from a depth-buffer capture; by default it temporarily disables viewer transparency for that capture through `pick_opaque=True`, then restores the previous viewer transparency state.
- Notebooks should call double-click selection from the update block when it mutates target state; render code should only draw the already computed click feedback.
- When object names can vary, pass or rely on `obj_body_prefix="body_obj"` rather than hardcoding every body name in the notebook.

## IK Marker Convention

- `InteractiveMarker` can replace Qt pose sliders when an IK target should be edited directly in the MuJoCo viewer.
- For body-coordinate IK, use `BodyPositionIKSolverRevPriBase.add_ik_coordinate_target(...)` instead of recreating body-local coordinate target math in notebooks.
- For dual-hand FFW SH5 body-coordinate IK, use one marker for the right hand target and one marker for the left hand target.
- Initialize FFW SH5 hand markers from `get_T_hand_from_sites(...)`; the current target body names are `hx5_r_base` and `hx5_l_base`.
- Keep marker polling, IK target construction, `solve_ik(...)`, and `env.forward(...)` in the update block. Keep marker drawing, IK target visualization, timer overlays, and `env.render()` in the render block.
- If `solve_dls(...)` reports `max|J|=0.0`, the target Jacobian restricted to `joints_use` is zero. Check the target body name and whether the selected joints can move that body before changing damping or solver gains.
- `InteractiveMarker.set_T(...)` defaults to `clear_state=True`. This is useful after snapping a marker to a corrected hand pose, but repeated per-frame calls can clear hover and drag state while a user is trying to click or drag the marker.

## Signed-Distance Depenetration Notes

- Keep render/query top-k and depenetration top-k separate in notebooks. Rendering five nearest pairs does not mean all five should become IK targets.
- Start with small depenetration steps such as `0.01` m when debugging. Large fixed targets can make `ik_err_array` increase and cause `idx_best == 0`, which means the solver intentionally kept the original pose.
- If a target arrow is visible but the robot does not move, inspect the target body and the `joints_use`-restricted Jacobian before adding nullspace flags or dynamic joint-selection logic.
- Keep package responsibilities narrow: `MuJoCoParser.get_dep_trgts_from_signed_dists(...)` converts signed-distance results into reusable body-point targets, while `BodyPositionIKSolverRevPriBase.add_body_trgts(...)` only appends those targets to the solver buffer.
- Leave workflow-specific policy in notebooks, including marker re-anchoring, solve flags, timing, safe-margin sliders, and render overlays.

## Timer Overlay Convention

- Timer lag diagnostics belong in `SimpleTimer.get_lag_info()` and `MuJoCoParser.plot_tmr_lag(...)`, not notebook-local helper functions.
- `plot_tmr_lag(tmr)` should show missed ticks, equivalent lag in milliseconds, and recent run time from `tmr.get_ems()`.
- Keep timer overlays in the render block so update-rate diagnostics do not add work to the update path.
