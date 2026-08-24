#### SOUL.md

This document defines a reusable coding and notebook style. It should work as
a general convention file across repositories, not as an index of one specific
project layout.

The goal is to keep research code, notebooks, package modules, native bindings,
rendering workflows, and validation examples explicit, readable, rerunnable,
easy to debug, and easy to later convert into clean reusable functions.

Avoid writing durable rules that depend on one repository name, one folder
name, one notebook filename, or one local machine path. Function names, class
names, and API names may appear when they describe an actual coding convention.

#### Single Source Of Style Rules

Use one root-level style document as the durable source of coding conventions.
Folder-level README files may describe local file lists, setup notes, or module
surfaces, but cross-repository coding rules belong in the single style
document.

When a new convention is learned from an advanced notebook or package pattern,
write it here in generic terms so the same convention can be reused in another
repository.

#### Core Philosophy

Write notebooks as executable explanations. A notebook should show the
experiment in the order a researcher needs to understand it:

- What is imported.
- What is built or compiled.
- What is loaded.
- What is configured.
- What is controlled.
- What is rendered.
- Which loop is running.
- What is timed or validated.

Default workflow:

1. Write the first working notebook in an explicit linear style.
2. Put variables near the block that uses them.
3. Keep the execution order reproducible.
4. Verify `Restart Kernel and Run All` when feasible.
5. Extract reusable functions only after the notebook behavior is clear.

For interactive notebooks, change one behavior at a time. Treat the last
working loop as the baseline. If behavior regresses, inspect the last change
before adding a workaround. Do not mix behavior changes, performance tuning,
API renaming, overlay cleanup, and refactoring in one edit unless that combined
change is explicitly requested.

#### General Module Style

Package code should be small, explicit, and easy to call from notebooks. A
module should provide reusable behavior, but it should not hide experiment
decisions that belong in an explained notebook.

Prefer a clear data flow:

1. Build a small explicit data object, often a dictionary.
2. Inspect or summarize that data when useful.
3. Pass that same data object to the method that mutates internal state.

Public methods should be direct and conservative:

- Use method names that describe the side effect.
- Return useful values when they help notebooks show what happened.
- Accept explicit source objects such as `env` when the method reads external
  state.
- Avoid implicit global state.
- Avoid adding dependencies only for nicer printing or display.
- Keep return shapes stable when adding accelerated or optional paths.

Internal helpers should be simple and local to the class or module. Use a
leading underscore for helpers that are not part of the notebook-facing API.
Do not turn every repeated line into a helper; extract only when it clarifies a
real state transition or prevents meaningful mismatch between notebook and
package behavior.

#### Documentation Style

Documentation should orient a reader quickly. It should not become a stale
mirror of the full codebase.

Good README content:

- A short purpose paragraph.
- A compact guide to important files, notebooks, or modules.
- Setup notes only when there are real environment assumptions.
- A public API snapshot when a module has a stable user-facing surface.
- Notes about cleanup, timing, viewer, asset, generated-file, or validation
  assumptions.

Keep README descriptions synchronized with actual names and behavior. If a
notebook or module is renamed, update the README. If an interaction pattern
changes, update the README. Avoid stale object counts, stale asset lists, old
filenames, and old screenshots.

#### Markdown Heading Scale

Prefer small headings in Markdown files and notebook Markdown cells. Use
`####` as the default heading level instead of `#` or `##` unless a file
already has an established hierarchy that should not be disturbed.

The preferred style is compact notes inside the editor, not a large rendered
web page. Educational notebook Markdown may be detailed, but headings should
stay visually modest.

#### Notebook Cell Structure

Most notebooks should follow this order:

1. Markdown purpose cell.
2. Import-only code cell.
3. Asset, file, or environment construction.
4. Configuration before runtime.
5. Build or compile cells when needed.
6. Validation or API check cells when needed.
7. Final rerunnable runtime cell.
8. One empty code cell at the very end is allowed.

The first code cell should contain imports only. Avoid mixing imports with path
setup, build logic, asset selection, environment creation, or runtime
variables.

Prefer project bootstrap imports when a package already provides them. Do not
add redundant imports such as `import numpy as np` when the standard project
import surface already provides `np`. Extra imports are fine when the notebook
actually needs them.

The first Markdown cell should briefly explain:

- The goal of the notebook.
- The main model, environment, or data source.
- The main interaction pattern such as viewer, sliders, markers, replay, IK,
  forward dynamics, native binding, or render process.
- Any important assumption before running.

Keep the first Markdown cell concise. It should orient the reader, not become a
long report.

#### Imports

Keep imports in one import-only cell whenever possible.

Preferred style:

```python
from package.bootstrap import *
from package.module_a import *
from package.module_b import *
```

Use the actual package import surface for the repository. The rule is not the
specific import name; the rule is that imports should be centralized, visible,
and not mixed with runtime logic.

#### Environment And Setup Cells

Use early setup cells for things such as:

- Asset or file selection.
- XML, scene, model, or dataset composition.
- Parser or environment construction.
- Auxiliary environments used for IK, target state, or validation.
- Object or robot selection that must happen before loading.

These cells should prepare the experiment. They should not start the viewer
and should not contain the long runtime loop.

Use compact one-line calls when they are easy to read. Use aligned keyword
arguments for longer multiline calls. Keep source inputs visible when a scene,
dataset, or model is generated from smaller pieces.

#### XML And Model Composition

Treat generated XML, merged scenes, and compiled models as products of visible
inputs, not as hidden magic.

Preferred XML workflow:

- Keep the source XML or MJCF component list visible in the notebook.
- Generate or merge a scene from those sources in one setup cell.
- Store generated scenes under a local generated-output location when the
  repository has one.
- Use generated XML as a reproducible intermediate, not as the only source of
  truth.
- Print or inspect important runtime names after loading when names may depend
  on the selected assets.
- Keep robot, object, fixture, floor, and camera inputs conceptually separate
  even when they are merged into one final model.

Use XML generation cells to answer these questions:

- Which components make the scene?
- Where is the generated scene written?
- Which parser or model object loads the generated scene?
- Which runtime body, joint, geom, site, camera, or actuator names matter?

Avoid hardcoding runtime names that come from a variable asset set unless the
notebook first checks or derives them from the loaded model. When a model is
fixed and educational, keeping a checked name visible is acceptable.

Generated XML files, logs, rollouts, and other derived outputs should not be
treated as hand-authored source files unless the notebook is specifically about
inspecting that generated file.

#### Parser And Environment Handling

Parser or environment objects should make ownership clear. A notebook may need
several environments for different roles, such as visible state, IK state,
target state, validation state, or render-process state. Name and use them by
role, not by accident.

Preferred parser conventions:

- Create parser or environment objects in setup cells, before runtime widgets
  and loops.
- Use one main environment for visible or authoritative notebook state.
- Use auxiliary environments when a solver, target, comparison, or validation
  path needs an independent state.
- Reset environments explicitly before starting an interactive runtime loop.
- Call `forward()` explicitly after direct state mutation when later code
  depends on kinematic or geometric state.
- Keep model-level mutations such as friction, actuator gains, geom colors, or
  margins in setup cells or in clearly marked runtime sections.
- Cache expensive geometry or point-cloud data once when it is static.
- Synchronize auxiliary environments from the main environment deliberately;
  do not rely on hidden cell execution order.

When multiple environments exist, prefer names that describe their role:

```python
env       = MuJoCoParser(rel_xml_path=xml_path,verbose=False)
env_ik    = MuJoCoParser(rel_xml_path=xml_path,verbose=False)
env_trgt  = MuJoCoParser(rel_xml_path=xml_path,verbose=False)
env_check = MuJoCoParser(rel_xml_path=xml_path,verbose=False)
```

The exact class name may differ by repository. The rule is that parser objects
should be explicit, role-labeled, and synchronized intentionally.

#### Configuration Cells

Configuration cells should define the knobs used by the runtime cell:

- Joint, body, control, or object grouping.
- Gains, frictions, controller settings, and solver options.
- IK joint names, end-effector names, object names, and target names.
- Viewer settings, marker settings, and render settings.
- Constants that are used by the final runtime cell.

Keep variables close to the code that uses them. If a value is only needed for
friction setup, define it immediately above the friction setup. If a value is
only needed for IK setup, define it near the IK setup.

Loop over dictionaries for repeated setup. Avoid writing several nearly
identical calls by hand when a small explicit dictionary makes the intent
clearer.

#### Final Runtime Cell

The final non-empty cell should be the main runtime cell. It should be safe to
rerun after closing the viewer or stopping the process.

For viewer-based notebooks, the final runtime cell should usually contain:

- Environment reset.
- Viewer or render-process initialization.
- Initial robot and object poses.
- Slider, widget, marker, or controller initialization.
- IK solver initialization.
- Timer initialization.
- Main loop.
- Update block.
- Render block.
- Cleanup in `finally`.

The final runtime cell should be designed so the user can repeatedly run only
that cell. It should not require restarting the kernel or rerunning all earlier
cells after closing an interactive window.

#### Important Looping Pattern

Interactive notebooks should use a clear loop contract. A loop should make it
obvious which state is updated, which state is rendered, how often each block
runs, and how the loop stops.

Preferred loop shape:

- Start from a clean reset.
- Create runtime-only objects inside the final cell.
- Start timers before the loop.
- Run a short always-on loop condition such as viewer alive, process alive, or
  user stop flag.
- Poll Qt or UI events once near the top of the loop.
- Put state mutation in the update block.
- Put drawing and rendering in the render block.
- End each timer inside the block it measured.
- Sleep briefly only to yield idle CPU time, not to control the render process.
- Close runtime objects in `finally`.

The update block should own:

- Slider or widget reads.
- Interactive marker reads.
- Controller target updates.
- IK solve.
- Collision or distance queries.
- State mutation.
- `forward()` or simulation stepping when needed.

The render block should own:

- Text overlays.
- Draw command construction.
- Plot calls.
- Image or RGB overlays.
- Viewer render or `render(...)`.

Do not let render code recompute the expensive state that the update block
already computed. If render needs contact, distance, IK, sensor, or timing
data, compute it in update and pass the cached result to render.

Use the simplest loop that shows the intended behavior. Add profiling,
subprocess rendering, native binding, or marker synchronization one at a time
after the baseline loop works.

Preferred structure:

```python
env.reset()

tmr_update = SimpleTimer(name="Update",Hz=100,verbose=False)
tmr_render = SimpleTimer(name="Render",Hz=30,verbose=False)
tmr_update.start()
tmr_render.start()

try:
    while env.is_viewer_alive():
        if tmr_update.do_run():
            # Poll widgets, update markers, solve IK, step simulation.
            tmr_update.end()

        if tmr_render.do_run():
            # Draw overlays, markers, frames, and render.
            env.render()
            tmr_render.end()
finally:
    env.close_viewer()
```

Use `try/finally` for cleanup rather than hiding exceptions. Cleanup should
close viewers, widgets, devices, sockets, subprocesses, and displays reliably
while still allowing real errors to surface.

#### Rerunnable Interactive Contract

Interactive notebooks should obey a strict rerunnable contract:

- Closing the viewer must not leave the notebook in a broken state.
- The final cell must be safe to run again.
- Runtime objects that depend on the viewer should be created inside the final
  cell.
- Timers should be created inside the final cell.
- Markers, sliders, widgets, and transient viewer objects should be created
  inside the final cell unless there is a clear reason they must persist.
- Cleanup should happen in `finally`.
- Use helper cleanup functions such as `close_objects(...)` when they make the
  final cell shorter and clearer.

If a runtime cell mutates model parameters such as geometry properties,
friction, equality settings, or actuator gains, store the baseline before the
loop and restore it at the start or in `finally` when needed.

Do not leave autoreload cells, monkey patches, reload experiments, or temporary
debug blocks in final notebooks.

#### Qt And Widget Handling

Qt widgets should be treated as runtime objects. Create them near the loop that
uses them, update Qt events in the loop, and close them with the rest of the
runtime objects.

Preferred Qt conventions:

- Keep Qt setup in the standard import/bootstrap cell when the repository
  provides one.
- Use `update_qt()` or the repository's equivalent helper instead of calling
  raw Qt event-processing code in notebook loops.
- Let widget constructors or `show()` methods handle ordinary show, raise, and
  activation behavior when possible.
- Avoid repeated manual calls such as `show()`, `raise_()`, and
  `activateWindow()` in notebooks when a widget API can own that behavior.
- Create sliders, image widgets, control panels, and marker panels in the final
  runtime cell unless they intentionally persist across runs.
- Close widgets with `close_objects(...)` or another small cleanup helper in
  `finally`.

Window layout arguments should be visible and compact. When a widget uses
positional layout values, keep them on the first continuation line:

```python
sliders = MultiSliderQtWidget(
    0.00, 0.05, 0.20, 0.30,
    title        = "target pose",
    label_texts  = ["X","Y","Z","Roll","Pitch","Yaw"],
    slider_vals  = [0.0,0.0,1.0,0.0,0.0,0.0],
)
```

The notebook should show what the user can manipulate. It should not expose raw
Qt boilerplate unless the notebook is specifically about debugging Qt.

#### Compact Multiline Style

Prefer compact multiline calls when the call remains readable. Long or
semantic arguments should stay on separate lines, but short keyword groups may
share one line to reduce notebook height.

This style is especially useful for UI widgets, render-process construction,
marker style, and timer name lists.

Preferred compact style:

```python
slider_safe_margin = SingleSliderQtWidget(
    0.00, 0.05, 0.20, 0.05, "Safe margin",
    slider_min = 0.00, slider_max = 0.10, slider_val = 0.02, resolution = 0.001,
)

rp = MuJoCoRenderProcess(
    0.2, 0.00, 0.6, 1.00,
    xml_path  = xml_path,
    title     = "interactive bind example",
    render_hz = 20.0,
    max_sec   = 300.0,
)
kwargs_marker = {"show_frame_toggle":False,"show_rotation_reset":False}
rp.add_interactive_marker(
    "right",
    get_T_hand_from_sites(env,site_name_common="right_site",hand_type="right"),
    axis_len = 0.20, ring_r = 0.20, alpha = 0.60, **kwargs_marker,
)
rp.add_interactive_marker(
    "left",
    get_T_hand_from_sites(env,site_name_common="left_site",hand_type="left"),
    axis_len = 0.20, ring_r = 0.20, alpha = 0.60, **kwargs_marker,
)
rp.start()
```

Use this compact form only when the resulting line remains easy to scan. Keep
large dictionaries, nested calls with many meanings, and long expressions
split across lines.

#### Variable Placement

Put variables near their point of use.

Avoid collecting every constant at the top of the notebook if that makes the
reader jump back and forth. A variable should usually appear immediately before
the block that uses it.

Avoid one-use alias variables in notebooks. If a string, number, or simple
expression is used only once, put it directly at the call site. Use a variable
when it is reused across separated blocks, printed, tuned by the user, saved,
passed to several APIs, or when the name adds important semantic information.

Use uppercase names for configuration dictionaries that describe experiment
settings, such as gain maps, friction maps, and target configuration maps.

Use clear quantity-style names for spatial values:

- `p_*` for position.
- `R_*` for rotation matrix.
- `T_*` for homogeneous transform.
- `q_*` or `qpos_*` for joint positions.
- `idxs_*` for index arrays.

When combining a quantity tag with an object tag, prefer quantity-first names
such as `p_cam`, `R_cam`, `T_cam`, `p_target`, and `T_target`.

#### Naming Convention

Names should explain role, unit, and lifecycle without becoming long.

General rules:

- Use short names for common runtime handles when the meaning is local and
  repeated, such as `env`, `be`, `rp`, `tts`, `tmr_update`, and `tmr_render`.
- Use role-specific suffixes when several objects share a type, such as
  `env_ik`, `env_trgt`, `ik_marker_solver`, and `ik_dep_solver`.
- Use `_bind` only at the end of a function or method name that calls a native
  bound backend path.
- Use `(bind)` in overlay text, not as a vague label suffix.
- Use `*_info` for dictionaries that summarize a result.
- Use `*_cmds` for lists of commands passed to another system, such as draw or
  overlay commands.
- Use `*_names` for lists of names and `idxs_*` for index arrays.
- Use `n_*` for counts and `is_*`, `has_*`, or `*_flag` for booleans.
- Use `TOPK_*`, `DEP_*`, `CTRL_*`, or similar uppercase names for true
  notebook configuration knobs.
- Prefer `set_trgt` or `target` labels when the code builds targets. Avoid
  vague labels such as `path`, `thing`, `data`, or `result` when a more
  specific role is known.

File and folder names should be ordered, descriptive, and stable:

- Use numeric prefixes when a folder is meant to teach a sequence.
- Use lowercase snake_case for notebooks, scripts, generated scenes, and helper
  modules unless the repository already uses another convention.
- Rename files when their purpose changes; do not leave old names that describe
  a previous implementation.
- Keep README entries synchronized with renamed files.
- Avoid names that encode temporary debugging goals after the notebook becomes
  a durable example.

Function names should match the user's call-site expectation. If the normal
Python path is `object.operation(...)`, the accelerated path should look like
`object.operation_bind(...,be=be)` rather than a separate backend object with a
surprising name. Tutorial functions may be direct backend calls such as
`be.foo(...)`; production-style calls should usually stay on the normal object
surface.

#### Timing And Profiling

Use timers to separate update logic from render logic.

Preferred timer split:

```python
tmr_update = SimpleTimer(name="Update",Hz=100,verbose=False)
tmr_render = SimpleTimer(name="Render",Hz=50,verbose=False)
```

Do simulation, marker polling, controller updates, IK updates, collision
queries, and state mutation in the update block. Do drawing, overlays, image
display, and viewer rendering in the render block.

Avoid expensive simulation recomputation in the render block. If contact
information, sensor values, or dynamic state are already computed in the update
block, pass that payload to the plotting function in the render block.

For multiple named timing sections, use one `TicTocs` object named `tts`:

```python
tts = TicTocs(
    "marker_total", "marker_target", "marker_solve", "marker_forward",
    "signed_distance", "dep_target_build", "dep_ik",
)
```

Prefer `tts.tic("name")`, `tts.toc("name")`, and `tts.ms("name")` over many
individual `TicToc` variables and matching `esec_*` variables.

Timer overlays should be concise and grouped. For timer status, prefer
`actual ms / period ms / target Hz`, for example
`"[9.40]ms / [10.0]ms / [100]Hz"`. The overlay should show both how long the
last measured block took and the timer period it was supposed to meet. Avoid
vague `lag`, `load`, or duplicated total overlays unless the notebook is
specifically diagnosing timing budget.

For grouped timing overlays, show the total first and internal timings below it
with visible indentation:

```python
overlay_cmds = [
    rp.overlay(
        "update timer",
        "[%.2f]ms / [%.1f]ms / [%d]Hz"%(update_run_ms,update_period_ms,tmr_update.Hz),
    ),
    rp.overlay(
        "render timer",
        "[%.2f]ms / [%.1f]ms / [%d]Hz"%(render_run_ms,render_period_ms,tmr_render.Hz),
    ),
    rp.overlay("marker ik total","[%.2f]ms"%(tts.ms("marker_total"))),
    rp.overlay("  marker ik set trgt","[%.2f]ms"%(tts.ms("marker_target"))),
    rp.overlay("  marker ik solve (bind)","[%.2f]ms"%(tts.ms("marker_solve"))),
    rp.overlay("  marker ik fk","[%.2f]ms"%(tts.ms("marker_forward"))),
    rp.overlay("dep total","[%.2f]ms"%dep_time_ms),
    rp.overlay("  signed dist (bind)","[%.2f]ms"%(tts.ms("signed_distance"))),
    rp.overlay("  set trgt","[%.2f]ms"%(tts.ms("dep_target_build"))),
    rp.overlay("  dep ik (bind)","[%.2f]ms"%(tts.ms("dep_ik"))),
]
```

Prefer `(bind)` inside labels over appending `bind` as a vague suffix. Use
explicit quality labels such as `marker ik err` and `dep ik err`.

#### Native C++ Bind Pattern

When adding a small C++ function that should be callable from Python, keep the
full build and binding path visible in the teaching notebook.

Preferred sequence:

- Declare the function in the matching `.hpp` file.
- Implement the function in the matching `.cpp` file.
- Add or update the feature-specific binding file.
- Expose the function with `m.def(...)`.
- Call the feature-specific binding registration from the module entry point.
- Keep the notebook build cell self-contained with `cpp_source_names`,
  `hpp_header_names`, `define_names`, and `module_name`.
- Build only the C++ files needed by that notebook example.
- Use distinct native module names for examples with different exported
  functions.
- Call `clean_native(module_name=module_name,verbose=True)` before rebuilding
  notebook-local native examples.
- After importing the backend as `be`, print the public API with
  `enumerate(...,start=1)` and `[%02d]` numbering.
- Verify the function from Python with explicit `be.some_function(...)` calls.

For notebook-local native examples, the notebook should teach which files
participate in the build and why. The build helper should build what the
notebook asks for; it should not hide the example's file selection in a
package-level default.

#### Production-Style Bind API

After a native function moves beyond the tutorial stage, expose it through a
Python-facing `_bind` method or function that keeps the normal package object
as the main caller and receives the compiled backend explicitly as `be=be`.

Preferred call shape:

```python
be = get_native_backend(required=True,module_name=module_name)

result = some_object.some_operation_bind(
    ...,
    be = be,
)
```

For `_bind` APIs:

- Always put `be` after `*` as a keyword-only argument.
- Keep `be` explicit at each accelerated call site.
- Do not store `be` silently inside a solver or parser object unless that style
  is explicitly chosen.
- Hide raw native addresses inside package-level wrappers.
- Notebooks should pass normal objects such as `env`, `model`, `data`, or
  solver instances.
- Preserve the non-bind Python API and return shape. For example,
  `solve_ik_bind(...,be=be)` should return the same tuple shape as
  `solve_ik(...)`.
- On unsupported platforms, prefer warning and falling back to the Python path
  over raising during normal notebook execution.
- For timing notebooks, compute both Python and bind paths only when the
  comparison itself is useful. Use the bind result for the visible workflow
  when validating the accelerated path.

For performance-sensitive bind work, move the full repeated loop into C++ when
possible. Binding only a tiny subroutine may not improve notebook
responsiveness because Python-to-C++ call overhead and Python-side allocation
can dominate.

#### Process-Separated Rendering

Some notebooks should render through a render-process object instead of opening
the viewer in the notebook process. This is useful when the notebook kernel
should keep owning sliders, Qt events, IK, control logic, or experiment state
while a separate subprocess owns its own model, data, viewer, and render loop.

Treat transport details such as sockets or ZMQ as implementation details of
the render-process class. The notebook-facing concept is a process-separated
render process.

Preferred shape:

```python
rp = MuJoCoRenderProcess(
    0.25, 0.00, 0.75, 1.00,
    xml_path  = xml_path,
    title     = "render process",
    render_hz = 30.0,
    max_sec   = 300.0,
)
rp.start()

try:
    while rp.is_running():
        update_qt()
        rp.update_markers()
        rp.render(
            env      = env,
            overlays = overlay_cmds,
            draws    = draw_cmds,
            mode     = "snapshot",
            profile  = False,
        )
        sleep(0.001)
finally:
    close_objects("rp")
```

When constructing `MuJoCoRenderProcess`, keep the first four positional window
layout arguments on the first continuation line, then align keyword arguments
vertically. This mirrors widget placement and makes notebook window layout easy
to scan.

Use `rp.overlay(...)` for text overlays and `rp.draw("plot_...",...)` for
render-process calls to existing plotting helpers. Do not expose raw sockets,
subprocess handles, protocol dictionaries, or transport imports inside
notebooks unless the notebook is explicitly about debugging the protocol.

Use `mode="snapshot"` when the notebook-side environment already called
`forward()` and the render process should only copy render-relevant state
before drawing. Use `mode="qpos"` when the render process should receive
`qpos` and run its own `forward()`.

Keep `profile=True` only when diagnosing render timing. Prefer `profile=False`
in polished notebook loops.

The notebook process and render process do not share Python objects. The
notebook must send JSON-safe scene state, overlays, and draw commands; the
render process must own and close its own viewer. Always close the render
process in `finally`. Use `max_sec` only as a safety guard against forgotten
subprocesses, not as the normal cleanup mechanism.

#### Render-Process Final Images

Final viewer images belong to the render process because that process owns the
viewer, OpenGL context, camera state, overlays, and draw commands. Do not add
`imshow(...)`, image capture, socket polling, sleeps, or timeout-based waits
inside the runtime render loop just to inspect the terminal frame; that changes
the loop timing and obscures the control flow.

For notebook-side inspection of the terminal frame, keep the loop unchanged and
put the display in cleanup:

```python
try:
    while rp.is_running():
        update_qt()
        if tmr_render.do_run():
            rp.render(env,overlay_cmds,draw_cmds,profile=True)
            tmr_render.end()
finally:
    imshow(rp.get_final_rgb_img())
    close_objects("rp")
```

The package API should do the process-boundary work: the render process detects
viewer shutdown, captures its own final RGB image, sends it through its feedback
channel, and `rp.get_final_rgb_img()` only drains that already-sent feedback and
returns a copy. Keep raw sockets, protocol messages, retry loops, and timeout
heuristics out of notebooks unless the notebook is explicitly testing the
transport layer.

#### Render-Process Interactive Markers

For render-process `InteractiveMarker` notebooks, mouse input belongs to the
render subprocess because that subprocess owns the viewer window.

Use render-process markers through the render-process object:

```python
rp.add_interactive_marker(
    "right",
    T_right_init,
    axis_len = 0.20, ring_r = 0.20, alpha = 0.60, **kwargs_marker,
)
rp.update_markers()
T_right_target = rp.get_marker_T("right")
rp.set_marker_T("right",T_right_current)
```

Marker count, names, initial poses, and style should be visible near
`rp.add_interactive_marker(...)`. Do not rebuild marker protocol dictionaries by hand
inside the runtime loop. `rp.update_markers()` should be called in the update
block, not the render block, because marker state usually drives IK or control.

#### Advanced Bind And Render-Process Convention

For advanced notebooks that combine C++ bind acceleration, collision-aware IK,
process-separated rendering, and render-process interactive markers, use this
division of responsibility:

- The notebook process owns the main environment, solver objects, native
  backend `be`, Qt widgets, update timers, bind calls, and experiment state.
- The render process owns its own model, data, viewer, render loop, and mouse
  input for render-process markers.
- The notebook sends render-relevant scene snapshots, draw commands, overlays,
  and marker updates.
- The render process should not recompute expensive IK or collision logic that
  the notebook process already computed.

For bind calls in this style, keep the backend explicit:

```python
geom_signed_distance_infos = env.get_geom_pair_signed_distance_infos_bind(
    candidate_geom_pairs,
    topk = TOPK_RENDER,
    be   = be,
)
q_marker,marker_info = ik_marker_solver.solve_ik_bind(
    env = env,
    be  = be,
)
```

Do not add a Python-vs-bind comparison path inside the advanced runtime loop
unless the notebook's purpose is profiling that comparison. For a polished bind
workflow, use the bind result for the visible environment and show bind-path
timing only.

Python helper steps may still build targets from bind results when that is the
clear package boundary, such as building depenetration targets from
signed-distance results.

#### Interactive Markers And IK

For robot IK notebooks, prefer interactive markers and explicit target setup.
The marker code should be visible enough that the user can understand which
marker controls which end-effector, body, hand, gripper, or target.

Preferred patterns:

- Use model-provided sites, bodies, or stable helper APIs to compute frames.
- Use body-based IK targets when site targets are not available.
- Create markers in the final runtime cell so they are recreated when the
  final cell is rerun.
- Poll marker transforms in the update block.
- Draw markers in the render block.
- Keep target body names, marker colors, and transforms explicit.

If a model may not have site definitions, handle that gracefully by falling
back to body transforms or another explicit target source.

#### Collision And Depenetration

Object and contact settings should be explicit. When setting frictions, gains,
or grouped body properties, define:

- A prefix dictionary.
- A grouped-name dictionary.
- A configuration dictionary.
- A short loop that applies each group.

For signed-distance depenetration:

- Keep render and solve counts as separate knobs. Use names such as
  `TOPK_RENDER` for plotted nearest pairs and `TOPK_DEPENETRATE` for the small
  number of pairs converted into IK targets.
- Use conservative step lengths.
- If the solver chooses the original pose, inspect whether the proposed target
  set made the error worse before changing gains.
- If targets are visible but the robot does not move, check target body names
  and the Jacobian before changing solver gains.
- If depenetration changes an end-effector pose in an interactive-marker
  notebook, re-anchor the marker target to the corrected pose.

#### Plotting And Rendering Order

Inside a render block, keep viewer calls in a stable order:

1. Text overlays.
2. Image or RGB overlays.
3. Plot calls.
4. Render call.

Treat timer-info helpers that draw text as part of the text-overlay group.
This keeps visual state easy to scan and prevents late overlay calls from being
scattered around the render path.

When a notebook teaches a rendering or sensing path, avoid broad composite
helpers that hide multiple stateful render operations. Prefer showing the
concrete order first, then replacing that sequence with a package helper only
after the behavior is clear.

#### Validation

Use the intended Python environment consistently for notebooks, package checks,
and terminal validation. If a terminal import fails but a notebook works, first
compare the Python executable before changing code or paths.

Validation expectations:

- After changing reusable package code, run focused validation notebooks or an
  equivalent smoke check for the touched surface.
- Validation notebooks should be small and narrow.
- Viewer, Qt, render-loop, teleoperation, socket, and device changes still
  need a manual interactive notebook check.
- Static imports and headless smoke tests cannot verify interactive behavior.
- Optional hardware or SDK integrations should default to non-fatal behavior in
  general notebooks.
- Use fail-fast behavior only in scripts or notebooks whose explicit purpose is
  to require that device.

External model, API, or credential-based notebooks should keep local tokens and
machine-specific paths out of committed outputs.

#### Debugging And Unit Notebook Workflow

When developing a complex feature, build it in small visible steps. The goal is
to isolate each assumption before integrating it into the final interactive
notebook.

Preferred debugging sequence:

1. Inspect the current folder, package, and notebook structure before editing.
2. Create the smallest possible unit notebook or unit script for the new
   concept.
3. Keep the first unit example trivial enough to prove the mechanism, not the
   final algorithm.
4. Print or echo the minimum state needed to verify that parser, model, data,
   backend, widget, process, or API handles are connected correctly.
5. Only then connect the mechanism to the real workflow.
6. Add timing after correctness is visible.
7. Add rendering or interaction after the compute path is visible.
8. Fold the successful convention back into Markdown docs.

Good unit examples are intentionally small:

- A native binding unit first proves that a tiny C++ function can be compiled,
  imported, listed, and called from Python.
- A parser bridge unit first proves that model and data fields can be read
  correctly.
- A render-process unit first proves that a subprocess can receive state and
  render it.
- An interactive-marker unit first proves that marker poses can be created,
  updated, read, and sent back to the main loop.
- A performance unit first compares the same semantic operation through two
  paths before the faster path becomes the default.

Keep unit notebooks self-contained enough that a reader can rerun them without
searching package defaults. Build cells should show which sources, headers, or
defines participate. Runtime cells should show which backend, parser, process,
or widget object is active.

Debug from the actual failure outward:

- Read the error message first.
- Verify that the expected variable was created in an earlier cell.
- Check whether the notebook relies on hidden execution order.
- Check whether the active Python environment matches the notebook kernel.
- Check whether the viewer, widget, subprocess, socket, or native extension is
  stale from a previous run.
- Clean or close only the object that is actually stale.
- Avoid broad rewrites while diagnosing a narrow failure.

When a prototype works, remove debug-only prints and stale cells, but keep
useful verification prints that teach the workflow or confirm an important
state transition.

#### Notebook Hygiene

Before a notebook is considered final:

- Restart the kernel and run all cells when feasible.
- Clear stale outputs when they are not part of the intended result.
- Clear execution counts.
- Remove empty code cells in the middle of a notebook.
- Remove debug-only cells.
- Remove print-only status cells that were used only while debugging.
- Remove inline debug prints that do not teach setup, verify a meaningful state
  transition, or summarize a payload the reader needs.
- Remove commented-out experiments unless they document a deliberate option.
- Make sure the final cell is rerunnable after closing the viewer.
- Make sure imports are in the import cell.
- Make sure runtime variables are not accidentally created only by old cell
  execution order.

Committed outputs are allowed only when the visible result is part of the
notebook's purpose, such as a final image, chart, compact table, or recorded
validation evidence. Execution counts should still be cleared before the
notebook is treated as polished.

A single empty code cell at the very end of a notebook is allowed. Other empty
or scratch cells should not remain in a finished notebook unless their purpose
is clear.

Do not leave a notebook that only works because cells were run in a hidden
manual order.

#### Editing Existing Notebooks

When modifying an existing notebook:

1. Preserve the author's current structure unless it conflicts with this
   document.
2. Avoid unrelated cleanup.
3. Keep changes focused on the requested behavior.
4. Do not silently remove user edits.
5. If the notebook already follows the explained style, extend it in that style
   rather than introducing a new abstraction.
6. Validate the notebook JSON after editing.

When the user says an explained notebook is the desired final style, treat that
notebook as the closest local reference for spacing, block order, naming, and
level of explicitness. Do not add new local helper functions or local package
dependencies unless the user explicitly asks for them.

When the user asks for a compact notebook, preserve behavior but reduce the
number of notebook lines. Prefer stable helper calls and concise cells while
keeping the main runtime cell rerunnable.

#### What To Avoid

Avoid:

- Starting with helper functions before the experiment is visible.
- Hiding important setup in a local notebook package.
- Splitting viewer initialization and render loop across many cells.
- Requiring a kernel restart after the viewer is closed.
- Leaving duplicated old cells after rewriting a notebook.
- Keeping stale output, execution counts, or empty code cells.
- Adding long explanatory text inside code cells.
- Over-abstracting gains, frictions, targets, and marker setup.
- Mixing imports, environment loading, setup, and runtime loop in one early
  cell.
- Diagnosing environment or import problems from the wrong Python environment.
- Promoting project-specific helpers into shared package code before removing
  project-specific constants and assumptions.
- Letting README descriptions drift away from actual names or interaction
  behavior.
- Requiring hardware, credentials, model downloads, or external services in
  ordinary validation notebooks.

The preferred notebook should feel like a clean lab script: everything needed
is visible, the order is natural, and the final cell can be rerun whenever the
interactive session needs to restart.
