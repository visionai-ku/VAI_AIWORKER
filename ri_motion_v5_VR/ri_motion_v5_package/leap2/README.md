# `ri_motion_v5_package.leap2`

Leap Motion integration helpers for the workspace.

## Modules

- `leap2_handler.py`
  - Runtime helpers for Leap Motion hand tracking data
- `leapmotion_mac/`
  - Vendored or local Ultraleap Gemini LeapC Python binding material for macOS

## Public API Snapshot

- `leap2_handler.py`: `LeapMotion2Handler`, `NullLeapConnection`
- `LeapMotion2Handler`: availability/status checks, connection setup, tracking mode/scale/offset configuration, hand parsing, bone/point accessors, flexion-ratio helpers, and MuJoCo bone plotting

## Import

```python
from ri_motion_v5_package.leap2 import *
```

## Notes

- Notebook examples live in `../../ri_motion_v5_notebook/40_leap_motion/`
- Hardware and SDK setup may require local machine configuration outside the Python package
- `LeapMotion2Handler(required=False)` is the default behavior: missing SDK bindings or connection setup failures are reported through `is_available()` and `get_status_msg()` instead of stopping notebook execution
- `is_device_detected()` remains false until a Leap device event is received, so notebooks can distinguish an SDK connection from actual hardware presence
- Use `LeapMotion2Handler(required=True)` when a notebook or script must fail fast without Leap Motion support
- Keep machine-specific SDK paths out of reusable package code when possible
