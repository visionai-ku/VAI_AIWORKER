# `ri_motion_v5_package.qt`

Qt widgets and environment-aware controls for interactive notebooks.

## Modules

- `qt_widgets.py`
  - Low-level multi-slider, radio-button, and Qt widget wrappers
- `env_widgets.py`
  - Environment-aware controls such as pose, delta pose, position, joint, control, and monitor widgets

## Public API Snapshot

- `qt_widgets.py`: `MultiSliderQtWidget`, `SingleSliderQtWidget`, `MultiRadioQtWidget`, `LivePlotWidget`, color/screen helpers, and `pick_pos_neg_points_qt(...)`
- `env_widgets.py`: `EnvPoseSlider`, `EnvJointSlider`, `EnvDeltaPoseSlider`, `EnvDeltaPositionSlider`, `JointPositionMonitorQtWidget`, `EnvCtrlSlider`

## Import

```python
from ri_motion_v5_package.qt import *
```

## Common Widgets

- `EnvPoseSlider`
- `EnvDeltaPoseSlider`
- `EnvDeltaPositionSlider`
- `EnvJointSlider`
- `EnvCtrlSlider`
- `JointPositionMonitorQtWidget`

## Notes

- Requires the notebook-side Qt bootstrap from `init_scripts`
- Prefer left-side widget placement in interactive notebooks
- `EnvJointSlider` colors labels and slider bars by the primary MuJoCo joint axis by default: X red, Y green, Z blue. Use `color_by_axis=False` or pass `label_colors` / `slider_colors` to override this.
- Close widgets in `finally` blocks together with viewer cleanup
