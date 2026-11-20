"""Edit the visual RG2 mount pose relative to the UR flange."""

from PyQt5.QtCore import QSettings, Qt
from PyQt5.QtWidgets import QDoubleSpinBox, QGridLayout, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from ..rg2_visual import DEFAULT_VISUAL_MOUNT, normalize_visual_mount
from .widgets import disclosure_button, make_label


class VisualMountEditor:
    def __init__(self, layout, *, simulation, robot_ip):
        self.settings = QSettings("VisionAI", "UR5e Teleoperation")
        self.group = f"rg2/robots/{robot_ip}" if robot_ip else None
        if simulation:
            self.group = "rg2/simulation"
        self.applied = self._load_mount()
        self.toggle = disclosure_button("RG2 Visual Mount")
        self.toggle.setToolTip("Adjust the rendered gripper relative to the tool flange in MuJoCo.")
        self.body = QWidget()
        self.body.hide()
        self.toggle.toggled.connect(self.body.setVisible)
        layout.addWidget(self.toggle, 0, Qt.AlignLeft)
        layout.addWidget(self.body)

        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(5)
        self.fields = []
        body.addLayout(self._build_fields())
        body.addLayout(self._build_actions())
        self._update_actions()

    def _build_fields(self):
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        for i, axis in enumerate(("X", "Y", "Z", "Roll", "Pitch", "Yaw")):
            translation = i < 3
            row = 0 if translation else 2
            col = i % 3
            label = make_label(f"{axis} ({'mm' if translation else '°'})", role="muted")
            field = QDoubleSpinBox()
            limit = 1000. if translation else 180.
            field.setRange(-limit, limit)
            field.setDecimals(1)
            field.setSingleStep(1.)
            field.setKeyboardTracking(False)
            field.setValue(self.applied[i])
            field.setAccessibleName(f"RG2 visual mount {label.text()}")
            direction = "XYZ"[col]
            tooltip = (f"Visual mount translation along flange {direction}, in millimetres." if translation else
                       f"Visual mount {axis.lower()} about fixed flange {direction}, in degrees (right-hand rule).")
            field.setToolTip(tooltip)
            label.setBuddy(field)
            grid.addWidget(label, row, col)
            grid.addWidget(field, row + 1, col)
            grid.setColumnStretch(col, 1)
            self.fields.append(field)
            field.valueChanged.connect(self._update_actions)
        return grid

    def _build_actions(self):
        actions = QHBoxLayout()
        actions.addStretch()
        self.reset = QPushButton("Reset")
        self.reset.setToolTip("Restore the visual gripper to the model's default flange mount.")
        self.reset.clicked.connect(self._reset)
        actions.addWidget(self.reset)
        self.apply = QPushButton("Apply")
        self.apply.setToolTip("Apply this mount to the MuJoCo view and remember it for this robot.")
        self.apply.clicked.connect(self._apply)
        actions.addWidget(self.apply)
        return actions

    def _load_mount(self):
        if not self.group:
            return DEFAULT_VISUAL_MOUNT
        key = f"{self.group}/pose"
        try:
            if self.settings.contains(key):
                values = self.settings.value(key)
                if isinstance(values, (tuple, list)) and len(values) == 6:
                    return normalize_visual_mount(values)
            else:
                yaw = float(self.settings.value(f"{self.group}/jaw_yaw_deg", 0.))
                if -180. <= yaw <= 180.:
                    return (*DEFAULT_VISUAL_MOUNT[:5], yaw)
        except (TypeError, ValueError, OverflowError):
            pass
        return DEFAULT_VISUAL_MOUNT

    def _values(self):
        return tuple(field.value() for field in self.fields)

    def _update_actions(self):
        current = self._values()
        self.apply.setEnabled(current != self.applied)
        self.reset.setEnabled(current != DEFAULT_VISUAL_MOUNT or self.applied != DEFAULT_VISUAL_MOUNT)

    def _apply(self):
        self.applied = self._values()
        if self.group:
            self.settings.setValue(f"{self.group}/pose", list(self.applied))
            self.settings.sync()
        self._update_actions()

    def _reset(self):
        for field, value in zip(self.fields, DEFAULT_VISUAL_MOUNT):
            field.setValue(value)
        self._apply()
