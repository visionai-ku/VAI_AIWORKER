"""Edit, validate and acknowledge the two sets of per-axis impedance gains."""

from PyQt5.QtCore import QSignalBlocker, QSize, Qt
from PyQt5.QtWidgets import (
    QApplication, QDoubleSpinBox, QGridLayout, QHBoxLayout, QPushButton,
    QToolButton, QVBoxLayout, QWidget,
)

from ..impedance_gains import default_impedance_gains, max_impedance_gains, validate_impedance_gains
from .widgets import make_label


class ImpedanceEditor:
    """Keep TCP and joint gain edits local until the operator applies a complete set."""

    linear_units = ("N/m", "N·s/m")
    angular_units = ("N·m/rad", "N·m·s/rad")

    def __init__(self, layout, submit):
        self.space = None
        self.confirmed = {space: default_impedance_gains(space) for space in ("eef", "joint")}
        self.drafts = {}

        self.toggle = self._make_disclosure()
        layout.addWidget(self.toggle, 0, Qt.AlignLeft)
        self.body = QWidget()
        self.body.hide()
        layout.addWidget(self.body)
        body = QVBoxLayout(self.body)
        body.setContentsMargins(0, 0, 0, 5)
        body.setSpacing(5)
        body.addLayout(self._build_gain_table())
        body.addLayout(self._build_actions(submit))
        self.toggle.toggled.connect(self._expand)

    @staticmethod
    def _make_disclosure():
        toggle = QToolButton()
        toggle.setObjectName("gainDisclosure")
        toggle.setFont(QApplication.font("QLabel"))
        arrow_size = toggle.fontMetrics().height() // 2
        toggle.setIconSize(QSize(arrow_size, arrow_size))
        toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.RightArrow)
        toggle.setCheckable(True)
        return toggle

    def _build_gain_table(self):
        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(3)
        for col in (1, 2):
            grid.setColumnStretch(col, 1)
        self.headers = []
        for row in (0, 4):
            headings = [make_label("", role="column") for _ in range(2)]
            for col, label in enumerate(headings, 1):
                grid.addWidget(label, row, col)
            self.headers.append(headings)
        self.axes = []
        self.inputs = {"stiffness": [], "damping": []}
        for row in (1, 2, 3, 5, 6, 7):
            axis = make_label("")
            self.axes.append(axis)
            grid.addWidget(axis, row, 0)
            for col, fields in enumerate(self.inputs.values(), 1):
                box = QDoubleSpinBox()
                box.setDecimals(2)
                box.setAlignment(Qt.AlignRight)
                box.valueChanged.connect(self._mark_edits)
                fields.append(box)
                grid.addWidget(box, row, col)
        return grid

    def _build_actions(self, submit):
        actions = QHBoxLayout()
        self.status = make_label("", role="muted", wrap=True)
        actions.addWidget(self.status, 1)
        defaults = QPushButton("기본값 불러오기")
        defaults.clicked.connect(lambda: self._set_values(default_impedance_gains(self.space)))
        actions.addWidget(defaults)
        self.apply = QPushButton("계수 적용")
        self.apply.clicked.connect(lambda: submit(self.space, self.values()))
        actions.addWidget(self.apply)
        return actions

    def _expand(self, expanded):
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.body.setVisible(expanded)

    def values(self):
        return {key: [field.value() for field in fields] for key, fields in self.inputs.items()}

    def configure(self, space, visible):
        self.toggle.setVisible(visible)
        if not visible:
            self.toggle.setChecked(False)
        if self.space == space:
            return
        if self.space is not None:
            self.drafts[self.space] = self.values()
        self.space = space
        joint = space == "joint"
        self.toggle.setText(f"{'관절' if joint else 'TCP'} 임피던스 · 강성·감쇠 설정")
        for col, name in enumerate(("강성", "감쇠")):
            top_units = self.angular_units if joint else self.linear_units
            self.headers[0][col].setText(f"{name}\n{top_units[col]}")
            self.headers[1][col].setText(f"{name}\n{self.angular_units[col]}")
            self.headers[1][col].setVisible(not joint)
        self._set_values(self.drafts.get(space, self.confirmed[space]))

    def _set_values(self, values):
        limits = max_impedance_gains(self.space)
        joint = self.space == "joint"
        for i in range(6):
            axis = f"J{i + 1}" if joint else ("X", "Y", "Z", "Rx", "Ry", "Rz")[i]
            self.axes[i].setText(axis)
            for col, (key, fields) in enumerate(self.inputs.items()):
                box = fields[i]
                name = ("강성", "감쇠")[col]
                unit = (self.angular_units if joint or i >= 3 else self.linear_units)[col]
                box.setAccessibleName(f"{axis} {name} ({unit})")
                box.setToolTip(f"{axis} {name} · 현재 적용값 {self.confirmed[self.space][key][i]:g} {unit}\n"
                               f"입력 범위 0–{limits[key][i]:g} {unit}")
                with QSignalBlocker(box):
                    box.setRange(0, limits[key][i])
                    box.setSingleStep(1 if limits[key][i] >= 20 else 0.1)
                    box.setValue(values[key][i])
        self._mark_edits()

    def _mark_edits(self):
        dirty = self.values() != self.confirmed[self.space]
        self.status.setText("수정 중 · 미적용" if dirty else "적용된 값")
        self.apply.setEnabled(dirty)

    def result(self, space, gains, accepted):
        if space not in self.confirmed or gains is None:
            return
        self.confirmed[space] = validate_impedance_gains(space, gains)
        if accepted:
            self.drafts.pop(space, None)
        if space == self.space:
            self._set_values(self.confirmed[space] if accepted else self.values())
