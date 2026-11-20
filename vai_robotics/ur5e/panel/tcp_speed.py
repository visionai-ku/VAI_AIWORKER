"""Read-only telemetry for a robot-connected MuJoCo preview."""

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QGridLayout, QHBoxLayout

from .widgets import make_label


class TCPSpeedComparison:
    def __init__(self, layout):
        self.section = layout.parentWidget()
        self.section.hide()
        heading = QHBoxLayout()
        heading.addWidget(make_label("TCP Speed", emphasis=2))
        heading.addStretch()
        self.status = make_label("", role="muted", wrap=True)
        heading.addWidget(self.status)
        layout.addLayout(heading)

        table = QGridLayout()
        table.setColumnStretch(0, 1)
        for col, unit in enumerate(("Linear (mm/s)", "Angular (°/s)"), start=1):
            label = make_label(unit, role="column")
            table.addWidget(label, 0, col, Qt.AlignRight)
        self.values = {}
        sources = (("REAL RUN", "real", "RTDE actual_TCP_speed"),
                   ("DRY RUN", "dry", "MuJoCo shadow using the same input"))
        for row, (source, key, description) in enumerate(sources, start=1):
            source_label = make_label(source)
            source_label.setToolTip(description)
            table.addWidget(source_label, row, 0)
            for col, unit in enumerate(("mm_s", "deg_s"), start=1):
                label = make_label("—")
                label.setAccessibleName(f"{source} TCP speed ({'mm/s' if col == 1 else '°/s'})")
                table.addWidget(label, row, col, Qt.AlignRight)
                self.values[f"{key}_{unit}"] = label
        layout.addLayout(table)
        self.payload = make_label("", role="muted", wrap=True)
        self.payload.setToolTip("Active payload configured on the robot. CoG is relative to the tool flange.")
        layout.addWidget(self.payload)

    def refresh(self, comparison, payload, *, dry_run):
        if comparison is None and payload is None and self.section.isHidden():
            return
        self.section.show()
        comparison = comparison or {}
        phase = comparison.get("phase")
        if phase == "live" and dry_run:
            phase = "waiting"  # A viewer reply from the preceding REAL RUN may still be in flight.
        captions = {
            "waiting": "Compare during REAL RUN" if dry_run else "Waiting for Clutch",
            "live": "Live",
            "last": "Last Clutch · Mean",
        }
        self.status.setText(captions.get(phase, "Comparison unavailable"))
        for key, label in self.values.items():
            value = comparison.get(key) if phase in ("live", "last") else None
            label.setText("—" if value is None else f"{value:.1f}")
        if payload is None:
            self.payload.setText("RTDE Payload unavailable")
        else:
            cog_mm = ", ".join(f"{metres * 1000:g}" for metres in payload["cog_m"])
            self.payload.setText(f"RTDE Payload {payload['mass_kg']:g} kg · Flange CoG [{cog_mm}] mm")
