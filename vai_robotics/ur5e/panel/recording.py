"""Recording controls for one demonstration episode at a time."""

from PyQt5.QtWidgets import QHBoxLayout, QPushButton

from .widgets import make_label, set_status_text


class EpisodeControls:
    def __init__(self, layout):
        self.section = layout.parentWidget()
        self.section.hide()
        self._event = None
        self._pending = False

        heading = QHBoxLayout()
        heading.addWidget(make_label("Recording", emphasis=2))
        self.status = make_label("", role="muted", wrap=True)
        heading.addWidget(self.status, 1)
        layout.addLayout(heading)

        row = QHBoxLayout()
        self.buttons = {}
        for command, caption in (("start", "Record"), ("save", "Save Episode"), ("discard", "Discard")):
            button = QPushButton(caption)
            button.clicked.connect(lambda checked, value=command: self._request(value))
            self.buttons[command] = button
            row.addWidget(button)
        layout.addLayout(row)

        self.destination = make_label("", role="muted", wrap=True)
        layout.addWidget(self.destination)
        self.message = make_label("", wrap=True)
        layout.addWidget(self.message)

    def _request(self, command):
        # Wait for the control loop to acknowledge this click, including refusals.
        self._event = command
        self._pending = True
        for button in self.buttons.values():
            button.setEnabled(False)

    def take_request(self):
        event, self._event = self._event, None
        return event

    def refresh(self, recording, *, acknowledged=False):
        if acknowledged:
            self._pending = False
        self.section.show()
        state = recording["state"]
        frames = recording["frames"]
        status = f"{state.title()} · {frames} frames"
        if recording.get("episodes_known", True):
            status += f" · {recording['episodes']} saved"
        self.status.setText(status)
        set_status_text(self.destination, f"Dataset   {recording['root']}\nTask   {recording['task']}")
        message = recording.get("message", "")
        set_status_text(self.message, message)
        self.message.setVisible(bool(message))

        can_start = state in ("idle", "error") and recording.get("can_start", True)
        self.buttons["start"].setEnabled(can_start and not self._pending)
        self.buttons["save"].setEnabled(state == "recording" and frames > 0 and not self._pending)
        self.buttons["discard"].setEnabled(state in ("preparing", "recording") and not self._pending)
