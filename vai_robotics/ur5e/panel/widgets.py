"""Shared Qt widgets and layout conventions for the operator panel."""

import re

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QComboBox, QFrame, QLabel, QSizePolicy, QVBoxLayout


class ModePicker(QComboBox):
    def wheelEvent(self, event):
        # Scrolling the panel should never request a controller change.
        event.ignore()


def make_label(text, *, role=None, wrap=False, emphasis=0):
    label = QLabel(text)
    label.setTextFormat(Qt.PlainText)
    if role:
        label.setProperty("role", role)
    if wrap:
        label.setWordWrap(True)
        label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
    if emphasis:
        font = label.font()
        font.setBold(True)
        font.setPointSizeF(font.pointSizeF() + emphasis)
        label.setFont(font)
    return label


def set_status_text(label, text):
    # QLabel only wraps at word boundaries. Driver names and addresses can be
    # longer than the whole status column on a small screen.
    def break_token(match):
        token = match.group()
        return "\u200b".join(token[i:i + 12] for i in range(0, len(token), 12))

    display = re.sub(r"\S{20,}", break_token, text)
    label.setText(display)
    label.setToolTip(text if display != text else "")


def add_section(parent_layout):
    frame = QFrame()
    frame.setObjectName("section")
    layout = QVBoxLayout(frame)
    layout.setContentsMargins(11, 11, 11, 12)
    layout.setSpacing(6)
    parent_layout.addWidget(frame)
    return layout


def add_divider(layout):
    rule = QFrame()
    rule.setObjectName("rule")
    rule.setFixedHeight(1)
    layout.addWidget(rule)
