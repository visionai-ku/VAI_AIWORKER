"""Calibration file selection in the MuJoCo command-center control window."""
import json
from pathlib import Path
import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QPushButton, QLabel

from .intent_calibration import MAX_PROFILE_BYTES, load_intent_profile, _digest


def profile_options(project_dir):
    """Find matrix profiles; raw trajectories and review records are not profiles."""
    root = Path(project_dir).resolve()
    options = []
    for directory in (root / 'calibration/profiles', root / 'calibration/diagnostics'):
        for path in sorted(directory.glob('*.json')):
            try:
                if path.stat().st_size > MAX_PROFILE_BYTES:
                    continue
                profile = json.loads(path.read_text(encoding='utf-8'))
            except (OSError, ValueError):
                continue
            if not isinstance(profile, dict) or profile.get('type') != 'quest_intent_profile':
                continue
            # Keep invalid/incomplete profiles visible; selecting one explains
            # the error without silently changing the active calibration.
            label = path.name + ' · ' + str(path.parent.relative_to(root))
            if path == root / 'calibration/diagnostics/v1.json':
                options.insert(0, ('v1 원본 (v1.json)', str(path.resolve())))
            else:
                options.append((label, str(path.resolve())))
    return options


class CommandCenterProfileSelector(QWidget):
    """Queue file clicks; apply only after a released/idle control tick."""

    def __init__(self, project_dir, settings, profile, parent=None):
        super().__init__(parent)
        self.project_dir = Path(project_dir)
        self.settings = settings
        self.profile = profile
        self._path = None if settings.PROFILE_PATH is None else Path(settings.PROFILE_PATH).resolve()
        self._request = None
        self._locked = False
        self.setObjectName('intent_profile_selector')
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel('작업자 보정 · 파일명 선택 시 적용', self))
        row = QHBoxLayout()
        self.combo = QComboBox(self)
        self.combo.setObjectName('profile_file_selector')
        self.combo.setMinimumContentsLength(22)
        self.combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.combo.setToolTip('대기 중 DRY RUN: 추적 없이 변경 가능 · REAL RUN: 양손 추적/해제 필요\n'
                             '조작·복귀·녹화 중에는 변경할 수 없습니다. REAL RUN 전환 후에도 선택을 유지합니다.')
        self.refresh_button = QPushButton('새로고침', self)
        self.refresh_button.setObjectName('profile_refresh')
        row.addWidget(self.combo, 1)
        row.addWidget(self.refresh_button)
        layout.addLayout(row)
        self.status = QLabel(self)
        self.status.setObjectName('profile_selection_status')
        self.status.setTextFormat(Qt.PlainText)
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self._populate()
        self.combo.activated[int].connect(self._selected)
        self.refresh_button.clicked.connect(self.refresh)
        self._show_applied()

    def _populate(self):
        self.combo.clear()
        self.combo.addItem('보정 미적용 (identity)', None)
        options = profile_options(self.project_dir)
        if self._path is not None and str(self._path) not in [value for _, value in options]:
            options.append((self._path.name + ' · 현재 경로', str(self._path)))
        for label, value in options:
            self.combo.addItem(label, value)
            self.combo.setItemData(self.combo.count() - 1, value, Qt.ToolTipRole)
        self._restore_selection()

    def _restore_selection(self):
        self.combo.setCurrentIndex(self.combo.findData(None if self._path is None else str(self._path)))

    def _show_applied(self):
        self.status.setText('적용: 보정 미적용 (identity)' if self.profile is None else
            '적용: %s\nID: %s\n%s' % (self._path.name, self.profile['profile_id'], self._path))

    def _selected(self, index):
        if self._locked:
            self._restore_selection()
            return
        self._request = (self.combo.itemData(index),)
        self.status.setText('선택한 보정 파일 확인 중…')

    def refresh(self):
        if self._locked:
            return
        self._request = None
        self._populate()
        self._show_applied()

    @staticmethod
    def _busy(runtime, on_tick):
        if (runtime._grip_engaged or runtime.armed or runtime._grip_home_active
                or runtime.home_pending or runtime.planner.homing or getattr(on_tick, 'lock_input', False)):
            return True
        record_control = getattr(on_tick, 'record_control', None)
        return record_control is not None and (
            record_control.recorder.recording
            or record_control.widget.get_button_option(label_text='Record') == 'REC')

    def poll(self, runtime, on_tick=None, now=None):
        """Return a log event on a click; never queue an automatic later retry."""
        self._locked = bool(self._busy(runtime, on_tick))
        self.combo.setEnabled(not self._locked)
        self.refresh_button.setEnabled(not self._locked)
        if self._request is None:
            return None
        selected, = self._request
        self._request = None
        try:
            if self._locked:
                raise ValueError('조작·복귀·녹화가 끝난 뒤 다시 선택하세요.')
            profile, matrices = (None, None) if selected is None else load_intent_profile(
                selected, real=self.settings.backend == 'real')
            runtime.set_intent_profile(matrices, None if profile is None else profile['profile_id'],
                                       now=time.monotonic() if now is None else now)
        except (OSError, ValueError, TypeError, KeyError) as exc:
            self._restore_selection()
            self._show_applied()
            self.status.setText(self.status.text() + '\n변경 실패 (%s): %s' % (
                'identity' if selected is None else Path(selected).name, exc))
            return dict(accepted=False, requested_path=selected, reason=str(exc))
        self._path = None if selected is None else Path(selected)
        self.settings.PROFILE_PATH = self._path
        self.profile = profile
        self._show_applied()
        return dict(accepted=True, profile_path=selected, intent_profile=profile,
                    profile_sha256=None if profile is None else _digest(profile),
                    profile_id=None if profile is None else profile['profile_id'])
