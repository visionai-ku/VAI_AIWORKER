"""Trajectory collection UI. Uses the Quest receiver without a robot connection."""
from datetime import datetime
from pathlib import Path
import time
import json
import math
import shutil
from PyQt5.QtCore import QRegularExpression
from PyQt5.QtGui import QRegularExpressionValidator
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QComboBox, QLineEdit
from ri_motion_v5_package.qt import use_pyqt_platform_plugins
from .intent_calibration import IntentTrajectoryCapture, LABELS, LABEL_TEXT, fit_intent_profile, save_json
from .quest_controller_server import QuestControllerServer, create_self_signed_certificate


def controller_status_text(frame, now, timeout_s, input_reason):
    """Display per-hand tracking signals; collection eligibility stays with the collector."""
    frame = frame if isinstance(frame, dict) else {}
    received = frame.get('received_at')
    age = (max(0., now - received) if type(received) in (int, float)
           and math.isfinite(received) else None)
    stale = age is not None and (age > timeout_s or input_reason == 'stale_pose')
    hands = frame.get('hands') or {}
    lines = []
    for hand, label in (('left', '왼손'), ('right', '오른손')):
        data = hands.get(hand) if isinstance(hands, dict) else None
        if not frame or age is None:
            state = '입력 대기'
        elif stale:
            state = '입력 만료 · 새 데이터 대기'
        elif frame.get('visibility') != 'visible':
            state = 'Quest 세션 비활성'
        elif not isinstance(data, dict) or data.get('valid') is not True:
            state = '추적 안 됨'
        elif data.get('emulated') is not False:
            state = '추정 위치 · 수집 불가'
        else:
            state = '추적 신호 정상'
        buttons = ''
        if isinstance(data, dict) and state == '추적 신호 정상':
            values = [data.get(key) for key in ('grip', 'trigger')]
            if all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values):
                buttons = ' · 그립 %.0f%% / 트리거 %.0f%%' % tuple(100*v for v in values)
        lines.append(label + ': ' + state + buttons)
    lines.append('마지막 패킷 수신: ' + ('없음' if age is None else '%.0fms 전' % (age*1000)))
    return '\n'.join(lines)


def fit_summary_text(profile):
    lines = ['보정 생성 완료' if profile['fit_success'] else '보정 미완료: ' + profile['reason']]
    for hand, title in (('left', '왼손'), ('right', '오른손')):
        data = profile.get('hands', {}).get(hand)
        if data:
            lines.append('%s · 행렬 조건수 %.3f' % (title, data['condition']))
    return '\n'.join(lines)


def calibration_save_paths(project_dir, name, now=None):
    """Build one portable, user-named set without changing calibration contents."""
    name = name.strip(' ')
    if not name:
        raise ValueError('Calibration 이름을 입력하세요.')
    if not name.isascii() or len(name) > 40 or any(not (c.isalnum() or c in ' _-') for c in name):
        raise ValueError('이름은 40자 이내의 영문(A–Z, a–z)·숫자·공백·_·-만 사용하세요.')
    stem = (now or datetime.now()).strftime('%y%m%d') + '_' + name
    root = Path(project_dir) / 'calibration'
    return {
        'capture_path': root / 'diagnostics' / (stem + '_trajectory.json'),
        'journal_path': root / 'diagnostics' / (stem + '_trajectory.jsonl'),
        'profile_path': root / 'profiles' / (stem + '.json'),
        'incomplete_profile_path': root / 'diagnostics' / (stem + '.json'),
    }


def _check_save_paths(paths):
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('같은 날짜·이름의 calibration이 있습니다. 다른 이름을 입력하세요.')


def run_intent_capture(project_dir, host, port=8443, prepare_s=3., capture_s=6., max_wrist_rotation_deg=30., resume_path=None):
    """Five repeats of the selected label, 3s preparation before every repeat.

    Keep wrists facing the same way; rest at the beginning/end for >=0.3s.
    Invalid tracking aborts only the current attempt; a reference change clears
    the training set through the existing collector and needs an explicit reset.
    """
    if not host: raise ValueError('Set the PC LAN address')
    if prepare_s < 1 or not 3 <= capture_s <= 10: raise ValueError('Invalid capture timing')
    root = Path(project_dir)
    stamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    output = root / 'calibration' / 'diagnostics' / (stamp + '-intent-trajectories.json')
    journal = output.with_suffix('.jsonl')
    collector = IntentTrajectoryCapture(max_wrist_rotation_deg=max_wrist_rotation_deg)
    if resume_path is not None:
        restored = collector.restore_capture(json.loads(Path(resume_path).read_text(encoding='utf-8')))
        print('저장 시연 재검사·복원:', restored['counts'])
        print('같은 Quest 세션·origin·헤드셋 기준인지 새 입력으로 확인한 후 이어 수집합니다.')
    use_pyqt_platform_plugins()
    app = QApplication.instance() or QApplication([])
    window = QWidget()
    window.setWindowTitle('개인 의도 축 수집 · 로봇 연결 없음')
    layout = QVBoxLayout(window)
    combo = QComboBox(window)
    names = {'forward': '전진', 'left': '왼쪽', 'up': '상승', 'right': '오른쪽'}
    for label in LABELS: combo.addItem(names[label] + ' (' + label + ')', label)
    remaining = [k for k,v in collector.status()['counts'].items() if v < 5]
    if remaining: combo.setCurrentIndex(LABELS.index(remaining[0]))
    instructions = QLabel('몸통 고정 · 두 손은 서로 벌리지 않고 함께 같은 방향으로 이동합니다.\n각 시연: 시작 0.3초 정지 → 이동 → 출발 위치로 복귀 → 끝 0.3초 정지.\n손목 방향은 유지하고, 자연스럽게 섞이는 교차축 이동은 그대로 기록합니다.', window)
    instructions.setWordWrap(True)
    instructions.setText(instructions.text() + '\n컨트롤러 회전 허용: 시작 방향에서 최대 %g°' % max_wrist_rotation_deg)
    motion_help = QLabel(LABEL_TEXT[combo.currentData()], window)
    motion_help.setWordWrap(True)
    combo.currentIndexChanged.connect(lambda _: motion_help.setText(LABEL_TEXT[combo.currentData()]))
    controller_status = QLabel('왼손: 입력 대기\n오른손: 입력 대기', window)
    controller_status.setObjectName('controller_input_status')
    controller_status.setWordWrap(True)
    readiness = QLabel('수집 시작 불가 · Quest 입력 대기', window)
    readiness.setObjectName('capture_readiness')
    readiness.setWordWrap(True)
    start = QPushButton('선택 동작 5회 수집 (매회 %g초 준비)' % prepare_s, window)
    cancel = QPushButton('현재 동작 중단', window)
    reset = QPushButton('전체 기준/수집 초기화', window)
    clear_label = QPushButton('선택 동작만 다시 수집', window)
    preview = QPushButton('보정 계산 미리보기', window)
    name_label = QLabel('Calibration 이름', window)
    name_input = QLineEdit(window)
    name_input.setObjectName('calibration_name_input')
    name_input.setMaxLength(40)
    name_input.setValidator(QRegularExpressionValidator(QRegularExpression(r'[A-Za-z0-9 _-]*'), name_input))
    name_input.setPlaceholderText('예: jaehoon_tableA (영문 입력)')
    name_label.setBuddy(name_input)
    save_hint = QLabel(window)
    save_hint.setObjectName('calibration_save_hint')
    save_hint.setWordWrap(True)
    finish = QPushButton('저장·보정 계산 후 종료', window)
    status = QLabel('Quest 연결 대기', window)
    status.setWordWrap(True)
    fit_status = QLabel('', window)
    fit_status.setWordWrap(True)
    for item in (instructions, controller_status, readiness, combo, motion_help, start, cancel, clear_label, reset, preview, name_label, name_input, save_hint, finish, status, fit_status): layout.addWidget(item)

    def named_paths():
        paths = calibration_save_paths(root, name_input.text())
        _check_save_paths(paths)
        return paths

    def update_save_hint():
        try:
            paths = named_paths()
        except (ValueError, FileExistsError) as exc:
            save_hint.setText(str(exc))
            save_hint.setStyleSheet('color: #9a5500;')
            finish.setEnabled(False)
            return None
        save_hint.setText('저장 이름: ' + paths['profile_path'].stem)
        save_hint.setStyleSheet('color: #16703a;')
        finish.setEnabled(True)
        return paths

    name_input.textChanged.connect(update_save_hint)
    update_save_hint()
    save_day = datetime.now().date()
    selected_paths = None
    actions = []
    for button, action in ((start, 'start'), (cancel, 'cancel'), (reset, 'reset'),
                           (clear_label, 'clear_label'), (preview, 'preview'), (finish, 'finish')):
        button.clicked.connect(lambda checked=False, a=action: actions.append(a))
    server = None
    phase, deadline, selected = 'IDLE', 0., None
    try:
        cert, key = create_self_signed_certificate(root / '.cache/fixed_quest/tls', host)
        server = QuestControllerServer(host='0.0.0.0', port=port, certfile=cert, keyfile=key).start()
        print('Quest Browser: https://%s:%d/' % (host, server.port))
        print('제어창에서 Calibration 이름을 입력한 뒤 저장하세요. 형식: YYMMDD_이름')
        window.resize(720, 500)
        window.show()
        while window.isVisible():
            app.processEvents()
            if not window.isVisible():
                break
            if datetime.now().date() != save_day:
                save_day = datetime.now().date()
                update_save_hint()
            now = time.monotonic()
            frame = server.latest()
            current = collector.add_frame(frame, now=now)
            if current['state'] == 'ABORTED': phase = 'IDLE'
            for action in actions:
                if action == 'finish':
                    selected_paths = update_save_hint()
                    if selected_paths is not None:
                        window.close()
                        break
                elif action == 'reset':
                    collector.reset(); phase = 'IDLE'; fit_status.setText('')
                elif action == 'clear_label' and phase == 'IDLE':
                    collector.clear_pose(combo.currentData()); fit_status.setText('')
                elif action == 'preview' and phase == 'IDLE':
                    fitted = fit_intent_profile(collector.snapshot())
                    fit_status.setText(fit_summary_text(fitted))
                elif action == 'cancel':
                    collector.cancel_trial('operator_cancelled'); phase = 'IDLE'
                elif action == 'start' and phase == 'IDLE':
                    selected = combo.currentData()
                    if current['input_valid'] and current['counts'][selected] < 5 and current['state'] != 'ABORTED':
                        phase, deadline = 'PREPARE', now + prepare_s
            actions.clear()
            if not window.isVisible():
                break
            if phase == 'PREPARE' and now >= deadline:
                collector.begin_trial(selected)
                phase, deadline = 'CAPTURE', now + capture_s
            elif phase == 'CAPTURE' and now >= deadline:
                report = collector.finish_trial()
                journal.parent.mkdir(parents=True, exist_ok=True)
                with journal.open('a', encoding='utf-8') as file:
                    file.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + '\n')
                if report['accepted'] and collector.status()['counts'][selected] < 5:
                    phase, deadline = 'PREPARE', now + prepare_s
                else: phase = 'IDLE'
            current = collector.status()
            combo.setEnabled(phase == 'IDLE')
            can_start = (phase == 'IDLE' and current['input_valid'] and current['state'] != 'ABORTED'
                         and current['counts'][combo.currentData()] < 5)
            start.setEnabled(can_start)
            controller_status.setText(controller_status_text(
                frame, now, collector.config['timeout_s'], current['input_reason']))
            if current['state'] == 'ABORTED':
                ready_text = '수집 시작 불가 · 전체 기준/수집 초기화 필요: ' + current['reason']
            elif not current['input_valid']:
                ready_text = '수집 입력 사용 불가: ' + current['input_reason']
            elif phase != 'IDLE':
                ready_text = '준비 중' if phase == 'PREPARE' else '기록 중'
            elif current['counts'][combo.currentData()] >= 5:
                ready_text = '선택 동작 5회 완료 · 다음 동작을 선택하세요'
            else:
                ready_text = '수집 시작 가능 · 선택 동작 5회 수집 버튼을 누르세요 (그립을 쥘 필요 없음)'
            readiness.setText(ready_text)
            readiness.setStyleSheet('font-weight: bold; color: %s;' % ('#16703a' if can_start else '#9a5500'))
            reset.setEnabled(phase == 'IDLE')
            clear_label.setEnabled(phase == 'IDLE' and current['counts'][combo.currentData()] > 0)
            preview.setEnabled(phase == 'IDLE')
            status.setText('%s · %.1f초\n%s\n%s\n%s' % (
                phase, max(0, deadline - now) if phase != 'IDLE' else 0,
                LABEL_TEXT.get(selected, ''), current['reason'],
                ' / '.join('%s %d/5' % (k, v) for k, v in current['counts'].items())))
            time.sleep(.01)
    finally:
        if collector.state == 'COLLECTING': collector.cancel_trial('window_closed')
        if server is not None: server.close()
        window.close()
        snapshot = collector.snapshot()
        # Keep a unique recovery copy until the named capture/profile are saved.
        save_json(output, snapshot)
        print('수집 임시 저장:', output)
    profile = fit_intent_profile(snapshot)
    if selected_paths is None:
        try:
            selected_paths = named_paths()
        except (ValueError, FileExistsError):
            print('유효한 새 이름 없이 창을 닫아 실행 시각으로 자동 저장합니다.')
    if selected_paths is not None:
        _check_save_paths(selected_paths)
        destination = selected_paths['profile_path' if profile['fit_success'] else 'incomplete_profile_path']
        save_json(selected_paths['capture_path'], snapshot)
        save_json(destination, profile)
        if journal.exists():
            with journal.open('r', encoding='utf-8') as source, selected_paths['journal_path'].open('x', encoding='utf-8') as target:
                shutil.copyfileobj(source, target)
            journal.unlink()
        output.unlink()
        output = selected_paths['capture_path']
    else:
        destination = root / 'calibration' / ('profiles' if profile['fit_success'] else 'diagnostics') / (stamp + '-intent-profile.json')
        save_json(destination, profile)
    print('수집 저장:', output)
    print('보정 결과:', profile['reason'], '|', destination)
    print(fit_summary_text(profile))
    if profile['fit_success']:
        print('MuJoCo 커맨드 센터의 작업자 보정 목록에서 파일명을 선택하세요.')
    else:
        print('보정 행렬 미완료: 수집 횟수와 계산 결과를 확인하세요. docs/CALIBRATION.md')
    return {'capture_path': output, 'profile_path': destination, 'profile': profile}
