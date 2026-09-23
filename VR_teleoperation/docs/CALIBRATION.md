# 개인 의도 축 보정

[보정 노트북](../calibration_notebook/calibrate_intent_axes.ipynb)은 Quest 양손 컨트롤러의 왕복 궤적으로 개인별 위치 보정 행렬을 만든다. 로봇 연결, bringup/inbound/outbound 실행, grip 입력은 필요 없다. 기존 SIM·REAL·기록용 REAL 노트북의 기본 설정은 자동으로 바뀌지 않는다.

## 준비

1. [README의 환경 설치](../README.md#실행)를 완료한다. 기존 `requirements-notebook.txt`의 Python 패키지를 사용하며 Quest HTTPS 인증서 생성에 `openssl`이 필요하다.
2. PC와 Quest를 같은 LAN에 연결한다. 다른 SIM·REAL·수집 창의 Quest 수신기를 STOP하여 기본 포트 `8443`을 비운다.
3. 보정 노트북을 새 커널에서 위부터 실행한다. `PC_HOST`가 Quest에서 접근할 수 있는 PC LAN 주소인지 확인한다. 기본값은 `QUEST_PORT=8443`, `PREPARE_S=3.`, `CAPTURE_S=6.`, `MAX_WRIST_ROTATION_DEG=30.`이다.
4. 새 보정은 `RESUME_CAPTURE_PATH=None`으로 시작한다. 실행 셀로 수집 창을 연 뒤 Quest 브라우저에서 출력된 `https://<PC_HOST>:8443/`에 접속해 VR 세션을 시작한다. 양손 추적과 마지막 패킷 수신 상태를 확인하고 **수집 시작 가능** 표시를 기다린다.

## 수집과 저장

Quest 방향·본인 위치·몸통을 유지하고 팔꿈치를 편하게 굽힌 작업 자세에서 시작한다. 손목 방향을 유지한 채 두 손을 함께 이동한다. 좌우 동작에서는 몸통을 회전하거나 양팔을 벌리지 않는다. 창에 표시되는 좌표 설명을 따르며 왼쪽은 +Y, 오른쪽은 -Y다.

| 동작 | 수행 | 사용 |
|---|---|---|
| 전진 | 높이를 유지하려는 의도로 앞으로 이동 후 복귀 | 1~4회 학습, 5회 검증 |
| 왼쪽 | 높이·앞뒤 거리를 유지하며 두 손을 +Y로 평행 이동 후 복귀 | 1~4회 학습, 5회 검증 |
| 상승 | 앞뒤·좌우 위치를 유지하려는 의도로 위로 이동 후 복귀 | 1~4회 학습, 5회 검증 |
| 오른쪽 | 두 손을 -Y로 평행 이동 후 복귀 | 1~4회 학습, 5회 검증 |
| 왼쪽＋상승 | +Y/+Z로 대각선 이동 후 복귀 | 5회 모두 검증 |
| 오른쪽＋상승 | -Y/+Z로 대각선 이동 후 복귀 | 5회 모두 검증 |

동작을 선택하고 **선택 동작 5회 수집**을 누른다. 매회 3초 준비 후 6초 동안 기록한다. 기록 시작과 끝에 각각 0.3초 이상 정지하고, 중간에 편한 범위로 이동했다가 시작 위치로 돌아온다. 10~20cm 정도부터 시작하며 복합 동작의 두 축 이동량을 같게 맞출 필요는 없다. 손목 회전은 시작 방향에서 30° 이내로 유지한다.

거절된 회차는 완료 횟수에 포함되지 않는다. 원인을 확인하고 같은 시작 버튼을 누르면 완료한 반복은 유지한 채 남은 회차를 수집한다. **선택 동작만 다시 수집**은 해당 동작의 완료 기록을 제외하고 다시 수집한다. Quest 세션·origin·헤드셋 기준이 바뀌어 중단되면 **전체 기준/수집 초기화** 후 전체 동작을 다시 수집한다.

6종 × 5회가 끝나면 **검증 결과 미리보기**로 확인한다. 제어창의 **Calibration 이름**에 이름을 입력하고 **저장·보정 계산 후 종료**를 누른다. 이름은 수집 중에도 입력·수정할 수 있으며 저장 이름을 미리 확인할 수 있다. 노트북 결과 셀에서 `reason`, `failed_checks`, 각 손의 `checks`와 저장 경로를 확인한다. 파일 이름은 저장 날짜(PC 현지 시각)의 `YYMMDD_이름` 형식이다. 예를 들어 2026년 9월 23일 `jaehoon`을 입력하면 보정 결과는 `260923_jaehoon.json`으로 저장된다. 이름은 40자 이내의 영문(A–Z, a–z)·숫자·공백·`_`·`-`만 사용할 수 있다. 빈 이름이나 같은 날짜·이름의 기존 파일이 있으면 저장 버튼이 비활성화되므로 다른 이름을 입력한다.

| 내용 | 저장 위치 |
|---|---|
| 전체 수집 원본 | `calibration/diagnostics/<YYMMDD_이름>_trajectory.json` |
| 반복별 수집 기록 | 같은 이름의 `.jsonl` |
| 자동 검사 통과 프로필 | `calibration/profiles/<YYMMDD_이름>.json` |
| 미완료 또는 검사 실패 결과 | `calibration/diagnostics/<YYMMDD_이름>.json` |

기존 파일을 덮어쓰지 않으며 새 개인 데이터·프로필은 Git에서 제외된다. 창을 X로 닫아도 유효한 새 이름이 있으면 같은 형식으로 저장한다. 이름이 비었거나 사용할 수 없으면 실행 시각 기반의 고유 이름으로 자동 저장하여 수집 내용을 보존한다. 원본 capture와 profile을 함께 보관한다. 이어 수집하려면 `RESUME_CAPTURE_PATH`에 원본 capture JSON을 지정할 수 있지만, 같은 Quest 세션·origin·헤드셋 기준과 본인 위치가 유지되어야 한다. 새 Quest 세션에서는 `None`으로 새로 수집한다.

## 결과 해석과 SIM 적용

`accepted=true`는 정해진 궤적 검사를 통과했다는 뜻이다. 실제 사용감, 작업 성공, 로봇 안전성을 보장하지 않는다. 반대로 `accepted=false`는 자동 검사가 통과하지 않았다는 뜻이며 사용감을 직접 측정한 결과는 아니다. 미완료·추적 오류·축 추정 실패와, 전체 수집 후 궤적 잔차 기준 실패를 구분해 `reason`과 반복별 결과를 확인한다.

통과 프로필은 먼저 [SIM 노트북](../sim_notebook/sim_vr_teleop.ipynb)의 `IntentTeleopSettings(...)`에서 다음 항목을 바꾼다. `<YYMMDD_이름>`은 실제 저장 이름으로 대체한다. 같은 사용자·Quest 방향·작업 위치를 확인한 후 `CONFIRM_CALIBRATION_SETUP=True`로 둔다.

```python
PROFILE_PATH=PROJECT_DIR / "calibration/profiles/<YYMMDD_이름>.json",
ALLOW_FAILED_PROFILE_IN_SIM=False,
SIM_CANDIDATE_CAPTURE_PATH=None,
CONFIRM_CALIBRATION_SETUP=True,
```

수집 창을 종료해 Quest 포트를 비우고 SIM의 설정 셀부터 다시 실행한다. 표시된 profile ID를 확인한 뒤 작은 전진·좌우·상승·대각선 움직임으로 사용감을 확인한다. 실행 중인 런타임에 파일 경로만 바꾸면 반영되지 않으므로 STOP 후 설정부터 다시 실행한다.

## REAL·기록용 REAL 적용

SIM에서 확인한 통과 프로필을 [REAL 노트북](../real_notebook/real_vr_teleop.ipynb) 또는 [기록용 REAL 노트북](../real_notebook/real_vr_teleop_record.ipynb)의 설정에 지정한다. 기존 v1 후보용 capture/review 경로를 아래처럼 비워 일반 통과 프로필 로더를 사용한다.

```python
PROFILE_PATH=PROJECT_DIR / "calibration/profiles/<YYMMDD_이름>.json",
REAL_CANDIDATE_CAPTURE_PATH=None,
REAL_CANDIDATE_REVIEW_PATH=None,
CONFIRM_CALIBRATION_SETUP=True,
DRY_RUN=True,
```

REAL은 DRY RUN에서도 실제 로봇 피드백이 필요하다. [worker 준비](WORKER_SETUP.md)를 따른 뒤 해당 노트북의 연결·DRY RUN 확인 절차를 진행한다.

## 자동 검사 실패 후보의 선택적 SIM 확인

전체 수집과 축 추정이 완료되어 `reason="trajectory_validation_failed"`인 후보는 기존 SIM 후보 로더로 시험할 수 있다. 원본 capture와 후보 profile이 모두 필요하며 사용자·Quest 방향·작업 위치를 명시적으로 확인한다. 미완료 또는 축 추정 실패 결과에는 이 경로를 사용할 수 없다.

```python
PROFILE_PATH=PROJECT_DIR / "calibration/diagnostics/<YYMMDD_이름>.json",
ALLOW_FAILED_PROFILE_IN_SIM=True,
SIM_CANDIDATE_CAPTURE_PATH=PROJECT_DIR / "calibration/diagnostics/<YYMMDD_이름>_trajectory.json",
CONFIRM_CALIBRATION_SETUP=True,
```

SIM 확인 후에도 원본 `accepted=false`는 유지된다. 이 후보를 REAL에서 시험하려면 해당 profile/capture 해시에 연결된 **새 SIM 확인 기록**이 필요하다. 배포된 `v1-sim-reviewed-real-trial.json`은 기존 v1 후보 전용이므로 새 후보에 재사용할 수 없다. 보정 노트북은 REAL 확인 기록을 생성하지 않는다.
