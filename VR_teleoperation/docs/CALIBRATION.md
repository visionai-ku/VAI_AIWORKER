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
| 전진 | 높이를 유지하려는 의도로 앞으로 이동 후 복귀 | 5회 모두 행렬 계산 |
| 왼쪽 | 높이·앞뒤 거리를 유지하며 두 손을 +Y로 평행 이동 후 복귀 | 5회 모두 행렬 계산 |
| 상승 | 앞뒤·좌우 위치를 유지하려는 의도로 위로 이동 후 복귀 | 5회 모두 행렬 계산 |
| 오른쪽 | 두 손을 -Y로 평행 이동 후 복귀 | 5회 모두 행렬 계산 |

동작을 선택하고 **선택 동작 5회 수집**을 누른다. 매회 3초 준비 후 6초 동안 기록한다. 기록 시작과 끝에 각각 0.3초 이상 정지하고, 중간에 편한 범위로 이동했다가 시작 위치로 돌아온다. 10~20cm 정도부터 시작한다. 손목 회전은 시작 방향에서 30° 이내로 유지한다.

거절된 회차는 완료 횟수에 포함되지 않는다. 원인을 확인하고 같은 시작 버튼을 누르면 완료한 반복은 유지한 채 남은 회차를 수집한다. **선택 동작만 다시 수집**은 해당 동작의 완료 기록을 제외하고 다시 수집한다. Quest 세션·origin·헤드셋 기준이 바뀌어 중단되면 **전체 기준/수집 초기화** 후 전체 동작을 다시 수집한다.

4종 × 5회가 끝나면 **보정 계산 미리보기**로 확인한다. 20회를 모두 행렬 계산에 사용하며, 대각선 수집과 궤적 점수 검사는 없다. 제어창의 **Calibration 이름**에 이름을 입력하고 **저장·보정 계산 후 종료**를 누른다. 이름은 수집 중에도 입력·수정할 수 있으며 저장 이름을 미리 확인할 수 있다. 노트북 결과 셀에서 `fit_success`, `reason`, 각 손의 조건수·행렬과 저장 경로를 확인한다. 파일 이름은 저장 날짜(PC 현지 시각)의 `YYMMDD_이름` 형식이다. 예를 들어 2026년 9월 23일 `jaehoon`을 입력하면 보정 결과는 `260923_jaehoon.json`으로 저장된다. 이름은 40자 이내의 영문(A–Z, a–z)·숫자·공백·`_`·`-`만 사용할 수 있다. 빈 이름이나 같은 날짜·이름의 기존 파일이 있으면 저장 버튼이 비활성화되므로 다른 이름을 입력한다.

| 내용 | 저장 위치 |
|---|---|
| 전체 수집 원본 | `calibration/diagnostics/<YYMMDD_이름>_trajectory.json` |
| 반복별 수집 기록 | 같은 이름의 `.jsonl` |
| 행렬 계산 완료 프로필 | `calibration/profiles/<YYMMDD_이름>.json` |
| 미완료 또는 행렬 계산 실패 결과 | `calibration/diagnostics/<YYMMDD_이름>.json` |

기존 파일을 덮어쓰지 않는다. `calibration/profiles/`와 `calibration/diagnostics/`의 작업자 보정 프로필 JSON·원본 궤적 JSON·보정 수집 로그 JSONL은 모두 Git에 포함하며, 새 작업자 데이터도 세 종류를 함께 커밋한다. 기존 `v1_trajectory.json`도 재계산·회귀 검사를 위해 보존한다. 창을 X로 닫아도 유효한 새 이름이 있으면 같은 형식으로 저장한다. 이름이 비었거나 사용할 수 없으면 실행 시각 기반의 고유 이름으로 자동 저장하여 수집 내용을 보존한다. 원본 capture와 profile을 함께 보관한다. 이어 수집하려면 `RESUME_CAPTURE_PATH`에 원본 capture JSON을 지정할 수 있지만, 같은 Quest 세션·origin·헤드셋 기준과 본인 위치가 유지되어야 한다. 새 Quest 세션에서는 `None`으로 새로 수집한다.

## 결과 해석

새 프로필은 `version=2`이며 `fit_success=true`가 20회 수집과 행렬 계산 완료를 뜻한다. 5회째도 학습에 사용하고 held-out 검사나 궤적 잔차 점수로 사용을 차단하지 않는다. 수집 단계의 추적·패킷·회전·왕복 구간 품질 확인, 행렬의 finite·크기·역행렬·조건수 확인은 유지한다. 미완료·계산 실패 결과는 `reason`을 확인한다.

## REAL·기록용 REAL 적용

1. 수집 창을 종료해 Quest 포트를 비운다.
2. [REAL 노트북](../real_notebook/real_vr_teleop.ipynb) 또는 [기록용 REAL 노트북](../real_notebook/real_vr_teleop_record.ipynb)을 기존 **초기화 → 설정 → 모델 준비 → 실행** 순서로 실행한다. 노트북에는 선택 위젯을 띄우지 않는다. 시작 보정은 설정의 `PROFILE_PATH`이며 기본값은 기존 v1이다. 시작 파일 검사는 기록 수신 스레드보다 먼저 수행한다.
3. MuJoCo 커맨드 센터의 **작업자 보정** 목록에서 파일명을 클릭한다. `calibration/profiles/`와 `calibration/diagnostics/`의 보정 프로필 JSON을 찾는다. 두 폴더에 `jaehoon.json`, `minsu.json`처럼 사람 이름으로 파일을 두면 표시된다. 파일명만 바꾸어도 내부 행렬·profile ID는 유지된다. 원본 궤적 `*_trajectory.json`과 SIM 확인 기록은 목록에서 제외한다.
4. **보정 미적용 (identity)**, **v1 원본 (v1.json)**, 작업자 파일 중 선택하면 별도 적용 버튼 없이 불러온다. v1 원본은 재계산하지 않는다. REAL RUN에서는 Quest 양손 추적이 유효하고 grip을 모두 놓은 상태에서 선택한다. DRY RUN 대기 상태에서는 추적·grip 확인 없이 선택할 수 있다. 적용된 파일명·경로·profile ID를 확인한다. 새 파일은 **새로고침**으로 목록에 추가한다.
5. 조작·복귀·녹화 중에는 선택·새로고침이 잠긴다. 추적 영역 밖에서 바꾸려면 **DRY RUN → 미리보기 대기 상태에서 프로필/XYZ 배율 적용 → REAL RUN** 순서로 전환한다. 같은 실행 창에서 전환하면 적용값은 유지된다. REAL RUN 재개에는 기존의 유효한 추적·양손 grip 해제 후 재그립 절차가 필요하며, 모드 전환만으로 로봇을 움직이지 않는다. 보정 행렬만 교체하며 다음 grip은 기존 방식으로 현재 EEF에 기준을 잡는다. 잘못된 파일이나 적용 불가 상태에서는 현재 보정을 유지하고 실패한 파일명과 이유를 표시한다. 실패한 요청을 나중에 자동 재시도하지 않는다.

프로필 JSON 하나만 필요하다. SIM 확인 기록, 원본 capture 연결, `CONFIRM_CALIBRATION_SETUP`은 필요하지 않다. 실행 로그에는 시작 프로필과 변경 시점의 경로·profile ID·내용 해시·전체 프로필을 남긴다. 변경 이벤트는 해당 tick 종료 후, 다음 tick부터 적용됨을 뜻한다. 설정 셀을 다시 실행하면 그 셀의 `PROFILE_PATH`로 초기화된다.

REAL은 기존 DRY RUN으로 시작하며 실제 로봇 피드백이 필요하다. [worker 준비](WORKER_SETUP.md)와 노트북의 연결·DRY RUN 확인 절차는 유지한다. Quest 추정 위치 모드(`ACCEPT_EMULATED_POSE=True`, `TRACKING_PAUSE_S=.1`)로 고정하고 모드 전환 UI만 제거한다. 그리퍼 속도 UI만 제거하며 팔 속도 UI는 유지한다. 일반 REAL의 팔/그리퍼 상한 60/60°/s, 기록용의 100/120°/s와 복귀·IK·기록 처리는 그대로다.

## 선택적 SIM 적용

[SIM 노트북](../sim_notebook/sim_vr_teleop.ipynb)도 같은 커맨드 센터 목록을 사용한다. 시작 프로필은 설정에서 지정할 수 있다. SIM 확인은 REAL 사용의 선행 조건이 아니다. SIM의 기존 추적 모드·그리퍼 속도 UI는 유지한다.

```python
PROFILE_PATH=PROJECT_DIR / "calibration/profiles/<YYMMDD_이름>.json",
```

보정 없이 비교하려면 `PROFILE_PATH=None`으로 설정한다.

## 기존 파일 재사용

- **기존 프로필 직접 사용:** v1의 `accepted=true` 또는 `accepted=false, reason="trajectory_validation_failed"` 프로필은 원래 profile ID 해시와 양손 행렬 유효성이 맞으면 직접 사용할 수 있다. 저장된 행렬을 재계산하거나 정규화하지 않으며 원본 JSON도 수정하지 않는다. 미완료·축 계산 실패·손상 파일은 사용하지 않는다. REAL에는 Quest에서 만든 프로필을 사용하며 synthetic 프로필은 허용하지 않는다.
- **기존 원본 capture로 새 행렬 계산:** 전진·왼쪽·상승·오른쪽의 유효한 각 5회가 있으면 새 계산기를 사용할 수 있다. 과거 대각선 기록은 계산에서 제외하되 원본 전체의 해시는 provenance로 보존한다. 과거 1~4회 대신 1~5회를 쓰므로 새 행렬은 기존 v1 행렬과 달라질 수 있다. 새 파일명으로 저장한다.
- `calibration/reviews/`의 기존 SIM 확인 기록은 로컬 과거 기록으로 보존하며, 현재 로딩과 Git 배포에는 포함하지 않는다.

기존 capture를 오프라인 재계산할 때는 라이브 수집을 열 필요가 없다. 프로젝트 초기화 셀을 실행한 후 별도 셀에서 다음처럼 사용한다.

```python
import json
from package.intent_calibration import fit_intent_profile, save_json
capture = json.loads((PROJECT_DIR / "calibration/diagnostics/v1_trajectory.json").read_text())
refitted = fit_intent_profile(capture)
if not refitted["fit_success"]:
    raise ValueError(refitted["reason"])
save_json(PROJECT_DIR / "calibration/profiles/v1_refit_5repeats.json", refitted)
```
