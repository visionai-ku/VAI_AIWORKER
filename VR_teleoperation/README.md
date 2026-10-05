# VR_teleoperation

FFW-SG2 SIM/REAL intent teleoperation 배포 폴더. 2026-09-17 현재 `FtF_VR/project_v2`에서 두 노트북의 실행 파일, 선택된 보정 데이터, 모델, 공용 Python 런타임, 최신 worker와 관련 오프라인 검사를 복사했다. 원본 `project_v2`와 저장소의 기존 `ri_motion_v5_VR`, `ros2_ws` 파일은 변경하지 않았다. 2026-09-18에 같은 원본의 추적 손실 모드 변경(공용 코드 3개, 두 노트북의 설정·안내 셀, 관련 테스트 2개)을 다시 반영했다.

## 실행

Python 3.12 환경을 기준으로 한다 (lerobot 0.6.1이 Python 3.12 이상을 요구한다. 학습 환경 `lerobot_act`와 같은 lerobot 계열). `requirements-notebook.txt`는 직접 의존성의 버전 목록이며 다른 OS에서 검증한 lockfile은 아니다. CUDA torch는 requirements보다 먼저 설치한다(`requirements.txt` 상단 주석).

```bash
cd VR_teleoperation
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name ffw-sg2-vr --display-name "FFW SG2 VR"
python -m jupyterlab
```

- [SIM 노트북](sim_notebook/sim_vr_teleop.ipynb)은 선택적으로 실행한다. REAL 사용에 SIM 확인 기록은 필요하지 않다.
- 새 개인 보정은 [보정 수집 노트북](calibration_notebook/calibrate_intent_axes.ipynb)에서 만든다. 준비·수집·SIM/REAL 적용 순서는 [보정 안내](docs/CALIBRATION.md)를 따른다.
- [REAL 노트북](real_notebook/real_vr_teleop.ipynb)은 DRY RUN으로 시작하며 실제 로봇 피드백이 필요하다. [로봇 worker 적용 안내](docs/WORKER_SETUP.md)를 따른다.
- 각 노트북은 새 커널에서 첫 셀부터 실행한다. 기존 프로젝트 모듈을 불러온 커널은 재시작한다.
- Ubuntu에서는 저장소 루트의 `bash setup_ubuntu_env.sh`로 `ri_motion_v5_py312` 환경과 Jupyter 커널을 만든다 (기존 Python 3.10 / lerobot 0.4.4 환경 `ri_motion_v5_env`는 건드리지 않는다). 이 스크립트는 환경·커널 안에서 `LD_LIBRARY_PATH`의 `/usr/lib/x86_64-linux-gnu`를 제외한다. 이 경로가 앞에 있으면(예: `~/.bashrc`의 CUDA/ROS 설정) PyQt5가 시스템 Qt 5.12를 불러와 `Could not load the Qt platform plugin "xcb"`로 종료된다.
- macOS Apple Silicon에서는 [환경 복원 안내](../environments/README.md)의 Python 3.12 / LeRobot 0.6.1 환경을 사용한다. 커널 이름은 `ri_motion_v5_py312`다.
- `QUEST_HOST` 자동 감지가 맞지 않으면 PC LAN IP를 지정한다. 기본 로봇 IP는 `192.168.6.2`다. HTTPS 인증서 생성에는 `openssl`이 필요하다.

숨김 파일 `.project_v2`는 작업공간 식별자로 보존했다. 폴더 이름은 `VR_teleoperation`이며 노트북 폴더와 저장소 루트 양쪽에서 찾도록 복사본의 경로 탐색을 조정했다. 실행에 필요한 `vendor/`와 `xml/`은 이 폴더 안의 사본을 사용한다.

## Joint·EEF·Base 컨트롤러

`VAI_merge`에서 검증한 공통 클라이언트와 컨트롤러를 반영했다. 다음 노트북은 기존 위치를 유지하지만 코드·모델·worker는 이 `VR_teleoperation` 폴더를 사용한다. `ri_motion_v5_py312` 새 커널에서 첫 셀부터 실행한다.

- [Joint 컨트롤러](../ri_motion_v5_VR/project/ffw_sg2_vr_teleoperation/real_notebook/real_joint_controller_sg2_teleop.ipynb)
- [EEF 컨트롤러](../ri_motion_v5_VR/project/ffw_sg2_vr_teleoperation/real_notebook/real_eef_controller_sg2_teleop.ipynb)
- [Base 컨트롤러](../ri_motion_v5_VR/project/ffw_sg2_vr_teleoperation/real_notebook/real_base_controller_sg2.ipynb)
- [상태 렌더러](../ri_motion_v5_VR/project/ffw_sg2_vr_teleoperation/real_notebook/real_render_MetaQuest3_sg2.ipynb)
- [녹화 세션 예제](../ri_motion_v5_VR/project/ffw_sg2_vr_teleoperation/real_notebook/real_sg2_record_session.ipynb)

VR·Joint·EEF·Base·정책 추론은 **JSON이 기본값**이며, 로봇 제어 세션은 한 번에 하나만 실행한다. Base는 새 worker의 `capabilities.base_json`을 확인하고, 0 속도 arming과 0 명령의 ACK가 끝난 뒤 키를 놓았다가 새로 누르면 이동한다. SPACE는 Robot command를 OFF로 바꾸고 0 stop을 전송한다. 평면 속도 크기 상한은 0.3 m/s, 회전은 0.6 rad/s다. 명령·피드백 만료 시 worker도 0 속도를 발행한다.

**Base를 사용하려면 로봇의 `inbound.py`, `outbound.py`, `fixed_quest_protocol.py`도 갱신하고 두 worker를 재시작해야 한다.** [worker 적용 안내](docs/WORKER_SETUP.md)를 따른다. 기존 worker에 JSON 설정만 추가해도 Base 명령 형식은 지원되지 않는다. Joint/EEF에는 일시적인 `joint_rate_limit` 거부 시 제한된 목표를 유지하고 ACK를 기다리는 처리를 반영했다.

Joint/EEF 준비 셀은 포함된 C++ 소스로 현재 Python ABI에 맞는 IK 모듈을 빌드한다. macOS는 Xcode Command Line Tools, Ubuntu는 `build-essential`이 필요하다. 컴파일 바이너리는 배포하지 않는다. 과거 REAL motion executor·MetaQuest 예제는 계속 legacy 모드이며, 실행 시 양쪽 worker에 `SG2_FIXED_QUEST=0`을 명시한다.

## 보정 데이터와 현재 동작

Initial pose 버튼과 컨트롤러 A·B의 머리 `head_joint1` 목표는 bringup과 동일한 **0.69 rad(약 39.5°)** 아래 방향으로 고정한다. `head_joint2`는 실측값을 유지한다. 변경된 공용 상수를 불러오려면 실행을 종료하고 노트북 커널을 재시작한 뒤 첫 셀부터 다시 실행한다.

2026-09-23 원본 `FtF_VR/project_v2`의 개인 의도 축 보정 노트북과 수집 UI를 복원했다. 새 수집은 `RESUME_CAPTURE_PATH=None`에서 시작하며 Quest 컨트롤러만 사용한다. 로봇 연결이나 grip 입력은 필요 없다. 현재는 전진·왼쪽·상승·오른쪽을 각 5회 수집하고 총 20회를 모두 행렬 계산에 사용한다. 대각선 수집과 궤적 점수 판정은 제거했다. 결과는 `YYMMDD_이름`의 새 파일로 저장한다. 프로필은 MuJoCo 커맨드 센터의 작업자 보정 목록에서 파일명으로 선택한다. `calibration/profiles/`와 `calibration/diagnostics/`에 사람 이름으로 보정 프로필 JSON을 두면 표시된다. 보정 미적용(identity)과 v1 원본도 별도 항목이며 파일명을 클릭하면 자동 적용한다. 조작·복귀·녹화 중에는 선택을 잠근다. DRY RUN 대기 중에는 추적 없이 프로필·XYZ 배율을 바꿀 수 있고, REAL RUN으로 전환해도 값을 유지한다. REAL RUN에서 직접 변경할 때는 기존 양손 추적·grip 해제 조건을 유지한다. 노트북 위젯이나 `ipywidgets`는 사용하지 않는다. 코드 갱신 후 커널을 재시작한다. 상세 절차와 결과 판정은 [보정 안내](docs/CALIBRATION.md)를 참고한다.

Teleoperation의 충돌 거리 검사는 `TELEOP_COLLISION_CHECK=False`로 해제했다(SIM·REAL·기록용 REAL 공통). 양팔·그리퍼를 가까이 접근시키는 조작과 grip 재시작, 추적 일시정지·IK 대기 중 유지 명령에도 같은 설정을 적용한다. Initial pose, B 버튼 Reset pose의 각 단계, 수동 관절 이동은 전체 경로 및 전송 전 충돌 검사를 유지한다. 관절 범위·속도·팔꿈치 자세·피드백·ACK 검사는 계속 적용한다. 실행 창과 기록 메타데이터에 충돌 검사 설정을 표시하며, Teleoperation에도 검사를 적용하려면 `TELEOP_COLLISION_CHECK=True`로 바꾼다. 코드 갱신 후 노트북 커널을 재시작하고 설정 셀부터 다시 실행해야 반영된다.

자동 이동의 기본 충돌 여유는 5mm다. 기록용 REAL 노트북의 기존 `COLLISION_ESCAPE_MIN_M=.001`도 유지하므로, 시작 간격이 1~5mm일 때 시작보다 더 가까워지지 않는 경로는 허용한다(0.2mm 허용오차). 자동 이동이 충돌 검사로 거부되면 Teleoperation으로 간격을 확보한 뒤 다시 요청한다.

2026-09-21 자세 유지 IK를 SIM·REAL·기록용 REAL 노트북에 적용했다. `POSTURE_IK_ENABLED=True`에서는 손목 위치·방향을 동시에 제약하고, 그 안에서 초기 팔 자세와 손목 각도를 선호한다. 첫 양손 grip으로 실제 조작이 시작될 때 실측 자세를 기준으로 저장하며, grip 해제·재시작·DRY RUN에서 REAL 전환 시에도 같은 기준을 유지한다. 새 기준을 잡으려면 실행을 종료하고 준비 자세에서 런타임을 다시 시작한다. 자세 기준은 HOME 목표와 별개다.

IK 후보의 허용오차는 위치 1 mm·회전 0.5°이며, 초기 팔꿈치 방향에서 85° 이내로 제한한다. 속도 제한 때문에 실제 명령과 로봇이 최종 목표에 도달하는 데는 시간이 걸린다. 자세 선호는 모든 관절을 고정하거나 손목 굽힘을 특정 각도 이하로 보장하는 제약은 아니다. 만족하는 해가 없으면 기존 IK_WAIT/HOLD 절차로 처리한다. 기록용 REAL의 회전 deadband는 정확한 상대 회전 추종을 위해 0°로 바꿨다.

테스트 방법·설정·기존 IK 비교는 [자세 유지 IK 안내](docs/POSTURE_IK.md)를 참고한다. 이 IK 옵션 전환에는 로봇 worker 변경이 필요하지 않으며 REAL의 `DRY_RUN=True` 시작 설정은 유지한다.

시작 보정과 원본 재계산·회귀 검사를 위해 다음 두 파일을 포함한다.

- `calibration/diagnostics/v1.json`
- `calibration/diagnostics/v1_trajectory.json`

`calibration/profiles/`와 `calibration/diagnostics/`의 작업자 보정 프로필 JSON·원본 궤적 JSON(`*_trajectory.json`, `*-intent-trajectories.json`)·보정 수집 로그 JSONL은 모두 Git에 포함한다. 새 작업자 데이터도 세 종류를 함께 커밋·배포할 수 있다. 커맨드 센터 선택 목록에는 보정 프로필만 표시된다.

기존 v1의 `accepted=false`와 저장 행렬은 그대로 보존한다. 공통 로더는 궤적 점수만 미통과한 v1도 파일 무결성과 행렬 유효성을 확인해 직접 읽는다. capture·SIM 확인 기록·setup 확인 옵션은 로딩에 필요하지 않다. 기존 review 파일은 로컬 과거 기록으로 보관하고 Git 배포에서 제외한다. 새 프로필은 `version=2`, `fit_success`로 계산 완료를 표시한다.

일반 REAL은 팔/그리퍼 60/60°/s, 기록용 REAL은 100/120°/s의 기존 설정을 유지한다. REAL의 그리퍼 속도 입력·적용 UI만 제거했으며 팔 속도 UI는 유지한다. 그리퍼는 노트북 설정의 상한을 사용하고 HOME/수동 조정의 기존 속도 처리는 유지한다. 기존 충돌 모델에는 로봇과 바닥만 있고 실제 책상이 포함되지 않는다. 최신 worker는 어깨·그리퍼의 0.002 rad 이내 경계 피드백을 정규화하며, 명령 관절 범위는 기존 제한을 유지한다. 장시간 실행의 모든 IK 중단 원인이 해결된 것은 아니다.

추적 손실 모드: 두 REAL 노트북은 기존 기본값인 `추정 위치 사용`으로 고정하며 모드 전환 버튼을 표시하지 않는다. 한 손의 광학 추적이 끊겨도 Quest 추정 위치로 조작을 이어가며, 컨트롤러가 아예 사라지면 0.1초 뒤 정지한다. 한 프레임 20cm 초과 점프·grip 해제·피드백 오류 시 정지하고 작업범위·속도 제한은 유지한다. SIM의 선택적 추적 모드·그리퍼 속도 UI는 유지한다. 제어 상태 전이, IK, worker, HOME/Reset pose, 녹화 절차는 변경하지 않았다.

## 오프라인 검증

이 폴더에서 실행한다. 실제 Quest/로봇 연결 없이 복사본의 모델, 모듈 경로, 선택된 프로필과 보정 노트북 준비 설정을 확인한다.

```bash
python scripts/check_workspace.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/validate_intent_pipeline.py
```

`check_workspace.py`는 SIM·REAL·기록용 REAL 노트북의 import·설정·모델 준비 셀과 보정 노트북의 준비 셀을 검사한다. 라이브 실행 셀은 실행하지 않는다. 포함된 회귀 검사는 메모리 기반 transport/ROS mock을 사용한다. 합성 검증은 이상적인 관절 추종을 사용하며 실제 모터·부하·통신 검증을 대신하지 않는다. 실행 결과와 생성 장면은 `.cache/`에 저장되고 Git에서 제외된다.

2026-09-29 공통 컨트롤러 반영 후 Python 3.12 환경의 전체 회귀 검사는 414개 중 412개 통과, 실제 checkpoint가 필요한 2개 skip이었다. 변경된 노트북 6개의 문법, 컨트롤러·도구 노트북 5개의 초기화 경로와 macOS native IK 컴파일·import도 확인했다. 실제 로봇 제어는 실행하지 않았다.

2026-10-05 보정·프로필 선택·REAL UI 변경 후 전체 회귀 검사는 442개 중 440개 통과, 실제 checkpoint가 필요한 2개 skip이었다. 기존 v1 행렬의 정확한 보존, 5회째 수집 반영, 커맨드 센터의 identity·v1 원본·작업자 파일 선택, 조작·복귀·녹화 중 잠금, 오류 시 현재 보정 보존, REAL/SIM 위젯 및 기록 시작·종료 순서를 확인했다. 보정 교체가 HOME·명령·속도·다음 grip 기준을 바꾸지 않는 것도 확인했다. DRY RUN 대기 중 추적 없는 프로필·배율 변경과 REAL RUN 전환 후 값 유지, 전환 후 기존 grip 재시작 조건도 확인했다. 워크스페이스 준비 검사와 합성 IK·추적 일시정지·HOME 복귀 검사도 통과했다. 합성 검사의 관찰 구간만 28초로 늘려 기존 HOME 완료를 확인했으며 제어 로직은 변경하지 않았다. 실제 로봇 제어는 실행하지 않았다.

## 배포 범위

`DEPLOYMENT_MANIFEST.json`에 2026-09-17 배포 당시 각 파일의 원본/배포 SHA-256과 조정 사항을 기록한다. 복사한 파일 중에서는 경로 안내를 바꾼 `init_project.py`, 두 노트북, `.gitignore`, 이 배포 범위에 맞춘 검사 스크립트만 조정했다.

로봇용 worker는 `package/worker/`가 정본이며 2026-09-21에 로봇에서 실행 중인 실물로 갱신했다. 이전에 두던 `deploy/20260917-home-gripper-worker/` 스냅샷은 내용이 뒤처져 제거했고, `check_workspace.py`의 번들 해시 검증도 함께 뺐다.

`SOURCE_MANIFEST.json`은 2026-09-15 작업공간 생성 당시의 출처 기록으로 보존한 것이며 현재 파일 해시 목록은 아니다. 출처·라이선스는 [THIRD_PARTY.md](THIRD_PARTY.md), 로봇 자산 고지는 [ROBOTIS_LICENSE](xml/ROBOTIS_LICENSE)에 있다.

실험/분석 산출물, 원본 `.cache`, 백업, 인증서/키와 컴파일 바이너리는 포함하지 않는다. Joint/EEF 실행에 필요한 native C++ 소스, 초기화 스크립트, qpos와 렌더러 모델은 이번에 추가했다. VR intent의 Python IK 실행에는 native 빌드가 필요하지 않다.
