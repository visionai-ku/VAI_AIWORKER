# VR_teleoperation

FFW-SG2 SIM/REAL intent teleoperation 배포 폴더. 2026-09-17 현재 `FtF_VR/project_v2`에서 두 노트북의 실행 파일, 선택된 보정 데이터, 모델, 공용 Python 런타임, 최신 worker와 관련 오프라인 검사를 복사했다. 원본 `project_v2`와 저장소의 기존 `ri_motion_v5_VR`, `ros2_ws` 파일은 변경하지 않았다. 2026-09-18에 같은 원본의 추적 손실 모드 변경(공용 코드 3개, 두 노트북의 설정·안내 셀, 관련 테스트 2개)을 다시 반영했다.

## 실행

Python 3.10 환경을 기준으로 한다. `requirements-notebook.txt`는 기존 macOS 환경의 버전 목록이며 다른 OS에서 검증한 lockfile은 아니다.

```bash
cd VR_teleoperation
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name ffw-sg2-vr --display-name "FFW SG2 VR"
python -m jupyterlab
```

- [SIM 노트북](sim_notebook/sim_vr_teleop.ipynb)을 먼저 실행한다.
- [REAL 노트북](real_notebook/real_vr_teleop.ipynb)은 DRY RUN으로 시작하며 실제 로봇 피드백이 필요하다. [로봇 worker 적용 안내](docs/WORKER_SETUP.md)를 따른다.
- 각 노트북은 새 커널에서 첫 셀부터 실행한다. 기존 프로젝트 모듈을 불러온 커널은 재시작한다.
- Ubuntu에서는 저장소 루트의 `bash setup_ubuntu_env.sh`로 `ri_motion_v5_env` 환경과 Jupyter 커널을 만든다. 이 스크립트는 환경·커널 안에서 `LD_LIBRARY_PATH`의 `/usr/lib/x86_64-linux-gnu`를 제외한다. 이 경로가 앞에 있으면(예: `~/.bashrc`의 CUDA/ROS 설정) PyQt5가 시스템 Qt 5.12를 불러와 `Could not load the Qt platform plugin "xcb"`로 종료된다.
- `QUEST_HOST` 자동 감지가 맞지 않으면 PC LAN IP를 지정한다. 기본 로봇 IP는 `192.168.6.2`다. HTTPS 인증서 생성에는 `openssl`이 필요하다.

숨김 파일 `.project_v2`는 작업공간 식별자로 보존했다. 폴더 이름은 `VR_teleoperation`이며 노트북 폴더와 저장소 루트 양쪽에서 찾도록 복사본의 경로 탐색을 조정했다. 실행에 필요한 `vendor/`와 `xml/`은 이 폴더 안의 사본을 사용한다.

## 보정 데이터와 현재 동작

Teleoperation의 충돌 거리 검사는 `TELEOP_COLLISION_CHECK=False`로 해제했다(SIM·REAL·기록용 REAL 공통). 양팔·그리퍼를 가까이 접근시키는 조작과 grip 재시작, 추적 일시정지·IK 대기 중 유지 명령에도 같은 설정을 적용한다. Initial pose, B 버튼 Reset pose의 각 단계, 수동 관절 이동은 전체 경로 및 전송 전 충돌 검사를 유지한다. 관절 범위·속도·팔꿈치 자세·피드백·ACK 검사는 계속 적용한다. 실행 창과 기록 메타데이터에 충돌 검사 설정을 표시하며, Teleoperation에도 검사를 적용하려면 `TELEOP_COLLISION_CHECK=True`로 바꾼다. 코드 갱신 후 노트북 커널을 재시작하고 설정 셀부터 다시 실행해야 반영된다.

자동 이동의 기본 충돌 여유는 5mm다. 기록용 REAL 노트북의 기존 `COLLISION_ESCAPE_MIN_M=.001`도 유지하므로, 시작 간격이 1~5mm일 때 시작보다 더 가까워지지 않는 경로는 허용한다(0.2mm 허용오차). 자동 이동이 충돌 검사로 거부되면 Teleoperation으로 간격을 확보한 뒤 다시 요청한다.

2026-09-21 자세 유지 IK를 SIM·REAL·기록용 REAL 노트북에 적용했다. `POSTURE_IK_ENABLED=True`에서는 손목 위치·방향을 동시에 제약하고, 그 안에서 초기 팔 자세와 손목 각도를 선호한다. 첫 양손 grip으로 실제 조작이 시작될 때 실측 자세를 기준으로 저장하며, grip 해제·재시작·DRY RUN에서 REAL 전환 시에도 같은 기준을 유지한다. 새 기준을 잡으려면 실행을 종료하고 준비 자세에서 런타임을 다시 시작한다. 자세 기준은 HOME 목표와 별개다.

IK 후보의 허용오차는 위치 1 mm·회전 0.5°이며, 초기 팔꿈치 방향에서 85° 이내로 제한한다. 속도 제한 때문에 실제 명령과 로봇이 최종 목표에 도달하는 데는 시간이 걸린다. 자세 선호는 모든 관절을 고정하거나 손목 굽힘을 특정 각도 이하로 보장하는 제약은 아니다. 만족하는 해가 없으면 기존 IK_WAIT/HOLD 절차로 처리한다. 기록용 REAL의 회전 deadband는 정확한 상대 회전 추종을 위해 0°로 바꿨다.

테스트 방법·설정·기존 IK 비교는 [자세 유지 IK 안내](docs/POSTURE_IK.md)를 참고한다. 로봇 worker 변경은 필요하지 않으며 REAL의 `DRY_RUN=True` 시작 설정은 유지한다.

노트북 설정 그대로 실행할 수 있도록 다음 세 파일을 포함했다. 복사본 `.gitignore`에는 이 파일들만 예외로 추가했다. 일반 `git add VR_teleoperation/`에 포함되며 추가로 수집한 개인 로그·프로필은 계속 제외된다.

- `calibration/diagnostics/v1.json`
- `calibration/diagnostics/v1_trajectory.json`
- `calibration/reviews/v1-sim-reviewed-real-trial.json`

이는 특정 사용자가 SIM에서 확인한 후보이며 `accepted=false`를 유지한다. REAL의 별도 확인 기록은 해당 프로필/시연 해시와 연결된다. 다른 사용자·Quest 방향·작업 위치에 대해 검증된 일반 보정값이라는 뜻은 아니다. 배포 시 확인 옵션이나 제어 설정은 변경하지 않았다.

팔과 그리퍼 시작 상한은 각각 60°/s다. 최신 worker에서 그리퍼 10~120°/s 설정을 지원하며 HOME/수동 조정은 최대 10°/s다. 기존 충돌 모델에는 로봇과 바닥만 있고 실제 책상이 포함되지 않는다. 장시간 실행에서 관찰된 lift/그리퍼 경계 피드백 거부와 IK 중단 등은 이번 파일 배포에서 수정하지 않았다.

추적 손실 모드: 두 노트북은 `추정 위치 사용`으로 시작한다. 한 손의 광학 추적이 끊겨도 Quest 추정 위치로 조작을 이어가며, 컨트롤러가 아예 사라지면 0.1초 뒤 정지한다. 실행 창의 `정지 0.5초` 버튼을 누르면 추적 손실 중 현재 명령을 최대 0.5초 유지하는 방식으로 바뀐다. 전환은 양손 grip을 놓은 상태에서 적용된다. 두 모드 모두 한 프레임 20cm 초과 점프·grip 해제·피드백 오류 시 정지하고, 작업범위·속도 제한은 같다. 충돌 검사는 두 추적 모드 모두 위의 Teleoperation/자동 이동 설정을 따른다. 로봇 worker는 바뀌지 않았다.

## 오프라인 검증

이 폴더에서 실행한다. 실제 Quest/로봇 연결 없이 복사본의 모델, 모듈 경로, 선택된 프로필과 worker 해시를 확인한다.

```bash
python scripts/check_workspace.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/validate_intent_pipeline.py
```

`check_workspace.py`는 두 노트북의 import·설정·모델 준비 셀만 실행한다. 라이브 실행 셀은 실행하지 않는다. 포함된 회귀 검사는 메모리 기반 transport/ROS mock을 사용한다. 합성 검증은 이상적인 관절 추종을 사용하며 실제 모터·부하·통신 검증을 대신하지 않는다. 실행 결과와 생성 장면은 `.cache/`에 저장되고 Git에서 제외된다.

## 배포 범위

`DEPLOYMENT_MANIFEST.json`에 2026-09-17 배포 당시 각 파일의 원본/배포 SHA-256과 조정 사항을 기록한다. 복사한 파일 중에서는 경로 안내를 바꾼 `init_project.py`, 두 노트북, `.gitignore`, 이 배포 범위에 맞춘 검사 스크립트만 조정했다.

로봇용 worker는 `package/worker/`가 정본이며 2026-09-21에 로봇에서 실행 중인 실물로 갱신했다. 이전에 두던 `deploy/20260917-home-gripper-worker/` 스냅샷은 내용이 뒤처져 제거했고, `check_workspace.py`의 번들 해시 검증도 함께 뺐다.

`SOURCE_MANIFEST.json`은 2026-09-15 작업공간 생성 당시의 출처 기록으로 보존한 것이며 현재 파일 해시 목록은 아니다. 출처·라이선스는 [THIRD_PARTY.md](THIRD_PARTY.md), 로봇 자산 고지는 [ROBOTIS_LICENSE](xml/ROBOTIS_LICENSE)에 있다.

과거 노트북, 실험/분석 산출물, 원본 `.cache`, 백업, 인증서/키, 컴파일 바이너리와 선택적 native C++ 빌드 소스는 포함하지 않았다. 현재 Python IK 실행에는 native 빌드가 필요하지 않다.
