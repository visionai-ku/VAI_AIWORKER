# VR_teleoperation

FFW-SG2 SIM/REAL intent teleoperation 배포 폴더. 2026-09-17 현재 `FtF_VR/project_v2`에서 두 노트북의 실행 파일, 선택된 보정 데이터, 모델, 공용 Python 런타임, 최신 worker와 관련 오프라인 검사를 복사했다. 원본 `project_v2`와 저장소의 기존 `ri_motion_v5_VR`, `ros2_ws` 파일은 변경하지 않았다.

## 실행

Python 3.10 환경을 기준으로 한다. `requirements-notebook.txt`는 기존 macOS 환경의 버전 목록이며 다른 OS에서 검증한 lockfile은 아니다.

```bash
cd VR_teleoperation
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name ffw-sg2-vr --display-name "FFW SG2 VR"
python -m jupyterlab
```

- [SIM 노트북](sim_notebook/sim_intent_teleop.ipynb)을 먼저 실행한다.
- [REAL 노트북](real_notebook/real_intent_teleop.ipynb)은 DRY RUN으로 시작하며 실제 로봇 피드백이 필요하다. [로봇 worker 적용 안내](docs/WORKER_SETUP.md)를 따른다.
- 각 노트북은 새 커널에서 첫 셀부터 실행한다. 기존 프로젝트 모듈을 불러온 커널은 재시작한다.
- `QUEST_HOST` 자동 감지가 맞지 않으면 PC LAN IP를 지정한다. 기본 로봇 IP는 `192.168.6.2`다. HTTPS 인증서 생성에는 `openssl`이 필요하다.

숨김 파일 `.project_v2`는 작업공간 식별자로 보존했다. 폴더 이름은 `VR_teleoperation`이며 노트북 폴더와 저장소 루트 양쪽에서 찾도록 복사본의 경로 탐색을 조정했다. 실행에 필요한 `vendor/`와 `xml/`은 이 폴더 안의 사본을 사용한다.

## 보정 데이터와 현재 동작

노트북 설정 그대로 실행할 수 있도록 다음 세 파일을 포함했다. 복사본 `.gitignore`에는 이 파일들만 예외로 추가했다. 일반 `git add VR_teleoperation/`에 포함되며 추가로 수집한 개인 로그·프로필은 계속 제외된다.

- `calibration/diagnostics/v1.json`
- `calibration/diagnostics/v1_trajectory.json`
- `calibration/reviews/v1-sim-reviewed-real-trial.json`

이는 특정 사용자가 SIM에서 확인한 후보이며 `accepted=false`를 유지한다. REAL의 별도 확인 기록은 해당 프로필/시연 해시와 연결된다. 다른 사용자·Quest 방향·작업 위치에 대해 검증된 일반 보정값이라는 뜻은 아니다. 배포 시 확인 옵션이나 제어 설정은 변경하지 않았다.

팔과 그리퍼 시작 상한은 각각 60°/s다. 최신 worker에서 그리퍼 10~120°/s 설정을 지원하며 HOME/수동 조정은 최대 10°/s다. 기존 충돌 모델에는 로봇과 바닥만 있고 실제 책상이 포함되지 않는다. 장시간 실행에서 관찰된 lift/그리퍼 경계 피드백 거부와 추적/IK 중단 등은 이번 파일 배포에서 수정하지 않았다.

## 오프라인 검증

이 폴더에서 실행한다. 실제 Quest/로봇 연결 없이 복사본의 모델, 모듈 경로, 선택된 프로필과 worker 해시를 확인한다.

```bash
python scripts/check_workspace.py
python -m unittest discover -s tests -p 'test_*.py'
python scripts/validate_intent_pipeline.py
```

`check_workspace.py`는 두 노트북의 import·설정·모델 준비 셀만 실행한다. 라이브 실행 셀은 실행하지 않는다. 포함된 회귀 검사는 메모리 기반 transport/ROS mock을 사용한다. 합성 검증은 이상적인 관절 추종을 사용하며 실제 모터·부하·통신 검증을 대신하지 않는다. 실행 결과와 생성 장면은 `.cache/`에 저장되고 Git에서 제외된다.

## 배포 범위

`DEPLOYMENT_MANIFEST.json`에 각 파일의 원본/배포 SHA-256과 조정 사항을 기록한다. 복사한 파일 중에서는 경로 안내를 바꾼 `init_project.py`, 두 노트북, `.gitignore`, 이 배포 범위에 맞춘 검사 스크립트만 조정했다. README와 worker 안내는 이 배포 폴더에 맞게 새로 작성했다. 그 외 복사한 실행 코드와 worker는 현재 원본과 같다.

`SOURCE_MANIFEST.json`은 2026-09-15 작업공간 생성 당시의 출처 기록으로 보존한 것이며 현재 파일 해시 목록은 아니다. 출처·라이선스는 [THIRD_PARTY.md](THIRD_PARTY.md), 로봇 자산 고지는 [ROBOTIS_LICENSE](xml/ROBOTIS_LICENSE)에 있다.

과거 노트북, 실험/분석 산출물, 원본 `.cache`, 백업, 인증서/키, 컴파일 바이너리와 선택적 native C++ 빌드 소스는 포함하지 않았다. 현재 Python IK 실행에는 native 빌드가 필요하지 않다.
