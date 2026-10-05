# VAI_AIWORKER

- [환경 설치](SETUP.md): Python 3.12 / LeRobot 0.6.1, `ri_motion_v5_py312` 커널
- [VR·Joint·EEF·Base 실행](VR_teleoperation/README.md): 공통 JSON 제어 코드와 노트북 위치
- [로봇 worker 업데이트](VR_teleoperation/docs/WORKER_SETUP.md): Base JSON 지원에 필요한 파일과 실행 명령
- [macOS 환경 복원](environments/README.md)

## 배포 파일 관리

- `VR_teleoperation/`의 실행 코드·모델·Quest 클라이언트와 기존 Joint·EEF·Base 노트북 진입점을 유지한다. 작업자 보정 프로필 JSON·원본 궤적 JSON·보정 수집 로그 JSONL은 모두 Git에 포함하며, 그 외 실행 로그·캐시·생성 바이너리는 제외한다.
- `ros2_ws/build/`, `ros2_ws/install/`, `ros2_ws/log/`, 기존 BVH 데이터와 Leap Motion 압축파일은 로컬 파일을 보존하고 Git 추적에서 제외했다. 이미 배포한 로봇의 설치 디렉터리를 삭제하라는 뜻은 아니다.
- ROS 소스의 11개 gitlink는 유지한다. 현재 루트 `.gitmodules`에 해당 URL이 등록되어 있지 않으므로, 새 로봇 환경을 재빌드하려면 소스 저장소 URL·버전과 빌드 절차를 먼저 복원해야 한다. Git에서 제외한 설치 산출물만으로 소스 복원 가능성을 가정하지 않는다.
- `VR_teleoperation/tests/`, 검증 스크립트, 기존 v1 원본 궤적은 재현·회귀 검사를 위해 소스 저장소에 유지한다. 과거 SIM 확인 기록은 로컬 보관하며 현재 실행에는 필요하지 않다.
