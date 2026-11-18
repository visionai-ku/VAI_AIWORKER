# UR5e teleoperation

Quest 3의 컨트롤러 한 개 또는 3Dconnexion SpaceMouse로 TCP의 3축 이동과 3축 회전을 조작한다. 시뮬레이션에서는 기본 모델의 플랜지(`attachment_site`), 실기에서는 티치 펜던트에 설정된 TCP가 기준이다. 두 환경에서 같은 지점을 비교하려면 툴 오프셋을 맞춘다.

## 설치

[기존 Python 환경](../../SETUP.md)을 활성화하고 저장소 루트에서 실행한다. **Quest + 시뮬레이션**은 이 환경만으로 실행할 수 있다. SpaceMouse에는 `pyspacemouse`와 시스템 HIDAPI, 실기에는 `ur-rtde`가 필요하다. 준비된 워크스테이션에는 두 드라이버를 함께 설치할 수 있다.

```bash
python -m pip install -r vai_robotics/requirements-teleop.txt
```

- **macOS**에서는 `brew install hidapi cmake boost`로 네이티브 의존성을 설치한다. Apple Silicon에서 Xcode Command Line Tools와 CMake 4를 사용해 검증한 명령은 `CMAKE_POLICY_VERSION_MINIMUM=3.5 python -m pip install -r vai_robotics/requirements-teleop.txt`다.
- **Ubuntu**에서는 HIDAPI 개발 패키지(`libhidapi-dev`)와 SpaceMouse의 hidraw 장치 접근 권한이 필요하다. 연결된 장치를 다른 드라이버가 독점하고 있으면 해당 드라이버를 종료한다. [PySpaceMouse 설치 안내](https://github.com/JakubAndrysek/PySpaceMouse)

## 실행

노트북은 [시뮬레이션](notebooks/sim_teleop.ipynb)과 [실기](notebooks/real_teleop.ipynb) 중 하나를 선택한다. 터미널에서는 아래 네 조합 중 하나를 실행한다. `QUEST_HOST`는 헤드셋에서 접속할 **PC의 LAN 주소**, `ROBOT_IP`는 **UR 컨트롤러 주소**로 설정한다.

```bash
python -m vai_robotics.ur5e.teleop --backend sim  --input quest --quest-host "$QUEST_HOST"
python -m vai_robotics.ur5e.teleop --backend sim  --input spacemouse
python -m vai_robotics.ur5e.teleop --backend real --input quest --quest-host "$QUEST_HOST" --robot-ip "$ROBOT_IP"
python -m vai_robotics.ur5e.teleop --backend real --input spacemouse --robot-ip "$ROBOT_IP"
```

**실기는 기본이 수신 전용 dry-run이다.** 클러치를 누르고 조작하면 현재 TCP 기준으로 계산한 목표가 터미널에 표시된다. 실제 동작을 시작할 때는 로봇의 TCP·안전 평면·작업 공간을 점검하고, 원격 제어 모드에서 기존 프로그램을 정지한 후 같은 명령에 `--enable-motion`을 추가한다. 실기 노트북의 대응 설정은 `ENABLE_MOTION = True`다.

## 장치 조작

| 장치 | 시작과 조작 |
|---|---|
| Quest 3 | PC와 헤드셋을 같은 네트워크에 연결한다. Quest Browser로 콘솔에 표시된 HTTPS 주소에 접속하고 인증서를 신뢰한 뒤 WebXR 세션을 시작한다. 추적이 잡힌 상태에서 로봇을 바라보고 grip을 놓아 기준 방향을 잡는다. 기본 오른손의 **측면 grip**을 누른 채 손을 옮기거나 돌린다. 왼손은 `--hand left`로 선택한다. |
| SpaceMouse | PC에 USB 케이블이나 수신기를 연결한다. 시작할 때 **0번 버튼을 한 번 눌렀다 놓는다**. 다시 버튼을 누른 채 캡을 밀거나 비튼다. 캡 변위가 TCP 이동·회전 속도를 결정한다. 다른 물리 버튼은 `--button 1`처럼 선택한다. |

기본 배치에서 작업자는 로봇의 +X 쪽에서 베이스를 바라본다. 손이나 캡을 앞으로 밀면 베이스 −X, 오른쪽으로 밀면 +Y, 위로 올리면 +Z 방향이다. 클러치를 놓으면 현재 자세를 유지하고, 다시 누르면 그 위치에서 이어진다. 추적 소실·입력 만료 후에는 클러치를 놓았다 다시 누른다. 종료는 `Ctrl+C` 또는 시뮬레이션 창 닫기다.

공통 루프는 50 Hz에서 입력을 처리하며 TCP 목표 변화율을 최대 0.10 m/s, 0.60 rad/s로 제한한다. 실기에서는 목표와 피드백 간 거리·각도, UR이 보고한 안전 상태와 목표 도달 가능성을 검사한다. 제어 프로세스의 명령이 끊기면 UR 컨트롤러의 RTDE watchdog이 제어 프로그램을 정지시킨다.
