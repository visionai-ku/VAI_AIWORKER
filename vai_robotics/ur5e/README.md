# UR5e teleoperation

Quest 3의 컨트롤러 한 개 또는 3Dconnexion SpaceMouse로 TCP의 3축 이동과 3축 회전을 조작한다. MuJoCo는 DRY RUN, SIM RUN, REAL RUN에서 UR5e 손목에 장착된 OnRobot RG2와 두 손가락을 표시한다. 시뮬레이션의 팔 제어 기준은 손목 플랜지(`attachment_site`), 실기의 기준은 티치 펜던트에 설정된 TCP다. 두 환경에서 같은 지점을 비교하려면 툴 오프셋을 맞춘다.

## 설치

[기존 Python 환경](../../SETUP.md)을 활성화하고 저장소 루트에서 실행한다. **Quest + 시뮬레이션**은 이 환경만으로 실행할 수 있다. SpaceMouse에는 `pyspacemouse`와 시스템 HIDAPI, 실물 UR5e와 RG2에는 각각 `ur-rtde`와 `pymodbus`를 사용한다. Python 패키지는 다음 명령으로 설치한다.

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

### 조작 패널

데스크톱에서 teleop을 실행하면 설정 패널이 함께 열린다. **DRY RUN**에서 클러치를 누른 채 조작하면 MuJoCo의 가상 팔만 움직인다. 클러치를 놓으면 가상 팔이 기준 자세로 돌아간다. 기준은 시뮬레이션에서는 SIM RUN에 보존된 현재 관절 자세이고, 실기에서는 UR에서 읽은 현재 관절 자세다. 다음에 클러치를 누르면 그 자세에서 새 미리보기를 시작한다. **입력 배율**은 이동과 회전에 함께 적용된다. 패널에서 실측·목표 위치(mm), 회전벡터(°), 클러치 상태를 확인할 수 있다.

- **시뮬레이션**은 **SIM RUN**으로 시작한다. 처음부터 미리보기를 열려면 `python -m vai_robotics.ur5e.teleop --backend sim --input spacemouse --dry-run`을 실행한다. 노트북에서는 `START_IN_DRY_RUN = True`로 설정한다. 실행 중에는 패널에서 모드를 바꿀 수 있다.
- **실기**는 수신 전용 **DRY RUN**으로 시작한다. 클러치를 놓고 **REAL RUN · 실기 구동**을 누르면 실기 제어가 시작된다. 같은 MuJoCo 창에서 가상 팔이 클러치 상태와 관계없이 UR에서 수신한 6개 관절각을 따라 움직인다. 패널에서 실측 TCP와 명령 목표를 비교할 수 있다. **DRY RUN**으로 돌아가면 서보를 정지하고 제어 연결을 해제하며, 같은 창에서 다시 조작을 미리 볼 수 있다. MuJoCo 화면은 RTDE 제어 루프와 별도 프로세스에서 그린다.

SpaceMouse를 처음 연결하거나 세션을 다시 열었을 때 **상태 확인 대기**가 보이면 클러치(기본 오른쪽 1번)를 한 번 눌렀다 놓는다. 두 버튼에서 모두 손을 뗀 상태로 **REAL RUN · 실기 구동**을 누른다. 패널은 전환 중이라는 문구를 표시하고, 연결되면 선택된 모드를 바꾼다. 전환에 실패하면 버튼 아래에 사유가 표시된다.

모드를 바꾸면 이전 목표가 지워진다. 클러치를 놓았다가 다시 누르면 현재 위치에서 시작한다. **세션 종료** 또는 창 닫기로 종료한다. 실기에서 패널과 MuJoCo 창 없이 CLI로 조작하려면 `--no-panel`을 사용한다. 실기를 직접 제어 모드로 시작하는 기존 옵션은 `--enable-motion`이다. 시뮬레이션의 `--headless`는 두 창을 모두 생략한다.

실기 뷰어 왼쪽 위의 `DRY RUN · Preview`와 `REAL RUN · Measured` 표시로 현재 화면을 구분한다. REAL RUN에서 팔은 실측 관절각을, 색깔 좌표축은 컨트롤러가 보고한 현재 TCP의 위치와 방향을 보여준다. 실기 제어 스크립트가 중단되고 자세 수신이 정상인 경우 패널에 사유를 표시하고 수신 전용 DRY RUN으로 돌아간다. 다시 구동하려면 클러치를 놓고 REAL RUN을 선택한다.

### 실물 UR5e 연결

1. 조작 PC와 UR 컨트롤러를 서로 통신할 수 있는 네트워크에 연결한다. 티치 펜던트의 네트워크 설정에서 **UR 컨트롤러의 IP**를 확인해 실기 노트북의 `ROBOT_IP` 또는 CLI의 `--robot-ip`에 입력한다. Quest를 함께 쓰면 `--quest-host`에는 헤드셋에서 접속할 **조작 PC의 주소**를 입력한다.
2. `--backend real`로 실행하면 PC가 `ur-rtde`의 수신 인터페이스로 컨트롤러의 RTDE 포트(TCP 30004)에 접속해 6개 관절 각도와 TCP 자세를 읽는다. 패널의 실측 TCP와 미리보기 가상 팔의 초기 자세를 확인한다. 기본 DRY RUN은 이 읽기 연결만 사용한다.
3. 실기를 움직이기 전 TCP·안전 평면·작업 공간을 확인하고, 펜던트에서 원격 제어 모드를 설정한 뒤 실행 중인 프로그램을 정지한다. **REAL RUN**으로 전환하면 PC가 같은 UR 컨트롤러에 제어 인터페이스를 추가로 열어 클러치 입력에 따라 `servoL`로 TCP 목표를 전송한다. 이때도 수신 연결을 유지해 현재 자세와 제어 상태를 계속 확인한다.

### OnRobot RG2 연결과 조작

실물 그리퍼를 함께 쓰려면 **OnRobot Compute Box의 IP**를 실기 노트북의 `GRIPPER_IP`에 입력하거나 CLI에 `--gripper-ip`로 전달한다. 팔의 `ROBOT_IP`와 별도로 설정하는 Modbus TCP 연결이다(포트 502, 단일 그리퍼 장치 ID 65). 그리퍼 주소를 생략해도 팔을 조작할 수 있으며, 이때 실물 RG2의 개도 수신과 구동은 꺼져 있다. 아래 명령은 현재 워크스테이션의 주소를 사용한다.

```bash
ROBOT_IP=192.168.0.252
GRIPPER_IP=192.168.0.72
python -m vai_robotics.ur5e.teleop --backend real --input spacemouse \
  --robot-ip "$ROBOT_IP" --gripper-ip "$GRIPPER_IP" --gripper-force-n 40
```

SpaceMouse의 **0번과 1번을 함께 누르는 동안 그리퍼 닫힘을 요청한다**. Compact에서는 왼쪽이 0번, 오른쪽이 1번이다. 두 버튼을 누른 채 캡을 움직이면 물체를 잡은 상태로 팔을 이동할 수 있다. `--button`은 팔 클러치만 변경하며 그리퍼 조합은 0번과 1번이다.

실기와 시뮬레이션 모두 `--gripper-force-n`으로 RG2 파지력을 설정한다(기본 40 N, 범위 3–40 N). 노트북에서는 `GRIPPER_FORCE_N`을 바꾼다. 예를 들어 `python -m vai_robotics.ur5e.teleop --backend sim --input spacemouse --dry-run --gripper-force-n 40`으로 실기와 같은 설정을 미리 볼 수 있다.

**DRY RUN**에서 두 버튼을 누르면 설정 파지력과 현재 개도에 해당하는 제조사 속도 곡선을 따라 가상 그리퍼가 닫힌다. 완전히 열려 있을 때는 천천히 움직이고 닫힐수록 빨라진다. 40 N에서 110 mm를 전부 닫는 데 약 1.1–1.2초, 50 mm에서 닫는 데 약 0.4초가 걸리는 모델이다. 버튼 하나를 놓으면 즉시 실물 RG2의 현재 개도 또는 보존된 SIM RUN 개도로 돌아간다. 실측값이 끊기면 마지막 기준 개도를 사용하며, 실측 이력이 없으면 110 mm에서 시작한다. [OnRobot RG2 데이터시트 3쪽의 속도 곡선](https://onrobot.com/storage/datasheets/rg2/datasheet_rg2_v1.8_en.pdf#page=3)

**SIM RUN**과 **REAL RUN**에서는 패널의 **열림 목표 폭**을 0–110 mm 범위에서 0.1 mm 단위로 설정한다. 기본값은 75 mm다. 두 버튼을 함께 누르면 0 mm로 닫고, 하나를 놓으면 설정한 폭으로 이동한다. 값을 바꾸면 다음 열림 명령부터 적용된다. SIM RUN에서는 가상 그리퍼가 설정한 폭에 도달할 때까지 움직인다. REAL RUN에서는 Compute Box에 목표 개도와 설정 파지력을 전송하고, 패널에 실측 개도를 표시한다. DRY RUN에서 버튼을 놓으면 현재 기준 개도로 즉시 돌아간다.

RG2 메시는 오사카대 Harada 연구실의 모델을 사용한다. 최대 열림 명령 110 mm에서 메시의 안쪽 패드 간격은 약 104 mm다. 손가락 링크는 시각화용이다. [모델 출처와 라이선스](assets/README.md#onrobot-rg2)

## 장치 조작

| 장치 | 시작과 조작 |
|---|---|
| Quest 3 | PC와 헤드셋을 같은 네트워크에 연결한다. Quest Browser로 콘솔에 표시된 HTTPS 주소에 접속하고 인증서를 신뢰한 뒤 WebXR 세션을 시작한다. 추적이 잡힌 상태에서 로봇을 바라보고 grip을 놓아 기준 방향을 잡는다. 기본 오른손의 **측면 grip**을 누른 채 손을 옮기거나 돌린다. 왼손은 `--hand left`로 선택한다. |
| SpaceMouse | PC에 USB 케이블이나 수신기를 연결한다. 시작할 때 **1번 버튼(Compact는 오른쪽)을 한 번 눌렀다 놓는다**. 다시 버튼을 누른 채 캡을 밀거나 비튼다. 캡 변위가 TCP 이동·회전 속도를 결정한다. 패널에서 `버튼 1: 누름`과 `버튼 1: 놓음`을 확인할 수 있다. 버튼 번호는 0부터 시작한다. 0번을 쓰려면 CLI에 `--button 0`을 추가하거나 노트북에서 `BUTTON = 0`으로 설정한다. |

기본 배치에서 작업자는 로봇의 +X 쪽에서 베이스를 바라본다. 손이나 캡을 앞으로 밀면 베이스 −X, 오른쪽으로 밀면 +Y, 위로 올리면 +Z 방향이다. SIM RUN과 REAL RUN에서는 클러치를 놓으면 그 위치에서 정지한다. 추적 소실·입력 만료 후에는 클러치를 놓았다 다시 누른다. 종료는 `Ctrl+C` 또는 조작 패널의 **세션 종료**다.

공통 루프는 50 Hz에서 입력을 처리하며 TCP 목표 변화율을 최대 0.10 m/s, 0.60 rad/s로 제한한다. 실기에서는 목표와 피드백 간 거리·각도, UR이 보고한 안전 상태와 목표 도달 가능성을 검사한다. 제어 프로세스의 명령이 끊기면 UR 컨트롤러의 RTDE watchdog이 제어 프로그램을 정지시킨다.
