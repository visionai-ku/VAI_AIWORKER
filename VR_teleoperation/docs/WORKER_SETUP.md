# REAL worker 배포

PC는 [real_vr_teleop.ipynb](../real_notebook/real_vr_teleop.ipynb)를 사용한다. B 버튼 reset의 정렬 이후 속도 변경을 적용하려면 아래 worker 파일을 로봇에도 업데이트해야 한다.

로봇 컨테이너의 실제 실행 위치는 `/root/ros2_ws/src/ai_worker/zmq/`다. 로봇 조작과 기존 inbound/outbound 프로세스를 정지하고 기존 파일을 별도 백업한 후 다음 네 파일을 한 세트로 복사한다.

```text
package/worker/inbound.py
package/worker/outbound.py
package/worker/fixed_quest_protocol.py
package/worker/camera_outbound.py
```

두 worker를 모두 재시작해야 새 protocol이 적용된다. ROS 환경이 설정된 컨테이너의 별도 터미널에서 실행하며 `<PC_IP>`는 노트북 PC LAN IP로 바꾼다.

```bash
ros2 launch ffw_bringup ffw_sg2_follower_ai.launch.py
```

```bash
SG2_FIXED_QUEST=1 python3 /root/ros2_ws/src/ai_worker/zmq/outbound.py
```

```bash
SG2_FIXED_QUEST=1 SG2_ZMQ_SUB_IP=<PC_IP> python3 /root/ros2_ws/src/ai_worker/zmq/inbound.py
```

### `invalid_json_feedback`가 계속 표시될 때

`utf-8 ... byte 0x80 in position 0` 또는 `protocol_mismatch_legacy_pickle`이면 로봇 상태 포트에서 기존 pickle 형식의 데이터가 수신되고 있는지 확인한다. 이 노트북은 strict JSON 상태와 명시적 command bridge ACK를 사용한다. `SG2_FIXED_QUEST`가 없거나 `1`이 아니면 현재 worker는 legacy pickle 모드로 시작한다.

노트북 조작을 중지하고 로봇 컨테이너에서 기존 inbound/outbound를 종료한 다음, 위 두 명령처럼 **각 worker 프로세스에 `SG2_FIXED_QUEST=1`을 전달하여** 다시 실행한다. 노트북 PC에만 환경변수를 설정하거나 이미 실행 중인 worker의 터미널에 뒤늦게 `export`해도 기존 프로세스 모드는 바뀌지 않는다. 기존 outbound에 `--meta`를 사용했다면 재시작 명령에도 유지한다.

outbound 시작 로그는 `Protocol: strict JSON with source freshness`, inbound는 `Protocol: strict JSON; explicit arm required`여야 한다. `legacy pickle`이 표시되면 시작 환경이 적용되지 않은 것이다. 기존 worker가 남아 있으면 5560 포트를 새 worker가 사용할 수 없으므로 먼저 기존 프로세스가 종료되었는지 확인한다. 그다음 노트북을 재실행한다. DRY RUN에서도 두 strict worker의 정상 피드백이 필요하다.

inbound 시작 로그의 상한은 arm 120, gripper 120, home 100°/s다. B 버튼 reset은 양손 위·바깥 이동과 첫 팔 정렬(`shutdown1`)까지 30°/s, 도달 확인 후 `shutdown2`부터 뒤로 접기·앞으로 펴기는 팔 100°/s다. B 팔 속도는 일반 조작 속도와 별도로 적용하며, 머리·그리퍼 상한과 lift 속도는 유지된다. 일반 Initial pose와 수동 조정은 기존 10°/s다. PC의 초기 팔/그리퍼 조작 설정은 60°/s다. 기본 연결은 로봇 IP `192.168.6.2`, 상태 5560, PC 명령 5561, Quest HTTPS 8443이다. 로봇 측에는 ROS2의 `rclpy`, 메시지 패키지와 `pyzmq`가 필요하다.

PC는 실제 bridge 피드백의 `home_joint_speed_rad_s`도 적용한다. 구버전 worker의 상한이 30/60°/s라면 업데이트 안내와 함께 모든 접기·펴기를 해당 상한 이내로 수행한다. 이를 무시하고 더 빠른 명령을 전송하면 `shutdown2`에서 `joint_rate_limit` 거절과 command lease 만료가 반복될 수 있다.

100°/s 적용 시 `FIXED_QUEST_VR_HOME_SPEED = math.radians(100)`뿐 아니라 **`fixed_quest_protocol.py`와 `inbound.py` 전체를 이번 버전으로 교체**한다. 이전 코드의 피드백·시작 로그는 머리 관절의 60°/s 상한을 HOME 전체 상한으로 보고하므로 상수만 수정해서는 PC가 팔 100°/s를 사용할 수 없다. 새 코드도 각 관절의 기존 상한을 적용하며 머리 상한은 60°/s다. inbound 재시작 후 `home 100.0 deg/s`, 피드백 `home_joint_speed_rad_s` 약 `1.74532925` rad/s를 확인한다. PC 코드 변경은 노트북 커널을 재시작해야 반영된다.

노트북도 새 커널에서 실행한다. SIM 확인 후 REAL DRY RUN으로 피드백과 동작을 확인하고 실제 운용을 진행한다. DRY RUN도 로봇 피드백이 필요하다. HOME와 수동 미리보기의 모델에는 실제 책상이 없다는 점은 [README](../README.md)의 현재 동작 설명과 같다.

이번 작업은 저장소에 파일을 준비한 것이다. 로봇 컨테이너 복사·프로세스 재시작·실제 명령 전송은 수행하지 않았다. 롤백 시 제어를 정지하고 로봇에서 백업한 세 파일과 대응하는 PC 버전을 복원한다.
