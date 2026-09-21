# REAL worker 배포

PC는 [real_vr_teleop.ipynb](../real_notebook/real_vr_teleop.ipynb)를 사용한다. `package/worker/`의 네 파일은 2026-09-21 기준 로봇에서 실행 중인 실물과 해시가 동일하다.

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

inbound 시작 로그의 상한은 arm 120, gripper 120, home 10°/s다. PC의 초기 팔/그리퍼 설정은 60°/s다. 기본 연결은 로봇 IP `192.168.6.2`, 상태 5560, PC 명령 5561, Quest HTTPS 8443이다. 로봇 측에는 ROS2의 `rclpy`, 메시지 패키지와 `pyzmq`가 필요하다.

노트북도 새 커널에서 실행한다. SIM 확인 후 REAL DRY RUN으로 피드백과 동작을 확인하고 실제 운용을 진행한다. DRY RUN도 로봇 피드백이 필요하다. HOME와 수동 미리보기의 모델에는 실제 책상이 없다는 점은 [README](../README.md)의 현재 동작 설명과 같다.

이번 작업은 저장소에 파일을 준비한 것이다. 로봇 컨테이너 복사·프로세스 재시작·실제 명령 전송은 수행하지 않았다. 롤백 시 제어를 정지하고 로봇에서 백업한 세 파일과 대응하는 PC 버전을 복원한다.
