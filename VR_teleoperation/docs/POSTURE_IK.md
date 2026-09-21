# 자세 유지 IK 테스트

SIM, REAL, 기록용 REAL의 공용 런타임에 적용된다. 새 커널에서 노트북을 처음부터 실행한다. REAL은 기존처럼 `DRY_RUN=True`로 시작한다.

## 조작과 비교

1. 팔꿈치 방향과 손목 각도가 편안한 준비 자세를 만든다.
2. 첫 양손 grip 조작이 활성화될 때 실측 관절 자세가 기준으로 저장된다. LIVE에서는 robot ACK 이후이며 DRY RUN에서도 기준을 저장한다.
3. grip을 놓고 다시 잡아도 팔 자세 기준은 유지한다. 컨트롤러의 상대 이동 기준점은 기존처럼 매 grip마다 다시 잡는다.
4. 앞뒤·좌우·상하 이동, 위치를 유지한 회전, 이동과 회전의 조합을 확인한다. J3/J5의 누적 비틀림, J6/J7 굽힘, 팔꿈치 방향, 목표 추종을 함께 본다.
5. 기존 IK와 비교하려면 설정 셀의 `POSTURE_IK_ENABLED=False`로 바꾸고 실행을 종료한 뒤 다시 시작한다. 공용 코드가 이미 로드된 상태에서 파일을 갱신했다면 커널도 재시작한다.

기준 자세를 바꾸려면 런타임을 종료하고 새 준비 자세에서 다시 시작한다. 현재 기준은 grip 해제, 제어 reset, 통신 재연결, DRY RUN→REAL 전환으로 자동 갱신되지 않는다. HOME 목표는 첫 로봇 피드백 자세이며 자세 유지 기준과 별개다. HOME/수동 관절 조정은 기존 경로를 사용한다.

## 설정

```python
POSTURE_IK_ENABLED=True
POSTURE_POSITION_TOLERANCE_M=.001
POSTURE_ROTATION_TOLERANCE_DEG=.5
POSTURE_ELBOW_LIMIT_DEG=85.
NULL_ANCHOR_ON_GRIP=False
```

새 IK에서는 `NULL_REFERENCE`와 `NULL_ANCHOR_ON_GRIP`의 기존 solver 설정 대신 고정된 초기 자세를 사용한다. 기록용 REAL의 `ROTATION_DEADBAND_DEG`도 0으로 설정하여 상대 회전을 그대로 목표에 반영한다.

1 mm/0.5°는 **이번 중간 목표에 대한 IK 후보의 허용오차**다. 최종 VR 목표, 속도 제한 후 명령 FK, 로봇 실측 FK의 오차는 별도로 관찰해야 한다. 외부 제어기의 속도 제한·추종 지연·모델 오차까지 보장하는 수치가 아니다.

## 계산 방식

`package/posture_ik.py`는 MuJoCo FK와 analytic Jacobian, SciPy SLSQP를 이용한다. 손목의 위치 3축·회전 3축을 등식 제약으로 두고, 관절 범위와 팔꿈치 방향 부등식 안에서 초기 자세 편차를 줄인다. 기존 vendor IK는 비교용으로 보존한다.

- J1/J2/J4는 상대적으로 자유롭게, J3/J5 비틀림과 J6/J7 손목 편차는 더 강하게 억제한다. 이는 자세 선호 비용이며 손목 각도의 추가 hard limit은 아니다.
- 팔꿈치 방향은 어깨(link2)→손목 제어점 축에 대한 팔꿈치(link4)의 방향으로 계산한다. 초기 방향을 현재 뻗는 방향으로 회전시켜 비교하므로 J4 각도 부호만으로 판단하지 않는다.
- 팔꿈치 방향 차이를 85° 이내로 제한한다. 팔이 펴져 방향 정의가 불안정한 경우나 반대 reach로 기준 회전이 정의되지 않는 경우에는 후보를 거부한다.
- 위치·회전 허용오차를 통과한 후보 중 자세 비용이 작은 것을 선택한다. 아주 작은 오차로 등식을 만족한 후보가 있으면 그것을 우선한다. 손목 목표가 이미 맞아 있어도 자세 복원을 시도한다.
- 이전 명령에서 시작하며, 한 solve의 관절 탐색 범위는 각 축 ±20°다. 최대 24 iteration, 시간 예산 40 ms/팔로 계산을 제한한다. 시간 예산은 Python 계산 중 확인하는 기준이며 운영체제 스케줄링을 포함한 hard real-time 보장은 아니다.
- 양팔 중 하나라도 허용 가능한 해를 찾지 못하면 양팔·그리퍼 명령을 유지하고 기존 IK 재시도 상태로 전환한다. 자세/관절 제약을 풀어 실패를 숨기지 않는다.
- 속도 제한, 실제 피드백·ACK 검사는 유지한다. 충돌 검사는 `TELEOP_COLLISION_CHECK=False`일 때 Teleoperation에서 해제하고, Initial pose·Reset pose·수동 조정에서는 유지한다. 속도 제한과 REAL pacing 이후 전송할 관절값에도 팔꿈치 방향 검사를 적용한다. HOME/수동 조정은 기존 검사를 사용한다.

손목 6D 목표와 팔 형상 제한을 동시에 만족하는 해가 없는 목표는 거부될 수 있다. 기존 충돌 장면의 범위와 샘플링 검사 특성도 그대로다.

## 로그와 오프라인 검사

공용 JSONL 로그의 `posture_ik`에 손별 위치·회전 오차, 자세 비용, 팔꿈치 각도(rad), 반복 횟수, 계산 시간, 예산 소진 여부가 기록된다. `posture_reference_captured`는 기준 저장 여부다. `posture_command_guard`는 REAL pacing 후 후보의 자세 검사 결과이며 거절된 시도는 `attempted_posture_ik`에 남는다.

```bash
python -m unittest discover -s tests -p 'test_posture_*.py' -v
python -m unittest discover -s tests -p 'test_*.py'
python scripts/check_workspace.py
python scripts/validate_intent_pipeline.py --output .cache/validation/posture-intent
```

모두 실제 로봇 명령 없이 실행한다. 합성 pipeline 검사는 이상적 관절 추종이며 실제 모터·통신 성능을 검증하는 것은 아니다.
