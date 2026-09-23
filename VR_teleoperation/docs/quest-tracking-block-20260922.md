# 17:35 Quest 입력 차단 확인

대상: `.cache/fixed_quest/recordings/20260922-173544-438204-intent-real.jsonl`.

- 제어 로그 21,306개 모두 `WAIT_INPUT`이며, 전 구간 `accept_emulated=True`였다.
- 주된 결과는 `tracking_lost_left` 14,825개, `no_frame` 5,111개,
  `stale_pose` 1,093개, `session_not_visible` 256개였다.
- 왼손 진단 중 9,581개는 visible/정상 viewer 상태에서 컨트롤러 장치가 없었다.
- 4,667개는 컨트롤러 추정 위치가 있었지만 viewer도 emulated여서
  `blocked_by=viewer_pose`였다. 클라이언트가 양손을 빈 입력으로 전송한다.
  PC의 컨트롤러 추정 위치 허용 설정으로는 이 조건을 해제하지 않는다.
- 위 수치는 고유 XR 프레임 수가 아니라 PC 제어 로그 수다.
- 새로 추가된 `diagnostics.transport`가 없어 이전 Quest 페이지가 계속 실행된
  정황이 있다. PC 수신의 `frame.transport`와는 다른 필드다.

이번 수정은 PC/Quest 표시를 바로잡는다. 헤드셋 위치 추적 차단과 XR 비활성을
우선 표시하고, 컨트롤러 추정 위치의 사용 여부는 PC 설정에 따라 표현한다.
추적 허용 조건이나 로봇 동작 코드는 변경하지 않았다.

적용/복구 확인: PC 실행 종료 → 커널 재시작 및 설정부터 실행 → Quest XR 종료,
입력 페이지 새로고침 및 XR 재시작. 헤드셋을 착용하고 양손 컨트롤러가 인식되는지,
viewer가 위치 유효인지 확인한다. 양손 grip을 놓은 상태에서 입력 복구를 확인한 뒤
다시 조작한다. 지속되면 Quest 자체의 위치/컨트롤러 추적 상태를 확인해야 한다.
로그만으로 Quest 추적 손실의 하드웨어/환경 원인은 확정할 수 없다.

검증: 기록 프레임의 8가지 진단 조합으로 PC 표시 확인, 기존 JavaScript 송신
검증 통과(ES2020+ V8 호스트), Python 구문 검사 및 diff 공백 검사 통과.
로봇 연결/동작 및 Quest 하드웨어 복구 확인은 수행하지 않았다.
