# 03. `null_reference` 패치 — §4-A 적용 기록

[02_ik_solver.md](./02_ik_solver.md) §4-A(null-space 기준점을 `q_home` → `q_prev` 로 변경)를 실제 코드에 적용한 내역.

작성일: 2026-08-23 · 대상 파일: `ri_motion_v5_VR/project/ffw_sg2_vr_teleoperation/package/utils_new.py`
원본 `utils.py` (1,954줄)는 수정하지 않음. 사본에 **추가만 33줄**, 삭제·변경 0줄.

---

## 핵심

```
before:  dq_null = k_null · (q_home_at_startup − q_curr)   "시작 자세로 가라"
after :  dq_null = k_null · (q_prev            − q_curr)   "직전 자세를 유지하라"
```

솔버는 매 solve 틱마다 `solver.q_home_rev_pri` 를 새로 읽는다 (Python: `compute_dq_rev_pri()` / C++: `_native.py:502`). 따라서 **속성만 재바인딩하면 양쪽 경로에 반영**되고, submodule(`ri_motion_v5_package`) 수정이 필요 없다.

---

## 변경 6곳

### ① L94 — 모드 상수

```python
SG2_NULL_REFERENCE_MODES = ("previous", "home")
```

### ② L1365 — 신규 함수

```python
def set_sg2_null_space_reference(ik_solver, env, null_reference="previous"):
    if null_reference not in SG2_NULL_REFERENCE_MODES:
        raise ValueError(...)
    if null_reference == "home":
        return None                                   # 기존 동작 보존

    q_prev_rev_pri = env.get_qpos(joint_names=env.rev_pri_joint_names)
    ik_solver.q_home_rev_pri = q_prev_rev_pri          # ← 기준점 재바인딩
    return q_prev_rev_pri
```

### ③ L1460 — 시그니처

`solve_sg2_collision_aware_wrist_ik(...)` 에 `null_reference="previous"` 추가.

### ④ L1493 / L1517 — 두 분기에서 IK 직전 호출

```python
# tracking 분기 (L1493)
set_sg2_null_space_reference(tracking_solver, env, null_reference)
out["upper_arm_yaw_info"] = relax_sg2_upper_arm_yaw_joints(env)
tracking_solver.reset_buffers(env=env)

# depenetration 분기 (L1517)
set_sg2_null_space_reference(depenetration_solver, env, null_reference)
depenetration_solver.reset_buffers(env=env)
```

### ⑤ L1482 / L1361 — 결과 딕셔너리

`out["null_reference"] = null_reference`, `make_idle_collision_ik_result()` 에도 `"null_reference": None` 으로 스키마 정렬. 렌더 오버레이에 현재 모드를 표시할 때 사용 가능.

### ⑥ L1910 / L1970 — `__all__` 등록

`SG2_NULL_REFERENCE_MODES`, `set_sg2_null_space_reference`.

---

## 설계 판단

**호출을 `relax_sg2_upper_arm_yaw_joints()` 앞에 배치했다.** relax가 `env.forward()` 로 `joint3` 를 1°/틱 밀어놓기 때문에, 뒤에 두면 null 기준점이 이미 밀린 자세가 된다. 앞에 둬야 `q_prev` 가 **순수한 직전 IK 해**가 되고, [02 문서](./02_ik_solver.md) §4-E(relax 끄기) 실험에서 두 메커니즘이 독립적으로 관측된다.

**기본값을 `"previous"` 로 했다.** 이 파일을 로드하면 수정이 켜진 상태다. A/B 비교는 키워드 하나로:

```python
collision_ik_result = solve_sg2_collision_aware_wrist_ik(
    ...,
    null_reference = "home",     # 기존 동작 재현
)
```

---

## ⚠️ 지금 상태로는 로드되지 않는다

`init_project.py` 가 `importlib.import_module("utils")` 로 **`utils` 만** 불러온다. `utils_new.py` 는 파일만 존재할 뿐 어디서도 import되지 않는다. 활성화하려면 둘 중 하나:

- **(a)** `utils.py` 를 백업하고 `utils_new.py` 를 그 이름으로 덮어쓰기
- **(b)** `init_project.py` 의 `"utils"` → `"utils_new"` 로 변경 (`sys.modules.pop("utils", None)` 도 함께)

---

## 미포함 / 검증 상태

**미포함** — [02 문서](./02_ik_solver.md) 의 나머지 항목은 반영하지 않았다.

| 항목 | 내용 | 비고 |
|---|---|---|
| §4-B | 관절별 가중 (`k_null` 벡터화 또는 기준점 혼합) | |
| §4-C | `create_sg2_collision_aware_arm_ik_solvers()` 에 `dls_damping` 명시 | **A와 독립적이고 한 줄짜리 — 같이 넣을 가치 있음** |
| §4-D | `idx_best` 선택 안정화 | submodule 수정 필요 |
| §4-E | `relax_sg2_upper_arm_yaw_joints()` 비활성화 | A 검증 시 함께 실험해야 함 |

**검증** — `ast` 구문 검사 통과, `__all__` 96개 항목 전수 정의 확인. **import·실행 테스트는 미실시** (작업 환경에 MuJoCo 없음). 시뮬 노트북에서 먼저 확인할 것.
