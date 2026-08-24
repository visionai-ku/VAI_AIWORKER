# 02. IK Solver — Null Space로 인한 Elbow 떨림

AI Worker 텔레오퍼레이션 중 **elbow(팔꿈치) 관절이 계속 흔들리는 문제**를, IK의 null space 관점에서 진단하고 수정 계획을 정리한 문서.

작성일: 2026-08-23 · 관련 문서: [00_workspace_overview.md](./00_workspace_overview.md)

---

## 0. 받은 디렉션

> teleoperation할 때 최성준 교수님이 주신 IK solver가 elbow joint가 계속 흔들리는 문제가 있었는데, 그게 IK null space가 있어서 생기는 문제라 (하나의 eef state에 여러 joint의 해가 생겨서) 이전 스텝의 joint 값을 기반으로 제일 적은 joint 움직임을 갖도록 하는 조건을 걸어주면 될 것 같아요. **IK solver 최적화할 때 이전 joint도 optimization에 들어가나 봐야할듯.**

### 확인 결과 (요약)

| 질문 | 답 |
|---|---|
| null space가 존재하는가 | **예.** 7-DOF 팔 + 6-DOF 손목 구속 → **1-DOF 잉여** = elbow swivel(팔꿈치 선회각) |
| null-space 제어 항이 있는가 | **예.** `k_null`, `q_home_rev_pri` 로 구현되어 있고 텔레오퍼레이션에서 **켜져 있다** |
| **이전 스텝 joint가 최적화에 들어가는가** | **아니오.** 당기는 대상은 `q_home` 이고, 이 값은 **솔버 생성 시점에 한 번 캡처된 고정 스냅샷**이다 |

**즉 디렉션의 진단은 정확하며, 필요한 수정은 "없는 기능을 새로 만드는 것"이 아니라 "이미 있는 null-space 항이 당기는 기준점을 `q_home` → `q_prev` 로 바꾸는 것"이다.** §4-A가 최소 수정안이다.

---

## 1. 관련 코드 위치

| 역할 | 파일 | 위치 |
|---|---|---|
| IK 솔버 본체 (Python) | `ri_motion_v5_package/mujoco_sim/mujoco_ik.py` | `BodyPositionIKSolverRevPriBase` (L1239~) |
| null-space 계산 | 〃 | `compute_dq_rev_pri()` L1887~2054, **step 6** L2008~ |
| null-space 투영 | 〃 | `project_dq_null_damped()` L128, `mask_dq_null_to_task_relevant_joints()` L184 |
| DLS 해 | 〃 | `solve_dls()` L49 |
| 네이티브 브리지 | `mujoco_sim/_native.py` | `_body_position_solve_ik_from_backend()` L379~, **`q_home_rev_pri` 읽는 곳 L502** |
| IK 솔버 본체 (C++) | `mujoco_sim/cpp/ik.cpp` | null-space 블록 **L462~482** |
| SG2 솔버 생성 | `project/ffw_sg2_vr_teleoperation/package/utils.py` | `create_sg2_collision_aware_arm_ik_solvers()` L1270 |
| SG2 매 스텝 solve | 〃 | `solve_sg2_collision_aware_wrist_ik()` L1422 |
| 손목 타겟 생성 | 〃 | `add_sg2_wrist_ik_coordinate_target()` L1223 |
| upper-arm yaw 완화 | 〃 | `relax_sg2_upper_arm_yaw_joints()` L1362 |
| 메인 루프 | `.../real_notebook/real_MetaQuest3_sg2_teleop.ipynb` | cell 6 (설정), cell 7 (루프) |

---

## 2. 현재 IK 구조 해부

### 2.1 Task 정의 — 왜 정확히 1-DOF가 남는가

`add_sg2_wrist_ik_coordinate_target()` 는 손목 하나당 **좌표축 3개의 끝점**(x/y/z, `axis_width = 0.10 m`)을 위치 타겟으로 추가한다.

```
손목 1개 = 3 point target × 3축(x,y,z 성분) = 9 task rows
```

**행이 9개라고 해서 구속이 9개는 아니다.** 세 점은 모두 같은 강체(`arm_?_link7`)에 붙어 있고, 강체 위 점들의 위치는 그 강체의 6-DOF 자세로 완전히 결정된다. 따라서 야코비안 `J ∈ ℝ^(9×7)` 의 **랭크는 6**이고 나머지 3행은 중복 정보다. 위치 타겟만으로 자세까지 구속하는 영리한 트릭이지만, 잉여 자유도를 없애주지는 못한다.

null space 차원을 정하는 것은 행 수가 아니라 **랭크**다:

```
dim(ker J) = 7 (관절) − rank(J) = 7 − 6 = 1
```

이 1-DOF가 **elbow swivel(arm angle)** 이다.

**SG2 팔의 실제 구조** (`xml/ffw_sg2.xml` 의 default class에서 확인):

| 관절 | axis | range | 역할 |
|---|---|---|---|
| joint1 / joint2 / joint3 | y / x / z | −3.14~3.14, −3.14~0, −3.14~3.14 | **어깨 3-DOF** |
| **joint4** | y | **−2.9361 ~ 1.0786** | **팔꿈치 굴곡** (비대칭 범위 = 한 방향 굽힘) |
| joint5 / joint6 / joint7 | z / y / x | — | **손목 3-DOF** |

직관적으로는 이렇게 이해하면 된다 — 어깨 중심 `S` 는 몸통에 붙어 고정, 손목 중심 `W` 는 목표가 정하므로 고정, 상완·전완 길이는 상수. 그러면 팔꿈치 `E` 는 **어깨–손목 축을 회전축으로 도는 곡선 위 어디에나** 놓일 수 있고, 그동안 손목 자세는 완전히 동일하다. 디렉션의 "하나의 eef state에 여러 joint의 해"가 정확히 이것.

> 사람 팔로 재현 가능: 손바닥을 책상에 붙여 고정한 채 팔꿈치만 위아래로 돌려보면, 손은 그대로인데 팔꿈치가 움직인다.

> ⚠️ **이 그림은 이상화 모델이다.** 교과서의 "구 두 개의 교집합 = 정확한 원" 은 어깨 3축이 한 점에서 만나고 링크가 직선일 때 성립한다. SG2는 그렇지 않다 — 어깨는 joint2→joint3 사이에 `0 0 -0.165` 오프셋이 있어 세 축이 한 점에서 만나지 않고, 팔꿈치 주변에는 link4 `+0.041004` / link5 `−0.041004` 의 횡방향 지그재그가 있다. 따라서 **실제 swivel 궤적은 정확한 원이 아니라 변형된 폐곡선**이다.
>
> 다만 **차원이 1이라는 결론은 위의 랭크 논증에서 나오므로 이 이상화와 무관하게 정확하다.** 해석적 swivel 각 해를 유도하려 한다면 이 오프셋들을 반드시 반영해야 한다.

**어느 관절이 흔들리는가** — swivel 방향으로 실제 움직이는 것은 **어깨 3관절(joint1/2/3)과 이를 보상하는 손목 3관절(joint5/6/7)** 이고, 그중 축이 상완 방향(z)인 **joint3가 지배적**이다.

**`joint4`(팔꿈치 굴곡)는 null space에서 자유롭지 않다.** 어깨–손목 거리가 코사인 법칙으로 굴곡각을 유일하게 결정하기 때문이다. 손목 목표가 정해지면 joint4도 정해진다. 억제 대상을 고를 때 joint4를 넣으면 안 된다.

**관절 구성별 null 차원** — 실제 솔버는 양팔 14 관절을 하나의 솔버에서 동시에 푼다 (`SG2_COLLISION_IK_JOINT_NAMES = arm_l 7 + arm_r 7`).

| 상황 | 관절 | rank(J) | dim(ker J) |
|---|---|---|---|
| 팔 하나 단독 | 7 | 6 | **1** |
| 양팔 솔버, 양손 attach | 14 | 12 | **2** (팔당 1) |
| 양팔 솔버, 한 손만 attach | 14 | 6 | **8** |

마지막 경우 8개 중 7개는 노는 팔 전체이고, `task_null_mask_flag=True` 가 야코비안 열이 0인 그 관절들의 null-space 이동을 죽인다. 실질적으로 남는 것은 활성 팔의 swivel **1개**다.

> 특이점(팔을 곧게 편 자세) 근처에서는 `rank(J)` 가 6 아래로 떨어져 **null space가 더 커진다.** 원인 (2) `dls_damping` 과 직결되는 지점.

### 2.2 null-space 항의 실제 수식

`compute_dq_rev_pri()` step 6 (Python) / `ik.cpp` L462~482 (C++), 동일한 로직:

```python
q_err_use = self.q_home_rev_pri[idxs] - q_rev_pri_curr[idxs]   # ← 기준점은 q_home
dq_null   = self.k_null * q_err_use
if task_null_mask_flag:
    dq_null = mask_dq_null_to_task_relevant_joints(J_use_valid, dq_null)
dq_add    = project_dq_null_damped(J_use=J_use_valid, dq_null=dq_null, lam=lam_valid)
dq_use[valid_local] += dq_add * steps_use[valid_local]
```

`project_dq_null_damped()` 는 `dq_null` 에서 task 공간 성분을 빼서 **task를 방해하지 않는 성분만** 남긴다:

```
dq_add = dq_null − (JᵀJ + λI)⁻¹ Jᵀ J dq_null
```

구조 자체는 교과서적으로 옳다. **문제는 `q_home` 이 무엇이냐다.**

### 2.3 `q_home` 이 무엇인가 — 핵심 발견

`utils.py:1270` `create_sg2_collision_aware_arm_ik_solvers()`:

```python
tracking_solver = body_ik_solver_cls(
    ...
    k_null         = 0.5,
    q_home_rev_pri = env.get_qpos(joint_names=env.rev_pri_joint_names),   # ← 생성 시점 1회
)
```

- 이 값은 **노트북 cell 6이 실행되는 순간의 관절값**으로 고정된다.
- 루프 안에서 갱신되지 않는다. 매 스텝 null space는 **"실행 시작 시 자세"로 팔꿈치를 당긴다.**
- 사용자가 팔을 움직이면 현재 자세와 `q_home` 의 차이가 커지고, `dq_null` 도 계속 커진다. 그런데 그 방향은 지금 손목 목표와 아무 상관이 없다.

**결과**: task 항은 손목을 따라가려 하고, null-space 항은 팔꿈치를 시작 자세로 되돌리려 한다. 두 힘이 매 프레임 경쟁하면서 팔꿈치가 진동한다.

---

## 3. 떨림의 원인 후보 4가지

우선순위 순.

### (1) `q_home` 이 고정 스냅샷 — **주범**

§2.3. null-space가 "적게 움직이기"가 아니라 "시작 자세로 돌아가기"를 하고 있다. 디렉션이 지적한 바로 그 지점.

### (2) `dls_damping` 미지정 — 유력한 공범

두 솔버 생성 함수를 비교하면:

| 함수 | `dls_damping` |
|---|---|
| `create_sg2_arm_ik_solvers()` (팔별 개별 솔버) | `1e-6 * ones(n)`, **마지막 관절만 `1e-4`** |
| `create_sg2_collision_aware_arm_ik_solvers()` (실제 사용) | **인자를 아예 넘기지 않음 → 기본값 `1e-6` 스칼라** |

`1e-6` 는 매우 낮은 감쇠다. 특이점(팔을 곧게 폈을 때) 근처에서 `solve_dls()` 가 큰 `dq` 를 뱉는다. **실제 텔레오퍼레이션에 쓰는 솔버만 감쇠 설정이 빠져 있다** — 의도된 것인지 누락인지 확인이 필요하다.

### (3) `idx_best = argmin(ik_err_array)` — 이산 선택 점프

`solve_ik()` / `solve_ik_bind()` 는 8틱을 돌린 뒤 **task error가 최소인 틱의 자세**를 고른다.

```python
idx_best       = int(np.argmin(ik_err_array))
qpos_used_best = qpos_full_best[...]
```

null space 때문에 서로 다른 틱이 거의 같은 error를 가지면, 프레임마다 `idx_best` 가 3 → 7 → 4 처럼 튄다. **task error는 같지만 관절 자세는 다르다** → 그대로 관절 점프가 된다. null space가 있는 한 이 선택은 본질적으로 불안정하다.

### (4) `relax_sg2_upper_arm_yaw_joints()` 와의 충돌

`solve_sg2_collision_aware_wrist_ik()` 는 tracking 모드에서 **IK를 풀기 직전** 매번 이걸 호출한다:

```python
out["upper_arm_yaw_info"] = relax_sg2_upper_arm_yaw_joints(env)
```

내용은 `arm_r_joint3` / `arm_l_joint3`(= upper-arm yaw, 팔꿈치를 선회시키는 관절)을 **0을 향해 최대 1°/스텝씩 강제 이동**시키는 것이다.

```python
SG2_UPPER_ARM_YAW_JOINT_NAMES = ("arm_r_joint3", "arm_l_joint3")
q_next = q_prev + np.clip(q_home - q_prev, -max_step_rad, +max_step_rad)  # max_step_rad = 1°
```

**대상 관절이 `joint3`(상완 요축)인 것이 결정적 단서다.** joint3를 돌리면 팔꿈치가 어깨–손목 축 주위로 도는 방향, 즉 **§2.1의 swivel 방향과 거의 일치**한다. 다시 말해 이 함수는 null space 자유도를 직접 붙잡으려던 시도다 — **동일한 elbow 문제에 대한 기존 임시방편**으로 보인다.

20 Hz × 1° = 초당 20°의 지속적 당김이 null-space 항과 별개로 작동하므로, 둘이 서로 다른 목표를 향할 때 떨림을 만든다.

> §4의 수정을 넣을 때 **이 함수를 끄거나 약화시키는 것을 함께 실험해야 한다.** 두 메커니즘을 동시에 켜두면 어느 쪽 효과인지 구분할 수 없다.

---

## 4. 수정안

### A. 최소 수정 — `q_home` 을 이전 스텝 값으로 (권장 시작점)

`_native.py:502` 가 **매 호출마다 `solver.q_home_rev_pri` 를 새로 읽는다.**

```python
q_home_rev_pri = (... if solver.q_home_rev_pri is None
                  else np.ascontiguousarray(solver.q_home_rev_pri, ...))
```

Python 경로(`compute_dq_rev_pri`)도 `self.q_home_rev_pri` 를 매 틱 참조한다. 따라서 **솔버 속성만 갱신하면 Python·C++ 양쪽 경로에 그대로 반영된다.** 솔버 코드를 건드릴 필요가 없다.

수식으로는 null-space 목표가 이렇게 바뀐다:

```
before:  dq_null = k_null · (q_home_at_startup − q_curr)      "시작 자세로 돌아가라"
after:   dq_null = k_null · (q_prev            − q_curr)      "직전 자세에서 벗어나지 마라"
```

이것이 곧 디렉션의 **"이전 스텝의 joint 값을 기반으로 제일 적은 joint 움직임"** 이다.

**수정 위치**: `utils.py` `solve_sg2_collision_aware_wrist_ik()` — IK를 풀기 전에 기준점을 갱신.

```python
# solve_sg2_collision_aware_wrist_ik(...) 안, tracking_solver.reset_buffers(env=env) 직전
tracking_solver.q_home_rev_pri = env.get_qpos(joint_names=env.rev_pri_joint_names)
```

> 코드 컨벤션상 노트북에는 이 배관이 보이면 안 된다. `utils.py` 안에 넣고, 동작 선택은 의미 있는 파라미터 하나로 노출한다:
> `solve_sg2_collision_aware_wrist_ik(..., null_reference="previous")` (`"previous"` / `"home"`)

**장점**: 3줄, 되돌리기 쉬움, 솔버 패키지(submodule) 미수정 → 충돌 없음.
**한계**: 관절별 가중치를 줄 수 없다(`k_null` 이 스칼라). 팔꿈치만 강하게 잡고 손목은 자유롭게 두는 건 불가능.

### B. 관절별 가중 — `k_null` 을 벡터로

swivel 지배 관절(주로 `arm_?_joint3`)만 강하게 억제하고 싶다면 `k_null` 이 관절별 벡터여야 한다. 현재는 스칼라 하나가 Python과 C++ 양쪽에 그대로 쓰인다 (`ik.cpp:472` `k_null*(q_home[rev_idx] - q_rev[rev_idx])`).

**두 가지 경로:**

| 경로 | 수정 범위 | 비고 |
|---|---|---|
| **B-1. 솔버 패키지 수정** | `mujoco_ik.py` + `_native.py` + `cpp/ik.cpp` + 재컴파일 | 정공법이지만 **submodule(`sjchoi86/ri_motion_v5_package`) 수정** → 업스트림과 분기. 교수님 쪽과 조율 필요 |
| **B-2. 프로젝트 측 우회** | `utils.py` 만 | `q_home` 을 관절별로 다르게 섞어 가중 효과를 흉내: 억제하고 싶은 관절은 `q_prev`, 자유롭게 둘 관절은 현재값(=차이 0 → null 항 없음) |

**B-2 스케치** — `k_null` 은 그대로 두고 기준점만 관절별로 구성:

```
q_ref[j] = q_prev[j]   (억제 대상: arm_?_joint3 — swivel 지배 관절. 필요시 joint1/2 추가)
q_ref[j] = q_curr[j]   (자유 대상: 나머지 → q_err = 0 이 되어 null 항이 사라짐)
```

> 억제 대상에 **`joint4`(팔꿈치 굴곡)는 넣지 않는다.** §2.1대로 joint4는 손목 목표가 유일하게 결정하므로 null space 자유도가 아니다. 여기에 억제를 걸면 task 추종만 방해한다.

효과적으로 관절별 on/off 가중이 된다. 연속 가중이 필요하면 `q_ref = q_curr + w_j·(q_prev − q_curr)` 로 `w_j ∈ [0,1]` 을 섞는다. **패키지 수정 없이 관절별 제어가 가능하므로 B-1보다 먼저 시도할 것.**

### C. `dls_damping` 명시

`create_sg2_collision_aware_arm_ik_solvers()` 에 `create_sg2_arm_ik_solvers()` 와 같은 감쇠 프로파일을 넣는다.

```python
dls_damping = 1e-6 * np.ones(len(joints_use))
dls_damping[6]  = 1e-4   # arm_l_joint7 (손목 롤)
dls_damping[13] = 1e-4   # arm_r_joint7
```

인덱스는 `joints_use` 순서(`SG2_COLLISION_IK_JOINT_NAMES` = arm_l 1~7 + arm_r 1~7)를 따른다. 특이점 근처 폭주를 줄이는 독립적 개선이라 **A와 별개로 넣어도 된다.**

### D. `idx_best` 선택 안정화 (선택)

원인 (3)에 대한 대응. `argmin` 대신 "error가 임계 이내면 이전 자세에 가장 가까운 틱"을 고르는 방식.

```
후보 = { tick : ik_err[tick] < ik_err_best · (1 + tol) }
선택 = argmin_{tick ∈ 후보} ‖q[tick] − q_prev‖
```

**솔버 패키지 수정이 필요**하므로(B-1과 동급) 우선순위는 낮다. A~C를 넣고도 틱 점프가 남을 때 검토한다.

### E. `relax_sg2_upper_arm_yaw_joints()` 정리

A가 제대로 동작하면 이 임시방편은 불필요해지거나 오히려 방해가 된다.

- 1단계: `max_step_rad` 를 0으로 (사실상 비활성) 하고 A만으로 떨림이 잡히는지 확인
- 2단계: 잡히면 호출 자체를 제거 또는 옵션화
- 3단계: 안 잡히면 A와 병행하되 `max_step_rad` 를 0.1~0.3° 수준으로 낮춰 재실험

---

## 5. 수정 대상 파일 총괄

| # | 파일 | 변경 | 범위 | 우선순위 |
|---|---|---|---|---|
| 1 | `project/ffw_sg2_vr_teleoperation/package/utils.py` | `solve_sg2_collision_aware_wrist_ik()` 에 `null_reference` 파라미터 + `q_home_rev_pri` 갱신 (§4-A) | 프로젝트 | **1** |
| 2 | 〃 | `create_sg2_collision_aware_arm_ik_solvers()` 에 `dls_damping` 명시 (§4-C) | 프로젝트 | **2** |
| 3 | 〃 | `relax_sg2_upper_arm_yaw_joints()` 호출 옵션화 (§4-E) | 프로젝트 | **3** |
| 4 | 〃 | 관절별 가중 기준점 (§4-B-2) | 프로젝트 | 4 |
| 5 | `.../sim_notebook/sim_MetaQuest3_sg2_teleop.ipynb` | 새 파라미터 노출 · **검증은 여기서 먼저** | 노트북 | 1과 동시 |
| 6 | `.../real_notebook/real_MetaQuest3_sg2_teleop.ipynb` | 동일 | 노트북 | 시뮬 통과 후 |
| 7 | `.../sim_notebook/sim_interactive_marker_wrist_ik_test.ipynb` | **재현 테스트베드로 활용** (§6) | 노트북 | 0 — 가장 먼저 |
| 8 | `ri_motion_v5_package/mujoco_sim/mujoco_ik.py` · `_native.py` · `cpp/ik.cpp` | `k_null` 벡터화 (§4-B-1), `idx_best` 안정화 (§4-D) | **submodule** | 5 — 최후 |

> ⚠️ 8번은 **`sjchoi86/ri_motion_v5_package` submodule** 이다. 여기를 고치면 업스트림과 분기되고 `git submodule update --remote` 시 충돌한다. 1~4로 해결되면 손대지 않는 편이 좋고, 필요하다면 교수님 쪽에 이슈/PR로 올리는 경로를 먼저 검토할 것.

---

## 6. 검증 계획

### 6.1 재현 환경 — 실기 없이 시작

**`sim_notebook/sim_interactive_marker_wrist_ik_test.ipynb` 를 먼저 쓴다.**

이 노트북은 기즈모 두 개로 좌우 손목 타겟을 직접 끌 수 있다. Quest도 로봇도 필요 없고, **같은 타겟을 반복 재현**할 수 있어 A/B 비교에 이상적이다.

> 단, 이 노트북은 충돌회피가 아닌 **일반 DLS 솔버**를 쓴다 (cell 3 주석: "One small (non-collision-aware) DLS IK solver per arm"). 실제 텔레오퍼레이션 솔버와 조건이 다르므로, 여기서 원인을 좁힌 뒤 `sim_MetaQuest3_sg2_teleop.ipynb` 로 옮겨 재확인한다.

### 6.2 정량 지표

떨림을 눈이 아니라 숫자로 본다. 손목 타겟을 **고정**한 채 IK를 계속 돌리면서:

| 지표 | 정의 | 기대 |
|---|---|---|
| 관절 변화량 | `‖q_t − q_{t−1}‖` (관절별로도) | 수정 후 크게 감소. **특히 어깨 joint1/2/3** |
| Elbow 선회각 | `arm_?_joint3` 시계열 | 진동 진폭 감소 |
| Task error | `ik_info["ik_err_best"]` | **악화되지 않아야 함** — null-space 항은 task를 방해하면 안 됨 |
| `idx_best` 분포 | `ik_info["idx_best"]` 히스토그램 | 튀는 정도 감소 (원인 3 확인용) |
| 반복 재현성 | 같은 타겟 → 같은 자세가 나오는가 | null space가 잡히면 재현성이 생김 |

**핵심 판정**: 관절 변화량은 줄고 task error는 유지. 둘 다 나빠지면 `k_null` 이 과도한 것이다.

### 6.3 실험 파라미터

| 파라미터 | 현재 | 스윕 범위 | 의미 |
|---|---|---|---|
| `k_null` | 0.5 | 0.0 / 0.25 / 0.5 / 1.0 / 2.0 | 0 = null 제어 없음(기준선), 크면 팔꿈치 고착 |
| `null_reference` | (`home` 상당) | `home` / `previous` | §4-A의 A/B |
| `dls_damping` | 1e-6 | 1e-6 / 1e-5 / 1e-4 | 특이점 근처 안정성 |
| `max_ik_tick` | 8 | 4 / 8 / 16 | 틱 수와 `idx_best` 점프의 관계 |
| `relax_...max_step_rad` | 1.0° | 0° / 0.3° / 1.0° | §4-E |
| `ik_stepsize_rev` | 8° | 4° / 8° | 스텝이 크면 오버슛 |

**한 번에 하나씩.** 먼저 `k_null=0` 으로 두고 기준선을 잡으면 null-space 항이 떨림의 원인인지 아닌지가 즉시 갈린다.

### 6.4 순서

```text
[ ] 0. sim_interactive_marker 에서 손목 타겟 고정 → q_t 로깅, 떨림 재현 확인
[ ] 1. k_null = 0 으로 기준선 측정  → 떨림이 사라지면 null-space 항이 원인 확정
[ ] 2. relax_upper_arm_yaw max_step_rad = 0 → 이 임시방편의 기여도 분리
[ ] 3. §4-A 적용 (q_home ← q_prev) → 지표 재측정
[ ] 4. §4-C 적용 (dls_damping) → 지표 재측정
[ ] 5. k_null 스윕으로 최적점 탐색
[ ] 6. sim_MetaQuest3_sg2_teleop 로 이전, Quest 입력으로 재확인
[ ] 7. 실기 적용 (E-stop 대기 필수 — 00 문서 §7.6)
[ ] 8. 전후 그래프를 mkdown/ 에 기록
```

---

## 7. 주의 · 트레이드오프

- **null space를 과하게 잡으면 팔꿈치가 굳는다.** `k_null` 이 크면 팔꿈치가 이전 자세에 붙박여, 충돌 회피나 관절 한계 회피에 필요한 재배치를 못 한다. 특히 `solve_sg2_collision_aware_wrist_ik()` 의 depenetration 모드는 자세 재배치가 목적이므로 **tracking 모드에만** 강한 억제를 적용하는 편이 안전하다 (depenetration 솔버는 이미 `k_null=0.0`).
- **`q_prev` 기준은 드리프트를 만들 수 있다.** "직전 자세 유지"만 하면 팔꿈치가 어색한 자세에 갇혀도 빠져나올 유인이 없다. 필요하면 `q_ref = α·q_prev + (1−α)·q_home` 으로 두 기준을 섞어 소량의 home 복원력을 남긴다.
- **gain 문제와 구분할 것.** 이 문서는 **IK 해 자체가 흔들리는 문제**(명령이 진동)를 다룬다. 명령은 매끈한데 로봇의 추종이 떨리는 것은 DYNAMIXEL gain·프로파일 쪽 문제로, 별개의 조사가 필요하다. **`q_use` 시계열을 보면 즉시 구분된다** — 시뮬만 돌려도 IK 출력이 떨리면 이 문서의 문제, 시뮬은 매끈한데 실기만 떨리면 gain 쪽이다.
- **submodule 경계를 넘지 말 것.** §5의 1~4는 프로젝트 폴더 안이라 자유롭게 고칠 수 있다. 8번(패키지/C++)은 다른 사람들이 공유하는 코드다.

---

## 8. 미해결 / 확인 필요

- `create_sg2_collision_aware_arm_ik_solvers()` 의 `dls_damping` 누락이 **의도인지 실수인지** — 원 저자 확인 필요
- `relax_sg2_upper_arm_yaw_joints()` 가 왜 도입되었는지 — 같은 elbow 문제 대응이라면 §4-A로 대체 가능
- 떨림이 **양팔 동시 attach 시에만** 심한지 (한 손만 attach면 원시 null 차원이 8로 커지지만 `task_null_mask` 가 노는 팔을 죽이므로 실질 1-DOF는 동일 → 두 경우의 떨림 정도가 비슷해야 정상. 다르다면 마스킹이 기대대로 동작하지 않는 것)
- 특이점 근처(팔을 곧게 편 자세)에서만 심한지 → 그렇다면 원인 (2) `dls_damping` 비중이 큼
- 시뮬에서도 재현되는지 → **재현되면 순수 IK 문제로 확정**, 실기에서만이면 gain/통신 요인이 섞인 것
- `ros2_ws/src/cyclo_control` 에 **OSQP 기반 QP 최적화 컨트롤러**(`cyclo_motion_controller_core/{kinematics,controllers,optimization}`, `osqp_eigen_vendor`)가 이미 들어 있다. Dockerfile에 `motion_controller_hand='... controller_type:=vr hand:=true'` alias도 있다. null space를 QP 제약으로 다루는 경로가 로봇 쪽에 준비되어 있을 수 있으므로, MuJoCo 솔버를 손대기 전에 **이 패키지가 무엇을 제공하는지 먼저 확인할 것**
