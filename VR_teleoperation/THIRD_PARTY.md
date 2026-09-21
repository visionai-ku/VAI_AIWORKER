# Source provenance

이 폴더는 로컬 `FtF_VR/ri_motion_v5_VR` 작업본에서 필요한 파일을 선별한 snapshot이다. 원격 저장소의 특정 commit과 같다고 가정하지 않는다. 파일별 출처와 원본/복사본 SHA-256은 `SOURCE_MANIFEST.json`에 기록한다.

- 공용 런타임 `vendor/ri_motion_v5_package`: 원본 `.gitmodules`에 기록된 출처는 `https://github.com/sjchoi86/ri_motion_v5_package.git`이다. `kinematics`, `utility`, `mujoco_sim`, `qt` 소스와 선택적 native 빌드 소스만 가져왔다. 이 범위의 로컬 snapshot에는 별도의 LICENSE 파일이 없었으며, 이 문서는 새 라이선스를 부여하지 않는다. 기존 소스 주석·저자 표시를 보존했다.
- 프로젝트 코드와 노트북: 로컬 `ri_motion_v5_VR/project/ffw_sg2_vr_teleoperation`에서 가져왔다. 상위 README의 저장소 출처는 `https://github.com/Junha02/ri_motion_junha.git`이다. 로컬에서 진행한 Fixed Quest 수정도 포함되어 있다.
- 로봇 XML·메시: 원본과 함께 있던 Apache 2.0 고지 `xml/ROBOTIS_LICENSE`를 그대로 보존했다. 이 고지를 공용 Python 코드 전체의 라이선스로 확대 해석하지 않는다.

upstream의 pyproject/requirements는 생략한 다른 패키지와 무관한 대형 의존성을 포함해 그대로 복사하지 않았다. v2 requirements에는 현재 경로의 직접 의존성만 기록했다.
