# 환경 세팅 (lerobot 0.6.1 / Python 3.12)

노트북(teleop·녹화·추론)은 Python 3.12 환경 `ri_motion_v5_py312`를 사용한다. 기존 `ri_motion_v5_env`(Python 3.10, lerobot 0.4.4)에서는 녹화·추론이 동작하지 않는다.

**Ubuntu**

```bash
bash setup_ubuntu_env.sh
```

설치 후 Jupyter 커널을 `ri_motion_v5_py312`로 선택한다.

**Mac (Apple Silicon만 가능, Intel Mac은 불가)**

PyTorch가 Intel Mac용 빌드를 2.2까지만 제공하고 lerobot 0.6.1은 torch 2.7 이상이 필요하다. 레포 루트에서 Claude Code를 열고 다음과 같이 요청한다.

```
setup_ubuntu_env.sh를 참고해서 macOS용 Python 3.12 환경(ri_motion_v5_py312)을 만들어줘.
CUDA torch 대신 기본 torch, LD_LIBRARY_PATH 설정은 빼고,
마지막에 opencv-python GUI 버전 재설치랑 Jupyter 커널 등록까지 해줘.
```

추론 노트북의 `DEVICE = None`은 Mac에서 자동으로 `mps`를 사용한다.

자세한 실행 방법은 [VR_teleoperation/README.md](VR_teleoperation/README.md)를 참고한다.
