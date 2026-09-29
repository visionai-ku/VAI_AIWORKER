# 환경 세팅 (lerobot 0.6.1 / Python 3.12)

노트북(teleop·녹화·추론)은 Python 3.12 환경 `ri_motion_v5_py312`를 사용한다. 기존 `ri_motion_v5_env`(Python 3.10, lerobot 0.4.4)에서는 녹화·추론이 동작하지 않는다.

**Ubuntu**

```bash
bash setup_ubuntu_env.sh
```

설치 후 Jupyter 커널을 `ri_motion_v5_py312`로 선택한다.

**Mac (Apple Silicon만 가능, Intel Mac은 불가)**

검증한 환경 버전은 [`environments/ri_motion_v5_py312.macos-arm64.yml`](environments/ri_motion_v5_py312.macos-arm64.yml)에 기록되어 있다. [복원 안내](environments/README.md)에 따라 환경을 만든 뒤 OpenCV GUI 재설치와 Jupyter 커널 등록을 진행한다.

```bash
conda env create -f environments/ri_motion_v5_py312.macos-arm64.yml
conda activate ri_motion_v5_py312
python -m pip install --force-reinstall --no-deps opencv-python==4.12.0.88
python -m ipykernel install --user --name ri_motion_v5_py312 --display-name ri_motion_v5_py312
```

Mac에서는 CUDA wheel이나 Linux용 `LD_LIBRARY_PATH` 설정을 사용하지 않는다. 이미 검증한 같은 환경을 사용하는 경우 환경 생성은 생략한다. 추론 노트북의 `DEVICE`로 `cpu` 또는 `mps`를 지정할 수 있다.

자세한 실행 방법은 [VR_teleoperation/README.md](VR_teleoperation/README.md)를 참고한다.
