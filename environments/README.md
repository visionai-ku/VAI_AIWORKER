# Python 3.12 / LeRobot 0.6.1 환경

[`ri_motion_v5_py312.macos-arm64.yml`](ri_motion_v5_py312.macos-arm64.yml)은
2026-09-29에 설치하고 확인한 `ri_motion_v5_py312` conda 환경의 버전 목록이다.

- 대상: macOS Apple Silicon (`osx-arm64`)
- Python 3.12.14, `lerobot[dataset]` 0.6.1, PyArrow 25.0.1
- PyTorch 2.11.0, TorchVision 0.26.0, TorchCodec 0.11.1, FFmpeg 8.1.2
- JupyterLab 4.5.1, ipykernel 6.29.5

Conda와 pip의 직접·간접 의존성 버전을 함께 기록했다. 개인 설치 경로와
커널의 절대 경로는 포함하지 않았다. 빌드 번호와 배포 파일 해시는 고정하지
않았으므로 동일 바이너리를 보장하는 잠금 파일은 아니다. Ubuntu 또는 Intel Mac용
설정으로 사용하지 않는다.

## 복원

Conda가 설치된 macOS Apple Silicon에서 저장소 루트를 기준으로 실행한다.
앞서 검증한 `ri_motion_v5_py312` 환경을 그대로 사용하는 경우 생성 명령은 생략한다.

```bash
conda env create -f environments/ri_motion_v5_py312.macos-arm64.yml
conda activate ri_motion_v5_py312

# LeRobot의 headless OpenCV와 cv2 파일이 겹치므로 GUI 버전을 마지막에 설치한다.
python -m pip install --force-reinstall --no-deps opencv-python==4.12.0.88

# 현재 컴퓨터의 Python 경로를 사용해 커널을 등록한다.
python -m ipykernel install --user \
  --name ri_motion_v5_py312 --display-name ri_motion_v5_py312
```

VS Code 또는 Jupyter에서 **ri_motion_v5_py312** 커널을 선택한다.
Linux용 `LD_LIBRARY_PATH` 설정은 이 macOS 환경에 적용하지 않는다.

## 설치 확인

```bash
python -m pip check
python -c "import sys, pyarrow; from importlib.metadata import version; print(sys.version); print('lerobot', version('lerobot')); print('pyarrow', pyarrow.__version__)"
python -c "import cv2; gui = next(line.strip() for line in cv2.getBuildInformation().splitlines() if 'GUI:' in line); print(gui); assert 'COCOA' in gui"
jupyter kernelspec list
```

`pip check`에서 의존성 충돌이 없어야 하며, macOS OpenCV의 GUI 표시는 `COCOA`여야 한다.

## 코드 호환 범위

이 환경의 녹화·추론 import와 녹화 데이터 저장·재개·영상 읽기 검증은
[`VAI_AIWORKER`의 `642e45b8`](https://github.com/jaehoondata/VAI_AIWORKER/tree/642e45b8f0543d3281e97f2233e2d16884aafe6e)
코드로 수행했다. 실제 로봇 제어 또는 정책 추론 실행을 검증한 것은 아니다.

`VR_teleoperation/requirements.txt`도 `lerobot[dataset]==0.6.1`을 지정한다.
최신 코드의 녹화 생성·이어쓰기 API, 데이터 검토 노트북, 정책 추론의 import 경로와
전처리·후처리 장치 설정이 이 저장소에 반영되어 있다. 환경 파일은 설치된 패키지를
복원하며, 실행 코드는 이 저장소의 `VR_teleoperation/`를 사용한다.
