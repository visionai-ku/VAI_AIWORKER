# `ri_motion_v5_package.hf_models`

Hugging Face model wrappers and helpers for vision, vision-language, correspondence, depth, segmentation, and face workflows.

## Modules

- `hf_common.py`
  - Shared device, dtype, loading, and utility helpers
- `hf_vision.py`
  - Vision model helpers
- `hf_llm.py`
  - Vision-language and language-model helpers
- `hf_corresp.py`
  - Dense or feature correspondence helpers
- `hf_face.py`
  - Face identification helpers
- `hf_viz.py`
  - Visualization helpers for model outputs

## Public API Snapshot

- `hf_common.py`: package/checkpoint helpers, device selection, and JSON object extraction
- `hf_vision.py`: `PointSam2`, `PromptOwlv2`, `PromptOwlv2Sam2`, `DepthAnythingV2`
- `hf_llm.py`: `HuggingFaceLLM`
- `hf_corresp.py`: `DINOv2Correspondence`, `get_fig_h_from_image_shape(...)`
- `hf_face.py`: `YoloFaceDet`, `FaceEmbedding`, `FaceIdentification`
- `hf_viz.py`: image/point normalization and Matplotlib annotation helpers such as `rgb_img_np(...)`, `viz_label(...)`, and `viz_close(...)`

## Import

```python
from ri_motion_v5_package.hf_models import *
```

## Notes

- Notebook examples live in `../../ri_motion_v5_notebook/21_hugging_face_models/`
- Keep model download/cache behavior explicit in notebooks when it affects runtime or storage
