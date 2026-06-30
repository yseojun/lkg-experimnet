# OMG4-FTGS 1-View And Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OMG4-FTGS package that renders the approved cook_spinach checkpoint as a single PNG and as a same-camera full-time MP4/PNG sequence.

**Architecture:** Add a focused `lkg_experiment.omg4_ftgs` package parallel to `rtgs_coherent`. The package separates compressed model loading, camera/timestamp loading, rendering, and CLI output orchestration so the normal gsplat backend can later be swapped for CoherentRaster.

**Tech Stack:** Python 3.11, PyTorch CUDA, tiny-cuda-nn, dahuffman, gsplat, PIL, imageio when available.

---

## File Structure

- Create `experiment/src/lkg_experiment/omg4_ftgs/__init__.py`: package marker and public summary.
- Create `experiment/src/lkg_experiment/omg4_ftgs/model.py`: OMG4-FTGS checkpoint decode and per-timestamp tensor materialization.
- Create `experiment/src/lkg_experiment/omg4_ftgs/camera.py`: test timestamp and N3DV pose/intrinsics loading.
- Create `experiment/src/lkg_experiment/omg4_ftgs/render.py`: normal `gsplat.rasterization` render backend.
- Create `experiment/src/lkg_experiment/omg4_ftgs/cli.py`: `single`, `video`, and `both` command orchestration.
- Create `experiment/omg4_ftgs_1view.py`: top-level script matching RTGS entrypoint style.
- Create `experiment/tests/test_omg4_ftgs.py`: unit tests for defaults, path resolution, timestamp loading, and camera matrix shapes.
- Modify `experiment/pyproject.toml`: add a console script entry.

## Task 1: Package Skeleton And Defaults

**Files:**
- Create: `experiment/src/lkg_experiment/omg4_ftgs/__init__.py`
- Create: `experiment/src/lkg_experiment/omg4_ftgs/cli.py`
- Create: `experiment/omg4_ftgs_1view.py`
- Test: `experiment/tests/test_omg4_ftgs.py`

- [ ] **Step 1: Write failing tests for default paths and parser**

Create tests that assert:

```python
from pathlib import Path
from lkg_experiment.omg4_ftgs import cli

def test_default_cook_spinach_paths():
    assert cli.DEFAULT_CHECKPOINT_PATH == Path("/data/ysj/result/4dgs/OMG4-FTGS_weights/ours_L_weight/cook_spinach.xz")
    assert cli.DEFAULT_DATA_PATH == Path("/data/ysj/dataset/N3DV/cook_spinach")

def test_parser_defaults_to_both_mode():
    args = cli.build_parser().parse_args([])
    assert args.mode == "both"
    assert args.camera_index == 0
    assert args.frame_index == 0
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs.Omg4FtgsTest.test_default_cook_spinach_paths -v
```

Expected: import failure for `lkg_experiment.omg4_ftgs`.

- [ ] **Step 3: Add minimal package, parser, and top-level script**

Implement constants:

```python
DEFAULT_WEIGHTS_ROOT = Path("/data/ysj/result/4dgs/OMG4-FTGS_weights")
DEFAULT_CHECKPOINT_PATH = DEFAULT_WEIGHTS_ROOT / "ours_L_weight" / "cook_spinach.xz"
DEFAULT_DATA_PATH = Path("/data/ysj/dataset/N3DV/cook_spinach")
DEFAULT_OUTPUT_ROOT = Path("/data/ysj/result/coherent-raster/generated/omg4_ftgs")
```

Parser arguments:

```python
--mode {single,video,both}
--checkpoint-path
--data-path
--output-dir
--camera-index
--frame-index
--resolution
--width
--height
--fps
--save-frames
--gsplat-root
--device
```

- [ ] **Step 4: Run test to verify it passes**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: parser/default tests pass; later tests are not present yet.

## Task 2: Camera And Timestamp Loading

**Files:**
- Create: `experiment/src/lkg_experiment/omg4_ftgs/camera.py`
- Test: `experiment/tests/test_omg4_ftgs.py`

- [ ] **Step 1: Write failing tests for timestamps and camera shapes**

Test:

```python
def test_load_test_timestamps_from_real_dataset():
    frames = camera.load_test_frames(Path("/data/ysj/dataset/N3DV/cook_spinach"))
    assert len(frames) > 0
    assert isinstance(frames[0].timestamp, float)

def test_load_pose_camera_shapes_from_real_dataset():
    cam = camera.load_pose_camera(Path("/data/ysj/dataset/N3DV/cook_spinach"), camera_index=0, resolution=2)
    assert cam.viewmats.ndim == 3
    assert cam.viewmats.shape[-2:] == (4, 4)
    assert cam.K.shape == (3, 3)
    assert cam.width > 0
    assert cam.height > 0
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: missing `camera` module or missing functions.

- [ ] **Step 3: Implement `camera.py`**

Implement:

```python
@dataclass(frozen=True)
class TestFrame:
    index: int
    image_path: Path
    timestamp: float

@dataclass(frozen=True)
class FtgsCameraSet:
    viewmats: np.ndarray
    K: np.ndarray
    width: int
    height: int
    source_width: int
    source_height: int
    focal: float
```

Load `transforms_test.json` frames and `poses_bounds.npy` with the same conversion used by OMG4-FTGS:

```python
c2ws = cam[..., :4]
c2ws = np.stack([c2ws[..., 1], c2ws[..., 0], -c2ws[..., 2], c2ws[..., 3]], axis=1)
bottom = np.array([0, 0, 0, 1]).reshape((1, 4, 1)).repeat(len(c2ws), 0)
c2ws = np.concatenate([c2ws, bottom], axis=2)
viewmats = np.transpose(np.linalg.inv(c2ws), (0, 2, 1)).astype(np.float32)
```

- [ ] **Step 4: Run tests to verify pass**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: all current tests pass.

## Task 3: Model Decode And Materialization

**Files:**
- Create: `experiment/src/lkg_experiment/omg4_ftgs/model.py`
- Test: `experiment/tests/test_omg4_ftgs.py`

- [ ] **Step 1: Write non-CUDA metadata test and CUDA smoke guard**

Test checkpoint exists and can read top-level keys without CUDA:

```python
def test_default_checkpoint_has_expected_encoded_keys():
    keys = model.peek_checkpoint_keys(cli.DEFAULT_CHECKPOINT_PATH)
    assert "means" in keys
    assert "MLP_cont" in keys
    assert "scale_code" in keys
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: missing `model` module or `peek_checkpoint_keys`.

- [ ] **Step 3: Implement `model.py`**

Implement a lightweight `DynamicGaussians` using OMG4-FTGS logic:

```python
def peek_checkpoint_keys(path: Path | str) -> set[str]:
    with lzma.open(Path(path).expanduser(), "rb") as f:
        payload = pickle.load(f)
    return set(payload)
```

For full decode, import `huffman_decode` from `utils.compress_utils` after adding `OMG4` to `sys.path`, build tiny-cuda-nn networks, restore `MLP_*` params, and expose:

```python
def load_dynamic_gaussians(path: Path | str, *, omg4_root: Path | str, device: str) -> DynamicGaussians
def materialize(timestamp: float) -> FtgsSplats
```

`FtgsSplats` contains `means`, `quats`, `scales`, `opacities`, `colors`, and `sh_degree`.

- [ ] **Step 4: Run tests**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: CPU metadata tests pass.

## Task 4: Single-View Render

**Files:**
- Create: `experiment/src/lkg_experiment/omg4_ftgs/render.py`
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/cli.py`
- Test: `experiment/tests/test_omg4_ftgs.py`

- [ ] **Step 1: Add render backend and CLI manifest tests with mocks**

Patch `render.render_splats_gsplat` in a CLI test to return a small tensor and assert `single.png` plus `manifest.json` are written.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: missing render/backend orchestration.

- [ ] **Step 3: Implement render and single mode**

Render:

```python
image, _alpha, _meta = gsplat.rasterization(
    means=splats.means,
    quats=splats.quats,
    scales=splats.scales,
    opacities=splats.opacities,
    colors=splats.colors,
    viewmats=viewmat.unsqueeze(0),
    Ks=K.unsqueeze(0),
    sh_degree=splats.sh_degree,
    width=width,
    height=height,
)
return image[0].permute(2, 0, 1).clamp(0.0, 1.0).contiguous()
```

- [ ] **Step 4: Run unit tests**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: unit tests pass.

- [ ] **Step 5: Run real single-view smoke**

Run:

```bash
cd experiment
PYTHONPATH=src python omg4_ftgs_1view.py --mode single --run-label cook_spinach_smoke_single
```

Expected: output directory contains `single.png` and `manifest.json`.

## Task 5: Full-Time Video Render

**Files:**
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/cli.py`
- Test: `experiment/tests/test_omg4_ftgs.py`

- [ ] **Step 1: Add mocked video test**

Mock frame rendering for three timestamps and assert:

```python
frames/frame_0000.png
frames/frame_0001.png
frames/frame_0002.png
manifest.json
```

exist. Assert MP4 status is present in manifest.

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: video orchestration missing.

- [ ] **Step 3: Implement video mode**

Loop over timestamps from `load_test_frames(data_path)`, render same camera pose, save PNGs, and attempt MP4 with `imageio.v3.imwrite(video_path, frames, fps=args.fps)`.

Manifest records:

```json
{
  "mode": "both",
  "checkpoint_path": "...cook_spinach.xz",
  "data_path": "...cook_spinach",
  "camera_index": 0,
  "frame_count": 0,
  "timestamps": [],
  "mp4": {"written": true, "path": "video.mp4", "error": null}
}
```

- [ ] **Step 4: Run unit tests**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: unit tests pass.

- [ ] **Step 5: Run real both-mode smoke**

Run:

```bash
cd experiment
PYTHONPATH=src python omg4_ftgs_1view.py --mode both --run-label cook_spinach_smoke_both
```

Expected: `single.png`, frame PNGs, `manifest.json`, and preferably `video.mp4`.

## Self-Review

- Spec coverage: package, decode, camera loading, single PNG, full-time video, manifests, and MP4 fallback are covered.
- Placeholder scan: no `TBD`, `TODO`, or unspecified "add tests" steps remain.
- Type consistency: `FtgsSplats`, `TestFrame`, `FtgsCameraSet`, and CLI defaults are introduced before use.
