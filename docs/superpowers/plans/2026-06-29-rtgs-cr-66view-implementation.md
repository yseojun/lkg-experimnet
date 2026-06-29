# RTGS CoherentRaster 66-View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a new RTGS + CoherentRaster 66-view command that renders all 66 views for an LKG interlaced image while saving only 5 sampled full-frame views for inspection.

**Architecture:** Create a separate `rtgs_cr_66views.py` entrypoint that reuses the Task 3 official RTGS runtime and snapshot path. The first correctness baseline uses `compose` interlacing: render each CR view with the per-view Task 3 `rtgscompat` adapter in memory, save only sampled views, and compose the final LKG image from all 66 rendered views. A later direct multi-view CR kernel mode can be added after N3DV per-view projection compensation is solved in the kernel.

**Tech Stack:** Python, PyTorch CUDA, RTGS official renderer, gsplat CoherentRaster single-view adapter, NumPy/PIL, `unittest`.

---

## Technical Constraint

The Task 3 N3DV `rtgscompat` adapter is view-dependent: it scales camera-space mean `x/y` for one view while keeping covariance unchanged. A single direct multi-view CR kernel call cannot apply a different mean-only adapter for all 66 views without changing the CUDA kernel interface. Therefore the first Task 4 implementation must use per-view CR rendering plus in-memory LKG interlacing for correctness. This still uses all 66 views for the interlaced output and saves only 5 sampled full-frame views.

## Files

- Create: `experiment/src/lkg_experiment/rtgs_coherent/cr_66views.py`
- Create: `experiment/rtgs_cr_66views.py`
- Create: `experiment/tests/test_rtgs_cr_66views.py`
- Modify: `docs/superpowers/specs/2026-06-29-rtgs-cr-66view-design.md`
- Modify: `docs/superpowers/plans/2026-06-29-rtgs-coherent-raster-lkg.md`

## Task 1: Unit Tests For CLI And Pure Helpers

**Files:**
- Create: `experiment/tests/test_rtgs_cr_66views.py`

- [x] **Step 1: Write failing tests**

Add tests for:

```python
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from lkg_experiment.rtgs_coherent import cr_66views


class RtgsCr66ViewsTest(unittest.TestCase):
    def test_default_output_path_is_external_and_mode_separated(self):
        path = cr_66views.default_cr66_output_path(
            Path("/data/ysj/result/4dgs/RTGS/coffee_martini"),
            split="test",
            camera_index=0,
            timestamp=0.0,
            run_label="cr66",
        )

        self.assertIn("rtgs_cr_66views", path.parts)
        self.assertEqual(path.name, "cr66")

    def test_parser_defaults_save_five_sampled_views_and_all_66_for_interlaced(self):
        args = cr_66views.build_parser().parse_args(["--dataset-kind", "dnerf"])

        self.assertEqual(args.views, 66)
        self.assertEqual(args.sample_save_views, 5)
        self.assertEqual(args.interlace_mode, "compose")
        self.assertTrue(args.write_interlaced)
        self.assertTrue(args.compare_official_sampled)

    def test_evenly_spaced_sample_indices_cover_first_middle_last(self):
        self.assertEqual(cr_66views.resolve_sample_view_indices(66, 5, None), [0, 16, 32, 49, 65])

    def test_explicit_sample_indices_are_validated(self):
        self.assertEqual(cr_66views.resolve_sample_view_indices(66, 5, "0, 7 65"), [0, 7, 65])
        with self.assertRaises(ValueError):
            cr_66views.resolve_sample_view_indices(66, 5, "0, 66")

    def test_validate_viewpoint_index_rejects_out_of_range_view_id(self):
        viewpoint_index = np.zeros((2, 2, 3), dtype=np.int32)
        viewpoint_index[0, 0, 0] = 66

        with self.assertRaises(ValueError):
            cr_66views.validate_viewpoint_index(viewpoint_index, source_view_count=66)

    def test_compose_interlaced_image_uses_all_subpixel_channels(self):
        views = torch.zeros((2, 3, 2, 2), dtype=torch.float32)
        views[0] = 0.25
        views[1] = 0.75
        viewpoint_index = np.zeros((2, 2, 3), dtype=np.int32)
        viewpoint_index[:, :, 1] = 1

        image = cr_66views.compose_interlaced_from_views(views, viewpoint_index)

        self.assertTrue(torch.allclose(image[0], torch.full((2, 2), 0.25)))
        self.assertTrue(torch.allclose(image[1], torch.full((2, 2), 0.75)))
        self.assertTrue(torch.allclose(image[2], torch.full((2, 2), 0.25)))
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_cr_66views -v
```

Expected: FAIL because `cr_66views` does not exist.

## Task 2: Implement Parser, Paths, Sampling, And Interlace Helpers

**Files:**
- Create: `experiment/src/lkg_experiment/rtgs_coherent/cr_66views.py`
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/__init__.py` only if importing the module requires it

- [x] **Step 1: Implement minimal pure helpers**

Implement:

```python
DEFAULT_CR66_OUTPUT_ROOT = DEFAULT_GENERATED_ROOT / "rtgs_cr_66views"

def default_cr66_output_path(...): ...
def build_parser() -> argparse.ArgumentParser: ...
def resolve_sample_view_indices(views: int, sample_save_views: int, explicit: str | None) -> list[int]: ...
def validate_viewpoint_index(viewpoint_index: np.ndarray, *, source_view_count: int) -> dict[str, int]: ...
def compose_interlaced_from_views(views: torch.Tensor, viewpoint_index: np.ndarray) -> torch.Tensor: ...
```

`compose_interlaced_from_views` must accept views shaped `[V, 3, H, W]` and viewpoint index shaped `[H, W, 3]`, then return `[3, H, W]`.

- [x] **Step 2: Run tests and verify GREEN**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_cr_66views -v
```

Expected: PASS.

## Task 3: Add Runtime Helpers For Official Comparisons

**Files:**
- Modify: `experiment/tests/test_rtgs_cr_66views.py`
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/cr_66views.py`

- [x] **Step 1: Write failing sentinel-preservation test**

Add a test using a simple fake camera object:

```python
def test_synthetic_camera_preserves_n3dv_negative_fov_sentinel(self):
    anchor = SimpleNamespace(
        image_width=1352,
        image_height=1014,
        FoVx=-1.0,
        FoVy=-1.0,
        fl_x=730.0,
        fl_y=730.0,
        cx=676.0,
        cy=507.0,
        image=torch.zeros((3, 1014, 1352), dtype=torch.float32),
    )
    viewmat = torch.eye(4)

    camera = cr_66views.synthetic_camera_from_viewmat_preserving_rtgs_contract(
        anchor,
        viewmat,
        uid=0,
        image_name="view_000",
        timestamp=0.0,
        device="cuda",
        width=1440,
        height=2560,
        crop_to_fill=True,
    )

    self.assertEqual(camera.FoVx, -1.0)
    self.assertEqual(camera.FoVy, -1.0)
    self.assertGreater(camera.fl_x, 0.0)
    self.assertGreater(camera.fl_y, 0.0)
```

- [x] **Step 2: Run test and verify RED**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_cr_66views.RtgsCr66ViewsTest.test_synthetic_camera_preserves_n3dv_negative_fov_sentinel -v
```

Expected: FAIL because the helper does not exist.

- [x] **Step 3: Implement sentinel-preserving synthetic camera helper**

Implement a local helper instead of using `rtgs_camera_from_gsplat_viewmat()` directly. If the anchor camera has positive `fl_x/fl_y` and nonpositive `FoVx/FoVy`, preserve the negative FoV sentinel and scale explicit intrinsics to the requested output size.

- [x] **Step 4: Run test and verify GREEN**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_cr_66views -v
```

Expected: PASS.

## Task 4: Implement 66-View Render Command

**Files:**
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/cr_66views.py`
- Create: `experiment/rtgs_cr_66views.py`

- [x] **Step 1: Implement the executable wrapper**

Create `experiment/rtgs_cr_66views.py` with the same wrapper pattern as `rtgs_cr_1view.py`.

- [x] **Step 2: Implement CUDA render flow**

Implement `render_rtgs_cr_66views(args)`:

```text
prepare_official_rtgs_1view(args)
normalize_explicit_intrinsics_fov(enabled=False by default)
materialize_rtgs_geometry once
synthesize 66 viewmats
resolve 5 sampled view ids
for each of 66 views:
  evaluate RTGS color for that view center
  build snapshot
  apply adapt_snapshot_for_rtgs_compat_projection for that view
  render single-view CR
  keep the tensor in memory for interlacing
  save PNG only if view id is sampled
compose LKG interlaced image from all 66 view tensors
optionally render official sampled synthetic views and metrics
write manifest.json and metrics.json
```

- [x] **Step 3: Run non-CUDA tests**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_cr_66views tests.test_rtgs_cr_1view tests.test_rtgs_official_1view -v
```

Expected: PASS.

## Task 5: Update Docs And Run Smoke Commands

**Files:**
- Modify: `docs/superpowers/specs/2026-06-29-rtgs-cr-66view-design.md`
- Modify: `docs/superpowers/plans/2026-06-29-rtgs-coherent-raster-lkg.md`

- [x] **Step 1: Document compose-mode baseline**

Update the spec and main plan to say Task 4 correctness uses `--interlace-mode compose` by default because N3DV `rtgscompat` is view-dependent.

- [x] **Step 2: Run CUDA smoke if available**

Run:

```bash
cd experiment
conda run -n rtgs-coherent-cu121 env PYTHONPATH=src python rtgs_cr_66views.py \
  --dataset-kind dnerf \
  --model-path /data/ysj/result/4dgs/RTGS/jumpingjacks \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --gsplat-root /home/ysj/lkg-experiment/gsplat \
  --dataset-root /data/ysj/dataset/dnerf \
  --split test \
  --camera-index 0 \
  --views 66 \
  --sample-save-views 5 \
  --cluster-size 1 \
  --map-mode linear \
  --width 400 \
  --height 400 \
  --run-label jumpingjacks_cam00_cr66_smoke
```

Then run the equivalent N3DV command at a reduced smoke resolution first:

```bash
cd experiment
conda run -n rtgs-coherent-cu121 env PYTHONPATH=src python rtgs_cr_66views.py \
  --dataset-kind n3dv \
  --model-path /data/ysj/result/4dgs/RTGS/coffee_martini \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --gsplat-root /home/ysj/lkg-experiment/gsplat \
  --n3dv-root /data/ysj/dataset/N3DV \
  --split test \
  --camera-index 0 \
  --n3dv-frame-index 0 \
  --views 66 \
  --sample-save-views 5 \
  --cluster-size 1 \
  --map-mode linear \
  --width 676 \
  --height 507 \
  --run-label coffee_martini_cam00_frame0000_cr66_smoke
```

Expected: sampled views, sampled comparisons, `rtgs_cr_lkg_interlaced.png`, `metrics.json`, and `manifest.json` are written under `/data/ysj/result/coherent-raster/generated/rtgs_cr_66views/...`.

Smoke results, 2026-06-29:

```text
dnerf jumpingjacks, 160x160, map-mode linear:
  output: /data/ysj/result/coherent-raster/generated/rtgs_cr_66views/jumpingjacks/jumpingjacks_cam00_cr66_smoke_160
  sampled views: [0, 16, 32, 49, 65]
  sampled mean PSNR vs official: 79.762 dB
  adapter summary: 0/66 applied, reason=fov_not_nonpositive_sentinel

N3DV coffee_martini, 338x254, map-mode linear:
  output: /data/ysj/result/coherent-raster/generated/rtgs_cr_66views/coffee_martini/coffee_martini_cam00_frame0000_cr66_smoke_338x254
  sampled views: [0, 16, 32, 49, 65]
  sampled mean PSNR vs official: 57.394 dB
  camera_fov_normalization.applied: false
  adapter summary: 66/66 applied, reason=explicit_intrinsics_with_nonpositive_fov_sentinel
```

- [ ] **Step 3: Commit**

Commit only source, tests, and docs. Do not commit generated images.
