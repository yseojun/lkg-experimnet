# RTGS Coherent Raster LKG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** RTGS checkpoint를 official RTGS 렌더링 경로로 1-view baseline까지 먼저 고정하고, 그 baseline을 기준으로 RTGS + coherent-raster 1-view와 LKG multi-view 확장을 단계적으로 검증한다.

**Architecture:** 기존 `rtgs_basic_1view.py` wrapper를 계속 보정하지 않고, 먼저 RTGS 원본 구성 요소인 `arguments`, `Scene`, `GaussianModel`, `gaussian_renderer.render()` 흐름을 얇게 감싼 official-flow baseline을 만든다. 그 다음 coherent-raster는 별도 1-view entrypoint로 분리해 원본 RTGS render와 같은 camera, timestamp, background, color evaluation을 공유하도록 붙인다. Multi-view는 1-view RTGS + CR 결과가 원본 RTGS와 맞는 것을 확인한 뒤에만 `views66` 계열로 확장한다.

**Tech Stack:** Python, PyTorch CUDA, RTGS local checkout at `4d-gaussian-splatting`, gsplat/coherent-raster local checkouts, `experiment` package, `unittest`, LKG 66-view mapping NPZ artifacts.

---

## Execution Rule

이 계획은 한 번에 실행하지 않는다. 아래 Task 1, Task 2, Task 3, Task 4는 사용자가 다음 단계 진행을 확인한 뒤에만 시작한다.

Task 1은 "RTGS official 1-view render가 LKG display에 들어갈 단일 view 이미지로 정상 렌더링된다"는 되돌림 지점이다. Task 1 검증이 끝나면 반드시 커밋한다. Task 2는 구현 없이 설계 브레인스토밍 문서만 만든다. Task 3은 RTGS + coherent-raster 1-view를 official RTGS baseline과 별도 entrypoint로 비교한다. Task 4는 Task 3 결과가 정상일 때만 LKG multi-view로 확장한다.

## Current Repository Facts

- `4d-gaussian-splatting` checkout에는 현재 top-level `render.py`가 없다.
- 로컬 RTGS official-flow 기준은 `4d-gaussian-splatting/arguments/__init__.py`, `4d-gaussian-splatting/scene/__init__.py`, `4d-gaussian-splatting/scene/gaussian_model.py`, `4d-gaussian-splatting/gaussian_renderer/__init__.py`를 직접 사용하는 것으로 정의한다.
- 로컬 RTGS `Scene`은 N3DV raw dynamic frames를 직접 읽지 못하고 `colmap/sparse` static camera만 읽는다. N3DV 1-view baseline은 RTGS model output의 `cameras.json`과 raw `poses_bounds.npy`를 사용해 dynamic camera를 구성한다.
- 현재 실패 기록과 재설계 결정은 `experiment/docs/rtgs_original_render_rewrite_decision.md`에 있다.
- 기존 wrapper entrypoint는 `experiment/rtgs_basic_1view.py`이고 실제 구현은 `experiment/src/lkg_experiment/rtgs_coherent/basic_1view.py`와 `experiment/src/lkg_experiment/rtgs_coherent/cli.py`에 있다.
- 기존 RTGS + CR multi-view 후보 구현은 `experiment/src/lkg_experiment/rtgs_coherent/views66.py`에 있다.
- 기존 테스트는 `experiment/tests/test_rtgs_basic_1view.py`, `experiment/tests/test_rtgs_coherent.py`, `experiment/tests/test_rtgs_66_lkg_script.py`에 분산되어 있다.

## Files By Phase

- Create: `experiment/src/lkg_experiment/rtgs_coherent/official_1view.py`
- Create: `experiment/rtgs_official_1view.py`
- Create: `experiment/tests/test_rtgs_official_1view.py`
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/__init__.py`
- Modify: `experiment/docs/rtgs_original_render_rewrite_decision.md`
- Create: `experiment/docs/rtgs_coherent_raster_brainstorm.md`
- Create: `experiment/src/lkg_experiment/rtgs_coherent/cr_1view.py`
- Create: `experiment/rtgs_cr_1view.py`
- Create: `experiment/tests/test_rtgs_cr_1view.py`
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/views66.py`
- Modify: `experiment/scripts/run_rtgs_66_lkg_all.sh`
- Modify: `experiment/tests/test_rtgs_66_lkg_script.py`

## Baseline Invariants

- Official baseline에는 `RtgsLiteGaussianModel`을 사용하지 않는다.
- dnerf official baseline에는 custom camera loader를 사용하지 않는다.
- dnerf official baseline camera는 RTGS `Scene(...).getTrainCameras()` 또는 `Scene(...).getTestCameras()`에서 나온 camera를 사용한다.
- N3DV official baseline camera는 local RTGS `Scene`의 static colmap camera가 아니라 `model_path/cameras.json`과 `n3dv_root/<scene>/poses_bounds.npy`로 만든 dynamic camera를 사용한다.
- Official baseline model은 RTGS `GaussianModel.restore(model_params, opt)` 또는 RTGS `Scene`의 PLY loading path 중 하나로만 로드한다.
- Official baseline render call은 `gaussian_renderer.render(camera.cuda(), gaussians, pipe, background)["render"]` 형태를 유지한다.
- dnerf와 N3DV dataset root는 CLI에서 명시적으로 분리한다.
- output directory는 mode별로 분리한다: `generated/rtgs_official_1view`, `generated/rtgs_cr_1view`, `generated/rtgs_coherent`.
- diagnostic output은 `--run-label`이 없을 때 같은 디렉터리를 덮어쓰지 않는다.
- 모든 render manifest는 `rtgs_code_root`, RTGS git commit, `model_path`, `checkpoint_path`, `dataset_kind`, `source_path`, `config_path`, split, camera index, timestamp, `ModelParams`, `PipelineParams`, output image path를 기록한다.

---

### Task 1: RTGS Official 1-View Baseline

**Files:**
- Create: `experiment/src/lkg_experiment/rtgs_coherent/official_1view.py`
- Create: `experiment/rtgs_official_1view.py`
- Create: `experiment/tests/test_rtgs_official_1view.py`
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/__init__.py`
- Modify: `experiment/docs/rtgs_original_render_rewrite_decision.md`

**Current Status, 2026-06-29:** Steps 1-9 are implemented and verified. Regular Codex sandbox execution does not expose `/dev/nvidia*`, but escalated execution sees the GPUs and produced dnerf/N3DV image artifacts. Since the LKG display is not connected, Step 9 is scoped to saved image artifacts and `metrics.json` manifest records. The loader now follows the official RTGS flat argparse flow: instantiate `ModelParams`, `OptimizationParams`, `PipelineParams`; add train.py top-level args; recursively merge YAML leaf values into the same namespace; then call each group `extract()`.

**Re-analysis, 2026-06-29:** dnerf `jumpingjacks` matches the saved official reference when the dirty RTGS checkout is isolated through a clean HEAD snapshot: render vs reference render is about 61.6 dB. N3DV `coffee_martini` now selects the correct dynamic GT/order (`cam00_0000` maps closest to official `gt/00196.png`) and reproduces the saved official render after preserving the original RTGS explicit-intrinsics camera contract: `FoVx=FoVy=-1.0` while `fl_x/fl_y/cx/cy` drive the center-shift projection matrix. Before this fix, recomputing positive FoV values from focal length produced about 20.1 dB vs GT; with the sentinel FoV contract, `rtgs_vs_gt_psnr` is about 27.52 dB and saved PNG render vs official `renders/00196.png` is about 51.2 dB.

**Milestone 1 Complete, 2026-06-29:** Initial plan item 1 is complete: RTGS model loading plus official RTGS 1-view rendering is fixed for both dnerf and N3DV as image artifacts, with LKG display output intentionally deferred because no display is connected. This commit is the restore point before brainstorming and implementing RTGS + coherent-raster rendering.

**Final 1-view sweep, 2026-06-29:** The official 1-view renderer completed on all configured RTGS scenes: dnerf `bouncingballs`, `hellwarrior`, `hook`, `jumpingjacks`, `lego`, `mutant`, `standup`, `trex`, and N3DV `coffee_martini`, `cook_spinach`, `cut_roasted_beef`, `flame_salmon`, `flame_steak`, `sear_steak`. `flame_salmon` required one loader robustness fix because its `cameras.json` includes frame 300 while raw `cam00/images` ends at 299; the loader now validates only the requested `--n3dv-frame-index` unless all frames are requested. A batch helper for user-run video generation is available at `experiment/scripts/run_rtgs_1view_videos_all.sh`.

- [x] **Step 1: Define official baseline CLI contract**

`experiment/src/lkg_experiment/rtgs_coherent/official_1view.py`의 parser는 아래 option을 제공한다.

```text
--model-path
--checkpoint
--rtgs-code-root
--dataset-kind {dnerf,n3dv}
--dataset-root
--n3dv-root
--source-path
--config
--output-dir
--run-label
--split {train,test}
--camera-index
--n3dv-frame-index
--device
--checkpoint-load-device
--background {auto,black,white}
--no-ssim
```

- [x] **Step 2: Write parser and path tests first**

Create `experiment/tests/test_rtgs_official_1view.py` with tests that assert:

```python
from pathlib import Path
import unittest

from lkg_experiment.rtgs_coherent import official_1view


class RtgsOfficial1ViewTest(unittest.TestCase):
    def test_default_output_path_is_mode_separated(self):
        path = official_1view.default_official_output_path(
            Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            split="test",
            camera_index=0,
            timestamp=0.0,
            run_label="baseline",
        )

        self.assertIn("rtgs_official_1view", path.parts)
        self.assertEqual(path.name, "baseline")

    def test_parser_requires_dataset_kind_to_avoid_implicit_fallback(self):
        parser = official_1view.build_parser()
        args = parser.parse_args(["--dataset-kind", "dnerf"])

        self.assertEqual(args.dataset_kind, "dnerf")
        self.assertEqual(args.split, "test")
        self.assertEqual(args.camera_index, 0)
        self.assertEqual(args.background, "auto")


if __name__ == "__main__":
    unittest.main()
```

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_official_1view -v
```

Expected: FAIL because `official_1view` does not exist yet.

- [x] **Step 3: Implement official-flow loader**

The loader must:

1. insert `--rtgs-code-root` into `sys.path`
2. instantiate RTGS `ModelParams`, `PipelineParams`, `OptimizationParams`
3. recursively merge RTGS YAML config leaves into the same flat argparse namespace used by official `train.py`
4. create RTGS `GaussianModel`
5. restore `checkpoints/chkpnt_best.pth`
6. create RTGS `Scene` using RTGS `Scene` and select one train/test camera

The key import boundary is:

```python
from arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import Scene
from scene.gaussian_model import GaussianModel
```

- [x] **Step 4: Implement official-flow render**

The render function must call RTGS directly:

```python
with torch.no_grad():
    image = render(camera.cuda(), gaussians, pipe, background)["render"]
    image = image.detach().clamp(0.0, 1.0).contiguous()
```

It must save:

```text
gt.png
rtgs_official_render.png
comparison.png
metrics.json
```

- [x] **Step 5: Add thin executable wrapper**

Create `experiment/rtgs_official_1view.py` with the same wrapper style as `experiment/rtgs_basic_1view.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lkg_experiment.rtgs_coherent.official_1view import main


if __name__ == "__main__":
    raise SystemExit(main())
```

- [x] **Step 6: Run non-CUDA tests**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_official_1view -v
PYTHONPATH=src python -m unittest discover -s tests -v
```

Expected: PASS for parser/path tests and no regression in existing non-CUDA tests.

- [x] **Step 7: Run CUDA smoke render for dnerf jumpingjacks**

Run on the CUDA server:

```bash
cd experiment
PYTHONPATH=src python rtgs_official_1view.py \
  --dataset-kind dnerf \
  --model-path /data/ysj/result/4dgs/RTGS/jumpingjacks \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --dataset-root /data/ysj/dataset/dnerf \
  --split test \
  --camera-index 0 \
  --run-label jumpingjacks_test0_official
```

Expected:

```text
generated/rtgs_official_1view/jumpingjacks/jumpingjacks_test0_official/rtgs_official_render.png
generated/rtgs_official_1view/jumpingjacks/jumpingjacks_test0_official/metrics.json
```

- [x] **Step 8: Run CUDA smoke render for N3DV coffee_martini**

Run on the CUDA server:

```bash
cd experiment
PYTHONPATH=src python rtgs_official_1view.py \
  --dataset-kind n3dv \
  --model-path /data/ysj/result/4dgs/RTGS/coffee_martini \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --n3dv-root /data/ysj/dataset/N3DV \
  --split test \
  --camera-index 0 \
  --n3dv-frame-index 0 \
  --run-label coffee_martini_cam00_0000_official
```

Expected:

```text
generated/rtgs_official_1view/coffee_martini/coffee_martini_cam00_0000_official/rtgs_official_render.png
generated/rtgs_official_1view/coffee_martini/coffee_martini_cam00_0000_official/metrics.json
```

- [x] **Step 9: Inspect LKG single-view image artifact**

Use the generated `rtgs_official_render.png` as the single-view image sent to the LKG display path. Record in `metrics.json` or the phase note whether the image is visually valid, not black, not transposed, not mirrored unexpectedly, and uses the expected scene/frame.

- [x] **Step 10: Commit Task 1 baseline**

Only after Step 6, Step 7, Step 8, and Step 9 pass:

```bash
git add experiment/rtgs_official_1view.py \
  experiment/src/lkg_experiment/rtgs_coherent/official_1view.py \
  experiment/src/lkg_experiment/rtgs_coherent/__init__.py \
  experiment/tests/test_rtgs_official_1view.py \
  experiment/docs/rtgs_original_render_rewrite_decision.md \
  docs/superpowers/plans/2026-06-29-rtgs-coherent-raster-lkg.md
git commit -m "baseline: add RTGS official one-view render"
```

Completed in commit `34fce60 Establish RTGS official 1-view baseline`. This is the restore point before Task 2/3 CR work.

---

### Task 2: RTGS + Coherent-Raster Design Brainstorm

**Files:**
- Create: `experiment/docs/rtgs_coherent_raster_brainstorm.md`
- Modify: `docs/superpowers/plans/2026-06-29-rtgs-coherent-raster-lkg.md`

**Current Status, 2026-06-29:** The paper analysis and RTGS application brainstorm are documented in `experiment/docs/rtgs_coherent_raster_brainstorm.md`. The recommended direction is adapter-first gsplat CoherentRaster: load the model/camera through the Step 1 official RTGS path, materialize one timestamp-specific 3D snapshot from the 4D Gaussian model, precompute RTGS SH/4D SH RGB for CR representative views, and call the existing gsplat CR kernels. Native RTGS CUDA CR is deferred until this adapter path proves visually valid.

- [x] **Step 1: Capture official baseline facts**

Write the following values from Task 1 manifests into `experiment/docs/rtgs_coherent_raster_brainstorm.md`:

```text
scene
dataset_kind
source_path
checkpoint_path
camera index/name
timestamp
image width/height
PipelineParams
ModelParams
RTGS git commit
rtgs_vs_gt metrics
visual inspection result
```

- [x] **Step 2: Map RTGS render inputs to CR inputs**

Document this mapping table:

```text
RTGS pc.get_xyz -> gsplat means
RTGS evaluated covariance at timestamp -> CR covars
RTGS opacity with temporal marginal -> CR opacities
RTGS SH color evaluated at timestamp and camera center -> CR colors
RTGS camera world_view_transform -> gsplat viewmat
RTGS projection/intrinsics -> gsplat K
RTGS background -> CR backgrounds
```

- [x] **Step 3: Decide the 1-view CR invariant**

Record this design decision:

```text
For 1-view CR validation, view_idx_matrix is all zeros and adjacent_viewmats has shape [1, 1, 4, 4]. The output must match the official RTGS 1-view render for the same camera, timestamp, and image size before any LKG-specific multi-view remapping is introduced.
```

- [x] **Step 4: Identify the first failure split**

Use these comparison layers in order:

```text
official RTGS render image
RTGS snapshot-reference render image
RTGS materialized snapshot statistics
RTGS + CR 1-view image
RTGS + CR 1-view metrics against official image
RTGS + CR LKG interlaced output
```

The document must state that Task 3 cannot start until the expected output filenames and comparison metric thresholds are written down.

- [x] **Step 5: Subagent plan review**

Ask a separate subagent to inspect `experiment/docs/rtgs_coherent_raster_brainstorm.md` and this plan for technical gaps, incorrect assumptions, missing validation gates, and contradictory sequencing. Patch the documents before asking for user review.

**Review result:** The plan was updated to require timestamp-conditioned means for RTGS 4D SH color evaluation, official-vs-snapshot gating before judging CR, `cluster_size=1`/remapping-off multi-view isolation, N3DV sentinel preservation for synthetic comparison cameras, projection sanity checks, cluster map validation, and richer tensor diagnostics.

- [ ] **Step 6: Review with user**

Stop after saving the brainstorm document. Do not edit CR rendering code in Task 2.

---

### Task 3: RTGS + Coherent-Raster 1-View Render

**Files:**
- Create: `experiment/src/lkg_experiment/rtgs_coherent/cr_1view.py`
- Create: `experiment/rtgs_cr_1view.py`
- Create: `experiment/tests/test_rtgs_cr_1view.py`
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/cli.py`
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/__init__.py`
- Modify: `experiment/docs/rtgs_coherent_raster_brainstorm.md`

**Current Status, 2026-06-29:** Task 3 is implemented and accepted for 1-view validation. `experiment/rtgs_cr_1view.py` keeps RTGS + CoherentRaster separate from the Task 1 official baseline while reusing the official runtime loader, camera contract, checkpoint restore, timestamp, background, and GT selection. Outputs now default to `/data/ysj/result/coherent-raster/generated/rtgs_cr_1view/...`.

The accepted `rtgscompat` results are:

```text
dnerf jumpingjacks cam0:
  official_vs_snapshot_psnr: 78.680 dB
  snapshot_vs_cr_psnr:       80.720 dB
  official_vs_cr_psnr:       76.551 dB
  cr_vs_gt_psnr:             27.650 dB

N3DV coffee_martini cam00 frame0000:
  official_vs_snapshot_psnr: 51.485 dB
  snapshot_vs_cr_psnr:       54.001 dB
  official_vs_cr_psnr:       49.560 dB
  cr_vs_gt_psnr:             27.515 dB
```

For N3DV, the valid path is explicitly the RTGS-compatible projection adapter: keep `FoVx=FoVy=-1.0`, leave explicit-intrinsics FoV normalization disabled, and adapt the CR input so gsplat/CR reproduces the RTGS official sentinel-FoV convention. The diagnostic FoV-normalized path reproduces the old bad N3DV rendering and must not be used for acceptance.

- [x] **Step 1: Extract an official runtime adapter**

Refactor `official_1view.py` only enough to expose a helper that returns the official model, camera, GT, pipeline namespace, and background without rendering. CR code must reuse this helper so dnerf and N3DV preserve the Step 1 camera/model contracts.

- [x] **Step 2: Write CR 1-view unit tests first**

Create tests that assert:

```python
from pathlib import Path
import unittest

from lkg_experiment.rtgs_coherent import cr_1view


class RtgsCr1ViewTest(unittest.TestCase):
    def test_default_output_path_is_separate_from_official_baseline(self):
        path = cr_1view.default_cr_output_path(
            Path("/data/ysj/result/4dgs/RTGS/jumpingjacks"),
            split="test",
            camera_index=0,
            timestamp=0.0,
            run_label="cr_one_view",
        )

        self.assertIn("rtgs_cr_1view", path.parts)
        self.assertEqual(path.name, "cr_one_view")

    def test_parser_defaults_compare_against_official_render(self):
        args = cr_1view.build_parser().parse_args(["--dataset-kind", "dnerf"])

        self.assertTrue(args.compare_official)
        self.assertEqual(args.views, 1)
        self.assertEqual(args.camera_index, 0)


if __name__ == "__main__":
    unittest.main()
```

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_cr_1view -v
```

Expected: FAIL because `cr_1view` does not exist yet.

- [x] **Step 3: Share official-flow loading with Task 1**

`cr_1view.py` must reuse the Task 1 official loader so camera/model/path semantics match. It must not call `load_rtgs_checkpoint()` from the current wrapper path until the official-flow baseline and wrapper path have been compared and the difference is documented.

- [x] **Step 4: Add snapshot-reference diagnostics**

Before blaming CR, render or otherwise validate a snapshot-reference path using the same timestamp-conditioned mean/covariance/opacity/color tensors that CR will consume. Save the snapshot tensor statistics and compare against official RTGS default output when feasible. For `rot_4d=True`, SH/4D SH color direction must use timestamp-conditioned means, not only base `pc.get_xyz`.

Expected manifest fields:

```text
gaussians_total
gaussians_after_temporal_mask
means_shape
covars_shape
opacities_shape
colors_shape
color_direction_source
tensor_nan_inf_counts
opacity_min_max_mean
color_min_max_mean
covariance_sanity
projection_sanity_rtgs_vs_gsplat
official_vs_snapshot_metrics
```

- [x] **Step 5: Materialize RTGS snapshot for one timestamp**

Use the existing materialization concepts from `experiment/src/lkg_experiment/rtgs_coherent/cli.py`:

```text
materialize_rtgs_geometry
evaluate_rtgs_colors
snapshot_from_geometry
```

The snapshot must be produced from the official `GaussianModel` object used for Task 1, not from `RtgsLiteGaussianModel`.

- [x] **Step 6: Render CR with a 1-view lookup**

Use the same invariant from Task 2:

```text
view_idx_matrix: all zeros
subpixel_coord_matrix: generated through coherent-raster lookup helpers
adjacent_viewmats: one camera only, shape [1, 1, 4, 4]
Ks: one camera only, shape [1, 3, 3]
```

Save:

```text
gt.png
rtgs_official_render.png
rtgs_snapshot_reference_render.png
rtgs_cr_render.png
comparison_official_vs_snapshot.png
comparison_snapshot_vs_cr.png
comparison_official_vs_cr.png
metrics.json
manifest.json
```

- [x] **Step 7: Run non-CUDA tests**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_cr_1view -v
PYTHONPATH=src python -m unittest discover -s tests -v
```

Expected: PASS.

- [x] **Step 8: Run CUDA CR 1-view comparison**

Run:

```bash
cd experiment
PYTHONPATH=src python rtgs_cr_1view.py \
  --dataset-kind dnerf \
  --model-path /data/ysj/result/4dgs/RTGS/jumpingjacks \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --dataset-root /data/ysj/dataset/dnerf \
  --split test \
  --camera-index 0 \
  --run-label jumpingjacks_test0_cr_1view
```

Expected:

```text
generated/rtgs_cr_1view/jumpingjacks/jumpingjacks_test0_cr_1view/rtgs_cr_render.png
generated/rtgs_cr_1view/jumpingjacks/jumpingjacks_test0_cr_1view/metrics.json
```

- [x] **Step 9: Run CUDA CR 1-view comparison for N3DV**

Run:

```bash
cd experiment
PYTHONPATH=src python rtgs_cr_1view.py \
  --dataset-kind n3dv \
  --model-path /data/ysj/result/4dgs/RTGS/coffee_martini \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --n3dv-root /data/ysj/dataset/N3DV \
  --split test \
  --camera-index 0 \
  --n3dv-frame-index 0 \
  --run-label coffee_martini_cam00_0000_cr_1view
```

Expected:

```text
generated/rtgs_cr_1view/coffee_martini/coffee_martini_cam00_0000_cr_1view/rtgs_cr_render.png
generated/rtgs_cr_1view/coffee_martini/coffee_martini_cam00_0000_cr_1view/metrics.json
```

- [x] **Step 10: Gate Task 3 completion**

Task 3 is complete only when:

```text
rtgs_cr_render.png is visually valid
official-vs-snapshot passes the initial alert threshold before official-vs-CR is used to judge CR quality
snapshot-vs-CR is the primary CR adapter correctness metric
official-vs-CR is recorded as an end-to-end diagnostic
rtgs_cr_render.png is compared against rtgs_official_render.png from the same camera after the snapshot gate passes
metrics.json records PSNR/MAE/MSE for official vs snapshot, snapshot vs CR, and CR vs official
initial numeric alert thresholds from experiment/docs/rtgs_coherent_raster_brainstorm.md are evaluated
the difference source is documented if CR and official do not match
```

Stop after reporting Task 3 status. Do not start Task 4 without user confirmation.

**Milestone 3 Complete, 2026-06-29:** dnerf and N3DV both produce valid 1-view CR renders under the external generated directory. The latest verification command was:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_official_1view tests.test_rtgs_cr_1view -v
```

Result: 31 tests passed. The user visually confirmed the N3DV `rtgscompat` output is normal. Task 4 remains gated until explicit user confirmation.

---

### Task 4: RTGS + Coherent-Raster Multi-View LKG Extension

**Files:**
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/views66.py`
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/cli.py`
- Modify: `experiment/scripts/run_rtgs_66_lkg_all.sh`
- Modify: `experiment/tests/test_rtgs_66_lkg_script.py`
- Modify: `experiment/docs/rtgs_coherent_raster_brainstorm.md`

- [ ] **Step 1: Promote Task 3 snapshot path into multi-view**

`views66.py` must use the same model, camera, timestamp, color, covariance, opacity, and background path validated by Task 3. The only new variable in Task 4 is view synthesis and LKG viewpoint indexing. This is a blocking gate: do not accept metrics from the current `load_rtgs_checkpoint()` / lightweight wrapper path as Task 4 evidence.

- [ ] **Step 2: Preserve N3DV synthetic camera sentinel**

When `rtgs_camera_from_gsplat_viewmat()` or its replacement creates sampled official comparison cameras from an N3DV anchor, it must preserve `FoVx=FoVy=-1.0` and use `fl_x/fl_y/cx/cy` for projection. Add a manifest assertion and unit test for this before trusting sampled N3DV comparisons.

- [ ] **Step 3: Add remapping control and cluster map validation**

Expose a CR remapping switch instead of hardcoding `use_remapping=True`. Use this CLI shape unless implementation discovers a better local convention:

```text
--cr-remapping {on,off}
```

Before rendering quality images, validate:

```text
viewpoint_index.max() < source_view_count
viewpoint_index.min() >= 0
non-divisible cluster tails are deterministic
color-coded view-index debug render matches expected subpixel placement
```

- [ ] **Step 4: Compare sampled views against official RTGS**

Keep `--compare-original-views` enabled for a small sample of synthesized views. Each sampled view must save:

```text
official_view.png
coherent_view.png
comparison.png
metrics.json
```

- [ ] **Step 5: Run isolated multi-view sequence before LKG completion**

Run in this order:

```text
first: cluster_size=1, remapping disabled, linear debug map, sampled full-frame comparisons only
second: cluster_size=1, remapping enabled, linear debug map, sampled full-frame comparisons only
third: cluster_size=1, remapping enabled, real LKG map, sampled comparisons plus interlaced output
fourth: cluster_size=2 or 4, remapping enabled, sampled comparisons
fifth: cluster_size=8, remapping enabled, sampled comparisons plus interlaced output
```

- [ ] **Step 6: Render LKG interlaced output**

Use the existing calibration artifact unless a different mapping is explicitly selected:

```text
/data/ysj/result/coherent-raster/generated/lkg_go_1440x2560_66_views_lkg_calibration.npz
```

Expected output:

```text
/data/ysj/result/coherent-raster/generated/rtgs_coherent/<scene>/<run_label>/rtgs_coherent_lkg.png
/data/ysj/result/coherent-raster/generated/rtgs_coherent/<scene>/<run_label>/metrics.json
```

- [ ] **Step 7: Run tests and CUDA smoke**

Run non-CUDA tests:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_rtgs_66_lkg_script -v
PYTHONPATH=src python -m unittest discover -s tests -v
```

Run CUDA smoke:

```bash
cd experiment
PYTHONPATH=src python rtgs_coherent.py \
  --render-mode views66 \
  --dataset-kind dnerf \
  --model-path /data/ysj/result/4dgs/RTGS/jumpingjacks \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --dataset-root /data/ysj/dataset/dnerf \
  --split test \
  --camera-index 0 \
  --views 66 \
  --cluster-size 1 \
  --cr-remapping off \
  --map-mode linear \
  --compare-original-views 5 \
  --no-write-interlaced \
  --run-label jumpingjacks_test0_lkg_66
```

Expected: PASS for tests and valid sampled view comparisons. A visually valid `rtgs_coherent_lkg.png` is required only after the isolated sequence allows real LKG interlaced output.

- [ ] **Step 8: Gate multi-view completion**

Task 4 is complete only when:

```text
sampled coherent views are visually valid
sampled coherent views have recorded metrics against official RTGS
interlaced LKG image is visually valid on the target display
manifest records mapping artifact path and view synthesis parameters
```

Stop after reporting Task 4 status and ask whether to commit or open a PR.

## Review Checklist

- [x] Task 1 establishes a commit-able RTGS official 1-view baseline.
- [x] Task 2 contains no code changes.
- [x] Task 3 keeps RTGS + CR 1-view separate from official RTGS 1-view.
- [ ] Task 4 starts only after Task 3 is visually and metrically acceptable.
- [x] No phase overwrites another phase's generated output directory.
- [x] Every phase has a stop point for user review.
