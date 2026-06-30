# OMG4-FTGS CoherentRaster 1-View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add and verify a CoherentRaster single-view renderer for OMG4-FTGS.

**Architecture:** Keep OMG4 model and camera loading unchanged. Add a renderer selector in the CLI, and add a CoherentRaster helper beside the existing gsplat helper so both paths consume the same materialized splats, view matrix, intrinsics, width, height, and timestamp.

**Tech Stack:** Python, unittest, PyTorch, gsplat, CoherentRaster lookup helpers, PIL, ffmpeg/ffprobe for existing media handling.

---

### Task 1: Renderer Selection Tests

**Files:**
- Modify: `experiment/tests/test_omg4_ftgs.py`
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/cli.py`

- [ ] **Step 1: Write the failing parser test**

```python
def test_parser_accepts_renderer_modes(self):
    default_args = cli.build_parser().parse_args([])
    self.assertEqual(default_args.renderer, "gsplat")

    coherent_args = cli.build_parser().parse_args(["--renderer", "coherent"])
    self.assertEqual(coherent_args.renderer, "coherent")

    both_args = cli.build_parser().parse_args(["--renderer", "both"])
    self.assertEqual(both_args.renderer, "both")
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs.Omg4FtgsTest.test_parser_accepts_renderer_modes -v
```

Expected: FAIL because `--renderer` is not implemented.

- [ ] **Step 3: Add `--renderer` to the parser**

Add:

```python
parser.add_argument("--renderer", choices=("gsplat", "coherent", "both"), default="gsplat")
```

- [ ] **Step 4: Re-run the focused test**

Expected: PASS.

### Task 2: CoherentRaster Helper Tests

**Files:**
- Modify: `experiment/tests/test_omg4_ftgs.py`
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/render.py`

- [ ] **Step 1: Write a failing mocked helper test**

Patch `gsplat.rendering_coherent_raster.rasterization_CR` and `coherent_raster.utils.utils_coherent_raster` so the test can verify that `render_splats_coherent` passes `adjacent_viewmats`, lookup matrices, dimensions, and splat tensors.

- [ ] **Step 2: Run the focused test and verify it fails**

Expected: FAIL because `render_splats_coherent` is missing.

- [ ] **Step 3: Implement `render_splats_coherent`**

Use the RTGS 1-view pattern: all-zero viewpoint index, `build_cr_lookup_arrays`, `lookup_arrays_to_torch`, `rasterization_CR`, `unpatchify_image_shape_matrix`, `unpad`, then return CHW float RGB.

- [ ] **Step 4: Re-run the focused helper test**

Expected: PASS.

### Task 3: Single-View Output Tests

**Files:**
- Modify: `experiment/tests/test_omg4_ftgs.py`
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/cli.py`

- [ ] **Step 1: Write a failing mocked `--renderer both` single-view test**

Verify that `single_gsplat.png`, `single_coherent.png`, `single_diff.png`, and manifest renderer entries are written.

- [ ] **Step 2: Run the focused test and verify it fails**

Expected: FAIL because the CLI only writes `single.png`.

- [ ] **Step 3: Implement renderer dispatch and metrics**

For `gsplat`, keep `single.png`. For `coherent`, write `single_coherent.png`. For `both`, write both renderer outputs plus `single_diff.png` and simple MSE/MAE/PSNR metrics.

- [ ] **Step 4: Run OMG4 tests**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
conda run -n rtgs-coherent-cu121 env PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: PASS.

### Task 4: GPU Verification

**Files:**
- Runtime outputs under `/data/ysj/result/coherent-raster/generated/omg4_ftgs`

- [ ] **Step 1: Run outside sandbox on GPU**

```bash
cd experiment
conda run -n rtgs-coherent-cu121 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python omg4_ftgs_1view.py --mode single --renderer both --run-label cook_spinach_cr_single
```

- [ ] **Step 2: Verify outputs**

Check `manifest.json`, file sizes, and open `single_gsplat.png` and `single_coherent.png`.
