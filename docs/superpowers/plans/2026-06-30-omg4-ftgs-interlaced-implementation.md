# OMG4-FTGS Multi-View Interlaced Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a CoherentRaster multi-view interlaced renderer for one OMG4-FTGS camera timestamp.

**Architecture:** Keep OMG4 checkpoint, camera, and single-view rendering code intact. Add focused interlaced helpers to `lkg_experiment.omg4_ftgs.render`, and route `--mode interlaced` from the existing CLI so the same materialized splats feed grouped CoherentRaster rendering.

**Tech Stack:** Python, unittest, PyTorch, gsplat CoherentRaster, existing LKG mapping helpers, PIL.

---

### Task 1: Parser and Map Selection

**Files:**
- Modify: `experiment/tests/test_omg4_ftgs.py`
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/cli.py`
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/render.py`

- [ ] **Step 1: Write failing parser/map tests**

Add tests that parse `--mode interlaced`, assert defaults `views=66`, `panel_width=1440`, `panel_height=2560`, `cluster_size=8`, and verify a helper can build a linear viewpoint index with shape `(height, width, 3)`.

- [ ] **Step 2: Verify RED**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs.Omg4FtgsTest.test_parser_accepts_interlaced_defaults tests.test_omg4_ftgs.Omg4FtgsTest.test_build_interlaced_viewpoint_index_linear -v
```

Expected: FAIL because `interlaced` parser options and helper are missing.

- [ ] **Step 3: Implement parser defaults and map helper**

Add parser arguments for interlaced rendering and add `build_interlaced_viewpoint_index(args, width, height)` that uses `load_viewpoint_index_file` for file mode and `build_linear_viewpoint_index` for linear mode.

- [ ] **Step 4: Verify GREEN**

Re-run the focused tests. Expected: PASS.

### Task 2: Orbit View Matrix Helpers

**Files:**
- Modify: `experiment/tests/test_omg4_ftgs.py`
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/render.py`

- [ ] **Step 1: Write failing orbit helper tests**

Add tests for `estimate_orbit_center` and `synthesize_interlaced_viewmats`, checking grouped shape `(ceil(views / cluster_size), cluster_size, 4, 4)`.

- [ ] **Step 2: Verify RED**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs.Omg4FtgsTest.test_synthesize_interlaced_viewmats_groups_views -v
```

Expected: FAIL because helpers are missing.

- [ ] **Step 3: Implement helper functions**

Use `OrbitViewSynthesizer` and the same orbit center estimate as the RTGS path: explicit `orbit_center_distance` when positive, otherwise average splat means relative to the source camera position.

- [ ] **Step 4: Verify GREEN**

Re-run the focused test. Expected: PASS.

### Task 3: CoherentRaster Interlaced Renderer

**Files:**
- Modify: `experiment/tests/test_omg4_ftgs.py`
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/render.py`

- [ ] **Step 1: Write failing mocked renderer test**

Patch `rasterization_CR`, `unpatchify_image_shape_matrix`, and `unpad`. Verify `render_splats_interlaced_coherent` passes grouped view matrices, lookup tensors, panel dimensions, splat tensors, SH degree, and returns a CHW image.

- [ ] **Step 2: Verify RED**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs.Omg4FtgsTest.test_render_splats_interlaced_coherent_uses_grouped_views_and_mapping -v
```

Expected: FAIL because the interlaced renderer is missing.

- [ ] **Step 3: Implement renderer**

Add `render_splats_interlaced_coherent` beside the single-view CoherentRaster helper. Use `build_cr_lookup_arrays`, `lookup_arrays_to_torch`, grouped adjacent view matrices, and `rasterization_CR`.

- [ ] **Step 4: Verify GREEN**

Re-run the focused test. Expected: PASS.

### Task 4: CLI Interlaced Output

**Files:**
- Modify: `experiment/tests/test_omg4_ftgs.py`
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/cli.py`

- [ ] **Step 1: Write failing mocked CLI test**

Patch model/camera/frame loading and `render_splats_interlaced_coherent`. Verify `--mode interlaced --map-mode linear --panel-width 4 --panel-height 3 --views 4 --cluster-size 2` writes `omg4_ftgs_lkg_interlaced.png` and manifest `interlaced`.

- [ ] **Step 2: Verify RED**

Run:

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs.Omg4FtgsTest.test_interlaced_mode_writes_panel_and_manifest_with_mock_renderer -v
```

Expected: FAIL because CLI dispatch is missing.

- [ ] **Step 3: Implement CLI dispatch**

Add `manifest["interlaced"]`, route `--mode interlaced`, save the interlaced PNG, and write panel/render/map/orbit metadata.

- [ ] **Step 4: Verify GREEN**

Re-run the focused test. Expected: PASS.

### Task 5: Full Verification

**Files:**
- Runtime outputs under `/data/ysj/result/coherent-raster/generated/omg4_ftgs`

- [ ] **Step 1: Run unit tests**

```bash
cd experiment
PYTHONPATH=src python -m unittest tests.test_omg4_ftgs tests.test_package_layout tests.test_lkg_experiment_defaults -v
conda run -n rtgs-coherent-cu121 env PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: all tests pass.

- [ ] **Step 2: Run GPU interlaced render outside sandbox**

```bash
cd experiment
conda run -n rtgs-coherent-cu121 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python omg4_ftgs_1view.py --mode interlaced --run-label cook_spinach_lkg_interlaced
```

Expected: exit code 0, `omg4_ftgs_lkg_interlaced.png` exists.

- [ ] **Step 3: Verify artifacts**

Use PIL to verify image mode and size. Use `python -m json.tool` to verify manifest fields.
