# OMG4-FTGS CR Lookup Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move OMG4-FTGS interlaced CoherentRaster lookup construction out of the per-render hot path and expose timing breakdowns that separate lookup setup from CR core rendering.

**Architecture:** Build the LKG CR lookup once after the panel viewpoint map is loaded, transfer the lookup tensors to the target device once, and pass the prebuilt tensors into `render_splats_interlaced_coherent()`. Keep the existing `viewpoint_index` API as a compatibility fallback for tests and one-off callers, but make the CLI interlaced path use the prebuilt lookup.

**Tech Stack:** Python, PyTorch, NumPy, gsplat CoherentRaster, `unittest`.

---

## Problem Summary

The current OMG4 interlaced manifest reports `render_ms=1631.56 ms` for `1440x2560`, 66 views, cluster size 8. A stage profiler showed that the CR kernel is not the bottleneck:

- `materialize`: 3.42 ms
- `build_cr_lookup_arrays()`: 1110.61 ms
- `lookup_arrays_to_torch()`: 39.23 ms
- `rasterization_CR()`: 42.02 ms
- `unpatchify + unpad`: 0.33 ms

The expensive lookup work is currently done inside `render_splats_interlaced_coherent()` and is included in the CLI render timer. RTGS one-shot metrics do not include this lookup construction cost, so the current comparison is not fair.

## Files

- Modify: `experiment/src/lkg_experiment/omg4_ftgs/render.py`
  - Add a small `PreparedCrLookup` dataclass.
  - Add `prepare_cr_lookup_tensors()`.
  - Allow `render_splats_interlaced_coherent()` to accept prebuilt `view_idx_matrix` and `subpixel_coord_matrix`.
  - Add optional `return_meta` and `return_timing` support so callers can record CR internal timing.

- Modify: `experiment/src/lkg_experiment/omg4_ftgs/cli.py`
  - Prepare lookup tensors before starting the render timer.
  - Record `lookup_cpu_ms`, `lookup_h2d_ms`, `cr_core_ms`, `post_ms`, and `frame_ms_excluding_lookup`.
  - Keep the existing image output and manifest fields.

- Modify: `experiment/tests/test_omg4_ftgs.py`
  - Add a failing test proving the interlaced renderer can use prebuilt lookup tensors without calling `build_cr_lookup_arrays()` in the hot path.
  - Add a CLI manifest test proving lookup timing is present and prebuilt tensors are passed to the renderer.

## Behavioral Requirements

- Existing single-view coherent rendering keeps its current behavior.
- Existing callers that pass only `viewpoint_index` to `render_splats_interlaced_coherent()` still work.
- The OMG4 interlaced CLI prepares lookup tensors once per frame before the render timer starts.
- The interlaced manifest keeps `interlaced.render_ms` for backward compatibility, but adds explicit timing fields:
  - `lookup_cpu_ms`
  - `lookup_h2d_ms`
  - `frame_ms_excluding_lookup`
  - `cr_timing_ms`
  - `post_ms` when available
- The CR output image must remain pixel-equivalent for the same input map, camera, splats, and renderer settings.

## Risk Check Before Implementation

- **API compatibility:** Low risk. The renderer can keep `viewpoint_index` as an optional fallback and add optional prebuilt tensor arguments.
- **Device mismatch:** Medium risk. `prepare_cr_lookup_tensors()` must transfer tensors to the requested device and the renderer should use them directly. Tests should assert object identity so accidental rebuilds are caught.
- **Timing accuracy:** Medium risk. CPU lookup timing and H2D timing need CUDA synchronization when the target is CUDA. Use the existing CLI `_sync_device()` helper around timed blocks.
- **Memory use:** Low to medium risk. Prebuilding lookup tensors does not add a new steady-state allocation compared with the current path; it only extends lifetime until render finishes. This is acceptable for one interlaced frame and necessary for video/interactive reuse later.
- **RTGS comparison fairness:** This change makes OMG4 manifest report both lookup-inclusive and lookup-exclusive timings. It does not alter RTGS metrics.

## Tasks

### Task 1: Add Prebuilt Lookup Renderer Test

**Files:**
- Modify: `experiment/tests/test_omg4_ftgs.py`

- [ ] **Step 1: Write the failing test**

Add a test that patches `build_cr_lookup_arrays` to raise if called and passes explicit `view_idx_matrix` and `subpixel_coord_matrix` to `render_splats_interlaced_coherent()`.

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_omg4_ftgs.Omg4FtgsTest.test_render_splats_interlaced_coherent_uses_prebuilt_lookup_tensors -v
```

Expected: fail because `render_splats_interlaced_coherent()` does not yet accept the prebuilt tensor arguments.

### Task 2: Implement Prebuilt Lookup Support

**Files:**
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/render.py`

- [ ] **Step 1: Add `PreparedCrLookup` and `prepare_cr_lookup_tensors()`**

The helper should call `build_cr_lookup_arrays()`, convert to torch tensors, and return both tensors plus timing fields.

- [ ] **Step 2: Update `render_splats_interlaced_coherent()`**

Accept optional `view_idx_matrix`, `subpixel_coord_matrix`, `return_meta`, and `return_timing`. If tensors are absent, keep the existing fallback path that builds them from `viewpoint_index`.

- [ ] **Step 3: Run the renderer test**

Run the test from Task 1. Expected: pass.

### Task 3: Move CLI Lookup Preparation Out of Render Timer

**Files:**
- Modify: `experiment/src/lkg_experiment/omg4_ftgs/cli.py`
- Modify: `experiment/tests/test_omg4_ftgs.py`

- [ ] **Step 1: Write the failing CLI manifest test**

Update the interlaced mock test to assert that `view_idx_matrix` and `subpixel_coord_matrix` are passed to the renderer and that manifest timing includes lookup fields.

- [ ] **Step 2: Run the test and confirm failure**

Run:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_omg4_ftgs.Omg4FtgsTest.test_interlaced_mode_writes_panel_and_manifest_with_mock_renderer -v
```

Expected: fail because the CLI still passes `viewpoint_index` only and does not write lookup timing fields.

- [ ] **Step 3: Implement CLI preparation**

Call `prepare_cr_lookup_tensors()` immediately after `build_interlaced_viewpoint_index()`. Start `render_ms` timing after lookup preparation. Pass the prepared tensors to `render_splats_interlaced_coherent()`.

- [ ] **Step 4: Record manifest timing**

Add lookup and CR timing fields without removing existing fields.

### Task 4: Verify and Benchmark

**Files:**
- No production file changes unless tests reveal an issue.

- [ ] **Step 1: Run unit tests**

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_omg4_ftgs -v
```

Expected: all OMG4 tests pass.

- [ ] **Step 2: Run package/default smoke tests**

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_package_layout tests.test_lkg_experiment_defaults -v
```

Expected: pass.

- [ ] **Step 3: Run GPU interlaced smoke benchmark**

```bash
cd /home/ysj/lkg-experiment/experiment
conda run -n rtgs-coherent-cu121 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python omg4_ftgs_1view.py --mode interlaced --run-label cook_spinach_lkg_interlaced_lookup_cached
```

Expected: output image is written, manifest has `lookup_cpu_ms`, `lookup_h2d_ms`, and `frame_ms_excluding_lookup`. `frame_ms_excluding_lookup` should be close to the profiled CR/core scope, not the previous 1.6s lookup-inclusive value.

For a single CLI process with no warmup, `frame_ms_excluding_lookup` may still include first CR kernel cold-start overhead. If that happens, run a same-process repeated benchmark with prebuilt lookup tensors and treat the post-warmup average as the steady-state comparison value.

## Self-Review

- No placeholders remain.
- The plan changes only OMG4-FTGS CR lookup handling and manifest timing.
- Existing caller compatibility is preserved by keeping `viewpoint_index` fallback.
- Tests are written before production code changes.
