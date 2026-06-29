# RTGS + CoherentRaster One-Shot Clustered Rendering Design

Created: 2026-06-29

## Status

Reviewed by sub-agent and revised.

This document plans the next RTGS + CoherentRaster step: replacing the current `compose` correctness path, which loops over every synthetic view, with a 3DGS CoherentRaster-style one-shot clustered render that directly produces the LKG interlaced image.

## Problem

The current verified RTGS + CR 66-view path is correct but not the same execution model as the CoherentRaster paper or the existing 3DGS experiment path.

Current `compose` behavior:

```text
for view_id in 0..65:
  build synthetic RTGS camera
  evaluate RTGS view-dependent color
  apply N3DV RTGS-compatible projection adapter when required
  call CR as a single-view renderer
  copy that view's selected subpixels into the interlaced tensor
```

This produces one final interlaced image, but it performs 66 independent rasterizer calls per frame. With timing defaults of `warmup_iters=3` and `measure_iters=5`, one experiment row can perform `8 * 66 = 528` CR single-view renders.

Target one-shot behavior:

```text
build all synthetic views
group views into clusters
build one LKG subpixel-to-view lookup
call rasterization_CR once
return one interlaced tensor
```

This is how the 3DGS CoherentRaster experiment path works.

## Existing Reference

The 3DGS path in `experiment/src/lkg_experiment/coherent_default/run_coherent_raster_experiment.py` already performs one-shot CR rendering:

```text
lookup = build_cr_lookup_arrays(viewpoint_index, tile_size, use_remapping)
view_idx_matrix, subpixel_coord_matrix = lookup_arrays_to_torch(lookup)
adjacent_viewmats = OrbitViewSynthesizer(..., cluster_size=variant.cluster_size)(c2w, orbit_center)
interlaced, timing = time_interlaced_render(
  coherent_renderer,
  adjacent_viewmats=adjacent_viewmats,
  K=K,
  view_idx_matrix=view_idx_matrix,
  subpixel_coord_matrix=subpixel_coord_matrix,
)
```

The underlying `gsplat.rendering_coherent_raster.rasterization_CR` contract is:

```text
means: float tensor [N, 3]
covars: float tensor [N, 6], or quats [N,4] and scales [N,3]
opacities: float tensor [N]
colors:
  precomputed RGB [N,3], broadcast to every cluster reference; or
  precomputed RGB [C,N,3], one color set per cluster reference; or
  SH coefficients [N,K,3] / [C,N,K,3]
adjacent_viewmats: float tensor [C, S, 4, 4]
Ks: float tensor [1,3,3] for the currently supported shared-intrinsics path
view_idx_matrix: uint32 tensor [tile_h,tile_w,3,tile_size,tile_size]
subpixel_coord_matrix: uint32 tensor [tile_h,tile_w,3,tile_size,tile_size,3]
raw CR output: patchified float tensor [tile_h,tile_w,3,tile_size,tile_size]
final output: [3,H,W] after unpatchify_image_shape_matrix() and unpad(H,W)
```

Where `C` is the number of view clusters and `S` is cluster size.

The current CR path is not a general per-camera-intrinsics renderer. `rasterization_CR()` internally repeats a shared `Ks` tensor for cluster reference projection and `isect_tiles_CR()` receives only `Ks[0]`. Therefore the one-shot plan must keep one shared viewport-size `K` unless the CR wrapper and tile-intersection code are extended.

## Key RTGS Differences

RTGS introduces three differences from the static 3DGS path.

### Time-Conditioned Geometry

For one timestamp, RTGS geometry can still be materialized once:

```text
GaussianModel + timestamp
  -> means [N,3]
  -> covars [N,6]
  -> opacities [N]
  -> active temporal mask
```

This part is compatible with one-shot CR.

### View-Dependent Color

RTGS color depends on camera center through SH or 4D SH evaluation. For clustered CR this maps naturally to the paper's reuse rule:

```text
colors_per_cluster = evaluate_rtgs_colors_for_centers(cluster_reference_centers)
colors_per_cluster shape: [C, N, 3]
```

For `cluster_size=1`, `C == 66`, so colors are per view and should match the compose baseline.

For `cluster_size>1`, colors are reused from each cluster representative, matching CoherentRaster's cross-view attribute reuse.

The representative view is the middle view inside each group:

```text
reference_index = cluster_size // 2
reference_centers = inverse(adjacent_viewmats[:, reference_index])[:, :3, 3]
```

### N3DV Sentinel-FoV Projection Adapter

N3DV RTGS checkpoints use explicit intrinsics with sentinel FoV values:

```text
FoVx = -1.0
FoVy = -1.0
fl_x > 0
fl_y > 0
```

The verified 1-view and compose paths handle this using `adapt_snapshot_for_rtgs_compat_projection()`. That adapter is camera-projection dependent:

```text
camera-space mean x *= scale_x(view)
camera-space mean y *= scale_y(view)
K.fx = fov-derived fx
K.fy = fov-derived fy
3D covars are not directly scaled
```

For the current N3DV cameras the scale values are expected to be shared across the synthetic views because width, height, explicit focal length, and sentinel FoV are shared. The implementation should still store adapter metadata in a broadcastable form:

```text
rtgs_mean_camera_scales: [1,2] for current shared scale
optional future shape: [C,S,2] if view-specific intrinsics are introduced
```

This cannot be represented by the current `rasterization_CR` Python API because it accepts one global `means [N,3]` tensor for all views. It also cannot be fixed by changing the Python caller alone: the CR CUDA path computes both reference projection and adjacent-view translation from the unadapted 3D mean. This is the main blocker for a correct N3DV one-shot path.

## Recommended Architecture

Current status: `cluster_size=1` one-shot has matched the old compose baseline for dnerf and N3DV, so `rtgs_cr_experiment.py` is now a one-shot-only experiment runner. The user-facing engine name is `one_shot`; the old `compose` and `clustered` experiment-engine names should not be used in new commands.

The implementation should introduce a focused one-shot module instead of growing `cr_experiment.py` further:

```text
experiment/src/lkg_experiment/rtgs_coherent/cr_one_shot.py
```

Responsibilities:

- build viewport-aware CR lookup tensors;
- synthesize grouped view matrices;
- evaluate RTGS colors at cluster representative centers;
- build optional RTGS projection-adapter metadata;
- call `rasterization_CR` once;
- unpatchify/unpad the viewport output;
- paste the viewport output into the final panel-sized interlaced tensor.

`cr_experiment.py` should time one-shot variants, save artifacts, and write metrics.

## Viewport Handling

The one-shot path supports aspect-preserving 1440x2560 rendering by rendering only the content viewport and placing it inside the panel.

The one-shot path should support the same viewport contract. The viewport crop must happen before lookup construction:

```text
panel viewpoint_index: [panel_h, panel_w, 3]
viewport = resolve_aspect_viewport(...)
viewport_viewpoint_index = panel viewpoint_index[
  offset_y : offset_y + render_height,
  offset_x : offset_x + render_width,
  :
]
```

Then call CR once for the viewport:

```text
rasterization_CR(
  width=render_width,
  height=render_height,
  view_idx_matrix=lookup(viewport_viewpoint_index),
  subpixel_coord_matrix=lookup(viewport_viewpoint_index)
)
unpatchify_image_shape_matrix(...)
unpad(render_height, render_width)
```

After rendering:

```text
panel = background-filled [3, panel_h, panel_w]
panel[:, offset_y:offset_y+render_height, offset_x:offset_x+render_width] = viewport_interlaced
```

This avoids stretching N3DV into the full 9:16 panel and avoids adding a full-panel mask to the CUDA kernel.

## Engine Phases

### Phase A: dnerf / Positive-FoV One-Shot

Goal: prove the Python-side one-shot plumbing matches the existing 3DGS CR path.

Inputs:

```text
geometry.means: [N,3]
geometry.covars: [N,6]
geometry.opacities: [N]
colors_per_cluster: [C,N,3]
adjacent_viewmats: [C,S,4,4]
K: [1,3,3]
viewport lookup tensors
```

Required behavior:

- `cluster_size=1` must produce an interlaced image that closely matches `compose`.
- `cluster_size=2,4,8` can trade quality for speed and should be measured against `compose`.
- dnerf should not need the N3DV projection adapter.

Acceptance:

```text
dnerf cluster_size=1:
  one-shot vs compose interlaced PSNR >= 55 dB, or explain the numeric difference
  sampled one-shot views visually align with official RTGS
  one-shot performs one rasterization_CR call per measured frame
```

### Phase B: N3DV One-Shot With Adapter Disabled As A Diagnostic

Goal: reproduce the known bad behavior intentionally so the failure mode is measurable.

Run N3DV with:

```text
rtgs_compat_projection = false
cluster_size = 1
```

Expected:

- output resembles the pre-FoV-fix bad render;
- metrics are worse than compose;
- manifest records that this is a diagnostic, not acceptance.

This protects against accidentally accepting a visually wrong N3DV one-shot result.

### Phase C: N3DV One-Shot With Kernel-Level Mean Adapter

Goal: make one-shot CR match the verified compose path for N3DV.

Add an optional RTGS projection adapter to the CR projection path:

```text
rtgs_mean2d_adapter_enabled: bool
rtgs_mean_camera_scales: [1,2] or [C,S,2]
```

The adapter should apply the same camera-space x/y mean scaling currently performed by `adapt_snapshot_for_rtgs_compat_projection()`, but inside every CR CUDA path that converts world means to image-space centers.

Required insertion points:

```text
1. Reference projection:
   gsplat/gsplat/cuda/csrc/CoherentRaster_ModifiedKernel.cu
   projection_ewa_3dgs_fused_fwd_kernel_CR()
   after posW2C(..., mean_c), before persp_proj(...)

2. Adjacent-view translation generation:
   gsplat/gsplat/cuda/csrc/CoherentRaster_CRKernel.cu
   intersect_tile_kernel_CR_AccuTile()
   after world_to_cam_coord(..., adj_mean_cam), before cam_to_image_coord(...)
```

Important rule:

```text
scale camera-space mean x/y before projection;
use FoV-derived shared K at the viewport render size;
do not directly scale the input 3D covars;
allow the existing projection function to compute conics from adapted mean_c and unchanged covar_c.
```

This mirrors the verified Python adapter, where adapted means are passed with unchanged `covars`.

Implementation location:

```text
gsplat/gsplat/rendering_coherent_raster.py
gsplat/gsplat/cuda/_wrapper_coherent_raster.py
gsplat/gsplat/cuda/ext.cpp
gsplat/gsplat/cuda/include/Ops_CoherentRaster.h
gsplat/gsplat/cuda/csrc/CoherentRaster.cpp
gsplat/gsplat/cuda/csrc/CoherentRaster.h
gsplat/gsplat/cuda/csrc/CoherentRaster_Modified.cuh
gsplat/gsplat/cuda/csrc/CoherentRaster_ModifiedWrapper.cu
gsplat/gsplat/cuda/csrc/CoherentRaster_ModifiedKernel.cu
gsplat/gsplat/cuda/csrc/CoherentRaster_CR.cuh
gsplat/gsplat/cuda/csrc/CoherentRaster_CRKernel.cu
```

If the existing CUDA projection kernel cannot support this cleanly, the fallback is to extend the Python API to accept per-camera means:

```text
means_per_view: [V,N,3]
```

That fallback is less desirable because it increases memory pressure and deviates from the current CR kernel structure.

Acceptance:

```text
tiny synthetic kernel-adapter test:
  CR kernel adapter output matches existing Python adapter within a tight tolerance

N3DV cluster_size=1:
  one-shot sampled views match compose sampled views within 0.2 dB PSNR
  one-shot interlaced output matches compose interlaced output closely
  output does not regress to the old sentinel-FoV failure appearance
```

### Phase D: Cluster Reuse Quality/Speed Sweep

After `cluster_size=1` passes for both datasets:

```text
cluster_size=2
cluster_size=4
cluster_size=8
cluster_size=16
without_remap
without_reuse
```

Report:

- FPS;
- frame time;
- peak VRAM;
- one-shot vs compose interlaced PSNR;
- sampled view PSNR against official RTGS;
- dataset kind, scene, timestamp, viewport, adapter status.

## Required Tests

Unit tests should be added before implementation.

### Viewport Lookup Tests

File:

```text
experiment/tests/test_rtgs_cr_one_shot.py
```

Cases:

- crops `viewpoint_index` to `AspectViewport`;
- lookup dimensions match `render_width/render_height`, not panel size;
- final paste restores panel-sized tensor;
- letterbox area is background-filled.

### Input Shape Tests

Cases:

- `colors_per_cluster` shape is `[num_clusters, gaussian_count, 3]`;
- `adjacent_viewmats` shape is `[num_clusters, cluster_size, 4, 4]`;
- `cluster_size=1` creates one reference color per view;
- padded final cluster duplicates the last view consistently.

### Adapter Decision Tests

Cases:

- dnerf positive FoV disables adapter;
- N3DV sentinel FoV enables adapter;
- adapter metadata records scales per view or per reference view;
- `clustered` accepts N3DV sentinel-FoV cameras when the kernel adapter is available.
- adapter K is FoV-derived at viewport render size, not panel size.

### Kernel Adapter Equivalence Tests

Cases:

- a tiny synthetic camera-space mean set is adapted by the CUDA projection adapter and by `adapt_snapshot_for_rtgs_compat_projection()`;
- projected 2D means match within tolerance;
- adjacent-view translation values change consistently when the adapter is enabled;
- input 3D covariance tensors are not pre-scaled by the adapter.

### Experiment Runner Tests

Cases:

- `--engine compose` is rejected and the default engine is `one_shot`;
- one-shot rows preserve the existing metrics schema;
- timing, quality, resource, and resolution columns follow `experiment/docs/rtgs_cr_metric_columns.md` and the Korean reference `experiment/docs/rtgs_cr_metric_columns_ko.md`;
- manifest records `render_call_count=1` for one-shot timing iterations;
- the experiment script does not pass an engine flag and defaults to one-shot.

### Clustered Metric Tests

Cases:

- sampled one-shot views are rendered through a constant-view lookup, equivalent to the 3DGS helper `constant_view_index_like()`;
- one-shot sampled metrics compare one-shot CR views against official RTGS sampled views;
- interlaced one-shot output is compared against compose interlaced output;
- `compute_sampled_view_metric_stats()` is the sampled-view metric entry point for one-shot experiment rows.

## Validation Commands

Use small smoke tests before full-resolution LKG runs.

dnerf one-shot diagnostic:

```bash
cd /home/ysj/lkg-experiment/experiment
conda activate rtgs-coherent-cu121

PYTHONPATH=src python rtgs_cr_experiment.py \
  --dataset-kind dnerf \
  --model-path /data/ysj/result/4dgs/RTGS/jumpingjacks \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --gsplat-root /home/ysj/lkg-experiment/gsplat \
  --dataset-root /data/ysj/dataset/dnerf \
  --artifact-dir /data/ysj/result/coherent-raster/generated/rtgs_cr_experiments \
  --run-id jumpingjacks_one_shot_smoke \
  --width 320 \
  --height 320 \
  --views 66 \
  --map-mode linear \
  --clusters 1 \
  --warmup-iters 0 \
  --measure-iters 1 \
  --max-metric-views 5 \
  --skip-web-assets
```

N3DV one-shot acceptance command:

```bash
PYTHONPATH=src python rtgs_cr_experiment.py \
  --dataset-kind n3dv \
  --model-path /data/ysj/result/4dgs/RTGS/coffee_martini \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --gsplat-root /home/ysj/lkg-experiment/gsplat \
  --n3dv-root /data/ysj/dataset/N3DV \
  --artifact-dir /data/ysj/result/coherent-raster/generated/rtgs_cr_experiments \
  --run-id coffee_martini_one_shot_smoke \
  --width 320 \
  --height 568 \
  --views 66 \
  --map-mode linear \
  --clusters 1 \
  --warmup-iters 0 \
  --measure-iters 1 \
  --max-metric-views 5 \
  --skip-web-assets
```

## Risks

- N3DV correctness depends on kernel-level support for the RTGS sentinel-FoV mean adapter. Without it, one-shot CR will likely reproduce the old bad N3DV render.
- The current `rasterization_CR` effectively supports one shared `K` in this path. If per-cluster or per-view intrinsics are required later, `rasterization_CR()`, `isect_tiles_CR()`, and `intersect_tile_kernel_CR_AccuTile()` must all be extended together.
- `cluster_size>1` intentionally reuses covariance/depth/color and may reduce quality. `cluster_size=1` must remain the correctness baseline.
- Full-resolution 1440x2560 one-shot may expose VRAM pressure from `means2d/conics/colors [C,N,...]`, especially for `cluster_size=1` where `C=66`.
- One-shot rows can be misleading if they only report old compose-style sampled metrics. The experiment must report one-shot sampled metrics and, for acceptance runs, one-shot-vs-compose interlaced PSNR.

## Recommended Next Implementation Order

1. Add `cr_one_shot.py` with viewport lookup, grouped viewmat, color evaluation, and one-shot call helpers.
2. Add tests for shape contracts, viewport cropping, and adapter decision metadata.
3. Wire `cr_experiment.py` to the new one-shot helper for dnerf/FoV-positive cameras.
4. Validate dnerf `cluster_size=1` against compose.
5. Add explicit N3DV diagnostic failure mode so old bad behavior is documented and cannot be mistaken for success.
6. Extend CR projection kernels for the RTGS mean adapter.
7. Validate N3DV `cluster_size=1` against compose and official sampled views.
8. Run cluster-size sweep and compare FPS/PSNR/VRAM.

## Implementation Progress

- 2026-06-29: Added `lkg_experiment.rtgs_coherent.cr_one_shot` for viewport-cropped CR lookup construction, one-shot clustered rendering, and panel reinsertion.
- 2026-06-29: Wired `cr_experiment.py --engine clustered` to delegate rendering through the new one-shot module.
- 2026-06-29: Removed the positive-FoV letterbox guard because one-shot now renders the content viewport instead of the full panel.
- 2026-06-29: Validated a 64x64 dnerf `jumpingjacks` smoke run. `cluster_size=1` one-shot and compose interlaced PNGs matched exactly (`MSE=0`, `MAE=0`, `PSNR=inf`).
- 2026-06-29: Validated full LKG 1440x2560 dnerf `jumpingjacks` runs:
  - clustered one-shot: `jumpingjacks_clustered_lkg1440x2560_codex_20260629`, `frame_ms=458.176`, `fps=2.183`, `peak_vram_gb=2.846`;
  - compose baseline: `jumpingjacks_compose_lkg1440x2560_codex_20260629`, `frame_ms=32525.305`, `fps=0.031`, `peak_vram_gb=2.252`;
  - interlaced PNG comparison: `MSE=0`, `MAE=0`, `PSNR=inf`.
- 2026-06-29: Validated a 64x112 N3DV `coffee_martini` compose smoke run. The content viewport resolved to `64x48+0+32`, confirming the letterboxed N3DV compose path still works.
- 2026-06-29: Confirmed N3DV clustered with `rtgs_compat_projection=True` stops at the intended sentinel-FoV guard. A diagnostic `--no-rtgs-compat-projection` clustered run executed and produced a 64x112 image; versus compose it measured `MSE=0.00021566`, `MAE=0.00593724`, `PSNR=36.662 dB`. This is diagnostic only and does not replace Phase C kernel adapter acceptance.
- 2026-06-29: Validated full LKG 1440x2560 N3DV `coffee_martini` runs:
  - compose baseline: `coffee_martini_compose_lkg1440x2560_codex_20260629`, content viewport `1440x1080+0+740`, `frame_ms=27172.960`, `fps=0.037`, `peak_vram_gb=10.516`;
  - clustered with `rtgs_compat_projection=True`: stopped at the intended sentinel-FoV guard;
  - diagnostic clustered with `--no-rtgs-compat-projection`: `coffee_martini_clustered_no_adapter_lkg1440x2560_codex_20260629`, `frame_ms=582.383`, `fps=1.717`, `peak_vram_gb=19.857`;
  - diagnostic clustered versus compose: `MSE=0.00487815`, `MAE=0.03059597`, `PSNR=23.117 dB`, confirming this is not an acceptance result before the Phase C adapter.
- 2026-06-29: Added the sentinel-FoV RTGS projection adapter to the grouped CR CUDA projection and tile-intersection kernels, removed the temporary N3DV clustered guard, and validated `coffee_martini` clustered `cluster=2` with `rtgs_compat_projection=True`:
  - 64x112 smoke with generated linear view map: `n3dv_cluster_adapter_smoke`;
  - 1440x2560 LKG run with calibration view map: `n3dv_cluster_adapter_fullres`, `frame_ms=526.160`, `fps=1.901`, `peak_vram_gb=15.289`.
- 2026-06-29: Renamed the experiment runner engine to `one_shot`, removed the old compose experiment branch, made `cluster_1` part of the default sweep (`1,2,4,8,16`), and updated `run_rtgs_cr_experiments_all.sh` to stop passing `--engine`.

## Sub-Agent Review Notes

Sub-agent `Avicenna` reviewed the draft by static code reading. The review agreed that the compose-vs-one-shot execution model was correctly described, and identified four required corrections that are now incorporated:

- the N3DV adapter must affect both reference projection and adjacent-view translation generation;
- the CR tensor contract must list exact patchified lookup shapes and the shared-`K` limitation;
- clustered metrics must validate clustered output instead of only reusing compose metrics;
- viewport lookup construction must crop to the content viewport before patchification and unpad to `render_height/render_width`.
