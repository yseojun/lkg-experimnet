# RTGS + CoherentRaster 66-View Design

Created: 2026-06-29

## Decision

Task 4 will add a new RTGS + CoherentRaster 66-view entrypoint instead of rewriting the existing `views66.py` path in place. The correctness renderer will use all 66 synthesized views for the LKG interlaced output, but it will save only a small sampled set of full-frame views for visual/debug inspection.

This keeps the previously implemented 3DGS/RTGS 66-view experiment code available as reference while allowing the new path to strictly preserve the Task 3 official RTGS runtime and N3DV `rtgscompat` projection contract.

The initial implementation uses `--interlace-mode compose`: render each CR view with the per-view Task 3 adapter in memory, then compose the LKG interlaced image from the 66 rendered views. A single direct multi-view CR kernel call remains a later optimization because the current N3DV `rtgscompat` adapter is view-dependent and cannot be represented as one global mean/covariance transform without kernel changes.

## Goals

- Render a 66-view LKG interlaced image from one RTGS checkpoint and one timestamp.
- Save only 5 sampled full-frame CR views by default, because saving every full-frame view is unnecessary for normal visual validation.
- Compare the same 5 sampled CR views against official RTGS synthetic-view renders when requested.
- Preserve the Task 3 dnerf and N3DV camera/model behavior:
  - dnerf uses the official RTGS scene camera.
  - N3DV uses the dynamic `cameras.json`/`poses_bounds.npy` camera.
  - N3DV keeps `FoVx=FoVy=-1.0` and disables explicit-intrinsics FoV normalization.
  - CR input applies the Task 3 RTGS-compatible projection adapter when the sentinel-FoV explicit-intrinsics condition is present.
- Write all outputs under `/data/ysj/result/coherent-raster/generated`.

## Non-Goals

- Do not save all 66 full-frame views by default.
- Do not physically validate on the LKG display while the display is disconnected.
- Do not tune CoherentRaster cluster reuse for performance before `cluster_size=1` correctness is validated.
- Do not use the older lightweight RTGS checkpoint loader as acceptance evidence for this path.
- Do not treat the `compose` interlacing baseline as a final CoherentRaster speed benchmark. It is a correctness baseline that preserves N3DV `rtgscompat`.

## New Files

- `experiment/rtgs_cr_66views.py`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_66views.py`
- `experiment/tests/test_rtgs_cr_66views.py`
- optional batch script after the smoke path works: `experiment/scripts/run_rtgs_cr_66views_all.sh`

The existing `experiment/src/lkg_experiment/rtgs_coherent/views66.py` remains a reference path unless later cleanup is explicitly requested.

## CLI Contract

The new command should mirror the Task 3 `rtgs_cr_1view.py` arguments and add 66-view controls:

```bash
PYTHONPATH=src python rtgs_cr_66views.py \
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
  --viewpoint-index-path /data/ysj/result/coherent-raster/generated/lkg_go_1440x2560_66_views_lkg_calibration.npz \
  --run-label coffee_martini_cam00_frame0000_cr66
```

Required additional options:

```text
--views
--sample-save-views
--sample-view-indices
--width
--height
--view-degree
--orbit-direction
--orbit-center-distance
--viewpoint-index-path
--map-mode {file,linear}
--cluster-size
--cr-remapping {on,off}
--interlace-mode {compose}
--compare-official-sampled / --no-compare-official-sampled
--write-interlaced / --no-write-interlaced
```

Defaults:

```text
views: 66
sample_save_views: 5
sample_view_indices: evenly spaced across [0, views - 1]
width: 1440
height: 2560
cluster_size: 1 for correctness smoke
cr_remapping: on
interlace_mode: compose
write_interlaced: true
compare_official_sampled: true
output root: /data/ysj/result/coherent-raster/generated/rtgs_cr_66views
```

## Data Flow

1. Load official RTGS runtime through `prepare_official_rtgs_1view(args)`.
2. Move the anchor camera to CUDA and keep the Task 3 camera contract unchanged.
3. Materialize timestamp-conditioned RTGS geometry once:

```text
GaussianModel + timestamp
  -> means, covars, opacities, temporal mask
```

4. Synthesize 66 view matrices around the anchor camera using the existing `OrbitViewSynthesizer`.
5. For each cluster representative view, evaluate RTGS color from the representative camera center.
6. For the sampled 5 views:
   - evaluate per-view color from that view's camera center;
   - apply the RTGS-compatible CR projection adapter when needed;
   - render one full-frame CR image;
   - optionally render the official RTGS synthetic view for comparison.
7. For the LKG interlaced image:
   - build the 66-view viewpoint-index map from the calibration NPZ or linear debug map;
   - validate `viewpoint_index.min() >= 0` and `viewpoint_index.max() < 66`;
   - use all 66 in-memory CR renders to fill the subpixels selected by the viewpoint-index map;
   - save one interlaced image.

## Output Layout

For a run label `coffee_martini_cam00_frame0000_cr66`:

```text
/data/ysj/result/coherent-raster/generated/rtgs_cr_66views/coffee_martini/coffee_martini_cam00_frame0000_cr66/
  rtgs_cr_lkg_interlaced.png
  sampled_views/
    view_000_cr.png
    view_016_cr.png
    view_032_cr.png
    view_049_cr.png
    view_065_cr.png
  sampled_comparisons/
    view_000_official.png
    view_000_comparison.png
    view_000_metrics.json
    ...
  metrics.json
  manifest.json
```

If `--sample-view-indices` is provided, the sampled filenames follow those explicit indices. If `--no-compare-official-sampled` is used, `sampled_comparisons` can be omitted but `sampled_views` remains required.

## N3DV Projection Rule

The N3DV acceptance path must match Task 3:

```text
camera_fov_normalization.enabled: false
camera_fov_normalization.applied: false
FoVx: -1.0
FoVy: -1.0
cr_projection_adapter.applied: true
cr_projection_adapter.reason: explicit_intrinsics_with_nonpositive_fov_sentinel
```

The diagnostic FoV-normalized path must not be used for Task 4 acceptance because it reproduces the earlier bad N3DV rendering.

## Validation Order

1. Unit tests for parser defaults, sampled index selection, output path separation, viewpoint-index validation, and N3DV projection adapter manifest fields.
2. dnerf isolated debug smoke with `cluster_size=1`, `cr-remapping=off`, `map-mode=linear`, and sampled comparisons enabled.
3. N3DV isolated debug smoke with `cluster_size=1`, `cr-remapping=off`, `map-mode=linear`, and sampled comparisons enabled.
4. Real LKG map smoke with `cluster_size=1`, `cr-remapping=on`, 66-view interlaced output enabled.
5. Cluster reuse experiments only after the above pass:

```text
cluster_size=2
cluster_size=4
cluster_size=8
```

## Acceptance Gates

- `rtgs_cr_lkg_interlaced.png` exists and is nonblack.
- Exactly 5 sampled CR full-frame views are saved by default.
- Sampled official comparisons record PSNR/MAE/MSE when enabled.
- dnerf sampled CR views are visually aligned with official synthetic views.
- N3DV sampled CR views use the Task 3 `rtgscompat` adapter and are visually aligned with official synthetic views.
- `manifest.json` records:
  - dataset kind, scene, model path, checkpoint path;
  - timestamp, split, camera index, N3DV frame index;
  - 66 view synthesis parameters;
  - sampled view indices;
  - LKG viewpoint-index path and metadata;
  - cluster size and remapping mode;
  - camera FoV normalization status;
  - CR projection adapter status;
  - render timings for sampled views, sampled official comparisons, and interlaced output.

## Failure Handling

- If sampled official comparisons fail, debug synthetic camera construction before inspecting the interlaced image.
- If dnerf passes but N3DV fails, first inspect `FoVx/FoVy`, explicit intrinsics, and `cr_projection_adapter`.
- If sampled views pass but interlaced output fails, inspect viewpoint-index map shape, view-id range, patchification, remapping, and cluster grouping.
- If only larger cluster sizes fail, treat it as a cross-view attribute reuse issue and keep `cluster_size=1` as the correctness baseline.

## Documentation Follow-Up

After implementation, update the main implementation plan with the actual smoke commands and output directories. Task 4 remains incomplete until the user confirms the sampled views and interlaced image are visually acceptable.
