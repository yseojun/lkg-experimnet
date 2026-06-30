# RTGS CoherentRaster Gsplat Patch Archive

This directory preserves the `gsplat` changes needed by the RTGS + CoherentRaster path in `lkg-experiment`.

It is intentionally **not** a full `gsplat` fork. The canonical replay artifacts are ordered git patches in `patches/`, and `modified-files/` contains the final file snapshots for review or manual recovery.

## Scope

These changes affect the CoherentRaster-specific `gsplat` path:

- `gsplat.rendering_coherent_raster.rasterization_CR`
- CoherentRaster Python/CUDA wrappers under `gsplat/cuda/*CoherentRaster*`

They do **not** modify the normal 3DGS renderer entrypoint:

- `gsplat.rendering.rasterization`

For ordinary 3DGS rendering that does not call `rasterization_CR`, there should be no image-output change.

## Source Commits

The archive was generated from the external repo at `/home/ysj/lkg-experiment/gsplat`.

| Commit | Purpose |
| --- | --- |
| `938f6a9` | Add RTGS projection adapter to CoherentRaster. |
| `ede313e` | Add CoherentRaster timing metrics. |

Recommended base before applying these patches:

```text
6f6609a Support precomputed RGB in coherent raster
```

An equivalent CoherentRaster-enabled `gsplat` checkout can also work, but vanilla upstream `gsplat` is not enough because these patches assume the CoherentRaster files and APIs already exist.

## What Changed

### RTGS Projection Adapter

RTGS N3DV checkpoints use a camera contract where positive explicit intrinsics (`fl_x`, `fl_y`, `cx`, `cy`) can coexist with non-positive FoV sentinel values (`FoVx=FoVy=-1.0`). Official RTGS rendering uses this contract differently from the naive `gsplat` projection path.

The adapter is an optional tensor:

```text
rtgs_projection_adapter = [fov_fx, fov_fy, scale_x, scale_y, cx, cy]
```

When the adapter is present, CoherentRaster:

- uses `fov_fx` and `fov_fy` for covariance/radius projection,
- scales camera-space Gaussian mean `x/y` by `scale_x/scale_y` for mean projection,
- uses explicit `cx/cy` as the principal point,
- applies the same adapted projection when computing adjacent-view translation values.

When the adapter is `None`, the existing CoherentRaster projection behavior is preserved.

### Timing Metrics

`rasterization_CR(..., return_timing=True)` now returns timing values in `meta["timing_ms"]`:

```text
cr_projection_ms
cr_keygen_ms
cr_sort_ms
cr_blend_ms
```

The default is `return_timing=False`, so existing callers do not need to change.

## Directory Layout

```text
rtgs-coherent-raster/
  README.md
  manifest.json
  apply_patches.sh
  verify_snapshot.py
  patches/
    0001-Add-RTGS-projection-adapter-to-coherent-raster.patch
    0002-Add-coherent-raster-timing-metrics.patch
  modified-files/
    gsplat/...
```

## Apply To Another Environment

Set these paths first:

```bash
export LKG_EXPERIMENT=/home/ysj/lkg-experiment
export TARGET_GSPLAT=/path/to/gsplat
```

The target must be a git checkout:

```bash
git -C "$TARGET_GSPLAT" status --short
git -C "$TARGET_GSPLAT" log -1 --oneline
```

Apply the patches with commit metadata:

```bash
git -C "$TARGET_GSPLAT" am --3way \
  "$LKG_EXPERIMENT"/rtgs-coherent-raster/patches/*.patch
```

Or use the helper script:

```bash
"$LKG_EXPERIMENT"/rtgs-coherent-raster/apply_patches.sh "$TARGET_GSPLAT"
```

To check patch applicability without applying:

```bash
"$LKG_EXPERIMENT"/rtgs-coherent-raster/apply_patches.sh --check "$TARGET_GSPLAT"
```

## Manual Fallback

If the target checkout is not close enough for `git am`, inspect the patch failure first:

```bash
git -C "$TARGET_GSPLAT" am --abort
```

Then compare the snapshot files:

```bash
"$LKG_EXPERIMENT"/rtgs-coherent-raster/verify_snapshot.py "$TARGET_GSPLAT"
```

If you intentionally want to overwrite the target CoherentRaster files with the archived versions:

```bash
rsync -av \
  "$LKG_EXPERIMENT"/rtgs-coherent-raster/modified-files/ \
  "$TARGET_GSPLAT"/
```

After manual copy, inspect the diff before committing:

```bash
git -C "$TARGET_GSPLAT" diff --stat
git -C "$TARGET_GSPLAT" diff -- gsplat/rendering_coherent_raster.py
```

## CUDA Extension Rebuild

These patches change Python/CUDA extension interfaces. Rebuild the extension after applying.

Use a fresh extension cache:

```bash
export TORCH_EXTENSIONS_DIR=/data/ysj/result/coherent-raster/generated/torch_extensions_lkg_rtgs/rtgs_official
rm -rf "$TORCH_EXTENSIONS_DIR"
```

Then run an RTGS + CR command or tests that import `gsplat.rendering_coherent_raster`; PyTorch will rebuild the extension.

## Verification

From `lkg-experiment/experiment`:

```bash
PYTHONPATH=src conda run -n rtgs-coherent-cu121 \
  python -m unittest tests.test_gsplat_cr_adapter tests.test_rtgs_cr_one_shot -v
```

For broader experiment coverage:

```bash
PYTHONPATH=src conda run -n rtgs-coherent-cu121 \
  python -m unittest \
  tests.test_batch_experiments \
  tests.test_rtgs_66_lkg_script \
  tests.test_rtgs_cr_experiment \
  tests.test_rtgs_cr_66views \
  tests.test_rtgs_cr_one_shot \
  tests.test_rtgs_lkg_web_server \
  tests.test_gsplat_cr_adapter \
  tests.test_lkg_experiment_defaults -v
```

The helper can verify that this archive still matches the local external `gsplat` checkout:

```bash
rtgs-coherent-raster/verify_snapshot.py gsplat
```

## Expected Impact

| Path | Impact |
| --- | --- |
| Normal `gsplat.rendering.rasterization` | No code-path change. |
| 3DGS CoherentRaster without `rtgs_projection_adapter` | Projection output should remain unchanged. |
| RTGS + CoherentRaster with adapter | Uses RTGS-compatible projection for sentinel-FoV explicit-intrinsics cameras. |
| CoherentRaster with `return_timing=True` | Adds timing values to `meta["timing_ms"]`. |
