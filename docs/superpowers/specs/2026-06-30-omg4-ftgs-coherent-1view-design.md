# OMG4-FTGS CoherentRaster 1-View Design

## Goal

Add a CoherentRaster renderer path to the existing OMG4-FTGS single-view workflow and verify that it renders the same checkpoint, camera, and timestamp as the existing gsplat baseline.

## Scope

This step only targets one camera and one timestamp. Video rendering and multi-view Looking Glass interlacing remain out of scope for this change.

## Approach

The CLI gains a renderer selector: `--renderer gsplat|coherent|both`. The default remains `gsplat` to preserve the current behavior. For validation, `--renderer both --mode single` writes a gsplat baseline, a CoherentRaster output, a visual diff, and manifest metrics.

The CoherentRaster path mirrors the RTGS 1-view path. It builds an all-zero `height x width x 3` viewpoint map, converts it with `build_cr_lookup_arrays`, and calls `gsplat.rendering_coherent_raster.rasterization_CR`. OMG4-FTGS already materializes means, quaternions, scales, opacities, and SH colors, so the CoherentRaster call uses the quats/scales path rather than precomputed covariances.

## Outputs

For `--mode single --renderer both`, the output directory contains:

- `single_gsplat.png`
- `single_coherent.png`
- `single_diff.png`
- `manifest.json`

For `--renderer gsplat`, the existing `single.png` output is preserved for compatibility.

## Verification

Automated tests cover parser behavior, dispatch to the selected renderer, and mocked CoherentRaster output writing. GPU verification runs outside the sandbox in `rtgs-coherent-cu121` with `CUDA_VISIBLE_DEVICES=0`, using:

```bash
conda run -n rtgs-coherent-cu121 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python omg4_ftgs_1view.py --mode single --renderer both --run-label cook_spinach_cr_single
```

Success means the process exits with code 0, both rendered PNGs are non-empty, the manifest records CoherentRaster metadata, and the images can be opened for visual inspection.
