# OMG4-FTGS Multi-View Interlaced Design

## Goal

Generate a Looking Glass style multi-view interlaced image from an OMG4-FTGS checkpoint using CoherentRaster.

## Scope

This change renders one timestamp from one source camera. It does not render full-time interlaced video yet. It also does not add an interactive web playback path.

## Approach

The existing OMG4-FTGS CLI gains `--mode interlaced`. The path loads the same checkpoint, dataset camera, and frame timestamp used by the single-view renderer, materializes OMG4 splats once, synthesizes an orbit of adjacent view matrices, and renders one CoherentRaster interlaced panel.

The interlaced renderer reuses existing project primitives:

- `load_viewpoint_index_file` for the 1440x2560 66-view LKG mapping file.
- `build_linear_viewpoint_index` as a deterministic fallback map for smoke tests.
- `OrbitViewSynthesizer` for grouped adjacent view matrices.
- `gsplat.rendering_coherent_raster.rasterization_CR` for grouped multi-view rasterization.

OMG4 colors are passed as SH coefficients to CoherentRaster. Unlike RTGS, OMG4 does not need per-view Python color evaluation; CoherentRaster evaluates SH color internally from the reference view directions.

## CLI

New interlaced options:

- `--mode interlaced`
- `--views`, default `66`
- `--panel-width`, default `1440`
- `--panel-height`, default `2560`
- `--map-mode file|linear`, default `file`
- `--viewpoint-index-path`, default `/data/ysj/result/coherent-raster/generated/lkg_go_1440x2560_66_views_lkg_calibration.npz`
- `--cluster-size`, default `8`
- `--view-degree`, default `53.0`
- `--orbit-direction`, default `-1`
- `--orbit-center-distance`, default `0.0`, meaning estimate from splat center
- `--coherent-quantize floor|nearest`, default `floor`

## Outputs

The run writes:

- `omg4_ftgs_lkg_interlaced.png`
- `manifest.json`

The manifest records checkpoint/data paths, source camera, frame timestamp, panel dimensions, view count, cluster size, map metadata, orbit parameters, Gaussian count, and render timing.

## Verification

Unit tests cover parser defaults, map selection, orbit view matrix shape, and mocked interlaced output writing. GPU verification runs outside the sandbox:

```bash
conda run -n rtgs-coherent-cu121 env CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src python omg4_ftgs_1view.py --mode interlaced --run-label cook_spinach_lkg_interlaced
```

Success means the process exits with code 0, `omg4_ftgs_lkg_interlaced.png` is an RGB `1440x2560` image, the manifest records `mode=interlaced`, and the image opens with non-empty pixel ranges.
