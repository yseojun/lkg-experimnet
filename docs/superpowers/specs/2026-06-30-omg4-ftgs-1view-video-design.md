# OMG4-FTGS 1-View And Video Rendering Design

## Goal

Add an OMG4-FTGS-specific source package that can load the stored OMG4-FTGS compressed checkpoints, render one fixed camera view, and render a full-time video for that same camera. The first validation target is:

- checkpoint: `/data/ysj/result/4dgs/OMG4-FTGS_weights/ours_L_weight/cook_spinach.xz`
- dataset: `/data/ysj/dataset/N3DV/cook_spinach`
- camera: test camera index `0`

This first step validates OMG4-FTGS decode, camera conversion, and normal `gsplat.rasterization` output before adapting the path to CoherentRaster.

## Scope

Create `experiment/src/lkg_experiment/omg4_ftgs/` as a dedicated package, parallel to `rtgs_coherent/`.

The initial implementation includes:

- checkpoint decoding from OMG4-FTGS `comp.xz`
- N3DV/DyNeRF-style camera loading from `poses_bounds.npy` and `transforms_test.json`
- one-view PNG rendering at a chosen timestamp
- same-camera video rendering over all timestamps in `transforms_test.json`
- output manifests recording checkpoint, dataset, camera, dimensions, timestamps, and render timing

The initial implementation does not yet replace `gsplat.rasterization` with `rasterization_CR`. It keeps renderer boundaries compatible with a later CoherentRaster adapter.

## Architecture

`lkg_experiment.omg4_ftgs.model`

- Defines a lightweight `DynamicGaussians` loader based on `OMG4/OMG4_FTGS/_gaussians.py`.
- Decodes `comp.xz` using `lzma`, `pickle`, and the local OMG4 Huffman helpers.
- Materializes time-dependent tensors:
  - `means_t = means + (timestamp - times) * velocities`
  - `scales = exp(scales)`
  - `quats`
  - `opacities = sigmoid(mlp_opacity(...)) * temporal_opacity`
  - SH colors from the OMG4-FTGS MLPs

`lkg_experiment.omg4_ftgs.camera`

- Loads test frames from `transforms_test.json`.
- Loads N3DV camera poses and intrinsics from `poses_bounds.npy`, matching OMG4-FTGS axis conversion.
- Converts the selected camera to `viewmat` and `K` tensors accepted by gsplat.

`lkg_experiment.omg4_ftgs.render`

- Calls normal `gsplat.rasterization` first.
- Returns `3,H,W` float RGB tensors clamped to `[0,1]`.
- Keeps a function boundary that can later swap the backend to CoherentRaster.

`lkg_experiment.omg4_ftgs.cli`

- Provides a CLI with modes:
  - `single`: render one PNG
  - `video`: render the full timestamp sequence for one camera and encode MP4 when possible
  - `both`: run single and video in one command
- Defaults target the approved cook_spinach setup.

Top-level script:

- `experiment/omg4_ftgs_1view.py` dispatches to the package CLI, matching existing RTGS script style.

## Data Flow

1. Resolve checkpoint and dataset paths.
2. Load checkpoint into `DynamicGaussians`.
3. Load camera `0` from `poses_bounds.npy` and test timestamps from `transforms_test.json`.
4. Render one timestamp for PNG validation.
5. Render every test timestamp with the same camera pose and intrinsics.
6. Save PNG frames, MP4 if available, and JSON manifest.

## Output

Default output root:

`/data/ysj/result/coherent-raster/generated/omg4_ftgs`

Expected files for the default run:

- `single.png`
- `frames/frame_0000.png`, `frames/frame_0001.png`, ...
- `video.mp4` when MP4 encoding is available
- `manifest.json`

## Error Handling

- Missing checkpoint or dataset files raise clear `FileNotFoundError`.
- Missing CUDA raises `RuntimeError`, because OMG4-FTGS decode and rendering require CUDA/tiny-cuda-nn.
- MP4 encoding failure does not discard rendered PNG frames; it records the failure in the manifest.
- If `tinycudann`, `dahuffman`, or gsplat are unavailable, the CLI reports the missing dependency.

## Testing And Verification

Unit tests cover path resolution, transform timestamp loading, camera matrix shapes, and default argument behavior.

Manual verification for this step:

```bash
cd experiment
PYTHONPATH=src python omg4_ftgs_1view.py --mode both
```

Successful verification means:

- one PNG is written
- the timestamp sequence frames are written
- MP4 is written, or the manifest records a codec/ffmpeg failure while frames exist
- manifest records the cook_spinach L checkpoint, dataset path, selected camera, and frame count

## Future CoherentRaster Adapter

After the normal gsplat path is validated, add a CR backend that reuses the materialized tensors from `model.py` and feeds `rasterization_CR`. The FTGS representation is well suited to the non-covariance CR path because it naturally materializes `means`, `quats`, `scales`, `opacities`, and SH colors per timestamp.
