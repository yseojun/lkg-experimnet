# RTGS + CoherentRaster Metric Columns

Created: 2026-06-29

Korean version: `experiment/docs/rtgs_cr_metric_columns_ko.md`

This document defines the `metrics.csv` / `metrics.json` columns for RTGS + CoherentRaster experiments. The goal is to keep the experiment output readable after adding dynamic RTGS timing breakdowns.

The timing scope is one rendered frame at one timestamp. Checkpoint loading, CUDA extension build time, image file writing, web asset generation, and metric/reference rendering are not part of render timing unless a column explicitly says otherwise.

## Identifier Columns

| Column | Meaning |
| --- | --- |
| `dataset_kind` | Dataset family, for example `dnerf` or `n3dv`. |
| `scene` | Scene/checkpoint directory name. |
| `checkpoint` | Checkpoint path used for the run. |
| `camera_split` | Camera split selected by the experiment, for example `train` or `test`. |
| `camera_index` | Anchor camera index used for the frame. |
| `frame_index` | Dataset frame index. For N3DV this maps to `--n3dv-frame-index`; for dnerf it should identify the selected timestamp frame. |
| `timestamp` | RTGS timestamp used to materialize dynamic Gaussians. |
| `output_prefix` | Optional filename prefix used when saving image artifacts. |
| `engine` | Experiment engine. Current RTGS + CR experiment uses `one_shot`. |
| `variant` | Variant name, for example `cluster_8`, `without_remap`, `without_reuse`, or `without_reuse_without_remap`. `without_reuse` is the official RTGS reference/GT variant in RTGS + CR experiments. |
| `group` | Variant group used for aggregation/comparison. |

## Configuration Columns

| Column | Meaning |
| --- | --- |
| `render_width` | Width passed to the renderer for the viewport image. |
| `render_height` | Height passed to the renderer for the viewport image. |
| `source_views` | Number of LKG source views used to build the interlaced image. |
| `saved_views` | Number of individual source-view images saved for visual inspection. This does not change the rendered interlaced view count. |
| `cluster_size` | Number of adjacent views grouped into one CoherentRaster cluster. |
| `color_eval_views` | Number of camera centers used for RTGS color evaluation. For one-shot clustered rendering this is usually the number of clusters, not necessarily `source_views`. |
| `use_remapping` | Whether CoherentRaster view/subpixel remapping is enabled. |
| `reuse_enabled` | Whether clustered attribute reuse is enabled. If false, rendering should behave like cluster size 1 for reuse-sensitive work. |
| `tile_size` | CoherentRaster tile size. |
| `map_mode` | LKG view mapping mode used to generate the interlaced lookup. |
| `camera_aspect_mode` | Synthetic camera aspect policy. `expand` builds a full-panel 9:16 camera for LKG output; `preserve` keeps the legacy `aspect_fit` viewport/camera behavior. |

## RTGS Dynamic Timing

These columns measure work added by dynamic RTGS before the CoherentRaster core.

| Column | Meaning |
| --- | --- |
| `dynamic_geometry_ms` | Time to compute timestamp-conditioned Gaussian means/covariances, including RTGS mean offsets when used. |
| `temporal_opacity_ms` | Time to compute timestamp marginal weights, opacity modulation, and temporal active mask. |
| `snapshot_compaction_ms` | Time to compact masked tensors into render-ready contiguous tensors. |
| `dynamic_color_ms` | Time to evaluate RTGS view-dependent color, including 4D SH when applicable. |
| `rtgs_dynamic_total_ms` | Sum of `dynamic_geometry_ms`, `temporal_opacity_ms`, `snapshot_compaction_ms`, and `dynamic_color_ms`. |

## CoherentRaster Core Timing

These columns correspond to the core CoherentRaster pipeline and should be comparable with the static 3DGS CoherentRaster breakdown.

| Column | Meaning |
| --- | --- |
| `cr_projection_ms` | Gaussian projection and projected footprint preparation. |
| `cr_keygen_ms` | Tile/view/subpixel key generation. |
| `cr_sort_ms` | Key/depth sorting and range preparation. |
| `cr_blend_ms` | Alpha blending / color accumulation into the CR output buffer. |
| `cr_core_total_ms` | Sum of `cr_projection_ms`, `cr_keygen_ms`, `cr_sort_ms`, and `cr_blend_ms`. |

## LKG Interlace Post Timing

This is recorded separately from CoherentRaster core timing because it is the LKG display image construction step.

| Column | Meaning |
| --- | --- |
| `lkg_unpatchify_ms` | Time to convert CoherentRaster patch/tile output into a viewport image tensor. |
| `lkg_unpad_ms` | Time to remove tile padding and clamp/crop back to `render_width` x `render_height`. |
| `lkg_panel_paste_ms` | Time to paste the viewport image into the final LKG panel tensor. |
| `lkg_interlace_post_ms` | Sum of `lkg_unpatchify_ms`, `lkg_unpad_ms`, and `lkg_panel_paste_ms`. |

## Aggregate Timing Columns

| Column | Meaning |
| --- | --- |
| `frame_ms_without_lkg` | `rtgs_dynamic_total_ms + cr_core_total_ms`. This is the dynamic RTGS + CR render time before LKG interlace post processing. |
| `fps_without_lkg` | `1000 / frame_ms_without_lkg`. |
| `frame_ms_with_lkg_interlace` | `frame_ms_without_lkg + lkg_interlace_post_ms`. This is the frame time until the interlaced LKG tensor is ready. |
| `fps_with_lkg_interlace` | `1000 / frame_ms_with_lkg_interlace`. |
| `frame_ms` | Backward-compatible alias for `frame_ms_with_lkg_interlace` after detailed timing is implemented. |
| `fps` | Backward-compatible alias for `fps_with_lkg_interlace` after detailed timing is implemented. |

If a run does not collect detailed timing yet, only the legacy `frame_ms` and `fps` columns may be present. In that case they should be interpreted according to the runner version that produced the file.

## Quality Columns

These metrics compare sampled RTGS + CR source-view renders against the official RTGS renderer. They are not the same as LKG interlaced image similarity unless explicitly computed by a separate interlaced metric.

| Column | Meaning |
| --- | --- |
| `psnr_mean` | Mean PSNR over sampled metric views. |
| `psnr_std` | Standard deviation of sampled-view PSNR. |
| `ssim_mean` | Mean SSIM over sampled metric views. |
| `ssim_std` | Standard deviation of sampled-view SSIM. |
| `lpips_mean` | Mean LPIPS over sampled metric views, if collected. Use `nan` when not collected. |
| `lpips_std` | Standard deviation of sampled-view LPIPS, if collected. Use `nan` when not collected. |
| `metric_view_count` | Number of sampled views used for quality metrics. |

## Resource And Scene Columns

| Column | Meaning |
| --- | --- |
| `total_gaussians` | Number of Gaussians in the loaded checkpoint before timestamp masking. |
| `active_gaussians` | Number of Gaussians remaining after timestamp mask/compaction. |
| `active_gaussian_ratio` | `active_gaussians / total_gaussians`. |
| `peak_vram_gb` | Peak CUDA reserved memory during the measured experiment variant, in GiB. |

## Notes

- `lkg_interlace_post_ms` is intentionally not folded into `cr_core_total_ms`.
- The current CR Python wrapper records `cr_keygen_ms` around the tile-intersection/key path. `cr_sort_ms` is kept as a schema column and records `0.0` until the CUDA path exposes sorting as a separately timed stage.
- `frame_ms_without_lkg` is the fairer number for comparing dynamic RTGS + CR against renderer-only baselines.
- `frame_ms_with_lkg_interlace` is the practical number for producing an LKG-ready tensor.
- Disk output, video encoding, GLFW texture upload, and display swap should be measured in separate output/display columns if needed later.
