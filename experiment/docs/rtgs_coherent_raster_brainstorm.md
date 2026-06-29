# RTGS + CoherentRaster Brainstorm

Created: 2026-06-29

## Purpose

이 문서는 1단계 RTGS official 1-view baseline 이후, CoherentRaster를 RTGS renderer에 적용하기 위한 2단계 분석/계획 문서다. 목표는 바로 CUDA 코드를 고치는 것이 아니라, CoherentRaster 논문의 핵심 가정과 현재 RTGS 렌더링 계약을 대조해 다음 구현 단계의 실패 지점을 줄이는 것이다.

## CoherentRaster Paper Analysis

CoherentRaster는 Light Field Display(LFD)의 최종 출력이 full RGB view들의 배열이 아니라 interlaced subpixel image라는 점을 이용한다. LFD calibration은 각 LCD subpixel `(x, y, u)`가 어떤 viewpoint `j`를 표시해야 하는지 정하는 viewpoint index matrix `V[x,y,u]`를 만든다. 기존 full-frame 방식은 모든 viewpoint를 완전한 RGB frame으로 렌더링한 뒤 필요한 subpixel만 뽑기 때문에 viewpoint 수에 비례해 projection, sorting, blending 비용이 증가한다.

논문의 pipeline은 다음 세 최적화로 요약된다.

1. **Subpixel-level rasterization**
   - 최종 LFD image에 필요한 subpixel만 직접 렌더링한다.
   - 각 subpixel은 `V[x,y,u]`로 viewpoint를 선택하고, 해당 viewpoint 또는 cluster의 Gaussian list를 alpha blend한다.

2. **Cross-view Coherent Attribute Reuse**
   - viewpoint를 인접 cluster로 묶는다.
   - 2D mean은 viewpoint마다 계산한다. 작은 viewpoint 변화도 tile coverage를 바꾸므로 재사용하면 geometric artifact가 커진다.
   - 2D covariance, depth, SH color는 cluster representative view에서 계산해 같은 cluster에 재사용한다.
   - key는 `(tile_id, cluster_id, depth)` 구조가 되어 per-view Gaussian-tile pair 수와 sorting 비용을 줄인다.

3. **View-coherent Remapping**
   - LFD layout에서는 인접 subpixel이 서로 다른 viewpoint를 가리킬 수 있어 warp memory access가 흩어진다.
   - tile 내부 subpixel을 viewpoint index 기준으로 정렬한 lookup table `Psi`를 미리 만들어, 연속 thread가 같은/인접 viewpoint의 list를 읽게 한다.
   - lens geometry가 고정되어 있으므로 lookup은 frame마다 다시 만들 필요가 없다.

논문의 중요한 한계도 RTGS 적용에서 그대로 유효하다.

- CoherentRaster는 정적 3DGS를 대상으로 한다.
- Cross-view reuse는 cluster 안에서 viewpoint 변화가 작다는 가정에 의존한다.
- specular/high-frequency view-dependent color는 cluster size가 커질수록 artifact가 생길 수 있다.
- 논문의 quality metric은 실제 LFD capture GT가 아니라 full-frame 3DGS render를 pseudo ground-truth로 사용한다.

RTGS 적용에서는 이 한계를 "같은 timestamp에서 4D Gaussian을 3D snapshot으로 materialize한 뒤 CR을 적용한다"로 해석한다. 즉 cluster는 viewpoint 축에서만 만들고, 시간축으로는 묶지 않는다. video/frame sequence에서는 매 frame timestamp마다 snapshot을 새로 만든다.

## Step 1 Baseline Facts

1단계 restore point는 commit `34fce60 Establish RTGS official 1-view baseline`이다. Official harness는 dirty RTGS checkout을 그대로 쓰지 않고 clean snapshot을 만들며, effective RTGS commit은 `63725f21d4adc29669e565ae10e6b3ad6e0d1250`이다.

대표 baseline A: dnerf `jumpingjacks`

```text
output_dir: experiment/generated/rtgs_official_1view/jumpingjacks/step1_all_1view_final_verify
dataset_kind: dnerf
camera_source: official_scene_camera_dataset
camera_image_name: r_013
timestamp: 0.6834170854271356
resolution: 400 x 400
gaussians_total: 780611
gaussian_dim: 4
active_sh_degree: 3
active_sh_degree_t: 2
pipeline: compute_cov3D_python=False, convert_SHs_python=False, eval_shfs_4d=True
camera contract: FoVx=0.6911112070083618, FoVy=0.6911112070083618, fl_x=-0.5, fl_y=-0.5, cx=-0.5, cy=-0.5
rtgs_vs_gt_psnr: 27.651799646428646 dB
visual artifact check: nonblack RGB image, min=0, max=255, per-channel mean=[12.509, 9.628, 6.828]
```

대표 baseline B: N3DV `coffee_martini`

```text
output_dir: experiment/generated/rtgs_official_1view/coffee_martini/step1_all_1view_final_verify
dataset_kind: n3dv
camera_source: n3dv_dynamic_cameras_json
camera_image_name: cam00_0000
timestamp: 0.0
resolution: 1352 x 1014
gaussians_total: 4744560
gaussian_dim: 4
active_sh_degree: 3
active_sh_degree_t: 2
pipeline: compute_cov3D_python=False, convert_SHs_python=False, eval_shfs_4d=True
camera contract: FoVx=-1.0, FoVy=-1.0, fl_x=730.3771702081667, fl_y=730.3771702081667, cx=676.0, cy=507.0
rtgs_vs_gt_psnr: 27.520009474912857 dB
visual artifact check: nonblack RGB image, min=0, max=255, per-channel mean=[124.51, 121.51, 122.05]
```

The all-scene 1-view sweep passed for 8 dnerf scenes and 6 N3DV scenes. The `flame_salmon` loader edge case is documented in the main implementation plan.

## RTGS To CoherentRaster Input Mapping

CoherentRaster/gsplat should not receive raw RTGS 4D parameters directly in the first implementation. The safer contract is to first materialize a timestamp-specific 3D snapshot using RTGS's own formulas, then pass that snapshot into `gsplat.rendering_coherent_raster.rasterization_CR()`.

```text
RTGS official GaussianModel.get_xyz
  -> timestamp-conditioned means after get_current_covariance_and_mean_offset()

RTGS official GaussianModel covariance
  -> timestamp-conditioned 3D covariance covars [N,6]

RTGS opacity
  -> opacity * get_marginal_t(timestamp), then mask marginal_t > 0.05

RTGS SH / 4D SH color
  -> precomputed RGB colors from RTGS eval_sh/eval_shfs_4d for each CR representative view

RTGS camera.world_view_transform
  -> gsplat viewmat [4,4]

RTGS camera.full_proj_transform / explicit intrinsics
  -> reference contract for projection sanity checks; CR itself consumes gsplat viewmat plus K [3,3]

RTGS explicit intrinsics
  -> gsplat K [3,3], preserving the N3DV FoV=-1 sentinel behavior

RTGS background
  -> CR backgrounds tensor [3] or None using the same auto black/white rule

LKG viewpoint index matrix
  -> CR view_idx_matrix after patchify/remap

LKG subpixel coordinates
  -> CR subpixel_coord_matrix after patchify/remap
```

Important implementation detail: the local CR bridge already patchifies lookup arrays. The raw paper notation says `V[x,y,u]`, but the Python wrapper currently passes patchified tensors shaped like `[n_tile_h, n_tile_w, 3, tile_size, tile_size]` plus coordinate tensors shaped like `[n_tile_h, n_tile_w, 3, tile_size, tile_size, 3]`.

Known high-risk adapter details:

- RTGS official default usually has `compute_cov3D_python=False` and `convert_SHs_python=False`, so temporal covariance, temporal opacity, and 4D SH color are evaluated inside the RTGS CUDA rasterizer. The CR adapter will bypass that CUDA path.
- The CR snapshot path must evaluate color with the timestamp-conditioned Gaussian mean used for rendering, not blindly with base `pc.get_xyz`, otherwise view-dependent SH directions can be wrong for `rot_4d=True` models.
- The temporal opacity mask `marginal_t > 0.05` must be recorded. A different mask from the official CUDA path can change visibility and should be diagnosed before tuning CR.
- For N3DV, K conversion must use `fl_x/fl_y/cx/cy`; do not derive focal length from `FoVx=-1.0`.
- For N3DV sampled-view comparisons in Task 4, synthetic RTGS cameras must preserve the anchor sentinel contract: keep `FoVx=FoVy=-1.0` while `fl_x/fl_y/cx/cy` drive projection.
- Before rendering CR, project a small deterministic sample of snapshot means through both RTGS `full_proj_transform` and gsplat `viewmat + K`. Pixel coordinates should agree for dnerf and N3DV, otherwise CR images are not interpretable.

## Brainstormed Approaches

### Approach A: Full-frame RTGS baseline plus interlace only

Render every target view using official RTGS, then interlace with the LKG viewpoint index map.

Pros:
- Lowest implementation risk.
- Best pseudo-GT for later CR comparisons.
- Preserves official RTGS CUDA path exactly.

Cons:
- Does not implement CoherentRaster speedups.
- Very expensive for 66+ views and video.

Use this as a control path, not as the final implementation.

### Approach B: Adapter-first gsplat CoherentRaster

Load model/camera through the Step 1 official RTGS harness, materialize a 3D snapshot at one timestamp, precompute representative-view RGB with RTGS SH/4D SH evaluation, and call the existing gsplat CR kernels.

Pros:
- Reuses the existing CoherentRaster CUDA implementation.
- Keeps RTGS official loading/camera contract fixed.
- Gives a clean 1-view validation path before LKG multi-view.
- Allows cluster/remap experiments without modifying RTGS diff-gaussian-rasterization first.

Cons:
- Must prove Python materialization matches the official RTGS CUDA render path closely enough.
- Uses gsplat rasterization semantics, so exact pixel equality with RTGS diff rasterizer is not guaranteed.
- Needs careful handling for N3DV explicit intrinsics and RTGS 4D SH color.

Recommended for Task 3 and Task 4.

### Approach C: Native RTGS CUDA CoherentRaster

Port CoherentRaster ideas directly into `4d-gaussian-splatting/diff-gaussian-rasterization`.

Pros:
- The final renderer would stay closest to official RTGS rasterizer semantics.
- Avoids cross-library projection/rasterization differences.

Cons:
- Highest risk and slowest feedback cycle.
- Requires modifying RTGS CUDA projection, binning, sorting, and blending together.
- Harder to isolate whether a failure is from RTGS 4D logic or CR logic.

Keep this as a later optimization only after Approach B demonstrates visually valid 1-view and multi-view behavior.

## Recommended Design

Use Approach B, but add one extra validation layer before rendering CR:

```text
official RTGS default render
  -> official RTGS snapshot-reference render
  -> RTGS + CR 1-view render
  -> RTGS + CR multi-view/interlaced render
```

The snapshot-reference render means: use the same official `GaussianModel` and camera, but force the RTGS renderer through a path equivalent to the snapshot adapter when possible, especially Python covariance/color evaluation. If the snapshot-reference render differs strongly from the official default render, do not blame CR yet. First document whether the mismatch comes from covariance, mean offset, temporal opacity mask, SH/4D SH color, or camera intrinsics.

## 1-View CR Invariant

For Task 3, CR must first render a single view with no LKG-specific viewpoint distribution:

```text
viewpoint_index_hwc: zeros with shape [height, width, 3]
view_idx_matrix: patchified/remapped all-zero map
subpixel_coord_matrix: patchified/remapped panel coordinates
adjacent_viewmats: shape [1, 1, 4, 4]
Ks: shape [1, 3, 3]
colors: shape [N,3] or [1,N,3] with sh_degree=None
```

This is not a CoherentRaster speed benchmark. It is an adapter correctness test. It answers only: "Can RTGS model/camera/timestamp/background/color/covariance be converted into the CR renderer without destroying one ordinary view?"

Expected Task 3 outputs:

```text
gt.png
rtgs_official_render.png
rtgs_snapshot_reference_render.png
rtgs_cr_render.png
comparison_official_vs_snapshot.png
comparison_snapshot_vs_cr.png
comparison_official_vs_cr.png
metrics.json
manifest.json
```

## Multi-View CR Invariant

For Task 4, all views in an LKG frame share the same RTGS timestamp. The new dimensions are only viewpoint synthesis, cluster partitioning, and viewpoint index mapping.

```text
timestamp: fixed per output frame
geometry/covariance/opacities: materialized once per timestamp
colors: one RGB table per cluster representative view, shape [num_clusters,N,3]
adjacent_viewmats: grouped as [num_clusters, cluster_size, 4, 4]
viewpoint_index_hwc: real LKG calibration map or linear debug map
cluster_size: start with 1 for correctness, then 2/4/8 for reuse tests
remapping: off/on ablation after cluster_size=1 passes
```

The first correctness run should use `cluster_size=1` and `use_remapping=False` or a logically equivalent setting if the current CR wrapper requires remapping. This removes cross-view attribute reuse from the first multi-view test. After sampled views match the full-frame RTGS pseudo-GT, enable remapping, then increase cluster size.

## Failure Split

When an image is bad, debug in this order:

1. **Official baseline failure**
   - Re-run `rtgs_official_1view.py`.
   - If this fails, return to Step 1. Do not inspect CR yet.

2. **Camera/K conversion failure**
   - Check dnerf FoV path and N3DV explicit-intrinsics sentinel path separately.
   - For N3DV, `FoVx=FoVy=-1.0` must remain in the RTGS camera contract while `fl_x/fl_y/cx/cy` drive projection.

3. **Snapshot materialization failure**
   - Compare Gaussian count before/after temporal mask.
   - Compare means/covars/opacities statistics.
   - Render snapshot-reference through RTGS if possible.

4. **Color evaluation failure**
   - Compare RTGS CUDA SH path to Python `eval_shfs_4d` path.
   - Check `active_sh_degree`, `active_sh_degree_t`, `force_sh_3d`, `time_duration`.

5. **CR rasterization convention failure**
   - Confirm viewmat convention, K convention, image shape, CHW/HWC conversion, near/far plane, and background.

6. **LKG mapping failure**
   - Only inspect viewpoint index/remapping after 1-view CR is valid.

## Acceptance Gates

Task 3 can start after this document is reviewed.

Task 3 is not complete until:

- dnerf and N3DV both produce `rtgs_cr_render.png`.
- `metrics.json` records official-vs-snapshot, snapshot-vs-CR, and official-vs-CR PSNR/MAE/MSE.
- Output is visually nonblack and not transposed/mirrored.
- Official-vs-snapshot passes first; if it fails, stop and debug materialization before using official-vs-CR to judge CoherentRaster.
- Snapshot-vs-CR is the primary CR adapter correctness metric. Official-vs-CR is still recorded, but it is an end-to-end diagnostic after the snapshot gate passes.
- If official-vs-CR stays below the 25 dB alert threshold after the snapshot gate passes, the dominant mismatch layer is documented before trying multi-view.

Initial numeric alert thresholds:

```text
official_vs_snapshot_psnr < 40 dB:
  Treat snapshot materialization as suspect. Do not use this result to judge CR quality yet.

snapshot_vs_cr_psnr < 25 dB or obvious geometry corruption:
  Treat CR rasterizer convention/K/viewmat/background as suspect. Do not start Task 4.

official_vs_cr_psnr < 25 dB:
  Do not start LKG multi-view unless the mismatch is visually benign and explicitly documented.
```

These are debugging gates, not final publication thresholds. After the first valid CR smoke, tighten them if the cross-library rasterizer difference is smaller than expected.

Task 4 is not complete until:

- `cluster_size=1` sampled views are compared against official full-frame RTGS.
- remapping on/off behavior is recorded.
- at least one larger cluster size, preferably 4 or 8, is tested for quality/speed trade-off.
- final interlaced image records viewpoint map path, cluster partition, view synthesis parameters, and frame timestamp.

Task 4 must also validate the LKG/view cluster map before any image-quality claim:

```text
viewpoint_index.max() < source_view_count
viewpoint_index.min() >= 0
each source view is either covered by the map or explicitly marked unused
non-divisible cluster tails are padded or truncated deterministically
color-coded view-index debug render matches expected subpixel placement
```

## Implementation Plan For Next Phase

### Task 3A: Extract an official runtime adapter

Create a small shared adapter around `official_1view.py` so future CR code can reuse official loading without importing the old lightweight wrapper path.

Files:
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/official_1view.py`
- Create or modify: `experiment/tests/test_rtgs_official_1view.py`

Checks:
- Existing official 1-view tests still pass.
- The adapter returns `official`, `camera`, `gt`, `pipe`, and `background` without rendering.

### Task 3B: Add snapshot-reference diagnostics

Add a helper that materializes RTGS geometry/color for one timestamp and records tensor statistics.

Files:
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/cli.py` or a new focused module if `cli.py` grows too much.
- Test: `experiment/tests/test_rtgs_cr_1view.py`

Checks:
- Shape contract for dnerf/N3DV is deterministic.
- Temporal mask count and color shape are written to `manifest.json`.
- Color direction uses timestamp-conditioned means for `rot_4d=True`, matching the official Python SH branch as closely as possible.
- Tensor diagnostics include NaN/Inf counts, mask ratio, opacity min/max/mean, color min/max/mean, and covariance min/max or eigenvalue sanity.
- A projection sanity check compares RTGS `full_proj_transform` against gsplat `viewmat + K` for sampled snapshot means.

### Task 3C: Implement separate CR 1-view entrypoint

Create a new command that renders official RTGS and CR side by side without modifying the Step 1 baseline command.

Files:
- Create: `experiment/src/lkg_experiment/rtgs_coherent/cr_1view.py`
- Create: `experiment/rtgs_cr_1view.py`
- Create: `experiment/tests/test_rtgs_cr_1view.py`

Checks:
- Non-CUDA tests cover parser defaults, output path separation, and all-zero viewpoint index contract.
- CUDA smoke runs for `jumpingjacks` and `coffee_martini`.

### Task 3D: Decide whether Task 4 is allowed

After CR 1-view outputs exist, compare metrics and images in gate order. If official-vs-snapshot fails, stop and debug snapshot materialization. If snapshot-vs-CR fails, stop and debug CR rasterizer convention, K, viewmat, background, and lookup shape. If the only remaining mismatch is official-vs-CR and the images are still interpretable, continue only with a clearly documented threshold. If the image has geometric corruption, do not start multi-view.

### Task 4A: Promote snapshot path into `views66.py`

Replace the old RTGS wrapper loading path in `views66.py` with the same official runtime/snapshot path validated by Task 3.

This is a blocking gate. Metrics from the existing `load_rtgs_checkpoint()` / lightweight wrapper path must not be used as Task 4 acceptance evidence.

### Task 4B: Multi-view debug order

Run:

```text
1. cluster_size=1, remap disabled, linear debug map, sampled full-frame comparisons only
2. cluster_size=1, remap enabled, linear debug map, same comparisons
3. cluster_size=1, remap enabled, real LKG map, sampled comparisons plus interlaced output
4. cluster_size=2 or 4, remap enabled, sampled comparisons
5. cluster_size=8, remap enabled, sampled comparisons plus interlaced output
```

Only then run all-scene or video scripts.

## Open Questions

1. The exact CR-vs-official PSNR threshold should be set after the first valid CR smoke, because gsplat CR and RTGS diff rasterizer are not pixel-identical renderers.
2. If Python snapshot-reference already differs from official default render, we need to decide whether to force official comparison through Python covariance/color for CR validation or port RTGS CUDA 4D evaluation into gsplat CR.
3. LKG physical validation remains deferred until the display is connected. For now, visual validation is image artifact only.
