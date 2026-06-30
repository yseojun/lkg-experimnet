# RTGS + CoherentRaster Structure Explainer Design

## Status

Approved for spec writing on 2026-06-30 after two read-only sub-agent reviews:

- planning review: identify important RTGS rendering concepts, CR application points, and page structure;
- clarity/correctness review: tighten beginner flow, qualify technical claims, and reorder lookup before one-shot CR.

## Goal

Build a static Korean web explainer that helps a reader with limited Gaussian Splatting background understand:

- how an original/official RTGS render is produced;
- why RTGS cannot be treated as a static 3DGS checkpoint;
- how the current implementation converts RTGS into a timestamp-specific snapshot before CoherentRaster;
- how CR is used first for one-view validation, then for 66-view/LKG interlaced rendering;
- what correctness caveats matter when reading metrics, images, and timing.

The page should be detailed enough to serve as a code-grounded architecture explanation, but it should not assume the reader already understands RTGS, 4D SH, LKG subpixels, or CR lookup tensors.

## Non-Goals

- Do not build a live renderer, experiment dashboard, or web server.
- Do not create a marketing-style landing page.
- Do not claim that CR modifies the RTGS model or training process.
- Do not present the old 66-view compose path as the final CoherentRaster performance model.
- Do not claim arbitrary per-view intrinsics support in one-shot CR. The current path assumes a shared `K` contract.

## Recommended Approach

Create a checked-in static page under:

```text
docs/rtgs-cr-structure-explainer/index.html
```

Use local CSS under:

```text
docs/rtgs-cr-structure-explainer/assets/styles.css
```

This should follow the existing `docs/omg4-ftgs-cr-structure-explainer/` pattern: no network dependencies, left-side table of contents, stable section anchors, concise code evidence blocks, and explanatory tables. The page can reuse the same visual grammar, but the information order must be more beginner-oriented than the OMG4 page.

## Audience

The primary reader is technical and can read Python/CUDA snippets, but does not yet know the RTGS or CoherentRaster render path. The page should answer:

- What is a Gaussian splat and how does a pixel get color?
- What changes when Gaussian parameters depend on timestamp?
- What did official RTGS do before CR?
- What is a timestamp-specific RTGS snapshot?
- Why are `marginal_t`, `active_sh_degree_t`, camera center, and `FoVx/FoVy` important?
- Where does `rasterization_CR()` enter the path?
- Why do `view_idx_matrix` and `subpixel_coord_matrix` exist?
- Why are compose, one-view CR, and one-shot CR different evidence stages?

## Information Architecture

### 1. `overview`

Title: `RTGS + CR 한눈에 보기`

Teach the whole flow first:

```text
RTGS checkpoint + official Scene/camera/timestamp
  -> official RTGS render
  -> timestamp-specific 3D snapshot
  -> CR one-view validation
  -> 66-view compose correctness bridge
  -> one-shot CR LKG interlaced output
```

The first screen should separate three products:

- official RTGS image: validates official loading, camera, timestamp, and renderer;
- snapshot/CR single-view image: validates the RTGS-to-CR adapter;
- LKG interlaced panel image: validates subpixel view selection and one-shot CR output.

### 2. `3dgs-basics`

Title: `GS 렌더링을 이해하기 위한 최소 개념`

Explain the pixel-level mental model before RTGS details:

- A 3D Gaussian is projected to a 2D ellipse-like footprint.
- Each Gaussian contributes color and alpha to covered pixels.
- Front-to-back alpha blending accumulates the final pixel color.
- SH color is not always fixed RGB; it can be evaluated from camera direction.

Include a compact beginner table:

| Term | Simple meaning |
| --- | --- |
| `means` | Gaussian center positions |
| `covars` / `scales` / `quats` | shape and orientation |
| `opacities` | how strongly each Gaussian contributes |
| `colors` / SH | color coefficients or evaluated RGB |
| rasterizer | projects, sorts, and blends |

### 3. `rtgs-dynamics`

Title: `RTGS는 timestamp마다 3DGS 상태가 달라진다`

Use precise wording:

> `gaussian_dim=4`인 RTGS는 시간 관련 파라미터를 저장합니다. timestamp `t`에서 opacity는 `marginal_t`로 조정되고, `rot_4d`가 켜져 있으면 mean/covariance도 시간 조건으로 바뀝니다. color는 view-dependent이며, 4D SH를 쓰면 timestamp에도 의존할 수 있습니다.

Code evidence should point to:

- `experiment/src/lkg_experiment/rtgs_coherent/cli.py::materialize_rtgs_geometry`
- `experiment/src/lkg_experiment/rtgs_coherent/cli.py::evaluate_rtgs_colors`
- official RTGS `gaussian_renderer.render()` behavior through the checked-out RTGS code when available.

### 4. `camera-contract`

Title: `카메라 계약: official RTGS와 gsplat/CR 사이의 번역`

Explain that official RTGS consumes RTGS camera fields, while CR uses gsplat-style `viewmat` and `K`.

Required points:

- official RTGS uses `world_view_transform`, `full_proj_transform`, FoV/timestamp fields;
- CR adapter converts the same camera into `viewmat` and `K`;
- explicit intrinsics take priority when `fl_x/fl_y` are positive;
- N3DV uses `FoVx=FoVy=-1.0` as a sentinel while `fl_x/fl_y/cx/cy` drive projection;
- normalizing the FoV is diagnostic only, not the acceptance path.

Evidence:

- `experiment/src/lkg_experiment/rtgs_coherent/cli.py::rtgs_camera_to_gsplat_inputs`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_1view.py::normalize_explicit_intrinsics_fov`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_1view.py::adapt_snapshot_for_rtgs_compat_projection`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_one_shot.py::build_rtgs_projection_adapter`

### 5. `official-flow`

Title: `기존 RTGS official 렌더링 흐름`

Explain that the official baseline is not “same checkpoint loaded by a small wrapper.” It is the official object/runtime/render flow:

```text
RTGS config + checkpoint
  -> ModelParams / PipelineParams / OptimizationParams
  -> Scene
  -> GaussianModel restore
  -> selected camera + background
  -> gaussian_renderer.render(camera, gaussians, pipe, background)
```

Evidence:

- `experiment/src/lkg_experiment/rtgs_coherent/official_1view.py`
- `experiment/docs/rtgs_original_render_rewrite_decision.md`

### 6. `snapshot-materialization`

Title: `CR로 넘기기 전: timestamp snapshot 만들기`

Plain explanation:

> materialization은 시간 `t`를 하나 고른 뒤, 4D/동적 Gaussian을 그 순간의 3D Gaussian 목록으로 접는 과정입니다.

Show the adapter contract:

```text
GaussianModel + timestamp + camera center
  -> means [N,3]
  -> covars [N,6]
  -> opacities [N]
  -> colors [N,3] or [C,N,3]
  -> mask [N_total]
```

Required caveats:

- snapshot path explicitly multiplies opacity by `marginal_t`;
- active mask is `marginal_t > 0.05` for 4D Gaussian snapshots;
- color must be evaluated with timestamp-conditioned means and the relevant camera center;
- official-vs-snapshot mismatch is not automatically a CR bug.

Evidence:

- `experiment/src/lkg_experiment/rtgs_coherent/cli.py::materialize_rtgs_snapshot`
- `experiment/src/lkg_experiment/rtgs_coherent/cli.py::materialize_rtgs_geometry`
- `experiment/src/lkg_experiment/rtgs_coherent/cli.py::evaluate_rtgs_colors`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_1view.py::SnapshotGaussianProxy`

### 7. `cr-1view-ladder`

Title: `1-view CR 검증: official -> snapshot -> CR`

Explain the validation ladder:

```text
official RTGS render
  -> snapshot reference render
  -> RTGS snapshot through CoherentRaster
```

The one-view CR path uses an all-zero `viewpoint_index`, so every RGB subpixel selects view 0. This validates whether the RTGS snapshot and camera conversion can pass through CR before LKG interlacing is introduced.

Evidence:

- `experiment/src/lkg_experiment/rtgs_coherent/cr_1view.py::render_rtgs_cr_1view`
- `experiment/src/lkg_experiment/rtgs_coherent/cli.py::render_rtgs_coherent`
- `experiment/tests/test_rtgs_cr_1view.py`

### 8. `lkg-lookup`

Title: `LKG viewpoint lookup과 remapping`

Explain raw lookup before CUDA layout:

```text
viewpoint_index[y, x, c] = v
```

Example:

```text
panel pixel (10, 20)
  red subpixel   -> view 12
  green subpixel -> view 13
  blue subpixel  -> view 13
```

Then show conversion:

```text
raw HxWx3 view map
  -> pad to tile multiple
  -> patchify to [tile_h, tile_w, 3, tile_size, tile_size]
  -> optional stable remap by view id
  -> preserve original channel/y/x through subpixel_coord_matrix
```

Important wording:

- remapping changes tile-internal processing order;
- it does not change which output subpixel each result belongs to;
- `subpixel_coord_matrix` preserves original coordinates.

Evidence:

- `experiment/src/lkg_experiment/coherent_default/coherent_gsplat_bridge.py::build_cr_lookup_arrays`
- `experiment/src/lkg_experiment/coherent_default/coherent_gsplat_bridge.py::_remap_subpixel_coord`
- `rtgs-coherent-raster/modified-files/gsplat/cuda/csrc/CoherentRaster_CRKernel.cu`

### 9. `cr-66-compose`

Title: `66-view compose 경로: correctness bridge`

Explain that compose renders each synthetic view through CR single-view, then copies selected subpixels into the final interlaced panel. It is useful because it preserves the one-view adapter behavior per view and gives sampled full-frame comparison images.

Required caveat:

> compose는 correctness bridge입니다. 66개 view를 각각 렌더하므로 CoherentRaster 논문식 one-shot 성능 모델로 해석하면 안 됩니다.

Evidence:

- `experiment/src/lkg_experiment/rtgs_coherent/cr_66views.py::render_rtgs_cr_66views`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_66views.py::accumulate_interlaced_view`
- `docs/superpowers/specs/2026-06-29-rtgs-cr-66view-design.md`

### 10. `one-shot-cr`

Title: `one-shot CR: interlaced panel을 한 번의 CR 호출로 만들기`

Explain:

- materialize RTGS geometry once per timestamp;
- synthesize grouped adjacent view matrices;
- evaluate color at cluster representative camera centers;
- build viewport-cropped CR lookup tensors;
- call `rasterization_CR()` once;
- unpatchify/unpad the viewport image;
- paste the viewport into the panel image.

Use precise wording:

> one-shot CR은 모든 view별 full image를 먼저 만들고 섞는 방식이 아닙니다. grouped/reference-view projection과 lookup tensor를 사용해 interlaced target을 한 CR 경로에서 만듭니다.

Required caveats:

- clustered color reuse is an approximation because RTGS color is view-dependent;
- `cluster_size=1` is the correctness reference;
- current path uses shared `K`;
- one-shot path passes CUDA `rtgs_projection_adapter` into `rasterization_CR()` for sentinel-FoV cameras.

Evidence:

- `experiment/src/lkg_experiment/rtgs_coherent/cr_one_shot.py::prepare_rtgs_cr_one_shot_state`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_one_shot.py::render_rtgs_cr_one_shot_viewport_image`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_one_shot.py::render_rtgs_cr_one_shot_interlaced_once`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_experiment.py`
- `rtgs-coherent-raster/modified-files/gsplat/rendering_coherent_raster.py::rasterization_CR`

### 11. `cuda-contract`

Title: `CR CUDA 계약`

Summarize the inputs the CUDA side needs:

| Input | Role |
| --- | --- |
| `adjacent_viewmats [C,S,4,4]` | grouped views around the anchor/source view |
| `view_idx_matrix` | which view id a tile/subpixel thread should use |
| `subpixel_coord_matrix` | original output channel/y/x after remapping |
| `translation_values` | per-Gaussian, per-view 2D projection offset |
| `rtgs_projection_adapter [6]` | optional sentinel-FoV projection compensation |

Evidence:

- `rtgs-coherent-raster/modified-files/gsplat/rendering_coherent_raster.py`
- `rtgs-coherent-raster/modified-files/gsplat/cuda/csrc/CoherentRaster_CRKernel.cu`
- `rtgs-coherent-raster/patches/0001-Add-RTGS-projection-adapter-to-coherent-raster.patch`
- `rtgs-coherent-raster/patches/0002-Add-coherent-raster-timing-metrics.patch`

### 12. `timing-and-metrics`

Title: `timing과 metric을 어떻게 읽을 것인가`

Explain the RTGS + CR metric groups:

- RTGS dynamic timing: geometry, temporal opacity, compaction, color;
- CR core timing: projection, keygen/isect/sort/offset, blend;
- LKG post timing: unpatchify, unpad, panel paste;
- aggregate timing: `frame_ms_without_lkg`, `frame_ms_with_lkg_interlace`.

Evidence:

- `experiment/docs/rtgs_cr_metric_columns_ko.md`
- `experiment/src/lkg_experiment/rtgs_coherent/cr_one_shot.py::RtgsCrFrameTiming`

### 13. `caveats`

Title: `틀리기 쉬운 해석`

Include warning cards:

- CR does not modify the RTGS model.
- A snapshot-vs-official difference must be diagnosed before blaming CR.
- RTGS color is not fixed RGB.
- N3DV `FoV=-1` is a sentinel contract, not automatically an invalid camera.
- compose is not the one-shot performance model.
- remapping is not changing the view map; it changes processing order and preserves output coordinates.
- one-shot CR currently assumes shared `K`.

### 14. `takeaway`

Title: `최종 요약`

Final message:

> RTGS의 핵심은 시간, color, camera contract입니다. CoherentRaster는 그 model stage를 바꾸지 않고, timestamp snapshot을 CR-compatible tensor로 만든 뒤 rasterization/backend와 LKG subpixel view selection을 바꿉니다. 그래서 올바른 해석 순서는 official RTGS, snapshot reference, CR one-view, compose bridge, one-shot CR입니다.

## Visual And Layout Rules

- Use text-first static layout with left-side navigation.
- Use compact tables, code evidence blocks, and pipeline boxes.
- Avoid decorative hero imagery or generated images. This is a technical explainer, not a landing page.
- Keep cards only for repeated summary/warning items.
- Preserve mobile readability with single-column breakpoints.
- Use local assets only.

## Required Wording Rules

Use these formulations:

- `CR은 RTGS가 만든 timestamp snapshot을 다른 rasterizer/backend로 렌더링한다.`
- `compose는 correctness bridge이고, one-shot이 CR식 실행 모델이다.`
- `N3DV에서는 FoV=-1이 explicit intrinsics를 쓰기 위한 sentinel contract다.`
- `snapshot과 official 차이는 먼저 분리해야 하며, 자동으로 CR 오류라고 볼 수 없다.`
- `RTGS color는 SH/4D SH로 view/time에 따라 평가될 수 있다.`
- `remapping은 tile 내부 처리 순서를 바꾸고 원래 좌표는 subpixel_coord_matrix가 보존한다.`
- `현재 one-shot 경로는 shared K 계약을 전제로 한다.`

Avoid these formulations:

- `CR이 RTGS 모델을 바꿨다.`
- `RTGS 4D Gaussian을 그대로 CR에 넣었다.`
- `66-view compose가 CoherentRaster 최종 성능이다.`
- `FoV=-1은 잘못된 camera 값이다.`
- `색은 Gaussian마다 고정 RGB다.`
- `remapping은 view id를 바꾼다.`
- `one-shot은 모든 view별 K를 자유롭게 지원한다.`

## Verification Plan

Add or update a static page test similar to `experiment/tests/test_explanatory_web_page.py`.

The test should verify:

- `docs/rtgs-cr-structure-explainer/index.html` exists;
- `lang="ko"` is present;
- all required section ids are present;
- only local assets are referenced;
- key terms appear:
  - `RTGS`
  - `CoherentRaster`
  - `timestamp`
  - `materialize`
  - `marginal_t`
  - `viewpoint_index`
  - `view_idx_matrix`
  - `subpixel_coord_matrix`
  - `rtgs_projection_adapter`
  - `rasterization_CR`
  - `frame_ms_without_lkg`
  - `frame_ms_with_lkg_interlace`
  - `최종 요약`

Suggested local verification command:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_explanatory_web_page -v
```

If adding a separate test class/file is cleaner, use a focused RTGS explainer test rather than overloading the OMG4 page assertions.

## Implementation Notes

- Reuse style concepts from `docs/omg4-ftgs-cr-structure-explainer/assets/styles.css`, but adjust copy and section anchors for RTGS.
- Keep code snippets compact. Prefer pseudo-code when exact snippets would be too long.
- Ground every technical section in file/function references.
- Mention that `experiment/src/lkg_experiment/rtgs_coherent/views66.py` exists as an older/reference path only if useful; the page should focus on `cr_1view.py`, `cr_66views.py`, `cr_one_shot.py`, and `cr_experiment.py`.
- Do not link to external resources; all evidence is local repository context.
