# RTGS + CoherentRaster Metric 컬럼

작성일: 2026-06-29

이 문서는 RTGS + CoherentRaster 실험에서 생성되는 `metrics.csv` / `metrics.json` 컬럼의 의미를 정의한다. dynamic RTGS timing breakdown을 추가한 뒤에도 실험 산출물을 일관되게 해석할 수 있도록 하는 것이 목적이다.

타이밍 기준은 하나의 timestamp에서 한 프레임을 렌더링하는 과정이다. checkpoint 로딩, CUDA extension 빌드, 이미지 파일 저장, web asset 생성, metric/reference 렌더링은 별도 컬럼에서 명시하지 않는 한 render timing에 포함하지 않는다.

## 식별자 컬럼

| 컬럼 | 의미 |
| --- | --- |
| `dataset_kind` | 데이터셋 종류. 예: `dnerf`, `n3dv`. |
| `scene` | scene 또는 checkpoint 디렉토리 이름. |
| `checkpoint` | 실험에 사용한 checkpoint 경로. |
| `camera_split` | 실험에서 선택한 camera split. 예: `train`, `test`. |
| `camera_index` | 해당 프레임의 anchor camera index. |
| `frame_index` | 데이터셋 frame index. N3DV에서는 `--n3dv-frame-index`에 대응하며, dnerf에서는 선택된 timestamp frame을 식별해야 한다. |
| `timestamp` | dynamic Gaussian을 materialize할 때 사용한 RTGS timestamp. |
| `output_prefix` | image artifact 저장 시 사용하는 선택적 파일명 prefix. |
| `engine` | 실험 engine. 현재 RTGS + CR experiment는 `one_shot`을 사용한다. |
| `variant` | variant 이름. 예: `cluster_8`, `without_remap`, `without_reuse`, `without_reuse_without_remap`. RTGS + CR 실험에서 `without_reuse`는 official RTGS reference/GT variant로 사용한다. |
| `group` | aggregation/comparison에 사용하는 variant group. |

## 설정 컬럼

| 컬럼 | 의미 |
| --- | --- |
| `render_width` | viewport image 렌더링에 사용한 width. |
| `render_height` | viewport image 렌더링에 사용한 height. |
| `source_views` | interlaced image를 구성하는 LKG source view 수. |
| `saved_views` | 육안 확인용으로 저장한 개별 source-view image 수. interlaced view 구성에 사용되는 view 수를 바꾸지는 않는다. |
| `cluster_size` | 하나의 CoherentRaster cluster로 묶는 adjacent view 수. |
| `color_eval_views` | RTGS color evaluation에 사용한 camera center 수. one-shot clustered rendering에서는 보통 `source_views`가 아니라 cluster 수와 같다. |
| `use_remapping` | CoherentRaster view/subpixel remapping 사용 여부. |
| `reuse_enabled` | clustered attribute reuse 사용 여부. false이면 reuse-sensitive 작업에서는 cluster size 1과 같은 동작을 해야 한다. |
| `tile_size` | CoherentRaster tile size. |
| `map_mode` | interlaced lookup 생성을 위해 사용한 LKG view mapping mode. |
| `camera_aspect_mode` | synthetic camera aspect 정책. `expand`는 LKG 출력을 위해 full-panel 9:16 camera를 만들고, `preserve`는 기존 `aspect_fit` viewport/camera 동작을 유지한다. |

## RTGS Dynamic Timing

이 컬럼들은 CoherentRaster core 이전에 dynamic RTGS 때문에 추가되는 작업 시간을 측정한다.

| 컬럼 | 의미 |
| --- | --- |
| `dynamic_geometry_ms` | timestamp-conditioned Gaussian mean/covariance 계산 시간. RTGS mean offset을 사용하는 경우 해당 계산을 포함한다. |
| `temporal_opacity_ms` | timestamp marginal weight, opacity modulation, temporal active mask 계산 시간. |
| `snapshot_compaction_ms` | mask가 적용된 tensor를 render-ready contiguous tensor로 압축하는 시간. |
| `dynamic_color_ms` | RTGS view-dependent color evaluation 시간. 4D SH를 사용하는 경우 해당 계산을 포함한다. |
| `rtgs_dynamic_total_ms` | `dynamic_geometry_ms`, `temporal_opacity_ms`, `snapshot_compaction_ms`, `dynamic_color_ms`의 합. |

## CoherentRaster Core Timing

이 컬럼들은 CoherentRaster의 core pipeline에 해당하며, static 3DGS CoherentRaster breakdown과 비교하기 위한 기준이다.

| 컬럼 | 의미 |
| --- | --- |
| `cr_projection_ms` | Gaussian projection과 projected footprint 준비 시간. |
| `cr_keygen_ms` | tile/view/subpixel key 생성 시간. |
| `cr_sort_ms` | key/depth sorting과 range 준비 시간. |
| `cr_blend_ms` | alpha blending 및 CR output buffer color accumulation 시간. |
| `cr_core_total_ms` | `cr_projection_ms`, `cr_keygen_ms`, `cr_sort_ms`, `cr_blend_ms`의 합. |

## LKG Interlace Post Timing

이 단계는 CoherentRaster core timing과 분리해서 기록한다. CR 결과를 LKG display용 image tensor로 구성하는 과정이기 때문이다.

| 컬럼 | 의미 |
| --- | --- |
| `lkg_unpatchify_ms` | CoherentRaster patch/tile output을 viewport image tensor로 변환하는 시간. |
| `lkg_unpad_ms` | tile padding을 제거하고 `render_width` x `render_height` 크기로 clamp/crop하는 시간. |
| `lkg_panel_paste_ms` | viewport image를 최종 LKG panel tensor에 배치하는 시간. |
| `lkg_interlace_post_ms` | `lkg_unpatchify_ms`, `lkg_unpad_ms`, `lkg_panel_paste_ms`의 합. |

## Aggregate Timing 컬럼

| 컬럼 | 의미 |
| --- | --- |
| `frame_ms_without_lkg` | `rtgs_dynamic_total_ms + cr_core_total_ms`. LKG interlace post processing 이전의 dynamic RTGS + CR render time. |
| `fps_without_lkg` | `1000 / frame_ms_without_lkg`. |
| `frame_ms_with_lkg_interlace` | `frame_ms_without_lkg + lkg_interlace_post_ms`. interlaced LKG tensor가 준비될 때까지의 frame time. |
| `fps_with_lkg_interlace` | `1000 / frame_ms_with_lkg_interlace`. |
| `frame_ms` | 상세 timing 구현 이후에는 `frame_ms_with_lkg_interlace`의 backward-compatible alias. |
| `fps` | 상세 timing 구현 이후에는 `fps_with_lkg_interlace`의 backward-compatible alias. |

상세 timing을 아직 수집하지 않는 run에서는 legacy `frame_ms`, `fps` 컬럼만 존재할 수 있다. 이 경우 해당 값은 파일을 생성한 runner version 기준으로 해석해야 한다.

## 품질 컬럼

이 metric들은 sampled RTGS + CR source-view render와 official RTGS renderer 결과를 비교한다. 별도 interlaced metric을 계산하지 않는 한 LKG interlaced image similarity와는 다르다.

| 컬럼 | 의미 |
| --- | --- |
| `psnr_mean` | sampled metric view들의 평균 PSNR. |
| `psnr_std` | sampled-view PSNR의 표준편차. |
| `ssim_mean` | sampled metric view들의 평균 SSIM. |
| `ssim_std` | sampled-view SSIM의 표준편차. |
| `lpips_mean` | LPIPS를 수집한 경우 sampled metric view들의 평균 LPIPS. 수집하지 않으면 `nan`. |
| `lpips_std` | LPIPS를 수집한 경우 sampled-view LPIPS의 표준편차. 수집하지 않으면 `nan`. |
| `metric_view_count` | quality metric에 사용한 sampled view 수. |

## Resource 및 Scene 컬럼

| 컬럼 | 의미 |
| --- | --- |
| `total_gaussians` | timestamp masking 이전 checkpoint에 포함된 Gaussian 수. |
| `active_gaussians` | timestamp mask/compaction 이후 남은 Gaussian 수. |
| `active_gaussian_ratio` | `active_gaussians / total_gaussians`. |
| `peak_vram_gb` | 측정된 experiment variant 실행 중 CUDA reserved memory peak. 단위는 GiB. |

## 해석 메모

- `lkg_interlace_post_ms`는 의도적으로 `cr_core_total_ms`에 포함하지 않는다.
- 현재 CR Python wrapper는 tile-intersection/key 경로를 `cr_keygen_ms`로 측정한다. `cr_sort_ms`는 schema 컬럼으로 유지하되, CUDA 경로에서 sort를 별도 timing stage로 노출하기 전까지 `0.0`으로 기록한다.
- `frame_ms_without_lkg`는 dynamic RTGS + CR을 renderer-only baseline과 비교할 때 더 공정한 값이다.
- `frame_ms_with_lkg_interlace`는 LKG-ready tensor를 실제로 생성하는 데 필요한 실용적인 값이다.
- disk output, video encoding, GLFW texture upload, display swap은 필요 시 별도 output/display 컬럼으로 측정해야 한다.
