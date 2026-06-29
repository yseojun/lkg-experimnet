# RTGS original render rewrite decision log

## Purpose

이 문서는 `rtgs_basic_1view.py`와 관련 wrapper를 계속 부분 수정하지 않기 위한 실패 기록이다. 목표는 앞으로 RTGS original render를 다시 설계할 때, 이미 실패한 가설과 우회 구현을 반복하지 않는 것이다.

현재 결론은 명확하다.

- 기존 wrapper는 original RTGS render 흐름을 재현한다고 보기 어렵다.
- command option, camera mapping, rotation convention 같은 개별 보정만으로는 dnerf/N3DV 모두를 안정적으로 맞추기 어렵다.
- 다음 단계는 wrapper를 더 고치는 것이 아니라, official/original `render.py` 실행 흐름을 기준으로 렌더링 harness를 처음부터 다시 세우는 것이다.

## What Was Tried

### 1. Official coffee_martini output alignment

처음에는 N3DV `coffee_martini`의 official render와 wrapper render를 1-view로 비교했다.

사용한 기준 결과:

- Scene: `coffee_martini`
- Camera/frame: `cam00_0000`
- Official output index: `00196.png`
- `gt` vs `official_gt`: 약 51 dB
- `rtgs_render` vs `official_render`: 약 20 dB

이 실험에서 확인한 점:

- GT frame/order mapping은 맞았다.
- camera pose와 큰 scene layout도 대체로 맞았다.
- 문제는 GT 선택이나 output index보다는 RTGS render 자체의 Gaussian evaluation/rasterization 경로에 있었다.

실패한 점:

- 이 비교는 "어느 layer가 official과 다른지"를 정확히 분리하지 못했다.
- wrapper가 만든 camera/checkpoint/pipeline이 original RTGS의 `render.py`와 완전히 같은지 먼저 증명하지 못한 상태에서, 후속 수정들이 특정 layer를 원인으로 가정했다.

### 2. Camera/frame/order 문제 배제 시도

N3DV official output은 original RTGS의 train-then-test shuffle 저장 순서를 따른다고 보고, `cam00_0000`이 `00196.png`에 대응하도록 맞췄다. `gt`와 `official_gt`가 약 51 dB로 맞았기 때문에 frame/order mismatch는 1차 원인에서 제외했다.

실패한 점:

- GT alignment가 맞는 것은 render input 전체가 맞는다는 뜻이 아니다.
- original render는 dataset reader, scene construction, camera list ordering, timestamp handling, background, pipeline flags까지 함께 묶인 흐름인데, wrapper는 그 흐름을 재구성해서 흉내 낸다.
- 따라서 GT alignment 하나만으로 wrapper camera가 original `Scene` camera와 동일하다고 결론 내리면 안 된다.

### 3. Checkpoint vs PLY artifact 비교

`checkpoints/chkpnt_best.pth`와 `point_cloud/iteration_best/point_cloud.ply`의 주요 field를 샘플링 비교했다.

확인한 field:

- `xyz`
- `opacity`
- `scale`
- `rotation`
- `t`
- `scale_t`
- `rotation_r`
- `f_dc`
- `f_rest`

샘플 기준으로는 checkpoint와 PLY가 일치했다.

실패한 점:

- artifact 자체가 일치해도, original render가 어떤 artifact를 어떤 code path로 restore했는지는 별도 문제다.
- 특히 wrapper의 `RtgsLiteGaussianModel`은 original `GaussianModel`을 완전히 사용하는 것이 아니라, restore와 property subset을 재구현한다.
- 이 비교는 "model bytes가 전혀 다른가"만 배제했지, restore 이후 object behavior가 동일한지는 증명하지 못했다.

### 4. 4D rotation convention mismatch 가설

현재 `4d-gaussian-splatting/main`에는 `900c592 Flip the 4D rotation matrix for easier static regularization` 이후의 convention 변경이 들어있다. official reference가 pre-flip convention에서 생성되었을 가능성이 있어 다음을 변경했다.

변경한 내용:

- `utils/general_utils.py`의 `build_rotation_4d()`에서 `.flip(1,2)` 제거
- CUDA `forward.cu`의 `computeCov3D_conditional()` matrix block을 legacy convention으로 변경
- CUDA `backward.cu`의 matrix block과 quaternion gradient sign을 legacy convention으로 변경
- wrapper 기본 `--rtgs-rotation-convention`을 `legacy`로 변경
- legacy라는 이유만으로 `compute_cov3D_python=True`를 강제하던 로직 제거
- `render_rtgs_original()`에서 legacy + Python covariance일 때 `override_color`를 자동 주입하던 우회 제거

검증한 내용:

- source/static test는 legacy matrix block을 확인하도록 추가했다.
- `python -m unittest discover -s experiment/tests`는 통과했다.

실패한 점:

- 이 수정은 local unit/static level에서만 검증되었고, CUDA render verification은 실행 환경에 CUDA가 없어 완료하지 못했다.
- 이후 사용자가 실제 환경에서 여전히 오류를 확인했다.
- 즉 rotation convention이 문제의 일부일 수는 있어도, original render 실패의 root cause로 확정되지 않았다.
- 더 중요한 문제는 wrapper가 original render 흐름 자체를 그대로 실행하지 않는다는 점이다.

### 5. dnerf command option 수정

dnerf `jumpingjacks`에 대해 처음 사용한 명령은 다음 형태였다.

```bash
PYTHONPATH=src python rtgs_basic_1view.py \
  --model-path /data/ysj/result/4dgs/RTGS/jumpingjacks/ \
  --n3dv-root /data/ysj/dataset/dnerf/ \
  --camera-index 0 \
  --no-ssim
```

이 명령에서 문제로 본 점:

- dnerf root는 `--n3dv-root`가 아니라 `--dataset-root`로 전달해야 한다.
- `--n3dv-root`는 N3DV scene fallback용이다.

제안했던 수정 명령:

```bash
PYTHONPATH=src python rtgs_basic_1view.py \
  --model-path /data/ysj/result/4dgs/RTGS/jumpingjacks \
  --dataset-root /data/ysj/dataset/dnerf \
  --camera-index 0 \
  --run-label dnerf_legacy_cuda \
  --no-ssim \
  --no-compare-official
```

실패한 점:

- 사용자가 이 수정 후에도 오류를 확인했다.
- 정확한 최신 stack trace는 아직 이 문서에 기록되지 않았다.
- command option correction은 scene path resolution 문제만 다룬다. original render와 wrapper render의 구조적 차이는 해결하지 못한다.

## Root Problem Pattern

지금까지의 실패 패턴은 다음과 같다.

1. Wrapper가 original RTGS render 흐름 일부를 재구현한다.
2. 특정 scene에서 문제가 보이면 camera, dataset root, convention, covariance/color branch 같은 개별 가설을 세운다.
3. 해당 부분만 보정한다.
4. 다른 scene 또는 다른 code path에서 다시 오류가 난다.

이 패턴은 단일 버그라기보다 설계 문제에 가깝다. 특히 `rtgs_basic_1view.py`는 이름상 original render baseline이어야 하지만, 실제로는 다음 책임을 동시에 가진다.

- RTGS config path 추론
- dnerf/N3DV dataset path 추론
- original camera loader 일부 재구현
- N3DV dynamic camera handling
- checkpoint restore를 lightweight model로 재구현
- pipeline namespace 재구성
- official reference index 추론
- diagnostics/metrics/output 저장

이 많은 책임이 한 script에 섞여 있으면, "original render와 같은가"를 확인하기 어렵다.

## What Not To Repeat

다음 방식은 반복하지 않는다.

- original `render.py`를 완전히 읽고 실행 흐름을 고정하기 전에 wrapper에 또 다른 option 보정을 추가하지 않는다.
- dnerf와 N3DV를 하나의 추론 함수에서 암묵적으로 fallback시키지 않는다.
- `RtgsLiteGaussianModel`이 original `GaussianModel`과 같다고 가정하지 않는다.
- camera pose가 시각적으로 맞는다는 이유로 camera loader가 original과 같다고 결론 내리지 않는다.
- GT alignment가 맞는다는 이유로 render input 전체가 맞다고 결론 내리지 않는다.
- CUDA convention, Python covariance, SH color override 같은 우회를 original render baseline에 섞지 않는다.
- generated image/metric 결과를 덮어쓰는 명령을 run label 없이 실행하지 않는다.

## Redesign Direction

새 설계는 wrapper-first가 아니라 official-flow-first여야 한다.

### Layer 1. Official RTGS render harness

처음에는 original RTGS repository의 render 흐름을 그대로 호출하는 얇은 harness를 만든다.

원칙:

- original `arguments.ModelParams`, `PipelineParams`, `OptimizationParams` 초기화 방식을 따른다.
- original `Scene` 생성 방식을 따른다.
- original `GaussianModel` restore/loading 방식을 따른다.
- original `gaussian_renderer.render()` 호출 signature를 그대로 사용한다.
- wrapper-specific camera class나 lite model을 사용하지 않는다.
- output 저장만 experiment 쪽에서 담당한다.

이 layer의 목표는 "공식 RTGS render.py와 같은 이미지가 나오는가"만 검증하는 것이다.

### Layer 2. Dataset-specific entrypoints

dnerf와 N3DV는 entrypoint를 분리한다.

- dnerf: `--dataset-root`, `transforms_train.json`, `transforms_test.json`, `frame_ratio`, `time_duration`
- N3DV: raw N3DV camera/json path, transformed RTGS dataset path, official output order handling

공통 fallback 추론은 최소화한다. scene kind는 CLI에서 명시하거나, config에서 한 번만 결정하고 manifest에 기록한다.

### Layer 3. Adapter comparison

official harness가 맞은 뒤에만 adapter를 비교한다.

비교 순서:

1. official `render.py` output
2. new official-flow harness output
3. current `rtgs_basic_1view.py` output
4. CR/gsplat snapshot output

이 순서가 지켜져야 어느 layer에서 차이가 생겼는지 분리할 수 있다.

## Minimum Acceptance Tests For Rewrite

새 구현은 최소한 다음 검증을 통과해야 한다.

- dnerf `jumpingjacks`, fixed camera index 0 render가 original RTGS `render.py`와 같은 output을 낸다.
- N3DV `coffee_martini`, `cam00_0000`, timestamp 0 render가 official reference 또는 original `render.py` rerun과 같은 output을 낸다.
- dnerf command에는 `--n3dv-root`가 필요 없어야 한다.
- N3DV command에는 dnerf `--dataset-root` fallback에 의존하지 않아야 한다.
- output directory는 `--run-label` 또는 equivalent run id 없이는 diagnostic variant를 덮어쓰지 않아야 한다.
- manifest에는 다음 값이 반드시 기록되어야 한다.
  - actual RTGS code root
  - RTGS git commit if available
  - selected dataset kind
  - selected source path
  - selected config path
  - checkpoint path
  - split
  - camera uid/name/index
  - timestamp
  - PipelineParams
  - ModelParams relevant fields
  - extension cache path

## Open Items

- 사용자가 확인한 최신 dnerf 오류의 full stack trace를 아직 문서화하지 못했다.
- patched legacy CUDA convention이 coffee_martini official PSNR을 개선하는지 CUDA 환경에서 최종 확인하지 못했다.
- original RTGS `render.py`와 current wrapper의 call graph diff를 아직 작성하지 않았다.
- `RtgsLiteGaussianModel`을 original render baseline에서 완전히 제거할지, adapter comparison 단계에서만 유지할지 결정해야 한다.

## Current Decision

현재 wrapper에 추가 보정을 계속 넣는 방식은 중단한다.

다음 작업은 original RTGS render code를 기준으로 한 새 1-view harness 설계다. 이 harness가 original output과 일치한 뒤에만, 기존 wrapper/CR/gsplat adapter를 비교 대상으로 삼는다.

## Task 1 Implementation Note: Official 1-View Harness

`experiment/rtgs_official_1view.py`와 `lkg_experiment.rtgs_coherent.official_1view`를 추가했다. 이 harness는 기존 `RtgsLiteGaussianModel` 경로를 쓰지 않고, RTGS checkout의 실제 `arguments`, `Scene`, `GaussianModel`, `gaussian_renderer.render()`를 사용한다.

RTGS official training/rendering code의 인자 흐름은 nested config object가 아니라 flat `argparse.Namespace`다.

1. `ArgumentParser`를 만든다.
2. `ModelParams(parser)`, `OptimizationParams(parser)`, `PipelineParams(parser)`를 생성해 각 group option을 parser에 등록한다.
3. `--config`, `--gaussian_dim`, `--time_duration`, `--num_pts`, `--rot_4d` 같은 train.py top-level option을 같은 parser에 추가한다.
4. YAML config를 recursive merge하면서 leaf key를 같은 flat namespace attribute에 `setattr(args, key, value)`로 덮어쓴다.
5. `lp.extract(args)`, `op.extract(args)`, `pp.extract(args)`로 RTGS group namespace를 만든다.
6. `GaussianModel(..., gaussian_dim=args.gaussian_dim, time_duration=args.time_duration, rot_4d=args.rot_4d, ...)`와 `Scene(..., num_pts=args.num_pts, num_pts_ratio=args.num_pts_ratio, time_duration=args.time_duration)`에 그대로 전달한다.

현재 official 1-view harness도 위 순서를 따른다. Harness CLI의 `--model-path`, `--source-path`, `--device` 같은 로컬 실행 값은 YAML merge 이후 `defaults.model_path`, `defaults.source_path`, `defaults.data_device`에 명시적으로 반영한다.

구현 원칙:

- `--dataset-kind dnerf|n3dv`를 필수로 받아 dnerf/N3DV 경로 fallback을 막는다.
- dnerf는 `--dataset-root/<scene>`를 source 후보로 사용한다.
- N3DV는 `--n3dv-root/<scene>` 또는 alias를 사용하고, `colmap/sparse`가 있으면 RTGS `Scene`이 읽을 수 있는 `colmap` 경로를 source로 사용한다.
- `--source-path`를 주면 dataset kind와 무관하게 명시 source path를 사용하되, RTGS `Scene`이 읽을 수 있는 `transforms_train.json` 또는 `sparse` layout인지 먼저 검증한다.
- LKG display가 연결되지 않은 환경에서는 display 전송을 시도하지 않고 `rtgs_official_render.png`, `gt.png`, `comparison.png`, `metrics.json`만 저장한다.
- manifest의 `lkg_display.validation_mode`는 `image_artifact_only`로 기록한다.

현재 환경 검증:

- 일반 Codex sandbox 안에서는 `/dev/nvidia*`가 없고 `torch.cuda.is_available()`가 `False`다.
- sandbox 밖 escalated 실행에서는 `nvidia-smi`가 GPU 3개를 확인했고, `rtgs-coherent-cu121` PyTorch도 CUDA를 정상 인식했다.
- `conda run -n rtgs-coherent-cu121 ... python -m unittest discover -s tests -v`는 통과했다.
- dnerf `jumpingjacks` official 1-view 재검증 산출물:
  `experiment/generated/rtgs_official_1view/jumpingjacks/jumpingjacks_test0_official_argscheck/`
- N3DV `coffee_martini` official 1-view 재검증 산출물:
  `experiment/generated/rtgs_official_1view/coffee_martini/coffee_martini_cam00_0000_official_argscheck/`
- dnerf `jumpingjacks`와 N3DV `coffee_martini` 모두 escalated CUDA 실행으로 1-view image artifact를 생성했다.

재실행 명령은 다음과 같다.

```bash
conda run -n rtgs-coherent-cu121 bash -lc 'cd experiment && PYTHONPATH=src python rtgs_official_1view.py \
  --dataset-kind dnerf \
  --model-path /data/ysj/result/4dgs/RTGS/jumpingjacks \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --dataset-root /data/ysj/dataset/dnerf \
  --split test \
  --camera-index 0 \
  --run-label jumpingjacks_test0_official'
```

```bash
conda run -n rtgs-coherent-cu121 bash -lc 'cd experiment && PYTHONPATH=src python rtgs_official_1view.py \
  --dataset-kind n3dv \
  --model-path /data/ysj/result/4dgs/RTGS/coffee_martini \
  --checkpoint checkpoints/chkpnt_best.pth \
  --rtgs-code-root /home/ysj/lkg-experiment/4d-gaussian-splatting \
  --n3dv-root /data/ysj/dataset/N3DV \
  --split test \
  --camera-index 0 \
  --n3dv-frame-index 0 \
  --run-label coffee_martini_cam00_0000_official'
```

생성된 산출물:

- `experiment/generated/rtgs_official_1view/jumpingjacks/jumpingjacks_test0_official/rtgs_official_render.png`
- `experiment/generated/rtgs_official_1view/jumpingjacks/jumpingjacks_test0_official/gt.png`
- `experiment/generated/rtgs_official_1view/jumpingjacks/jumpingjacks_test0_official/comparison.png`
- `experiment/generated/rtgs_official_1view/jumpingjacks/jumpingjacks_test0_official/metrics.json`
- `experiment/generated/rtgs_official_1view/coffee_martini/coffee_martini_cam00_0000_official/rtgs_official_render.png`
- `experiment/generated/rtgs_official_1view/coffee_martini/coffee_martini_cam00_0000_official/gt.png`
- `experiment/generated/rtgs_official_1view/coffee_martini/coffee_martini_cam00_0000_official/comparison.png`
- `experiment/generated/rtgs_official_1view/coffee_martini/coffee_martini_cam00_0000_official/metrics.json`

## Task 1 Re-analysis: Official Flow Input Contract

2026-06-29에 official 1-view harness를 다시 검증했다.

확정한 내용:

- dnerf `jumpingjacks`는 dirty RTGS checkout 대신 clean HEAD snapshot을 사용할 때 saved official reference와 렌더가 일치한다. `rtgs_official_render.png` vs official `renders/00000.png`는 약 61.6 dB다.
- `4d-gaussian-splatting` working tree에는 CUDA rasterizer/general utility local edit가 있다. 기본 `--rtgs-code-policy clean`은 이 dirty tree를 직접 쓰지 않고 `/tmp/lkg_rtgs_official_code/...` 아래 clean archive snapshot을 사용한다.
- N3DV `coffee_martini`는 local RTGS `Scene`만 사용하면 static colmap camera가 선택된다. 올바른 dynamic frame 검증에는 `model_path/cameras.json`과 raw `poses_bounds.npy` 기반 camera가 필요하다.
- N3DV dynamic loader는 `cam00_0000` GT를 선택하며, 이 GT는 saved official `gt/00196.png`와 가장 가깝다. raw frame과 harness GT는 동일하고, official GT와의 차이는 원본 full-resolution/downsample artifact 차이로 보인다.
- checkpoint `chkpnt_best.pth`와 `point_cloud/iteration_best/point_cloud.ply`는 sampled field 기준으로 같은 model artifact다. `f_rest`는 PLY flatten order만 다르며 transpose하면 일치한다.

Resolved N3DV render mismatch:

- Root cause: N3DV dynamic loader recomputed positive `FoVx/FoVy` from focal length. RTGS official `readCamerasFromTransforms()` keeps `FovX=FovY=-1.0` when explicit intrinsics are available, while `fl_x/fl_y/cx/cy` drive `getProjectionMatrixCenterShift()`. The rasterizer still receives `tan(FoVx/2)`, so this sentinel is part of the official render contract.
- Harness before fix: `cam00_0000`, `rtgs_vs_gt_psnr` 약 20.1 dB.
- Harness after fix: `coffee_martini_fov_sentinel_fix`, `rtgs_vs_gt_psnr` 약 27.52 dB; saved PNG render vs official `renders/00196.png` 약 51.2 dB.
- Tested and rejected as primary causes before finding the FoV contract issue: `chkpnt_best` vs `chkpnt_30000`, env map (`env_map` is `None` in checkpoint), focal scaling, `compute_cov3D_python=True`, `force_sh_3d=True`, `scaling_modifier`, RTGS `d61f57d`, and RTGS pre-flip `5094ca8`.

1단계 완료 판정:

- RTGS official 1-view render baseline is now fixed as the return point for later coherent-raster experiments.
- dnerf and N3DV are selected explicitly with `--dataset-kind`; N3DV uses dynamic `cameras.json` plus raw `poses_bounds.npy`, while dnerf continues to use the official `Scene` camera dataset.
- The LKG display path is intentionally not exercised in this milestone because no display is connected; validation is by saved image artifacts and manifest metrics.
- Final 1-view sweep passed for all configured RTGS scenes: 8 dnerf scenes and 6 N3DV scenes.
- `flame_salmon` exposed an edge case where `cameras.json` contains frame 300 but raw `cam00/images` ends at `0299.png`; the N3DV dynamic loader now skips unrequested frame indices before validating image files.
- `experiment/scripts/run_rtgs_1view_videos_all.sh` can generate per-scene 1-view videos by rendering official 1-view frame sequences and encoding them with ffmpeg.

Manifest additions:

- `render_contract_tensor_shapes` now includes `env_map`.
- `camera_render_contract` records width, height, FoV, focal, principal point, near/far, and timestamp.
- `gaussian_model_constructor_kwargs` records the exact constructor kwargs used for `GaussianModel`.
- The harness adapts to older RTGS commits whose `GaussianModel.__init__` does not accept `prefilter_var`.
