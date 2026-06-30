# External RTGS + CoherentRaster Setup Script

Use this script when asking another engineer or coding agent to prepare a new
machine for RTGS-based Looking Glass rendering with the `rtgs-coherent-raster`
patch archive from this repository.

## Mission

Set up a CUDA Linux environment that can:

1. Load RTGS checkpoints trained with `fudan-zvg/4d-gaussian-splatting`.
2. Apply the local `rtgs-coherent-raster` patch archive to a CoherentRaster
   enabled `gsplat` checkout.
3. Run the one-shot RTGS + CoherentRaster experiment renderer.
4. Run the real-time LKG web viewer path, which must also use the one-shot
   interlaced renderer.

Do not use the old per-view compose path as the final performance/rendering
path. Use `rtgs_cr_experiment.py` for offline experiments and
`rtgs_lkg_web_server.py` for the interactive LKG viewer.

## Required Inputs

Fill these paths before starting:

```bash
export WORKSPACE="$HOME/lkg-experiment-workspace"
export LKG_EXPERIMENT_REPO="https://github.com/yseojun/lkg-experimnet.git"

export MODEL_ROOT="/data/ysj/result/4dgs/RTGS"
export DNERF_ROOT="/data/ysj/dataset/dnerf"
export N3DV_ROOT="/data/ysj/dataset/N3DV"
export GENERATED_ROOT="/data/ysj/result/coherent-raster/generated"

export DNERF_SCENE="jumpingjacks"
export N3DV_SCENE="coffee_martini"
export CHECKPOINT="checkpoints/chkpnt_best.pth"
```

Expected RTGS result layout:

```text
$MODEL_ROOT/<scene>/
  cameras.json
  cfg_args
  checkpoints/chkpnt_best.pth
```

Expected dataset layout:

```text
$DNERF_ROOT/<dnerf_scene>/transforms_train.json
$DNERF_ROOT/<dnerf_scene>/transforms_test.json
$N3DV_ROOT/<n3dv_scene>/...
```

## 1. Clone Repositories

```bash
set -euo pipefail

mkdir -p "$WORKSPACE"
cd "$WORKSPACE"

git clone "$LKG_EXPERIMENT_REPO" lkg-experiment
git clone --recursive https://github.com/fudan-zvg/4d-gaussian-splatting.git 4d-gaussian-splatting
git clone https://github.com/sgj0402/coherent-raster.git coherent-raster
git clone --recursive https://github.com/nerfstudio-project/gsplat.git gsplat
git clone https://github.com/daikiad/Bridge-Python-SDK-Lab.git Bridge-Python-SDK-Lab
```

Recommended RTGS checkout used by the current project:

```bash
git -C "$WORKSPACE/4d-gaussian-splatting" checkout 63725f2
```

Prepare `gsplat` for the original CoherentRaster patch first:

```bash
git -C "$WORKSPACE/gsplat" checkout a8d88d387f6e554b18153d309f5536696882de5c
git -C "$WORKSPACE/gsplat" switch -c cr-rtgs

cd "$WORKSPACE/coherent-raster"
python install.py
```

The `coherent-raster/install.py` script is expected to patch the sibling
`$WORKSPACE/gsplat` checkout. After it finishes, inspect the result:

```bash
git -C "$WORKSPACE/gsplat" status --short
git -C "$WORKSPACE/gsplat" log --oneline -5
```

## 2. Apply RTGS CoherentRaster Patches

The `lkg-experiment/rtgs-coherent-raster` directory is the portable patch
archive for RTGS-specific `gsplat` changes. It adds:

- RTGS projection adapter support for explicit-intrinsics N3DV cameras.
- CoherentRaster timing metrics.

Apply it to the CoherentRaster-enabled `gsplat` checkout:

```bash
cd "$WORKSPACE/lkg-experiment"

./rtgs-coherent-raster/apply_patches.sh --check "$WORKSPACE/gsplat"
./rtgs-coherent-raster/apply_patches.sh "$WORKSPACE/gsplat"
./rtgs-coherent-raster/verify_snapshot.py "$WORKSPACE/gsplat"
```

If the patch check fails, do not continue with rendering. First confirm that
`$WORKSPACE/gsplat` already contains the original CoherentRaster files. Vanilla
upstream `gsplat` is not enough.

## 3. Create The Conda Environment

Use Python 3.11 and CUDA 12.1 to match the current experiment environment.

```bash
conda create -n rtgs-coherent-cu121 python=3.11 -y
conda activate rtgs-coherent-cu121

python -m pip install --upgrade pip setuptools wheel ninja
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install \
  numpy Pillow tqdm torchmetrics imagesize kornia omegaconf \
  opencv-python imageio imageio-ffmpeg plyfile scipy scikit-image \
  lpips configargparse matplotlib pandas PyOpenGL glfw
```

Install the local experiment package:

```bash
cd "$WORKSPACE/lkg-experiment/experiment"
python -m pip install -e .
```

Install RTGS local CUDA/Python extensions:

```bash
cd "$WORKSPACE/4d-gaussian-splatting"
python -m pip install -e ./simple-knn
python -m pip install -e ./pointops2
python -m pip install -e ./diff-gaussian-rasterization
```

Install the patched `gsplat` checkout:

```bash
cd "$WORKSPACE/gsplat"
python -m pip install --no-build-isolation -e .
```

## 4. Set Runtime Environment Variables

Use a dedicated PyTorch extension cache. This prevents repeated CUDA rebuilds
from mixing incompatible extension variants.

```bash
export LKG_EXPERIMENT="$WORKSPACE/lkg-experiment"
export RTGS_CODE_ROOT="$WORKSPACE/4d-gaussian-splatting"
export GSPLAT_ROOT="$WORKSPACE/gsplat"
export BRIDGE_SDK_ROOT="$WORKSPACE/Bridge-Python-SDK-Lab"

export PYTHONPATH="$LKG_EXPERIMENT/experiment/src${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_EXTENSIONS_DIR="$GENERATED_ROOT/torch_extensions_lkg_rtgs/rtgs_coherent_external"

mkdir -p "$GENERATED_ROOT" "$TORCH_EXTENSIONS_DIR"
```

Quick CUDA/import check:

```bash
python - <<'PY'
import torch
print("cuda_available:", torch.cuda.is_available())
print("torch_cuda:", torch.version.cuda)
from gsplat.rendering_coherent_raster import rasterization_CR
print("rasterization_CR import: ok")
PY
```

## 5. Prepare The LKG View Mapping

If a Looking Glass display is connected, generate the calibration NPZ:

```bash
cd "$LKG_EXPERIMENT/experiment"

python -m lkg_experiment.coherent_default.build_lut_npz \
  --bridge-sdk-root "$BRIDGE_SDK_ROOT" \
  --display-index 0 \
  --output "$GENERATED_ROOT/lkg_go_1440x2560_66_views_lkg_calibration.npz"
```

If no display is connected, use `--map-mode linear` for non-LKG smoke tests.
For real LKG display runs, use the generated calibration file:

```bash
export VIEWPOINT_INDEX_PATH="$GENERATED_ROOT/lkg_go_1440x2560_66_views_lkg_calibration.npz"
```

## 6. Smoke Test: One RTGS + CR View

Run D-NeRF first:

```bash
cd "$LKG_EXPERIMENT/experiment"

python rtgs_cr_1view.py \
  --dataset-kind dnerf \
  --model-path "$MODEL_ROOT/$DNERF_SCENE" \
  --checkpoint "$CHECKPOINT" \
  --rtgs-code-root "$RTGS_CODE_ROOT" \
  --rtgs-code-policy as-is \
  --gsplat-root "$GSPLAT_ROOT" \
  --dataset-root "$DNERF_ROOT" \
  --n3dv-root "$N3DV_ROOT" \
  --split test \
  --camera-index 0 \
  --output-dir "$GENERATED_ROOT/smoke_rtgs_cr_1view" \
  --run-label "${DNERF_SCENE}_dnerf_1view"
```

Run N3DV next:

```bash
python rtgs_cr_1view.py \
  --dataset-kind n3dv \
  --model-path "$MODEL_ROOT/$N3DV_SCENE" \
  --checkpoint "$CHECKPOINT" \
  --rtgs-code-root "$RTGS_CODE_ROOT" \
  --rtgs-code-policy as-is \
  --gsplat-root "$GSPLAT_ROOT" \
  --dataset-root "$DNERF_ROOT" \
  --n3dv-root "$N3DV_ROOT" \
  --split test \
  --camera-index 0 \
  --n3dv-frame-index 0 \
  --output-dir "$GENERATED_ROOT/smoke_rtgs_cr_1view" \
  --run-label "${N3DV_SCENE}_n3dv_1view"
```

Success criteria:

- The command exits with code 0.
- The output directory contains official, CR, diff, and manifest artifacts.
- N3DV should not show the old FoV/aspect-ratio failure mode.

## 7. Smoke Test: One-Shot Interlaced Renderer

Start with a small resolution to compile CUDA extensions and validate the
one-shot path quickly:

```bash
cd "$LKG_EXPERIMENT/experiment"

python rtgs_cr_experiment.py \
  --dataset-kind dnerf \
  --model-path "$MODEL_ROOT/$DNERF_SCENE" \
  --checkpoint "$CHECKPOINT" \
  --rtgs-code-root "$RTGS_CODE_ROOT" \
  --rtgs-code-policy as-is \
  --gsplat-root "$GSPLAT_ROOT" \
  --dataset-root "$DNERF_ROOT" \
  --n3dv-root "$N3DV_ROOT" \
  --artifact-dir "$GENERATED_ROOT/rtgs_cr_experiments" \
  --run-id "smoke_${DNERF_SCENE}_one_shot" \
  --split test \
  --camera-index 0 \
  --width 360 \
  --height 640 \
  --views 6 \
  --clusters 2 \
  --warmup-iters 1 \
  --measure-iters 1 \
  --skip-metrics \
  --no-reference-interlaced \
  --no-without-reuse \
  --skip-web-assets \
  --map-mode linear \
  --camera-aspect-mode expand \
  --aspect-fit contain \
  --no-ssim
```

Then run the full 1440x2560 LKG-shaped test:

```bash
python rtgs_cr_experiment.py \
  --dataset-kind n3dv \
  --model-path "$MODEL_ROOT/$N3DV_SCENE" \
  --checkpoint "$CHECKPOINT" \
  --rtgs-code-root "$RTGS_CODE_ROOT" \
  --rtgs-code-policy as-is \
  --gsplat-root "$GSPLAT_ROOT" \
  --dataset-root "$DNERF_ROOT" \
  --n3dv-root "$N3DV_ROOT" \
  --artifact-dir "$GENERATED_ROOT/rtgs_cr_experiments" \
  --run-id "smoke_${N3DV_SCENE}_one_shot_1440x2560" \
  --split test \
  --camera-index 0 \
  --n3dv-frame-index 0 \
  --width 1440 \
  --height 2560 \
  --views 66 \
  --clusters 2 \
  --warmup-iters 1 \
  --measure-iters 1 \
  --max-metric-views 1 \
  --sample-save-views 5 \
  --skip-web-assets \
  --map-mode file \
  --viewpoint-index-path "$VIEWPOINT_INDEX_PATH" \
  --camera-aspect-mode expand \
  --aspect-fit contain \
  --no-ssim
```

Success criteria:

- `metrics.csv`, `metrics.json`, and `manifest.json` are written.
- `looking_glass_tensor.png` is written for the experiment variants.
- The timing columns include one-shot timing fields such as
  `frame_ms_with_lkg_interlace`, `dynamic_deform_ms`, `projection_ms`,
  `color_eval_ms`, `cr_projection_ms`, `cr_keygen_ms`, `cr_sort_ms`,
  `cr_blend_ms`, and `interlace_post_ms`.

## 8. Run Full Batch Experiments

Use the maintained shell wrapper for all configured scenes:

```bash
cd "$LKG_EXPERIMENT/experiment"

MODEL_ROOT="$MODEL_ROOT" \
RTGS_CODE_ROOT="$RTGS_CODE_ROOT" \
RTGS_CODE_POLICY="as-is" \
GSPLAT_ROOT="$GSPLAT_ROOT" \
DNERF_ROOT="$DNERF_ROOT" \
N3DV_ROOT="$N3DV_ROOT" \
GENERATED_ROOT="$GENERATED_ROOT" \
VIEWPOINT_INDEX_PATH="$VIEWPOINT_INDEX_PATH" \
TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
CAMERA_ASPECT_MODE="expand" \
ASPECT_FIT="contain" \
bash scripts/run_rtgs_cr_experiments_all.sh
```

For full test-set view coverage:

```bash
cd "$LKG_EXPERIMENT/experiment"

MODEL_ROOT="$MODEL_ROOT" \
RTGS_CODE_ROOT="$RTGS_CODE_ROOT" \
RTGS_CODE_POLICY="as-is" \
GSPLAT_ROOT="$GSPLAT_ROOT" \
DNERF_ROOT="$DNERF_ROOT" \
N3DV_ROOT="$N3DV_ROOT" \
GENERATED_ROOT="$GENERATED_ROOT" \
VIEWPOINT_INDEX_PATH="$VIEWPOINT_INDEX_PATH" \
TORCH_EXTENSIONS_DIR="$TORCH_EXTENSIONS_DIR" \
CAMERA_ASPECT_MODE="expand" \
ASPECT_FIT="contain" \
bash scripts/run_rtgs_cr_experiments_all_full_views.sh
```

## 9. Run The Real-Time LKG Web Viewer

Use this when the Looking Glass display is connected:

```bash
cd "$LKG_EXPERIMENT/experiment"

python rtgs_lkg_web_server.py \
  --checkpoint-root "$MODEL_ROOT" \
  --initial-scene "$DNERF_SCENE" \
  --initial-checkpoint "$CHECKPOINT" \
  --rtgs-code-root "$RTGS_CODE_ROOT" \
  --rtgs-code-policy as-is \
  --gsplat-root "$GSPLAT_ROOT" \
  --dataset-root "$DNERF_ROOT" \
  --n3dv-root "$N3DV_ROOT" \
  --viewpoint-index-path "$VIEWPOINT_INDEX_PATH" \
  --width 1440 \
  --height 2560 \
  --views 66 \
  --camera-aspect-mode expand \
  --aspect-fit contain \
  --cluster-size 2 \
  --display-mode glfw \
  --host 0.0.0.0 \
  --port 8088
```

Open the controller in a browser:

```text
http://<server-ip>:8088
```

Use `--display-mode none` when no LKG display is connected. The web page will
still show the latest interlaced JPEG and status/FPS information:

```bash
python rtgs_lkg_web_server.py \
  --checkpoint-root "$MODEL_ROOT" \
  --initial-scene "$DNERF_SCENE" \
  --initial-checkpoint "$CHECKPOINT" \
  --rtgs-code-root "$RTGS_CODE_ROOT" \
  --rtgs-code-policy as-is \
  --gsplat-root "$GSPLAT_ROOT" \
  --dataset-root "$DNERF_ROOT" \
  --n3dv-root "$N3DV_ROOT" \
  --width 1440 \
  --height 2560 \
  --views 66 \
  --camera-aspect-mode expand \
  --aspect-fit contain \
  --cluster-size 2 \
  --display-mode none \
  --host 0.0.0.0 \
  --port 8088
```

Success criteria:

- The web server starts and lists checkpoints under `$MODEL_ROOT`.
- The loading overlay appears while checkpoints are being switched.
- The rendered frame updates in the browser.
- `render_fps` measures model render input through interlaced tensor creation.
- `display_fps` measures the GLFW upload/display path when `--display-mode glfw`
  is used.

## Troubleshooting Checklist

- If CUDA extensions rebuild every run, keep `TORCH_EXTENSIONS_DIR` fixed and do
  not delete it between runs.
- If a CUDA extension build fails after changing patches, delete only the
  dedicated cache directory:

  ```bash
  rm -rf "$TORCH_EXTENSIONS_DIR"
  mkdir -p "$TORCH_EXTENSIONS_DIR"
  ```

- If N3DV looks cropped or vertically wrong, verify that the command includes
  `--camera-aspect-mode expand --aspect-fit contain`.
- If N3DV resembles the old broken FoV result, verify that the RTGS
  CoherentRaster patches were applied to `gsplat` and that the code imports the
  patched checkout:

  ```bash
  python - <<'PY'
  import gsplat, pathlib
  print(pathlib.Path(gsplat.__file__).resolve())
  PY
  ```

- If `apply_patches.sh --check` fails, inspect the target `gsplat` base. The
  RTGS patches assume a CoherentRaster-enabled `gsplat`; they are not intended
  for vanilla upstream `gsplat`.
- If the LKG display is unavailable, use `--map-mode linear` for offline smoke
  tests and `--display-mode none` for the web viewer.
