# LKG Experiment Workspace

This repository keeps the experiment package separate from the rendering
dependencies. The dependency repositories live next to `experiment/` in the
workspace root and are ignored by Git.

## Layout

```text
<workspace>/
├── Bridge-Python-SDK-Lab/   # external Bridge SDK checkout
├── coherent-raster/         # external CoherentRaster checkout
├── gsplat/                  # external gsplat checkout with CoherentRaster patch
└── experiment/              # tracked experiment package
```

The workspace directory name does not matter. The Python entrypoints resolve
dependency defaults from their own location.

## Clone

```bash
git clone https://github.com/yseojun/lkg-experimnet.git lkg-experiment
cd lkg-experiment
```

## External Dependencies

Clone or copy these sibling checkouts into the workspace root:

```bash
git clone https://github.com/daikiad/Bridge-Python-SDK-Lab.git Bridge-Python-SDK-Lab

git clone https://github.com/sgj0402/coherent-raster.git coherent-raster

git clone --recursive https://github.com/nerfstudio-project/gsplat.git gsplat
git -C gsplat checkout a8d88d387f6e554b18153d309f5536696882de5c
git -C gsplat switch -c cr-server
```

Apply the CoherentRaster patch from `coherent-raster` into sibling `gsplat`:

```bash
cd coherent-raster
python install.py
```

Then install `gsplat` editable into the conda environment used for experiments:

```bash
cd ../gsplat
conda run -n coherent_raster python -m pip install --no-build-isolation -e .
```

If the machine uses CUDA 12.2 with a default compiler newer than GCC 12, build
with an older host compiler and undefine CUDA half/bfloat16 guard macros:

```bash
CC=/usr/bin/gcc-11 \
CXX=/usr/bin/g++-11 \
MAX_JOBS=4 \
TORCH_CUDA_ARCH_LIST=8.9 \
NVCC_FLAGS="-U__CUDA_NO_HALF_CONVERSIONS__ -U__CUDA_NO_BFLOAT16_CONVERSIONS__ -U__CUDA_NO_HALF_OPERATORS__ -U__CUDA_NO_HALF2_OPERATORS__" \
conda run -n coherent_raster python -m pip install --no-build-isolation -e .
```

The current prepared dependency states are:

| Directory | Remote | Branch | Current commit |
| --- | --- | --- | --- |
| `Bridge-Python-SDK-Lab` | `https://github.com/daikiad/Bridge-Python-SDK-Lab.git` | `main` | `0d080ed` |
| `coherent-raster` | `https://github.com/sgj0402/coherent-raster.git` | `main` | `181b989` |
| `gsplat` | `https://github.com/nerfstudio-project/gsplat.git` | `cr-server` | `eec8ab6` |

## Data And Results

Set these environment variables for each server:

```bash
export DATADIR=/path/to/datasets
export RESULTDIR=/path/to/results
```

Expected dataset/result layout:

```text
$DATADIR/
├── nerf_synthetic/
│   └── drums/
└── MipNeRF_360/
    └── garden/

$RESULTDIR/
├── blender_MCMC100000_init50000/
│   └── drums/ckpts/ckpt_29999_rank0.pt
├── blender_MCMC500000/
└── MipNeRF360_MCMC500000/
```

If `DATADIR` or `RESULTDIR` is omitted, the code tries common local defaults
such as `/data/ysj/dataset`, `/data/ysj/result/coherent-raster`,
`~/Data/datasets`, `~/Data/results`, `~/data/dataset`,
`~/data/result`, `/data/dataset`, and `/data/result`.

## Run The Drums Experiment

```bash
cd experiment
DATADIR=/path/to/datasets \
RESULTDIR=/path/to/results \
PYTHON_BIN="$(command -v python)" \
./scripts/run_drums_66_views.sh --run-id drums_smoke
```

You can also call the Python entrypoint directly:

```bash
cd experiment
DATADIR=/path/to/datasets \
RESULTDIR=/path/to/results \
python run_coherent_raster_experiment.py --run-id drums_smoke
```

Experiment outputs are generated under
`/data/ysj/result/generated/coherent_raster_experiments/` by default. Override
`LKG_RESULT_BASE`, `LKG_GENERATED_DIR`, or `ARTIFACT_DIR` when a server needs a
different storage root.
