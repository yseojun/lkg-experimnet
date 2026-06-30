# RTGS CoherentRaster Gsplat Snapshot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the RTGS-specific `gsplat` CoherentRaster modifications inside the main `lkg-experiment` repository so they can be inspected and reapplied in another environment.

**Architecture:** Store the changes as both ordered git patches and a final modified-file snapshot. The patches are the canonical replay path; the snapshot is a readable reference for reviewing the final file contents without checking out the external `gsplat` repo.

**Tech Stack:** Git format-patch, Python standard library verification, Markdown documentation.

---

### Task 1: Create Patch Archive

**Files:**
- Create: `rtgs-coherent-raster/patches/0001-Add-RTGS-projection-adapter-to-coherent-raster.patch`
- Create: `rtgs-coherent-raster/patches/0002-Add-coherent-raster-timing-metrics.patch`

- [ ] **Step 1: Generate patches from the external gsplat repo**

Run:

```bash
mkdir -p rtgs-coherent-raster/patches
git -C gsplat format-patch -2 -o ../rtgs-coherent-raster/patches
```

Expected: two patch files are created in commit order.

- [ ] **Step 2: Verify patch order**

Run:

```bash
ls rtgs-coherent-raster/patches
```

Expected:

```text
0001-Add-RTGS-projection-adapter-to-coherent-raster.patch
0002-Add-coherent-raster-timing-metrics.patch
```

### Task 2: Create Modified File Snapshot

**Files:**
- Create under: `rtgs-coherent-raster/modified-files/gsplat/`

- [ ] **Step 1: Copy the current modified files**

Run:

```bash
python - <<'PY'
from pathlib import Path
import shutil

files = [
    "gsplat/cuda/_wrapper_coherent_raster.py",
    "gsplat/cuda/csrc/CoherentRaster.cpp",
    "gsplat/cuda/csrc/CoherentRaster.h",
    "gsplat/cuda/csrc/CoherentRaster_CR.cuh",
    "gsplat/cuda/csrc/CoherentRaster_CRKernel.cu",
    "gsplat/cuda/csrc/CoherentRaster_Modified.cuh",
    "gsplat/cuda/csrc/CoherentRaster_ModifiedKernel.cu",
    "gsplat/cuda/csrc/CoherentRaster_ModifiedWrapper.cu",
    "gsplat/cuda/include/Ops_CoherentRaster.h",
    "gsplat/rendering_coherent_raster.py",
]
root = Path("gsplat")
out = Path("rtgs-coherent-raster/modified-files")
for rel in files:
    src = root / rel
    dst = out / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
PY
```

Expected: every file listed above exists under `rtgs-coherent-raster/modified-files/`.

### Task 3: Write Reapplication Documentation

**Files:**
- Create: `rtgs-coherent-raster/README.md`
- Create: `rtgs-coherent-raster/manifest.json`

- [ ] **Step 1: Document purpose and scope**

The README must state that these patches affect CoherentRaster-specific `gsplat` code, not the normal `gsplat.rendering.rasterization` path.

- [ ] **Step 2: Document apply commands**

The README must include commands for:

```bash
git -C /path/to/gsplat am /path/to/lkg-experiment/rtgs-coherent-raster/patches/*.patch
```

and the helper script:

```bash
rtgs-coherent-raster/apply_patches.sh --check /path/to/gsplat
rtgs-coherent-raster/apply_patches.sh /path/to/gsplat
```

- [ ] **Step 3: Document rebuild and verification**

The README must include deleting or changing `TORCH_EXTENSIONS_DIR`, then running RTGS + CR tests or smoke renders so CUDA extensions rebuild against the patched sources.

### Task 4: Verify Archive

**Files:**
- Read: `rtgs-coherent-raster/patches/*.patch`
- Read: `rtgs-coherent-raster/modified-files/**`

- [ ] **Step 1: Verify files are present**

Run:

```bash
python - <<'PY'
from pathlib import Path
required = [
    "rtgs-coherent-raster/README.md",
    "rtgs-coherent-raster/manifest.json",
    "rtgs-coherent-raster/patches/0001-Add-RTGS-projection-adapter-to-coherent-raster.patch",
    "rtgs-coherent-raster/patches/0002-Add-coherent-raster-timing-metrics.patch",
]
missing = [path for path in required if not Path(path).is_file()]
if missing:
    raise SystemExit("missing: " + ", ".join(missing))
PY
```

Expected: exit code 0.

- [ ] **Step 2: Verify snapshot matches external gsplat**

Run a Python file comparison over all files listed in `manifest.json`.

Expected: every copied snapshot file is byte-identical to the corresponding current file in `gsplat/`.
