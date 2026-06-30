# LKG 9:16 Camera Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicit LKG camera policy so RTGS + CoherentRaster experiments render with a 1440x2560 9:16 camera frustum instead of placing a landscape camera inside the panel.

**Architecture:** Keep `aspect_fit` as the viewport/layout policy, and add `camera_aspect_mode` as the camera/intrinsics policy. `preserve` keeps the current behavior; `expand` forces full-panel rendering with `crop_to_fill=False`, preserving the source camera scale on the limiting axis and expanding the other axis FoV.

**Tech Stack:** Python argparse/dataclasses, existing RTGS camera helpers, unittest, shell runner defaults.

---

### Task 1: Add Camera Policy Tests

**Files:**
- Modify: `experiment/tests/test_rtgs_cr_66views.py`
- Modify: `experiment/tests/test_rtgs_cr_experiment.py`

- [ ] **Step 1: Add failing tests**

Add tests that assert:
- parser default `camera_aspect_mode == "expand"`;
- N3DV `expand` viewport uses full panel `1440x2560`, offset `0,0`, and `crop_to_fill=False`;
- `preserve` keeps current `contain` behavior;
- batch script exports `CAMERA_ASPECT_MODE="${CAMERA_ASPECT_MODE:-expand}"` and passes `--camera-aspect-mode`.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
cd /home/ysj/lkg-experiment/experiment
conda run -n rtgs-coherent-cu121 bash -lc 'PYTHONPATH=src python -m unittest tests.test_rtgs_cr_66views tests.test_rtgs_cr_experiment -v'
```

Expected: fail because `camera_aspect_mode` does not exist.

### Task 2: Implement Camera Policy

**Files:**
- Modify: `experiment/src/lkg_experiment/rtgs_coherent/cr_66views.py`
- Modify: `experiment/scripts/run_rtgs_cr_experiments_all.sh`

- [ ] **Step 1: Add argparse option**

Add `--camera-aspect-mode` with choices `preserve` and `expand`, default `expand`.

- [ ] **Step 2: Add viewport resolver helper**

Add `resolve_lkg_camera_viewport(...)`:
- if `camera_aspect_mode == "expand"`, return full target panel viewport with `aspect_fit="fit"`;
- if `camera_aspect_mode == "preserve"`, delegate to `resolve_aspect_viewport(..., aspect_fit=args.aspect_fit)`.

- [ ] **Step 3: Use helper in both context paths**

Replace direct calls to `resolve_aspect_viewport` in `prepare_rtgs_cr_66_context()` and `render_rtgs_cr_66views()` with the helper.

- [ ] **Step 4: Update script defaults**

Set `CAMERA_ASPECT_MODE="${CAMERA_ASPECT_MODE:-expand}"` and pass `--camera-aspect-mode "$CAMERA_ASPECT_MODE"`.

### Task 3: Verify

**Files:**
- No new production files.

- [ ] **Step 1: Run unit tests**

```bash
cd /home/ysj/lkg-experiment/experiment
conda run -n rtgs-coherent-cu121 bash -lc 'PYTHONPATH=src python -m unittest tests.test_rtgs_cr_66views tests.test_rtgs_cr_experiment tests.test_rtgs_cr_one_shot -v'
```

- [ ] **Step 2: Run syntax checks**

```bash
cd /home/ysj/lkg-experiment
python -m py_compile experiment/src/lkg_experiment/rtgs_coherent/cr_66views.py
cd /home/ysj/lkg-experiment/experiment
bash -n scripts/run_rtgs_cr_experiments_all.sh
```

- [ ] **Step 3: Run a small N3DV smoke**

Run a small `coffee_martini` one-shot render and verify manifest:

```text
content_viewport.render_width = requested width
content_viewport.render_height = requested height
content_viewport.offset_x = 0
content_viewport.offset_y = 0
```
