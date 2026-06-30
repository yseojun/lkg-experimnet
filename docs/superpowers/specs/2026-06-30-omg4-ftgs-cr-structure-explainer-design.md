# OMG4-FTGS + CoherentRaster Structure Explainer Design

## Goal

Build a static explanatory web page that helps a technical reader understand the full OMG4-FTGS to CoherentRaster rendering structure. The page should explain the original FTGS rendering path, where the CoherentRaster path was injected, how single-view CR differs from interlaced CR, and why the latest lookup/timing split was added.

The page is not a live rendering controller. It is a code-and-architecture explainer grounded in this repository's current files, diffs, tests, and expected artifacts.

## Audience

The primary reader is someone who can read Python/CUDA code but does not yet have the FTGS and CoherentRaster data flow in their head. The page should answer:

- What tensors does FTGS produce for a timestamp?
- What did the baseline `gsplat.rasterization()` path do with those tensors?
- Where did `rasterization_CR()` enter the path?
- Why are `view_idx_matrix`, `subpixel_coord_matrix`, and `translation_values` necessary for interlaced CR?
- Which recent code changes moved lookup construction out of the render hot path?
- Which timing number should be used when comparing CR core rendering versus lookup-inclusive frame cost?

## Recommended Approach

Use a checked-in static page under:

```text
docs/omg4-ftgs-cr-structure-explainer/index.html
```

If separate CSS or JavaScript is needed, keep it under:

```text
docs/omg4-ftgs-cr-structure-explainer/assets/
```

This matches the repository's existing static web pattern better than adding a Node, React, Vite, or live Python server dependency. The page should work without network access and without generated runtime artifacts. If local generated images or manifests exist later, they can be linked as optional evidence, but the initial version should be complete from code and text alone.

## Information Architecture

The page should be organized as a left-navigation technical review page with stable anchors:

1. `Overview`
2. `FTGS Materialization`
3. `Baseline Gsplat Rendering`
4. `CoherentRaster Injection`
5. `Single-View vs Interlaced CR`
6. `CUDA Contract`
7. `Recent Lookup/Timing Diff`
8. `Timing Interpretation`
9. `Verification`
10. `Final Takeaway`

The first screen should show the whole pipeline before any detailed diff:

```text
OMG4 checkpoint
  -> timestamp materialization
  -> FtgsSplats(means, quats, scales, opacities, colors)
  -> baseline gsplat single render
  -> CR single-view validation
  -> CR interlaced LKG panel render
  -> manifest/timing evidence
```

The first screen should also state three facts plainly:

- The baseline gsplat path is preserved.
- CoherentRaster is an additional rendering path after FTGS materialization.
- The latest change separates lookup setup cost from CR core render cost.

## Content Model

Each major section should follow the same pattern:

```text
Concept
  Short explanation of what the stage does.

Code Evidence
  File references and compact snippets or summarized diff hunks.

Why It Matters
  The reason this stage exists and what would be misleading without it.

What To Inspect
  Output files, manifest fields, or tests that prove the behavior.
```

This structure keeps the page from becoming either a prose essay or a raw diff dump.

## Key Sections

### Overview

Explain that OMG4-FTGS first decodes and materializes dynamic Gaussians for a timestamp. CR does not replace that model logic. CR consumes the materialized splats as an alternate rasterization backend.

The overview should include a small before/after table:

```text
Before:
  materialize FTGS splats -> gsplat rasterization

After:
  materialize FTGS splats -> gsplat baseline OR CoherentRaster path

Latest timing change:
  CR lookup setup is prepared before the render timer and reported separately.
```

### FTGS Materialization

Explain the FTGS-specific model stage:

- Load compressed checkpoint.
- Decode means, times, scales, rotations, durations, velocities, appearance codebooks, and MLP weights.
- For a timestamp, compute `means_t`.
- Apply temporal opacity.
- Run MLPs to produce SH/DC color coefficients and opacity.
- Package the result as `FtgsSplats`.

Primary code references:

- `experiment/src/lkg_experiment/omg4_ftgs/model.py`
- `OMG4/OMG4_FTGS/render.py`

### Baseline Gsplat Rendering

Explain that the baseline path takes the materialized `FtgsSplats` and calls `gsplat.rasterization()` with one view matrix and intrinsics matrix.

Primary code references:

- `experiment/src/lkg_experiment/omg4_ftgs/render.py::render_splats_gsplat`
- `gsplat/gsplat/rendering.py::rasterization`

This section should emphasize that baseline output is still useful as the comparison reference.

### CoherentRaster Injection

Explain that CR is injected after FTGS materialization, not inside FTGS checkpoint decoding or MLP evaluation.

For single-view validation, the page should show:

- The same materialized splats are passed to `rasterization_CR()`.
- `adjacent_viewmats` is shaped as a single group containing one view.
- The viewpoint map is all zero, so every subpixel resolves to the same view.
- The output is unpatchified and unpadded back into an RGB image.

Primary code references:

- `experiment/src/lkg_experiment/omg4_ftgs/render.py::render_splats_coherent`
- `gsplat/gsplat/rendering_coherent_raster.py::rasterization_CR`

### Single-View vs Interlaced CR

Explain the difference between validation and real LKG panel rendering:

- Single-view CR uses one view and an all-zero map.
- Interlaced CR uses a panel-sized viewpoint index map and synthesized adjacent view matrices.
- The panel map tells the CUDA kernel which view each RGB subpixel should display.

Primary code references:

- `experiment/src/lkg_experiment/omg4_ftgs/render.py::build_interlaced_viewpoint_index`
- `experiment/src/lkg_experiment/omg4_ftgs/render.py::synthesize_interlaced_viewmats`
- `experiment/src/lkg_experiment/omg4_ftgs/render.py::render_splats_interlaced_coherent`
- `experiment/src/lkg_experiment/coherent_default/coherent_gsplat_bridge.py::build_cr_lookup_arrays`

### CUDA Contract

Explain the CR CUDA inputs at a practical level:

- `view_idx_matrix`: for each tile/subpixel thread, which view number to render.
- `subpixel_coord_matrix`: where that thread's result belongs in the original panel coordinate system.
- `translation_values`: per-Gaussian, per-view translation from reference view projection to adjacent view projection.
- `isect_offsets` and `flatten_ids`: tile/list ranges used for alpha blending.

The reader should understand that CR avoids rendering all full view images and then interlacing them. It renders the interlaced target directly at subpixel level.

Primary code references:

- `rtgs-coherent-raster/modified-files/gsplat/cuda/csrc/CoherentRaster_CRKernel.cu`
- `rtgs-coherent-raster/modified-files/gsplat/cuda/csrc/CoherentRaster.cpp`
- `rtgs-coherent-raster/modified-files/gsplat/cuda/_wrapper_coherent_raster.py`

### Recent Lookup/Timing Diff

Explain the current working-tree change as a focused optimization and measurement correction:

Before:

```text
render_splats_interlaced_coherent()
  -> build_cr_lookup_arrays()
  -> lookup_arrays_to_torch()
  -> rasterization_CR()
  -> unpatchify/unpad
```

After:

```text
CLI interlaced path
  -> prepare_cr_lookup_tensors()
      -> build_cr_lookup_arrays()
      -> lookup_arrays_to_torch()
  -> start render timer
  -> render_splats_interlaced_coherent(prebuilt lookup tensors)
      -> rasterization_CR()
      -> unpatchify/unpad
```

Primary code references:

- `experiment/src/lkg_experiment/omg4_ftgs/render.py::PreparedCrLookup`
- `experiment/src/lkg_experiment/omg4_ftgs/render.py::prepare_cr_lookup_tensors`
- `experiment/src/lkg_experiment/omg4_ftgs/render.py::render_splats_interlaced_coherent`
- `experiment/src/lkg_experiment/omg4_ftgs/cli.py::_render_interlaced`
- `experiment/tests/test_omg4_ftgs.py`

### Timing Interpretation

The page should present the motivating timing breakdown:

```text
Previous manifest render_ms: 1631.56 ms
Measured lookup CPU:        1110.61 ms
Measured lookup H2D:          39.23 ms
Measured rasterization_CR:    42.02 ms
Measured postprocess:          0.33 ms
```

Then explain:

- `frame_ms_including_lookup` is the full one-shot frame setup cost.
- `frame_ms_excluding_lookup` is the render call after lookup is already prepared.
- `cr_core_ms` is the sum of CR projection/keygen/sort/blend timing when available.
- Lookup-inclusive timing is useful for first-frame cost.
- Lookup-exclusive timing is more useful for steady-state video or interactive rendering where the panel lookup can be reused.

### Verification

The page should list the expected tests and evidence:

- Parser and dispatch tests for `--renderer`.
- Mocked CR single-view call test.
- Mocked interlaced CR call test.
- Prebuilt lookup tensor pass-through test.
- CLI manifest timing test.
- GPU smoke command for single-view `--renderer both`.
- GPU smoke command for interlaced output.

The page should also list manifest fields to inspect:

- `lookup_cpu_ms`
- `lookup_h2d_ms`
- `frame_ms_excluding_lookup`
- `frame_ms_including_lookup`
- `cr_core_ms`
- `cr_timing_ms`
- `post_ms`
- `gaussians`
- `views`
- `cluster_size`

## UI Design

The UI should be dense and code-forward, not marketing-style.

Layout:

- Fixed or sticky left navigation on desktop.
- Single-column content on mobile.
- A compact top summary panel.
- Section blocks with plain headings, short paragraphs, and code evidence panels.
- Diff hunks grouped by intent rather than raw file order.
- Tables for before/after and timing interpretation.

Visual style:

- Use a restrained light theme.
- Avoid decorative gradients, hero artwork, and oversized typography.
- Use monospace blocks for code paths, function names, manifest fields, and commands.
- Use small colored labels for categories such as `FTGS`, `gsplat`, `CR`, `lookup`, `timing`, and `test`.

Interactive behavior should be minimal:

- Anchor links should jump to sections.
- If JavaScript is included, it can support section filtering or copyable code snippets.
- The initial version should not require JavaScript to understand the content.

## Data And Assets

The page should be complete without generated artifacts. Optional artifacts can be referenced as examples, but they should not be required for tests to pass.

Expected static assets:

- `index.html`
- Optional `assets/style.css`
- Optional `assets/app.js`

No remote CSS, JavaScript, fonts, or image dependencies are allowed.

## Error Handling

Because the page is static, error handling is mostly about avoiding broken assumptions:

- If optional local artifacts are referenced, label them as optional and do not require them to exist.
- Asset links in the committed page must resolve locally.
- The page should not depend on `/data/...` paths at runtime.
- Any shell commands should be examples, not automatically executed.

## Testing

Use TDD before implementation. Add a focused test file:

```text
experiment/tests/test_explanatory_web_page.py
```

The tests should assert:

- `docs/omg4-ftgs-cr-structure-explainer/index.html` exists.
- Required anchors exist:
  - `overview`
  - `ftgs-materialization`
  - `baseline-gsplat-rendering`
  - `coherentraster-injection`
  - `single-view-vs-interlaced-cr`
  - `cuda-contract`
  - `recent-lookup-timing-diff`
  - `timing-interpretation`
  - `verification`
  - `final-takeaway`
- Required phrases appear:
  - `FtgsSplats`
  - `rasterization_CR`
  - `view_idx_matrix`
  - `subpixel_coord_matrix`
  - `translation_values`
  - `frame_ms_excluding_lookup`
  - `frame_ms_including_lookup`
- Local `href` and `src` asset references resolve on disk.
- No remote `http://` or `https://` assets are required.

Verification commands:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_explanatory_web_page -v
env PYTHONPATH=src python -m unittest tests.test_package_layout tests.test_lkg_experiment_defaults -v
```

Manual static preview command:

```bash
cd /home/ysj/lkg-experiment
python -m http.server 8000 -d docs/omg4-ftgs-cr-structure-explainer
```

Open:

```text
http://127.0.0.1:8000/index.html
```

## Scope Boundaries

In scope:

- A static explainer page.
- Local tests for page structure and asset integrity.
- Code references and short snippets that explain architecture and diff intent.
- Timing explanation for lookup-inclusive and lookup-exclusive metrics.

Out of scope:

- A live renderer.
- A new React/Vite/Next app.
- Automatic parsing of arbitrary git diffs.
- Automatic loading of `/data/...` experiment outputs.
- Modifying FTGS, gsplat, or CUDA behavior.

## Final Takeaway

The page should leave the reader with one coherent mental model:

FTGS creates timestamp-specific Gaussian splats. The baseline path renders those splats with normal gsplat. The CoherentRaster path is injected after splat materialization and uses extra lookup matrices to render a light-field interlaced panel directly at subpixel level. The latest lookup/timing change does not alter the CR math; it moves reusable panel lookup setup out of the render hot path and reports timing in a way that separates first-frame setup cost from CR core rendering.
