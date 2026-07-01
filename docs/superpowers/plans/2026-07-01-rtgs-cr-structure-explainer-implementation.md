# RTGS CR Structure Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the static Korean RTGS + CoherentRaster explainer page specified in `docs/superpowers/specs/2026-06-30-rtgs-cr-structure-explainer-design.md`.

**Architecture:** Add a standalone static documentation page under `docs/rtgs-cr-structure-explainer/` with local CSS and no runtime dependencies. Add a focused stdlib `unittest` page parser test that validates required Korean structure, anchors, terms, and local-only assets. Keep the existing OMG4 explainer test untouched.

**Tech Stack:** Static HTML, CSS, Python stdlib `unittest`/`html.parser`.

---

## File Structure

- Create: `experiment/tests/test_rtgs_explanatory_web_page.py`
  - Tests the RTGS explainer page structure and asset locality.
- Create: `docs/rtgs-cr-structure-explainer/index.html`
  - Korean static technical explainer with left navigation and code-grounded sections.
- Create: `docs/rtgs-cr-structure-explainer/assets/styles.css`
  - Local responsive styling adapted from the OMG4 explainer.

## Task 1: Add Failing Page Contract Test

**Files:**
- Create: `experiment/tests/test_rtgs_explanatory_web_page.py`

- [ ] **Step 1: Write the failing test**

Create `experiment/tests/test_rtgs_explanatory_web_page.py` with a parser test that requires the new RTGS page, all planned anchors, key terms, and local assets:

```python
from __future__ import annotations

import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE_ROOT = REPO_ROOT / "docs" / "rtgs-cr-structure-explainer"
INDEX_PATH = PAGE_ROOT / "index.html"


class _PageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.assets: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        data = {name: value for name, value in attrs if value is not None}
        if "id" in data:
            self.ids.add(data["id"])
        if tag == "link" and "href" in data:
            self.assets.append(data["href"])
        if tag in {"script", "img"} and "src" in data:
            self.assets.append(data["src"])


class RtgsCrExplainerPageTest(unittest.TestCase):
    def _parse_page(self) -> tuple[str, _PageParser]:
        self.assertTrue(INDEX_PATH.is_file(), f"missing explainer page: {INDEX_PATH}")
        html = INDEX_PATH.read_text(encoding="utf-8")
        parser = _PageParser()
        parser.feed(html)
        return html, parser

    def test_page_has_required_korean_structure(self) -> None:
        html, parser = self._parse_page()
        required_ids = {
            "overview",
            "3dgs-basics",
            "rtgs-dynamics",
            "camera-contract",
            "official-flow",
            "snapshot-materialization",
            "cr-1view-ladder",
            "lkg-lookup",
            "cr-66-compose",
            "one-shot-cr",
            "cuda-contract",
            "timing-and-metrics",
            "caveats",
            "takeaway",
        }
        self.assertTrue(required_ids.issubset(parser.ids), required_ids - parser.ids)
        required_phrases = [
            'lang="ko"',
            "RTGS",
            "CoherentRaster",
            "timestamp",
            "materialize",
            "marginal_t",
            "viewpoint_index",
            "view_idx_matrix",
            "subpixel_coord_matrix",
            "rtgs_projection_adapter",
            "rasterization_CR",
            "frame_ms_without_lkg",
            "frame_ms_with_lkg_interlace",
            "SnapshotGaussianProxy",
            "FoVx=FoVy=-1.0",
            "최종 요약",
        ]
        for phrase in required_phrases:
            self.assertIn(phrase, html)

    def test_page_uses_only_local_assets(self) -> None:
        _, parser = self._parse_page()
        self.assertGreater(parser.assets, [], "expected at least one local stylesheet asset")
        for asset in parser.assets:
            parsed = urlparse(asset)
            self.assertEqual(parsed.scheme, "", asset)
            self.assertFalse(asset.startswith("//"), asset)
            if asset.startswith("#"):
                continue
            asset_path = asset.split("#", 1)[0].split("?", 1)[0]
            resolved = (PAGE_ROOT / asset_path).resolve()
            self.assertTrue(str(resolved).startswith(str(PAGE_ROOT.resolve())), asset)
            self.assertTrue(resolved.is_file(), asset)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify RED**

Run:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_rtgs_explanatory_web_page -v
```

Expected: FAIL because `docs/rtgs-cr-structure-explainer/index.html` does not exist yet.

## Task 2: Implement Static HTML/CSS Page

**Files:**
- Create: `docs/rtgs-cr-structure-explainer/index.html`
- Create: `docs/rtgs-cr-structure-explainer/assets/styles.css`

- [ ] **Step 1: Create the HTML page**

Implement `index.html` as a Korean static explainer with:

- `lang="ko"`;
- local stylesheet `assets/styles.css`;
- left TOC with all required anchors;
- first-screen summary grid for official RTGS, snapshot/CR, and LKG/one-shot;
- sections matching the test ids and the approved design spec;
- compact code evidence blocks referencing:
  - `official_1view.py`;
  - `cli.py::materialize_rtgs_geometry`;
  - `cli.py::evaluate_rtgs_colors`;
  - `cr_1view.py::render_rtgs_cr_1view`;
  - `coherent_gsplat_bridge.py::build_cr_lookup_arrays`;
  - `cr_66views.py::accumulate_interlaced_view`;
  - `cr_one_shot.py::render_rtgs_cr_one_shot_viewport_image`;
  - `rendering_coherent_raster.py::rasterization_CR`;
  - `CoherentRaster_CRKernel.cu`.

- [ ] **Step 2: Create the CSS**

Implement `assets/styles.css` with:

- responsive two-column shell: sticky TOC + main content;
- no external fonts or assets;
- readable code blocks and tables;
- warning cards and pipeline boxes;
- mobile breakpoint under 980px and 620px;
- stable dimensions for cards/tables/code blocks.

- [ ] **Step 3: Run the focused test to verify GREEN**

Run:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_rtgs_explanatory_web_page -v
```

Expected: PASS.

## Task 3: Run Related Verification And Review

**Files:**
- Verify: `experiment/tests/test_rtgs_explanatory_web_page.py`
- Verify: `experiment/tests/test_explanatory_web_page.py`
- Verify: `docs/rtgs-cr-structure-explainer/index.html`
- Verify: `docs/rtgs-cr-structure-explainer/assets/styles.css`

- [ ] **Step 1: Run focused and related explainer tests**

Run:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_rtgs_explanatory_web_page tests.test_explanatory_web_page -v
```

Expected: PASS for both page tests.

- [ ] **Step 2: Run whitespace/static diff checks**

Run:

```bash
cd /home/ysj/lkg-experiment
git diff --check -- docs/rtgs-cr-structure-explainer experiment/tests/test_rtgs_explanatory_web_page.py docs/superpowers/plans/2026-07-01-rtgs-cr-structure-explainer-implementation.md
```

Expected: no output and exit 0.

- [ ] **Step 3: Review scoped git diff**

Run:

```bash
cd /home/ysj/lkg-experiment
git diff -- docs/rtgs-cr-structure-explainer experiment/tests/test_rtgs_explanatory_web_page.py docs/superpowers/plans/2026-07-01-rtgs-cr-structure-explainer-implementation.md
```

Expected: diff only contains the RTGS explainer implementation, its test, and this plan.

## Task 4: Commit Scoped Implementation

**Files:**
- Add: `docs/rtgs-cr-structure-explainer/index.html`
- Add: `docs/rtgs-cr-structure-explainer/assets/styles.css`
- Add: `experiment/tests/test_rtgs_explanatory_web_page.py`
- Add: `docs/superpowers/plans/2026-07-01-rtgs-cr-structure-explainer-implementation.md`

- [ ] **Step 1: Stage only scoped files**

Run:

```bash
cd /home/ysj/lkg-experiment
git add docs/rtgs-cr-structure-explainer/index.html docs/rtgs-cr-structure-explainer/assets/styles.css experiment/tests/test_rtgs_explanatory_web_page.py docs/superpowers/plans/2026-07-01-rtgs-cr-structure-explainer-implementation.md
```

Expected: staged file list contains only these four files.

- [ ] **Step 2: Commit**

Run:

```bash
cd /home/ysj/lkg-experiment
git commit -m "Add RTGS CR structure explainer"
```

Expected: commit succeeds with only the scoped implementation files.

## Self-Review

- Spec coverage: all approved sections from `2026-06-30-rtgs-cr-structure-explainer-design.md` map to HTML sections and required test anchors.
- Placeholder scan: this plan contains no TBD/TODO placeholders.
- Type/path consistency: all referenced new files use exact repository paths and the test points to `docs/rtgs-cr-structure-explainer/index.html`.
