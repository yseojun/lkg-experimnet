# OMG4 FTGS CR Structure Explainer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a checked-in Korean static web page that explains how OMG4-FTGS rendering flows into the baseline gsplat path and where CoherentRaster is injected.

**Architecture:** The page is a local static document under `docs/omg4-ftgs-cr-structure-explainer/`, with one HTML file and one CSS file. A focused unittest validates the required anchors, Korean technical explanation markers, and local-only asset references.

**Tech Stack:** HTML, CSS, Python `unittest`, standard-library `html.parser`.

---

## File Structure

- Create: `experiment/tests/test_explanatory_web_page.py`
  - Validates the explainer exists, is Korean, includes the required FTGS/CR anchors and phrases, and does not depend on remote assets.
- Create: `docs/omg4-ftgs-cr-structure-explainer/index.html`
  - Korean explanatory page with pipeline overview, code-diff panels, CUDA contract notes, timing interpretation, and verification checklist.
- Create: `docs/omg4-ftgs-cr-structure-explainer/assets/styles.css`
  - Local responsive layout and code-forward visual style.

## Task 1: Add Failing Page Contract Test

**Files:**
- Create: `experiment/tests/test_explanatory_web_page.py`

- [ ] **Step 1: Write the failing test**

```python
from __future__ import annotations

import unittest
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE_ROOT = REPO_ROOT / "docs" / "omg4-ftgs-cr-structure-explainer"
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


class Omg4FtgsCrExplainerPageTest(unittest.TestCase):
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
            "ftgs-materialization",
            "baseline-gsplat-rendering",
            "coherentraster-injection",
            "single-view-vs-interlaced-cr",
            "cuda-contract",
            "recent-lookup-timing-diff",
            "timing-interpretation",
            "verification",
            "final-takeaway",
        }
        self.assertTrue(required_ids.issubset(parser.ids), required_ids - parser.ids)
        required_phrases = [
            'lang="ko"',
            "FTGS",
            "CoherentRaster",
            "FtgsSplats",
            "rasterization_CR",
            "view_idx_matrix",
            "subpixel_coord_matrix",
            "translation_values",
            "frame_ms_excluding_lookup",
            "frame_ms_including_lookup",
            "왜 수정했나",
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

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_explanatory_web_page -v
```

Expected: FAIL because `docs/omg4-ftgs-cr-structure-explainer/index.html` does not exist yet.

## Task 2: Implement Static Korean Explainer

**Files:**
- Create: `docs/omg4-ftgs-cr-structure-explainer/index.html`
- Create: `docs/omg4-ftgs-cr-structure-explainer/assets/styles.css`
- Test: `experiment/tests/test_explanatory_web_page.py`

- [ ] **Step 1: Add the HTML page**

The page must include `lang="ko"`, a local stylesheet link to `assets/styles.css`, and sections with these ids:

```text
overview
ftgs-materialization
baseline-gsplat-rendering
coherentraster-injection
single-view-vs-interlaced-cr
cuda-contract
recent-lookup-timing-diff
timing-interpretation
verification
final-takeaway
```

The Korean content must explain that FTGS materializes timestamp-specific `FtgsSplats`, the baseline path calls `gsplat.rasterization`, CoherentRaster is injected after materialization through `rasterization_CR`, interlaced CR uses `view_idx_matrix`, `subpixel_coord_matrix`, and `translation_values`, and the latest lookup diff separates `frame_ms_excluding_lookup` from `frame_ms_including_lookup`.

- [ ] **Step 2: Add the CSS page style**

The CSS must implement a dense technical review layout: sticky navigation on desktop, one-column mobile layout, local-only styling, readable code blocks, tables, and no remote font/image dependency.

- [ ] **Step 3: Run focused test to verify it passes**

Run:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_explanatory_web_page -v
```

Expected: PASS.

## Task 3: Verify And Preview

**Files:**
- Read: `docs/omg4-ftgs-cr-structure-explainer/index.html`
- Read: `docs/omg4-ftgs-cr-structure-explainer/assets/styles.css`

- [ ] **Step 1: Run baseline smoke tests**

Run:

```bash
cd /home/ysj/lkg-experiment/experiment
env PYTHONPATH=src python -m unittest tests.test_package_layout tests.test_lkg_experiment_defaults -v
```

Expected: PASS.

- [ ] **Step 2: Start a static preview server**

Run:

```bash
cd /home/ysj/lkg-experiment
python -m http.server 8000 -d docs/omg4-ftgs-cr-structure-explainer
```

Expected: server listens at `http://127.0.0.1:8000/index.html`. If port 8000 is occupied, use 8001.

- [ ] **Step 3: Verify the served page responds**

Run:

```bash
curl -I http://127.0.0.1:8000/index.html
```

Expected: HTTP 200.

## Self-Review

- Spec coverage: the tasks cover the static page location, required anchors, Korean explanation, code-diff content, timing interpretation, local assets, and focused tests.
- Placeholder scan: the plan contains no deferred implementation markers.
- Type consistency: test class names, paths, and asset parsing helpers are used consistently.
