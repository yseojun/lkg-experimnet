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
