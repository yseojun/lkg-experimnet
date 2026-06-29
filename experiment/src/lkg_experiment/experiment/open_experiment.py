#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import webbrowser
from dataclasses import dataclass
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from lkg_experiment.coherent_default.coherent_raster_experiment import build_experiment_web_assets


DEFAULT_EXPERIMENTS_ROOT = Path("/data/ysj/result/coherent-raster/generated/coherent_raster_experiments")


@dataclass(frozen=True)
class ExperimentTarget:
    experiments_root: Path
    run_id: Optional[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve CoherentRaster experiment web assets and open a selected run in a browser"
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_EXPERIMENTS_ROOT),
        help="Experiment run directory containing manifest.json, or the experiments root directory",
    )
    parser.add_argument("--host", default="127.0.0.1", help="HTTP bind host")
    parser.add_argument("--port", default=8000, type=int, help="First HTTP port to try")
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Print the URL and serve it without calling the local browser opener",
    )
    parser.add_argument(
        "--no-regenerate-previews",
        action="store_true",
        help="Only rebuild index.html/viewer assets; do not regenerate mapping PNG previews from raw_mapping.npz",
    )
    return parser


def resolve_experiment_target(path: Path | str) -> ExperimentTarget:
    target_path = Path(path).expanduser().resolve()
    if not target_path.exists():
        raise FileNotFoundError(f"experiment path does not exist: {target_path}")
    if not target_path.is_dir():
        raise NotADirectoryError(f"experiment path is not a directory: {target_path}")

    if (target_path / "manifest.json").is_file():
        return ExperimentTarget(experiments_root=target_path.parent, run_id=target_path.name)

    run_dirs = [child for child in target_path.iterdir() if child.is_dir() and (child / "manifest.json").is_file()]
    if run_dirs:
        return ExperimentTarget(experiments_root=target_path, run_id=None)

    raise ValueError(
        "path must be an experiment run directory containing manifest.json "
        f"or an experiments root containing run directories: {target_path}"
    )


def build_experiment_url(host: str, port: int, run_id: Optional[str]) -> str:
    url = f"http://{host}:{int(port)}/index.html"
    if run_id:
        url = f"{url}#{urlencode({'run': run_id})}"
    return url


def serve_experiment_root(experiments_root: Path, host: str, preferred_port: int) -> tuple[ThreadingHTTPServer, int]:
    handler = partial(SimpleHTTPRequestHandler, directory=str(experiments_root))
    last_error: Optional[OSError] = None
    for port in range(int(preferred_port), int(preferred_port) + 100):
        try:
            server = ThreadingHTTPServer((host, port), handler)
        except OSError as error:
            last_error = error
            continue
        return server, port
    raise RuntimeError(f"could not bind an HTTP server near port {preferred_port}: {last_error}")


def main() -> None:
    args = build_parser().parse_args()
    target = resolve_experiment_target(args.path)

    index_path = build_experiment_web_assets(
        target.experiments_root,
        regenerate_previews=not args.no_regenerate_previews,
    )
    server, port = serve_experiment_root(target.experiments_root, args.host, args.port)
    url = build_experiment_url(args.host, port, target.run_id)

    print(f"Wrote {index_path}", flush=True)
    print(f"Serving {target.experiments_root} at http://{args.host}:{port}/", flush=True)
    print(f"Open: {url}", flush=True)

    if not args.no_browser:
        opened = webbrowser.open(url, new=2)
        if not opened:
            print("Browser opener did not report success; open the printed URL manually.", file=sys.stderr, flush=True)

    print("Press Ctrl+C to stop the server.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
