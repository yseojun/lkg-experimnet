#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys


SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lkg_experiment.run_coherent_raster_experiment import main


if __name__ == "__main__":
    main()
