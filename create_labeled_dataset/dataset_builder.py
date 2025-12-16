"""
Backwards-compatible entrypoint.

Use 1_build_dataset.py going forward.
"""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    target = Path(__file__).with_name("1_build_dataset.py")
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
