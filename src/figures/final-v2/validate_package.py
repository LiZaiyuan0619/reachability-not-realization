#!/usr/bin/env python3
"""Compatibility entry point for package-level validation."""

from __future__ import annotations

import runpy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


if __name__ == "__main__":
    runpy.run_path(str(ROOT / "scripts" / "validate_package.py"), run_name="__main__")
