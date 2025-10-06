"""Ensure local Python dependencies from python_libs are discoverable."""

from __future__ import annotations

import sys
from pathlib import Path

_VENDOR_DIR = (Path(__file__).resolve().parent / 'python_libs').resolve()
if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))
