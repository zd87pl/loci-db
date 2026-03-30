"""Test bootstrap helpers.

Keep the repository root importable so pytest can load ``loci`` directly from the
working tree without requiring ``pip install -e .`` first.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
root_str = str(ROOT)
if root_str not in sys.path:
    sys.path.insert(0, root_str)
