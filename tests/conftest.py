from __future__ import annotations

import sys
from pathlib import Path

# Keep the offline unit suite runnable from a freshly synchronized checkout,
# before the remote bootstrap performs an editable package installation.
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SOURCE_ROOT))

