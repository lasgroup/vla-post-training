from pathlib import Path
import sys


_ROOT = Path(__file__).resolve().parents[2]
for _rel in ("", "openpi/src", "openpi/packages/openpi-client/src"):
    _p = str(_ROOT / _rel) if _rel else str(_ROOT)
    if _p not in sys.path:
        sys.path.insert(0, _p)
