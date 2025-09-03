from __future__ import annotations
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[2]
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    from wellco_features.analysis.auto_report import generate
except ModuleNotFoundError:
    from .auto_report import generate

if __name__ == "__main__":
    out_dir = HERE.parents[1] / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = generate(out_dir)
    print("[REPORT] Open this in your browser:", (plots / "index.html").resolve())
