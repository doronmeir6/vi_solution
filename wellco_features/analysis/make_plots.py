# final_vi/analysis/make_plots.py
from __future__ import annotations
from pathlib import Path
import sys

# Ensure the project root (the parent of 'final_vi') is on sys.path
HERE = Path(__file__).resolve()
PKG_ROOT = HERE.parents[2]   # .../<project_root>/
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    # Preferred: import via package
    from vi_solution.wellco_features.analysis.auto_report import generate
except ModuleNotFoundError:
    # Fallback: relative import if executed inside the package context
    from .auto_report import generate

if __name__ == "__main__":
    # Default outputs folder next to 'final_vi'
    out_dir = HERE.parents[1] / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = generate(out_dir)
    print("[REPORT] Open this in your browser:", (plots / "index.html").resolve())
