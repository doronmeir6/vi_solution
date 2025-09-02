# final_vi/main_hardcoded.py
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from orchestrator import run_all

def _project_root() -> Path:
    # This file lives inside .../final_vi/, so parent is the project root
    return Path(__file__).resolve().parent

if __name__ == "__main__":
    root = _project_root()

    # ✅ Force-create out/ BEFORE anything else (independent of Config)
    out_dir = (root / "out").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Drop a sentinel so you can see it even if training crashes early
    (out_dir / "_created.txt").write_text(f"created at {datetime.now().isoformat()}\n", encoding="utf-8")

    # Data directory next to this file (adjust if yours is elsewhere)
    data_dir = (root / "data").resolve()

    # Run end-to-end
    res = run_all(
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        feature_select="mi",
        feature_topk=200,
        candidate_models=("logreg", "xgb", "lgbm"),
        tune_lgbm_trials=80,
        benefit_per_saved=150.0,
        cost_per_outcare=30.0,
    )

    print(f"[DONE] Outputs under: {out_dir}")
    print(f"Exists? {out_dir.exists()}")
