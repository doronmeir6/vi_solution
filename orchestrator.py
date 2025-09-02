from __future__ import annotations
from pathlib import Path
import json, re
import pandas as pd

from wellco_features.config import Config
from wellco_features.pipeline import FeaturePipeline
from wellco_features.training.model_api import (
    train_two_best_models, build_uplift_tlearner, choose_n_from_netvalue
)

def _read_baseline_auc_txt(path: Path) -> float | None:
    if not path.exists(): return None
    txt = path.read_text(errors="ignore")
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
    return float(nums[0]) if nums else None

def run_all(
    data_dir: str | Path = "data",
    out_dir: str | Path = "out",
    feature_select: str = "mi",
    feature_topk: int = 200,
    candidate_models=("logreg","xgb","lgbm"),
    tune_lgbm_trials: int = 80,
    benefit_per_saved: float = 150.0,
    cost_per_outcare: float = 30.0,
    resampler: str = "none",
) -> dict:
    cfg = Config(Path(data_dir), Path(out_dir))

    # 1) Features
    features_path = FeaturePipeline(cfg).run()
    feats = pd.read_csv(features_path)

    # 2) Train & pick best (ROC-AUC) for churn + outreach
    res = train_two_best_models(
        features_df=feats,
        feature_select=feature_select,
        feature_topk=feature_topk,
        candidate_models=candidate_models,
        tune_lgbm_trials=tune_lgbm_trials,
        resampler=resampler,
    )
    metrics_all = pd.concat([res["churn"]["metrics"], res["outreach"]["metrics"]], ignore_index=True)
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    metrics_all.to_csv(cfg.out_dir / "metrics_two_best.csv", index=False)
    res["churn"]["scores_val"].to_csv(cfg.out_dir / "scores_best_churn.csv", index=False)
    res["churn"]["ranking_val"].to_csv(cfg.out_dir / "ranking_risk_churn.csv", index=False)
    res["outreach"]["scores_val"].to_csv(cfg.out_dir / "scores_best_outreach.csv", index=False)
    res["outreach"]["ranking_val"].to_csv(cfg.out_dir / "ranking_prop_outreach.csv", index=False)

    # 3) Compact baseline comparison
    baseline_auc = _read_baseline_auc_txt(Path(cfg.data_dir) / "auc_baseline.txt") or \
                   _read_baseline_auc_txt(Path.cwd() / "auc_baseline.txt")
    rows = []
    for target in ("churn","outreach"):
        best_name = res[target]["best_name"]
        sub = metrics_all.loc[metrics_all["target"]==target].sort_values("auc_roc", ascending=False).head(1)
        if len(sub):
            row = {
                "target": target,
                "best_model": best_name,
                "auc_roc": float(sub["auc_roc"].iloc[0]),
                "auc_pr": float(sub["auc_pr"].iloc[0]),
                "accuracy": float(sub["accuracy"].iloc[0]),
                "precision": float(sub["precision"].iloc[0]),
                "recall": float(sub["recall"].iloc[0]),
                "f1": float(sub["f1"].iloc[0]),
                "base_rate": float(sub["base_rate"].iloc[0]),
            }
            if target == "churn" and baseline_auc is not None:
                row["baseline_auc_roc"] = float(baseline_auc)
                row["delta_auc_vs_baseline"] = row["auc_roc"] - row["baseline_auc_roc"]
            rows.append(row)
    pd.DataFrame(rows).to_csv(cfg.out_dir / "summary_compact.csv", index=False)

    # 4) Uplift uses the *same* algo as selected best churn model
    uplift = build_uplift_tlearner(
        features_df=feats,
        prefer_model="from_best",
        prefer_map={"churn": res["churn"]["best_name"]},
        benefit_per_saved=benefit_per_saved,
        cost_per_outcare=cost_per_outcare,
    )
    uplift["scores"].to_csv(cfg.out_dir / "scores_uplift.csv", index=False)
    uplift["ranking_uplift"].to_csv(cfg.out_dir / "ranking_uplift.csv", index=False)
    uplift["ranking_netvalue"].to_csv(cfg.out_dir / "ranking_uplift_netvalue.csv", index=False)

    # 5) Choose N (cost-aware)
    n_star, n_summary, outreach_list = choose_n_from_netvalue(
        uplift["ranking_netvalue"],
        cost_per_outcare=cost_per_outcare,
        benefit_per_saved=benefit_per_saved,
    )
    n_summary.update({
        "uplift_algo_used": uplift.get("uplift_algo_used", "unknown"),
        "best_churn_model": res["churn"]["best_name"],
        "best_outreach_model": res["outreach"]["best_name"],
    })
    (cfg.out_dir / "recommended_n.json").write_text(json.dumps(n_summary, indent=2))
    pd.DataFrame([n_summary]).to_csv(cfg.out_dir / "n_summary.csv", index=False)
    outreach_list.to_csv(cfg.out_dir / "outreach_list_uplift_netvalue.csv", index=False)

    # 6) Plots/report
    from vi_solution.wellco_features.analysis.auto_report import generate as generate_report
    report_dir = generate_report(cfg.out_dir)
    print("[REPORT] Open this:", (report_dir / "index.html").resolve())

    # 7) Manifest
    manifest = {
        "best_churn_model":   res["churn"]["best_name"],
        "best_outreach_model":res["outreach"]["best_name"],
        "selection_metric": "auc_roc",
        "tune_lgbm_trials": tune_lgbm_trials,
        "feature_select": feature_select,
        "feature_topk": feature_topk,
        "resampler": resampler,
    }
    (cfg.out_dir / "best_models.json").write_text(json.dumps(manifest, indent=2))

    return {
        "features_path": str(features_path),
        "best_models": manifest,
        "recommended_n": n_summary,
        "out_dir": str(cfg.out_dir),
    }
