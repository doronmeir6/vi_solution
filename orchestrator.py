from __future__ import annotations
from pathlib import Path
import json, re
import pandas as pd
import numpy as np

from wellco_features.config import Config
from wellco_features.pipeline import FeaturePipeline
from wellco_features.training.model_api import (
    train_two_best_models,                    # full training (when no cache)
    train_best_for_target,                    # light refit to rebuild predictors
    build_uplift_tlearner,
    build_uplift_tlearner_with_threshold_gate,
    tune_threshold_gate_for_ev,               # returns (best_dict, grid_df) in our fixed version
    choose_n_from_netvalue,                   # we’ll still compute and plot EV curve; this remains for compatibility
)

# Plotting deps
from sklearn.metrics import f1_score, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings


def _read_baseline_auc_txt(path: Path) -> float | None:
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
    return float(nums[0]) if nums else None


def _infer_best_from_metrics(metrics_df: pd.DataFrame, target: str, selection_metric: str) -> str | None:
    sub = metrics_df.loc[metrics_df["target"] == target].copy()
    if sub.empty:
        return None
    metric_col = selection_metric if selection_metric in sub.columns else ("auc_roc" if "auc_roc" in sub.columns else None)
    if metric_col is None:
        return None
    sub = sub.sort_values(metric_col, ascending=False)
    if "model" in sub.columns and not sub.empty:
        return str(sub.iloc[0]["model"])
    return None


def run_all(
    data_dir: str | Path = "data",
    out_dir: str | Path = "out",
    feature_select: str = "mi",
    feature_topk: int = 200,
    candidate_models=("logreg", "xgb", "lgbm"),
    tune_lgbm_trials: int = 0,
    tune_logreg_trials: int = 60,
    tune_xgb_trials: int = 60,
    selection_metric: str = "auc_pr",
    benefit_per_saved: float = 150.0,
    cost_per_outcare: float = 30.0,
    resampler: str = "none",
    # EV gating + capacity
    use_ev_gate: bool = True,
    gate_mode: str = "and",                   # "and" | "mission_union"
    capacity_k_override: int | None = None,   # optional hard cap for N (otherwise use n_star from EV)
    # Soft penalty for EV (optional)
    use_soft_penalty: bool = False,
    soft_alpha: float = 0.3,
    soft_beta: float = 0.2,
) -> dict:
    """
    End-to-end with EV-tuned thresholds, cached-run light refit, and FULL plotting suite.

    Behavior:
      • If training artifacts exist, SKIP heavy training but LIGHT-REFIT the chosen best models
        to rebuild `predict_full`/`predict_full_raw`.
      • Always recomputes uplift (+ EV gating if enabled) on current features.
      • N is dynamic by default: taken from EV gating n_star, or overridden by capacity_k_override.
      • Plots and filenames match your old orchestrator:
          out/plots/auc_pr_best_models.png
          out/plots/f1_best_threshold.png
          out/plots/pr_curve_churn.png
          out/plots/pr_curve_outreach.png
          out/plots/ev_curve_with_n_star.png
      • Generates an HTML report if available and writes out/index.url to open it.
    """
    cfg = Config(Path(data_dir), Path(out_dir))
    cfg.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Features
    features_path = FeaturePipeline(cfg).run()
    feats = pd.read_csv(features_path)

    # 2) Load or train models
    metrics_file            = cfg.out_dir / "metrics_two_best.csv"
    churn_scores_file       = cfg.out_dir / "scores_best_churn.csv"
    outreach_scores_file    = cfg.out_dir / "scores_best_outreach.csv"
    churn_ranking_file      = cfg.out_dir / "ranking_risk_churn.csv"
    outreach_ranking_file   = cfg.out_dir / "ranking_prop_outreach.csv"

    have_training_artifacts = all([
        metrics_file.exists(),
        churn_scores_file.exists(),
        outreach_scores_file.exists(),
        churn_ranking_file.exists(),
        outreach_ranking_file.exists(),
    ])

    if have_training_artifacts:
        print("[INFO] Skipping heavy training – using cached artifacts and light refit for predictors.")
        metrics_all = pd.read_csv(metrics_file)

        # Infer best models (by selection metric)
        best_churn_algo = _infer_best_from_metrics(metrics_all, "churn", selection_metric) or "logreg"
        best_out_algo   = _infer_best_from_metrics(metrics_all, "outreach", selection_metric) or "logreg"

        # Light refit to rebuild predict_full/predict_full_raw
        churn_res = train_best_for_target(
            features_df=feats,
            target="churn",
            candidate_models=(best_churn_algo,),
            feature_select=feature_select,
            feature_topk=feature_topk,
            validation_size=0.30,
            tune_lgbm_trials=0,
            selection_metric=selection_metric,
            resampler=resampler,
            tune_logreg_trials=0,
            tune_xgb_trials=0,
        )
        outreach_res = train_best_for_target(
            features_df=feats,
            target="outreach",
            candidate_models=(best_out_algo,),
            feature_select=feature_select,
            feature_topk=feature_topk,
            validation_size=0.30,
            tune_lgbm_trials=0,
            selection_metric=selection_metric,
            resampler=resampler,
            tune_logreg_trials=0,
            tune_xgb_trials=0,
        )

        # Keep previously saved artifacts for consistency / CSV outputs
        churn_scores = pd.read_csv(churn_scores_file)
        churn_rank   = pd.read_csv(churn_ranking_file)
        out_scores   = pd.read_csv(outreach_scores_file)
        out_rank     = pd.read_csv(outreach_ranking_file)

        res = {
            "churn": {
                "best_name": best_churn_algo,
                "metrics": metrics_all[metrics_all["target"]=="churn"],
                "scores_val": churn_scores,
                "ranking_val": churn_rank,
                **churn_res
            },
            "outreach": {
                "best_name": best_out_algo,
                "metrics": metrics_all[metrics_all["target"]=="outreach"],
                "scores_val": out_scores,
                "ranking_val": out_rank,
                **outreach_res
            },
        }

    else:
        res = train_two_best_models(
            features_df=feats,
            feature_select=feature_select,
            feature_topk=feature_topk,
            candidate_models=candidate_models,
            tune_lgbm_trials=tune_lgbm_trials,
            resampler=resampler,
            selection_metric=selection_metric,
            tune_logreg_trials=tune_logreg_trials,
            tune_xgb_trials=tune_xgb_trials,
        )

        metrics_all = pd.concat([res["churn"]["metrics"], res["outreach"]["metrics"]], ignore_index=True)
        cfg.out_dir.mkdir(parents=True, exist_ok=True)
        metrics_all.to_csv(metrics_file, index=False)
        res["churn"]["scores_val"].to_csv(churn_scores_file, index=False)
        res["churn"]["ranking_val"].to_csv(churn_ranking_file, index=False)
        res["outreach"]["scores_val"].to_csv(outreach_scores_file, index=False)
        res["outreach"]["ranking_val"].to_csv(outreach_ranking_file, index=False)

    # 3) Compact baseline comparison (same outputs as old)
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

    # 4) Uplift (ungated + optional EV gating as in your old flow)
    uplift_ungated = build_uplift_tlearner(
        features_df=feats,
        prefer_model="from_best",
        prefer_map={"churn": res["churn"]["best_name"]},
        benefit_per_saved=benefit_per_saved,
        cost_per_outcare=cost_per_outcare,
        gate_by_scores=None
    )
    uplift_ungated["scores"].to_csv(cfg.out_dir / "scores_uplift.csv", index=False)
    uplift_ungated["ranking_netvalue"].to_csv(cfg.out_dir / "ranking_uplift_netvalue.csv", index=False)
    uplift_ungated["ranking_uplift"].to_csv(cfg.out_dir / "ranking_uplift.csv", index=False)

    if use_ev_gate:
        # Our fixed tune function returns (best_dict, grid_df)
        tuned_res, tuned_grid = tune_threshold_gate_for_ev(
            features_df=feats,
            churn_result=res["churn"],
            outreach_result=res["outreach"],
            benefit_per_saved=benefit_per_saved,
            cost_per_outcare=cost_per_outcare,
            mode=gate_mode,
            use_soft_penalty=use_soft_penalty,
            soft_alpha=soft_alpha,
            soft_beta=soft_beta,
        )
        uplift = build_uplift_tlearner_with_threshold_gate(
            features_df=feats,
            churn_result=res["churn"],
            outreach_result=res["outreach"],
            benefit_per_saved=benefit_per_saved,
            cost_per_outcare=cost_per_outcare,
            gate_mode=gate_mode,
            thr_churn=float(tuned_res["thr_churn"]),
            thr_outreach=float(tuned_res["thr_outreach"]),
            use_soft_penalty=use_soft_penalty,
            soft_alpha=soft_alpha,
            soft_beta=soft_beta,
        )
        uplift["ranking_netvalue"].to_csv(cfg.out_dir / "ranking_uplift_netvalue__eligible.csv", index=False)
        uplift["ranking_uplift"].to_csv(cfg.out_dir / "ranking_uplift__eligible.csv", index=False)
        tuned_grid.to_csv(cfg.out_dir / "gate_grid_results.csv", index=False)
    else:
        uplift = build_uplift_tlearner_with_threshold_gate(
            features_df=feats,
            churn_result=res["churn"],
            outreach_result=res["outreach"],
            benefit_per_saved=benefit_per_saved,
            cost_per_outcare=cost_per_outcare,
            gate_mode=gate_mode,
        )

    # 5) Choose N (dynamic by default from EV; keep same filenames as old)
    ranking_nv = uplift["ranking_netvalue"]
    # Derive N*: prefer EV-optimal n_star if we tuned; else fall back to best cumulative EV on ranking
    if use_ev_gate and isinstance(tuned_res, dict) and "n_star" in tuned_res:
        n_star = int(tuned_res["n_star"])
    else:
        # generic argmax cumulative policy/net value
        ev_col = "policy_net_value" if "policy_net_value" in ranking_nv.columns else "net_value"
        rank_nv = ranking_nv.sort_values(ev_col, ascending=False).reset_index(drop=True)
        rank_nv["cum"] = rank_nv[ev_col].cumsum()
        n_star = int(rank_nv["cum"].idxmax() + 1) if len(rank_nv) else 0

    N = int(capacity_k_override) if (capacity_k_override is not None and capacity_k_override > 0) else (
        n_star if n_star > 0 else len(ranking_nv)
    )

    # Build outreach list and summary (keeping your filenames)
    ev_col = "policy_net_value" if "policy_net_value" in ranking_nv.columns else "net_value"
    rank_sorted = ranking_nv.sort_values(ev_col, ascending=False).reset_index(drop=True)
    outreach_list = rank_sorted.head(N).copy()
    outreach_list.to_csv(cfg.out_dir / "outreach_list_uplift_netvalue.csv", index=False)

    # Summary to match old fields as much as possible
    n_summary = {
        "n_star": int(N),
        "benefit_per_saved": float(benefit_per_saved),
        "cost_per_outcare": float(cost_per_outcare),
        "gate_mode": gate_mode,
        "use_ev_gate": bool(use_ev_gate),
        "best_churn_model": res["churn"]["best_name"],
        "best_outreach_model": res["outreach"]["best_name"],
        "selection_metric": selection_metric,
        "ev_total_at_ev_gate_topK": float(outreach_list[ev_col].sum()),
        "eligible_pool_size": int(len(rank_sorted)),
        "overridden_capacity_k": int(capacity_k_override) if capacity_k_override is not None else None,
    }
    if use_ev_gate and isinstance(tuned_res, dict):
        n_summary.update({
            "thr_churn_ev": float(tuned_res.get("thr_churn", tuned_res.get("thr_churn_ev", 0.0))),
            "thr_outreach_ev": float(tuned_res.get("thr_outreach", tuned_res.get("thr_outreach_ev", 0.0))),
            "n_star_from_ev": int(tuned_res.get("n_star", N)),
        })
    else:
        # F1 thresholds from the model selection, like in your old manifest
        n_summary.update({
            "thr_churn_f1": float(res["churn"].get("thr_f1", 0.0)),
            "thr_outreach_f1": float(res["outreach"].get("thr_f1", 0.0)),
        })

    (cfg.out_dir / "recommended_n.json").write_text(json.dumps(n_summary, indent=2))
    pd.DataFrame([n_summary]).to_csv(cfg.out_dir / "n_summary.csv", index=False)

    # 6) PLOTS — EXACT directory and filenames like your old version
    plots_dir = cfg.out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # AUC-PR (best models)
    try:
        aucpr = []
        for target in ("churn", "outreach"):
            best_name = res[target]["best_name"]
            sub = metrics_all[(metrics_all["target"] == target) & (metrics_all["model"] == best_name)]
            aucpr.append(float(sub["auc_pr"].iloc[0]) if len(sub) else 0.0)
        plt.figure()
        plt.bar(["churn", "outreach"], aucpr)
        plt.title("AUC-PR (best models)")
        plt.ylabel("AUC-PR")
        plt.ylim(0, 1)
        plt.savefig(plots_dir / "auc_pr_best_models.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        warnings.warn(f"[WARN] AUC-PR plot failed: {e}")

    # F1 at calibrated threshold
    def _f1_at_thr(scores_df: pd.DataFrame, thr: float) -> float:
        pcols = [c for c in scores_df.columns if c.startswith("p_")]
        if not pcols or "y_true" not in scores_df.columns:
            return 0.0
        pcol = pcols[0]
        yhat = (scores_df[pcol].values >= float(thr)).astype(int)
        return float(f1_score(scores_df["y_true"].values, yhat, zero_division=0))

    try:
        f1s = [
            _f1_at_thr(res["churn"]["scores_val"], res["churn"].get("thr_f1", 0.5)),
            _f1_at_thr(res["outreach"]["scores_val"], res["outreach"].get("thr_f1", 0.5)),
        ]
        plt.figure()
        plt.bar(["churn", "outreach"], f1s)
        plt.title("F1 at calibrated threshold (best models)")
        plt.ylabel("F1")
        plt.ylim(0, 1)
        plt.savefig(plots_dir / "f1_best_threshold.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        warnings.warn(f"[WARN] F1 plot failed: {e}")

    # Precision–Recall curves (from saved validation scores)
    for target in ("churn", "outreach"):
        try:
            sv = res[target]["scores_val"].copy()
            pcols = [c for c in sv.columns if c.startswith("p_")]
            if not pcols or "y_true" not in sv.columns:
                raise ValueError("scores_val missing probability or y_true")
            pcol = pcols[0]
            y_true = pd.to_numeric(sv["y_true"], errors="coerce")
            keep = ~y_true.isna()
            y_true = y_true[keep].values
            y_score = pd.to_numeric(sv[pcol], errors="coerce")[keep].values
            P, R, _ = precision_recall_curve(y_true, y_score)
            plt.figure()
            plt.plot(R, P)
            plt.title(f"PR Curve – {target} ({res[target]['best_name']})")
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.ylim(0, 1); plt.xlim(0, 1)
            plt.savefig(plots_dir / f"pr_curve_{target}.png", bbox_inches="tight")
            plt.close()
        except Exception as e:
            warnings.warn(f"[WARN] PR curve for {target} failed: {e}")

    # EV curve (cumulative net value) with N* marker — from active ranking
    try:
        evc = rank_sorted[ev_col].cumsum()
        plt.figure()
        plt.plot(range(1, len(evc) + 1), evc.values)
        if N > 0:
            plt.axvline(N, linestyle="--")
        gate_label = "EV gate" if use_ev_gate else "F1 gate"
        plt.title(f"Cumulative Net Value & N* ({N}) – {gate_label}")
        plt.xlabel("Top-N"); plt.ylabel("Cumulative Net Value")
        plt.savefig(plots_dir / "ev_curve_with_n_star.png", bbox_inches="tight")
        plt.close()
    except Exception as e:
        warnings.warn(f"[WARN] EV curve plot failed: {e}")

    # 7) Optional HTML report and index.url
    report_index_path = None
    try:
        from wellco_features.analysis.auto_report import generate as generate_report
        report_dir = generate_report(cfg.out_dir)  # returns a directory containing index.html
        report_index_path = (report_dir / "index.html").resolve()
        print("[REPORT] Open this:", report_index_path)
    except Exception as e:
        print("[WARN] auto_report failed:", e)

    try:
        # Write a Windows Internet Shortcut that opens either the report or the plots folder
        url_target = str(report_index_path) if report_index_path else str((cfg.out_dir / "plots").resolve())
        (cfg.out_dir / "index.url").write_text(f"[InternetShortcut]\nURL=file:///{url_target}\n", encoding="utf-8")
    except Exception as e:
        print("[WARN] writing index.url failed:", e)

    # 8) Manifest (similar to your old one)
    manifest = {
        "best_churn_model":   res["churn"]["best_name"],
        "best_outreach_model":res["outreach"]["best_name"],
        "selection_metric": selection_metric,
        "tune_lgbm_trials": tune_lgbm_trials,
        "tune_logreg_trials": tune_logreg_trials,
        "tune_xgb_trials": tune_xgb_trials,
        "feature_select": feature_select,
        "feature_topk": feature_topk,
        "resampler": resampler,
        "gate_mode": gate_mode,
        "use_ev_gate": bool(use_ev_gate),
    }
    if use_ev_gate and isinstance(tuned_res, dict):
        manifest.update({
            "thr_churn_ev": float(tuned_res.get("thr_churn", tuned_res.get("thr_churn_ev", 0.0))),
            "thr_outreach_ev": float(tuned_res.get("thr_outreach", tuned_res.get("thr_outreach_ev", 0.0))),
        })
    else:
        manifest.update({
            "thr_churn_f1": float(res["churn"].get("thr_f1", 0.0)),
            "thr_outreach_f1": float(res["outreach"].get("thr_f1", 0.0)),
        })
    (cfg.out_dir / "best_models.json").write_text(json.dumps(manifest, indent=2))

    return {
        "features_path": str(features_path),
        "best_models": manifest,
        "recommended_n": n_summary,
        "out_dir": str(cfg.out_dir),
    }
