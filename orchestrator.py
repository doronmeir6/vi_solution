from __future__ import annotations
from pathlib import Path
import json, re
import pandas as pd

from wellco_features.config import Config
from wellco_features.pipeline import FeaturePipeline
from wellco_features.training.model_api import (
    train_two_best_models,
    build_uplift_tlearner,
    build_uplift_tlearner_with_threshold_gate,
    choose_n_from_netvalue,
    tune_threshold_gate_for_ev,  # supports soft-penalty
)

# For plots
from sklearn.metrics import precision_recall_curve, f1_score
import warnings


def _read_baseline_auc_txt(path: Path) -> float | None:
    if not path.exists():
        return None
    txt = path.read_text(errors="ignore")
    nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
    return float(nums[0]) if nums else None


def run_all(
        data_dir: str | Path = "data",
        out_dir: str | Path = "out",
        feature_select: str = "mi",
        feature_topk: int = 200,
        candidate_models=("logreg", "xgb"),
        tune_lgbm_trials: int = 0,
        tune_logreg_trials: int = 60,
        tune_xgb_trials: int = 60,
        selection_metric: str = "auc_pr",
        benefit_per_saved: float = 150.0,
        cost_per_outcare: float = 30.0,
        resampler: str = "none",
        # Gating / capacity
        gate_mode: str = "and",  # "and" or "mission_union"
        capacity_k: int = 265,  # fixed N you want
        # Soft penalty (NEW)
        use_soft_penalty: bool = True,
        soft_alpha: float = 0.3,  # penalty on high p_outreach
        soft_beta: float = 0.2,  # penalty on low p_churn
) -> dict:
    """
    End-to-end with EV-tuned thresholds for a fixed capacity K and optional soft penalty:
      - soft penalty re-scores net_value = uplift*B - C*(1 + α*p_outreach + β*(1 - p_churn))
    """
    cfg = Config(Path(data_dir), Path(out_dir))

    # 1) Features
    features_path = FeaturePipeline(cfg).run()
    feats = pd.read_csv(features_path)

    # 2) Train & pick best (AUC-PR default; can switch to F1)
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
    metrics_all.to_csv(cfg.out_dir / "metrics_two_best.csv", index=False)
    res["churn"]["scores_val"].to_csv(cfg.out_dir / "scores_best_churn.csv", index=False)
    res["churn"]["ranking_val"].to_csv(cfg.out_dir / "ranking_risk_churn.csv", index=False)
    res["outreach"]["scores_val"].to_csv(cfg.out_dir / "scores_best_outreach.csv", index=False)
    res["outreach"]["ranking_val"].to_csv(cfg.out_dir / "ranking_prop_outreach.csv", index=False)

    # 3) Baseline compact table
    baseline_auc = _read_baseline_auc_txt(Path(cfg.data_dir) / "auc_baseline.txt") or \
                   _read_baseline_auc_txt(Path.cwd() / "auc_baseline.txt")
    rows = []
    for target in ("churn", "outreach"):
        best_name = res[target]["best_name"]
        sub = metrics_all.loc[metrics_all["target"] == target].sort_values("auc_roc", ascending=False).head(1)
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
            rows.append(row)
    pd.DataFrame(rows).to_csv(cfg.out_dir / "summary_compact.csv", index=False)

    # 4) Uplift without gating (baseline uplift ranking)
    uplift_ungated = build_uplift_tlearner(
        features_df=feats,
        prefer_model="from_best",
        prefer_map={"churn": res["churn"]["best_name"]},
        benefit_per_saved=benefit_per_saved,
        cost_per_outcare=cost_per_outcare,
    )
    uplift_ungated["scores"].to_csv(cfg.out_dir / "scores_uplift.csv", index=False)
    uplift_ungated["ranking_netvalue"].to_csv(cfg.out_dir / "ranking_uplift_netvalue.csv", index=False)
    uplift_ungated["ranking_uplift"].to_csv(cfg.out_dir / "ranking_uplift.csv", index=False)

    # 5) EV-tune thresholds (capacity K is applied later when slicing Top-K)
    tuned = tune_threshold_gate_for_ev(
        features_df=feats,
        churn_result=res["churn"],
        outreach_result=res["outreach"],
        benefit_per_saved=benefit_per_saved,
        cost_per_outcare=cost_per_outcare,
        mode=gate_mode,  # "and" | "mission_union"
        use_soft_penalty=use_soft_penalty,
        soft_alpha=soft_alpha,
        soft_beta=soft_beta,
        # soft_gamma_mission stays default unless you set it explicitly
    )

    # Apply tuned gate (and soft penalty) to build eligible ranking
    gate = {
        "p_churn": tuned["p_churn_all"],
        "p_outreach": tuned["p_outreach_all"],
        "thr_churn": tuned["thr_churn_ev"],
        "thr_outreach": tuned["thr_outreach_ev"],
        "mode": gate_mode
    }
    uplift = build_uplift_tlearner(
        features_df=feats,
        prefer_model="from_best",
        prefer_map={"churn": res["churn"]["best_name"]},
        benefit_per_saved=benefit_per_saved,
        cost_per_outcare=cost_per_outcare,
        gate_by_scores=gate,
        soft_penalty=(dict(alpha=soft_alpha, beta=soft_beta) if use_soft_penalty else None),
    )
    uplift["ranking_netvalue"].to_csv(cfg.out_dir / "ranking_uplift_netvalue__eligible.csv", index=False)
    uplift["ranking_uplift"].to_csv(cfg.out_dir / "ranking_uplift__eligible.csv", index=False)
    tuned["grid"].to_csv(cfg.out_dir / "gate_grid_results.csv", index=False)

    # 6) FINAL LIST: take Top-K from the eligible net-value ranking
    rank_nv = uplift["ranking_netvalue"].sort_values("net_value", ascending=False).reset_index(drop=True)
    outreach_list = rank_nv.head(int(capacity_k)).copy()
    outreach_list.to_csv(cfg.out_dir / "outreach_list_uplift_netvalue.csv", index=False)

    # 7) N summary (+ record soft-penalty settings)
    n_summary = {
        "n_star": int(capacity_k),
        "benefit_per_saved": float(benefit_per_saved),
        "cost_per_outcare": float(cost_per_outcare),
        "thr_churn_ev": float(tuned["thr_churn_ev"]),
        "thr_outreach_ev": float(tuned["thr_outreach_ev"]),
        "gate_mode": gate_mode,
        "use_ev_gate": True,
        "best_churn_model": res["churn"]["best_name"],
        "best_outreach_model": res["outreach"]["best_name"],
        "selection_metric": selection_metric,
        "ev_total_at_ev_gate_topK": float(outreach_list["net_value"].sum()),
        "eligible_pool_size": int(len(rank_nv)),
        "use_soft_penalty": bool(use_soft_penalty),
        "soft_alpha": float(soft_alpha) if use_soft_penalty else 0.0,
        "soft_beta": float(soft_beta) if use_soft_penalty else 0.0,
    }
    (cfg.out_dir / "recommended_n.json").write_text(json.dumps(n_summary, indent=2))
    pd.DataFrame([n_summary]).to_csv(cfg.out_dir / "n_summary.csv", index=False)

    # 8) Label-mix audit on Top-K (checks (0,0) and (1,1) are reduced)
    try:
        lab = feats[["member_id", "churn", "outreach"]].drop_duplicates()
        topK = outreach_list.merge(lab, on="member_id", how="left")
        mix = (topK.assign(
            group=lambda d: d["churn"].astype(int).astype(str) + "_" + d["outreach"].astype(int).astype(str))
               .groupby("group")["member_id"].count().rename("count").reset_index())
        mix["share"] = mix["count"] / len(topK)
        mix.to_csv(cfg.out_dir / "audit_topK_label_mix.csv", index=False)
    except Exception as e:
        print("[AUDIT] skipped:", e)

    # 9) Plots
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plots_dir = cfg.out_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # PR curves
        for target in ("churn", "outreach"):
            sv = res[target]["scores_val"].copy()
            pcol = [c for c in sv.columns if c.startswith("p_")][0]
            P, R, _ = precision_recall_curve(sv["y_true"].values, sv[pcol].values)
            plt.figure();
            plt.plot(R, P)
            plt.title(f"PR Curve – {target} ({res[target]['best_name']})")
            plt.xlabel("Recall");
            plt.ylabel("Precision");
            plt.ylim(0, 1);
            plt.xlim(0, 1)
            plt.savefig(plots_dir / f"pr_curve_{target}.png", bbox_inches="tight");
            plt.close()

        # EV curve (eligible) with K marker
        rank_nv["cum_net_value"] = rank_nv["net_value"].cumsum()
        plt.figure()
        plt.plot(range(1, len(rank_nv) + 1), rank_nv["cum_net_value"].values)
        plt.axvline(int(capacity_k), linestyle="--")
        title_suffix = " + soft penalty" if use_soft_penalty else ""
        plt.title(f"Cumulative Net Value – EV-gated (K={capacity_k}){title_suffix}")
        plt.xlabel("Top-N");
        plt.ylabel("Cumulative Net Value")
        plt.savefig(plots_dir / "ev_curve_ev_gated_fixedK.png", bbox_inches="tight");
        plt.close()
    except Exception as e:
        print("[WARN] plotting failed:", e)

    # 10) Manifest
    manifest = {
        "best_churn_model": res["churn"]["best_name"],
        "best_outreach_model": res["outreach"]["best_name"],
        "selection_metric": selection_metric,
        "feature_select": feature_select,
        "feature_topk": feature_topk,
        "resampler": resampler,
        "gate_mode": gate_mode,
        "capacity_k": int(capacity_k),
        "use_ev_gate": True,
        "thr_churn_ev": float(tuned["thr_churn_ev"]),
        "thr_outreach_ev": float(tuned["thr_outreach_ev"]),
        "use_soft_penalty": bool(use_soft_penalty),
        "soft_alpha": float(soft_alpha) if use_soft_penalty else 0.0,
        "soft_beta": float(soft_beta) if use_soft_penalty else 0.0,
    }
    (cfg.out_dir / "best_models.json").write_text(json.dumps(manifest, indent=2))

    return {
        "features_path": str(features_path),
        "best_models": manifest,
        "recommended_n": n_summary,
        "out_dir": str(cfg.out_dir),
    }
