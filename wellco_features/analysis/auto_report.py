from __future__ import annotations
from pathlib import Path
import json, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- helpers ----------
def _first_prob_col(df: pd.DataFrame) -> str:
    for c in df.columns:
        if c.startswith("p_"):
            return c
    raise ValueError("No probability column (p_*) found.")

def _maybe_read_baseline(out_dir: Path) -> float | None:
    for p in [out_dir.parent / "auc_baseline.txt", out_dir / "auc_baseline.txt"]:
        if p.exists():
            txt = p.read_text(errors="ignore")
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", txt)
            if nums: return float(nums[0])
    return None

def _load_best_rows(metrics_csv: Path) -> pd.DataFrame:
    if (metrics_csv.parent / "summary_compact.csv").exists():
        return pd.read_csv(metrics_csv.parent / "summary_compact.csv")
    if not metrics_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(metrics_csv)
    if "target" not in df.columns or "auc_roc" not in df.columns:
        return pd.DataFrame()
    best = (df.sort_values(["target","auc_roc"], ascending=[True, False])
              .groupby("target", as_index=False).first())
    if "model" in best.columns:
        best = best.rename(columns={"model":"best_model"})
    for c in ["best_model","auc_pr","accuracy","precision","recall","f1","base_rate"]:
        if c not in best.columns: best[c] = np.nan
    return best[["target","best_model","auc_roc","auc_pr","accuracy","precision","recall","f1","base_rate"]]

# ---------- plots ----------
def _roc_pr_curves(scores_csv: Path, title: str, out_png_prefix: Path):
    if not scores_csv.exists(): return
    df = pd.read_csv(scores_csv)
    y = df["y_true"].astype(int).values
    p = df[_first_prob_col(df)].astype(float).values
    from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    fpr, tpr, _ = roc_curve(y, p); roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y, p); ap = average_precision_score(y, p)

    # ROC
    plt.figure(figsize=(6,4.5))
    plt.plot(fpr, tpr); plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{title} ROC (AUC={roc_auc:.3f})")
    plt.tight_layout(); plt.savefig(out_png_prefix.with_suffix(".roc.png")); plt.close()

    # PR
    plt.figure(figsize=(6,4.5))
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"{title} PR (AP={ap:.3f})")
    plt.tight_layout(); plt.savefig(out_png_prefix.with_suffix(".pr.png")); plt.close()

def _metrics_grouped_bars(best_metrics: pd.DataFrame, out_png: Path):
    if best_metrics.empty: return
    targets = best_metrics["target"].str.title().tolist()
    acc = best_metrics["accuracy"].astype(float).values
    pre = best_metrics["precision"].astype(float).values
    rec = best_metrics["recall"].astype(float).values
    f1  = best_metrics["f1"].astype(float).values

    x = np.arange(len(targets)); w = 0.2
    plt.figure(figsize=(8,4.5))
    plt.bar(x - 1.5*w, acc, width=w, label="Accuracy")
    plt.bar(x - 0.5*w, pre, width=w, label="Precision")
    plt.bar(x + 0.5*w, rec, width=w, label="Recall")
    plt.bar(x + 1.5*w, f1,  width=w, label="F1")
    plt.xticks(x, targets); plt.ylim(0, 1.0)
    plt.title("Validation metrics by target (best model)")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()

def _baseline_vs_best(metrics_csv: Path, out_png: Path, baseline_auc: float | None):
    df = _load_best_rows(metrics_csv)
    if df.empty: return
    labels = df["target"].str.title().tolist()
    aucs = df["auc_roc"].astype(float).values

    plt.figure(figsize=(6,4))
    x = np.arange(len(labels))
    plt.bar(x, aucs, width=0.5, label="Best AUC-ROC")
    if baseline_auc is not None:
        plt.hlines(baseline_auc, xmin=-0.5, xmax=len(labels)-0.5, linestyles="--", label=f"Baseline {baseline_auc:.3f}")
    plt.xticks(x, labels); plt.ylabel("AUC-ROC"); plt.title("Baseline vs Best (by target)")
    plt.legend(); plt.tight_layout(); plt.savefig(out_png); plt.close()

def _uplift_ev_curve(ranking_nv_csv: Path, rec_json: Path, out_png: Path):
    # prefer eligible ranking if present
    eligible_path = ranking_nv_csv.with_name("ranking_uplift_netvalue__eligible.csv")
    path = eligible_path if eligible_path.exists() else ranking_nv_csv
    if not path.exists(): return

    df = pd.read_csv(path)
    if "net_value" not in df.columns or df.empty: return
    nv = df["net_value"].astype(float).values
    cum = np.cumsum(nv)

    # robust N finder (supports multiple key names)
    N, algo, total = None, None, None
    if rec_json.exists():
        r = json.loads(rec_json.read_text())
        for k in ["n_star", "recommended_n", "argmax_ev", "N", "N_selected",
                  "recommended_n_validation", "recommended_n_all"]:
            v = r.get(k, None)
            if isinstance(v, (int, float)) and v and v > 0:
                N = int(v); break
        algo  = r.get("uplift_algo_used", None)
        total = r.get("total_net_value", None)

    if N is not None and N > len(cum):
        N = len(cum)

    plt.figure(figsize=(7,4.5))
    plt.plot(np.arange(1, len(cum)+1), cum)
    title = "Cost-aware uplift: cumulative net value vs N"
    if algo: title += f" (uplift={algo})"
    if eligible_path.exists(): title += " [eligible only]"
    plt.title(title)
    plt.xlabel("N (top-N)"); plt.ylabel("Cumulative net value")

    if N and 1 <= N <= len(cum):
        yN = float(cum[N-1])
        plt.axvline(N, linestyle="--", color="black", alpha=0.7)
        plt.scatter([N], [yN], zorder=5)
        plt.annotate(
            f"N* = {N}\nEV = {yN:.1f}",
            xy=(N, yN),
            xytext=(12, 12),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.6", alpha=0.95),
            arrowprops=dict(arrowstyle="->", lw=0.8, color="0.3"),
        )

    plt.tight_layout(); plt.savefig(out_png); plt.close()

def _uplift_hist(scores_uplift_csv: Path, out_png: Path):
    if not scores_uplift_csv.exists(): return
    df = pd.read_csv(scores_uplift_csv)
    if "uplift" not in df.columns or df["uplift"].dropna().empty: return
    plt.figure(figsize=(6,4))
    plt.hist(df["uplift"].dropna().values, bins=40)
    plt.xlabel("Predicted uplift"); plt.ylabel("Count")
    plt.title("Uplift distribution")
    plt.tight_layout(); plt.savefig(out_png); plt.close()

# ---------- entry ----------
def generate(out_dir: Path | str = "final_vi/out") -> Path:
    out_dir = Path(out_dir)
    plots = out_dir / "plots"; plots.mkdir(parents=True, exist_ok=True)

    # ROC/PR for best churn & outreach
    _roc_pr_curves(out_dir / "scores_best_churn.csv", "Churn", plots / "churn")
    _roc_pr_curves(out_dir / "scores_best_outreach.csv", "Outreach", plots / "outreach")

    # Baseline vs best + grouped bars
    baseline_auc = _maybe_read_baseline(out_dir)
    _baseline_vs_best(out_dir / "metrics_two_best.csv", plots / "baseline_vs_best.png", baseline_auc)
    best_rows = _load_best_rows(out_dir / "metrics_two_best.csv")
    _metrics_grouped_bars(best_rows, plots / "metrics_aprf.png")

    # Uplift EV curve (prefers eligible ranking) + hist
    _uplift_ev_curve(out_dir / "ranking_uplift_netvalue.csv",
                     out_dir / "recommended_n.json",
                     plots / "uplift_ev.png")
    _uplift_hist(out_dir / "scores_uplift.csv", plots / "uplift_hist.png")

    # header
    n_text = ""
    rec_json = out_dir / "recommended_n.json"
    if rec_json.exists():
        r = json.loads(rec_json.read_text())
        n_text = f"N*={r.get('n_star','?')}, Total EV={r.get('total_net_value','?')}, Avg/member={r.get('avg_net_value_per_member','?')}, B={r.get('benefit_per_saved','?')}, C={r.get('cost_per_outcare','?')}, uplift={r.get('uplift_algo_used','?')}, filter={r.get('eligibility_filter','?')}"

    html = f"""
    <html><head><meta charset="utf-8"><title>WellCo Report</title></head>
    <body>
      <h2>Model & Uplift Report</h2>
      <ul>
        <li><a href="../summary_compact.csv">summary_compact.csv</a></li>
        <li><a href="../metrics_two_best.csv">metrics_two_best.csv</a></li>
        <li><a href="../n_summary.csv">n_summary.csv</a> &nbsp; <a href="../recommended_n.json">recommended_n.json</a> — {n_text}</li>
        <li>Rankings:
          <ul>
            <li><a href="../ranking_uplift_netvalue__eligible.csv">ranking_uplift_netvalue__eligible.csv</a> (used for N)</li>
            <li><a href="../ranking_uplift__eligible.csv">ranking_uplift__eligible.csv</a></li>
            <li><a href="../ranking_uplift_netvalue.csv">ranking_uplift_netvalue.csv</a> (all)</li>
            <li><a href="../ranking_uplift.csv">ranking_uplift.csv</a> (all)</li>
          </ul>
        </li>
        <li><a href="../outreach_list_uplift_netvalue.csv">outreach_list_uplift_netvalue.csv</a></li>
      </ul>
      <h3>Baseline vs Best (AUC)</h3>
      <img src="baseline_vs_best.png" width="520">
      <h3>Validation metrics</h3>
      <img src="metrics_aprf.png" width="700">
      <h3>Churn</h3>
      <img src="churn.roc.png" width="420"> <img src="churn.pr.png" width="420">
      <h3>Outreach</h3>
      <img src="outreach.roc.png" width="420"> <img src="outreach.pr.png" width="420">
      <h3>Uplift</h3>
      <img src="uplift_ev.png" width="650"><br>
      <img src="uplift_hist.png" width="650">
    </body></html>
    """
    (plots / "index.html").write_text(html.strip(), encoding="utf-8")
    return plots
