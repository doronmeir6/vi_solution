# WellCo – Churn & Outreach Models + Uplift (End-to-End)

This project trains **two binary models** (one for **churn**, one for **outreach/propensity**) and then builds a **cost-aware uplift ranking** to decide **who to contact** and **how many (N)** to contact under a benefit/cost policy.

## What’s included
**Feature engineering** (claims + app + web), with:
  *   true sessionization (30-min gap),
  *   recency-weighted (half-life 30d) engagement,
  *   medical intent on web (health domains + path/query patterns),
  *   cross-features (`engagement_balance`, `health_interest`, etc.).
**Model selection** per target:
  * Candidates: **LogisticRegression**, **LightGBM** (tuned via **Optuna**), **XGBoost**.
  *Best model is chosen by ROC-AUC** on a validation split.
**Uplift (T-learner)**:
  * Predict churn separately on treated vs control; uplift = `P(churn|control) - P(churn|treat)`.
**Auto mode uses the *same algorithm type* as your selected best churn model** (`from_best`), ensuring uplift is aligned with how the single-task model wins.
**N selection** (cost-aware):
  For benefit **B** and cost **C**, we sort by **net_value = B·uplift − C** and choose **N\*** that maximizes cumulative net value.
**Outputs**:
  * `summary_compact.csv` — one table with **best model per target**, **AUC/PR-AUC/Accuracy/Precision/Recall/F1**, plus **baseline AUC** and **Δ vs baseline** (if `auc_baseline.txt` is present).
  * `n_summary.csv` / `recommended_n.json` — **N\***, total/avg net value, **B**, **C**, and **which uplift algorithm** was used.
  * `outreach_list_uplift_netvalue.csv` — final **top-N** members to contact.
  * `out/plots/index.html` — open in browser to see **all charts** (ROC/PR, Baseline vs Best, uplift EV curve, uplift histogram).

## Install
```bash
python -m venv .venv
# (or conda create -n wellco python=3.10)
. .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
