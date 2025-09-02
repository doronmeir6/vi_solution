# final_vi/training/model_api.py
from __future__ import annotations
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score, accuracy_score
)
from sklearn.feature_selection import mutual_info_classif

try:
    from xgboost import XGBClassifier
    XGB_OK = True
except Exception:
    XGB_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False

try:
    import optuna
    OPTUNA_OK = True
except Exception:
    OPTUNA_OK = False

RANDOM_STATE = 42

def _ensure_treatment_col(df: pd.DataFrame) -> str:
    if "outreach" in df.columns: return "outreach"
    if "outcare"  in df.columns: return "outcare"
    raise KeyError("Treatment column not found (need 'outreach' or 'outcare').")

def _prep_for_target(df_feat: pd.DataFrame, target: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    drop_cols = ["member_id", "signup_date", "outreach", "outcare", "churn"]
    Xdf = df_feat.drop(columns=[c for c in drop_cols if c in df_feat.columns], errors="ignore")
    feature_names = Xdf.columns.to_list()
    X = np.nan_to_num(Xdf.values)
    y = df_feat[target].astype(int).values
    return X, y, feature_names

def _scale_pos_weight(y: np.ndarray) -> float:
    pos = float(np.sum(y == 1)); neg = float(np.sum(y == 0))
    return (neg / max(pos, 1.0))

def _clf_metrics_full(y_true: np.ndarray, scores: np.ndarray, thr: float = 0.5) -> Dict:
    yhat = (scores >= thr).astype(int)
    return {
        "auc_roc": float(roc_auc_score(y_true, scores)),
        "auc_pr":  float(average_precision_score(y_true, scores)),
        "accuracy": float(accuracy_score(y_true, yhat)),
        "precision": float(precision_score(y_true, yhat, zero_division=0)),
        "recall": float(recall_score(y_true, yhat, zero_division=0)),
        "f1": float(f1_score(y_true, yhat, zero_division=0)),
        "base_rate": float(np.mean(y_true)),
    }

def _metrics(y_true: np.ndarray, scores: np.ndarray, model_name: str, target: str) -> Dict:
    m = _clf_metrics_full(y_true, scores, thr=0.5)
    m.update({"target": target, "model": model_name})
    return m

def _feature_select(Xtr, Xva, feature_names, method="none", topk=150, ytr=None):
    if method == "none": return Xtr, Xva, feature_names
    if method == "topk":
        stds = np.std(Xtr, axis=0)
        idx = np.argsort(stds)[::-1][:min(topk, Xtr.shape[1])]
    elif method == "mi":
        if ytr is None: raise ValueError("feature_select(method='mi') requires ytr.")
        mi = mutual_info_classif(Xtr, ytr, discrete_features=False, random_state=RANDOM_STATE)
        idx = np.argsort(mi)[::-1][:min(topk, Xtr.shape[1])]
    else:
        return Xtr, Xva, feature_names
    return Xtr[:, idx], Xva[:, idx], [feature_names[i] for i in idx]

def _fit_lr(Xtr, ytr, Xva):
    sc = StandardScaler(with_mean=False)
    Xtr_s = sc.fit_transform(Xtr); Xva_s = sc.transform(Xva)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
    lr.fit(Xtr_s, ytr)
    pred = lambda Z: lr.predict_proba(sc.transform(Z))[:, 1]
    return {"name": "logreg", "predict": pred}

def _fit_xgb(Xtr, ytr, Xva, yva):
    if not XGB_OK: return None
    xgb = XGBClassifier(
        n_estimators=1000, max_depth=5, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="auc",
        n_jobs=4, random_state=RANDOM_STATE, scale_pos_weight=_scale_pos_weight(ytr),
    )
    xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    pred = lambda Z: xgb.predict_proba(Z)[:, 1]
    return {"name": "xgb", "predict": pred}

def _tune_lgbm_optuna(X: np.ndarray, y: np.ndarray, n_trials: int = 80, metric: str = "auc_roc") -> Dict:
    if not (LGB_OK and OPTUNA_OK):
        return {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def _cv_score(params) -> float:
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            Xtr, Xva = X[tr_idx], X[va_idx]
            ytr, yva = y[tr_idx], y[va_idx]
            model = lgb.LGBMClassifier(
                n_estimators=int(params["n_estimators"]),
                num_leaves=int(params["num_leaves"]),
                learning_rate=float(params["learning_rate"]),
                feature_fraction=float(params["feature_fraction"]),
                bagging_fraction=float(params["bagging_fraction"]),
                bagging_freq=int(params["bagging_freq"]),
                reg_lambda=float(params["reg_lambda"]),
                objective="binary",
                n_jobs=4,
                random_state=RANDOM_STATE,
                scale_pos_weight=_scale_pos_weight(ytr),
            )
            model.fit(Xtr, ytr)
            s = model.predict_proba(Xva)[:, 1]
            if metric == "auc_pr":
                from sklearn.metrics import average_precision_score
                scores.append(average_precision_score(yva, s))
            else:
                from sklearn.metrics import roc_auc_score
                scores.append(roc_auc_score(yva, s))
        return float(np.mean(scores))

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 3000, step=50),
            "num_leaves":   trial.suggest_int("num_leaves", 31, 255),
            "learning_rate":trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 5.0),
        }
        return _cv_score(params)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params

def _fit_lgbm(Xtr, ytr, Xva, yva, tuned_params: Dict | None):
    if not LGB_OK: return None
    base = {
        "n_estimators": 2000, "num_leaves": 63, "learning_rate": 0.02,
        "feature_fraction": 0.8, "bagging_fraction": 0.9, "bagging_freq": 1,
        "reg_lambda": 1.0, "objective": "binary", "n_jobs": 4, "random_state": RANDOM_STATE,
        "scale_pos_weight": _scale_pos_weight(ytr),
    }
    if tuned_params: base.update(tuned_params)
    lgbm = lgb.LGBMClassifier(**base)
    lgbm.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
             callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)])
    pred = lambda Z: lgbm.predict_proba(Z)[:, 1]
    return {"name": "lgbm", "predict": pred}

def train_best_for_target(
    features_df: pd.DataFrame,
    target: str,
    candidate_models: Tuple[str, ...] = ("logreg", "xgb", "lgbm"),
    feature_select: str = "mi",
    feature_topk: int = 200,
    validation_size: float = 0.30,
    random_state: int = RANDOM_STATE,
    tune_lgbm_trials: Optional[int] = 80,
    selection_metric: str = "auc_roc",
    resampler: str = "none",                   # "none"|"ros"|"smote"|"rus"
) -> Dict:
    X_all, y_all, feat_names = _prep_for_target(features_df, target)
    idx_all = np.arange(len(y_all))
    Xtr, Xva, ytr, yva, id_tr, id_va = train_test_split(
        X_all, y_all, idx_all, test_size=validation_size, random_state=random_state, stratify=y_all
    )

    # feature selection
    Xtr_sel, Xva_sel, sel_names = _feature_select(Xtr, Xva, feat_names, feature_select, feature_topk, ytr)

    # optional resampling
    if resampler != "none":
        try:
            if resampler == "ros":
                from imblearn.over_sampling import RandomOverSampler
                Xtr_sel, ytr = RandomOverSampler(random_state=random_state).fit_resample(Xtr_sel, ytr)
            elif resampler == "smote":
                from imblearn.over_sampling import SMOTE
                Xtr_sel, ytr = SMOTE(random_state=random_state).fit_resample(Xtr_sel, ytr)
            elif resampler == "rus":
                from imblearn.under_sampling import RandomUnderSampler
                Xtr_sel, ytr = RandomUnderSampler(random_state=random_state).fit_resample(Xtr_sel, ytr)
        except Exception:
            pass

    learners, metrics_rows = [], []

    if "logreg" in candidate_models:
        lr = _fit_lr(Xtr_sel, ytr, Xva_sel)
        s = lr["predict"](Xva_sel)
        learners.append(lr)
        metrics_rows.append(_metrics(yva, s, "logreg", target))

    tuned_lgbm_params = None
    if "lgbm" in candidate_models and tune_lgbm_trials:
        tuned_lgbm_params = _tune_lgbm_optuna(Xtr_sel, ytr, n_trials=int(tune_lgbm_trials), metric="auc_roc")
    if "lgbm" in candidate_models:
        lgbm = _fit_lgbm(Xtr_sel, ytr, Xva_sel, yva, tuned_lgbm_params)
        if lgbm is not None:
            s = lgbm["predict"](Xva_sel)
            learners.append(lgbm)
            metrics_rows.append(_metrics(yva, s, "lgbm", target))

    if "xgb" in candidate_models:
        xgb = _fit_xgb(Xtr_sel, ytr, Xva_sel, yva)
        if xgb is not None:
            s = xgb["predict"](Xva_sel)
            learners.append(xgb)
            metrics_rows.append(_metrics(yva, s, "xgb", target))

    if not learners:
        raise RuntimeError("No learners available.")

    def auc_of(learner):
        s = learner["predict"](Xva_sel)
        return float(roc_auc_score(yva, s))
    best = max(learners, key=auc_of)
    best_name = best["name"]; s_best = best["predict"](Xva_sel)

    metrics_df = pd.DataFrame(metrics_rows)
    member_ids_va = features_df["member_id"].iloc[id_va].values
    ranking_val = (pd.DataFrame({"member_id": member_ids_va, "score": s_best})
                   .sort_values("score", ascending=False).reset_index(drop=True))
    ranking_val["rank"] = np.arange(1, len(ranking_val) + 1)

    scores_val = pd.DataFrame({"member_id": member_ids_va, "y_true": yva, f"p_{best_name}": s_best})

    return {
        "best_name": best_name,
        "metrics": metrics_df,
        "ranking_val": ranking_val,
        "scores_val": scores_val,
        "predict_full": best["predict"],
        "feature_names_used": sel_names,
        "val_index": id_va,
        "y_val": yva,
        "_Xtr_sel": Xtr_sel, "_Xva_sel": Xva_sel, "_feat_names_sel": sel_names,
    }

def _algo_to_fit_fn(algo: str):
    algo = (algo or "").lower()
    if algo == "xgb" and XGB_OK: return "xgb"
    if algo == "lgbm" and LGB_OK: return "lgbm"
    if algo == "logreg": return "logreg"
    # fallbacks
    if XGB_OK: return "xgb"
    if LGB_OK: return "lgbm"
    return "logreg"

def build_uplift_tlearner(
    features_df: pd.DataFrame,
    prefer_model: str = "auto",              # "auto"|"logreg"|"lgbm"|"xgb"|"from_best"
    prefer_map: Optional[Dict[str,str]] = None,  # when from_best: {"churn": "logreg"/"lgbm"/"xgb"}
    benefit_per_saved: float = 100.0,
    cost_per_outcare: float = 10.0
) -> Dict[str, pd.DataFrame]:
    """
    T-learner uplift on churn: fit one model on treated, one on control, both predicting churn.
    If prefer_model=="from_best", we use prefer_map["churn"] for both heads (keeps algo aligned with selected best churn model).
    """
    tcol = _ensure_treatment_col(features_df)
    if "churn" not in features_df.columns:
        raise KeyError("'churn' column not found.")

    drop_cols = ["member_id","signup_date","outreach","outcare","churn"]
    Xdf = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns], errors="ignore")
    X = np.nan_to_num(Xdf.values)
    y = features_df["churn"].astype(int).values
    t = features_df[tcol].astype(int).values
    mid = features_df["member_id"].values

    if t.sum() == 0 or t.sum() == len(t):
        empty = pd.DataFrame({"member_id": mid, "p_treat": np.nan, "p_control": np.nan,
                              "uplift": np.nan, "net_value": np.nan})
        return {"scores": empty, "ranking_uplift": empty.head(0), "ranking_netvalue": empty.head(0)}

    # choose algorithm
    if prefer_model == "from_best" and prefer_map and "churn" in prefer_map:
        algo = _algo_to_fit_fn(prefer_map["churn"])
    elif prefer_model in {"logreg","lgbm","xgb"}:
        algo = _algo_to_fit_fn(prefer_model)
    else:
        algo = _algo_to_fit_fn("auto")

    def _fit_bin(Xtr, ytr):
        if algo == "xgb" and XGB_OK:
            m = XGBClassifier(n_estimators=1000, max_depth=5, learning_rate=0.03,
                              subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
                              objective="binary:logistic", eval_metric="auc",
                              n_jobs=4, random_state=RANDOM_STATE)
            m.fit(Xtr, ytr)
            return lambda Z: m.predict_proba(Z)[:,1]
        if algo == "lgbm" and LGB_OK:
            m = lgb.LGBMClassifier(n_estimators=2000, num_leaves=63, learning_rate=0.02,
                                   feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=1,
                                   reg_lambda=1.0, objective="binary", n_jobs=4, random_state=RANDOM_STATE)
            m.fit(Xtr, ytr)
            return lambda Z: m.predict_proba(Z)[:,1]
        sc = StandardScaler(with_mean=False); Xs = sc.fit_transform(Xtr)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
        lr.fit(Xs, ytr)
        return lambda Z: lr.predict_proba(sc.transform(Z))[:,1]

    p_t = _fit_bin(X[t==1], y[t==1])(X)
    p_c = _fit_bin(X[t==0], y[t==0])(X)
    uplift = p_c - p_t
    net_value = uplift * float(benefit_per_saved) - float(cost_per_outcare)

    scores = pd.DataFrame({"member_id": mid, "p_treat": p_t, "p_control": p_c,
                           "uplift": uplift, "net_value": net_value})
    ranking_u = scores.sort_values("uplift", ascending=False).reset_index(drop=True)
    ranking_u["rank"] = np.arange(1, len(ranking_u)+1)
    ranking_nv = scores.sort_values("net_value", ascending=False).reset_index(drop=True)
    ranking_nv["rank"] = np.arange(1, len(ranking_nv)+1)

    return {
        "scores": scores,
        "ranking_uplift": ranking_u[["member_id","uplift","rank"]],
        "ranking_netvalue": ranking_nv[["member_id","net_value","uplift","rank"]],
        "uplift_algo_used": algo
    }

def choose_n_from_netvalue(ranking_netvalue: pd.DataFrame,
                           cost_per_outcare: float,
                           benefit_per_saved: float,
                           n_fixed: Optional[int] = None,
                           max_total_cost: Optional[float] = None) -> Tuple[int, Dict, pd.DataFrame]:
    if ranking_netvalue.empty:
        return 0, {"note":"empty ranking_netvalue"}, ranking_netvalue
    df = ranking_netvalue.copy().sort_values("net_value", ascending=False).reset_index(drop=True)
    df["cum_net_value"] = df["net_value"].cumsum()

    if n_fixed is not None:
        n_star = int(max(0, min(n_fixed, len(df))))
    elif max_total_cost is not None:
        max_n_cost = int(min(len(df), np.floor(max_total_cost / float(cost_per_outcare))))
        n_star = int(df.loc[:max_n_cost-1, "cum_net_value"].idxmax() + 1) if max_n_cost>0 else 0
    else:
        n_star = int(df["cum_net_value"].idxmax() + 1)
        if df.loc[n_star-1, "cum_net_value"] <= 0: n_star = 0

    chosen = df.head(n_star).copy()
    summary = {
        "n_star": int(n_star),
        "total_net_value": float(chosen["net_value"].sum()) if n_star>0 else 0.0,
        "avg_net_value_per_member": float(chosen["net_value"].mean()) if n_star>0 else 0.0,
        "benefit_per_saved": float(benefit_per_saved),
        "cost_per_outcare": float(cost_per_outcare),
    }
    return n_star, summary, chosen

def train_two_best_models(
    features_df: pd.DataFrame,
    *,
    feature_select: str = "mi",
    feature_topk: int = 200,
    candidate_models: Tuple[str, ...] = ("logreg", "xgb", "lgbm"),
    tune_lgbm_trials: int | None = 80,
    resampler: str = "none",   # "none"|"ros"|"smote"|"rus"
) -> Dict[str, Dict]:
    """
    Train & select the best model for 'churn' and 'outreach' (by ROC-AUC on a validation split).
    Returns:
      {
        "churn": {
           "best_name": str,
           "metrics": pd.DataFrame,
           "ranking_val": pd.DataFrame,
           "scores_val": pd.DataFrame,
           "predict_full": callable,
           "feature_names_used": List[str],
           "val_index": np.ndarray,
           "y_val": np.ndarray,
        },
        "outreach": { ...same keys... }
      }
    """
    out: Dict[str, Dict] = {}
    for target in ("churn", "outreach"):
        out[target] = train_best_for_target(
            features_df=features_df,
            target=target,
            candidate_models=candidate_models,
            feature_select=feature_select,
            feature_topk=feature_topk,
            tune_lgbm_trials=tune_lgbm_trials,
            resampler=resampler,
        )
    return out
