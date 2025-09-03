from __future__ import annotations

from pathlib import Path
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

# Optional libs (guarded)
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

# ---------- helpers ----------
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
    """Return a non-empty feature matrix. If selection gives 0 columns, fall back to no selection."""
    if method == "none" or Xtr.shape[1] == 0:
        return Xtr, Xva, feature_names

    if method == "topk":
        stds = np.std(Xtr, axis=0)
        idx = np.argsort(stds)[::-1][:min(topk, Xtr.shape[1])]
    elif method == "mi":
        if ytr is None:
            raise ValueError("feature_select(method='mi') requires ytr.")
        mi = mutual_info_classif(Xtr, ytr, discrete_features=False, random_state=RANDOM_STATE)
        idx = np.argsort(mi)[::-1][:min(topk, Xtr.shape[1])]
    else:
        return Xtr, Xva, feature_names

    if idx is None or len(idx) == 0:
        return Xtr, Xva, feature_names

    Xtr_sel, Xva_sel = Xtr[:, idx], Xva[:, idx]
    keep = np.where(np.nanstd(Xtr_sel, axis=0) > 0)[0]
    if len(keep) == 0:
        return Xtr, Xva, feature_names
    Xtr_sel, Xva_sel = Xtr_sel[:, keep], Xva_sel[:, keep]
    sel_names = [feature_names[i] for i in np.array(idx)[keep]]
    return Xtr_sel, Xva_sel, sel_names

def _best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Return (thr, f1_at_thr) maximizing F1 on given scores."""
    if len(scores) == 0:
        return 0.5, 0.0
    qs = np.linspace(0.01, 0.99, 99)
    thrs = np.unique(np.quantile(scores, qs))
    best_thr, best_f1 = 0.5, -1.0
    for t in thrs:
        yhat = (scores >= t).astype(int)
        f1 = f1_score(y_true, yhat, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, float(t)
    return best_thr, best_f1

# ---------- learners ----------
def _fit_lr(Xtr, ytr, Xva, tuned_params: Optional[Dict]=None):
    sc = StandardScaler(with_mean=False)
    Xtr_s = sc.fit_transform(Xtr); _ = sc.transform(Xva)
    params = dict(
        max_iter=1000, class_weight="balanced", solver="liblinear",
        random_state=RANDOM_STATE, C=1.0, penalty="l2"
    )
    if tuned_params:
        params.update({k: tuned_params[k] for k in ["C","penalty"] if k in tuned_params})
        params["solver"] = "liblinear"
    lr = LogisticRegression(**params)
    lr.fit(Xtr_s, ytr)
    def pred(Z): return lr.predict_proba(sc.transform(Z))[:, 1]
    return {"name": "logreg", "predict": pred, "model": lr, "scaler": sc}

def _fit_xgb(Xtr, ytr, Xva, yva, tuned_params: Optional[Dict]=None):
    if not XGB_OK: return None
    base = dict(
        n_estimators=1000, max_depth=5, learning_rate=0.03,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        objective="binary:logistic", eval_metric="auc",
        n_jobs=4, random_state=RANDOM_STATE, scale_pos_weight=_scale_pos_weight(ytr),
    )
    if tuned_params:
        base.update(tuned_params)
        base["scale_pos_weight"] = _scale_pos_weight(ytr)
    xgb = XGBClassifier(**base)
    try:
        xgb.fit(Xtr, ytr, eval_set=[(Xva, yva)], early_stopping_rounds=100, verbose=False)
    except Exception:
        xgb.fit(Xtr, ytr)
    def pred(Z): return xgb.predict_proba(Z)[:, 1]
    return {"name": "xgb", "predict": pred, "model": xgb, "scaler": None}

def _fit_lgbm(Xtr, ytr, Xva, yva, tuned_params: Optional[Dict]=None):
    """Kept for compatibility if you still include 'lgbm' in candidate_models."""
    if not LGB_OK: return None
    if Xtr.shape[1] == 0 or np.all(np.nanstd(Xtr, axis=0) == 0): return None
    base = dict(
        n_estimators=2000, num_leaves=31, learning_rate=0.02,
        feature_fraction=0.8, bagging_fraction=0.9, bagging_freq=1,
        reg_lambda=1.0, min_data_in_leaf=max(5, int(0.01 * Xtr.shape[0])),
        min_split_gain=0.0, max_bin=255, feature_pre_filter=False,
        objective="binary", n_jobs=4, random_state=RANDOM_STATE,
        scale_pos_weight=_scale_pos_weight(ytr), verbosity=-1,
    )
    if tuned_params: base.update(tuned_params)
    lgbm = lgb.LGBMClassifier(**base)
    lgbm.fit(
        Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                   lgb.log_evaluation(period=0)]
    )
    def pred(Z): return lgbm.predict_proba(Z)[:, 1]
    return {"name": "lgbm", "predict": pred, "model": lgbm, "scaler": None}

# ---------- Optuna tuners ----------
def _cv_score_metric(y_true, scores, metric: str) -> float:
    if metric == "f1":
        _, f1b = _best_f1_threshold(y_true, scores)
        return float(f1b)
    if metric == "auc_pr":
        return float(average_precision_score(y_true, scores))
    return float(roc_auc_score(y_true, scores))

def _tune_logreg_optuna(X: np.ndarray, y: np.ndarray, n_trials: int = 60, metric: str = "auc_pr") -> Dict:
    if not OPTUNA_OK:
        return {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2"])
        scores = []
        for tr, va in skf.split(X, y):
            Xtr, Xva = X[tr], X[va]; ytr, yva = y[tr], y[va]
            sc = StandardScaler(with_mean=False)
            Xtr_s = sc.fit_transform(Xtr); Xva_s = sc.transform(Xva)
            lr = LogisticRegression(
                C=C, penalty=penalty, solver="liblinear",
                max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
            )
            lr.fit(Xtr_s, ytr)
            s = lr.predict_proba(Xva_s)[:, 1]
            scores.append(_cv_score_metric(yva, s, metric))
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    return {"C": float(study.best_params["C"]), "penalty": str(study.best_params["penalty"])}

def _tune_xgb_optuna(X: np.ndarray, y: np.ndarray, n_trials: int = 60, metric: str = "auc_pr") -> Dict:
    if not (OPTUNA_OK and XGB_OK):
        return {}
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)

    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 2000, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 8),
            learning_rate=trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 0.0, 5.0),
            min_child_weight=trial.suggest_float("min_child_weight", 1.0, 10.0),
            gamma=trial.suggest_float("gamma", 0.0, 5.0),
        )
        vals = []
        for tr, va in skf.split(X, y):
            Xtr, Xva = X[tr], X[va]; ytr, yva = y[tr], y[va]
            model = XGBClassifier(
                **params,
                objective="binary:logistic", eval_metric="auc",
                n_jobs=4, random_state=RANDOM_STATE,
                scale_pos_weight=_scale_pos_weight(ytr),
            )
            model.fit(Xtr, ytr)
            s = model.predict_proba(Xva)[:, 1]
            vals.append(_cv_score_metric(yva, s, metric))
        return float(np.mean(vals))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=int(n_trials), show_progress_bar=False)
    best = study.best_params.copy()
    best["n_estimators"] = int(best["n_estimators"])
    best["max_depth"] = int(best["max_depth"])
    return best

def _tune_lgbm_optuna(X: np.ndarray, y: np.ndarray, n_trials: int = 80, metric: str = "auc_roc") -> Dict:
    if not (LGB_OK and OPTUNA_OK):
        return {}
    from sklearn.metrics import average_precision_score as aps
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
                min_split_gain=0.0,
                min_data_in_leaf=max(5, int(0.01*Xtr.shape[0])),
                feature_pre_filter=False,
                verbosity=-1,
            )
            model.fit(Xtr, ytr)
            s = model.predict_proba(Xva)[:, 1]
            scores.append(aps(yva, s) if metric=="auc_pr" else roc_auc_score(yva, s))
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

# ---------- training APIs ----------
def train_best_for_target(
    features_df: pd.DataFrame,
    target: str,
    candidate_models: Tuple[str, ...] = ("logreg", "xgb", "lgbm"),
    feature_select: str = "mi",
    feature_topk: int = 200,
    validation_size: float = 0.30,
    random_state: int = RANDOM_STATE,
    tune_lgbm_trials: Optional[int] = 80,
    selection_metric: str = "auc_pr",              # "auc_pr" or "f1"
    resampler: str = "none",
    tune_logreg_trials: Optional[int] = 60,        # NEW
    tune_xgb_trials: Optional[int] = 60,           # NEW
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

    # Optuna tuning
    tuned_lr_params = None
    if "logreg" in candidate_models and tune_logreg_trials:
        try:
            tuned_lr_params = _tune_logreg_optuna(Xtr_sel, ytr, n_trials=int(tune_logreg_trials), metric=selection_metric)
        except Exception:
            tuned_lr_params = None

    tuned_xgb_params = None
    if "xgb" in candidate_models and tune_xgb_trials:
        try:
            tuned_xgb_params = _tune_xgb_optuna(Xtr_sel, ytr, n_trials=int(tune_xgb_trials), metric=selection_metric)
        except Exception:
            tuned_xgb_params = None

    tuned_lgbm_params = None
    if "lgbm" in candidate_models and tune_lgbm_trials:
        try:
            tuned_lgbm_params = _tune_lgbm_optuna(Xtr_sel, ytr, n_trials=int(tune_lgbm_trials), metric="auc_pr")
        except Exception:
            tuned_lgbm_params = None

    # LOGREG
    if "logreg" in candidate_models:
        lr = _fit_lr(Xtr_sel, ytr, Xva_sel, tuned_lr_params)
        s = lr["predict"](Xva_sel)
        learners.append(lr)
        metrics_rows.append(_metrics(yva, s, "logreg", target))
        thr, f1b = _best_f1_threshold(yva, s)
        thr_logreg, f1_logreg = thr, f1b
    else:
        thr_logreg, f1_logreg = None, None

    # LGBM (optional)
    if "lgbm" in candidate_models:
        lgbm = _fit_lgbm(Xtr_sel, ytr, Xva_sel, yva, tuned_lgbm_params)
        if lgbm is not None:
            s = lgbm["predict"](Xva_sel)
            learners.append(lgbm)
            metrics_rows.append(_metrics(yva, s, "lgbm", target))

    # XGB
    if "xgb" in candidate_models:
        xgb = _fit_xgb(Xtr_sel, ytr, Xva_sel, yva, tuned_xgb_params)
        if xgb is not None:
            s = xgb["predict"](Xva_sel)
            learners.append(xgb)
            metrics_rows.append(_metrics(yva, s, "xgb", target))

    if not learners:
        raise RuntimeError("No learners available.")

    # pick best by selection_metric ("auc_pr" or "f1")
    def score_of(learner):
        s = learner["predict"](Xva_sel)
        if selection_metric == "f1":
            _, f1b = _best_f1_threshold(yva, s)
            return f1b
        return float(average_precision_score(yva, s))

    best = max(learners, key=score_of)
    best_name = best["name"]; s_best = best["predict"](Xva_sel)

    # threshold calibrated by F1 for the chosen model (diagnostic)
    thr_best, f1_best = _best_f1_threshold(yva, s_best)

    metrics_df = pd.DataFrame(metrics_rows)
    member_ids_va = features_df["member_id"].iloc[id_va].values
    ranking_val = (pd.DataFrame({"member_id": member_ids_va, "score": s_best})
                   .sort_values("score", ascending=False).reset_index(drop=True))
    ranking_val["rank"] = np.arange(1, len(ranking_val) + 1)

    scores_val = pd.DataFrame({"member_id": member_ids_va, "y_true": yva, f"p_{best_name}": s_best})

    # full-predict on RAW features (uses selected features & scaler if needed)
    sel_names = sel_names or []
    feat_names_all = features_df.drop(columns=["member_id","signup_date","outreach","outcare","churn"], errors="ignore").columns.tolist()

    def predict_full_raw(Z_raw: np.ndarray) -> np.ndarray:
        Zs = Z_raw[:, [feat_names_all.index(n) for n in sel_names]] if sel_names else Z_raw
        if best.get("scaler", None) is not None:
            Zs = best["scaler"].transform(Zs)
        return best["model"].predict_proba(Zs)[:, 1]

    return {
        "best_name": best_name,
        "metrics": metrics_df,
        "ranking_val": ranking_val,
        "scores_val": scores_val,
        "predict_full": best["predict"],           # expects selected features
        "predict_full_raw": predict_full_raw,      # works on raw features
        "feature_names_used": sel_names,
        "val_index": id_va,
        "y_val": yva,
        "thr_f1": float(thr_best),
        "f1_at_thr": float(f1_best),
    }

def train_two_best_models(
    features_df: pd.DataFrame,
    *,
    feature_select: str = "mi",
    feature_topk: int = 200,
    candidate_models: Tuple[str, ...] = ("logreg", "xgb", "lgbm"),
    tune_lgbm_trials: int | None = 80,
    resampler: str = "none",
    selection_metric: str = "auc_pr",
    tune_logreg_trials: Optional[int] = 60,
    tune_xgb_trials: Optional[int] = 60,
) -> Dict[str, Dict]:
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
            selection_metric=selection_metric,
            tune_logreg_trials=tune_logreg_trials,
            tune_xgb_trials=tune_xgb_trials,
        )
    return out

# ---------- uplift & policy scoring (with soft-penalty + mission bonus) ----------
def _algo_to_fit_fn(algo: str):
    algo = (algo or "").lower()
    if algo == "xgb" and XGB_OK: return "xgb"
    if algo == "lgbm" and LGB_OK: return "lgbm"
    if algo == "logreg": return "logreg"
    if XGB_OK: return "xgb"
    if LGB_OK: return "lgbm"
    return "logreg"

def build_uplift_tlearner(
    features_df: pd.DataFrame,
    prefer_model: str = "auto",              # "auto"|"logreg"|"lgbm"|"xgb"|"from_best"
    prefer_map: Optional[Dict[str,str]] = None,  # when from_best: {"churn": "logreg"/"lgbm"/"xgb"}
    benefit_per_saved: float = 100.0,
    cost_per_outcare: float = 10.0,
    gate_by_scores: Optional[Dict] = None,   # Optional hard gate (kept for compatibility)
    soft_penalty: Optional[Dict[str, float]] = None,  # {"alpha":..., "beta":..., "gamma_mission":...}
) -> Dict[str, pd.DataFrame]:
    """
    policy_net_value = uplift*B - C * (1 + alpha*p_out + beta*(1-p_ch)) + gamma_mission*B*(p_ch*(1-p_out) + (1-p_ch)*p_out)
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
            params = dict(
                n_estimators=1500, num_leaves=31, learning_rate=0.03,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=1,
                reg_lambda=1.0, min_data_in_leaf=max(5, int(0.01 * Xtr.shape[0])),
                min_split_gain=0.0, max_bin=255, feature_pre_filter=False,
                objective="binary", n_jobs=4, random_state=RANDOM_STATE, verbosity=-1,
            )
            m = lgb.LGBMClassifier(**params)
            m.fit(Xtr, ytr, callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False),
                                       lgb.log_evaluation(period=0)])
            return lambda Z: m.predict_proba(Z)[:, 1]
        sc = StandardScaler(with_mean=False); Xs = sc.fit_transform(Xtr)
        lr = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=RANDOM_STATE)
        lr.fit(Xs, ytr)
        return lambda Z: lr.predict_proba(sc.transform(Z))[:,1]

    # T-learner probs
    p_t = _fit_bin(X[t==1], y[t==1])(X)
    p_c = _fit_bin(X[t==0], y[t==0])(X)
    uplift = p_c - p_t

    # Base net value
    B = float(benefit_per_saved)
    C = float(cost_per_outcare)
    net_value = uplift * B - C

    # Need p_churn/p_outreach for policy; if not supplied via gate_by_scores, default to neutral
    if gate_by_scores is not None:
        p_out = np.asarray(gate_by_scores.get("p_outreach", np.zeros(len(uplift))))
        p_ch  = np.asarray(gate_by_scores.get("p_churn", np.ones(len(uplift))))
    else:
        # If you want strict model-based probs, pass them in via gate_by_scores from orchestrator
        p_out = np.zeros(len(uplift))
        p_ch  = np.ones(len(uplift))

    # Optional SOFT PENALTY + MISSION BONUS
    if soft_penalty is not None:
        alpha = float(soft_penalty.get("alpha", 0.0))
        beta  = float(soft_penalty.get("beta", 0.0))
        gamma = float(soft_penalty.get("gamma_mission", 0.0))
        # XOR-like mission preference: high when (1,0) or (0,1)
        mission_pref = (p_ch * (1.0 - p_out)) + ((1.0 - p_ch) * p_out)
        net_value_policy = (uplift * B
                            - C * (1.0 + alpha * p_out + beta * (1.0 - p_ch))
                            + gamma * B * mission_pref)
    else:
        net_value_policy = net_value.copy()

    scores = pd.DataFrame({
        "member_id": mid,
        "p_treat": p_t,
        "p_control": p_c,
        "uplift": uplift,
        "net_value": net_value,
        "policy_net_value": net_value_policy
    })

    # Optional hard gate by calibrated scores
    if gate_by_scores is not None and ("thr_churn" in gate_by_scores or "thr_outreach" in gate_by_scores):
        thr_c = float(gate_by_scores.get("thr_churn", 0.5))
        thr_o = float(gate_by_scores.get("thr_outreach", 0.5))
        mode  = str(gate_by_scores.get("mode", "and")).lower()
        if mode == "mission_union":
            mask = (((p_ch >= thr_c) & (p_out <= thr_o)) |
                    ((p_ch <  thr_c) & (p_out >= thr_o)))
        else:
            mask = ((p_ch >= thr_c) & (p_out <= thr_o))
        scores = scores.loc[mask].reset_index(drop=True)

    ranking_u = scores.sort_values("uplift", ascending=False).reset_index(drop=True)
    ranking_u["rank"] = np.arange(1, len(ranking_u)+1)
    ranking_nv = scores.sort_values("policy_net_value", ascending=False).reset_index(drop=True)
    ranking_nv["rank"] = np.arange(1, len(ranking_nv)+1)

    return {
        "scores": scores,
        "ranking_uplift": ranking_u[["member_id","uplift","rank"]],
        "ranking_netvalue": ranking_nv[["member_id","policy_net_value","net_value","uplift","rank"]],
        "uplift_algo_used": algo
    }

# ---------- N selection ----------
def choose_n_from_policy_netvalue(
    ranking_netvalue: pd.DataFrame,
    *,
    use_column: str = "policy_net_value",
    n_fixed: Optional[int] = None,
    cost_per_outcare: Optional[float] = None,
    max_total_cost: Optional[float] = None
) -> Tuple[int, Dict, pd.DataFrame]:
    """
    Pick N by argmax of cumulative sum of `use_column` (default: policy_net_value).
    If n_fixed is given, just take top-n_fixed.
    If max_total_cost is set and cost_per_outcare>0, cap N so that N*C <= budget, then argmax within.
    """
    if ranking_netvalue.empty:
        return 0, {"note": "empty ranking_netvalue"}, ranking_netvalue

    df = ranking_netvalue.copy().sort_values(use_column, ascending=False).reset_index(drop=True)
    df["cum_policy"] = df[use_column].cumsum()

    if n_fixed is not None:
        k = max(0, min(int(n_fixed), len(df)))
        chosen = df.head(k).copy()
        summary = {
            "n_star": int(k),
            "total_policy_value": float(chosen[use_column].sum()),
            "avg_policy_value": float(chosen[use_column].mean()) if k>0 else 0.0
        }
        return int(k), summary, chosen

    # Budget mode
    if max_total_cost is not None and cost_per_outcare and cost_per_outcare > 0:
        max_n = int(min(len(df), np.floor(max_total_cost / float(cost_per_outcare))))
        if max_n <= 0:
            return 0, {"n_star":0, "total_policy_value":0.0, "avg_policy_value":0.0}, df.head(0)
        sub = df.head(max_n).copy()
        idx = int(sub["cum_policy"].idxmax())
        n_star = int(idx + 1)
        chosen = sub.head(n_star).copy()
        summary = {
            "n_star": n_star,
            "total_policy_value": float(chosen[use_column].sum()),
            "avg_policy_value": float(chosen[use_column].mean()) if n_star>0 else 0.0
        }
        return n_star, summary, chosen

    # Standard EV argmax
    idx = int(df["cum_policy"].idxmax())
    n_star = int(idx + 1)
    if df.loc[idx, "cum_policy"] <= 0:
        n_star = 0
    chosen = df.head(n_star).copy()
    summary = {
        "n_star": n_star,
        "total_policy_value": float(chosen[use_column].sum()) if n_star>0 else 0.0,
        "avg_policy_value": float(chosen[use_column].mean()) if n_star>0 else 0.0
    }
    return n_star, summary, chosen
# -------------------------------------------------------------------
# Backwards-compatibility shims (so older code keeps working)
# -------------------------------------------------------------------
from typing import Optional, Dict
import numpy as np
import pandas as pd

def build_uplift_tlearner_with_threshold_gate(
    features_df: pd.DataFrame,
    *,
    churn_result: Dict,
    outreach_result: Dict,
    benefit_per_saved: float = 150.0,
    cost_per_outcare: float = 30.0,
    gate_mode: str = "and",                # "and" | "mission_union"
    thr_churn: Optional[float] = None,     # if None, uses churn_result["thr_f1"] or 0.5
    thr_outreach: Optional[float] = None,  # if None, uses outreach_result["thr_f1"] or 0.5
    use_soft_penalty: bool = True,
    soft_alpha: float = 0.3,
    soft_beta: float = 0.2,
    soft_gamma_mission: float = 0.3,
) -> Dict[str, pd.DataFrame]:
    """
    Compatibility wrapper that reproduces the old API by:
      1) computing p_churn and p_outreach for ALL members,
      2) applying an optional hard gate (thr_churn, thr_outreach, mode),
      3) applying the soft penalty + mission bonus inside policy_net_value.
    """
    # Build raw feature matrix Z for predict_full_raw
    drop_cols = ["member_id", "signup_date", "outreach", "outcare", "churn"]
    Xdf = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns], errors="ignore")
    Z = np.nan_to_num(Xdf.values)

    # Get probabilities from the chosen best models
    if "predict_full_raw" in churn_result:
        p_ch = churn_result["predict_full_raw"](Z)
    else:
        # last resort: try selected-feature predictor (expects selected columns)
        p_ch = churn_result["predict_full"](Z)
    if "predict_full_raw" in outreach_result:
        p_out = outreach_result["predict_full_raw"](Z)
    else:
        p_out = outreach_result["predict_full"](Z)

    # Default thresholds from F1 calibration if not supplied
    if thr_churn is None:
        thr_churn = float(churn_result.get("thr_f1", 0.5))
    if thr_outreach is None:
        thr_outreach = float(outreach_result.get("thr_f1", 0.5))

    gate_by_scores = {
        "p_churn": p_ch,
        "p_outreach": p_out,
        "thr_churn": float(thr_churn),
        "thr_outreach": float(thr_outreach),
        "mode": str(gate_mode),
    }
    soft_penalty = ({"alpha": float(soft_alpha),
                     "beta": float(soft_beta),
                     "gamma_mission": float(soft_gamma_mission)}
                    if use_soft_penalty else None)

    prefer_map = {"churn": churn_result.get("best_name", "logreg")}
    return build_uplift_tlearner(
        features_df=features_df,
        prefer_model="from_best",
        prefer_map=prefer_map,
        benefit_per_saved=benefit_per_saved,
        cost_per_outcare=cost_per_outcare,
        gate_by_scores=gate_by_scores,
        soft_penalty=soft_penalty,
    )


def choose_n_from_netvalue(
    ranking_netvalue: pd.DataFrame,
    cost_per_outcare: float,
    benefit_per_saved: float,
    n_fixed: Optional[int] = None,
    max_total_cost: Optional[float] = None,
) -> tuple[int, Dict, pd.DataFrame]:
    """
    Compatibility wrapper. If 'policy_net_value' exists, select N* on that;
    otherwise fall back to plain 'net_value'.
    """
    use_col = "policy_net_value" if "policy_net_value" in ranking_netvalue.columns else "net_value"
    return choose_n_from_policy_netvalue(
        ranking_netvalue,
        use_column=use_col,
        n_fixed=n_fixed,
        cost_per_outcare=cost_per_outcare,
        max_total_cost=max_total_cost,
    )
# -------------------------------------------------------------------
# Legacy compatibility: tune_threshold_gate_for_ev
# -------------------------------------------------------------------
def tune_threshold_gate_for_ev(
    features_df: pd.DataFrame,
    *,
    churn_result: Dict,
    outreach_result: Dict,
    benefit_per_saved: float = 150.0,
    cost_per_outcare: float = 30.0,
    thr_grid_churn: Optional[List[float]] = None,
    thr_grid_outreach: Optional[List[float]] = None,
    mode: str = "and",               # "and" | "mission_union"
    use_soft_penalty: bool = True,
    soft_alpha: float = 0.3,
    soft_beta: float = 0.2,
    soft_gamma_mission: float = 0.3,
    save_grid_path: Optional[str] = None,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Grid-search thresholds on (p_churn, p_outreach) to maximize EV with penalty-aware policy.
    Returns (best_summary_dict, full_grid_df).

    best_summary contains:
      - thr_churn, thr_outreach, mode
      - eligible_pool_size
      - n_star (argmax cumulative policy value within the eligible subset)
      - ev_total (policy EV at N*)
    """

    # --- 1) Pre-compute probabilities for ALL members
    drop_cols = ["member_id", "signup_date", "outreach", "outcare", "churn"]
    Xdf = features_df.drop(columns=[c for c in drop_cols if c in features_df.columns], errors="ignore")
    Z = np.nan_to_num(Xdf.values)

    if "predict_full_raw" in churn_result:
        p_ch = churn_result["predict_full_raw"](Z)
    else:
        p_ch = churn_result["predict_full"](Z)

    if "predict_full_raw" in outreach_result:
        p_out = outreach_result["predict_full_raw"](Z)
    else:
        p_out = outreach_result["predict_full"](Z)

    # --- 2) Pre-compute uplift & base net_value once (no gate at this step)
    base = build_uplift_tlearner(
        features_df=features_df,
        prefer_model="from_best",
        prefer_map={"churn": churn_result.get("best_name", "logreg")},
        benefit_per_saved=benefit_per_saved,
        cost_per_outcare=cost_per_outcare,
        gate_by_scores={"p_churn": p_ch, "p_outreach": p_out},  # passed for shape; no thresholds here
        soft_penalty=None,   # raw uplift / net_value; penalty applied below per gate
    )
    sc = base["scores"][["member_id", "uplift", "net_value"]].copy()
    sc["p_churn"] = p_ch
    sc["p_outreach"] = p_out

    B = float(benefit_per_saved)
    C = float(cost_per_outcare)

    def policy_value(u, pc, po, alpha, beta, gamma):
        # penalty + mission bonus (prefers (1,0) and (0,1))
        mission_pref = pc * (1.0 - po) + (1.0 - pc) * po
        return (u * B
                - C * (1.0 + alpha * po + beta * (1.0 - pc))
                + gamma * B * mission_pref)

    # --- 3) Threshold grids
    if thr_grid_churn is None:
        thr_grid_churn = [0.30, 0.40, 0.50, 0.60, 0.70]
    if thr_grid_outreach is None:
        thr_grid_outreach = [0.30, 0.40, 0.50, 0.60, 0.70]

    rows = []
    for tc in thr_grid_churn:
        for to in thr_grid_outreach:
            if mode == "mission_union":
                mask = (((sc["p_churn"] >= tc) & (sc["p_outreach"] <= to)) |
                        ((sc["p_churn"] <  tc) & (sc["p_outreach"] >  to)))
            else:  # "and"
                mask = ((sc["p_churn"] >= tc) & (sc["p_outreach"] <= to))

            sub = sc.loc[mask].copy()
            n_elig = len(sub)
            if n_elig == 0:
                rows.append({
                    "thr_churn": tc, "thr_outreach": to, "mode": mode,
                    "eligible_pool_size": 0, "n_star": 0, "ev_total": -np.inf
                })
                continue

            if use_soft_penalty:
                pv = policy_value(sub["uplift"].values,
                                  sub["p_churn"].values,
                                  sub["p_outreach"].values,
                                  soft_alpha, soft_beta, soft_gamma_mission)
            else:
                pv = (sub["uplift"].values * B - C)

            order = np.argsort(-pv)
            pv_sorted = pv[order]
            cum = np.cumsum(pv_sorted)
            idx = int(np.argmax(cum))
            n_star = int(idx + 1)
            ev_total = float(cum[idx])

            rows.append({
                "thr_churn": tc,
                "thr_outreach": to,
                "mode": mode,
                "eligible_pool_size": int(n_elig),
                "n_star": n_star,
                "ev_total": ev_total,
            })

    grid = pd.DataFrame(rows).sort_values("ev_total", ascending=False).reset_index(drop=True)
    best = grid.iloc[0].to_dict()

    if save_grid_path:
        try:
            Path(save_grid_path).parent.mkdir(parents=True, exist_ok=True)
            grid.to_csv(save_grid_path, index=False)
        except Exception:
            pass

    return best, grid
