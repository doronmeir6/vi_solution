from __future__ import annotations
import numpy as np
import pandas as pd

__all__ = ["combine_features", "FeatureCombiner"]

def _normalize_key(df: pd.DataFrame, key: str = "member_id") -> pd.DataFrame:
    df = df.copy()
    if key in df.columns:
        df[key] = pd.to_numeric(df[key], errors="coerce").astype("Int64")
    return df

def _safe_ratio(a: pd.Series, b: pd.Series) -> pd.Series:
    b = b.replace(0, np.nan)
    r = a / b
    return r.fillna(0.0)

def combine_features(labels: pd.DataFrame,
                     webF: pd.DataFrame,
                     appF: pd.DataFrame,
                     claimsF: pd.DataFrame,
                     now_ts) -> pd.DataFrame:
    labels = _normalize_key(labels); webF = _normalize_key(webF)
    appF   = _normalize_key(appF);   claimsF = _normalize_key(claimsF)

    df = labels.copy()
    for part in (webF, appF, claimsF):
        if part is not None and not part.empty:
            if df["member_id"].dtype != part["member_id"].dtype:
                raise TypeError(f"dtype mismatch: {df['member_id'].dtype} vs {part['member_id'].dtype}")
            df = df.merge(part, on="member_id", how="left")

    df["tenure_days"] = (now_ts - pd.to_datetime(df["signup_date"], errors="coerce")).dt.days

    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        if c.endswith("_days_since_last"):
            df[c] = df[c].fillna(df[c].max() + 1 if df[c].notna().any() else 9999)
        else:
            df[c] = df[c].fillna(0)

    if "app_events" in df.columns and "wv_visits" in df.columns:
        df["ratio_app_to_web"] = _safe_ratio(df["app_events"], df["wv_visits"])

    claims_total = df.filter(regex=r"^claims_n_\d+d$").sum(axis=1)
    if "app_events" in df.columns:
        df["ratio_claims_to_usage"] = _safe_ratio(claims_total, df["app_events"])
    elif "wv_visits" in df.columns:
        df["ratio_claims_to_usage"] = _safe_ratio(claims_total, df["wv_visits"])

    if {"web_visits_decayed_hl30","app_events_decayed_hl30"}.issubset(df.columns):
        z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
        df["engagement_balance"] = z(df["app_events_decayed_hl30"]) - z(df["web_visits_decayed_hl30"])

    if {"web_health_domain_visits_90d","web_medical_path_hits_90d"}.issubset(df.columns):
        z = lambda s: (s - s.mean()) / (s.std() + 1e-9)
        df["health_interest"] = z(df["web_health_domain_visits_90d"]) + z(df["web_medical_path_hits_90d"])

    if {"health_interest","engagement_balance"}.issubset(df.columns):
        df["health_x_balance"] = df["health_interest"] * df["engagement_balance"]

    return df

class FeatureCombiner:
    def __init__(self, cfg=None):
        self.cfg = cfg
    def combine(self, labels, webF, appF, claimsF, now_ts):
        return combine_features(labels, webF, appF, claimsF, now_ts)
