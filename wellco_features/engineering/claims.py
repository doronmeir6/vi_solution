from __future__ import annotations
import numpy as np
import pandas as pd

def _icd_norm(s: pd.Series) -> pd.Series:
    return s.astype(str).str.upper().str.replace(".", "", regex=False).str.strip()

def build_claim_features(labels: pd.DataFrame, claims: pd.DataFrame, now_ts: pd.Timestamp) -> pd.DataFrame:
    if claims is None or claims.empty:
        feat = pd.DataFrame({"member_id": labels["member_id"].values})
        feat["member_id"] = pd.to_numeric(feat["member_id"], errors="coerce").astype("Int64")
        return feat

    c = claims.copy()
    c["icd_code"] = c.get("icd_code", "").astype(str)
    c["icd_norm"] = _icd_norm(c["icd_code"])
    c["icd_letter"] = c["icd_norm"].str[:1]

    g = c.groupby("member_id")
    last_dt = g["diagnosis_date"].max()
    uniq_days = g["diagnosis_date"].apply(lambda s: s.dt.date.nunique())

    n_90  = c[c["diagnosis_date"] >= now_ts - pd.Timedelta(days=90)].groupby("member_id").size()
    n_180 = c[c["diagnosis_date"] >= now_ts - pd.Timedelta(days=180)].groupby("member_id").size()
    n_365 = c[c["diagnosis_date"] >= now_ts - pd.Timedelta(days=365)].groupby("member_id").size()

    letter_counts = c.pivot_table(index="member_id", columns="icd_letter",
                                  values="icd_norm", aggfunc="count").fillna(0.0)
    if isinstance(letter_counts.columns, pd.Index):
        letter_counts.columns = [f"icd_{col}_cnt" for col in letter_counts.columns]

    flags = pd.DataFrame({"member_id": c["member_id"]})
    flags["cohort_T2D_E11x"]  = c["icd_norm"].str.startswith("E11").astype(int)
    flags["cohort_HTN_I10"]   = c["icd_norm"].str.startswith("I10").astype(int)
    flags["cohort_DIET_Z713"] = c["icd_norm"].str.startswith("Z713").astype(int)
    flags = flags.groupby("member_id").max()

    feat = pd.DataFrame({"member_id": labels["member_id"].values})
    feat = feat.merge(last_dt.rename("claims_last_dt"), on="member_id", how="left")
    feat = feat.merge(n_90.rename("claims_n_90d"),   on="member_id", how="left")
    feat = feat.merge(n_180.rename("claims_n_180d"), on="member_id", how="left")
    feat = feat.merge(n_365.rename("claims_n_365d"), on="member_id", how="left")
    feat = feat.merge(uniq_days.rename("claims_unique_days"), on="member_id", how="left")
    feat = feat.merge(letter_counts, on="member_id", how="left")
    feat = feat.merge(flags, on="member_id", how="left")

    feat["claims_days_since_last"] = (now_ts - feat["claims_last_dt"]).dt.days
    feat = feat.drop(columns=["claims_last_dt"])

    num_cols = feat.select_dtypes(include=[np.number]).columns
    for ccol in num_cols:
        if ccol.endswith("_days_since_last"):
            feat[ccol] = feat[ccol].fillna(feat[ccol].max() + 1 if feat[ccol].notna().any() else 9999)
        else:
            feat[ccol] = feat[ccol].fillna(0)

    feat["member_id"] = pd.to_numeric(feat["member_id"], errors="coerce").astype("Int64")
    return feat
