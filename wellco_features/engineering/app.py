from __future__ import annotations
import numpy as np
import pandas as pd

def _sessionize(ts: pd.Series, gap_minutes: int = 30) -> pd.Series:
    dt = ts.sort_values()
    gap = dt.diff().dt.total_seconds().div(60).fillna(1e9)
    return (gap > gap_minutes).cumsum()

def _decay_sum(days_ago: pd.Series, half_life_days: float = 30.0) -> float:
    lam = np.log(2.0) / half_life_days
    w = np.exp(-lam * days_ago.clip(lower=0).astype(float))
    return float(w.sum())

def build_app_features(labels: pd.DataFrame, app: pd.DataFrame, now_ts: pd.Timestamp) -> pd.DataFrame:
    if app is None or app.empty:
        feat = pd.DataFrame({"member_id": labels["member_id"].values})
        feat["member_id"] = pd.to_numeric(feat["member_id"], errors="coerce").astype("Int64")
        return feat

    a = app.copy()
    a["timestamp"] = pd.to_datetime(a["timestamp"], errors="coerce")

    g = a.groupby("member_id")
    last_dt = g["timestamp"].max()
    events  = g.size()
    active_days = g["timestamp"].apply(lambda s: s.dt.date.nunique())

    a90 = a[a["timestamp"] >= now_ts - pd.Timedelta(days=90)].sort_values(["member_id","timestamp"])
    ses = a90.groupby("member_id")["timestamp"].apply(lambda s: _sessionize(s, gap_minutes=30))
    a90 = a90.assign(session_id=ses.values)
    ses_counts = a90.groupby(["member_id","session_id"]).size().rename("events_in_session")
    ses_per_member = ses_counts.groupby("member_id").size().rename("app_sessions_90d")
    events_per_ses = ses_counts.groupby("member_id").mean().rename("app_events_per_session_90d")

    feat = pd.DataFrame({"member_id": labels["member_id"].values})
    feat = feat.merge(events.rename("app_events"), on="member_id", how="left")
    feat = feat.merge(active_days.rename("app_active_days"), on="member_id", how="left")
    feat = feat.merge(ses_per_member, on="member_id", how="left")
    feat = feat.merge(events_per_ses, on="member_id", how="left")
    feat = feat.merge(last_dt.rename("app_last_dt"), on="member_id", how="left")
    feat["app_days_since_last"] = (now_ts - feat["app_last_dt"]).dt.days
    feat = feat.drop(columns=["app_last_dt"])

    a["days_ago"] = (now_ts - a["timestamp"]).dt.days
    decayed = a.groupby("member_id")["days_ago"].apply(lambda s: _decay_sum(s, 30))
    feat = feat.merge(decayed.rename("app_events_decayed_hl30"), on="member_id", how="left")

    for c in feat.columns:
        if c == "member_id": continue
        if c.endswith("_days_since_last"):
            mx = feat[c].max()
            feat[c] = feat[c].fillna((mx + 1) if pd.notna(mx) else 9999)
        else:
            feat[c] = feat[c].fillna(0)

    feat["member_id"] = pd.to_numeric(feat["member_id"], errors="coerce").astype("Int64")
    return feat
