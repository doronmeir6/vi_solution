from __future__ import annotations
import numpy as np
import pandas as pd
import re

BRIEF_BUCKETS = {
    "web_nutrition_90d": ["nutrition","balanced diet","healthy eating","mediterranean","fiber","high fiber","weight","weight loss","bmi","cholesterol","lipid"],
    "web_diabetes_90d":  ["diabetes","blood glucose","glucose","insulin","glycemic","hba1c","a1c"],
    "web_bp_90d":        ["hypertension","blood pressure","bp","systolic","diastolic","heart health","cardiometabolic"],
    "web_activity_90d":  ["exercise","aerobic","strength training","cardio","cardiovascular fitness","movement"],
    "web_sleep_90d":     ["sleep","sleep hygiene","restorative sleep","sleep quality","apnea"],
    "web_stress_90d":    ["stress","stress management","mindfulness","meditation","wellbeing","mental health","resilience"],
}

MEDICAL_DOMAINS = {
    "nih.gov","ncbi.nlm.nih.gov","cdc.gov","who.int","mayoclinic.org","webmd.com",
    "nhs.uk","clevelandclinic.org","hopkinsmedicine.org","healthline.com","verywellhealth.com"
}
MEDICAL_PATH_PAT = re.compile(r"/(health|conditions|disease|diabetes|hypertension|bp|cholesterol|lipid|heart|obesity|sleep|stress|wellbeing)/", re.I)
MEDICAL_QUERY_PAT = re.compile(r"[?&](q|query|utm_term|s|search)=.*(diabetes|hypertension|bp|cholesterol|lipid|sleep|stress|depression|diet|nutrition)", re.I)

def _domain_of(url: pd.Series) -> pd.Series:
    s = url.astype(str).str.lower()
    dom = s.str.extract(r"://([^/]+)")[0].fillna("")
    return dom.str.replace(r"^www\.", "", regex=True)

def _sessionize(ts: pd.Series, gap_minutes: int = 30) -> pd.Series:
    dt = ts.sort_values()
    gap = dt.diff().dt.total_seconds().div(60).fillna(1e9)
    return (gap > gap_minutes).cumsum()

def _decay_sum(days_ago: pd.Series, half_life_days: float = 30.0) -> float:
    lam = np.log(2.0) / half_life_days
    w = np.exp(-lam * days_ago.clip(lower=0).astype(float))
    return float(w.sum())

def build_web_features(labels: pd.DataFrame, web: pd.DataFrame, now_ts: pd.Timestamp) -> pd.DataFrame:
    if web is None or web.empty:
        feat = pd.DataFrame({"member_id": labels["member_id"].values})
        feat["member_id"] = pd.to_numeric(feat["member_id"], errors="coerce").astype("Int64")
        return feat

    w = web.copy()
    w["timestamp"] = pd.to_datetime(w["timestamp"], errors="coerce")
    w["text"] = (w.get("title","").astype(str) + " " +
                 w.get("description","").astype(str) + " " +
                 w.get("url","").astype(str)).str.lower()

    last_dt = w.groupby("member_id")["timestamp"].max()
    visits  = w.groupby("member_id").size()
    dom = _domain_of(w.get("url","").astype(str))
    dom_nunique = dom.groupby(w["member_id"]).nunique()

    feat = pd.DataFrame({"member_id": labels["member_id"].values})
    feat = feat.merge(visits.rename("wv_visits"), on="member_id", how="left")
    feat = feat.merge(dom_nunique.rename("wv_domains"), on="member_id", how="left")
    feat = feat.merge(last_dt.rename("wv_last_dt"), on="member_id", how="left")
    feat["wv_days_since_last"] = (now_ts - feat["wv_last_dt"]).dt.days
    feat = feat.drop(columns=["wv_last_dt"])

    w90 = w[w["timestamp"] >= now_ts - pd.Timedelta(days=90)].copy()
    dom90 = _domain_of(w90.get("url","").astype(str))

    for col, kws in BRIEF_BUCKETS.items():
        mask = pd.Series(False, index=w90.index)
        for kw in kws:
            mask = mask | w90["text"].str.contains(re.escape(kw), na=False)
        counts = w90[mask].groupby("member_id").size()
        feat = feat.merge(counts.rename(col), on="member_id", how="left")

    med_dom_hits = dom90.isin(MEDICAL_DOMAINS)
    med_path_hits = w90.get("url","").astype(str).str.contains(MEDICAL_PATH_PAT, na=False)
    med_query_hits= w90.get("url","").astype(str).str.contains(MEDICAL_QUERY_PAT, na=False)
    md_cnt = med_dom_hits.groupby(w90["member_id"]).sum(min_count=1)
    mp_cnt = med_path_hits.groupby(w90["member_id"]).sum(min_count=1)
    mq_cnt = med_query_hits.groupby(w90["member_id"]).sum(min_count=1)
    feat = feat.merge(md_cnt.rename("web_health_domain_visits_90d"), on="member_id", how="left")
    feat = feat.merge(mp_cnt.rename("web_medical_path_hits_90d"),    on="member_id", how="left")
    feat = feat.merge(mq_cnt.rename("web_medical_query_hits_90d"),   on="member_id", how="left")

    v90 = w90.groupby("member_id").size()
    share = (md_cnt / v90).replace([np.inf, -np.inf], np.nan)
    feat = feat.merge(share.rename("web_health_domain_share_90d"), on="member_id", how="left")

    w90 = w90.sort_values(["member_id","timestamp"])
    ses = (w90.groupby("member_id")["timestamp"].apply(lambda s: _sessionize(s, gap_minutes=30)))
    w90 = w90.assign(session_id=ses.values)
    ses_counts = w90.groupby(["member_id","session_id"]).size().rename("events_in_session")
    ses_per_member = ses_counts.groupby("member_id").size().rename("web_sessions_90d")
    events_per_ses  = ses_counts.groupby("member_id").mean().rename("web_events_per_session_90d")
    feat = feat.merge(ses_per_member, on="member_id", how="left")
    feat = feat.merge(events_per_ses,  on="member_id", how="left")

    w["days_ago"] = (now_ts - w["timestamp"]).dt.days
    decayed = (w.groupby("member_id")["days_ago"].apply(lambda s: _decay_sum(s, half_life_days=30)))
    feat = feat.merge(decayed.rename("web_visits_decayed_hl30"), on="member_id", how="left")

    for c in feat.columns:
        if c == "member_id": continue
        if c.endswith("_days_since_last"):
            mx = feat[c].max()
            feat[c] = feat[c].fillna((mx + 1) if pd.notna(mx) else 9999)
        else:
            feat[c] = feat[c].fillna(0)

    feat["member_id"] = pd.to_numeric(feat["member_id"], errors="coerce").astype("Int64")
    return feat
