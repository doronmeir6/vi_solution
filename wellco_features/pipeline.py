from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import pandas as pd

from .config import Config
from .engineering.claims import build_claim_features
from .engineering.web import build_web_features
from .engineering.app import build_app_features
from .engineering.combine import combine_features

@dataclass
class FeaturePipeline:
    cfg: Config

    def run(self) -> Path:
        labels = self.cfg.read_labels()
        claims = self.cfg.read_claims()
        app    = self.cfg.read_app()
        web    = self.cfg.read_web()

        candidates = []
        if not claims.empty and "diagnosis_date" in claims: candidates.append(claims["diagnosis_date"].max())
        if not app.empty and "timestamp" in app: candidates.append(app["timestamp"].max())
        if not web.empty and "timestamp" in web: candidates.append(web["timestamp"].max())
        now_ts = (max(candidates) if candidates else pd.Timestamp("2021-01-01")) + pd.Timedelta(days=1)

        claimsF = build_claim_features(labels, claims, now_ts)
        appF    = build_app_features(labels, app, now_ts)
        webF    = build_web_features(labels, web, now_ts)

        feats = combine_features(labels, webF, appF, claimsF, now_ts)
        out_path = self.cfg.out_dir / "features.csv"
        feats.to_csv(out_path, index=False)
        return out_path
