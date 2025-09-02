# final_vi/wellco_features/config.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import os
import pandas as pd
from pandas.errors import EmptyDataError

def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def _resolve_dir(p: Path | str) -> Path:
    p = Path(p)
    root = _project_root()
    if not p.is_absolute():
        p = (root / p).resolve()
    # de-dupe nested final_vi segments
    parts, seen = [], 0
    for seg in p.parts:
        if seg == "final_vi":
            seen += 1
            if seen > 1:  # skip repeated
                continue
        parts.append(seg)
    return Path(*parts)

# -------- safe CSV reader --------
def _read_csv_safe(
    path: Path,
    *,
    dtype: dict | None = None,
    parse_dates: list[str] | None = None,
    required: bool = False,
    required_name: str = "",
) -> pd.DataFrame:
    """
    Robust CSV reader:
    - Missing file -> empty DF (unless required=True, then raise).
    - Empty or header-less file -> empty DF (unless required=True, then raise with hint).
    - Parses dates and applies dtype when possible.
    """
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Required file not found: {path}")
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")
    except EmptyDataError:
        if required:
            raise EmptyDataError(
                f"Required file is empty or has no header: {path}\n"
                f"Hint: Ensure the first row contains column names like "
                f"'member_id, signup_date, churn, outreach'."
            )
        return pd.DataFrame()
    except Exception as e:
        # Common gotchas: wrong delimiter, Excel file mistakenly saved as .csv
        raise RuntimeError(
            f"Failed reading {path} as CSV: {e}\n"
            f"Checks: Is it really CSV? Does it have a header row? Is the delimiter a comma?"
        ) from e

    # Normalize dates if present
    if parse_dates:
        for c in parse_dates:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    # Normalize member_id dtype if present
    if dtype and "member_id" in (dtype or {}) and "member_id" in df.columns:
        df["member_id"] = pd.to_numeric(df["member_id"], errors="coerce").astype("Int64")

    return df

@dataclass
class Config:
    data_dir: Path
    out_dir: Path

    col_member: str = "member_id"
    col_signup: str = "signup_date"
    col_churn: str = "churn"
    col_outreach: str = "outreach"  # sometimes called 'outcare'

    def __post_init__(self):
        # Normalize & ensure out_dir exists early (so you always see final_vi/out/)
        self.data_dir = _resolve_dir(self.data_dir)
        self.out_dir  = _resolve_dir(self.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # Resolve data_dir if the provided one doesn't exist
        if not self.data_dir.exists():
            env_dir = os.environ.get("WELLCO_DATA_DIR", "").strip()
            candidates = [
                self.data_dir,
                _project_root() / "data",
                Path.cwd()  / "data",
                Path.cwd() / "data",
                Path(env_dir).expanduser() if env_dir else None,
            ]
            candidates = [c for c in candidates if c is not None]
            for c in candidates:
                if c.exists():
                    self.data_dir = c
                    break
            else:
                tried = "\n  ".join(str(c) for c in candidates)
                raise FileNotFoundError(f"Could not locate data dir. Tried:\n  {tried}")

    # ---------- Readers (robust to empty files) ----------
    def read_labels(self) -> pd.DataFrame:
        """Required. Must contain: member_id, signup_date, churn (0/1), outreach/outcare (0/1)."""
        p = self.data_dir / "churn_labels.csv"
        df = _read_csv_safe(
            p,
            dtype={self.col_member: "Int64"},
            parse_dates=[self.col_signup],
            required=True,
            required_name="labels",
        )
        # Basic schema sanity
        need = {self.col_member, self.col_signup, self.col_churn}
        if not need.issubset(df.columns):
            raise ValueError(
                f"{p} is missing required columns. Need at least: {sorted(need)}. "
                f"Got: {sorted(df.columns.tolist())}"
            )
        # Accept either 'outreach' or 'outcare' as treatment; not strictly required for churn-only runs
        if ("outreach" not in df.columns) and ("outcare" not in df.columns):
            # Add a 0 default if completely missing to keep pipeline running
            df["outreach"] = 0
        return df

    def read_claims(self) -> pd.DataFrame:
        p = self.data_dir / "claims.csv"
        df = _read_csv_safe(p, dtype={self.col_member: "Int64"})
        if df.empty:
            return df
        # Map any date-ish column into 'diagnosis_date' if present
        for c in ("diagnosis_date", "claim_date", "date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")
                if "diagnosis_date" not in df.columns:
                    df["diagnosis_date"] = df[c]
                break
        return df

    def read_app(self) -> pd.DataFrame:
        p = self.data_dir / "app_usage.csv"
        df = _read_csv_safe(p, dtype={self.col_member: "Int64"})
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df

    def read_web(self) -> pd.DataFrame:
        p = self.data_dir / "web_visits.csv"
        df = _read_csv_safe(p, dtype={self.col_member: "Int64"})
        if df.empty:
            return df
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        return df
