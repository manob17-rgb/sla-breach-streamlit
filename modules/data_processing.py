"""Module 1 helpers - dataset loading, schema validation, cleaning.

Cleaning rules are reused from the midterm notebook (PART II). Pure pandas
functions (no Streamlit calls) so they can be unit-tested and reused by
train_model.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = ROOT / "data" / "cleaned_student_requests.csv"

# Columns that must be in an uploaded CSV (the 14 raw midterm columns).
REQUIRED_COLUMNS = [
    "Request_ID", "Service_Type", "Requester_Type", "Request_Channel",
    "Priority", "Assigned_Office", "Queue_Length", "Status", "Reopened",
    "Processing_Days", "SLA_Days", "Satisfaction_Score", "Submission_Date",
    "SLA_Breached",
]
NUMERIC_COLUMNS = ["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]

# Model definition from the midterm (submission-time features only).
FEATURES = ["Service_Type", "Priority", "Queue_Category", "Assigned_Office", "Submission_Weekday"]
CAT_FEATS = ["Service_Type", "Priority", "Queue_Category", "Assigned_Office"]
# Fields that exist only AFTER the request is processed -> never used for prediction.
LEAKAGE_FIELDS = ["Processing_Days", "Status", "Reopened", "Satisfaction_Score", "SLA_Breached"]

QUEUE_BINS = [-1, 5, 15, float("inf")]
QUEUE_LABELS = ["Low", "Moderate", "Critical"]
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def queue_category(queue_length) -> str:
    """Low (<=5), Moderate (6-15), Critical (>15) - same bins as the midterm."""
    cat = pd.cut([queue_length], bins=QUEUE_BINS, labels=QUEUE_LABELS)[0]
    return str(cat) if pd.notna(cat) else "Unknown"


def load_sample() -> pd.DataFrame:
    return pd.read_csv(SAMPLE_PATH)


def validate_schema(df: pd.DataFrame) -> dict:
    """Return {'ok': bool, 'errors': [...], 'warnings': [...]} for an uploaded file."""
    errors, warnings = [], []
    if df is None or df.empty:
        return {"ok": False, "errors": ["The file is empty (no data rows)."], "warnings": []}
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append("Missing required column(s): " + ", ".join(missing))
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS
             and c not in ("Submission_Weekday", "Queue_Category")]
    if extra:
        warnings.append("Unrecognised extra column(s) will be ignored: " + ", ".join(extra))
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            bad = pd.to_numeric(df[col], errors="coerce").isna() & df[col].notna()
            if bad.any():
                errors.append(f"Column '{col}' has {int(bad.sum())} non-numeric value(s).")
    if "Submission_Date" in df.columns:
        bad = pd.to_datetime(df["Submission_Date"], errors="coerce").isna() & df["Submission_Date"].notna()
        if bad.any():
            errors.append(f"Column 'Submission_Date' has {int(bad.sum())} unparseable date(s).")
    return {"ok": not errors, "errors": errors, "warnings": warnings}


def null_duplicate_summary(df: pd.DataFrame) -> dict:
    nulls = df.isnull().sum()
    return {
        "null_table": nulls[nulls > 0].rename("Null count").to_frame(),
        "total_nulls": int(nulls.sum()),
        "duplicate_ids": int(df["Request_ID"].duplicated().sum()) if "Request_ID" in df else 0,
        "duplicate_rows": int(df.duplicated().sum()),
    }


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Midterm cleaning + feature engineering (notebook PART II)."""
    out = df.copy()
    if "Request_ID" in out:
        out = out.drop_duplicates("Request_ID")
    for col in ["Priority", "Status", "Service_Type", "Assigned_Office"]:
        out[col] = out[col].astype(str).str.strip().str.capitalize()
    for col in ["Queue_Length", "Processing_Days", "SLA_Days"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
        out.loc[out[col] < 0, col] = np.nan            # negatives -> NaN
    out["Submission_Date"] = pd.to_datetime(out["Submission_Date"], errors="coerce")
    out["Submission_Weekday"] = out["Submission_Date"].dt.dayofweek
    out["Queue_Category"] = pd.cut(out["Queue_Length"], bins=QUEUE_BINS,
                                   labels=QUEUE_LABELS).astype(object)
    # SLA_Breached = Processing_Days > SLA_Days, for COMPLETED requests only.
    done = (out["Status"].str.lower() == "completed") & out["Processing_Days"].notna() & out["SLA_Days"].notna()
    out["SLA_Breached"] = (pd.to_numeric(out["SLA_Breached"], errors="coerce")
                           if "SLA_Breached" in out.columns else np.nan)
    out.loc[done, "SLA_Breached"] = (out.loc[done, "Processing_Days"] > out.loc[done, "SLA_Days"]).astype(int)
    return out


def modeling_subset(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for training/evaluation (completed/cancelled with a known label and features)."""
    m = clean_df[clean_df["Status"].str.lower().isin(["completed", "cancelled"])].copy()
    return m.dropna(subset=FEATURES + ["SLA_Breached"])
