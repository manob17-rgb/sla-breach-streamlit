"""
train_model.py - Step 1 of the Streamlit deployment exercise.

Reuses the EXACT modeling code from the midterm notebook (miles.ipynb):
  * same cleaning rules          * same engineered features
  * same 5 submission-time features (no leakage)
  * same 80/20 stratified split, random_state=42
  * same DecisionTreeClassifier(max_depth=3, random_state=42) pipeline

It then SAVES the fitted preprocessing + model pipeline with joblib so the
Streamlit app never retrains on a page refresh.

Run once:  python train_model.py
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from modules.data_processing import (CAT_FEATS, FEATURES, clean_dataframe,
                                     modeling_subset)

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "cleaned_student_requests.csv"
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"
MODEL_VERSION = "v1.0-midterm-DT-depth3"


def build_pipeline() -> Pipeline:
    """Identical to the midterm notebook (PART IV)."""
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(drop="first", sparse_output=False,
                                            handle_unknown="ignore"), CAT_FEATS)],
        remainder="passthrough",
    )
    return Pipeline(steps=[("prep", pre),
                           ("dt", DecisionTreeClassifier(max_depth=3, random_state=42))])


def train_and_evaluate(raw_df: pd.DataFrame):
    clean_df = clean_dataframe(raw_df)
    model_df = modeling_subset(clean_df)
    X = model_df[FEATURES]
    y = model_df["SLA_Breached"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = build_pipeline()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    baseline = float((y_test == 0).mean())
    acc = float(accuracy_score(y_test, y_pred))
    names = model.named_steps["prep"].get_feature_names_out()
    imp = (pd.Series(model.named_steps["dt"].feature_importances_, index=names)
           .sort_values(ascending=False))

    test_cases = X_test.copy()
    test_cases.insert(0, "Request_ID", model_df.loc[X_test.index, "Request_ID"])
    test_cases["Actual"] = y_test.values
    test_cases["Predicted"] = y_pred
    test_cases["P_breach"] = model.predict_proba(X_test)[:, 1]

    meta = {
        "model_version": MODEL_VERSION,
        "algorithm": "DecisionTreeClassifier(max_depth=3, random_state=42)",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "features": FEATURES,
        "target": "SLA_Breached",
        "n_total": int(len(model_df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": {
            "accuracy": acc,
            "precision": float(precision_score(y_test, y_pred)),
            "recall": float(recall_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred)),
            "baseline_accuracy": baseline,
            "improvement_over_baseline": acc - baseline,
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": {k: float(v) for k, v in imp.items() if v > 0},
        "test_cases": json.loads(test_cases.to_json(orient="records")),
        "known_categories": {c: sorted(model_df[c].dropna().unique().tolist())
                             for c in CAT_FEATS},
    }
    return model, meta


if __name__ == "__main__":
    raw = pd.read_csv(DATA)
    model, meta = train_and_evaluate(raw)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Saved {MODEL_PATH.name} and {META_PATH.name}")
    print("Confusion matrix:", meta["confusion_matrix"])
    for k, v in meta["metrics"].items():
        print(f"  {k:>26}: {v:.4f}")
    print("Top features:", list(meta["feature_importance"].items())[:3])
