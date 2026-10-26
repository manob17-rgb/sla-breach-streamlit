"""Run with:  pytest -q   (from the repository root)
Checks that the deployed model matches the midterm notebook and that
validation/cleaning behave as documented."""
import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from modules import data_processing as dp
from modules import prediction as pr

ROOT = Path(__file__).resolve().parent.parent
PIPE = joblib.load(ROOT / "models" / "trained_model.joblib")
META = json.loads((ROOT / "models" / "model_metadata.json").read_text())


def test_midterm_metrics_reproduced():
    m = META["metrics"]
    assert META["confusion_matrix"] == [[3, 1], [0, 4]]
    assert m["accuracy"] == pytest.approx(0.875)
    assert m["precision"] == pytest.approx(0.8)
    assert m["recall"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(0.8889, abs=1e-4)
    assert m["baseline_accuracy"] == pytest.approx(0.5)


def test_saved_model_matches_midterm_test_cases():
    tc = pd.DataFrame(META["test_cases"])
    assert (PIPE.predict(tc[dp.FEATURES]) == tc["Predicted"]).all()


def test_no_leakage_features():
    assert not set(dp.FEATURES) & set(dp.LEAKAGE_FIELDS)


def test_queue_category_bins():
    assert [dp.queue_category(n) for n in (0, 5, 6, 15, 16, 40)] == \
        ["Low", "Low", "Moderate", "Moderate", "Critical", "Critical"]


def test_input_row_matches_training_schema():
    row = pr.build_input_row("Transcript of records", "High", 30, "Records office", "2026-10-26")
    assert list(row.columns) == dp.FEATURES
    assert row.at[0, "Queue_Category"] == "Critical" and row.at[0, "Submission_Weekday"] == 0
    assert PIPE.predict(row)[0] in (0, 1)


@pytest.mark.parametrize("q", [-1, None, 2.5])
def test_invalid_queue_rejected(q):
    errs = pr.validate_inputs("Document request", "Low", q, "Records office", "2026-10-26", META["known_categories"])
    assert errs


def test_schema_validation():
    assert dp.validate_schema(dp.load_sample())["ok"]
    assert not dp.validate_schema(pd.DataFrame())["ok"]
    assert not dp.validate_schema(pd.DataFrame({"a": [1]}))["ok"]


def test_cleaning_handles_negatives_and_duplicates():
    df = dp.load_sample()
    dirty = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    dirty.loc[1, "Queue_Length"] = -4
    clean = dp.clean_dataframe(dirty)
    assert len(clean) == 40 and pd.isna(clean.loc[1, "Queue_Length"])
