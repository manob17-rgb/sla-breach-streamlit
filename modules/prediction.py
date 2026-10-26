"""Module 3 - Predictive modeling (inference with the SAVED midterm pipeline).

The pipeline is loaded once with @st.cache_resource - it is never retrained on
a page refresh. Prediction inputs are limited to the five submission-time
features used in the midterm (no Processing_Days / Status / Reopened).
"""
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from modules.data_processing import (FEATURES, WEEKDAYS, queue_category)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "trained_model.joblib"
META_PATH = ROOT / "models" / "model_metadata.json"

# In the midterm data each service type is handled by exactly one office.
DEFAULT_OFFICE = {
    "Document request": "Records office", "Academic record": "Registrar",
    "Id replacement": "Student affairs", "Enrollment verification": "Registrar",
    "Transcript of records": "Records office", "Leave of absence": "Student affairs",
    "Graduation clearance": "Registrar", "Subject overload": "Academic affairs",
    "Scholarship renewal": "Scholarship unit", "Transfer credentials": "Records office",
}
FLAG_THRESHOLD = 0.5   # same decision rule as DecisionTreeClassifier.predict


# ---------------------------------------------------------------- pure logic
def risk_level(p: float) -> str:
    if p >= 0.8:
        return "High"
    if p >= FLAG_THRESHOLD:
        return "Elevated"
    return "Low"


def validate_inputs(service, priority, queue_length, office, submit_date, known) -> list[str]:
    """Return a list of error strings (empty list = inputs are valid)."""
    errs = []
    if service not in known["Service_Type"]:
        errs.append(f"Unknown service type: {service!r}.")
    if priority not in known["Priority"]:
        errs.append(f"Unknown priority: {priority!r}.")
    if office not in known["Assigned_Office"]:
        errs.append(f"Unknown office: {office!r}.")
    if queue_length is None or pd.isna(queue_length):
        errs.append("Queue length is required.")
    elif queue_length < 0:
        errs.append("Queue length cannot be negative.")
    elif queue_length != int(queue_length):
        errs.append("Queue length must be a whole number.")
    if submit_date is None or pd.isna(pd.Timestamp(submit_date)):
        errs.append("A valid submission date is required.")
    return errs


def build_input_row(service, priority, queue_length, office, submit_date) -> pd.DataFrame:
    """One-row DataFrame with the EXACT training feature names/order/types."""
    ts = pd.Timestamp(submit_date)
    row = {
        "Service_Type": str(service),
        "Priority": str(priority),
        "Queue_Category": queue_category(queue_length),
        "Assigned_Office": str(office),
        "Submission_Weekday": int(ts.dayofweek),
    }
    return pd.DataFrame([row], columns=FEATURES)


def predict_one(pipeline, row: pd.DataFrame) -> dict:
    label = int(pipeline.predict(row)[0])
    p = float(pipeline.predict_proba(row)[0][1]) if hasattr(pipeline, "predict_proba") else None
    return {"label": label, "probability": p,
            "risk_level": risk_level(p) if p is not None else ("High" if label else "Low"),
            "flagged": label == 1,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}


def score_dataframe(pipeline, clean_df: pd.DataFrame) -> pd.DataFrame:
    """Score every row whose five features are available (for the dashboard)."""
    ok = clean_df.dropna(subset=FEATURES).copy()
    if ok.empty:
        return ok.assign(P_breach=[], Predicted=[], Risk_Level=[])
    ok["P_breach"] = pipeline.predict_proba(ok[FEATURES])[:, 1]
    ok["Predicted"] = pipeline.predict(ok[FEATURES]).astype(int)
    ok["Risk_Level"] = ok["P_breach"].apply(risk_level)
    return ok


# ------------------------------------------------------------- model loading
@st.cache_resource(show_spinner="Loading saved model...")
def load_model():
    """Load the saved pipeline + metadata. Falls back to the midterm training
    routine ONLY if the .joblib cannot be read (e.g. scikit-learn version
    mismatch on the host) - the result is cached, so it runs once."""
    meta = json.loads(META_PATH.read_text())
    try:
        return joblib.load(MODEL_PATH), meta, None
    except Exception as exc:  # pragma: no cover - environment dependent
        import train_model
        from modules.data_processing import load_sample
        pipeline, meta = train_model.train_and_evaluate(load_sample())
        return pipeline, meta, f"Saved model could not be loaded ({type(exc).__name__}); rebuilt once from the midterm code."


# -------------------------------------------------------------------- UI
def render_metrics(meta: dict):
    m = meta["metrics"]
    st.subheader("Model metrics (midterm hold-out test set)")
    c = st.columns(5)
    c[0].metric("Accuracy", f"{m['accuracy']:.3f}")
    c[1].metric("Precision", f"{m['precision']:.3f}")
    c[2].metric("Recall", f"{m['recall']:.3f}")
    c[3].metric("F1", f"{m['f1']:.3f}")
    c[4].metric("Baseline (no-breach)", f"{m['baseline_accuracy']:.3f}",
                delta=f"+{m['improvement_over_baseline']:.3f} vs baseline")
    st.caption(f"Test set has only **{meta['n_test']} requests** (of {meta['n_total']}), so every "
               "metric moves by 12.5 percentage points per single case. Treat them as indicative, not proven.")
    cm = meta["confusion_matrix"]
    cm_df = pd.DataFrame(cm, index=["Actual: On time", "Actual: Breached"],
                         columns=["Predicted: On time", "Predicted: Breached"])
    left, right = st.columns(2)
    left.markdown("**Confusion matrix**")
    left.dataframe(cm_df)
    imp = pd.DataFrame({"Feature": [k.split("__")[-1] for k in meta["feature_importance"]],
                        "Importance": list(meta["feature_importance"].values())})
    right.markdown("**Feature importance (what the tree actually uses)**")
    right.dataframe(imp, hide_index=True)


def similar_history(history: pd.DataFrame, priority: str, qcat: str) -> tuple[int, float]:
    """How many past completed requests share this priority + queue category, and their breach rate."""
    h = history[history["SLA_Breached"].notna()]
    m = h[(h["Priority"] == priority) & (h["Queue_Category"] == qcat)]
    return int(len(m)), (float(m["SLA_Breached"].mean()) if len(m) else float("nan"))


def render(pipeline, meta, load_warning, can_record: bool, history: pd.DataFrame):
    st.header("Predictive Modeling - SLA breach risk at submission")
    if load_warning:
        st.warning(load_warning)
    st.info(f"**Model:** {meta['model_version']}  |  {meta['algorithm']}  |  trained "
            f"{meta['trained_at_utc']}  |  scikit-learn {meta['sklearn_version']}\n\n"
            "**Prediction checkpoint:** request submission. Only these inputs are used: "
            "service type, priority, queue size (converted to Low/Moderate/Critical), assigned office, "
            "and submission weekday. Processing days, status, reopened and satisfaction are excluded (leakage).")

    known = meta["known_categories"]
    st.subheader("1. Enter a new request")
    c1, c2, c3 = st.columns(3)
    service = c1.selectbox("Service type", known["Service_Type"])
    priority = c2.selectbox("Priority", ["Low", "Medium", "High"])
    queue_len = c3.number_input("Queue length (requests ahead)", min_value=-5, max_value=500, value=10, step=1,
                                help="Negative values are rejected by validation.")
    default_office = DEFAULT_OFFICE.get(service, known["Assigned_Office"][0])
    offices = known["Assigned_Office"]
    c4, c5 = st.columns(2)
    office = c4.selectbox("Assigned office", offices, index=offices.index(default_office)
                          if default_office in offices else 0)
    submit_date = c5.date_input("Submission date", value=date.today())
    request_id = st.text_input("Request ID (optional label)", value="NEW-001")

    if office != DEFAULT_OFFICE.get(service):
        st.warning("This service/office pairing never appears in the midterm data - the model has no "
                   "evidence for this combination, so treat the result with extra caution.")

    if st.button("Predict SLA breach risk", type="primary"):
        errs = validate_inputs(service, priority, queue_len, office, submit_date, known)
        if errs:
            for e in errs:
                st.error(e)
        else:
            row = build_input_row(service, priority, queue_len, office, submit_date)
            res = predict_one(pipeline, row)
            st.session_state["last_prediction"] = {
                "Request_ID": request_id or "NEW", "Service_Type": service, "Priority": priority,
                "Queue_Length": int(queue_len), "Queue_Category": row.at[0, "Queue_Category"],
                "Assigned_Office": office, "Submission_Weekday": int(row.at[0, "Submission_Weekday"]),
                "P_breach": res["probability"], "Risk_Level": res["risk_level"],
                "Flagged": res["flagged"], "Predicted_at": res["timestamp"],
                "Model_version": meta["model_version"], "row": row,
            }

    last = st.session_state.get("last_prediction")
    if last:
        st.subheader("2. Result")
        row = last["row"]
        label = "LIKELY TO BREACH SLA" if last["Flagged"] else "Likely to meet SLA"
        (st.error if last["Flagged"] else st.success)(
            f"**{label}** - risk level **{last['Risk_Level']}**"
            + (f", estimated breach probability **{last['P_breach']:.0%}**" if last["P_breach"] is not None else ""))
        a, b, c = st.columns(3)
        a.metric("Queue category", last["Queue_Category"])
        b.metric("Submission weekday", WEEKDAYS[last["Submission_Weekday"]])
        c.metric("Prediction time", last["Predicted_at"])
        st.caption(f"Model version: {last['Model_version']}")
        n_sim, rate_sim = similar_history(history, last["Priority"], last["Queue_Category"])
        if n_sim == 0:
            st.warning("**No evidence for this combination.** None of the historical requests had "
                       f"{last['Priority']} priority with a {last['Queue_Category']} queue, so this score is an "
                       "extrapolation by the tree and should not be trusted without staff judgement.")
        else:
            st.caption(f"Evidence: {n_sim} historical request(s) had {last['Priority']} priority and a "
                       f"{last['Queue_Category']} queue; {rate_sim:.0%} of them breached SLA.")
        with st.expander("Exact one-row input sent to the saved pipeline"):
            st.dataframe(row)
        st.warning("**Advisory only.** This is a decision-support signal from a small decision tree (depth 3, "
                   f"{meta['n_train']} training cases). The probability is the share of training cases in the "
                   "tree leaf, not a calibrated confidence. A staff member must review every flagged request; "
                   "the system never approves, rejects or denies a service.")
        if last["Flagged"]:
            if can_record:
                if st.button("Add to staff review queue"):
                    st.session_state.setdefault("new_predictions", [])
                    st.session_state["new_predictions"] = [
                        r for r in st.session_state["new_predictions"] if r["Request_ID"] != last["Request_ID"]]
                    st.session_state["new_predictions"].append({k: v for k, v in last.items() if k != "row"})
                    st.success("Added. Open the Dashboard page to review it.")
            else:
                st.info("Only staff roles can add requests to the review queue.")

    st.divider()
    render_metrics(meta)

    with st.expander("Midterm test cases replayed with the saved model (consistency check)"):
        tc = pd.DataFrame(meta["test_cases"])
        live = pipeline.predict(tc[FEATURES]).astype(int)
        tc["Saved_model_now"] = live
        tc["Matches_midterm"] = (tc["Predicted"] == tc["Saved_model_now"])
        st.dataframe(tc, hide_index=True)
        st.caption("'Predicted' is what the notebook-trained model gave at training time; 'Saved_model_now' is the "
                   "loaded joblib pipeline. All rows should match.")

    with st.expander("Limitations"):
        st.markdown(
            "- Trained on **40 synthetic requests** (32 train / 8 test) - far too small to generalise.\n"
            "- A depth-3 tree can only split on a few features; most of its signal is queue size and priority.\n"
            "- Each service maps to one office in the data, so service and office effects are confounded.\n"
            "- Probabilities are leaf proportions, not calibrated confidence.\n"
            "- Missing or out-of-range inputs reduce reliability; the app rejects negative queue lengths.\n"
            "- Retrain periodically with new completed requests; keep a human in the loop.")
