SAMPLE_CASES = [
    ("Transcript of records", "High", 28, "Records office", "2026-10-12"),
    ("Scholarship renewal", "Low", 2, "Scholarship unit", "2026-10-13"),
    ("Document request", "Medium", 9, "Records office", "2026-10-14"),
    ("Graduation clearance", "High", 22, "Registrar", "2026-10-15"),
    ("Leave of absence", "Medium", 4, "Student affairs", "2026-10-16"),
]


def sample_predictions(pipeline, history: pd.DataFrame, version: str) -> pd.DataFrame:
    """Run the five lab sample requests through the saved pipeline (same path as the form)."""
    rows = []
    for i, (svc, pri, q, off, dt) in enumerate(SAMPLE_CASES, 1):
        row = build_input_row(svc, pri, q, off, dt)
        res = predict_one(pipeline, row)
        n, rate = similar_history(history, pri, row.at[0, "Queue_Category"])
        rows.append({
            "#": i, "Service type": svc, "Priority": pri, "Queue length": q,
            "Queue category": row.at[0, "Queue_Category"], "Office": off,
            "Date": f"{dt} ({WEEKDAYS[row.at[0, 'Submission_Weekday']][:3]})",
            "Prediction": "WILL BREACH SLA" if res["flagged"] else "Will meet SLA",
            "P(breach)": f"{res['probability']:.0%}", "Risk level": res["risk_level"],
            "Human review": "Required" if res["flagged"] else "Not flagged",
            "Past similar cases": f"{n} cases, {rate:.0%} breached" if n else "none",
            "Model": version, "Predicted at": res["timestamp"],
        })
    return pd.DataFrame(rows)
