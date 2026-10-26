"""Module 5 - SAD integration: requirements, actors, use case diagram, workflow."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"

FUNCTIONAL = [
    ("FR1", "Dataset upload & validation",
     "The system shall accept a CSV upload, check it against the required 14-column schema, reject empty or malformed files with a clear message, and show null and duplicate summaries.",
     "The midterm data needed cleaning (duplicate IDs, inconsistent text, negative values); schema errors would silently corrupt every later result.",
     "Dataset page"),
    ("FR2", "Predict SLA breach risk at submission",
     "The system shall estimate breach risk for a new request using only service type, priority, queue size, assigned office and submission weekday.",
     "The midterm model beat the no-breach baseline (0.875 vs 0.50 accuracy on 8 test cases) using only submission-time fields; Processing_Days, Status and Reopened are excluded to avoid leakage.",
     "Prediction page"),
    ("FR3", "Show SLA performance by office and service",
     "The system shall display breach rate, request counts and average processing days by service type and by assigned office.",
     "Breach rates range from 0% to 100% across services and 0% to 67% across offices - targeted visibility is needed.",
     "Data Analysis and Dashboard pages"),
    ("FR4", "Flag high-risk requests for human review",
     "The system shall place requests predicted to breach into a review queue with risk level and recommended follow-up, and shall never approve, reject or deny a request automatically.",
     "The model is a small tree with a 12.5-point swing per test case (it produced 1 false alarm in 8); human review prevents unfair outcomes.",
     "Dashboard page"),
    ("FR5", "Record the final human decision",
     "The system shall let authorised staff record a decision, rationale, reviewer name, timestamp and model version for each flagged request, and export the log.",
     "Accountability: the model is advisory, so someone must own the outcome and the reasoning must be auditable.",
     "Dashboard page"),
]
NON_FUNCTIONAL = [
    ("NFR1", "Performance", "The trained pipeline shall load once per server session (cached) and never retrain on page refresh; pages shall respond in under 3 seconds for datasets of this size."),
    ("NFR2", "Security & privacy", "Only synthetic or authorised data shall be deployed publicly; no credentials or real student records are committed to the repository; decision-recording is limited to staff roles."),
    ("NFR3", "Transparency & reliability", "Every prediction shall show the model version, timestamp, limitations and a human-review notice; invalid inputs shall be rejected with explicit error messages."),
]
ACTORS = pd.DataFrame([
    ["Student", "Submits and checks a request", "No", "No"],
    ["Student Services Staff", "Reviews flagged requests, follows up with requesters", "Yes", "Yes"],
    ["System Administrator", "Manages data uploads, monitors performance, retrains the model", "Yes", "Yes"],
], columns=["Actor", "Main responsibility", "Can view predictions & dashboard", "Can record final decision"])


def render():
    st.header("System Design (SAD Integration)")
    st.subheader("Actors and permissions")
    st.dataframe(ACTORS, hide_index=True, use_container_width=True)
    st.caption("Roles in this app are a demo selector in the sidebar, not real authentication.")

    st.subheader("Functional requirements")
    for rid, title, text, evidence, where in FUNCTIONAL:
        with st.expander(f"{rid} - {title}"):
            st.markdown(f"**Requirement:** {text}")
            st.markdown(f"**Evidence:** {evidence}")
            st.markdown(f"**Implemented in:** {where}")

    st.subheader("Non-functional requirements")
    st.dataframe(pd.DataFrame(NON_FUNCTIONAL, columns=["ID", "Category", "Requirement"]),
                 hide_index=True, use_container_width=True)

    st.subheader("Updated use case diagram")
    p = ASSETS / "use_case_diagram.png"
    if p.exists():
        st.image(str(p), use_container_width=True)
    else:
        st.error("assets/use_case_diagram.png is missing.")

    st.subheader("Decision workflow and analytical process flow")
    w = ASSETS / "decision_workflow.png"
    if w.exists():
        st.image(str(w), use_container_width=True)
    else:
        st.error("assets/decision_workflow.png is missing.")
    st.markdown("**Who sees what:** staff and administrators see risk scores and the review queue; "
                "the Student role can open only the Home and System Design pages. **Only staff and administrators record the final decision**, "
                "and the model's output is one input to that decision, never the decision itself.")
