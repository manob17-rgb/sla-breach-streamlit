"""Module 4 - Decision-support dashboard: KPIs, risk/review queue,
interpretations, recommended human follow-up, and a decision log."""
from __future__ import annotations

from datetime import datetime

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.exploratory_analysis import breach_by, labelled
from modules.prediction import score_dataframe

DECISIONS = [
    "Expedite / escalate to office head",
    "Reassign to another staff member",
    "Contact requester with a realistic timeline",
    "Monitor - no action yet",
    "Override model - no action (reason required)",
]
FOLLOW_UP = {
    "High": "Review today. Check the office's queue, consider reassigning or adding staff, and tell the requester the expected timeline.",
    "Elevated": "Review within the working day. Confirm the request has an owner and watch it until it moves.",
    "Low": "No action needed beyond normal processing.",
}


# ---------------------------------------------------------------- pure logic
def kpis(clean_df: pd.DataFrame, scored: pd.DataFrame) -> dict:
    lab = labelled(clean_df)
    return {
        "total": int(len(clean_df)),
        "completed": int(len(lab)),
        "breach_rate": float(lab["SLA_Breached"].mean()) if len(lab) else float("nan"),
        "avg_days": float(lab["Processing_Days"].mean()) if len(lab) else float("nan"),
        "flagged": int((scored["Predicted"] == 1).sum()) if len(scored) else 0,
        "critical_queue": int((clean_df["Queue_Category"] == "Critical").sum()),
    }


def interpretations(clean_df: pd.DataFrame) -> list[str]:
    lab = labelled(clean_df)
    if lab.empty:
        return ["Not enough completed requests to interpret."]
    out = []
    o = breach_by(lab, "Assigned_Office")
    out.append(f"**{o.iloc[0]['Assigned_Office']}** has the highest breach rate ({o.iloc[0]['Breach_Rate']:.0%}); "
               f"**{o.iloc[-1]['Assigned_Office']}** has the lowest ({o.iloc[-1]['Breach_Rate']:.0%}).")
    q = lab.groupby("Queue_Category")["SLA_Breached"].mean()
    if "Critical" in q.index:
        out.append(f"Requests that entered a **Critical** queue (>15 waiting) breached SLA {q['Critical']:.0%} of the time.")
    if "Low" in q.index:
        out.append(f"Requests that entered a **Low** queue (<=5 waiting) breached SLA {q['Low']:.0%} of the time.")
    out.append("Backlog at the moment of submission is the clearest early warning sign in this data - "
               "staffing decisions should follow queue size, not only request priority.")
    return out


def build_queue(scored: pd.DataFrame, new_preds: list[dict], decisions: list[dict]) -> pd.DataFrame:
    cols = ["Source", "Request_ID", "Service_Type", "Assigned_Office", "Priority", "Queue_Category",
            "P_breach", "Risk_Level"]
    parts = []
    if len(scored):
        d = scored[scored["Predicted"] == 1].copy()
        d["Source"] = "Dataset"
        parts.append(d[cols])
    if new_preds:
        n = pd.DataFrame(new_preds)
        n["Source"] = "New submission"
        parts.append(n[cols])
    q = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)
    done = {d["Request_ID"] for d in decisions}
    q["Review_Status"] = q["Request_ID"].apply(lambda r: "Reviewed" if r in done else "Pending")
    return q.sort_values(["Review_Status", "P_breach"], ascending=[False, False]).reset_index(drop=True)


# ----------------------------------------------------------------------- UI
def render(pipeline, clean_df: pd.DataFrame, meta: dict, can_record: bool, role: str):
    st.header("Decision-Support Dashboard")
    st.caption("Predictions are advisory. Staff make and record the final decision; the system never denies a service.")
    scored = score_dataframe(pipeline, clean_df)
    k = kpis(clean_df, scored)

    c = st.columns(6)
    c[0].metric("Total requests", k["total"])
    c[1].metric("Completed", k["completed"])
    c[2].metric("SLA breach rate", "n/a" if pd.isna(k["breach_rate"]) else f"{k['breach_rate']:.0%}")
    c[3].metric("Avg processing days", "n/a" if pd.isna(k["avg_days"]) else f"{k['avg_days']:.1f}")
    c[4].metric("Flagged by model", k["flagged"])
    c[5].metric("Critical-queue requests", k["critical_queue"])

    st.subheader("Interpretation")
    for line in interpretations(clean_df):
        st.markdown("- " + line)

    lab = labelled(clean_df)
    if len(lab):
        a, b = st.columns(2)
        fo = breach_by(lab, "Assigned_Office")
        fig = px.bar(fo, x="Assigned_Office", y="Breach_Rate", text_auto=".0%", title="Breach rate by office")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        a.plotly_chart(fig, use_container_width=True)
        fq = px.box(lab, x="Queue_Category", y="Processing_Days", points="all",
                    category_orders={"Queue_Category": ["Low", "Moderate", "Critical"]},
                    title="Processing days by queue category")
        b.plotly_chart(fq, use_container_width=True)

    st.subheader("Risk / review queue")
    st.caption("Dataset rows are scored using only their submission-time features. Because the model was trained on "
               "most of these rows, their scores look better than they would for genuinely new requests.")
    decisions = st.session_state.setdefault("decision_log", [])
    queue = build_queue(scored, st.session_state.get("new_predictions", []), decisions)
    if queue.empty:
        st.success("No requests are currently flagged.")
    else:
        show = queue.copy()
        show["P_breach"] = show["P_breach"].map(lambda x: f"{x:.0%}")
        st.dataframe(show, hide_index=True, use_container_width=True)
        levels = queue["Risk_Level"].unique().tolist()
        with st.expander("Recommended human follow-up by risk level"):
            for lv in ["High", "Elevated", "Low"]:
                if lv in levels:
                    st.markdown(f"**{lv}:** {FOLLOW_UP[lv]}")

    st.subheader("Record a review decision")
    if not can_record:
        st.info(f"Your role (**{role}**) can view results but cannot record decisions. "
                "Switch to Student Services Staff or System Administrator in the sidebar (demo roles).")
    else:
        pending = queue[queue["Review_Status"] == "Pending"]["Request_ID"].tolist() if len(queue) else []
        if not pending:
            st.success("No pending flagged requests.")
        else:
            with st.form("decision_form", clear_on_submit=True):
                rid = st.selectbox("Flagged request", pending)
                decision = st.selectbox("Decision", DECISIONS)
                reviewer = st.text_input("Reviewer name")
                note = st.text_area("Rationale / note")
                submitted = st.form_submit_button("Save decision")
            if submitted:
                needs_reason = decision.startswith("Override")
                if not reviewer.strip():
                    st.error("Reviewer name is required for the audit trail.")
                elif needs_reason and not note.strip():
                    st.error("A rationale is required when overriding the model.")
                else:
                    row = queue[queue["Request_ID"] == rid].iloc[0]
                    decisions.append({
                        "Request_ID": rid, "Decision": decision, "Reviewer": reviewer.strip(), "Role": role,
                        "Note": note.strip(), "Model_risk": row["Risk_Level"],
                        "Model_P_breach": round(float(row["P_breach"]), 3),
                        "Model_version": meta["model_version"],
                        "Decided_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
                    st.success(f"Decision recorded for {rid}.")
                    st.rerun()

    if decisions:
        st.subheader("Decision log (this session)")
        log = pd.DataFrame(decisions)
        st.dataframe(log, hide_index=True, use_container_width=True)
        st.download_button("Download decision log (CSV)", log.to_csv(index=False).encode("utf-8"),
                           file_name="decision_log.csv", mime="text/csv")
        st.caption("The log lives in session memory only. A production system would write it to a database.")
