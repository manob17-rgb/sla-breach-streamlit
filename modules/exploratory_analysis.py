"""Module 2 - Exploratory data analysis (descriptive stats, class distribution,
and five interactive Plotly charts) built from the midterm EDA."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from modules.data_processing import QUEUE_LABELS


# ---------------------------------------------------------------- pure logic
def labelled(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Rows with a known SLA outcome (completed requests)."""
    out = clean_df[clean_df["SLA_Breached"].notna()].copy()
    out["SLA_Breached"] = out["SLA_Breached"].astype(int)
    out["Outcome"] = out["SLA_Breached"].map({0: "On time", 1: "Breached"})
    return out


def breach_by(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby(col)["SLA_Breached"].agg(Requests="count", Breach_Rate="mean").reset_index()
    return g.sort_values("Breach_Rate", ascending=False)


# ----------------------------------------------------------------------- UI
def render(clean_df: pd.DataFrame):
    st.header("Exploratory Data Analysis")
    df = labelled(clean_df)
    if df.empty:
        st.error("No completed requests with an SLA outcome were found, so outcome charts cannot be drawn.")
        return

    # interactive filters
    with st.expander("Filters (apply to every chart below)", expanded=False):
        offices = sorted(df["Assigned_Office"].dropna().unique())
        prios = sorted(df["Priority"].dropna().unique())
        sel_o = st.multiselect("Assigned office", offices, default=offices)
        sel_p = st.multiselect("Priority", prios, default=prios)
    df = df[df["Assigned_Office"].isin(sel_o) & df["Priority"].isin(sel_p)]
    if df.empty:
        st.warning("The filters removed every row. Select at least one office and priority.")
        return

    st.subheader("Descriptive statistics")
    st.dataframe(df[["Queue_Length", "Processing_Days", "SLA_Days", "Satisfaction_Score"]].describe().round(2))
    with st.expander("Category counts"):
        for col in ["Service_Type", "Priority", "Assigned_Office", "Request_Channel", "Requester_Type"]:
            if col in df:
                st.write(f"**{col}**")
                st.dataframe(df[col].value_counts(dropna=False).rename("Count").to_frame())

    st.subheader("Class distribution (target: SLA_Breached)")
    dist = df["Outcome"].value_counts().rename_axis("Outcome").reset_index(name="Count")
    c1, c2 = st.columns([1, 2])
    c1.dataframe(dist, hide_index=True)
    c1.metric("Breach rate", f"{df['SLA_Breached'].mean():.1%}")
    c2.plotly_chart(px.pie(dist, names="Outcome", values="Count", hole=0.45,
                           title="On time vs breached"), use_container_width=True)

    st.subheader("Interactive charts")
    t1, t2, t3, t4, t5 = st.tabs(["By service", "By office", "Queue vs time",
                                  "Queue scatter", "Priority"])
    with t1:
        s = breach_by(df, "Service_Type")
        fig = px.bar(s, x="Service_Type", y="Breach_Rate", hover_data=["Requests"],
                     title="SLA breach rate by service type", text_auto=".0%")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
        top = s[s["Breach_Rate"] == s["Breach_Rate"].max()]["Service_Type"].tolist()
        st.caption("Highest breach rate: " + ", ".join(top) + ".")
    with t2:
        o = breach_by(df, "Assigned_Office")
        fig = px.bar(o, x="Assigned_Office", y="Breach_Rate", hover_data=["Requests"],
                     title="SLA breach rate by assigned office", text_auto=".0%")
        fig.update_yaxes(tickformat=".0%", range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
    with t3:
        fig = px.box(df, x="Queue_Category", y="Processing_Days", points="all", color="Queue_Category",
                     category_orders={"Queue_Category": QUEUE_LABELS},
                     hover_data=["Request_ID", "Service_Type"], title="Processing days by queue category")
        st.plotly_chart(fig, use_container_width=True)
        med = df.groupby("Queue_Category")["Processing_Days"].median()
        if {"Low", "Critical"} <= set(med.index) and med["Low"] > 0:
            st.caption(f"Median processing time is {med['Critical'] / med['Low']:.1f}x higher in a Critical "
                       "queue than in a Low queue.")
    with t4:
        fig = px.scatter(df, x="Queue_Length", y="Processing_Days", color="Outcome",
                         hover_data=["Request_ID", "Service_Type", "Assigned_Office"],
                         title="Queue length vs processing days")
        st.plotly_chart(fig, use_container_width=True)
    with t5:
        p = df.groupby(["Priority", "Outcome"]).size().reset_index(name="Count")
        fig = px.bar(p, x="Priority", y="Count", color="Outcome", barmode="group",
                     title="Outcome by priority")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("In this (synthetic) dataset every High-priority request breached its SLA and no Low-priority "
                   "request did - a strong pattern, but one that real data is unlikely to repeat this cleanly.")
