"""Student Service Request Monitoring - Streamlit deployment of the midterm model.

Run locally:  streamlit run app.py
"""
import pandas as pd
import streamlit as st

from modules import (dashboard, data_processing as dp, exploratory_analysis,
                     prediction, system_design)

st.set_page_config(page_title="SLA Breach Monitoring - Student Services", page_icon="📋", layout="wide")

PAGES = ["Home", "Dataset", "Data Analysis", "Prediction", "Dashboard", "System Design"]
ROLES = {"Student": (False, False), "Student Services Staff": (True, True), "System Administrator": (True, True)}

page = st.sidebar.selectbox("Module", PAGES)
role = st.sidebar.selectbox("Role (demo only)", list(ROLES), index=1,
                            help="Demonstrates role-based access. This is not real authentication.")
can_view, can_record = ROLES[role]
st.sidebar.caption("Advisory predictions only - staff make the final decision.")


@st.cache_data(show_spinner=False)
def read_upload(raw: bytes) -> pd.DataFrame:
    import io
    return pd.read_csv(io.BytesIO(raw))


@st.cache_data(show_spinner=False)
def clean_cached(df: pd.DataFrame) -> pd.DataFrame:
    return dp.clean_dataframe(df)


def active_clean_df() -> pd.DataFrame:
    """Uploaded data if valid, otherwise the bundled midterm dataset."""
    if "raw_df" not in st.session_state:
        st.session_state["raw_df"], st.session_state["source"] = dp.load_sample(), "Bundled midterm dataset"
    return clean_cached(st.session_state["raw_df"])


def restricted():
    st.warning(f"The **{role}** role cannot open this module. Choose Student Services Staff or "
               "System Administrator in the sidebar (demo roles).")


pipeline, meta, load_warning = prediction.load_model()

if page == "Home":
    st.title("📋 Student Service Request Monitoring")
    st.subheader("Predicting SLA breaches at submission - deployed from the midterm exam")
    st.markdown(
        "This app turns the **Section B midterm** (Decision Tree, 40 service requests) into a working "
        "decision-support tool for student services staff.\n\n"
        "| Module | What it does |\n|---|---|\n"
        "| Dataset | Upload a CSV, validate the schema, see nulls and duplicates |\n"
        "| Data Analysis | Descriptive statistics, class balance, interactive charts |\n"
        "| Prediction | Enter a new request and get a breach-risk estimate from the saved model |\n"
        "| Dashboard | KPIs, review queue, recommended follow-up, decision log |\n"
        "| System Design | Requirements, actors, use case diagram, decision workflow |\n")
    c = st.columns(4)
    c[0].metric("Model", meta["model_version"].split("-")[0])
    c[1].metric("Accuracy (8 test cases)", f"{meta['metrics']['accuracy']:.3f}")
    c[2].metric("Recall", f"{meta['metrics']['recall']:.3f}")
    c[3].metric("Training rows", meta["n_train"])
    st.info("**Human in the loop:** predictions are advisory. The app never approves, rejects or denies a service. "
            "Data is synthetic; do not upload real student records to a public deployment.")

elif page == "Dataset":
    if not can_view:
        restricted()
    else:
        st.header("Dataset Upload and Validation")
        up = st.file_uploader("Upload a CSV with the 14 midterm columns", type="csv")
        if up is not None:
            try:
                df_up = read_upload(up.getvalue())
            except pd.errors.EmptyDataError:
                st.error("The file is empty - no columns could be read.")
                df_up = None
            except (pd.errors.ParserError, UnicodeDecodeError) as exc:
                st.error(f"This file could not be read as a CSV: {exc}")
                df_up = None
            if df_up is not None:
                res = dp.validate_schema(df_up)
                for e in res["errors"]:
                    st.error(e)
                for w in res["warnings"]:
                    st.warning(w)
                if res["ok"]:
                    st.session_state["raw_df"], st.session_state["source"] = df_up, f"Uploaded: {up.name}"
                    st.success(f"Schema valid - {len(df_up)} rows loaded.")
                else:
                    st.info("The file was NOT loaded; the previous dataset stays active.")
        if st.button("Reset to bundled midterm dataset"):
            st.session_state["raw_df"], st.session_state["source"] = dp.load_sample(), "Bundled midterm dataset"
        raw = st.session_state.get("raw_df", dp.load_sample())
        st.caption(f"Active dataset: **{st.session_state.get('source', 'Bundled midterm dataset')}**")
        a, b = st.columns(2)
        a.metric("Rows", len(raw))
        b.metric("Columns", raw.shape[1])
        st.subheader("Preview")
        st.dataframe(raw.head(20), use_container_width=True)
        s = dp.null_duplicate_summary(raw)
        st.subheader("Null and duplicate summary")
        d = st.columns(3)
        d[0].metric("Total null cells", s["total_nulls"])
        d[1].metric("Duplicate Request_IDs", s["duplicate_ids"])
        d[2].metric("Fully duplicated rows", s["duplicate_rows"])
        if s["total_nulls"]:
            st.dataframe(s["null_table"])
        else:
            st.success("No missing values.")
        st.caption("Cleaning applied before analysis (from the midterm): trim/standardise text, drop duplicate IDs, "
                   "negative numbers become NaN, weekday and queue category engineered, SLA_Breached recalculated "
                   "for completed requests.")

elif page == "Data Analysis":
    if not can_view:
        restricted()
    else:
        exploratory_analysis.render(active_clean_df())

elif page == "Prediction":
    if not can_view:
        restricted()
    else:
        prediction.render(pipeline, meta, load_warning, can_record, active_clean_df())

elif page == "Dashboard":
    if not can_view:
        restricted()
    else:
        dashboard.render(pipeline, active_clean_df(), meta, can_record, role)

elif page == "System Design":
    system_design.render()
