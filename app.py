"""
StatML Lab — an interactive workbench for statistics and machine learning.

Run:  streamlit run app.py
"""
import warnings

import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

from core.utils import get_df  # noqa: E402
from views import data_analysis as da, methods as sm, ml, reports, simulation  # noqa: E402

st.set_page_config(page_title="StatML Lab", page_icon="📐", layout="wide")

P = st.Page
nav = st.navigation({
    "Data Analysis": [
        P(da.page_load, title="Load data", icon=":material/upload_file:", url_path="load", default=True),
        P(da.page_diagnostics, title="Diagnostics", icon=":material/fact_check:", url_path="diagnostics"),
        P(da.page_missing, title="Missing data", icon=":material/grid_off:", url_path="missing"),
        P(da.page_visualization, title="Visualization", icon=":material/insights:", url_path="visualization"),
    ],
    "Statistical Methods": [
        P(sm.page_glm, title="Regression models (GLM)", icon=":material/functions:", url_path="glm"),
        P(sm.page_selection, title="Variable selection", icon=":material/filter_alt:", url_path="selection"),
        P(sm.page_causal, title="Causal inference", icon=":material/call_split:", url_path="causal"),
        P(sm.page_survival, title="Survival models", icon=":material/timeline:", url_path="survival"),
        P(sm.page_counts, title="Count models", icon=":material/tag:", url_path="counts"),
    ],
    "Machine Learning": [
        P(ml.page_prediction, title="Prediction", icon=":material/trending_up:", url_path="prediction"),
        P(ml.page_classification, title="Classification", icon=":material/category:", url_path="classification"),
        P(ml.page_clustering, title="Clustering & PCA", icon=":material/bubble_chart:", url_path="clustering"),
    ],
    "Simulation Laboratory": [
        P(simulation.page_simulation, title="Simulation studies", icon=":material/science:", url_path="simulation"),
    ],
    "Reports & Reproducibility": [
        P(reports.page_reports, title="Reports", icon=":material/description:", url_path="reports"),
    ],
})

with st.sidebar:
    st.markdown("### StatML Lab")
    df = get_df()
    if df is not None:
        st.caption(f"**Active data:** {st.session_state.get('df_name', 'data')}  \n"
                   f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        st.caption("No data loaded.")
    n_log = len(st.session_state.get("log", []))
    if n_log:
        st.caption(f"{n_log} analyses in the report")

nav.run()
