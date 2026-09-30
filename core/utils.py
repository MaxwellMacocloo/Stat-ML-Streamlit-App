"""Shared helpers used across StatML Lab pages."""
import hashlib
import json
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ACCENT = "#2B4C7E"
PALETTE = ["#2B4C7E", "#C0504D", "#4F9D69", "#8064A2", "#D9A21B", "#3FA7C4", "#7F7F7F"]

try:
    import xgboost  # noqa: F401
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


# ----------------------------------------------------------------------------- data state
def get_df():
    return st.session_state.get("df")


def set_df(df, name, truth=None, keep_original=False):
    st.session_state.df = df.reset_index(drop=True) if not keep_original else df
    st.session_state.df_name = name
    if not keep_original:
        st.session_state.df_original = st.session_state.df.copy()
        st.session_state.truth = truth or {}
    for k in [k for k in st.session_state if str(k).startswith("res_")]:
        del st.session_state[k]


def need_data():
    if get_df() is None:
        st.info("Load a dataset on the **Load data** page first.")
        st.stop()


def truth():
    return st.session_state.get("truth", {}) or {}


def numeric_cols(df):
    return df.select_dtypes(include=np.number).columns.tolist()


def is_binary(s):
    return s.dropna().nunique() == 2


def binary_cols(df):
    return [c for c in df.columns if is_binary(df[c])]


def to01(s):
    """Map a two-level series to 0/1 (higher/later level = 1)."""
    levels = sorted(s.dropna().unique())
    return (s == levels[-1]).astype(int), levels[-1]


def design_matrix(df, outcome, predictors, extra=()):
    cols = [outcome] + list(predictors) + [c for c in extra if c not in predictors and c != outcome]
    d = df[cols].dropna()
    X = pd.get_dummies(d[list(predictors)], drop_first=True, dtype=float)
    X.columns = [str(c) for c in X.columns]
    return X, d[outcome], len(df) - len(d), d


def data_fingerprint(df=None):
    df = get_df() if df is None else df
    if df is None:
        return None
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values.tobytes()).hexdigest()[:12]


def fmt_table(df, digits=3):
    return df.style.format(precision=digits, na_rep="—")


# ----------------------------------------------------------------------------- plots
def forest_plot(tab, est, lo, hi, ref, xlabel, log=True):
    tab = tab.iloc[::-1]
    fig = go.Figure(go.Scatter(
        x=tab[est], y=[str(i) for i in tab.index], mode="markers",
        marker=dict(size=10, color=ACCENT),
        error_x=dict(type="data", symmetric=False, array=tab[hi] - tab[est],
                     arrayminus=tab[est] - tab[lo], color=ACCENT),
        hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.add_vline(x=ref, line_dash="dash", line_color="grey")
    fig.update_layout(xaxis_title=xlabel, height=140 + 32 * len(tab),
                      margin=dict(l=10, r=10, t=20, b=10), showlegend=False)
    if log:
        fig.update_xaxes(type="log")
    return fig


# ----------------------------------------------------------------------------- report log
def _jsonable(x):
    return json.loads(json.dumps(x, default=str))


def log_result(title, params, tables=None, figs=None, notes=None):
    """Append an analysis record to the session report."""
    df = get_df()
    entry = dict(
        time=datetime.now().isoformat(timespec="seconds"),
        title=title,
        params=_jsonable(params),
        dataset=st.session_state.get("df_name"),
        shape=None if df is None else list(df.shape),
        fingerprint=data_fingerprint(),
        tables={k: v.copy() for k, v in (tables or {}).items()},
        figs=list(figs or []),
        notes=notes,
    )
    st.session_state.setdefault("log", []).append(entry)
    st.toast(f"Added to report: {title}")
