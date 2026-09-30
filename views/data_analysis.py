import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor

from core.data import SIMULATORS
from core.utils import (PALETTE, fmt_table, get_df, log_result, need_data, numeric_cols,
                        set_df, truth)


# ============================================================================= Load data
def page_load():
    st.header("Load data")
    src = st.radio("Source", ["Simulate a dataset", "Upload CSV"], horizontal=True)
    if src == "Upload CSV":
        f = st.file_uploader("CSV file", type=["csv"])
        if f is not None and st.session_state.get("upload_name") != f.name:
            set_df(pd.read_csv(f), f.name)
            st.session_state.upload_name = f.name
    else:
        c1, c2, c3 = st.columns([3, 1, 1])
        kind = c1.selectbox("Scenario", list(SIMULATORS))
        n = c2.number_input("n", 50, 50000, 1000, step=50)
        seed = c3.number_input("Seed", 0, 10**6, 2026)
        if st.button("Generate dataset", type="primary"):
            df, tr = SIMULATORS[kind](int(n), int(seed))
            set_df(df, f"{kind} (n={n}, seed={seed})", tr)
            st.session_state.upload_name = None

    df = get_df()
    if df is None:
        st.caption("No dataset loaded yet.")
        return

    st.success(f"Active dataset: **{st.session_state.df_name}** · "
               f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    if truth():
        with st.expander("Known truth for this simulated dataset"):
            st.json({k: v for k, v in truth().items() if k != "cate"})
    st.dataframe(df.head(200), width="stretch")

    c1, c2 = st.columns(2)
    c1.download_button("Download active data (CSV)", df.to_csv(index=False).encode(),
                       "statml_data.csv", "text/csv")
    orig = st.session_state.get("df_original")
    if orig is not None and not orig.equals(df):
        if c2.button("Restore original data"):
            set_df(orig.copy(), st.session_state.df_name.replace(" [modified]", ""),
                   truth(), keep_original=True)
            st.rerun()

    with st.expander("Inject missing values (to practise missing-data methods)"):
        c1, c2, c3 = st.columns([2, 1, 1])
        cols = c1.multiselect("Columns to receive missingness", df.columns)
        rate = c2.slider("Missing rate", 0.05, 0.6, 0.2, 0.05)
        mech = c3.selectbox("Mechanism", ["MCAR", "MAR"])
        driver = None
        if mech == "MAR":
            driver = st.selectbox("Missingness depends on", numeric_cols(df))
        if st.button("Inject") and cols:
            rng = np.random.default_rng(1)
            d = df.copy()
            for c in cols:
                if mech == "MCAR":
                    p = np.full(len(d), rate)
                else:
                    r = d[driver].rank(pct=True).fillna(0.5).to_numpy()
                    p = np.clip(2 * rate * r, 0, 0.95)
                d.loc[rng.random(len(d)) < p, c] = np.nan
            set_df(d, st.session_state.df_name + " [modified]", truth(), keep_original=True)
            st.rerun()


# ============================================================================= Diagnostics
def page_diagnostics():
    st.header("Diagnostics")
    need_data()
    df = get_df()
    num = numeric_cols(df)

    t1, t2, t3, t4 = st.tabs(["Overview", "Outliers", "Collinearity", "Distribution shape"])
    with t1:
        ov = pd.DataFrame({
            "type": df.dtypes.astype(str),
            "unique": df.nunique(),
            "missing %": 100 * df.isna().mean(),
            "constant": df.nunique() <= 1,
        })
        c1, c2, c3 = st.columns(3)
        c1.metric("Duplicate rows", int(df.duplicated().sum()))
        c2.metric("Constant columns", int(ov.constant.sum()))
        c3.metric("Columns with missing values", int((ov["missing %"] > 0).sum()))
        st.dataframe(fmt_table(ov, 1), width="stretch")
    with t2:
        k = st.slider("IQR multiplier", 1.0, 3.0, 1.5, 0.5)
        rows = []
        for c in num:
            x = df[c].dropna()
            q1, q3 = x.quantile([.25, .75])
            lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
            rows.append(dict(variable=c, lower_fence=lo, upper_fence=hi,
                             n_outliers=int(((x < lo) | (x > hi)).sum()),
                             max_abs_z=float(np.abs(stats.zscore(x)).max()) if x.std() > 0 else 0))
        out = pd.DataFrame(rows).set_index("variable").sort_values("n_outliers", ascending=False)
        st.dataframe(fmt_table(out), width="stretch")
        if num:
            v = st.selectbox("Inspect", num)
            st.plotly_chart(px.box(df, y=v, points="outliers"), width="stretch")
    with t3:
        if len(num) < 2:
            st.write("Need at least two numeric columns.")
        else:
            thr = st.slider("Flag |r| above", 0.5, 0.99, 0.8, 0.01)
            r = df[num].corr()
            pairs = (r.where(np.triu(np.ones(r.shape, bool), 1)).stack()
                     .rename("r").reset_index().rename(columns={"level_0": "var 1", "level_1": "var 2"}))
            st.dataframe(fmt_table(pairs[pairs.r.abs() > thr].sort_values("r", key=abs, ascending=False)),
                         width="stretch")
            st.subheader("Variance inflation factors")
            d = df[num].dropna()
            d = d.loc[:, d.std() > 0]
            if d.shape[1] > 60:
                st.caption("VIF shown for the first 60 numeric columns.")
                d = d.iloc[:, :60]
            Xc = np.column_stack([np.ones(len(d)), d.to_numpy()])
            vif = pd.Series([variance_inflation_factor(Xc, i + 1) for i in range(d.shape[1])],
                            index=d.columns, name="VIF").sort_values(ascending=False)
            st.dataframe(fmt_table(vif.to_frame(), 2), width="stretch")
            st.caption("VIF above 5–10 suggests problematic collinearity.")
    with t4:
        sh = pd.DataFrame({c: dict(skewness=stats.skew(df[c].dropna()),
                                   excess_kurtosis=stats.kurtosis(df[c].dropna()),
                                   shapiro_p=stats.shapiro(df[c].dropna().sample(
                                       min(5000, df[c].notna().sum()), random_state=0))[1]
                                   if df[c].notna().sum() >= 3 else np.nan)
                           for c in num}).T
        st.dataframe(fmt_table(sh), width="stretch")
        st.caption("Shapiro–Wilk uses at most 5,000 observations. With large n, tiny departures "
                   "are 'significant'; look at the skewness and Q–Q plots too.")


# ============================================================================= Missing data
def page_missing():
    st.header("Missing data")
    need_data()
    df = get_df()
    miss = df.isna().mean().sort_values(ascending=False)
    if miss.sum() == 0:
        st.info("No missing values in the active dataset. You can add some on the "
                "**Load data** page under *Inject missing values*.")
        return

    t1, t2, t3 = st.tabs(["Patterns", "Mechanism check", "Imputation"])
    with t1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Complete cases", f"{df.dropna().shape[0]:,} / {len(df):,}")
        c2.metric("Cells missing", f"{100 * df.isna().to_numpy().mean():.1f}%")
        c3.metric("Variables with missingness", int((miss > 0).sum()))
        st.plotly_chart(px.bar(miss[miss > 0], labels={"index": "", "value": "fraction missing"}),
                        width="stretch")
        cols = miss[miss > 0].index.tolist()
        pat = (df[cols].isna().astype(int).value_counts().rename("n").reset_index())
        pat["% rows"] = 100 * pat.n / len(df)
        st.subheader("Missingness patterns (1 = missing)")
        st.dataframe(pat.head(20), width="stretch")
        st.plotly_chart(px.imshow(df.isna().T.astype(int), aspect="auto",
                                  color_continuous_scale=["#EEF1F5", "#2B4C7E"]), width="stretch")

    with t2:
        st.write("Compare observed variables between rows where the target variable is missing "
                 "and rows where it is observed. Clear differences are evidence against MCAR. "
                 "Data alone cannot distinguish MAR from MNAR.")
        target = st.selectbox("Variable with missingness", miss[miss > 0].index)
        R = df[target].isna()
        rows = []
        for c in df.columns.drop(target):
            x = df[c]
            if pd.api.types.is_numeric_dtype(x) and x.nunique() > 5:
                a, b = x[R].dropna(), x[~R].dropna()
                if len(a) > 1 and len(b) > 1:
                    p = stats.ttest_ind(a, b, equal_var=False).pvalue
                    s = (a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2)
                    rows.append(dict(variable=c, test="Welch t", SMD=s, p=p))
            else:
                tab = pd.crosstab(R, x)
                if tab.shape[0] == 2 and tab.shape[1] > 1:
                    rows.append(dict(variable=c, test="Chi-square", SMD=np.nan,
                                     p=stats.chi2_contingency(tab)[1]))
        res = pd.DataFrame(rows).set_index("variable").sort_values("p")
        st.dataframe(fmt_table(res, 4), width="stretch")

    with t3:
        c1, c2, c3 = st.columns(3)
        method = c1.selectbox("Method", ["Complete cases (listwise deletion)", "Mean / mode",
                                         "Median / mode", "k-nearest neighbours",
                                         "Iterative (MICE-style, chained regressions)"])
        k = c2.slider("k (KNN)", 1, 25, 5)
        seed = c3.number_input("Seed", 0, 10**6, 0)
        if st.button("Impute", type="primary"):
            num = numeric_cols(df)
            cat = [c for c in df.columns if c not in num]
            d = df.copy()
            if method.startswith("Complete"):
                d = d.dropna()
            else:
                for c in cat:
                    if d[c].isna().any():
                        d[c] = d[c].fillna(d[c].mode().iloc[0])
                if num:
                    imp = {"Mean / mode": SimpleImputer(strategy="mean"),
                           "Median / mode": SimpleImputer(strategy="median"),
                           "k-nearest neighbours": KNNImputer(n_neighbors=k),
                           }.get(method, IterativeImputer(max_iter=20, random_state=seed,
                                                          sample_posterior=False))
                    d[num] = imp.fit_transform(d[num])
            st.session_state.res_imputed = (method, d)

        if "res_imputed" in st.session_state:
            method, d = st.session_state.res_imputed
            st.write(f"**{method}**: {len(d):,} rows, {int(d.isna().sum().sum())} missing cells remain.")
            num_miss = [c for c in miss[miss > 0].index if c in numeric_cols(df)]
            if num_miss and not method.startswith("Complete"):
                v = st.selectbox("Compare distributions", num_miss)
                was = df[v].isna()
                comp = pd.concat([pd.DataFrame({v: df[v].dropna(), "values": "observed"}),
                                  pd.DataFrame({v: d.loc[was, v], "values": "imputed"})])
                st.plotly_chart(px.histogram(comp, x=v, color="values", barmode="overlay",
                                             histnorm="probability density", opacity=.6,
                                             color_discrete_sequence=PALETTE), width="stretch")
            st.caption("Single imputation understates uncertainty. For inference, use multiple "
                       "imputation (e.g. mice in R) and combine with Rubin's rules.")
            if st.button("Use imputed data as active dataset"):
                set_df(d, st.session_state.df_name + " [imputed]",
                       truth(), keep_original=True)
                log_result("Imputation", {"method": method, "k": k},
                           tables={"Missing fraction before": miss.to_frame("fraction")})
                st.rerun()


# ============================================================================= Visualization
def page_visualization():
    st.header("Visualization")
    need_data()
    df = get_df()
    num = numeric_cols(df)
    t1, t2, t3, t4, t5 = st.tabs(["Distribution", "Relationships", "Group comparison",
                                  "Correlation", "Scatter matrix"])
    with t1:
        col = st.selectbox("Variable", df.columns, key="viz_var")
        by = st.selectbox("Colour by", ["—"] + df.columns.tolist(), key="viz_by")
        color = None if by == "—" else by
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() > 10:
            c1, c2 = st.columns(2)
            c1.plotly_chart(px.histogram(df, x=col, color=color, marginal="box", barmode="overlay",
                                         opacity=.7, color_discrete_sequence=PALETTE), width="stretch")
            qq = stats.probplot(df[col].dropna(), dist="norm")
            fig = go.Figure(go.Scatter(x=qq[0][0], y=qq[0][1], mode="markers"))
            fig.add_trace(go.Scatter(x=qq[0][0], y=qq[1][0] * qq[0][0] + qq[1][1], mode="lines",
                                     line=dict(dash="dash")))
            fig.update_layout(title="Normal Q–Q", showlegend=False,
                              xaxis_title="Theoretical quantiles", yaxis_title="Sample quantiles")
            c2.plotly_chart(fig, width="stretch")
        else:
            st.plotly_chart(px.histogram(df, x=col, color=color, barmode="group",
                                         color_discrete_sequence=PALETTE), width="stretch")
    with t2:
        if len(num) < 2:
            st.write("Need two numeric columns.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            x = c1.selectbox("x", num, key="rel_x")
            y = c2.selectbox("y", num, index=1, key="rel_y")
            color = c3.selectbox("Colour", ["—"] + df.columns.tolist(), key="rel_c")
            trend = c4.selectbox("Smoother", ["lowess", "ols", "none"])
            st.plotly_chart(px.scatter(df, x=x, y=y, color=None if color == "—" else color,
                                       trendline=None if trend == "none" else trend, opacity=.6,
                                       color_discrete_sequence=PALETTE), width="stretch")
    with t3:
        groups = [c for c in df.columns if df[c].nunique() <= 12]
        if not groups or not num:
            st.write("Need a grouping column (≤ 12 levels) and a numeric column.")
        else:
            c1, c2, c3 = st.columns(3)
            g = c1.selectbox("Group", groups)
            v = c2.selectbox("Measure", [c for c in num if c != g])
            kind = c3.radio("Plot", ["Box", "Violin"], horizontal=True)
            f = px.box if kind == "Box" else px.violin
            st.plotly_chart(f(df, x=g, y=v, color=g, points="outliers",
                              color_discrete_sequence=PALETTE), width="stretch")
            samples = [s.dropna() for _, s in df.groupby(g)[v]]
            samples = [s for s in samples if len(s) > 1]
            if len(samples) == 2:
                st.write(f"Welch t-test p = {stats.ttest_ind(*samples, equal_var=False).pvalue:.4g} · "
                         f"Mann–Whitney p = {stats.mannwhitneyu(*samples).pvalue:.4g}")
            elif len(samples) > 2:
                st.write(f"One-way ANOVA p = {stats.f_oneway(*samples).pvalue:.4g} · "
                         f"Kruskal–Wallis p = {stats.kruskal(*samples).pvalue:.4g}")
    with t4:
        if len(num) < 2:
            st.write("Need at least two numeric columns.")
        else:
            method = st.radio("Method", ["pearson", "spearman", "kendall"], horizontal=True)
            cols = num[:30]
            if len(num) > 30:
                st.caption("Showing the first 30 numeric columns.")
            st.plotly_chart(px.imshow(df[cols].corr(method=method), zmin=-1, zmax=1,
                                      color_continuous_scale="RdBu_r",
                                      text_auto=".2f" if len(cols) <= 12 else False), width="stretch")
    with t5:
        cols = st.multiselect("Variables", num, default=num[:4])
        color = st.selectbox("Colour", ["—"] + df.columns.tolist(), key="spm_c")
        if len(cols) >= 2:
            fig = px.scatter_matrix(df, dimensions=cols, color=None if color == "—" else color,
                                    opacity=.5, color_discrete_sequence=PALETTE)
            fig.update_traces(diagonal_visible=False, marker=dict(size=4))
            fig.update_layout(height=180 * len(cols))
            st.plotly_chart(fig, width="stretch")
