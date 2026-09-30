import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core import causal as cz
from core import selection as sel
from core.data import expit, sim_highdim
from core.utils import PALETTE, fmt_table, log_result


def page_simulation():
    st.header("Simulation laboratory")
    t1, t2, t3 = st.tabs(["Variable selection: FDR and power", "Causal estimators: double robustness",
                          "Confidence interval coverage"])
    with t1:
        _selection_study()
    with t2:
        _causal_study()
    with t3:
        _coverage_study()


# ----------------------------------------------------------------------------- selection
def _selection_study():
    st.write("Repeatedly simulate sparse linear models with AR(1)-correlated predictors and compare "
             "how often each method selects noise variables (FDR) and how many true signals it finds "
             "(power).")
    c = st.columns(6)
    n = c[0].number_input("n", 50, 5000, 300, 50)
    p = c[1].number_input("p", 10, 500, 50, 10)
    k = c[2].number_input("true signals k", 1, 50, 15)
    rho = c[3].slider("correlation ρ", 0.0, 0.9, 0.5, 0.1)
    amp = c[4].number_input("signal size", 0.05, 3.0, 0.4, 0.05)
    reps = c[5].number_input("replicates", 2, 500, 10)
    c = st.columns(5)
    methods = c[0].multiselect("Methods", sel.PENALTIES + ["Knockoff+"],
                               default=["Lasso", "SCAD", "TLP", "Knockoff+"])
    rule_lab = c[1].selectbox("λ choice", ["EBIC", "BIC", "CV (1-SE rule)", "CV (minimum error)"], key="vs_rule")
    rule = {"EBIC": "ebic", "BIC": "bic", "CV (1-SE rule)": "1se", "CV (minimum error)": "min"}[rule_lab]
    q = c[2].slider("Knockoff target FDR", 0.05, 0.3, 0.1, 0.05)
    tau = c[3].number_input("TLP τ", 0.01, 3.0, 0.1, 0.05, key="vs_tau")
    seed = c[4].number_input("Seed", 0, 10**6, 11, key="vs_seed")
    st.caption("Penalized methods search 15 λ values. CV rules are slower (5 folds) than (E)BIC. "
               "Knockoff+ cannot select anything unless it can make at least 1/q discoveries, so it "
               "needs k ≥ 1/q true signals to have power.")
    if not st.button("Run selection study", type="primary") or not methods:
        return
    rows = []
    prog = st.progress(0.0)
    for r in range(int(reps)):
        df, tr = sim_highdim(int(n), int(seed) + r, int(p), int(k), rho, amp)
        X = df.drop(columns="y").to_numpy()
        X = (X - X.mean(0)) / X.std(0)
        y = df.y.to_numpy()
        names = np.array(df.columns[1:])
        for m in methods:
            if m == "Knockoff+":
                chosen = sel.run_knockoffs(X, y, "gaussian", q, True, "lasso", int(seed) + r)["selected"]
            else:
                fit = sel.cv_penalized(X, y, m, "gaussian", 5, int(seed) + r, n_lam=15, rule=rule, n_iter=2,
                                       path=False, tau=tau)
                chosen = np.flatnonzero(fit["coef"])
            fdp, pw = sel.fdp_power(names[chosen], tr["active"])
            rows.append(dict(rep=r, method=m, FDP=fdp, power=pw, selected=len(chosen)))
        prog.progress((r + 1) / reps, f"Replicate {r + 1} of {reps}")
    prog.empty()
    res = pd.DataFrame(rows)
    summ = res.groupby("method", sort=False).agg(FDR=("FDP", "mean"), FDR_se=("FDP", "sem"),
                                                 power=("power", "mean"), power_se=("power", "sem"),
                                                 mean_selected=("selected", "mean"))
    long = pd.concat([pd.DataFrame(dict(method=summ.index, metric="FDR", value=summ.FDR, se=summ.FDR_se)),
                      pd.DataFrame(dict(method=summ.index, metric="Power", value=summ.power, se=summ.power_se))])
    fig = px.bar(long, x="method", y="value", color="metric", barmode="group", error_y=1.96 * long.se,
                 color_discrete_sequence=PALETTE, range_y=[0, 1.05])
    if "Knockoff+" in methods:
        fig.add_hline(y=q, line_dash="dash", annotation_text=f"q = {q}")
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fmt_table(summ), width="stretch")
    log_result("Simulation: variable selection", dict(n=n, p=p, k=k, rho=rho, amp=amp, reps=reps, methods=methods,
                                                      rule=rule, q=q, tau=tau, seed=seed),
               tables={"FDR and power": summ}, figs=[fig])


# ----------------------------------------------------------------------------- causal
def _ks(n, rng, tau):
    """Kang & Schafer (2007)-type design with a treatment effect tau."""
    X = rng.standard_normal((n, 4))
    e = expit(-X[:, 0] + 0.5 * X[:, 1] - 0.25 * X[:, 2] - 0.1 * X[:, 3])
    t = rng.binomial(1, e)
    y = 210 + 27.4 * X[:, 0] + 13.7 * (X[:, 1] + X[:, 2] + X[:, 3]) + tau * t + rng.standard_normal(n)
    Z = np.column_stack([np.exp(X[:, 0] / 2), X[:, 1] / (1 + np.exp(X[:, 0])) + 10,
                         (X[:, 0] * X[:, 2] / 25 + 0.6) ** 3, (X[:, 1] + X[:, 3] + 20) ** 2])
    return X, Z, t, y


def _causal_study():
    st.write("Kang & Schafer (2007)-style design. Each model is fitted either on the true covariates "
             "(**correct**) or on nonlinear transformations of them (**misspecified**). AIPW should stay "
             "unbiased when at least one of the two models is correct.")
    c = st.columns(5)
    n = c[0].number_input("n", 100, 20000, 1000, 100, key="cs_n")
    reps = c[1].number_input("replicates", 20, 5000, 300, 20, key="cs_reps")
    tau = c[2].number_input("true ATE", -20.0, 20.0, 10.0, 1.0)
    trim = c[3].slider("Propensity clipping ε", 0.0, 0.05, 0.01, 0.005, key="cs_trim")
    seed = c[4].number_input("Seed", 0, 10**6, 5, key="cs_seed")
    if not st.button("Run causal study", type="primary"):
        return
    rng = np.random.default_rng(int(seed))
    scen = {"both correct": (False, False), "propensity misspecified": (True, False),
            "outcome misspecified": (False, True), "both misspecified": (True, True)}
    rows = []
    prog = st.progress(0.0)
    for r in range(int(reps)):
        X, Z, t, y = _ks(int(n), rng, tau)
        for s, (ps_bad, or_bad) in scen.items():
            Xp, Xo = (Z if ps_bad else X), (Z if or_bad else X)
            e = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=5000)).fit(Xp, t).predict_proba(Xp)[:, 1]
            mu1 = LinearRegression().fit(Xo[t == 1], y[t == 1]).predict(Xo)
            mu0 = LinearRegression().fit(Xo[t == 0], y[t == 0]).predict(Xo)
            tab, _ = cz.ate_table(y, t, e, mu0, mu1, trim=max(trim, 1e-6))
            for est, row in tab.iterrows():
                cover = np.nan if pd.isna(row.SE) else float(row.lower <= tau <= row.upper)
                rows.append(dict(scenario=s, estimator=est, estimate=row.estimate, covered=cover))
        prog.progress((r + 1) / reps)
    prog.empty()
    res = pd.DataFrame(rows)
    res["error"] = res.estimate - tau
    summ = res.groupby(["scenario", "estimator"], sort=False).agg(
        bias=("error", "mean"), SD=("estimate", "std"),
        RMSE=("error", lambda e: np.sqrt(np.mean(e ** 2))), coverage=("covered", "mean"))
    fig = px.box(res, x="estimator", y="estimate", color="estimator", facet_col="scenario", facet_col_wrap=2,
                 points=False, color_discrete_sequence=PALETTE, height=650)
    fig.add_hline(y=tau, line_dash="dash")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig.update_xaxes(showticklabels=False, title=None)
    st.plotly_chart(fig, width="stretch")
    st.dataframe(fmt_table(summ), width="stretch")
    st.caption("Coverage uses nominal 95% Wald intervals; g-computation has no analytic SE here. "
               "When both models are wrong, AIPW can be worse than either single-model estimator.")
    log_result("Simulation: causal estimators", dict(n=n, reps=reps, tau=tau, trim=trim, seed=seed),
               tables={"Bias / SD / RMSE / coverage": summ.reset_index()}, figs=[fig])


# ----------------------------------------------------------------------------- coverage
def _coverage_study():
    st.write("How often do different 95% intervals for a population mean actually cover the truth?")
    c1, c2, c3 = st.columns(3)
    dist = c1.selectbox("Distribution", ["Normal(0,1)", "Exponential(1)", "Lognormal(0,1)", "t with 3 df",
                                         "Bernoulli(0.1)"])
    ns = c2.multiselect("Sample sizes", [10, 20, 30, 50, 100, 200], default=[10, 30, 100])
    reps = c3.slider("Monte Carlo replicates", 200, 3000, 1000, step=200)
    B = c3.slider("Bootstrap resamples", 200, 2000, 500, step=100)
    seed = c1.number_input("Seed", 0, 10**6, 7, key="cov_seed")
    if not ns or not st.button("Run coverage study", type="primary"):
        return
    rng = np.random.default_rng(seed)
    gen = {"Normal(0,1)": (lambda n: rng.normal(size=n), 0.0),
           "Exponential(1)": (lambda n: rng.exponential(size=n), 1.0),
           "Lognormal(0,1)": (lambda n: rng.lognormal(size=n), np.exp(.5)),
           "t with 3 df": (lambda n: rng.standard_t(3, size=n), 0.0),
           "Bernoulli(0.1)": (lambda n: rng.binomial(1, .1, size=n).astype(float), 0.1)}
    draw, mu = gen[dist]
    out = []
    prog = st.progress(0.0)
    for k, n in enumerate(sorted(ns)):
        hits = {"Wald (z)": 0, "Student t": 0, "Bootstrap percentile": 0}
        widths = {m: [] for m in hits}
        for _ in range(reps):
            x = draw(n)
            m, se = x.mean(), x.std(ddof=1) / np.sqrt(n)
            tq = stats.t.ppf(.975, n - 1)
            boot = x[rng.integers(0, n, size=(B, n))].mean(axis=1)
            ints = {"Wald (z)": (m - 1.96 * se, m + 1.96 * se), "Student t": (m - tq * se, m + tq * se),
                    "Bootstrap percentile": tuple(np.quantile(boot, [.025, .975]))}
            for meth, (lo, hi) in ints.items():
                hits[meth] += lo <= mu <= hi
                widths[meth].append(hi - lo)
        for meth in hits:
            cov = hits[meth] / reps
            out.append(dict(n=n, method=meth, coverage=cov, mc_se=np.sqrt(cov * (1 - cov) / reps),
                            mean_width=np.mean(widths[meth])))
        prog.progress((k + 1) / len(ns))
    prog.empty()
    res = pd.DataFrame(out)
    fig = px.line(res, x="n", y="coverage", color="method", markers=True, error_y=1.96 * res.mc_se, log_x=True,
                  color_discrete_sequence=PALETTE)
    fig.add_hline(y=.95, line_dash="dash", line_color="grey")
    fig.update_layout(yaxis_title="Empirical coverage", yaxis_range=[min(.7, res.coverage.min() - .02), 1])
    st.plotly_chart(fig, width="stretch")
    tab = res.pivot(index="n", columns="method", values=["coverage", "mean_width"])
    st.dataframe(fmt_table(tab), width="stretch")
    log_result("Simulation: CI coverage", dict(distribution=dist, n=ns, reps=reps, B=B, seed=seed),
               tables={"Coverage": res}, figs=[fig])
