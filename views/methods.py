import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test, proportional_hazard_test
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree

from core import causal as cz
from core import selection as sel
from core.utils import (ACCENT, PALETTE, binary_cols, design_matrix, fmt_table, forest_plot,
                        get_df, is_binary, log_result, need_data, numeric_cols, to01, truth)


# ============================================================================= GLM
def page_glm():
    st.header("Regression models (GLM)")
    st.caption("Linear, logistic and Poisson regression with interpretable effect estimates.")
    need_data()
    df = get_df()

    c1, c2 = st.columns([1, 2])
    outcome = c1.selectbox("Outcome", df.columns)
    fams = ["Gaussian (linear)", "Binomial (logistic)", "Poisson (log-linear)"]
    family = c1.selectbox("Family", fams, index=1 if is_binary(df[outcome]) else 0)
    others = [c for c in df.columns if c != outcome]
    preds = c2.multiselect("Predictors", others, default=others[:6])
    robust = c2.checkbox("Robust (HC3) standard errors")
    if not preds:
        st.stop()

    d = df[[outcome] + preds].dropna().copy()
    d.columns = [str(c).replace(" ", "_").replace(".", "_") for c in d.columns]
    y_name, x_names = d.columns[0], list(d.columns[1:])
    if family.startswith("Binomial"):
        if not is_binary(d[y_name]):
            st.error("Logistic regression needs a two-level outcome.")
            st.stop()
        d[y_name], lvl = to01(d[y_name])
        st.caption(f"Modelling P({outcome} = {lvl}).")
    formula = f"{y_name} ~ " + " + ".join(
        x if pd.api.types.is_numeric_dtype(d[x]) else f"C({x})" for x in x_names)
    st.code(formula, language="r")

    fam = {"Gaussian (linear)": sm.families.Gaussian(), "Binomial (logistic)": sm.families.Binomial(),
           "Poisson (log-linear)": sm.families.Poisson()}[family]
    try:
        fit = smf.glm(formula, d, family=fam).fit(cov_type="HC3" if robust else "nonrobust")
    except Exception as e:
        st.error(f"Model failed to fit: {e}")
        st.stop()

    ci = fit.conf_int()
    tab = pd.DataFrame({"estimate": fit.params, "lower": ci[0], "upper": ci[1], "SE": fit.bse,
                        "p": fit.pvalues})
    exp = not family.startswith("Gaussian")
    lab = {"Binomial (logistic)": "Odds ratio", "Poisson (log-linear)": "Rate ratio"}.get(family, "β")
    if exp:
        tab[["estimate", "lower", "upper"]] = np.exp(tab[["estimate", "lower", "upper"]])

    m1, m2, m3 = st.columns(3)
    m1.metric("n used", f"{int(fit.nobs):,}", f"{len(df) - len(d)} dropped (missing)", delta_color="off")
    m2.metric("AIC", f"{fit.aic:.1f}")
    m3.metric("Deviance / df", f"{fit.deviance / fit.df_resid:.3f}")
    if family.startswith("Poisson") and fit.pearson_chi2 / fit.df_resid > 1.5:
        st.warning("Pearson χ²/df > 1.5: the counts look overdispersed. Compare negative binomial "
                   "and zero-inflated models on the **Count models** page.")

    c1, c2 = st.columns(2)
    c1.subheader(f"Estimates ({lab})")
    c1.dataframe(fmt_table(tab, 4), width="stretch")
    c2.subheader("Forest plot")
    fp = forest_plot(tab.drop("Intercept"), "estimate", "lower", "upper", 1 if exp else 0,
                     f"{lab} (95% CI)", log=exp)
    c2.plotly_chart(fp, width="stretch")

    st.subheader("Diagnostics")
    c1, c2 = st.columns(2)
    c1.plotly_chart(px.scatter(x=fit.fittedvalues, y=fit.resid_deviance, opacity=.5, trendline="lowess",
                               labels={"x": "Fitted", "y": "Deviance residual"}), width="stretch")
    cook = fit.get_influence().cooks_distance[0]
    c2.plotly_chart(px.bar(y=cook, labels={"x": "Observation", "y": "Cook's distance"}), width="stretch")
    with st.expander("Full statsmodels output"):
        st.text(fit.summary().as_text())
    if st.button("Add to report"):
        log_result(f"GLM: {family}", {"formula": formula, "robust_se": robust},
                   tables={f"Estimates ({lab})": tab}, figs=[fp])


# ============================================================================= Variable selection
def page_selection():
    st.header("Variable selection")
    need_data()
    df = get_df()
    num = numeric_cols(df)
    if not num:
        st.warning("Need a numeric outcome.")
        st.stop()
    c1, c2 = st.columns([1, 2])
    outcome = c1.selectbox("Outcome", num)
    family = "binomial" if is_binary(df[outcome]) else "gaussian"
    c1.caption(f"Model: **{'logistic' if family == 'binomial' else 'linear'}**")
    others = [c for c in df.columns if c != outcome]
    preds = c2.multiselect("Candidate predictors", others, default=others)
    if len(preds) < 2:
        st.stop()
    X, y, dropped, _ = design_matrix(df, outcome, preds)
    if family == "binomial":
        y, _ = to01(y)
    names = list(X.columns)
    Xs = StandardScaler().fit_transform(X)
    yv = y.to_numpy(float)
    c2.caption(f"n = {len(y):,} ({dropped} dropped for missingness) · p = {len(names)} · "
               "predictors standardized")
    active = truth().get("active")
    active = active if active and set(active) <= set(names) else None

    t1, t2 = st.tabs(["Penalized regression (Lasso, SCAD, MCP, TLP)", "Knockoff filter"])

    # ---------------------------------------------------------------- penalized
    with t1:
        methods = st.multiselect("Penalties", sel.PENALTIES, default=["Lasso", "SCAD", "TLP"])
        c = st.columns(6)
        folds = c[0].slider("CV folds", 3, 10, 5)
        rule_lab = c[1].selectbox("Choose λ by", ["EBIC", "BIC", "CV (1-SE rule)", "CV (minimum error)"],
                                  help="CV minimizes prediction error and tends to over-select. (E)BIC "
                                       "targets the true support; EBIC adds a penalty for large p.")
        rule = {"EBIC": "ebic", "BIC": "bic", "CV (1-SE rule)": "1se", "CV (minimum error)": "min"}[rule_lab]
        a = c[2].number_input("SCAD a", 2.1, 10.0, 3.7, 0.1)
        gamma = c[3].number_input("MCP γ", 1.1, 10.0, 3.0, 0.1)
        tau = c[4].number_input("TLP τ", 0.01, 5.0, 0.1, 0.05,
                                help="Coefficients (standardized scale) larger than τ are not penalized.")
        seed = c[5].number_input("Seed", 0, 10**6, 0, key="pen_seed")
        with st.expander("How these penalties work"):
            st.markdown(
                "- **Lasso** penalizes every coefficient by λ|β|, which shrinks large effects and tends "
                "to admit extra noise variables when predictors are correlated.\n"
                "- **SCAD** (Fan & Li, 2001) applies the lasso rate near zero, then tapers the penalty "
                "off and leaves coefficients above aλ unpenalized — nearly unbiased for large effects.\n"
                "- **MCP** (Zhang, 2010) relaxes the penalty linearly to zero at γλ.\n"
                "- **TLP** (Shen, Pan & Zhu, 2012) uses λ·min(|β|, τ): an L1 penalty truncated at τ, "
                "a computational surrogate for the L0 penalty.\n\n"
                "Nonconvex penalties are fitted with the local linear approximation (a sequence of "
                "weighted lasso fits started from the lasso), which for TLP is the difference-of-convex "
                "algorithm. λ is chosen by EBIC/BIC on the full-data path or by K-fold CV (MSE for linear, "
                "deviance for logistic). Coefficients are on the standardized-predictor scale.")
        if st.button("Fit penalized models", type="primary") and methods:
            res = {}
            prog = st.progress(0.0, "Cross-validating…")
            for i, m in enumerate(methods):
                res[m] = sel.cv_penalized(Xs, yv, m, family, folds, seed, rule=rule, a=a, gamma=gamma, tau=tau)
                prog.progress((i + 1) / len(methods), f"Finished {m}")
            prog.empty()
            st.session_state.res_pen = dict(res=res, names=names, outcome=outcome)
            summ, coefs = _pen_summary(res, names, active)
            log_result("Variable selection: penalized regression",
                       dict(outcome=outcome, family=family, penalties=methods, folds=folds, rule=rule,
                            a=a, gamma=gamma, tau=tau, seed=seed),
                       tables={"Summary": summ, "Coefficients (standardized)": coefs})

        R = st.session_state.get("res_pen")
        if R and R["names"] == names and R["outcome"] == outcome:
            summ, coefs = _pen_summary(R["res"], names, active)
            st.dataframe(fmt_table(summ), width="stretch")
            if active:
                st.caption("FDP and power are computed against the simulated true active set.")
            st.subheader("Selected coefficients (standardized scale)")
            st.dataframe(coefs.style.format(precision=3, na_rep="·")
                         .background_gradient(cmap="RdBu_r", vmin=-1, vmax=1), width="stretch")
            m = st.selectbox("Show CV curve and path for", list(R["res"]))
            r = R["res"][m]
            c1, c2 = st.columns(2)
            if r["cv_mean"] is not None:
                cv = go.Figure(go.Scatter(x=np.log10(r["lams"]), y=r["cv_mean"], mode="lines+markers",
                                          error_y=dict(array=r["cv_se"]), line_color=ACCENT))
                cv.add_vline(x=np.log10(r["lam_min"]), line_dash="dash", annotation_text="min")
                cv.add_vline(x=np.log10(r["lam_1se"]), line_dash="dot", annotation_text="1se")
                cv.update_layout(title=f"{m}: cross-validation", xaxis_title="log10(λ)",
                                 yaxis_title="CV MSE" if family == "gaussian" else "CV deviance")
            else:
                cv = go.Figure(go.Scatter(x=np.log10(r["lams"]), y=r["ic"], mode="lines+markers",
                                          line_color=ACCENT))
                cv.add_vline(x=np.log10(r["lam"]), line_dash="dash", annotation_text="selected")
                cv.update_layout(title=f"{m}: {r['rule'].upper()} along the path", xaxis_title="log10(λ)",
                                 yaxis_title=r["rule"].upper())
            c1.plotly_chart(cv, width="stretch")
            path = go.Figure()
            chosen = set(np.flatnonzero(r["coef"]))
            for j, nm in enumerate(names):
                path.add_trace(go.Scatter(x=np.log10(r["lams"]), y=r["path"][:, j], mode="lines", name=nm,
                                          line=dict(width=2 if j in chosen else 0.7),
                                          opacity=1 if j in chosen else 0.35, showlegend=j in chosen))
            path.add_vline(x=np.log10(r["lam"]), line_dash="dash")
            path.update_layout(title=f"{m}: coefficient path", xaxis_title="log10(λ)",
                               yaxis_title="Coefficient")
            c2.plotly_chart(path, width="stretch")

    # ---------------------------------------------------------------- knockoffs
    with t2:
        st.write("The knockoff filter builds a synthetic 'fake' copy of each predictor that mimics the "
                 "correlation structure but is known to be null, then keeps predictors that beat "
                 "their knockoff by enough to control the **false discovery rate** at level q.")
        c = st.columns(5)
        q = c[0].slider("Target FDR q", 0.01, 0.5, 0.1, 0.01)
        plus = c[1].checkbox("Knockoff+ (exact FDR control)", True)
        stat = c[2].selectbox("Statistic W", ["Lasso coefficient difference", "Random-forest importance difference"])
        M = c[3].slider("Knockoff draws", 1, 30, 1, help="Repeat with fresh knockoffs to see selection stability.")
        kseed = c[4].number_input("Seed", 0, 10**6, 0, key="ko_seed")
        if st.button("Run knockoff filter", type="primary"):
            runs = []
            prog = st.progress(0.0, "Generating knockoffs…")
            for m in range(M):
                runs.append(sel.run_knockoffs(Xs, yv, family, q, plus,
                                              "lasso" if stat.startswith("Lasso") else "rf", kseed + m))
                prog.progress((m + 1) / M)
            prog.empty()
            st.session_state.res_ko = dict(runs=runs, names=names, outcome=outcome, q=q)
            first = [names[j] for j in runs[0]["selected"]]
            log_result("Variable selection: knockoff filter",
                       dict(outcome=outcome, q=q, knockoff_plus=plus, statistic=stat, draws=M, seed=kseed),
                       tables={"Selected (draw 1)": pd.DataFrame({"variable": first})})

        R = st.session_state.get("res_ko")
        if R and R["names"] == names and R["outcome"] == outcome:
            r0 = R["runs"][0]
            chosen = [names[j] for j in r0["selected"]]
            c1, c2, c3 = st.columns(3)
            c1.metric("Selected (draw 1)", len(chosen))
            c2.metric("Threshold T", "∞ (no selections)" if np.isinf(r0["T"]) else f"{r0['T']:.4f}")
            c3.metric("Equicorrelated s", f"{r0['s'][0]:.3f}",
                      help="Larger s = knockoffs less correlated with originals = more power.")
            if active:
                fdp, pw = sel.fdp_power(chosen, active)
                st.write(f"Against the true active set: FDP = **{fdp:.2f}**, power = **{pw:.2f}**")
            W = pd.Series(r0["W"], index=names).sort_values(ascending=False)
            fig = go.Figure(go.Bar(x=W.index, y=W.values,
                                   marker_color=[ACCENT if n in chosen else "#B8C2CF" for n in W.index]))
            if np.isfinite(r0["T"]):
                fig.add_hline(y=r0["T"], line_dash="dash", annotation_text="T")
                fig.add_hline(y=-r0["T"], line_dash="dot")
            fig.update_layout(title="Knockoff statistics W (dark = selected)", yaxis_title="W",
                              xaxis=dict(showticklabels=len(names) <= 60))
            st.plotly_chart(fig, width="stretch")
            st.write("**Selected:** " + (", ".join(chosen) if chosen else "none"))
            if len(R["runs"]) > 1:
                freq = pd.Series(0.0, index=names)
                for r in R["runs"]:
                    freq.iloc[r["selected"]] += 1 / len(R["runs"])
                freq = freq[freq > 0].sort_values(ascending=False)
                eta = st.slider("Keep variables selected in at least this fraction of draws", 0.1, 1.0, 0.5, 0.05)
                st.plotly_chart(px.bar(freq, labels={"index": "", "value": "selection frequency"}),
                                width="stretch")
                stable = freq[freq >= eta].index.tolist()
                st.write(f"**Stable set ({len(stable)}):** " + (", ".join(stable) or "none"))
                st.caption("Aggregating across draws by frequency is a stability heuristic; FDR "
                           "guarantees apply to each single draw (see Ren, Wei & Candès 2023 for "
                           "derandomized knockoffs with guarantees).")
            if np.isinf(r0["T"]):
                st.info(f"No selections. Knockoff+ needs at least {int(np.ceil(1 / R['q']))} discoveries "
                        "to certify FDR ≤ q; with few true signals, try a larger q or plain knockoffs.")
            st.caption("Uses Gaussian model-X knockoffs with a Ledoit–Wolf covariance estimate. The "
                       "guarantee assumes the predictor distribution is well approximated as Gaussian; "
                       "dummy-coded categorical predictors are handled only approximately.")


def _pen_summary(res, names, active):
    rows, coefs = [], {}
    for m, r in res.items():
        chosen = [names[j] for j in np.flatnonzero(r["coef"])]
        i = int(np.argmin(np.abs(r["lams"] - r["lam"])))
        crit = r["cv_mean"][i] if r["cv_mean"] is not None else r["ic"][i]
        row = dict(method=m, selected=len(chosen), **{"λ": r["lam"], r["rule"].upper() if r["cv_mean"] is None
                                                     else "CV error": crit})
        if active:
            row["FDP"], row["power"] = sel.fdp_power(chosen, active)
        rows.append(row)
        coefs[m] = pd.Series(r["coef"], index=names)
    coefs = pd.DataFrame(coefs)
    coefs = coefs[(coefs != 0).any(axis=1)].replace(0, np.nan)
    return pd.DataFrame(rows).set_index("method"), coefs


# ============================================================================= Causal inference
def page_causal():
    st.header("Causal inference: binary treatment")
    need_data()
    df = get_df()
    bins = binary_cols(df)
    if not bins:
        st.warning("No binary column available to use as the treatment.")
        st.stop()
    tr = truth()
    c1, c2, c3 = st.columns([1, 1, 2])
    treat = c1.selectbox("Treatment (binary)", bins,
                         index=bins.index(tr["treatment"]) if tr.get("treatment") in bins else 0)
    outs = [c for c in numeric_cols(df) if c != treat]
    outcome = c2.selectbox("Outcome", outs,
                           index=outs.index(tr["outcome"]) if tr.get("outcome") in outs else 0)
    others = [c for c in df.columns if c not in (treat, outcome)]
    covs = c3.multiselect("Confounders (adjust for)", others, default=others)
    if not covs:
        st.stop()

    c = st.columns(5)
    learner = c[0].selectbox("Nuisance learner", cz.LEARNERS,
                             help="Model for the propensity score e(x) and the outcome regressions μ₀(x), μ₁(x).")
    stage2 = c[1].selectbox("CATE second-stage learner", cz.LEARNERS, index=min(1, len(cz.LEARNERS) - 1),
                            help="Regularized model used by the X- and DR-learners on pseudo-outcomes.")
    folds = c[2].slider("Cross-fitting folds", 2, 10, 5)
    trim = c[3].slider("Clip propensity to [ε, 1−ε]", 0.0, 0.1, 0.01, 0.005)
    seed = c[4].number_input("Seed", 0, 10**6, 0, key="cz_seed")
    c = st.columns([2, 1, 1])
    which = c[0].multiselect("CATE meta-learners", ["T-learner", "S-learner", "X-learner", "DR-learner"],
                             default=["T-learner", "X-learner", "DR-learner"])
    boot = c[1].checkbox("Bootstrap SE for g-computation", value=learner.startswith("Parametric"))
    B = c[2].number_input("Bootstrap B", 50, 2000, 200, 50)

    with st.expander("Assumptions and estimators"):
        st.markdown(
            "Causal interpretation requires **consistency**, **no unmeasured confounding** given the "
            "selected covariates, and **overlap** (0 < e(x) < 1).\n\n"
            "- *Naive*: unadjusted difference in means.\n"
            "- *Outcome regression (g-computation)*: average of μ̂₁(x) − μ̂₀(x).\n"
            "- *IPW (Hajek)*: normalized inverse-propensity weighting; SE from the influence function "
            "treating ê as known (conservative for the ATE).\n"
            "- *AIPW*: doubly robust — consistent if either the propensity or the outcome model is right; "
            "with cross-fitting it attains √n inference even with flexible ML nuisances.\n\n"
            "**CATE** τ(x) = E[Y(1) − Y(0) | X = x] via meta-learners: S (one model with treatment as a "
            "feature), T (separate arm models), X (imputed effects, propensity-weighted), and DR "
            "(regress the AIPW pseudo-outcome on X). All CATE predictions are out-of-fold.")

    if st.button("Estimate effects", type="primary"):
        X, y, dropped, d = design_matrix(df, outcome, covs, extra=[treat])
        t, lvl = to01(d[treat])
        with st.spinner("Fitting cross-fitted nuisance models…"):
            nuis = cz.crossfit_nuisance(X, t, y, learner, folds, seed)
            gse = cz.bootstrap_gcomp(X, t, y, learner, int(B), seed) if boot else None
            tab, psi = cz.ate_table(y, t, nuis["e"], nuis["mu0"], nuis["mu1"], trim, gse)
        with st.spinner("Fitting CATE meta-learners…"):
            cates = cz.cate_learners(X, t, y, nuis, learner, which, folds, seed, trim, stage2) if which else {}
        st.session_state.res_causal = dict(X=X, t=t.to_numpy(), y=y.to_numpy(float), nuis=nuis, tab=tab,
                                           psi=psi, cates=cates, index=d.index, treat=treat, lvl=lvl,
                                           outcome=outcome, trim=trim, dropped=dropped, learner=learner)
        log_result("Causal inference: ATE / CATE",
                   dict(treatment=treat, outcome=outcome, covariates=covs, learner=learner, stage2=stage2,
                        folds=folds, trim=trim, cate_learners=which, bootstrap_B=B if boot else 0, seed=seed),
                   tables={"ATE": tab,
                           "CATE summary": pd.DataFrame({k: dict(mean=v.mean(), sd=v.std())
                                                         for k, v in cates.items()}).T})

    R = st.session_state.get("res_causal")
    if not R or R["treat"] != treat or R["outcome"] != outcome:
        return
    true_ate, true_cate = None, None
    if tr.get("treatment") == treat and tr.get("outcome") == outcome:
        true_ate = tr.get("ate")
        if tr.get("cate") is not None:
            true_cate = tr["cate"].reindex(R["index"]).to_numpy()

    t, e = R["t"], np.clip(R["nuis"]["e"], R["trim"], 1 - R["trim"])
    tA, tB, tC = st.tabs(["Average treatment effect", "Overlap & balance", "Heterogeneous effects (CATE)"])

    with tA:
        st.caption(f"Effect of {treat} = {R['lvl']} vs. not, on {outcome}. n = {len(t):,} "
                   f"({R['dropped']} dropped for missingness); {int(t.sum())} treated.")
        st.dataframe(fmt_table(R["tab"], 4), width="stretch")
        tb = R["tab"].dropna(subset=["SE"])
        fig = go.Figure(go.Scatter(x=tb.estimate, y=tb.index, mode="markers", marker=dict(size=11, color=ACCENT),
                                   error_x=dict(type="data", symmetric=False, array=tb.upper - tb.estimate,
                                                arrayminus=tb.estimate - tb.lower)))
        fig.add_vline(x=0, line_color="grey")
        if true_ate is not None:
            fig.add_vline(x=true_ate, line_dash="dash", line_color="#C0504D",
                          annotation_text=f"true ATE {true_ate:.2f}")
        fig.update_layout(xaxis_title="ATE (95% CI)", height=300, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, width="stretch")

    with tB:
        c1, c2 = st.columns(2)
        ps = pd.DataFrame({"propensity": R["nuis"]["e"],
                           "group": np.where(t == 1, "treated", "control")})
        c1.plotly_chart(px.histogram(ps, x="propensity", color="group", barmode="overlay", nbins=40,
                                     opacity=.6, color_discrete_sequence=PALETTE,
                                     title="Propensity score overlap"), width="stretch")
        c1.caption(f"Range among treated: {ps.propensity[t == 1].min():.3f}–{ps.propensity[t == 1].max():.3f}; "
                   f"controls: {ps.propensity[t == 0].min():.3f}–{ps.propensity[t == 0].max():.3f}. "
                   f"{int(((R['nuis']['e'] < R['trim']) | (R['nuis']['e'] > 1 - R['trim'])).sum())} "
                   "units clipped.")
        w = t / e + (1 - t) / (1 - e)
        bal = pd.DataFrame({"unweighted": cz.smd(R["X"], t), "IPW-weighted": cz.smd(R["X"], t, w)})
        bl = bal.reset_index(names="covariate").melt("covariate", var_name="sample", value_name="SMD")
        fig = px.scatter(bl, x="SMD", y="covariate", color="sample", color_discrete_sequence=PALETTE,
                         title="Covariate balance (love plot)")
        for v in (-0.1, 0.1):
            fig.add_vline(x=v, line_dash="dot", line_color="grey")
        fig.add_vline(x=0, line_color="grey")
        c2.plotly_chart(fig, width="stretch")
        ess = [w[t == g].sum() ** 2 / (w[t == g] ** 2).sum() for g in (1, 0)]
        c2.caption(f"|SMD| < 0.1 is a common balance target. Effective sample size after weighting: "
                   f"treated {ess[0]:.0f}, control {ess[1]:.0f}.")

    with tC:
        if not R["cates"]:
            st.info("Select at least one meta-learner and re-run.")
            return
        rows = {}
        for k, v in R["cates"].items():
            row = dict(mean=v.mean(), sd=v.std(), q10=np.quantile(v, .1), q90=np.quantile(v, .9))
            if true_cate is not None:
                row["PEHE (√MSE vs truth)"] = np.sqrt(np.mean((v - true_cate) ** 2))
                row["corr with truth"] = np.corrcoef(v, true_cate)[0, 1]
            rows[k] = row
        st.dataframe(fmt_table(pd.DataFrame(rows).T), width="stretch")

        m = st.selectbox("Inspect learner", list(R["cates"]))
        tau = R["cates"][m]
        c1, c2 = st.columns(2)
        c1.plotly_chart(px.histogram(x=tau, nbins=40, labels={"x": "Estimated CATE"},
                                     title="Distribution of estimated effects",
                                     color_discrete_sequence=[ACCENT]), width="stretch")
        xv = c2.selectbox("CATE against covariate", R["X"].columns)
        sc = go.Figure(go.Scatter(x=R["X"][xv], y=tau, mode="markers", opacity=.4, name="estimated",
                                  marker=dict(color=ACCENT, size=5)))
        if true_cate is not None:
            sc.add_trace(go.Scatter(x=R["X"][xv], y=true_cate, mode="markers", opacity=.4, name="true",
                                    marker=dict(color="#C0504D", size=4)))
        sc.update_layout(xaxis_title=xv, yaxis_title="CATE")
        c2.plotly_chart(sc, width="stretch")

        st.subheader("Calibration: sorted group effects (GATES)")
        G = cz.gates(tau, R["psi"])
        fig = go.Figure(go.Scatter(x=G.index, y=G.aipw, mode="markers", name="AIPW group effect",
                                   marker=dict(size=11, color=ACCENT),
                                   error_y=dict(type="data", symmetric=False, array=G.upper - G.aipw,
                                                arrayminus=G.aipw - G.lower)))
        fig.add_trace(go.Scatter(x=G.index, y=G.predicted, mode="lines+markers", name="mean predicted CATE",
                                 line=dict(dash="dash", color="#C0504D")))
        fig.update_layout(xaxis_title="CATE quintile (1 = smallest predicted effect)", yaxis_title="Effect")
        st.plotly_chart(fig, width="stretch")
        st.caption("If the learner captures real heterogeneity, the doubly robust effect estimated inside "
                   "each predicted-effect quintile should increase across groups and track the predictions.")

        st.subheader("What drives the heterogeneity?")
        depth = st.slider("Summary tree depth", 1, 4, 2)
        tree = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=max(20, len(tau) // 50)).fit(R["X"], tau)
        fig, ax = plt.subplots(figsize=(3.2 * 2 ** depth / 1.6 + 4, 2.2 * depth + 1.5))
        plot_tree(tree, feature_names=list(R["X"].columns), filled=True, rounded=True, precision=2,
                  fontsize=8, ax=ax, impurity=False)
        st.pyplot(fig)
        plt.close(fig)
        st.caption("A shallow regression tree fitted to the estimated CATEs: an interpretable summary of the "
                   "learner, not an independent estimate.")


# ============================================================================= Survival
def page_survival():
    st.header("Survival models")
    need_data()
    df = get_df()
    num = numeric_cols(df)
    if len(num) < 2:
        st.warning("Need a numeric time column and a 0/1 event column.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    tcol = c1.selectbox("Time", num, index=num.index("time") if "time" in num else 0)
    ecol = c2.selectbox("Event (1 = event)", num, index=num.index("event") if "event" in num else 1)
    group = c3.selectbox("Group for KM curves", ["—"] + [c for c in df.columns if c not in (tcol, ecol)])

    t1, t2 = st.tabs(["Kaplan–Meier", "Cox proportional hazards"])
    with t1:
        d = df[[tcol, ecol] + ([group] if group != "—" else [])].dropna()
        if group != "—" and d[group].nunique() > 8:
            d[group] = pd.qcut(d[group], 3, labels=["low", "mid", "high"])
            st.caption(f"{group} has many levels, so it was split into tertiles.")
        fig = go.Figure()
        groups = [("All", d)] if group == "—" else list(d.groupby(group, observed=True))
        med = []
        for i, (g, sub) in enumerate(groups):
            kmf = KaplanMeierFitter().fit(sub[tcol], sub[ecol], label=str(g))
            sf, ci = kmf.survival_function_, kmf.confidence_interval_
            x = sf.index.to_numpy()
            col = PALETTE[i % len(PALETTE)]
            fig.add_trace(go.Scatter(x=x, y=sf.iloc[:, 0], line_shape="hv", name=str(g), line_color=col))
            fig.add_trace(go.Scatter(x=np.r_[x, x[::-1]], y=np.r_[ci.iloc[:, 1], ci.iloc[::-1, 0]],
                                     fill="toself", line=dict(width=0), fillcolor=col, opacity=.15,
                                     line_shape="hv", showlegend=False, hoverinfo="skip"))
            med.append(dict(group=g, n=len(sub), events=int(sub[ecol].sum()), median=kmf.median_survival_time_))
        fig.update_layout(xaxis_title=tcol, yaxis_title="Survival probability", yaxis_range=[0, 1.02])
        st.plotly_chart(fig, width="stretch")
        st.dataframe(pd.DataFrame(med).set_index("group"), width="stretch")
        if group != "—":
            lr = multivariate_logrank_test(d[tcol], d[group], d[ecol])
            st.write(f"Log-rank test: χ² = {lr.test_statistic:.2f}, p = {lr.p_value:.4g}")

    with t2:
        others = [c for c in df.columns if c not in (tcol, ecol)]
        covs = st.multiselect("Covariates", others, default=others[:5])
        if not covs:
            st.stop()
        d = df[[tcol, ecol] + covs].dropna()
        d = pd.get_dummies(d, columns=[c for c in covs if not pd.api.types.is_numeric_dtype(d[c])],
                           drop_first=True, dtype=float)
        pen = st.slider("Ridge penalty (0 = standard Cox)", 0.0, 1.0, 0.0, .05)
        try:
            cph = CoxPHFitter(penalizer=pen).fit(d, tcol, ecol)
        except Exception as e:
            st.error(f"Cox model failed: {e}")
            st.stop()
        s = cph.summary
        tab = pd.DataFrame({"HR": s["exp(coef)"], "lower": s["exp(coef) lower 95%"],
                            "upper": s["exp(coef) upper 95%"], "p": s["p"]})
        m1, m2, m3 = st.columns(3)
        m1.metric("n / events", f"{len(d)} / {int(d[ecol].sum())}")
        m2.metric("Concordance", f"{cph.concordance_index_:.3f}")
        m3.metric("Partial AIC", f"{cph.AIC_partial_:.1f}")
        c1, c2 = st.columns(2)
        c1.dataframe(fmt_table(tab), width="stretch")
        fp = forest_plot(tab, "HR", "lower", "upper", 1, "Hazard ratio (95% CI)")
        c2.plotly_chart(fp, width="stretch")
        with st.expander("Proportional hazards check (Schoenfeld residual test)"):
            ph = proportional_hazard_test(cph, d, time_transform="rank").summary
            st.dataframe(fmt_table(ph, 4), width="stretch")
            st.caption("Small p-values suggest that covariate's hazard ratio changes over time.")
        if st.button("Add to report", key="surv_log"):
            log_result("Survival: Cox PH", dict(time=tcol, event=ecol, covariates=covs, penalizer=pen),
                       tables={"Hazard ratios": tab}, figs=[fp])


# ============================================================================= Count models
def page_counts():
    st.header("Count models: Poisson, negative binomial, zero-inflated Poisson")
    need_data()
    df = get_df()
    counts = [c for c in numeric_cols(df)
              if (df[c].dropna() >= 0).all() and np.allclose(df[c].dropna() % 1, 0) and df[c].nunique() > 2]
    if not counts:
        st.warning("No non-negative integer column found to use as a count outcome.")
        st.stop()
    c1, c2 = st.columns([1, 2])
    outcome = c1.selectbox("Count outcome", counts)
    others = [c for c in df.columns if c != outcome]
    pos = [c for c in numeric_cols(df) if c != outcome and (df[c].dropna() > 0).all()]
    offset = c1.selectbox("Exposure / follow-up time (log offset)", ["—"] + pos,
                          index=(["—"] + pos).index("followup") if "followup" in pos else 0)
    preds = c2.multiselect("Count-model predictors", [c for c in others if c != offset],
                           default=[c for c in others if c != offset][:5])
    infl = c2.multiselect("Zero-inflation predictors (ZIP)", [c for c in others if c != offset],
                          default=[c for c in ["smoker", "age"] if c in others] or preds[:2])
    models = st.multiselect("Models", ["Poisson", "Negative binomial (NB2)", "Zero-inflated Poisson (ZIP)"],
                            default=["Poisson", "Negative binomial (NB2)", "Zero-inflated Poisson (ZIP)"])
    with st.expander("Model definitions"):
        st.markdown(
            "- **Poisson**: log E[Y|x] = log(offset) + xᵀβ, with Var(Y|x) = E[Y|x].\n"
            "- **Negative binomial (NB2)**: same mean, Var = μ + αμ² — handles overdispersion.\n"
            "- **Zero-inflated Poisson**: a mixture — with probability π(z) = logit⁻¹(zᵀγ) the unit is a "
            "structural zero; otherwise Y ~ Poisson(μ(x)). exp(β) are rate ratios among the 'at-risk' "
            "population; exp(γ) are odds ratios for being a structural zero.")
    if not preds or not models or not st.button("Fit count models", type="primary"):
        if "res_counts" not in st.session_state or st.session_state.res_counts["outcome"] != outcome:
            st.stop()
    else:
        extra = infl + ([offset] if offset != "—" else [])
        X, y, dropped, d = design_matrix(df, outcome, preds, extra=extra)
        y = y.astype(int)
        Xc = sm.add_constant(X, has_constant="add")
        Z = (sm.add_constant(pd.get_dummies(d[infl], drop_first=True, dtype=float), has_constant="add")
             if infl else pd.DataFrame({"const": np.ones(len(d))}, index=d.index))
        off = np.log(d[offset].to_numpy(float)) if offset != "—" else np.zeros(len(d))
        fits = {}
        for m in models:
            try:
                if m == "Poisson":
                    fits[m] = sm.Poisson(y, Xc, offset=off).fit(disp=0, maxiter=300)
                elif m.startswith("Negative"):
                    fits[m] = sm.NegativeBinomial(y, Xc, loglike_method="nb2", offset=off).fit(disp=0, maxiter=500)
                else:
                    fits[m] = sm.ZeroInflatedPoisson(y, Xc, exog_infl=Z, inflation="logit",
                                                     offset=off).fit(disp=0, maxiter=2000, method="bfgs")
            except Exception as ex:
                st.error(f"{m} failed: {ex}")
        st.session_state.res_counts = dict(fits=fits, y=y.to_numpy(), Xc=Xc, Z=Z, off=off, outcome=outcome,
                                           dropped=dropped)
        comp, _ = _count_compare(st.session_state.res_counts)
        log_result("Count models", dict(outcome=outcome, predictors=preds, inflation=infl, offset=offset,
                                        models=models),
                   tables={"Model comparison": comp,
                           **{f"{m} estimates": _count_estimates(f, m)[0] for m, f in fits.items()}})

    R = st.session_state.res_counts
    if not R["fits"]:
        st.stop()
    comp, probs = _count_compare(R)
    st.subheader("Model comparison")
    st.dataframe(fmt_table(comp), width="stretch")
    if "Poisson" in R["fits"]:
        p = R["fits"]["Poisson"]
        mu = np.exp(R["Xc"].to_numpy() @ p.params[list(R["Xc"].columns)].to_numpy() + R["off"])
        disp = np.sum((R["y"] - mu) ** 2 / mu) / p.df_resid
        st.write(f"Poisson Pearson dispersion χ²/df = **{disp:.2f}** (≈ 1 if equidispersed).")
    if "Poisson" in R["fits"] and "Zero-inflated Poisson (ZIP)" in R["fits"]:
        v = _vuong(R["fits"]["Zero-inflated Poisson (ZIP)"], R["fits"]["Poisson"])
        st.write(f"Vuong test, ZIP vs Poisson: z = {v['z']:.2f} (p = {v['p']:.3g}); "
                 f"AIC-corrected z = {v['z_aic']:.2f} (p = {v['p_aic']:.3g}). Positive z favours ZIP.")
        st.caption("The Vuong test for zero-inflation is non-nested only in a limited sense (Wilson, 2015); "
                   "treat it alongside AIC/BIC and the fitted-frequency plot below.")

    kmax = min(int(R["y"].max()), 25)
    obs = np.bincount(R["y"], minlength=kmax + 1)[:kmax + 1]
    fig = go.Figure(go.Bar(x=np.arange(kmax + 1), y=obs, name="observed", marker_color="#B8C2CF"))
    for i, (m, P) in enumerate(probs.items()):
        fig.add_trace(go.Scatter(x=np.arange(kmax + 1), y=P[:, :kmax + 1].sum(0), mode="lines+markers",
                                 name=m, line_color=PALETTE[i]))
    fig.update_layout(title="Observed vs expected count frequencies", xaxis_title=R["outcome"],
                      yaxis_title="Number of units")
    st.plotly_chart(fig, width="stretch")

    st.subheader("Estimates")
    m = st.selectbox("Model", list(R["fits"]))
    tabs = _count_estimates(R["fits"][m], m)
    c1, c2 = st.columns(2)
    c1.markdown("**Count part: rate ratios**")
    c1.dataframe(fmt_table(tabs[0]), width="stretch")
    c1.plotly_chart(forest_plot(tabs[0].drop("const", errors="ignore"), "estimate", "lower", "upper", 1,
                                "Rate ratio (95% CI)"), width="stretch")
    if tabs[1] is not None:
        c2.markdown("**Zero-inflation part: odds of a structural zero**")
        c2.dataframe(fmt_table(tabs[1]), width="stretch")
        c2.plotly_chart(forest_plot(tabs[1].drop("const", errors="ignore"), "estimate", "lower", "upper", 1,
                                    "Odds ratio (95% CI)"), width="stretch")
    if m.startswith("Negative"):
        f = R["fits"][m]
        c2.metric("Overdispersion α", f"{f.params['alpha']:.3f}",
                  f"95% CI {f.conf_int().loc['alpha', 0]:.3f}–{f.conf_int().loc['alpha', 1]:.3f}",
                  delta_color="off")


def _count_estimates(fit, name):
    ci = fit.conf_int()
    t = pd.DataFrame({"estimate": fit.params, "lower": ci[0], "upper": ci[1], "p": fit.pvalues})
    t = t.drop(index="alpha", errors="ignore")
    infl = t.index.str.startswith("inflate_")
    E = t.copy()
    E[["estimate", "lower", "upper"]] = np.exp(E[["estimate", "lower", "upper"]])
    count = E[~infl]
    zero = E[infl].rename(index=lambda s: s.replace("inflate_", "")) if infl.any() else None
    return count, zero


def _count_probs(fit, name, R, kmax):
    k = np.arange(kmax + 1)
    p = fit.params
    beta = p[[c for c in R["Xc"].columns]].to_numpy()
    mu = np.exp(R["Xc"].to_numpy() @ beta + R["off"])
    if name == "Poisson":
        return stats.poisson.pmf(k[None, :], mu[:, None])
    if name.startswith("Negative"):
        a = p["alpha"]
        return stats.nbinom.pmf(k[None, :], 1 / a, (1 / (1 + a * mu))[:, None])
    gam = p[["inflate_" + c for c in R["Z"].columns]].to_numpy()
    pi = 1 / (1 + np.exp(-(R["Z"].to_numpy() @ gam)))
    P = (1 - pi)[:, None] * stats.poisson.pmf(k[None, :], mu[:, None])
    P[:, 0] += pi
    return P


def _count_compare(R):
    kmax = int(R["y"].max()) + 5
    rows, probs = {}, {}
    for m, f in R["fits"].items():
        P = _count_probs(f, m, R, kmax)
        probs[m] = P
        rows[m] = {"log-likelihood": f.llf, "parameters": len(f.params), "AIC": f.aic, "BIC": f.bic,
                   "observed zeros": int((R["y"] == 0).sum()), "expected zeros": P[:, 0].sum()}
    return pd.DataFrame(rows).T, probs


def _vuong(f1, f2):
    m = f1.model.loglikeobs(f1.params) - f2.model.loglikeobs(f2.params)
    n = len(m)
    z = np.sqrt(n) * m.mean() / m.std(ddof=1)
    adj = (len(f1.params) - len(f2.params)) / n
    z_aic = np.sqrt(n) * (m.mean() - adj) / m.std(ddof=1)
    return dict(z=z, p=stats.norm.sf(z), z_aic=z_aic, p_aic=stats.norm.sf(z_aic))
