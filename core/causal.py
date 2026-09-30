"""
Binary-treatment causal inference under no unmeasured confounding + overlap.

ATE estimators
    Naive difference in means, outcome regression (g-computation), IPW (Hajek),
    AIPW / doubly robust with K-fold cross-fitting (Chernozhukov et al. 2018).
CATE meta-learners (Kunzel et al. 2019; Kennedy 2023)
    S-, T-, X- and DR-learners, all producing out-of-fold predictions.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from core.utils import HAS_XGB

LEARNERS = ["Parametric (logistic / linear)", "Random forest", "Gradient boosting"] + (
    ["XGBoost"] if HAS_XGB else [])


def make_learner(name, kind, seed=0):
    """kind = 'clf' (propensity), 'reg' (outcome) or 'cate' (heavily regularized second
    stage fitted to noisy pseudo-outcomes)."""
    cate = kind == "cate"
    if name.startswith("Parametric"):
        return (make_pipeline(StandardScaler(), LogisticRegression(C=1e4, max_iter=5000))
                if kind == "clf" else LinearRegression())
    if name == "Random forest":
        kw = dict(n_estimators=200, min_samples_leaf=40 if cate else 10, random_state=seed, n_jobs=-1)
        return RandomForestClassifier(**kw) if kind == "clf" else RandomForestRegressor(**kw)
    if name == "Gradient boosting":
        kw = dict(n_estimators=100 if cate else 200, max_depth=2 if cate else 3, learning_rate=0.05,
                  subsample=0.5 if cate else 1.0, min_samples_leaf=40 if cate else 1, random_state=seed)
        return GradientBoostingClassifier(**kw) if kind == "clf" else GradientBoostingRegressor(**kw)
    if name == "XGBoost":
        from xgboost import XGBClassifier, XGBRegressor
        kw = dict(n_estimators=150 if cate else 300, max_depth=2 if cate else 3, learning_rate=0.05,
                  subsample=0.8, min_child_weight=40 if cate else 1, reg_lambda=10 if cate else 1,
                  random_state=seed, n_jobs=-1, verbosity=0)
        return XGBClassifier(**kw) if kind == "clf" else XGBRegressor(**kw)
    raise ValueError(name)


def crossfit_nuisance(X, t, y, learner, folds=5, seed=0):
    """Out-of-fold propensity e(x) and arm-specific outcome regressions mu0(x), mu1(x)."""
    X, t, y = np.asarray(X, float), np.asarray(t, int), np.asarray(y, float)
    n = len(y)
    e, mu0, mu1 = np.zeros(n), np.zeros(n), np.zeros(n)
    fold_id = np.zeros(n, int)
    for k, (tr, te) in enumerate(StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, t)):
        fold_id[te] = k
        e[te] = clone(make_learner(learner, "clf", seed)).fit(X[tr], t[tr]).predict_proba(X[te])[:, 1]
        tr1, tr0 = tr[t[tr] == 1], tr[t[tr] == 0]
        mu1[te] = clone(make_learner(learner, "reg", seed)).fit(X[tr1], y[tr1]).predict(X[te])
        mu0[te] = clone(make_learner(learner, "reg", seed)).fit(X[tr0], y[tr0]).predict(X[te])
    return dict(e=e, mu0=mu0, mu1=mu1, fold_id=fold_id)


def aipw_pseudo(y, t, e, mu0, mu1):
    return mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)


def ate_table(y, t, e, mu0, mu1, trim=0.01, gcomp_se=None, level=0.95):
    y, t = np.asarray(y, float), np.asarray(t, int)
    e = np.clip(e, trim, 1 - trim)
    n = len(y)
    z = stats.norm.ppf(0.5 + level / 2)
    rows = []

    y1, y0 = y[t == 1], y[t == 0]
    naive = y1.mean() - y0.mean()
    rows.append(("Naive difference", naive, np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))))

    rows.append(("Outcome regression (g-computation)", np.mean(mu1 - mu0), gcomp_se))

    w1, w0 = t / e, (1 - t) / (1 - e)
    m1, m0 = np.sum(w1 * y) / np.sum(w1), np.sum(w0 * y) / np.sum(w0)
    phi = w1 * (y - m1) / w1.mean() - w0 * (y - m0) / w0.mean()
    rows.append(("IPW (Hajek)", m1 - m0, phi.std(ddof=1) / np.sqrt(n)))

    psi = aipw_pseudo(y, t, e, mu0, mu1)
    rows.append(("AIPW (doubly robust)", psi.mean(), psi.std(ddof=1) / np.sqrt(n)))

    tab = pd.DataFrame(rows, columns=["estimator", "estimate", "SE"]).set_index("estimator")
    tab["lower"] = tab.estimate - z * tab.SE
    tab["upper"] = tab.estimate + z * tab.SE
    tab["p"] = 2 * stats.norm.sf(np.abs(tab.estimate / tab.SE))
    return tab, psi


def bootstrap_gcomp(X, t, y, learner, B=200, seed=0):
    X, t, y = np.asarray(X, float), np.asarray(t, int), np.asarray(y, float)
    rng = np.random.default_rng(seed)
    est = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        Xb, tb, yb = X[i], t[i], y[i]
        if tb.min() == tb.max():
            continue
        m1 = clone(make_learner(learner, "reg", seed)).fit(Xb[tb == 1], yb[tb == 1])
        m0 = clone(make_learner(learner, "reg", seed)).fit(Xb[tb == 0], yb[tb == 0])
        est.append(np.mean(m1.predict(Xb) - m0.predict(Xb)))
    return float(np.std(est, ddof=1))


def smd(X, t, w=None):
    """Standardized mean differences (denominator: unweighted pooled SD)."""
    X = pd.DataFrame(X)
    t = np.asarray(t, int)
    w = np.ones(len(t)) if w is None else np.asarray(w)
    out = {}
    for c in X:
        x = X[c].to_numpy(float)
        m1 = np.average(x[t == 1], weights=w[t == 1])
        m0 = np.average(x[t == 0], weights=w[t == 0])
        sd = np.sqrt((x[t == 1].var(ddof=1) + x[t == 0].var(ddof=1)) / 2)
        out[c] = (m1 - m0) / sd if sd > 0 else 0.0
    return pd.Series(out)


# ----------------------------------------------------------------------------- CATE
def cate_learners(X, t, y, nuis, learner, which, folds=5, seed=0, trim=0.01, stage2=None):
    stage2 = stage2 or learner
    X, t, y = np.asarray(X, float), np.asarray(t, int), np.asarray(y, float)
    e = np.clip(nuis["e"], trim, 1 - trim)
    mu0, mu1, fid = nuis["mu0"], nuis["mu1"], nuis["fold_id"]
    out = {}
    if "T-learner" in which:
        out["T-learner"] = mu1 - mu0
    if "S-learner" in which:
        tau = np.zeros(len(y))
        for k in range(folds):
            tr, te = fid != k, fid == k
            m = clone(make_learner(learner, "reg", seed)).fit(np.column_stack([X[tr], t[tr]]), y[tr])
            tau[te] = (m.predict(np.column_stack([X[te], np.ones(te.sum())]))
                       - m.predict(np.column_stack([X[te], np.zeros(te.sum())])))
        out["S-learner"] = tau
    if "X-learner" in which:
        tau = np.zeros(len(y))
        for k in range(folds):
            tr, te = np.where(fid != k)[0], fid == k
            tr1, tr0 = tr[t[tr] == 1], tr[t[tr] == 0]
            m1 = clone(make_learner(learner, "reg", seed)).fit(X[tr1], y[tr1])
            m0 = clone(make_learner(learner, "reg", seed)).fit(X[tr0], y[tr0])
            d1 = y[tr1] - m0.predict(X[tr1])
            d0 = m1.predict(X[tr0]) - y[tr0]
            g1 = clone(make_learner(stage2, "cate", seed)).fit(X[tr1], d1).predict(X[te])
            g0 = clone(make_learner(stage2, "cate", seed)).fit(X[tr0], d0).predict(X[te])
            tau[te] = e[te] * g0 + (1 - e[te]) * g1
        out["X-learner"] = tau
    if "DR-learner" in which:
        psi = aipw_pseudo(y, t, e, mu0, mu1)
        tau = np.zeros(len(y))
        for k in range(folds):
            tr, te = fid != k, fid == k
            tau[te] = clone(make_learner(stage2, "cate", seed)).fit(X[tr], psi[tr]).predict(X[te])
        out["DR-learner"] = tau
    return out


def gates(tau_hat, psi, groups=5, level=0.95):
    """Sorted group average treatment effects: AIPW effect within CATE quantile groups."""
    z = stats.norm.ppf(0.5 + level / 2)
    g = pd.qcut(pd.Series(tau_hat).rank(method="first"), groups, labels=False) + 1
    df = pd.DataFrame(dict(g=g, tau=tau_hat, psi=psi))
    agg = df.groupby("g").agg(predicted=("tau", "mean"), aipw=("psi", "mean"),
                              sd=("psi", "std"), n=("psi", "size"))
    agg["lower"] = agg.aipw - z * agg.sd / np.sqrt(agg.n)
    agg["upper"] = agg.aipw + z * agg.sd / np.sqrt(agg.n)
    return agg.drop(columns="sd")
