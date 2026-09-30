"""
Variable selection engines.

Penalized regression
--------------------
Lasso, SCAD (Fan & Li 2001), MCP (Zhang 2010) and TLP (Shen, Pan & Zhu 2012) are fitted
with the local linear approximation (LLA; Zou & Li 2008): starting from the lasso, each
step solves a weighted-L1 problem whose weights are the penalty derivative at the current
estimate. For TLP this is exactly the difference-of-convex algorithm of Shen et al.

    SCAD  p'(t) = lam                        t <= lam
                = (a*lam - t)_+ / (a - 1)    t >  lam
    MCP   p'(t) = (lam - t/gamma)_+
    TLP   p(t)  = lam * min(t, tau)  ->  p'(t) = lam * 1{t < tau}
          (Shen et al.'s  lam' * min(t/tau, 1)  with lam' = lam * tau)

Predictors are assumed standardized. Supports Gaussian and binomial families.

Knockoffs
---------
Model-X Gaussian knockoffs (Candes, Fan, Janson & Lv 2018) with the equicorrelated
construction and a Ledoit-Wolf covariance estimate; knockoff / knockoff+ thresholds
(Barber & Candes 2015).
"""
import warnings

import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LassoCV, LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold

PENALTIES = ["Lasso", "SCAD", "MCP", "TLP"]


# ----------------------------------------------------------------------------- penalties
def penalty_deriv(name, t, lam, a=3.7, gamma=3.0, tau=0.5):
    t = np.abs(t)
    if name == "Lasso":
        return np.full_like(t, lam, dtype=float)
    if name == "SCAD":
        return np.where(t <= lam, lam, np.maximum(a * lam - t, 0) / (a - 1))
    if name == "MCP":
        return np.maximum(lam - t / gamma, 0)
    if name == "TLP":
        return np.where(t < tau, lam, 0.0)
    raise ValueError(name)


def weighted_l1(X, y, lam, w, family):
    """argmin loss + lam * sum_j w_j |b_j|  via column rescaling."""
    w = np.maximum(w, 1e-3 if family == "gaussian" else 1e-2)   # weight ~0 -> ~unpenalized
    Xw = X / w
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if family == "gaussian":
            m = Lasso(alpha=lam, max_iter=20000, tol=1e-5).fit(Xw, y)
            return m.coef_ / w, float(m.intercept_)
        C = 1.0 / (lam * len(y))
        m = LogisticRegression(penalty="l1", C=C, solver="liblinear",
                               intercept_scaling=100, max_iter=300, tol=1e-4).fit(Xw, y)
        return m.coef_.ravel() / w, float(m.intercept_[0])


def lla_fit(X, y, lam, penalty, family, n_iter=3, **kw):
    p = X.shape[1]
    b, b0 = weighted_l1(X, y, lam, np.ones(p), family)
    if penalty == "Lasso":
        return b, b0
    for _ in range(n_iter):
        w = penalty_deriv(penalty, b, lam, **kw) / lam
        b_new, b0 = weighted_l1(X, y, lam, w, family)
        done = np.max(np.abs(b_new - b)) < 1e-4
        b = b_new
        if done:
            break
    return b, b0


def lambda_grid(X, y, n=20, ratio=None, family="gaussian"):
    lmax = np.max(np.abs(X.T @ (y - y.mean()))) / len(y)
    if ratio is None:
        ratio = 0.01 if (len(y) > X.shape[1] and family == "gaussian") else 0.05
    return np.logspace(np.log10(lmax), np.log10(lmax * ratio), n)


def _loss(y, eta, family):
    if family == "gaussian":
        return np.mean((y - eta) ** 2)
    p = np.clip(1 / (1 + np.exp(-eta)), 1e-8, 1 - 1e-8)
    return -2 * np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


def cv_penalized(X, y, penalty, family, folds=5, seed=0, n_lam=20, rule="min",
                 n_iter=3, path=True, ebic_gamma=0.5, **kw):
    """Tune lambda by K-fold CV ('min' / '1se') or by an information criterion on the
    full-data path ('bic' / 'ebic'; Chen & Chen 2008), then refit at the chosen lambda.
    CV targets prediction error and tends to over-select; (E)BIC targets the true support."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, p = X.shape
    lams = lambda_grid(X, y, n_lam, family=family)
    splitter = (StratifiedKFold if family == "binomial" else KFold)(folds, shuffle=True, random_state=seed)
    m = se = None
    i_min = i_1se = 0
    if rule in ("min", "1se"):
        err = np.zeros((len(lams), folds))
        for k, (tr, te) in enumerate(splitter.split(X, y)):
            for i, lam in enumerate(lams):
                b, b0 = lla_fit(X[tr], y[tr], lam, penalty, family, n_iter, **kw)
                err[i, k] = _loss(y[te], X[te] @ b + b0, family)
        m, se = err.mean(1), err.std(1, ddof=1) / np.sqrt(folds)
        i_min = int(np.argmin(m))
        i_1se = int(np.where(m <= m[i_min] + se[i_min])[0].min())

    coefs = ints = ic = None
    if path or rule in ("bic", "ebic"):
        fits = [lla_fit(X, y, lam, penalty, family, n_iter, **kw) for lam in lams]
        coefs = np.array([f[0] for f in fits])
        ints = np.array([f[1] for f in fits])
        df_ = (coefs != 0).sum(1)
        if family == "gaussian":
            rss = np.array([np.sum((y - X @ b - b0) ** 2) for b, b0 in fits])
            fit_term = n * np.log(rss / n)
        else:
            fit_term = n * np.array([_loss(y, X @ b + b0, family) for b, b0 in fits])
        ic = fit_term + df_ * np.log(n)
        if rule == "ebic":
            ic = ic + 2 * ebic_gamma * df_ * np.log(p)
    i_sel = {"min": i_min, "1se": i_1se}.get(rule) if rule in ("min", "1se") else int(np.argmin(ic))
    if coefs is not None:
        b, b0 = coefs[i_sel], float(ints[i_sel])
    else:
        b, b0 = lla_fit(X, y, lams[i_sel], penalty, family, n_iter, **kw)
    return dict(lams=lams, cv_mean=m, cv_se=se, lam=lams[i_sel],
                lam_min=lams[i_min] if m is not None else None,
                lam_1se=lams[i_1se] if m is not None else None,
                coef=b, intercept=b0, path=coefs, ic=ic, rule=rule)


# ----------------------------------------------------------------------------- knockoffs
def gaussian_knockoffs(X, rng):
    """Equicorrelated model-X Gaussian knockoffs for standardized X."""
    X = np.asarray(X, float)
    n, p = X.shape
    sd = X.std(0)
    sd[sd == 0] = 1
    Xs = (X - X.mean(0)) / sd
    S = LedoitWolf().fit(Xs).covariance_
    d = np.sqrt(np.diag(S))
    S = S / np.outer(d, d)
    lam_min = np.linalg.eigvalsh(S).min()
    s = np.full(p, min(1.0, 2 * lam_min) * 0.99)
    Sinv = np.linalg.inv(S)
    D = np.diag(s)
    mu = Xs - Xs @ Sinv @ D
    V = 2 * D - D @ Sinv @ D
    V = (V + V.T) / 2
    ev, U = np.linalg.eigh(V)
    L = U * np.sqrt(np.clip(ev, 0, None))
    Xk = mu + rng.standard_normal((n, p)) @ L.T
    return Xs, Xk, s


def knockoff_stats(Xs, Xk, y, family, seed, stat="lasso"):
    p = Xs.shape[1]
    Z = np.hstack([Xs, Xk])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        if stat == "lasso":
            if family == "gaussian":
                b = LassoCV(cv=5, n_alphas=50, random_state=seed, max_iter=10000).fit(Z, y).coef_
            else:
                b = LogisticRegressionCV(Cs=12, cv=5, penalty="l1", solver="liblinear",
                                         scoring="neg_log_loss", max_iter=3000,
                                         random_state=seed).fit(Z, y).coef_.ravel()
            imp = np.abs(b)
        else:
            RF = RandomForestRegressor if family == "gaussian" else RandomForestClassifier
            imp = RF(500, max_features=0.33, random_state=seed, n_jobs=-1).fit(Z, y).feature_importances_
    return imp[:p] - imp[p:]


def knockoff_threshold(W, q, plus=True):
    off = 1 if plus else 0
    for t in np.sort(np.abs(W[W != 0])):
        if (off + np.sum(W <= -t)) / max(1, np.sum(W >= t)) <= q:
            return float(t)
    return np.inf


def run_knockoffs(X, y, family, q=0.1, plus=True, stat="lasso", seed=0):
    rng = np.random.default_rng(seed)
    Xs, Xk, s = gaussian_knockoffs(X, rng)
    W = knockoff_stats(Xs, Xk, np.asarray(y), family, seed, stat)
    T = knockoff_threshold(W, q, plus)
    return dict(W=W, T=T, selected=np.where(W >= T)[0], s=s)


def fdp_power(selected, active):
    selected, active = set(selected), set(active)
    fdp = len(selected - active) / max(1, len(selected))
    power = len(selected & active) / max(1, len(active))
    return fdp, power
