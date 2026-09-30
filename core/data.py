"""Data-generating processes. Each simulator returns (DataFrame, truth dict)."""
import numpy as np
import pandas as pd


def expit(x):
    return 1 / (1 + np.exp(-x))


def sim_linear(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.normal(55, 10, n)
    bmi = rng.normal(27, 4, n)
    sex = rng.choice(["F", "M"], n)
    dose = rng.uniform(0, 10, n)
    y = 20 + 0.3 * age + 0.8 * bmi + 2 * (sex == "M") + 1.5 * np.sqrt(dose) + rng.normal(0, 4, n)
    return pd.DataFrame(dict(y=y, age=age, bmi=bmi, sex=sex, dose=dose)), {}


def sim_binary(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.normal(60, 9, n)
    support = rng.integers(0, 4, n)
    smoker = rng.binomial(1, 0.3, n)
    stage = rng.choice(["I", "II", "III", "IV"], n, p=[.2, .3, .3, .2])
    stage_eff = pd.Series(stage).map({"I": 0, "II": .3, "III": .7, "IV": 1.2}).to_numpy()
    lp = -1 + 0.03 * (age - 60) + 0.45 * support - 0.6 * smoker - stage_eff
    adherent = rng.binomial(1, expit(lp))
    return pd.DataFrame(dict(adherent=adherent, age=age, support=support,
                             smoker=smoker, stage=stage)), {}


def sim_survival(n, seed):
    rng = np.random.default_rng(seed)
    age = rng.normal(62, 8, n)
    treat = rng.binomial(1, 0.5, n)
    support = rng.choice(["low", "medium", "high"], n)
    sup_eff = pd.Series(support).map({"low": 0, "medium": -.3, "high": -.6}).to_numpy()
    lp = 0.03 * (age - 62) - 0.5 * treat + sup_eff
    t_event = rng.weibull(1.3, n) * 36 * np.exp(-lp / 1.3)
    t_cens = rng.uniform(12, 60, n)
    return pd.DataFrame(dict(time=np.minimum(t_event, t_cens),
                             event=(t_event <= t_cens).astype(int),
                             age=age, treat=treat, support=support)), {}


def sim_highdim(n, seed, p=100, k=10, rho=0.5, amp=0.6, family="gaussian"):
    """AR(1)-correlated design with k active predictors at random positions."""
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, p))
    X = np.empty_like(Z)
    X[:, 0] = Z[:, 0]
    for j in range(1, p):
        X[:, j] = rho * X[:, j - 1] + np.sqrt(1 - rho ** 2) * Z[:, j]
    active = np.sort(rng.choice(p, k, replace=False))
    beta = np.zeros(p)
    beta[active] = rng.choice([-1, 1], k) * amp
    eta = X @ beta
    y = eta + rng.standard_normal(n) if family == "gaussian" else rng.binomial(1, expit(eta))
    cols = [f"x{j + 1}" for j in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, "y", y)
    return df, {"active": [cols[j] for j in active],
                "beta": {cols[j]: float(beta[j]) for j in active}}


def sim_counts(n, seed):
    """Zero-inflated Poisson counts with a follow-up-time offset."""
    rng = np.random.default_rng(seed)
    age = rng.normal(50, 12, n)
    exposure = rng.uniform(0, 3, n)
    smoker = rng.binomial(1, .35, n)
    sex = rng.choice(["F", "M"], n)
    followup = rng.uniform(0.5, 2, n)
    pi = expit(-1.2 + 1.5 * smoker + 0.02 * (age - 50))
    mu = followup * np.exp(0.3 + 0.4 * exposure + 0.3 * (sex == "M") - 0.01 * (age - 50))
    y = np.where(rng.random(n) < pi, 0, rng.poisson(mu))
    df = pd.DataFrame(dict(visits=y, age=age, exposure=exposure, smoker=smoker,
                           sex=sex, followup=followup))
    return df, {"count_part": {"exposure": 0.4, "sex[M]": 0.3, "age": -0.01},
                "zero_part": {"smoker": 1.5, "age": 0.02}}


def sim_causal(n, seed):
    """Confounded binary treatment with a heterogeneous effect tau(x) = 1 + x1 + 1{x3 > 0}."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, 5))
    female = rng.binomial(1, .4, n)
    e = expit(-0.2 + 0.8 * X[:, 0] - 0.6 * X[:, 1] + 0.5 * female)
    t = rng.binomial(1, e)
    tau = 1 + X[:, 0] + (X[:, 2] > 0)
    y0 = (2 + 1.2 * X[:, 0] + 0.8 * X[:, 1] + 0.5 * X[:, 1] ** 2 + np.sin(X[:, 3])
          + 0.7 * female + rng.standard_normal(n))
    df = pd.DataFrame(X, columns=[f"x{j + 1}" for j in range(5)])
    df.insert(0, "treated", t)
    df.insert(0, "outcome", y0 + t * tau)
    df["female"] = female
    return df, {"ate": float(tau.mean()), "cate": pd.Series(tau, index=df.index),
                "treatment": "treated", "outcome": "outcome"}


def sim_clusters(n, seed):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0, 0, 0], [3, 3, 0, 1], [0, 3, 3, -1]], float)
    g = rng.choice(3, n, p=[.4, .35, .25])
    X = centers[g] + rng.standard_normal((n, 4)) * np.array([1, 1, .8, 1.2])
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["true_group"] = np.array(["A", "B", "C"])[g]
    return df, {"labels": "true_group"}


SIMULATORS = {
    "Continuous outcome (linear model)": sim_linear,
    "Binary outcome: treatment adherence": sim_binary,
    "Time-to-event: survival": sim_survival,
    "High-dimensional sparse (p=100, 10 true signals)": sim_highdim,
    "Counts with excess zeros (ZIP)": sim_counts,
    "Causal: confounded treatment, heterogeneous effect": sim_causal,
    "Clusters (3 groups, 4 features)": sim_clusters,
}
