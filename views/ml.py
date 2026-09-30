import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
from scipy.cluster.hierarchy import linkage
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import (GradientBoostingClassifier, GradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression, PoissonRegressor
from sklearn.metrics import (accuracy_score, adjusted_rand_score, balanced_accuracy_score,
                             brier_score_loss, confusion_matrix, f1_score, log_loss,
                             mean_absolute_error, mean_squared_error, normalized_mutual_info_score,
                             r2_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree

from core.utils import (ACCENT, HAS_XGB, PALETTE, design_matrix, fmt_table, get_df, log_result,
                        need_data, numeric_cols, set_df, truth)


def _hyperparams(prefix):
    with st.expander("Hyperparameters"):
        c = st.columns(4)
        hp = dict(
            C=c[0].number_input("Logistic C (inverse L2 strength)", 0.001, 1e4, 1.0, key=prefix + "C"),
            depth=c[1].slider("Tree max depth", 1, 20, 4, key=prefix + "depth"),
            leaf=c[2].slider("Tree min samples per leaf", 1, 100, 10, key=prefix + "leaf"),
            k=c[3].slider("k-NN neighbours", 1, 50, 15, key=prefix + "k"),
        )
        c = st.columns(4)
        hp.update(
            n_trees=c[0].slider("Forest trees", 50, 1000, 300, 50, key=prefix + "nt"),
            n_boost=c[1].slider("Boosting rounds", 50, 1000, 200, 50, key=prefix + "nb"),
            lr=c[2].select_slider("Boosting learning rate", [0.01, 0.03, 0.05, 0.1, 0.2, 0.3], 0.05,
                                  key=prefix + "lr"),
            bdepth=c[3].slider("Boosting tree depth", 1, 10, 3, key=prefix + "bd"),
        )
    return hp


def clf_zoo(hp, seed):
    zoo = {
        "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(C=hp["C"], max_iter=5000)),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Decision tree": DecisionTreeClassifier(max_depth=hp["depth"], min_samples_leaf=hp["leaf"],
                                                random_state=seed),
        "Random forest": RandomForestClassifier(hp["n_trees"], random_state=seed, n_jobs=-1),
        "Gradient boosting": GradientBoostingClassifier(n_estimators=hp["n_boost"], learning_rate=hp["lr"],
                                                        max_depth=hp["bdepth"], random_state=seed),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed)),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsClassifier(hp["k"])),
    }
    if HAS_XGB:
        from xgboost import XGBClassifier
        zoo["XGBoost"] = XGBClassifier(n_estimators=hp["n_boost"], learning_rate=hp["lr"], max_depth=hp["bdepth"],
                                       subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1,
                                       verbosity=0)
    return zoo


def reg_zoo(hp, seed, nonneg):
    zoo = {
        "Linear regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Decision tree": DecisionTreeRegressor(max_depth=hp["depth"], min_samples_leaf=hp["leaf"], random_state=seed),
        "Random forest": RandomForestRegressor(hp["n_trees"], random_state=seed, n_jobs=-1),
        "Gradient boosting": GradientBoostingRegressor(n_estimators=hp["n_boost"], learning_rate=hp["lr"],
                                                       max_depth=hp["bdepth"], random_state=seed),
        "SVR (RBF)": make_pipeline(StandardScaler(), SVR()),
        "k-NN": make_pipeline(StandardScaler(), KNeighborsRegressor(hp["k"])),
    }
    if nonneg:
        zoo["Poisson regression (GLM)"] = make_pipeline(StandardScaler(), PoissonRegressor(alpha=1e-4, max_iter=1000))
    if HAS_XGB:
        from xgboost import XGBRegressor
        zoo["XGBoost"] = XGBRegressor(n_estimators=hp["n_boost"], learning_rate=hp["lr"], max_depth=hp["bdepth"],
                                      subsample=0.8, colsample_bytree=0.8, random_state=seed, n_jobs=-1, verbosity=0)
    return zoo


def _importance(model, X, y, scoring, seed, stratify):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=seed,
                                          stratify=y if stratify else None)
    m = model.fit(Xtr, ytr)
    pi = permutation_importance(m, Xte, yte, n_repeats=10, random_state=seed, scoring=scoring, n_jobs=-1)
    return pd.DataFrame({"importance": pi.importances_mean, "sd": pi.importances_std},
                        index=X.columns).sort_values("importance")


def _tree_figure(tree, names, class_names=None, max_depth=3):
    depth = min(max_depth, tree.get_depth())
    fig, ax = plt.subplots(figsize=(min(24, 3.2 * 2 ** depth), 2.4 * depth + 2))
    plot_tree(tree, feature_names=names, class_names=class_names, filled=True, rounded=True,
              max_depth=max_depth, fontsize=8, precision=2, impurity=False, proportion=True, ax=ax)
    return fig


# ============================================================================= Classification
def page_classification():
    st.header("Classification")
    need_data()
    df = get_df()
    cands = [c for c in df.columns if 2 <= df[c].nunique() <= 10]
    if not cands:
        st.warning("No column with 2–10 classes to predict.")
        st.stop()
    c1, c2 = st.columns([1, 2])
    outcome = c1.selectbox("Class label", cands)
    others = [c for c in df.columns if c != outcome]
    preds = c2.multiselect("Features", others, default=others)
    c1, c2, c3 = st.columns([3, 1, 1])
    hp = _hyperparams("clf_")
    zoo = clf_zoo(hp, 0)
    default = [m for m in ["Logistic regression", "Naive Bayes (Gaussian)", "Decision tree",
                           "Gradient boosting", "XGBoost"] if m in zoo]
    chosen = c1.multiselect("Models", list(zoo), default=default)
    folds = c2.slider("CV folds", 3, 10, 5)
    seed = c3.number_input("Seed", 0, 10**6, 1, key="clf_seed")
    if not preds or not chosen:
        st.stop()

    if st.button("Run cross-validation", type="primary"):
        X, yraw, dropped, _ = design_matrix(df, outcome, preds)
        classes = np.array(sorted(yraw.unique(), key=str))
        y = pd.Series(pd.Categorical(yraw, categories=classes).codes, index=yraw.index)
        zoo = clf_zoo(hp, seed)
        cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
        proba, rows = {}, []
        prog = st.progress(0.0)
        for i, m in enumerate(chosen):
            prog.progress(i / len(chosen), f"Cross-validating {m}…")
            P = cross_val_predict(zoo[m], X, y, cv=cv, method="predict_proba")
            proba[m] = P
            rows.append(dict(model=m, **_clf_metrics(y, P)))
        prog.empty()
        res = pd.DataFrame(rows).set_index("model")
        st.session_state.res_clf = dict(proba=proba, y=y.to_numpy(), classes=classes, X=X, outcome=outcome,
                                        metrics=res, hp=hp, seed=seed, chosen=chosen, dropped=dropped)
        log_result("Classification benchmark", dict(outcome=outcome, features=preds, models=chosen, folds=folds,
                                                     seed=seed, hyperparameters=hp),
                   tables={"Out-of-fold metrics": res})

    R = st.session_state.get("res_clf")
    if not R or R["outcome"] != outcome:
        return
    y, classes, binary = R["y"], R["classes"], len(R["classes"]) == 2
    st.subheader(f"Out-of-fold performance ({folds}-fold CV)")
    st.caption(f"n = {len(y):,} ({R['dropped']} dropped for missingness) · classes: "
               + ", ".join(f"{c} ({(y == i).sum()})" for i, c in enumerate(classes)))
    met = R["metrics"]
    hi = [c for c in met if c in ("AUC", "accuracy", "balanced accuracy", "F1")]
    lo = [c for c in met if c in ("Brier", "log loss")]
    st.dataframe(met.style.format(precision=3).highlight_max(subset=hi, color="#DCE6F2")
                 .highlight_min(subset=lo, color="#DCE6F2"), width="stretch")

    if binary:
        c1, c2 = st.columns(2)
        roc = go.Figure()
        cal = go.Figure()
        for i, (m, P) in enumerate(R["proba"].items()):
            p = P[:, 1]
            fpr, tpr, _ = roc_curve(y, p)
            roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{m} ({roc_auc_score(y, p):.3f})",
                                     line_color=PALETTE[i % 7]))
            g = pd.DataFrame(dict(p=p, y=y)).groupby(pd.qcut(p, 10, duplicates="drop"), observed=True).mean()
            cal.add_trace(go.Scatter(x=g.p, y=g.y, mode="lines+markers", name=m, line_color=PALETTE[i % 7]))
        for f in (roc, cal):
            f.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="grey"))
        roc.update_layout(title="ROC curves", xaxis_title="1 − specificity", yaxis_title="Sensitivity")
        cal.update_layout(title="Calibration (deciles)", xaxis_title="Predicted probability",
                          yaxis_title="Observed proportion")
        c1.plotly_chart(roc, width="stretch")
        c2.plotly_chart(cal, width="stretch")

    st.subheader("Confusion matrix")
    c1, c2 = st.columns([1, 2])
    m = c1.selectbox("Model", list(R["proba"]), key="cm_model")
    P = R["proba"][m]
    if binary:
        thr = c1.slider(f"Threshold for '{classes[1]}'", 0.01, 0.99, 0.5, 0.01)
        pred = (P[:, 1] >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        c1.write(f"Sensitivity **{tp / max(1, tp + fn):.3f}** · Specificity **{tn / max(1, tn + fp):.3f}**  \n"
                 f"PPV **{tp / max(1, tp + fp):.3f}** · NPV **{tn / max(1, tn + fn):.3f}**")
    else:
        pred = P.argmax(1)
    cm = confusion_matrix(y, pred, labels=range(len(classes)))
    c2.plotly_chart(px.imshow(cm, text_auto=True, x=[str(c) for c in classes], y=[str(c) for c in classes],
                              labels=dict(x="Predicted", y="Actual"), color_continuous_scale="Blues"),
                    width="stretch")

    if "Decision tree" in R["proba"]:
        st.subheader("Decision tree (fitted on all data)")
        tree = clf_zoo(R["hp"], R["seed"])["Decision tree"].fit(R["X"], y)
        show = st.slider("Levels to display", 1, max(1, tree.get_depth()), min(3, max(1, tree.get_depth())))
        fig = _tree_figure(tree, list(R["X"].columns), [str(c) for c in classes], show)
        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Permutation importance (25% hold-out)")
    m2 = st.selectbox("Model", list(R["proba"]), key="imp_model")
    if st.button("Compute importance"):
        imp = _importance(clf_zoo(R["hp"], R["seed"])[m2], R["X"], y,
                          "roc_auc" if binary else "accuracy", R["seed"], True).tail(20)
        st.plotly_chart(px.bar(imp, x="importance", error_x="sd", orientation="h",
                               labels={"importance": "Drop in score when permuted", "index": ""}),
                        width="stretch")


def _clf_metrics(y, P):
    pred = P.argmax(1)
    out = {}
    if P.shape[1] == 2:
        out["AUC"] = roc_auc_score(y, P[:, 1])
        out["Brier"] = brier_score_loss(y, P[:, 1])
    else:
        out["AUC"] = roc_auc_score(y, P, multi_class="ovr")
    out.update({"accuracy": accuracy_score(y, pred), "balanced accuracy": balanced_accuracy_score(y, pred),
                "F1": f1_score(y, pred, average="binary" if P.shape[1] == 2 else "macro"),
                "log loss": log_loss(y, np.clip(P, 1e-15, 1), labels=range(P.shape[1]))})
    return out


# ============================================================================= Prediction
def page_prediction():
    st.header("Prediction (continuous and count outcomes)")
    need_data()
    df = get_df()
    num = [c for c in numeric_cols(df) if df[c].nunique() > 2]
    if not num:
        st.warning("No numeric outcome with more than two values.")
        st.stop()
    c1, c2 = st.columns([1, 2])
    outcome = c1.selectbox("Outcome", num)
    others = [c for c in df.columns if c != outcome]
    preds = c2.multiselect("Features", others, default=others)
    nonneg = bool((df[outcome].dropna() >= 0).all())
    hp = _hyperparams("reg_")
    zoo = reg_zoo(hp, 0, nonneg)
    c1, c2, c3 = st.columns([3, 1, 1])
    chosen = c1.multiselect("Models", list(zoo), default=[m for m in ["Linear regression", "Random forest",
                                                                      "Gradient boosting", "XGBoost"] if m in zoo])
    folds = c2.slider("CV folds", 3, 10, 5, key="reg_folds")
    seed = c3.number_input("Seed", 0, 10**6, 1, key="reg_seed")
    if not preds or not chosen:
        st.stop()

    if st.button("Run cross-validation", type="primary"):
        X, y, dropped, _ = design_matrix(df, outcome, preds)
        zoo = reg_zoo(hp, seed, nonneg)
        cv = KFold(folds, shuffle=True, random_state=seed)
        pred, rows = {}, []
        prog = st.progress(0.0)
        for i, m in enumerate(chosen):
            prog.progress(i / len(chosen), f"Cross-validating {m}…")
            p = cross_val_predict(zoo[m], X, y, cv=cv)
            pred[m] = p
            rows.append(dict(model=m, RMSE=np.sqrt(mean_squared_error(y, p)), MAE=mean_absolute_error(y, p),
                             R2=r2_score(y, p)))
        prog.empty()
        res = pd.DataFrame(rows).set_index("model").sort_values("RMSE")
        st.session_state.res_reg = dict(pred=pred, y=y.to_numpy(float), X=X, outcome=outcome, metrics=res,
                                        hp=hp, seed=seed, nonneg=nonneg, dropped=dropped)
        log_result("Prediction benchmark", dict(outcome=outcome, features=preds, models=chosen, folds=folds,
                                                seed=seed, hyperparameters=hp),
                   tables={"Out-of-fold metrics": res})

    R = st.session_state.get("res_reg")
    if not R or R["outcome"] != outcome:
        return
    st.subheader("Out-of-fold performance")
    st.dataframe(R["metrics"].style.format(precision=3).highlight_min(subset=["RMSE", "MAE"], color="#DCE6F2")
                 .highlight_max(subset=["R2"], color="#DCE6F2"), width="stretch")
    m = st.selectbox("Inspect model", list(R["pred"]))
    p, y = R["pred"][m], R["y"]
    c1, c2 = st.columns(2)
    f1 = px.scatter(x=p, y=y, opacity=.5, labels={"x": "Predicted (out-of-fold)", "y": "Observed"},
                    color_discrete_sequence=[ACCENT])
    lim = [min(p.min(), y.min()), max(p.max(), y.max())]
    f1.add_shape(type="line", x0=lim[0], y0=lim[0], x1=lim[1], y1=lim[1], line=dict(dash="dash", color="grey"))
    c1.plotly_chart(f1, width="stretch")
    c2.plotly_chart(px.scatter(x=p, y=y - p, opacity=.5, trendline="lowess",
                               labels={"x": "Predicted", "y": "Residual"}, color_discrete_sequence=[ACCENT]),
                    width="stretch")
    if m == "Decision tree":
        tree = reg_zoo(R["hp"], R["seed"], R["nonneg"])["Decision tree"].fit(R["X"], y)
        fig = _tree_figure(tree, list(R["X"].columns), None, 3)
        st.pyplot(fig)
        plt.close(fig)
    if st.button("Compute permutation importance (25% hold-out)"):
        imp = _importance(reg_zoo(R["hp"], R["seed"], R["nonneg"])[m], R["X"], y, "r2", R["seed"], False).tail(20)
        st.plotly_chart(px.bar(imp, x="importance", error_x="sd", orientation="h",
                               labels={"importance": "Drop in R² when permuted", "index": ""}), width="stretch")


# ============================================================================= Clustering & PCA
def page_clustering():
    st.header("Clustering & PCA")
    need_data()
    df = get_df()
    num = numeric_cols(df)
    if len(num) < 2:
        st.warning("Need at least two numeric columns.")
        st.stop()
    t1, t2 = st.tabs(["Principal component analysis", "Clustering"])

    with t1:
        cols = st.multiselect("Variables", num, default=num, key="pca_cols")
        std = st.checkbox("Standardize variables (PCA on the correlation matrix)", True)
        if len(cols) < 2:
            st.stop()
        d = df[cols].dropna()
        Z = StandardScaler().fit_transform(d) if std else (d - d.mean()).to_numpy()
        pca = PCA().fit(Z)
        ev, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
        c1, c2, c3 = st.columns(3)
        c1.metric("Components for 80% variance", int(np.searchsorted(cum, 0.8) + 1))
        c2.metric("Components for 90% variance", int(np.searchsorted(cum, 0.9) + 1))
        if std:
            c3.metric("Eigenvalues > 1 (Kaiser)", int((pca.explained_variance_ > 1).sum()))
        kshow = min(len(ev), 20)
        scree = go.Figure(go.Bar(x=np.arange(1, kshow + 1), y=ev[:kshow], name="proportion", marker_color=ACCENT))
        scree.add_trace(go.Scatter(x=np.arange(1, kshow + 1), y=cum[:kshow], name="cumulative",
                                   mode="lines+markers", line_color="#C0504D"))
        scree.update_layout(title="Scree plot", xaxis_title="Component", yaxis_title="Variance explained")
        st.plotly_chart(scree, width="stretch")

        scores = pca.transform(Z)
        c1, c2, c3 = st.columns(3)
        i = c1.selectbox("x axis", range(1, len(ev) + 1), 0, format_func=lambda k: f"PC{k}") - 1
        j = c2.selectbox("y axis", range(1, len(ev) + 1), 1, format_func=lambda k: f"PC{k}") - 1
        color = c3.selectbox("Colour points by", ["—"] + df.columns.tolist(), key="pca_c")
        bp = px.scatter(x=scores[:, i], y=scores[:, j], opacity=.55,
                        color=None if color == "—" else df.loc[d.index, color].astype(str)
                        if df[color].nunique() <= 12 else df.loc[d.index, color],
                        labels={"x": f"PC{i + 1} ({ev[i]:.1%})", "y": f"PC{j + 1} ({ev[j]:.1%})"},
                        color_discrete_sequence=PALETTE)
        load = pca.components_.T * np.sqrt(pca.explained_variance_)
        scale = np.abs(scores[:, [i, j]]).max() / max(1e-9, np.abs(load[:, [i, j]]).max()) * 0.8
        top = np.argsort(-np.hypot(load[:, i], load[:, j]))[:12]
        for v in top:
            bp.add_annotation(x=load[v, i] * scale, y=load[v, j] * scale, ax=0, ay=0, xref="x", yref="y",
                              axref="x", ayref="y", showarrow=True, arrowhead=2, arrowcolor="#333",
                              text=cols[v], font=dict(size=11))
        bp.update_layout(title="Biplot (arrows: loadings of the 12 strongest variables)")
        st.plotly_chart(bp, width="stretch")
        k = min(len(ev), 6)
        L = pd.DataFrame(load[:, :k], index=cols, columns=[f"PC{a + 1}" for a in range(k)])
        st.subheader("Loadings (correlation of each variable with each component)" if std else "Loadings")
        st.dataframe(L.style.format(precision=3).background_gradient(cmap="RdBu_r", vmin=-1, vmax=1),
                     width="stretch")
        c1, c2 = st.columns([1, 3])
        kk = c1.number_input("Scores to add", 1, len(ev), min(2, len(ev)))
        if c2.button("Add PC scores to the active dataset"):
            new = df.copy()
            for a in range(int(kk)):
                new.loc[d.index, f"PC{a + 1}"] = scores[:, a]
            set_df(new, st.session_state.df_name + " [+PCs]", truth(), keep_original=True)
            log_result("PCA", dict(variables=cols, standardized=std),
                       tables={"Variance explained": pd.DataFrame({"proportion": ev, "cumulative": cum}),
                               "Loadings": L}, figs=[scree])
            st.rerun()

    with t2:
        default = [c for c in num if df[c].nunique() > 2 and not c.startswith("PC")]
        cols = st.multiselect("Variables", num, default=default or num, key="cl_cols")
        c = st.columns(4)
        algo = c[0].selectbox("Algorithm", ["K-means", "Hierarchical (Ward)", "Hierarchical (average)",
                                            "Gaussian mixture"])
        k = c[1].slider("Number of clusters", 2, 12, 3)
        std2 = c[2].checkbox("Standardize", True, key="cl_std")
        seed = c[3].number_input("Seed", 0, 10**6, 0, key="cl_seed")
        if len(cols) < 2:
            st.stop()
        d = df[cols].dropna()
        Z = StandardScaler().fit_transform(d) if std2 else d.to_numpy(float)

        def fit(kk):
            if algo == "K-means":
                m = KMeans(kk, n_init=10, random_state=seed).fit(Z)
                return m.labels_, m.inertia_
            if algo == "Gaussian mixture":
                m = GaussianMixture(kk, n_init=3, random_state=seed).fit(Z)
                return m.predict(Z), m.bic(Z)
            link = "ward" if "Ward" in algo else "average"
            return AgglomerativeClustering(kk, linkage=link).fit_predict(Z), None

        with st.expander("Choose k: silhouette and elbow / BIC across k = 2…10"):
            if st.button("Evaluate k"):
                sub = np.random.default_rng(seed).choice(len(Z), min(len(Z), 3000), replace=False)
                rows = []
                for kk in range(2, 11):
                    lab, crit = fit(kk)
                    rows.append(dict(k=kk, silhouette=silhouette_score(Z[sub], lab[sub]), criterion=crit))
                ev = pd.DataFrame(rows)
                cc1, cc2 = st.columns(2)
                cc1.plotly_chart(px.line(ev, x="k", y="silhouette", markers=True, title="Mean silhouette (higher is better)"),
                                 width="stretch")
                if ev.criterion.notna().any():
                    cc2.plotly_chart(px.line(ev, x="k", y="criterion", markers=True,
                                             title="Within-cluster SS (elbow)" if algo == "K-means" else "BIC (lower is better)"),
                                     width="stretch")

        labels, _ = fit(k)
        labels = pd.Series(labels, index=d.index).astype(str)
        pcs = PCA(2).fit_transform(Z)
        c1, c2 = st.columns([2, 1])
        c1.plotly_chart(px.scatter(x=pcs[:, 0], y=pcs[:, 1], color=labels, opacity=.6,
                                   labels={"x": "PC1", "y": "PC2", "color": "cluster"},
                                   color_discrete_sequence=PALETTE, title=f"{algo}, k = {k} (shown in PC space)"),
                        width="stretch")
        sub = np.random.default_rng(seed).choice(len(Z), min(len(Z), 3000), replace=False)
        c2.metric("Mean silhouette", f"{silhouette_score(Z[sub], labels.to_numpy()[sub]):.3f}")
        c2.dataframe(labels.value_counts().rename("size").sort_index(), width="stretch")
        refs = [c for c in df.columns if 2 <= df[c].nunique() <= 20 and c not in cols]
        tr_lab = truth().get("labels")
        ref = c2.selectbox("Compare with reference labels", ["—"] + refs,
                           index=(["—"] + refs).index(tr_lab) if tr_lab in refs else 0)
        if ref != "—":
            r = df.loc[d.index, ref].astype(str)
            c2.write(f"ARI = **{adjusted_rand_score(r, labels):.3f}** · NMI = "
                     f"**{normalized_mutual_info_score(r, labels):.3f}**")
        st.subheader("Cluster profiles (means)")
        prof = df.loc[d.index, cols].groupby(labels).mean()
        st.dataframe(prof.style.format(precision=2).background_gradient(axis=0, cmap="Blues"), width="stretch")
        if algo.startswith("Hierarchical"):
            with st.expander("Dendrogram (random sample of up to 150 units)"):
                s = np.random.default_rng(seed).choice(len(Z), min(len(Z), 150), replace=False)
                method = "ward" if "Ward" in algo else "average"
                dg = ff.create_dendrogram(Z[s], linkagefun=lambda x: linkage(x, method))
                dg.update_layout(xaxis_showticklabels=False, height=400)
                st.plotly_chart(dg, width="stretch")
        if st.button("Add cluster labels to the active dataset"):
            new = df.copy()
            new.loc[d.index, "cluster"] = labels
            set_df(new, st.session_state.df_name + " [+clusters]", truth(), keep_original=True)
            log_result("Clustering", dict(variables=cols, algorithm=algo, k=k, standardized=std2, seed=seed),
                       tables={"Cluster profiles": prof})
            st.rerun()
