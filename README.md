# StatML Lab

An interactive statistics + machine-learning workbench built with Streamlit.

    pip install -r requirements.txt
    streamlit run app.py

## Layout

```
app.py                  navigation (sections below)
core/
  data.py               simulated datasets with known truth
  selection.py          Lasso / SCAD / MCP / TLP (LLA), CV + BIC/EBIC tuning, model-X knockoffs
  causal.py             cross-fitted nuisances, naive / g-comp / IPW / AIPW, S/T/X/DR-learners, GATES
  utils.py              shared helpers, forest plots, report logging
views/
  data_analysis.py      Load data · Diagnostics · Missing data · Visualization
  methods.py            GLM · Variable selection · Causal inference · Survival · Count models
  ml.py                 Prediction · Classification · Clustering & PCA
  simulation.py         Selection FDR/power · double robustness · CI coverage
  reports.py            HTML report, JSON log, session info, pinned requirements
```

## What each section does

**Data Analysis.** Simulate or upload data; inject MCAR/MAR missingness; diagnostics
(types, duplicates, IQR outliers, correlated pairs, VIF, skewness, Shapiro–Wilk); missingness
patterns, an MCAR check, and imputation (mean/median, KNN, iterative); plots.

**Statistical Methods.**
- GLM: linear, logistic, Poisson; OR/RR forest plots; HC3 SEs; diagnostics.
- Variable selection: Lasso, SCAD, MCP, TLP via local linear approximation (DC algorithm
  for TLP); λ by EBIC, BIC, or CV; coefficient paths; knockoff / knockoff+ filter with
  lasso or random-forest statistics and multi-draw stability.
- Causal inference (binary treatment): naive, g-computation, IPW, cross-fitted AIPW for the
  ATE; propensity overlap and love plot; S/T/X/DR-learners for the CATE; GATES calibration;
  summary tree of effect drivers.
- Survival: Kaplan–Meier + log-rank, Cox PH with Schoenfeld test.
- Count models: Poisson, NB2, zero-inflated Poisson with offsets, Vuong test, observed vs
  expected frequencies.

**Machine Learning.** Prediction (linear, Poisson GLM, tree, RF, GBM, XGBoost, SVR, k-NN) and
classification (logistic, Gaussian naive Bayes, decision tree, RF, GBM, XGBoost, SVM, k-NN)
with out-of-fold metrics, ROC, calibration, confusion matrices with threshold, tree plots,
and hold-out permutation importance. PCA (scree, biplot, loadings) and clustering (k-means,
hierarchical, Gaussian mixture; silhouette, elbow/BIC, ARI/NMI).

**Simulation Laboratory.** FDR and power of the selection methods; Kang–Schafer
double-robustness study; confidence-interval coverage.

**Reports & Reproducibility.** Every run is logged with settings, seed and a data
fingerprint; export a self-contained HTML report, a JSON log, or pinned requirements.

## Extending
Add a function to a `views/` module and register it with `st.Page(...)` in `app.py`.
