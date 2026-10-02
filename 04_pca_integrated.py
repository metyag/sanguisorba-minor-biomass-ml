# -*- coding: utf-8 -*-
"""
04 — PCA, INTEGRATED INTO THE MACHINE-LEARNING FRAMEWORK.

Reviewer 2 (comment 4) and the Editor both state that the PCA section is
descriptive only and is not used for model development, feature selection or
interpretation, and should therefore either be justified or removed.

This script gives PCA an explicit, testable role:
  (a) the descriptive PCA is recomputed and reported properly
      (eigenvalues, explained variance, communalities, Varimax-rotated loadings);
  (b) PC scores are then used AS PREDICTORS inside the same cross-validated
      pipeline, so that the contribution of PCA to predictive performance can be
      measured rather than asserted;
  (c) PCA is also used as a multicollinearity diagnostic (condition number, VIF).

Outputs (results/tables/):
  04_pca_eigenvalues.csv, 04_pca_loadings.csv, 04_pca_varimax.csv,
  04_pca_communalities.csv, 04_pca_vif.csv, 04_pca_ml_benchmark.csv,
  04_pca_summary.json
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold, GroupKFold

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import load_data, build_models, make_pipeline, SCORING, SEED, TAB_DIR, MORPH_FEATURES, TARGET_EN

pd.set_option("display.width", 240)
df, y, groups_cell, groups_geno, groups_year = load_data()

print("=" * 90)
print("04 — PCA INTEGRATED INTO THE ML FRAMEWORK")
print("=" * 90)

# The descriptive PCA runs on nine variables over the plant records: the eight
# morphological traits plus fresh plant weight. The manuscript does not report the
# components themselves; what it takes from this script is the collinearity
# diagnostics computed below, the condition number and the variance inflation
# factors of the eight predictors.
PCA_VARS = MORPH_FEATURES + [TARGET_EN]
print(f"Variables entered into the descriptive PCA (n = {len(PCA_VARS)}): {PCA_VARS}")
print(f"Observations: {len(df)}")

Z = StandardScaler().fit_transform(df[PCA_VARS].values)

# ---------------------------------------------------------------
# (a) Descriptive PCA
# ---------------------------------------------------------------
pca_full = PCA(n_components=len(PCA_VARS), random_state=SEED).fit(Z)
eig = pca_full.explained_variance_
evr = pca_full.explained_variance_ratio_
eig_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(eig))],
    "Eigenvalue": eig,
    "% of variance": 100 * evr,
    "Cumulative %": 100 * np.cumsum(evr),
})
eig_df.to_csv(os.path.join(TAB_DIR, "04_pca_eigenvalues.csv"), index=False)
print("\n### Eigenvalues / explained variance")
print(eig_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

n_kaiser = int((eig > 1).sum())
print(f"\nKaiser criterion (eigenvalue > 1) retains {n_kaiser} components, "
      f"explaining {100*evr[:n_kaiser].sum():.2f}% of total variance.")

# Loadings (component matrix = eigenvector * sqrt(eigenvalue))
load = pca_full.components_[:n_kaiser].T * np.sqrt(eig[:n_kaiser])
load_df = pd.DataFrame(load, index=PCA_VARS,
                       columns=[f"PC{i+1}" for i in range(n_kaiser)])
load_df.to_csv(os.path.join(TAB_DIR, "04_pca_loadings.csv"))
print("\n### Unrotated loadings")
print(load_df.to_string(float_format=lambda v: f"{v:.4f}"))

# Communalities
comm = (load ** 2).sum(axis=1)
comm_df = pd.DataFrame({"Variable": PCA_VARS, "Communality": comm}) \
    .sort_values("Communality", ascending=False)
comm_df.to_csv(os.path.join(TAB_DIR, "04_pca_communalities.csv"), index=False)
print("\n### Communalities (2-component solution)")
print(comm_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def varimax(Phi, gamma=1.0, q=100, tol=1e-6):
    """Kaiser Varimax rotation."""
    p, k = Phi.shape
    R = np.eye(k)
    d = 0.0
    for _ in range(q):
        d_old = d
        L = Phi @ R
        u, s, vh = np.linalg.svd(
            Phi.T @ (L ** 3 - (gamma / p) * L @ np.diag(np.diag(L.T @ L))))
        R = u @ vh
        d = float(np.sum(s))
        if d_old != 0 and d / d_old < 1 + tol:
            break
    return Phi @ R, R


rot, _ = varimax(load)
rot_df = pd.DataFrame(rot, index=PCA_VARS,
                      columns=[f"PC{i+1} (Varimax)" for i in range(n_kaiser)])
rot_df.to_csv(os.path.join(TAB_DIR, "04_pca_varimax.csv"))
print("\n### Varimax-rotated loadings")
print(rot_df.to_string(float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------
# PCA on the PREDICTORS ONLY (target excluded) — this is the version that can
# legitimately be used for prediction, since including the target would leak it.
# ---------------------------------------------------------------
print("\n" + "=" * 90)
print("PCA ON PREDICTORS ONLY (8 morphological traits, target excluded)")
print("=" * 90)
Zp = StandardScaler().fit_transform(df[MORPH_FEATURES].values)
pca_pred = PCA(n_components=len(MORPH_FEATURES), random_state=SEED).fit(Zp)
eig_p, evr_p = pca_pred.explained_variance_, pca_pred.explained_variance_ratio_
eigp_df = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(len(eig_p))],
    "Eigenvalue": eig_p, "% of variance": 100 * evr_p,
    "Cumulative %": 100 * np.cumsum(evr_p)})
eigp_df.to_csv(os.path.join(TAB_DIR, "04_pca_predictors_eigenvalues.csv"), index=False)
print(eigp_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
n_kaiser_p = int((eig_p > 1).sum())
print(f"\nKaiser criterion retains {n_kaiser_p} components "
      f"({100*evr_p[:n_kaiser_p].sum():.2f}% of predictor variance).")

loadp = pca_pred.components_[:n_kaiser_p].T * np.sqrt(eig_p[:n_kaiser_p])
loadp_df = pd.DataFrame(loadp, index=MORPH_FEATURES,
                        columns=[f"PC{i+1}" for i in range(n_kaiser_p)])
loadp_df.to_csv(os.path.join(TAB_DIR, "04_pca_predictors_loadings.csv"))
print("\n### Predictor-only loadings")
print(loadp_df.to_string(float_format=lambda v: f"{v:.4f}"))

# Correlation of each PC score with the target
scores = pca_pred.transform(Zp)
pc_corr = pd.DataFrame({
    "Component": [f"PC{i+1}" for i in range(scores.shape[1])],
    "Pearson r with fresh plant weight": [float(np.corrcoef(scores[:, i], y)[0, 1])
                                          for i in range(scores.shape[1])],
    "% of predictor variance": 100 * evr_p,
})
pc_corr.to_csv(os.path.join(TAB_DIR, "04_pca_score_target_correlation.csv"), index=False)
print("\n### Correlation of each PC score with the target")
print(pc_corr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------
# (c) Multicollinearity diagnostics
# ---------------------------------------------------------------
print("\n" + "=" * 90)
print("MULTICOLLINEARITY DIAGNOSTICS (the practical justification for PCA)")
print("=" * 90)
cond_number = float(np.sqrt(eig_p.max() / eig_p.min()))
print(f"Condition number of the standardised predictor matrix = {cond_number:.2f}")

vif_rows = []
from sklearn.linear_model import LinearRegression
for i, f in enumerate(MORPH_FEATURES):
    others = [j for j in range(len(MORPH_FEATURES)) if j != i]
    r2 = LinearRegression().fit(Zp[:, others], Zp[:, i]).score(Zp[:, others], Zp[:, i])
    vif_rows.append({"Feature": f, "R2 on other predictors": float(r2),
                     "VIF": float(1.0 / max(1e-12, 1.0 - r2))})
vif = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
vif.to_csv(os.path.join(TAB_DIR, "04_pca_vif.csv"), index=False)
print(vif.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------
# (b) PC scores AS PREDICTORS — cross-validated, PCA fitted inside each fold
# ---------------------------------------------------------------
print("\n" + "=" * 90)
print("DOES PCA HELP PREDICTION? PC scores used as model inputs")
print("(PCA is fitted inside each training fold only — no leakage)")
print("=" * 90)

DESIGNS = {
    "A. Random 5-fold": (KFold(n_splits=5, shuffle=True, random_state=SEED), None),
    "B. GroupKFold (genotype x year cell)": (GroupKFold(n_splits=5), groups_cell),
}

models = build_models()
rows = []
for dname, (splitter, grp) in DESIGNS.items():
    # Baseline: raw morphological traits
    for m_name, reg in models.items():
        pipe = make_pipeline(reg, MORPH_FEATURES, genotype_categorical=True)
        out = cross_validate(pipe, df[MORPH_FEATURES], y, cv=splitter, groups=grp,
                             scoring=SCORING, n_jobs=-1, error_score="raise")
        rows.append({"Design": dname, "Input representation": "Raw morphological traits (8)",
                     "Model": m_name,
                     "R2_mean": float(np.mean(out["test_r2"])), "R2_sd": float(np.std(out["test_r2"])),
                     "MAE_mean": float(-np.mean(out["test_mae"])),
                     "MAPE_mean": float(-np.mean(out["test_mape"]))})

    # PC representations
    for k in [2, 3, 4, len(MORPH_FEATURES)]:
        label = (f"PCA scores, first {k} PCs" if k < len(MORPH_FEATURES)
                 else f"PCA scores, all {k} PCs (rotation only)")
        for m_name, reg in models.items():
            from sklearn.base import clone
            pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=k, random_state=SEED)),
                ("model", clone(reg)),
            ])
            out = cross_validate(pipe, df[MORPH_FEATURES], y, cv=splitter, groups=grp,
                                 scoring=SCORING, n_jobs=-1, error_score="raise")
            rows.append({"Design": dname, "Input representation": label, "Model": m_name,
                         "R2_mean": float(np.mean(out["test_r2"])), "R2_sd": float(np.std(out["test_r2"])),
                         "MAE_mean": float(-np.mean(out["test_mae"])),
                         "MAPE_mean": float(-np.mean(out["test_mape"]))})

bench = pd.DataFrame(rows)
bench.to_csv(os.path.join(TAB_DIR, "04_pca_ml_benchmark.csv"), index=False)

piv = bench.pivot_table(index=["Design", "Model"], columns="Input representation",
                        values="R2_mean", aggfunc="first")
print("\n### R2 by input representation")
print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
piv.to_csv(os.path.join(TAB_DIR, "04_pca_ml_pivot.csv"))

print("\n### Best model per (design, representation)")
best = bench.loc[bench.groupby(["Design", "Input representation"])["R2_mean"].idxmax()]
print(best[["Design", "Input representation", "Model", "R2_mean", "R2_sd", "MAE_mean", "MAPE_mean"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
best.to_csv(os.path.join(TAB_DIR, "04_pca_ml_best.csv"), index=False)

summary = {
    "pca_variables": PCA_VARS,
    "n_observations": int(len(df)),
    "descriptive_pca_including_target": {
        "n_components_kaiser": n_kaiser,
        "eigenvalues": [float(v) for v in eig],
        "pct_variance": [float(v) for v in 100 * evr],
        "cumulative_pct_first_two": float(100 * evr[:2].sum()),
        "PC1_pct": float(100 * evr[0]), "PC2_pct": float(100 * evr[1]),
        "communalities": {k: float(v) for k, v in zip(PCA_VARS, comm)},
        "varimax_loadings": {v: {f"PC{i+1}": float(rot[j, i]) for i in range(n_kaiser)}
                             for j, v in enumerate(PCA_VARS)},
    },
    "predictor_only_pca": {
        "n_components_kaiser": n_kaiser_p,
        "eigenvalues": [float(v) for v in eig_p],
        "pct_variance": [float(v) for v in 100 * evr_p],
        "cumulative_pct_first_two": float(100 * evr_p[:2].sum()),
        "condition_number": cond_number,
        "max_VIF": float(vif["VIF"].max()),
        "max_VIF_feature": str(vif.iloc[0]["Feature"]),
    },
}
with open(os.path.join(TAB_DIR, "04_pca_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\n" + json.dumps(summary["descriptive_pca_including_target"], indent=2)[:2000])
print("\nSaved to", TAB_DIR)
