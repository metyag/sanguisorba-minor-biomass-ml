# -*- coding: utf-8 -*-
"""
03 — Feature-contribution analysis repeated under BOTH validation designs.

Reviewer 1 asked for a model built on genuine morphological traits only, and both
reviewers asked whether the forward-selection result is an artefact of the
validation strategy. This script therefore repeats every feature-contribution
analysis under:
   A. random 5-fold CV      (the original, leakage-prone design)
   B. GroupKFold by genotype x year cell (leakage-free design)
and for the FULL and MORPH-ONLY feature sets.

Outputs (results/tables/):
  03_univariate_r2.csv
  03_forward_selection.csv
  03_rf_importance.csv
  03_permutation_importance.csv
  03_oof_predictions_<tag>.csv
  03_diagnostics.json
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, cross_val_predict, GroupKFold, KFold
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (load_data, make_pipeline, make_preprocessor, SCORING, SEED, TAB_DIR,
                          FULL_FEATURES, MORPH_FEATURES, GENOTYPE, YEAR, TARGET_EN, mape)

pd.set_option("display.width", 240)

df, y, groups_cell, groups_geno, groups_year = load_data()

DESIGNS = {
    "A_random":    (KFold(n_splits=5, shuffle=True, random_state=SEED), None),
    "B_groupcell": (GroupKFold(n_splits=5), groups_cell),
}
SETS = {"FULL": FULL_FEATURES, "MORPH": MORPH_FEATURES}

BEST = ExtraTreesRegressor(n_estimators=800, random_state=SEED, n_jobs=-1)

print("=" * 90)
print("03 — FEATURE-CONTRIBUTION ANALYSIS UNDER RANDOM vs GROUPED CV")
print("=" * 90)

# ---------------------------------------------------------------
# 1) Univariate R2
# ---------------------------------------------------------------
print("\n### 1) Univariate R2 (each feature used alone, ExtraTrees)")
uni_rows = []
for dname, (splitter, grp) in DESIGNS.items():
    for feat in FULL_FEATURES:
        pipe = make_pipeline(BEST, [feat], genotype_categorical=True)
        out = cross_validate(pipe, df[[feat]], y, cv=splitter, groups=grp,
                             scoring=SCORING, n_jobs=-1, error_score="raise")
        uni_rows.append({
            "Design": dname, "Feature": feat,
            "R2_mean": float(np.mean(out["test_r2"])), "R2_sd": float(np.std(out["test_r2"])),
            "MAE": float(-np.mean(out["test_mae"])), "MAPE": float(-np.mean(out["test_mape"])),
        })
uni = pd.DataFrame(uni_rows)
uni.to_csv(os.path.join(TAB_DIR, "03_univariate_r2.csv"), index=False)
for dname in DESIGNS:
    sub = uni[uni.Design == dname].sort_values("R2_mean", ascending=False)
    print(f"\n-- {dname} --")
    print(sub[["Feature", "R2_mean", "R2_sd", "MAE", "MAPE"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------
# 2) Forward feature selection
# ---------------------------------------------------------------
print("\n\n### 2) Forward feature selection")
fwd_all = []
for dname, (splitter, grp) in DESIGNS.items():
    for sname, feats in SETS.items():
        selected, remaining, rows = [], list(feats), []
        while remaining:
            best_r2, best_feat, best_sd = -np.inf, None, 0.0
            for f in remaining:
                cand = selected + [f]
                pipe = make_pipeline(BEST, cand, genotype_categorical=True)
                out = cross_validate(pipe, df[cand], y, cv=splitter, groups=grp,
                                     scoring={"r2": "r2"}, n_jobs=-1, error_score="raise")
                m, s = float(np.mean(out["test_r2"])), float(np.std(out["test_r2"]))
                if m > best_r2:
                    best_r2, best_feat, best_sd = m, f, s
            selected.append(best_feat)
            remaining.remove(best_feat)
            rows.append({"Design": dname, "Feature set": sname, "Step": len(selected),
                         "Added feature": best_feat, "Cumulative R2_mean": best_r2,
                         "Cumulative R2_sd": best_sd, "Cumulative set": ", ".join(selected)})
        sub = pd.DataFrame(rows)
        sub["Delta R2"] = sub["Cumulative R2_mean"].diff()
        mx = sub["Cumulative R2_mean"].max()
        thr = 0.95 * mx
        mink = int(np.argmax(sub["Cumulative R2_mean"].values >= thr)) + 1
        sub["Max R2"] = mx
        sub["95% threshold"] = thr
        sub["Minimum feature count"] = mink
        fwd_all.append(sub)
        print(f"\n-- {dname} / {sname} --  max R2 = {mx:.4f}, 95% threshold = {thr:.4f}, "
              f"minimum features = {mink}")
        print(sub[["Step", "Added feature", "Cumulative R2_mean", "Cumulative R2_sd", "Delta R2"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
fwd = pd.concat(fwd_all, ignore_index=True)
fwd.to_csv(os.path.join(TAB_DIR, "03_forward_selection.csv"), index=False)

# ---------------------------------------------------------------
# 3) Random Forest impurity importance (fit on full data)
# ---------------------------------------------------------------
print("\n\n### 3) Random Forest impurity-based importance")
rf_rows = []
for sname, feats in SETS.items():
    pre = make_preprocessor(feats, genotype_categorical=True)
    Xp = pre.fit_transform(df[feats])
    names = []
    for tname, _, cols in pre.transformers_:
        if tname == "num":
            names += list(cols)
        elif tname == "cat":
            ohe = pre.named_transformers_["cat"].named_steps["onehot"]
            names += list(ohe.get_feature_names_out(cols))
    rf = RandomForestRegressor(n_estimators=600, random_state=SEED, n_jobs=-1).fit(Xp, y)
    agg = {}
    for nm, v in zip(names, rf.feature_importances_):
        orig = nm
        if nm.startswith(GENOTYPE + "_"):
            orig = GENOTYPE
        agg[orig] = agg.get(orig, 0.0) + float(v)
    for k, v in agg.items():
        rf_rows.append({"Feature set": sname, "Feature": k, "RF importance": v})
rf_imp = pd.DataFrame(rf_rows)
rf_imp.to_csv(os.path.join(TAB_DIR, "03_rf_importance.csv"), index=False)
for sname in SETS:
    sub = rf_imp[rf_imp["Feature set"] == sname].sort_values("RF importance", ascending=False)
    print(f"\n-- {sname} --")
    print(sub[["Feature", "RF importance"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------
# 4) Permutation importance — computed on HELD-OUT folds of the grouped design
#    (the original computed it on the training data itself, which inflates it)
# ---------------------------------------------------------------
print("\n\n### 4) Permutation importance on held-out folds")
perm_rows = []
for dname, (splitter, grp) in DESIGNS.items():
    for sname, feats in SETS.items():
        X = df[feats]
        fold_imp = {f: [] for f in feats}
        splits = list(splitter.split(X, y, groups=grp))
        for tr, te in splits:
            pipe = make_pipeline(BEST, feats, genotype_categorical=True)
            pipe.fit(X.iloc[tr], y[tr])
            r = permutation_importance(pipe, X.iloc[te], y[te], n_repeats=30,
                                       random_state=SEED, n_jobs=-1, scoring="r2")
            for j, f in enumerate(feats):
                fold_imp[f].append(float(r.importances_mean[j]))
        for f in feats:
            perm_rows.append({"Design": dname, "Feature set": sname, "Feature": f,
                              "Permutation dR2_mean": float(np.mean(fold_imp[f])),
                              "Permutation dR2_sd": float(np.std(fold_imp[f])),
                              "n_folds": len(splits)})
perm = pd.DataFrame(perm_rows)
perm.to_csv(os.path.join(TAB_DIR, "03_permutation_importance.csv"), index=False)
for dname in DESIGNS:
    for sname in SETS:
        sub = perm[(perm.Design == dname) & (perm["Feature set"] == sname)] \
            .sort_values("Permutation dR2_mean", ascending=False)
        print(f"\n-- {dname} / {sname} --")
        print(sub[["Feature", "Permutation dR2_mean", "Permutation dR2_sd"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------
# 5) Out-of-fold predictions + agreement diagnostics
# ---------------------------------------------------------------
print("\n\n### 5) Out-of-fold predictions and Bland-Altman agreement")
diag = {}
for dname, (splitter, grp) in DESIGNS.items():
    for sname, feats in SETS.items():
        tag = f"{dname}_{sname}"
        pipe = make_pipeline(BEST, feats, genotype_categorical=True)
        oof = cross_val_predict(pipe, df[feats], y, cv=splitter, groups=grp, n_jobs=-1)
        res = y - oof
        d = {
            "R2": float(r2_score(y, oof)),
            "RMSE": float(np.sqrt(mean_squared_error(y, oof))),
            "MAE": float(mean_absolute_error(y, oof)),
            "MAPE": float(mape(y, oof)),
            "BlandAltman_mean_difference": float(np.mean(res)),
            "BlandAltman_sd_difference": float(np.std(res, ddof=1)),
            "BlandAltman_LoA_lower": float(np.mean(res) - 1.96 * np.std(res, ddof=1)),
            "BlandAltman_LoA_upper": float(np.mean(res) + 1.96 * np.std(res, ddof=1)),
            "target_min": float(y.min()), "target_max": float(y.max()),
        }
        diag[tag] = d
        pd.DataFrame({"actual": y, "predicted_oof": oof, "residual": res,
                      "Year": df[YEAR].values, "Genotype": df[GENOTYPE].values}) \
            .to_csv(os.path.join(TAB_DIR, f"03_oof_predictions_{tag}.csv"), index=False)
        print(f"  {tag:20s} R2={d['R2']:.4f}  RMSE={d['RMSE']:.2f}  MAE={d['MAE']:.2f}  "
              f"MAPE={d['MAPE']:.2f}%  bias={d['BlandAltman_mean_difference']:+.2f}  "
              f"LoA=[{d['BlandAltman_LoA_lower']:.2f}, {d['BlandAltman_LoA_upper']:.2f}]")

# ---------------------------------------------------------------
# 6) Pearson correlation matrix (English labels)
# ---------------------------------------------------------------
corr_cols = FULL_FEATURES + [TARGET_EN]
corr = df[corr_cols].corr(method="pearson")
corr.to_csv(os.path.join(TAB_DIR, "03_correlation_matrix.csv"))
print("\n\n### 6) Pearson correlations with the target")
print(corr[TARGET_EN].sort_values(ascending=False).to_string(float_format=lambda v: f"{v:.4f}"))
print(f"\nCorrelation Year - Plant height = {corr.loc[YEAR, 'Plant height']:.4f}")

with open(os.path.join(TAB_DIR, "03_diagnostics.json"), "w", encoding="utf-8") as f:
    json.dump(diag, f, indent=2)
print("\nSaved to", TAB_DIR)
