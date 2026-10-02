# -*- coding: utf-8 -*-
"""
10 — Nested hyper-parameter tuning of the shortlist, and confirmation of the
     winning configuration under the two harder designs.

The submitted analysis performed no tuning at all. Here the tuning is done with an
INNER GroupKFold over the training folds only, so the search never sees a cell that
is in the outer held-out fold. This is the standard nested protocol; it is more
expensive than a single search but it is the only version that does not reintroduce
leakage through the hyper-parameters.

Runs:
  (1) Nested tuning on the best feature set / target found by script 09, evaluated
      under GroupKFold blocked on the genotype x year cell.
  (2) The tuned winner re-evaluated under leave-one-genotype-out and
      leave-one-year-out, so the cost of the season covariate is visible.

Outputs (results/tables/):
  10_tuned_results.csv, 10_final_scenarios.csv, 10_summary.json
"""
import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import loguniform, randint, uniform

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (load_data, make_pipeline, TAB_DIR, mape, SEED,
                          MORPH_FEATURES, YEAR, cv_schemes)
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (RandomForestRegressor, ExtraTreesRegressor,
                              GradientBoostingRegressor, HistGradientBoostingRegressor)
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

df, y, groups_cell, groups_geno, groups_year = load_data()

ENG_NAMES = ["Stem volume proxy", "Leaf area proxy", "Height x stems",
             "Total leaf length", "Height squared"]


def add_engineered(frame):
    X = frame.copy()
    h, d, ns = X["Plant height"], X["Main stem diameter"], X["Number of main stems"]
    nl, npl = X["Number of leaves"], X["Number of leaflets per leaf"]
    ll, lw, lfl = X["Leaflet length"], X["Leaflet width"], X["Leaf length"]
    X["Stem volume proxy"] = np.pi * (d / 2.0) ** 2 * h * ns
    X["Leaf area proxy"] = nl * npl * ll * lw
    X["Height x stems"] = h * ns
    X["Total leaf length"] = nl * lfl
    X["Height squared"] = h ** 2
    return X


DF = add_engineered(df)

# --- read the winner of script 09 -----------------------------------------
BEST09 = pd.read_csv(os.path.join(TAB_DIR, "09_improvement_best.csv"))
BEST09 = BEST09.sort_values("pooled_R2", ascending=False).reset_index(drop=True)
WIN_FS = str(BEST09.loc[0, "Feature set"])
WIN_TGT = str(BEST09.loc[0, "Target"])
FEATURE_SETS = {
    "A. Morphology only (8 raw traits)": MORPH_FEATURES,
    "B. Morphology + allometric features (13)": MORPH_FEATURES + ENG_NAMES,
    "C. Season + morphology (9)": [YEAR] + MORPH_FEATURES,
    "D. Season + morphology + allometric (14)": [YEAR] + MORPH_FEATURES + ENG_NAMES,
}
FEATS = FEATURE_SETS[WIN_FS]
print("=" * 100)
print("10 — NESTED HYPER-PARAMETER TUNING")
print("=" * 100)
print(f"Winning configuration from script 09: {WIN_FS} | target = {WIN_TGT} | "
      f"model = {BEST09.loc[0,'Model']} | R2 = {BEST09.loc[0,'pooled_R2']:.4f}")
print(f"Tuning on that feature set ({len(FEATS)} predictors). Genotype is not among them.\n")

TRANSFORM = (WIN_TGT == "log")

SEARCH = {
    "ExtraTrees": (ExtraTreesRegressor(random_state=SEED, n_jobs=-1), {
        "model__regressor__n_estimators" if TRANSFORM else "model__n_estimators": randint(300, 1200),
        "model__regressor__max_depth" if TRANSFORM else "model__max_depth": [None, 6, 10, 16, 24],
        "model__regressor__min_samples_leaf" if TRANSFORM else "model__min_samples_leaf": randint(1, 20),
        "model__regressor__max_features" if TRANSFORM else "model__max_features": ["sqrt", "log2", 0.5, 0.8, 1.0],
    }),
    "Random Forest": (RandomForestRegressor(random_state=SEED, n_jobs=-1), {
        "model__regressor__n_estimators" if TRANSFORM else "model__n_estimators": randint(300, 1000),
        "model__regressor__max_depth" if TRANSFORM else "model__max_depth": [None, 6, 10, 16],
        "model__regressor__min_samples_leaf" if TRANSFORM else "model__min_samples_leaf": randint(1, 20),
        "model__regressor__max_features" if TRANSFORM else "model__max_features": ["sqrt", 0.5, 0.8, 1.0],
    }),
    "Gradient Boosting": (GradientBoostingRegressor(random_state=SEED), {
        "model__regressor__n_estimators" if TRANSFORM else "model__n_estimators": randint(100, 800),
        "model__regressor__learning_rate" if TRANSFORM else "model__learning_rate": loguniform(0.005, 0.2),
        "model__regressor__max_depth" if TRANSFORM else "model__max_depth": randint(2, 6),
        "model__regressor__subsample" if TRANSFORM else "model__subsample": uniform(0.6, 0.4),
    }),
    "HistGradientBoosting": (HistGradientBoostingRegressor(random_state=SEED), {
        "model__regressor__max_depth" if TRANSFORM else "model__max_depth": [None, 3, 5, 8],
        "model__regressor__learning_rate" if TRANSFORM else "model__learning_rate": loguniform(0.01, 0.2),
        "model__regressor__min_samples_leaf" if TRANSFORM else "model__min_samples_leaf": randint(5, 40),
        "model__regressor__l2_regularization" if TRANSFORM else "model__l2_regularization": loguniform(1e-3, 10),
    }),
    "SVR (RBF)": (SVR(kernel="rbf"), {
        "model__regressor__C" if TRANSFORM else "model__C": loguniform(1, 1000),
        "model__regressor__gamma" if TRANSFORM else "model__gamma": loguniform(1e-3, 1),
        "model__regressor__epsilon" if TRANSFORM else "model__epsilon": loguniform(1e-3, 1),
    }),
    "ElasticNet": (ElasticNet(max_iter=20000), {
        "model__regressor__alpha" if TRANSFORM else "model__alpha": loguniform(1e-4, 10),
        "model__regressor__l1_ratio" if TRANSFORM else "model__l1_ratio": uniform(0, 1),
    }),
    "Huber": (HuberRegressor(max_iter=2000), {
        "model__regressor__epsilon" if TRANSFORM else "model__epsilon": uniform(1.05, 1.5),
        "model__regressor__alpha" if TRANSFORM else "model__alpha": loguniform(1e-5, 1),
    }),
    "Ridge": (Ridge(), {
        "model__regressor__alpha" if TRANSFORM else "model__alpha": loguniform(1e-3, 1000),
    }),
}
if HAS_XGB:
    SEARCH["XGBoost"] = (XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0), {
        "model__regressor__n_estimators" if TRANSFORM else "model__n_estimators": randint(200, 1500),
        "model__regressor__learning_rate" if TRANSFORM else "model__learning_rate": loguniform(0.005, 0.2),
        "model__regressor__max_depth" if TRANSFORM else "model__max_depth": randint(2, 8),
        "model__regressor__subsample" if TRANSFORM else "model__subsample": uniform(0.6, 0.4),
        "model__regressor__colsample_bytree" if TRANSFORM else "model__colsample_bytree": uniform(0.5, 0.5),
        "model__regressor__reg_lambda" if TRANSFORM else "model__reg_lambda": loguniform(1e-2, 50),
    })


def wrap(reg):
    return (TransformedTargetRegressor(regressor=clone(reg), func=np.log1p,
                                       inverse_func=np.expm1)
            if TRANSFORM else clone(reg))


def nested(reg, grid, splitter, groups, n_iter=30, inner_splits=4):
    X = DF[FEATS]
    oof = np.full(len(y), np.nan)
    chosen = []
    for tr, te in splitter.split(X, y, groups=groups):
        base = make_pipeline(wrap(reg), FEATS, genotype_categorical=False)
        g_tr = np.asarray(groups)[tr] if groups is not None else None
        inner = GroupKFold(n_splits=min(inner_splits, len(np.unique(g_tr))))
        srch = RandomizedSearchCV(base, grid, n_iter=n_iter, cv=inner, scoring="r2",
                                  n_jobs=-1, random_state=SEED, refit=True,
                                  error_score="raise")
        srch.fit(X.iloc[tr], y[tr], groups=g_tr)
        oof[te] = srch.best_estimator_.predict(X.iloc[te])
        chosen.append(srch.best_params_)
    m = ~np.isnan(oof)
    return {
        "pooled_R2": float(r2_score(y[m], oof[m])),
        "pooled_RMSE": float(np.sqrt(mean_squared_error(y[m], oof[m]))),
        "pooled_MAE": float(mean_absolute_error(y[m], oof[m])),
        "pooled_MAPE": float(mape(y[m], oof[m])),
    }, chosen, oof


schemes = cv_schemes(groups_cell, groups_geno, groups_year)

def _design(schemes, short_code):
    """Select a design from the output of cv_schemes by its short code.

    The long name is a display string and may change (for example 'x' -> '×');
    the short code ('random', 'groupcell', 'logo', 'loyo') is the identity of the
    design. Looking the design up by name broke silently with a KeyError whenever
    the name changed.
    """
    for _name, value in schemes.items():
        if value[2] == short_code:
            return value
    raise KeyError("short code not found: %s (available: %s)"
                   % (short_code, [v[2] for v in schemes.values()]))

splitter_B, groups_B, _ = _design(schemes, "groupcell")

rows = []
t0 = time.time()
for name, (reg, grid) in SEARCH.items():
    try:
        met, params, _ = nested(reg, grid, splitter_B, groups_B)
        rows.append({"Model": name, "Status": "ok", **met})
        print(f"  {name:<24} tuned R2 = {met['pooled_R2']:7.4f}   "
              f"RMSE = {met['pooled_RMSE']:7.2f}   MAE = {met['pooled_MAE']:7.2f}   "
              f"MAPE = {met['pooled_MAPE']:6.2f}", flush=True)
    except Exception as e:
        rows.append({"Model": name, "Status": f"FAILED: {e}"})
        print(f"  {name:<24} FAILED: {str(e)[:80]}", flush=True)

tuned = pd.DataFrame(rows)
tuned.to_csv(os.path.join(TAB_DIR, "10_tuned_results.csv"), index=False)
print(f"\nElapsed {time.time()-t0:.0f} s")

ok = tuned[tuned.Status == "ok"].sort_values("pooled_R2", ascending=False)
print("\n" + "=" * 100)
print("TUNED RANKING (GroupKFold by genotype x year cell)")
print("=" * 100)
print(ok.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------------------
# The winner under the two harder designs
# ---------------------------------------------------------------------------
win_name = str(ok.iloc[0]["Model"])
win_reg, win_grid = SEARCH[win_name]
print("\n" + "=" * 100)
print(f"CONFIRMING THE WINNER ({win_name}) UNDER THE HARDER DESIGNS")
print("=" * 100)

final = [{"Design": "B. GroupKFold (genotype x year cell)", "Model": win_name,
          **{k: float(ok.iloc[0][k]) for k in
             ["pooled_R2", "pooled_RMSE", "pooled_MAE", "pooled_MAPE"]}}]

for key, label in [("C. Leave-One-Genotype-Out (31 folds)", "C. Leave-one-genotype-out"),
                   ("D. Leave-One-Year-Out (2 folds)", "D. Leave-one-year-out")]:
    spl, grp, _ = schemes[key]
    try:
        met, _, _ = nested(win_reg, win_grid, spl, grp if grp is not None else groups_year,
                           n_iter=15, inner_splits=4)
        final.append({"Design": label, "Model": win_name, **met})
        print(f"  {label:<32} R2 = {met['pooled_R2']:7.4f}   MAE = {met['pooled_MAE']:7.2f}   "
              f"MAPE = {met['pooled_MAPE']:6.2f}")
    except Exception as e:
        print(f"  {label:<32} FAILED: {str(e)[:90]}")

fin = pd.DataFrame(final)
fin.to_csv(os.path.join(TAB_DIR, "10_final_scenarios.csv"), index=False)

summary = {
    "winning_feature_set": WIN_FS,
    "winning_target_transform": WIN_TGT,
    "untuned_best_R2": float(BEST09.loc[0, "pooled_R2"]),
    "untuned_best_model": str(BEST09.loc[0, "Model"]),
    "tuned_best_model": win_name,
    "tuned_best_R2": float(ok.iloc[0]["pooled_R2"]),
    "tuned_best_MAE": float(ok.iloc[0]["pooled_MAE"]),
    "tuned_best_MAPE": float(ok.iloc[0]["pooled_MAPE"]),
    "gain_from_tuning": float(ok.iloc[0]["pooled_R2"] - BEST09.loc[0, "pooled_R2"]),
    "across_designs": final,
}
with open(os.path.join(TAB_DIR, "10_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\n" + json.dumps(summary, indent=2))
