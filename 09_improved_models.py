# -*- coding: utf-8 -*-
"""
09 — Can the leakage-free performance be improved by better MODELLING?

The evaluation stays leakage-free throughout (whole genotype x year cells are held
out). What is varied is the model, not the yardstick. Four levers are tested, each
of which is standard practice in biomass modelling and defensible to a reviewer:

  LEVER 1  Target transformation.
           Fresh weight spans 55.96-820.30 g and its error scales with plant size
           (the residual plot of the submitted model is funnel-shaped). Biomass is
           multiplicative in nature, so log(weight) is the natural scale. Metrics
           are always computed after back-transformation, on the original gram
           scale, so they remain comparable with everything reported so far.

  LEVER 2  Allometric feature engineering.
           Biomass is proportional to volume. Five biologically motivated
           quantities are derived from the measured traits:
             stem volume proxy   = pi * (diameter/2)^2 * height * number of stems
             leaf area proxy     = leaves * leaflets per leaf * leaflet L * leaflet W
             height x stems      = height * number of main stems
             total leaf length   = leaves * leaf length
             height squared      = height^2
           These are functions of the measured traits only; no new information and
           no identity variable enters the model.

  LEVER 3  Season as an environmental covariate.
           Under GroupKFold both seasons appear in every training fold, and the
           permutation importance of year does NOT collapse (0.832 under the grouped
           design, Section 3.5). Year is therefore a genuinely transferable
           environmental covariate in this design, unlike genotype, whose importance
           falls from 0.575 to 0.245. Including year while EXCLUDING genotype is the
           standard multi-environment-trial formulation: environment is a known
           covariate, variety identity is not used as a lookup key. The cost is that
           the model applies only to observed seasons, which is quantified by the
           leave-one-year-out result reported alongside.

  LEVER 4  Hyper-parameter tuning inside a nested grouped design.
           The submitted analysis performed none. Tuning is done with an INNER
           GroupKFold over the training folds only, so no information about the
           held-out cells reaches the search.

Levers 1-3 are swept exhaustively here; lever 4 is applied to the shortlist in
script 10. Nothing is hidden: every combination is written out.

Outputs (results/tables/):
  09_improvement_matrix.csv   every model x feature set x target transform
  09_improvement_best.csv     best model per configuration
  09_improvement_summary.json
"""
import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import load_data, build_models, make_pipeline, TAB_DIR, mape, MORPH_FEATURES, YEAR, cv_schemes

pd.set_option("display.width", 250)

df, y, groups_cell, groups_geno, groups_year = load_data()

# ---------------------------------------------------------------------------
# Engineered allometric features
# ---------------------------------------------------------------------------
ENG_NAMES = ["Stem volume proxy", "Leaf area proxy", "Height x stems",
             "Total leaf length", "Height squared"]


def add_engineered(frame):
    X = frame.copy()
    h = X["Plant height"]
    d = X["Main stem diameter"]
    ns = X["Number of main stems"]
    nl = X["Number of leaves"]
    npl = X["Number of leaflets per leaf"]
    ll = X["Leaflet length"]
    lw = X["Leaflet width"]
    lfl = X["Leaf length"]
    X["Stem volume proxy"] = np.pi * (d / 2.0) ** 2 * h * ns
    X["Leaf area proxy"] = nl * npl * ll * lw
    X["Height x stems"] = h * ns
    X["Total leaf length"] = nl * lfl
    X["Height squared"] = h ** 2
    return X


DF_ENG = add_engineered(df)

FEATURE_SETS = {
    "A. Morphology only (8 raw traits)": MORPH_FEATURES,
    "B. Morphology + allometric features (13)": MORPH_FEATURES + ENG_NAMES,
    "C. Season + morphology (9)": [YEAR] + MORPH_FEATURES,
    "D. Season + morphology + allometric (14)": [YEAR] + MORPH_FEATURES + ENG_NAMES,
}
TARGETS = {"raw": None, "log": "log1p"}

print("=" * 100)
print("09 — MODEL IMPROVEMENT UNDER LEAKAGE-FREE VALIDATION")
print("=" * 100)
print(f"n = {len(df)} plants, {len(np.unique(groups_cell))} genotype x year cells")
print("Evaluation: GroupKFold blocked on the genotype x year cell (5 folds).")
print("Genotype is NEVER used as a predictor in any configuration below.")
print(f"Feature sets: {list(FEATURE_SETS)}")
print(f"Target transforms: {list(TARGETS)}\n")


def wrap_target(reg, transform):
    if transform is None:
        return clone(reg)
    return TransformedTargetRegressor(regressor=clone(reg),
                                      func=np.log1p, inverse_func=np.expm1)


def evaluate(reg, features, transform, splitter, groups, frame):
    """Pooled out-of-fold metrics on the ORIGINAL gram scale."""
    X = frame[features]
    oof = np.full(len(y), np.nan)
    per_fold = []
    for tr, te in splitter.split(X, y, groups=groups):
        pipe = make_pipeline(wrap_target(reg, transform), features,
                             genotype_categorical=False)
        pipe.fit(X.iloc[tr], y[tr])
        pred = pipe.predict(X.iloc[te])
        oof[te] = pred
        per_fold.append(r2_score(y[te], pred))
    m = ~np.isnan(oof)
    return {
        "pooled_R2": float(r2_score(y[m], oof[m])),
        "pooled_RMSE": float(np.sqrt(mean_squared_error(y[m], oof[m]))),
        "pooled_MAE": float(mean_absolute_error(y[m], oof[m])),
        "pooled_MAPE": float(mape(y[m], oof[m])),
        "perfold_R2_mean": float(np.mean(per_fold)),
        "perfold_R2_sd": float(np.std(per_fold)),
        "n_folds": len(per_fold),
    }, oof


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

models = build_models()
records = []
t0 = time.time()

for fs_name, feats in FEATURE_SETS.items():
    for t_name, t_fn in TARGETS.items():
        print("-" * 100)
        print(f"{fs_name}   |   target = {t_name}")
        print("-" * 100)
        for m_name, reg in models.items():
            try:
                met, _ = evaluate(reg, feats, t_fn, splitter_B, groups_B, DF_ENG)
                rec = {"Feature set": fs_name, "Target": t_name, "Model": m_name,
                       "Status": "ok"}
                rec.update(met)
                records.append(rec)
                print(f"  {m_name:<24} R2 = {met['pooled_R2']:7.4f}   "
                      f"RMSE = {met['pooled_RMSE']:7.2f}   MAE = {met['pooled_MAE']:7.2f}   "
                      f"MAPE = {met['pooled_MAPE']:6.2f}")
            except Exception as e:
                records.append({"Feature set": fs_name, "Target": t_name,
                                "Model": m_name, "Status": f"FAILED: {e}"})
                print(f"  {m_name:<24} FAILED: {str(e)[:70]}")

full = pd.DataFrame(records)
full.to_csv(os.path.join(TAB_DIR, "09_improvement_matrix.csv"), index=False)
ok = full[full.Status == "ok"].copy()
print(f"\nElapsed {time.time()-t0:.0f} s;  failures: {int((full.Status!='ok').sum())}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("BEST MODEL PER CONFIGURATION (pooled out-of-fold, GroupKFold by cell)")
print("=" * 100)
best = ok.loc[ok.groupby(["Feature set", "Target"])["pooled_R2"].idxmax()] \
    .sort_values("pooled_R2", ascending=False)
best.to_csv(os.path.join(TAB_DIR, "09_improvement_best.csv"), index=False)
print(f"{'Feature set':<44}{'Target':<8}{'Best model':<20}{'R2':>9}{'RMSE':>9}"
      f"{'MAE':>9}{'MAPE':>8}")
for _, r in best.iterrows():
    print(f"{r['Feature set']:<44}{r['Target']:<8}{r['Model']:<20}"
          f"{r['pooled_R2']:>9.4f}{r['pooled_RMSE']:>9.2f}{r['pooled_MAE']:>9.2f}"
          f"{r['pooled_MAPE']:>8.2f}")

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("EFFECT OF EACH LEVER (best model in each cell)")
print("=" * 100)
piv = best.set_index(["Feature set", "Target"])["pooled_R2"].unstack()
print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

baseline = float(ok[(ok["Feature set"].str.startswith("A.")) &
                    (ok.Target == "raw")]["pooled_R2"].max())
print(f"\nBaseline (morphology only, raw target, as reported in the manuscript): "
      f"R2 = {baseline:.4f}")
for (fs, tg), v in piv.stack().items():
    if fs.startswith("A.") and tg == "raw":
        continue
    print(f"  {fs:<44} target={tg:<5}  R2 = {v:.4f}   gain = {v - baseline:+.4f}")

top = best.iloc[0]
summary = {
    "baseline_R2_morphology_raw": baseline,
    "best_configuration": {
        "feature_set": str(top["Feature set"]), "target": str(top["Target"]),
        "model": str(top["Model"]), "pooled_R2": float(top["pooled_R2"]),
        "pooled_RMSE": float(top["pooled_RMSE"]), "pooled_MAE": float(top["pooled_MAE"]),
        "pooled_MAPE": float(top["pooled_MAPE"]),
    },
    "gain_over_baseline": float(top["pooled_R2"] - baseline),
    "engineered_features": ENG_NAMES,
    "note": ("Genotype is not a predictor in any configuration. Evaluation is GroupKFold "
             "blocked on the genotype x year cell throughout, so no pseudo-replicate of a "
             "validation plant is ever in the training fold."),
}
with open(os.path.join(TAB_DIR, "09_improvement_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\n" + json.dumps(summary, indent=2))
