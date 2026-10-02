# -*- coding: utf-8 -*-
"""
02 — THE CORE REVISION ANALYSIS.

Answers Reviewer 1 (Q1), Reviewer 2 (Q2, Q3) and the Editor's leakage comment by
re-evaluating all 17 models under four validation designs and three feature sets.

Validation designs
  A. Random 5-fold CV at the individual-plant level   [as originally submitted]
  B. GroupKFold by genotype x year cell (62 groups)   [no pseudo-replicate leakage]
  C. Leave-One-Genotype-Out (31 folds)                [generalisation to a NEW genotype]
  D. Leave-One-Year-Out (2 folds)                     [generalisation to a NEW season]

Feature sets
  FULL       : year + genotype + 8 morphological traits
  MORPH-ONLY : the 8 morphological traits (the non-destructive, deployable model)
  YEAR+GEN   : the two identifiers alone (the 99.35% claim under scrutiny)

METRIC CONVENTION
  Under Leave-One-Genotype-Out each held-out fold contains a single genotype, whose
  within-fold variance in fresh plant weight is very small. A per-fold R2 is
  therefore numerically unstable (its denominator nearly vanishes) and can take
  large negative values that say more about the fold than about the model. All
  designs are consequently summarised with POOLED out-of-fold metrics: every
  observation is predicted exactly once while held out, the 620 predictions are
  pooled, and one R2/MSE/MAE/MAPE is computed on the pooled set. Per-fold means
  and standard deviations are also reported for completeness.

Genotype is one-hot encoded (nominal), which is the statistically correct treatment.
A sensitivity run with genotype kept as an integer code (the original treatment)
is included for design A so the two can be compared directly.

Outputs (results/tables/):
  02_validation_matrix_full.csv    — every model x design x feature set
  02_validation_summary.csv        — compact table for the manuscript
  02_optimism_gap.csv              — random-CV optimism per model
  02_genotype_encoding_check.csv   — one-hot vs integer-code sensitivity
  02_headline_numbers.json         — verified numbers for the manuscript text
"""
import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (load_data, build_models, make_pipeline, TAB_DIR, mape,
                          FEATURE_SETS, FULL_FEATURES, cv_schemes, GENOTYPE, YEAR)

pd.set_option("display.width", 240)
pd.set_option("display.max_columns", 60)

df, y, groups_cell, groups_geno, groups_year = load_data()
models = build_models()
schemes = cv_schemes(groups_cell, groups_geno, groups_year)

print("=" * 92)
print("02 — VALIDATION-STRATEGY COMPARISON")
print("=" * 92)
print(f"n = {len(df)} plants | {df[GENOTYPE].nunique()} genotypes | {df[YEAR].nunique()} years "
      f"| {len(np.unique(groups_cell))} genotype x year cells")
print(f"{len(models)} models x {len(schemes)} designs x {len(FEATURE_SETS)} feature sets")
print("Primary metric: POOLED out-of-fold R2 (see module docstring).")


def evaluate(pipe, X, yv, splitter, groups):
    """Fit/predict over the splitter; return pooled and per-fold metrics."""
    oof = np.full(len(yv), np.nan)
    per_fold = []
    n_folds = 0
    for tr, te in splitter.split(X, yv, groups=groups):
        p = clone(pipe)
        p.fit(X.iloc[tr], yv[tr])
        pred = p.predict(X.iloc[te])
        oof[te] = pred
        per_fold.append({
            "r2": r2_score(yv[te], pred),
            "mse": mean_squared_error(yv[te], pred),
            "mae": mean_absolute_error(yv[te], pred),
            "mape": mape(yv[te], pred),
        })
        n_folds += 1
    pf = pd.DataFrame(per_fold)
    covered = ~np.isnan(oof)
    return {
        "pooled_R2": float(r2_score(yv[covered], oof[covered])),
        "pooled_MSE": float(mean_squared_error(yv[covered], oof[covered])),
        "pooled_MAE": float(mean_absolute_error(yv[covered], oof[covered])),
        "pooled_RMSE": float(np.sqrt(mean_squared_error(yv[covered], oof[covered]))),
        "pooled_MAPE": float(mape(yv[covered], oof[covered])),
        "perfold_R2_mean": float(pf["r2"].mean()), "perfold_R2_sd": float(pf["r2"].std(ddof=0)),
        "perfold_MSE_mean": float(pf["mse"].mean()), "perfold_MSE_sd": float(pf["mse"].std(ddof=0)),
        "perfold_MAE_mean": float(pf["mae"].mean()), "perfold_MAE_sd": float(pf["mae"].std(ddof=0)),
        "perfold_MAPE_mean": float(pf["mape"].mean()), "perfold_MAPE_sd": float(pf["mape"].std(ddof=0)),
        "n_folds": n_folds,
    }, oof


t0 = time.time()
records = []
oof_store = {}

for fs_name, feats in FEATURE_SETS.items():
    for sch_name, (splitter, groups, code) in schemes.items():
        note = ""
        if code == "logo" and GENOTYPE in feats:
            note = ("The genotype of the held-out fold never appears in training, so the "
                    "genotype term carries no usable information for that fold.")
        if code == "loyo" and YEAR in feats:
            note = ("The year of the held-out fold never appears in training, so the "
                    "year term carries no usable information for that fold.")

        print("\n" + "-" * 92)
        print(f"FEATURE SET : {fs_name}")
        print(f"DESIGN      : {sch_name}")
        if note:
            print(f"NOTE        : {note}")
        print("-" * 92)

        X = df[feats]
        for m_name, reg in models.items():
            pipe = make_pipeline(reg, feats, genotype_categorical=True)
            try:
                met, oof = evaluate(pipe, X, y, splitter, groups)
                rec = {"Feature set": fs_name, "Validation design": sch_name,
                       "design_code": code, "Model": m_name, "Note": note, "Status": "ok"}
                rec.update(met)
                records.append(rec)
                oof_store[(fs_name, code, m_name)] = oof
                print(f"  {m_name:22s} pooled R2 = {met['pooled_R2']:8.4f} | "
                      f"MAE = {met['pooled_MAE']:7.2f} | MAPE = {met['pooled_MAPE']:6.2f}% | "
                      f"per-fold R2 = {met['perfold_R2_mean']:8.4f} +/- {met['perfold_R2_sd']:.4f}")
            except Exception as e:
                records.append({"Feature set": fs_name, "Validation design": sch_name,
                                "design_code": code, "Model": m_name,
                                "Status": f"FAILED: {e}"})
                print(f"  {m_name:22s} FAILED: {e}")

full = pd.DataFrame(records)
full.to_csv(os.path.join(TAB_DIR, "02_validation_matrix_full.csv"), index=False)
print(f"\nElapsed: {time.time()-t0:.1f} s")

n_failed = int((full["Status"] != "ok").sum())
print(f"Failed model x design combinations: {n_failed}")
if n_failed:
    print(full[full["Status"] != "ok"][["Feature set", "Validation design", "Model", "Status"]]
          .to_string(index=False))

# ---------------------------------------------------------------
# Compact summary: pooled R2 by design
# ---------------------------------------------------------------
ok = full[full["Status"] == "ok"].copy()
piv = ok.pivot_table(index=["Feature set", "Model"], columns="design_code",
                     values="pooled_R2", aggfunc="first")
piv = piv.reindex(columns=["random", "groupcell", "logo", "loyo"])
piv.columns = ["A. Random 5-fold", "B. GroupKFold (cell)",
               "C. Leave-One-Genotype-Out", "D. Leave-One-Year-Out"]
piv.to_csv(os.path.join(TAB_DIR, "02_validation_summary.csv"))
print("\n" + "=" * 92)
print("POOLED OUT-OF-FOLD R2 BY VALIDATION DESIGN")
print("=" * 92)
print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

print("\n" + "=" * 92)
print("OPTIMISM INDUCED BY RANDOM CV  (R2 random  -  R2 grouped)")
print("=" * 92)
gap = piv.copy()
gap["Optimism (A - B)"] = gap["A. Random 5-fold"] - gap["B. GroupKFold (cell)"]
gap.to_csv(os.path.join(TAB_DIR, "02_optimism_gap.csv"))
print(gap[["A. Random 5-fold", "B. GroupKFold (cell)", "Optimism (A - B)"]]
      .sort_values("Optimism (A - B)", ascending=False)
      .to_string(float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------
# Genotype encoding sensitivity (design A, full feature set)
# ---------------------------------------------------------------
print("\n" + "=" * 92)
print("SENSITIVITY: genotype one-hot encoded vs kept as an integer code (design A)")
print("=" * 92)

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

splitter_A, groups_A, _ = _design(schemes, "random")
enc_rows = []
for m_name, reg in models.items():
    row = {"Model": m_name}
    for label, flag in [("one-hot", True), ("integer code", False)]:
        pipe = make_pipeline(reg, FULL_FEATURES, genotype_categorical=flag)
        met, _ = evaluate(pipe, df[FULL_FEATURES], y, splitter_A, groups_A)
        row[f"pooled R2 ({label})"] = met["pooled_R2"]
    row["Difference"] = row["pooled R2 (one-hot)"] - row["pooled R2 (integer code)"]
    enc_rows.append(row)
    print(f"  {m_name:22s} one-hot = {row['pooled R2 (one-hot)']:.4f}   "
          f"integer = {row['pooled R2 (integer code)']:.4f}   diff = {row['Difference']:+.4f}")
pd.DataFrame(enc_rows).to_csv(os.path.join(TAB_DIR, "02_genotype_encoding_check.csv"), index=False)

# ---------------------------------------------------------------
# Headline numbers
# ---------------------------------------------------------------
def pack(r):
    return {"model": r["Model"],
            "pooled_R2": float(r["pooled_R2"]), "pooled_MAE": float(r["pooled_MAE"]),
            "pooled_MSE": float(r["pooled_MSE"]), "pooled_RMSE": float(r["pooled_RMSE"]),
            "pooled_MAPE": float(r["pooled_MAPE"]),
            "perfold_R2_mean": float(r["perfold_R2_mean"]),
            "perfold_R2_sd": float(r["perfold_R2_sd"]),
            "perfold_MAE_mean": float(r["perfold_MAE_mean"]),
            "perfold_MAE_sd": float(r["perfold_MAE_sd"]),
            "perfold_MSE_mean": float(r["perfold_MSE_mean"]),
            "perfold_MSE_sd": float(r["perfold_MSE_sd"]),
            "perfold_MAPE_mean": float(r["perfold_MAPE_mean"]),
            "perfold_MAPE_sd": float(r["perfold_MAPE_sd"]),
            "n_folds": int(r["n_folds"])}

headline, et = {}, {}
for fs_key, short in [("FULL", "full"), ("MORPH-ONLY", "morph"), ("YEAR+GENOTYPE", "yeargen")]:
    for code in ["random", "groupcell", "logo", "loyo"]:
        sub = ok[(ok["Feature set"].str.startswith(fs_key)) & (ok["design_code"] == code)]
        if not sub.empty:
            headline[f"BEST__{short}__{code}"] = pack(sub.loc[sub["pooled_R2"].idxmax()])
            e = sub[sub["Model"] == "ExtraTrees"]
            if not e.empty:
                et[f"ExtraTrees__{short}__{code}"] = pack(e.iloc[0])

out = {"n_obs": int(len(df)), "n_genotypes": int(df[GENOTYPE].nunique()),
       "n_cells": int(len(np.unique(groups_cell))),
       "best_model_per_cell": headline, "extratrees": et}
with open(os.path.join(TAB_DIR, "02_headline_numbers.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 92)
print("HEADLINE NUMBERS")
print("=" * 92)
print(json.dumps(out, indent=2))
print("\nSaved to", TAB_DIR)
