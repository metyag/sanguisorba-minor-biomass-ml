# -*- coding: utf-8 -*-
"""
18 — The wide sweep: every estimator x every data view x every target transform.

Two questions are asked here.

(a) DATA INTEGRITY.  Before adding any more modelling machinery it is worth
    establishing what the 620 plant records actually are.  Two structural
    features are measured and reported: whether the nine first records of each
    genotype x year cell are ordered rather than independently sampled, and
    whether the tenth record behaves like the other nine.  Both are reported as
    counts, without any assumption about their cause.

(b) MODEL SPACE.  The sweep adds three things the earlier scripts did not have:
    the log-log parameterisation that a power law actually implies (logarithms
    of the *predictors* as well as of the response), four further target
    transforms including a fitted Yeo-Johnson and Box-Cox, and several estimator
    families that were missing (CatBoost, monotone-constrained gradient
    boosting, generalised additive models, robust and quantile regression).

The sweep is scored under grouped five-fold over cells, which is cheap; the
shortlist is then re-scored under leave-one-cell-out, leave-one-genotype-out and
leave-one-year-out.  Nothing here is a headline number: the maximum of a search
this wide is optimistically biased by construction, and script 21 measures that
bias by putting the whole search inside the validation loop.

Outputs (results/tables/):
  18_integrity.csv, 18_integrity.json, 18_sweep_cell.csv,
  18_sweep_shortlist.csv, 18_summary.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arsenal import (GroupKFoldArray, GroupOut, VIEWS, build_frames, build_pool,
                     evaluate, family_of, view_columns)
from common_setup import GENOTYPE, MORPH_FEATURES, TAB_DIR, TARGET_EN, YEAR

t_start = time.time()
print("=" * 100)
print("18 — WIDE SWEEP OVER THE FULL ARSENAL")
print("=" * 100, flush=True)

# ===========================================================================
# (a) DATA INTEGRITY
# ===========================================================================
PLANT, CELL, GENO = build_frames(exclude_last_plant=False)
PLANT9, CELL9, GENO9 = build_frames(exclude_last_plant=True)

ALL_TRAITS = MORPH_FEATURES + [TARGET_EN]
PLANT = PLANT.copy()
PLANT["_pos"] = PLANT.groupby([YEAR, GENOTYPE]).cumcount() + 1
n_cells = PLANT.groupby([YEAR, GENOTYPE]).ngroups

rows = []
for t in ALL_TRAITS:
    first9 = PLANT[PLANT["_pos"] <= 9]
    mono9 = first9.groupby([YEAR, GENOTYPE])[t].apply(
        lambda s: bool(np.all(np.diff(s.values) >= 0))).sum()
    arith9 = first9.groupby([YEAR, GENOTYPE])[t].apply(
        lambda s: bool(len(set(np.round(np.diff(s.values), 6))) == 1)).sum()
    z, ratio = [], []
    for _, sub in PLANT.groupby([YEAR, GENOTYPE]):
        v = sub.sort_values("_pos")[t].values
        sd = v[:9].std(ddof=1)
        if sd > 0:
            z.append((v[9] - v[:9].mean()) / sd)
        if v[8] > 0:
            ratio.append(v[9] / v[8])
    rows.append({
        "Trait": t,
        "Cells": int(n_cells),
        "Records 1-9 monotone": int(mono9),
        "Records 1-9 arithmetic": int(arith9),
        "Record 10 median z": float(np.median(z)) if z else np.nan,
        "Record 10 |z|>3 count": int(np.sum(np.abs(z) > 3)) if z else 0,
        "Record 10 / record 9 median ratio": float(np.median(ratio)) if ratio else np.nan,
    })
integ = pd.DataFrame(rows)
integ.to_csv(os.path.join(TAB_DIR, "18_integrity.csv"), index=False)

print("\n(a) DATA INTEGRITY — structure of the ten records within each cell")
print("-" * 100)
print(integ.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# probability that a genuinely random sample of nine is monotone
p_mono = 1.0 / np.math.factorial(9) if hasattr(np, "math") else 1.0 / 362880
integ_json = {
    "n_cells": int(n_cells),
    "p_monotone_by_chance": float(1.0 / 362880),
    "expected_monotone_cells_by_chance": float(n_cells / 362880),
    "traits_monotone_in_all_cells": [r["Trait"] for r in rows
                                     if r["Records 1-9 monotone"] == n_cells],
    "mean_shift_in_cell_mean_from_record10_pct": {
        t: float(np.mean(100 * (CELL[t] - CELL9[t]) / CELL9[t])) for t in ALL_TRAITS},
}
with open(os.path.join(TAB_DIR, "18_integrity.json"), "w", encoding="utf-8") as f:
    json.dump(integ_json, f, indent=2)

print(f"\n  A genuinely independent sample of nine plants is monotone in every trait")
print(f"  with probability 1/9! = {1/362880:.2e}; over {n_cells} cells that predicts")
print(f"  {n_cells/362880:.5f} such cells by chance.")
print(f"  Effect of record 10 on the cell mean, per cent:")
for t in ALL_TRAITS:
    print(f"    {t:32} {integ_json['mean_shift_in_cell_mean_from_record10_pct'][t]:6.2f}")

# ===========================================================================
# (b) THE SWEEP
# ===========================================================================
yc = CELL[TARGET_EN].values.astype(float)
g_cell = (CELL[YEAR].astype(str) + "_" + CELL[GENOTYPE].astype(str)).values
g_geno = CELL[GENOTYPE].astype(str).values
g_year = CELL[YEAR].astype(str).values

POOL = build_pool("cell")
TARGET_LIST = ["raw", "log", "sqrt", "yeo", "boxcox"]
CONFIGS = [(m, v, t) for m in POOL for v in VIEWS for t in TARGET_LIST]

print("\n" + "=" * 100)
print("(b) SWEEP — grouped five-fold over cells")
print("=" * 100)
print(f"  estimators      : {len(POOL)}")
print(f"  data views      : {len(VIEWS)}  ({', '.join(VIEWS)})")
print(f"  target transforms: {len(TARGET_LIST)}  ({', '.join(TARGET_LIST)})")
print(f"  configurations  : {len(CONFIGS)}", flush=True)

inner = GroupKFoldArray(g_cell, n_splits=5)


def one(cfg):
    m, v, t = cfg
    try:
        met, _ = evaluate(POOL[m], CELL, view_columns(v), t, yc, inner)
        return {"Model": m, "View": v, "Target": t, "Family": family_of(m),
                "Status": "ok", **met}
    except Exception as e:
        return {"Model": m, "View": v, "Target": t, "Family": family_of(m),
                "Status": f"FAILED: {type(e).__name__}"}


t0 = time.time()
res = Parallel(n_jobs=-1, verbose=0, batch_size=8)(delayed(one)(c) for c in CONFIGS)
sweep = pd.DataFrame(res)
sweep.to_csv(os.path.join(TAB_DIR, "18_sweep_cell.csv"), index=False)
ok = sweep[sweep.Status == "ok"].copy()
print(f"\n  completed {len(ok)}/{len(CONFIGS)} configurations in {time.time()-t0:.0f} s", flush=True)

print("\nTOP 25 CONFIGURATIONS (grouped five-fold over cells)")
print("-" * 100)
top = ok.sort_values("pooled_R2", ascending=False).head(25)
print(top[["Model", "View", "Target", "pooled_R2", "pooled_MAE", "pooled_MAPE",
           "spearman"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nBEST BY FAMILY")
print("-" * 100)
fam = (ok.sort_values("pooled_R2", ascending=False)
         .groupby("Family").head(1)
         .sort_values("pooled_R2", ascending=False))
print(fam[["Family", "Model", "View", "Target", "pooled_R2", "pooled_MAE"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
fam.to_csv(os.path.join(TAB_DIR, "18_best_by_family.csv"), index=False)

# The manuscript quotes how far the last family trails the first. That is a
# difference of two numbers in the table above rather than a number in it, so
# the document verifier cannot trace it unless it is written out here.
pd.DataFrame([
    {"Quantity": f"{fam.iloc[0].Family} minus {fam.iloc[-1].Family}, grouped five-fold",
     "Value": float(fam.iloc[0].pooled_R2 - fam.iloc[-1].pooled_R2)},
    {"Quantity": f"{fam.iloc[0].Family} minus {fam.iloc[1].Family} (leading pair)",
     "Value": float(fam.iloc[0].pooled_R2 - fam.iloc[1].pooled_R2)},
]).to_csv(os.path.join(TAB_DIR, "18_family_gaps.csv"), index=False)

print("\nBEST BY DATA VIEW  (does the log-log parameterisation help?)")
print("-" * 100)
byview = (ok.sort_values("pooled_R2", ascending=False)
            .groupby("View").head(1).sort_values("pooled_R2", ascending=False))
print(byview[["View", "Model", "Target", "pooled_R2", "pooled_MAE"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nBEST BY TARGET TRANSFORM")
print("-" * 100)
bytgt = (ok.sort_values("pooled_R2", ascending=False)
           .groupby("Target").head(1).sort_values("pooled_R2", ascending=False))
print(bytgt[["Target", "Model", "View", "pooled_R2", "pooled_MAE"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ===========================================================================
# (c) SHORTLIST UNDER THE EXPENSIVE DESIGNS
# ===========================================================================
SHORT = ok.sort_values("pooled_R2", ascending=False).head(15)
DESIGNS = {
    "LOCO leave-one-cell-out (62)": GroupOut(g_cell),
    "LOGO leave-one-genotype-out (31)": GroupOut(g_geno),
    "LOYO leave-one-year-out (2)": GroupOut(g_year),
}

print("\n" + "=" * 100)
print("(c) SHORTLIST — top 15 re-scored under the expensive designs")
print("=" * 100, flush=True)


def one_design(args):
    cfg, dname = args
    m, v, t = cfg
    try:
        met, _ = evaluate(POOL[m], CELL, view_columns(v), t, yc, DESIGNS[dname])
        return {"Design": dname, "Model": m, "View": v, "Target": t,
                "Family": family_of(m), "Status": "ok", **met}
    except Exception as e:
        return {"Design": dname, "Model": m, "View": v, "Target": t,
                "Family": family_of(m), "Status": f"FAILED: {type(e).__name__}"}


jobs = [((r.Model, r.View, r.Target), d)
        for r in SHORT.itertuples() for d in DESIGNS]
t0 = time.time()
res2 = Parallel(n_jobs=-1, batch_size=4)(delayed(one_design)(j) for j in jobs)
sl = pd.DataFrame(res2)
sl.to_csv(os.path.join(TAB_DIR, "18_sweep_shortlist.csv"), index=False)
print(f"  done in {time.time()-t0:.0f} s\n")

for d in DESIGNS:
    sub = sl[(sl.Design == d) & (sl.Status == "ok")].sort_values("pooled_R2",
                                                                ascending=False)
    print(f"--- {d} ---")
    print(sub[["Model", "View", "Target", "pooled_R2", "pooled_MAE", "pooled_MAPE",
               "spearman"]].head(8).to_string(index=False,
                                              float_format=lambda v: f"{v:.4f}"))
    print()

# ===========================================================================
# (d) THE SAME SWEEP WITHOUT RECORD 10, AS A SENSITIVITY ANALYSIS
# ===========================================================================
print("=" * 100)
print("(d) SENSITIVITY — the top 15 configurations refitted on records 1-9 only")
print("=" * 100, flush=True)
yc9 = CELL9[TARGET_EN].values.astype(float)
g_cell9 = (CELL9[YEAR].astype(str) + "_" + CELL9[GENOTYPE].astype(str)).values
loco9 = GroupOut(g_cell9)


def one9(cfg):
    m, v, t = cfg
    try:
        met, _ = evaluate(POOL[m], CELL9, view_columns(v), t, yc9, loco9)
        return {"Model": m, "View": v, "Target": t, "Status": "ok", **met}
    except Exception as e:
        return {"Model": m, "View": v, "Target": t, "Status": f"FAILED: {type(e).__name__}"}


res3 = Parallel(n_jobs=-1, batch_size=4)(
    delayed(one9)((r.Model, r.View, r.Target)) for r in SHORT.itertuples())
s9 = pd.DataFrame(res3)
loco_all = sl[(sl.Design == "LOCO leave-one-cell-out (62)") & (sl.Status == "ok")]
cmp = loco_all.merge(s9, on=["Model", "View", "Target"], suffixes=("_all10", "_first9"))
cmp["delta_R2"] = cmp["pooled_R2_first9"] - cmp["pooled_R2_all10"]
cmp.to_csv(os.path.join(TAB_DIR, "18_sensitivity_record10.csv"), index=False)
print(cmp[["Model", "View", "Target", "pooled_R2_all10", "pooled_R2_first9",
           "delta_R2"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(f"\n  mean change in R2 from dropping record 10: {cmp['delta_R2'].mean():+.4f}")

best = ok.loc[ok["pooled_R2"].idxmax()]
summary = {
    "n_configurations": int(len(CONFIGS)),
    "n_completed": int(len(ok)),
    "n_estimators": int(len(POOL)),
    "n_views": int(len(VIEWS)),
    "n_targets": int(len(TARGET_LIST)),
    "best_grouped5fold": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                          for k, v in best.items()},
    "best_by_view": {r["View"]: float(r["pooled_R2"]) for _, r in byview.iterrows()},
    "best_by_family": {r["Family"]: float(r["pooled_R2"]) for _, r in fam.iterrows()},
    "best_by_target": {r["Target"]: float(r["pooled_R2"]) for _, r in bytgt.iterrows()},
    "mean_delta_R2_dropping_record10": float(cmp["delta_R2"].mean()),
    "elapsed_s": float(time.time() - t_start),
}
for d in DESIGNS:
    sub = sl[(sl.Design == d) & (sl.Status == "ok")]
    if len(sub):
        b = sub.loc[sub["pooled_R2"].idxmax()]
        summary[f"best_{d.split()[0]}"] = {
            k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
            for k, v in b.items()}
with open(os.path.join(TAB_DIR, "18_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print(f"\nTotal elapsed {time.time()-t_start:.0f} s")
print("Saved to", TAB_DIR)
