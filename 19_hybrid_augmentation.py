# -*- coding: utf-8 -*-
"""
19 — Hybrid strategies: bottom-up aggregation, data augmentation, and blending.

Script 18 searched over estimators, data views and target transforms while
holding the *strategy* fixed: one row per genotype x year cell, fitted directly.
This script varies the strategy itself.

BOTTOM-UP.  A cell mean can also be predicted from below.  Fit the model to the
individual plants of the training cells, which is ten times as many rows, then
predict each plant of the held-out cell and average.  In deployment this is the
natural procedure, because the traits of the ten plants are measured before any
plant is cut; only the weights are unknown.  Two variants are separated: the
mean of the predictions, and the prediction from the mean traits.  They differ
whenever the model is non-linear, and the gap between them measures how much
that curvature costs.

Caveat, stated here because it bears on how the bottom-up result should be
read: script 18 established that the nine first records of each cell are
ordered rather than independently sampled.  A plant-level fit therefore partly
learns that ordering.  The between-cell signal, which is what the held-out
prediction depends on, is unaffected by any within-cell ordering, but the
plant-level residual variance is not a genuine measure of plant-to-plant
variation and is not interpreted as one.

AUGMENTATION.  Six schemes generate additional training rows from the training
cells only: bootstrap resampling of a cell's plants, resampling seven of ten,
leave-one-plant-out jackknife means, mixup, the regression-specific C-mixup
that restricts interpolation to cells of similar yield, and Gaussian noise
scaled by each cell's own standard error.  None of them adds information; all
of them change the effective regularisation, which at n = 62 can matter.

BLENDING.  Equal-weight and non-negative-least-squares combinations of the
strongest candidates, with the blending weights estimated on an inner grouped
cross-validation of the training cells so that they never see a held-out cell.

Outputs (results/tables/): 19_hybrid.csv, 19_blends.csv, 19_summary.json
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

from arsenal import (GroupKFoldArray, GroupOut, build_frames, build_pool,
                     family_of, metrics)
from candidates import Context, run_candidate, spec_label
from common_setup import TAB_DIR

t_start = time.time()
print("=" * 100)
print("19 — HYBRID STRATEGIES: BOTTOM-UP, AUGMENTATION, BLENDING")
print("=" * 100, flush=True)

PLANT, CELL, _ = build_frames()
ctx = Context(PLANT, CELL)
POOL = build_pool("cell")
POOL_PLANT = build_pool("plant")

DESIGNS = {
    "grouped 5-fold": GroupKFoldArray(ctx.g_cell, n_splits=5),
    "LOCO": GroupOut(ctx.g_cell),
    "LOGO": GroupOut(ctx.g_geno),
    "LOYO": GroupOut(ctx.g_year),
}

# Estimators cheap enough to refit 62 times on 610 plant rows.
FAST = {"Linear", "Ridge", "RidgeCV", "Huber", "Huber (eps 1.1)", "Huber (eps 2.0)",
        "Bayesian Ridge", "ARD Regression", "TheilSen", "Quantile (median)", "OMP",
        "LassoCV", "ElasticNetCV", "SVR (linear)", "PLS (2 comp)", "PLS (3 comp)",
        "PLS (4 comp)", "PLS (5 comp)", "PCR (3 comp)", "PCR (4 comp)", "PCR (5 comp)",
        "GP (RBF)", "GP (Matern 1.5)", "GP (Matern 2.5)", "GP (RatQuad)",
        "GP (linear+RBF)", "GAM (splines)", "LightGBM", "LightGBM (monotone)",
        "CatBoost", "Kernel Ridge (poly2)"}

sweep_path = os.path.join(TAB_DIR, "18_sweep_cell.csv")
if not os.path.exists(sweep_path):
    raise SystemExit("run 18_arsenal_sweep.py first")
sw = pd.read_csv(sweep_path)
sw = sw[(sw.Status == "ok") & (sw.Model.isin(FAST))].sort_values("pooled_R2",
                                                                ascending=False)
# one configuration per estimator, so the base set is diverse rather than ten
# near-copies of the same winner
BASE = [(r.Model, r.View, r.Target) for r in sw.groupby("Model").head(1)
        .sort_values("pooled_R2", ascending=False).head(10).itertuples()]
print("\nBase configurations taken from the script-18 sweep:")
for b in BASE:
    print(f"   {b[0]:22} {b[1]:12} {b[2]}")

# ===========================================================================
# 1. STRATEGY COMPARISON — direct vs bottom-up, under every design
# ===========================================================================
print("\n" + "=" * 100)
print("1. DIRECT vs BOTTOM-UP")
print("=" * 100, flush=True)

jobs = []
for (m, v, t) in BASE:
    for kind in ("direct", "bottomup", "bottomup_pm"):
        for d in DESIGNS:
            jobs.append(((kind, m, v, t), d))


def run_one(job):
    spec, d = job
    try:
        pool = POOL_PLANT if spec[0].startswith("bottomup") else POOL
        oof = run_candidate(ctx, spec, pool, DESIGNS[d])
        return {"Strategy": spec[0], "Model": spec[1], "View": spec[2],
                "Target": spec[3], "Design": d, "Family": family_of(spec[1]),
                "Status": "ok", **metrics(ctx.y_cell, oof)}
    except Exception as e:
        return {"Strategy": spec[0], "Model": spec[1], "View": spec[2],
                "Target": spec[3], "Design": d, "Family": family_of(spec[1]),
                "Status": f"FAILED: {type(e).__name__}: {str(e)[:60]}"}


t0 = time.time()
res = Parallel(n_jobs=-1, batch_size=2)(delayed(run_one)(j) for j in jobs)
strat = pd.DataFrame(res)
print(f"  {len(strat)} evaluations in {time.time()-t0:.0f} s\n")

for d in DESIGNS:
    sub = strat[(strat.Design == d) & (strat.Status == "ok")]
    if not len(sub):
        continue
    print(f"--- {d} ---")
    piv = sub.pivot_table(index=["Model", "View", "Target"], columns="Strategy",
                          values="pooled_R2")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print()

# ===========================================================================
# 2. AUGMENTATION
# ===========================================================================
print("=" * 100)
print("2. AUGMENTATION SCHEMES")
print("=" * 100, flush=True)

SCHEMES = ["bootstrap", "subsample7", "jackknife", "mixup", "cmixup", "noise"]
AUG_BASE = BASE[:5]
aug_jobs = [(("aug", m, v, t, s, 20), d)
            for (m, v, t) in AUG_BASE for s in SCHEMES
            for d in ("grouped 5-fold", "LOCO")]


def run_aug(job):
    spec, d = job
    try:
        oof = run_candidate(ctx, spec, POOL, DESIGNS[d])
        return {"Strategy": f"aug:{spec[4]}", "Scheme": spec[4], "Model": spec[1],
                "View": spec[2], "Target": spec[3], "Design": d, "Status": "ok",
                **metrics(ctx.y_cell, oof)}
    except Exception as e:
        return {"Strategy": f"aug:{spec[4]}", "Scheme": spec[4], "Model": spec[1],
                "View": spec[2], "Target": spec[3], "Design": d,
                "Status": f"FAILED: {type(e).__name__}: {str(e)[:60]}"}


t0 = time.time()
res2 = Parallel(n_jobs=-1, batch_size=2)(delayed(run_aug)(j) for j in aug_jobs)
aug = pd.DataFrame(res2)
print(f"  {len(aug)} evaluations in {time.time()-t0:.0f} s\n")

for d in ("grouped 5-fold", "LOCO"):
    sub = aug[(aug.Design == d) & (aug.Status == "ok")]
    if not len(sub):
        continue
    print(f"--- {d} ---")
    piv = sub.pivot_table(index=["Model", "View", "Target"], columns="Scheme",
                          values="pooled_R2")
    base_col = strat[(strat.Design == d) & (strat.Strategy == "direct")] \
        .set_index(["Model", "View", "Target"])["pooled_R2"]
    piv.insert(0, "no augmentation", base_col.reindex(piv.index))
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print()

allrows = pd.concat([strat, aug], ignore_index=True)
allrows.to_csv(os.path.join(TAB_DIR, "19_hybrid.csv"), index=False)

# ===========================================================================
# 3. BLENDS
# ===========================================================================
print("=" * 100)
print("3. BLENDS")
print("=" * 100, flush=True)

loco = strat[(strat.Design == "LOCO") & (strat.Status == "ok")] \
    .sort_values("pooled_R2", ascending=False)
loco_aug = aug[(aug.Design == "LOCO") & (aug.Status == "ok")] \
    .sort_values("pooled_R2", ascending=False)

# members: the best candidate of each distinct family, plus the best bottom-up
members = []
seen_fam = set()
for r in loco.itertuples():
    key = (r.Family, r.Strategy)
    if key in seen_fam:
        continue
    seen_fam.add(key)
    members.append((r.Strategy, r.Model, r.View, r.Target))
    if len(members) >= 6:
        break
print("Blend members:")
for m in members:
    print("   ", spec_label(m))


def run_blend(job):
    spec, d, label = job
    try:
        pool = POOL_PLANT if any(s[0].startswith("bottomup") for s in spec[1]) else POOL
        oof = run_candidate(ctx, spec, pool, DESIGNS[d])
        return {"Blend": label, "Design": d, "Members": len(spec[1]), "Status": "ok",
                **metrics(ctx.y_cell, oof)}
    except Exception as e:
        return {"Blend": label, "Design": d, "Members": len(spec[1]),
                "Status": f"FAILED: {type(e).__name__}: {str(e)[:80]}"}


blend_jobs = []
for k in (2, 3, 4, 5, 6):
    blend_jobs.append((("blend", members[:k]), "LOCO", f"equal weights, top {k}"))
blend_jobs.append((("blendfit", members), "LOCO", "NNLS weights, 6 members"))
blend_jobs.append((("blendfit", members[:3]), "LOCO", "NNLS weights, 3 members"))
for d in ("grouped 5-fold", "LOGO", "LOYO"):
    blend_jobs.append((("blend", members[:3]), d, "equal weights, top 3"))
    blend_jobs.append((("blendfit", members), d, "NNLS weights, 6 members"))

t0 = time.time()
res3 = Parallel(n_jobs=-1, batch_size=1)(delayed(run_blend)(j) for j in blend_jobs)
bl = pd.DataFrame(res3)
bl.to_csv(os.path.join(TAB_DIR, "19_blends.csv"), index=False)
print(f"\n  {len(bl)} blends in {time.time()-t0:.0f} s\n")
print(bl[["Blend", "Design", "Status", "pooled_R2", "pooled_MAE", "pooled_MAPE",
          "spearman"]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ===========================================================================
# SUMMARY
# ===========================================================================
best_rows = []
for src, name in ((strat, "strategy"), (aug, "augmentation")):
    s = src[(src.Design == "LOCO") & (src.Status == "ok")]
    if len(s):
        b = s.loc[s["pooled_R2"].idxmax()]
        best_rows.append({"group": name, **{k: (float(v) if isinstance(v, (int, float, np.floating))
                                               else str(v)) for k, v in b.items()}})
b_bl = bl[(bl.Design == "LOCO") & (bl.Status == "ok")]
if len(b_bl):
    b = b_bl.loc[b_bl["pooled_R2"].idxmax()]
    best_rows.append({"group": "blend", **{k: (float(v) if isinstance(v, (int, float, np.floating))
                                              else str(v)) for k, v in b.items()}})

summary = {"best_per_group_LOCO": best_rows,
           "base_configurations": [list(b) for b in BASE],
           "blend_members": [list(m) for m in members],
           "elapsed_s": float(time.time() - t_start)}
with open(os.path.join(TAB_DIR, "19_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n" + "=" * 100)
print("BEST OF EACH GROUP, leave-one-cell-out")
print("=" * 100)
for r in best_rows:
    lab = r.get("Blend") or f"{r.get('Strategy')} {r.get('Model')} {r.get('View')} {r.get('Target')}"
    print(f"  {r['group']:14} {lab:58} R2={r['pooled_R2']:.4f}  MAE={r['pooled_MAE']:.2f}")
print(f"\nTotal elapsed {time.time()-t_start:.0f} s")
