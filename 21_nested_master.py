# -*- coding: utf-8 -*-
"""
21 — Nested selection over the enlarged search space: the honest headline.

Scripts 18-20 searched thousands of configurations and reported the best of
them.  That maximum is optimistically biased by construction, and the size of
the bias grows with the size of the search.  Since the search has now grown by
more than an order of magnitude relative to script 14, the bias has to be
re-measured rather than assumed unchanged.

The measurement is the same as before: the entire selection procedure is placed
inside the validation loop.  For every outer fold the held-out cells are set
aside, an inner grouped cross-validation over the remaining cells alone chooses
the configuration, and that choice is refitted and applied to the held-out
cells.  The pooled out-of-fold statistics then describe what the *procedure*
achieves on cells it has never seen, which is what a reader needs.

HOW THE CANDIDATE SPACE IS DEFINED, AND WHY IT MATTERS.  The candidates are
specified by a structural rule fixed in advance --- every estimator whose
single fit is cheap enough for a 62-fold nested loop, crossed with the data
views and target transforms that correspond to a recognisable statistical model
--- and *not* by taking the winners of the script-18 sweep.  Seeding the nested
loop with the sweep's winners would import exactly the optimism the nesting is
meant to remove, because those winners were chosen using all 62 cells.

Two spaces are evaluated.

  SPACE A (estimator x view x target).  Directly comparable with script 14,
          which used 88 candidates; this one is larger.

  SPACE B (SPACE A restricted to a core estimator set, crossed with the
          modelling strategies from script 19: direct fitting, bottom-up
          aggregation from the plant level, and three augmentation schemes).
          This asks whether the strategy dimension survives honest selection.

Outputs (results/tables/): 21_nested_master.csv, 21_selection_frequency.csv,
  21_oof_*.csv, 21_summary.json
"""
import json
import os
import sys
import time
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arsenal import GroupKFoldArray, GroupOut, build_frames, build_pool, metrics
from candidates import Context, build_candidate, spec_label
from common_setup import GENOTYPE, TAB_DIR, YEAR

t_start = time.time()
print("=" * 100)
print("21 — NESTED SELECTION OVER THE ENLARGED SEARCH SPACE")
print("=" * 100, flush=True)

PLANT, CELL, _ = build_frames()
ctx = Context(PLANT, CELL)
POOL = build_pool("cell")
POOL_PLANT = build_pool("plant")
yc = ctx.y_cell

# ---------------------------------------------------------------------------
# structural rule: estimators cheap enough to refit inside a 62-fold nested loop
# ---------------------------------------------------------------------------
NESTED_ESTIMATORS = [
    "Linear", "Ridge", "RidgeCV", "LassoCV", "ElasticNetCV",
    "Huber", "Huber (eps 1.1)", "Huber (eps 2.0)",
    "Bayesian Ridge", "ARD Regression", "TheilSen", "Quantile (median)", "OMP",
    "PLS (2 comp)", "PLS (3 comp)", "PLS (4 comp)", "PLS (5 comp)",
    "PCR (3 comp)", "PCR (4 comp)", "PCR (5 comp)",
    "Kernel Ridge (poly2)", "SVR (linear)",
    "GP (RBF)", "GP (Matern 1.5)", "GP (RatQuad)", "GP (linear+RBF)",
    "KNN (9, dist)", "LightGBM", "LightGBM (monotone)", "GAM (splines)",
]
NESTED_VIEWS = ["raw8", "log8", "raw8+year", "log8+year", "raw13", "log13"]
NESTED_TARGETS = ["raw", "log", "sqrt"]

SPACE_A = [("direct", m, v, t)
           for m in NESTED_ESTIMATORS for v in NESTED_VIEWS for t in NESTED_TARGETS]

CORE = ["Ridge", "Huber", "Quantile (median)", "Bayesian Ridge", "ARD Regression",
        "PLS (3 comp)"]
# A Gaussian process is excluded from the bottom-up arm only: its cost is cubic
# in the number of training rows, and the bottom-up arm trains on 610 plants
# rather than 62 cells, which would dominate the whole nested run.
SPACE_B = ([("direct", m, v, t) for m in CORE
            for v in ["raw8", "log8"] for t in ["raw", "log"]]
           + [("bottomup", m, v, t) for m in CORE
              for v in ["raw8", "log8"] for t in ["raw", "log"]]
           + [("bottomup_pm", m, "raw8", t) for m in CORE for t in ["raw", "log"]]
           + [("aug", m, v, t, s, 20) for m in CORE
              for v in ["raw8"] for t in ["log"]
              for s in ["bootstrap", "jackknife", "cmixup"]]
           + [("resid", m, r, "raw8", "log") for m in CORE
              for r in ["LightGBM", "Random Forest"]])

print(f"  SPACE A : {len(SPACE_A)} candidates "
      f"({len(NESTED_ESTIMATORS)} estimators x {len(NESTED_VIEWS)} views "
      f"x {len(NESTED_TARGETS)} target transforms)")
print(f"  SPACE B : {len(SPACE_B)} candidates "
      f"({len(CORE)} estimators x direct / bottom-up / three augmentation schemes)",
      flush=True)

DESIGNS = {
    "Leave-one-cell-out (62 outer folds)": GroupOut(ctx.g_cell),
    "Leave-one-genotype-out (31 outer folds)": GroupOut(ctx.g_geno),
    "Leave-one-year-out (2 outer folds)": GroupOut(ctx.g_year),
}


def _pool_for(spec):
    return POOL_PLANT if spec[0].startswith("bottomup") else POOL


def inner_score(spec, tr, n_splits=5):
    """Grouped inner cross-validation over the training cells only."""
    inner = GroupKFoldArray(ctx.g_cell[tr], n_splits=min(n_splits, len(tr)))
    fp = build_candidate(ctx, spec, _pool_for(spec))
    o = np.full(len(tr), np.nan)
    try:
        for a, b in inner.split(np.arange(len(tr))):
            o[b] = fp(tr[a], tr[b])
    except Exception:
        return -np.inf
    m = np.isfinite(o)
    if m.sum() < 3:
        return -np.inf
    try:
        return float(r2_score(yc[tr][m], o[m]))
    except Exception:
        return -np.inf


def one_fold(args):
    """Choose a configuration on the training cells, then predict the held-out."""
    space, tr, te = args
    scores = [inner_score(s, tr) for s in space]
    k = int(np.argmax(scores))
    best = space[k]
    try:
        pred = build_candidate(ctx, best, _pool_for(best))(tr, te)
    except Exception:
        pred = np.full(len(te), float(np.mean(yc[tr])))
    return te, np.asarray(pred).ravel(), best, float(scores[k])


def nested(space, design_name, label):
    spl = DESIGNS[design_name]
    folds = list(spl.split())
    t0 = time.time()
    out = Parallel(n_jobs=-1, batch_size=1, verbose=0)(
        delayed(one_fold)((space, tr, te)) for tr, te in folds)
    oof = np.full(len(yc), np.nan)
    picks = []
    for te, pred, best, s in out:
        oof[te] = pred
        picks.append(best)
    met = metrics(yc, oof)
    print(f"    {label:10} {design_name:42} R2={met['pooled_R2']:7.4f}  "
          f"MAE={met['pooled_MAE']:7.2f}  MAPE={met['pooled_MAPE']:6.2f}  "
          f"rho={met.get('spearman', float('nan')):.3f}   ({time.time()-t0:.0f} s)",
          flush=True)
    return met, picks, oof


results, freq_rows = [], []
oof_store = {}
for space, label in ((SPACE_A, "SPACE A"), (SPACE_B, "SPACE B")):
    print(f"\n--- {label} ---", flush=True)
    for dname in DESIGNS:
        met, picks, oof = nested(space, dname, label)
        results.append({"Space": label, "Design": dname, "n_candidates": len(space), **met})
        oof_store[f"{label}|{dname}"] = oof
        c = Counter(spec_label(p) for p in picks)
        for cfg, n in c.most_common():
            freq_rows.append({"Space": label, "Design": dname, "Configuration": cfg,
                              "Times selected": n, "Share %": 100 * n / len(picks)})
        short = dname.split()[0].lower().replace("-", "")
        pd.DataFrame({"actual": yc, "predicted": oof,
                      YEAR: CELL[YEAR], GENOTYPE: CELL[GENOTYPE]}).to_csv(
            os.path.join(TAB_DIR, f"21_oof_{label[-1]}_{short}.csv"), index=False)

res = pd.DataFrame(results)
res.to_csv(os.path.join(TAB_DIR, "21_nested_master.csv"), index=False)
freq = pd.DataFrame(freq_rows)
freq.to_csv(os.path.join(TAB_DIR, "21_selection_frequency.csv"), index=False)

print("\n" + "=" * 100)
print("NESTED RESULTS (the selection is inside the loop)")
print("=" * 100)
show = [c for c in ["Space", "Design", "n_candidates", "pooled_R2", "pooled_RMSE",
                    "pooled_MAE", "pooled_MAPE", "spearman", "kendall"]
        if c in res.columns]
print(res[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nWHICH CONFIGURATION THE INNER LOOP CHOSE")
print("-" * 100)
for (sp, d), sub in freq.groupby(["Space", "Design"]):
    print(f"\n{sp} / {d}")
    print(sub.sort_values("Times selected", ascending=False)
          [["Configuration", "Times selected", "Share %"]].head(6)
          .to_string(index=False, float_format=lambda v: f"{v:.1f}"))

summary = {"results": results, "selection_frequency": freq_rows,
           "space_A_size": len(SPACE_A), "space_B_size": len(SPACE_B),
           "nested_estimators": NESTED_ESTIMATORS, "views": NESTED_VIEWS,
           "targets": NESTED_TARGETS, "core_estimators": CORE,
           "elapsed_s": float(time.time() - t_start)}
with open(os.path.join(TAB_DIR, "21_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nTotal elapsed {time.time()-t_start:.0f} s")
print("Saved to", TAB_DIR)
