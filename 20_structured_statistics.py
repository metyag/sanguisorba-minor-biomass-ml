# -*- coding: utf-8 -*-
"""
20 — Structured statistical approaches that a pure estimator sweep cannot reach.

Four questions, each of which is about the *form* of the analysis rather than
about which learner is applied.

1. WHICH TRAITS.  All 255 non-empty subsets of the eight traits are scored, so
   that the choice of predictors is made by exhaustive search rather than by a
   greedy forward selection.  Because choosing the best of 255 subsets on the
   same folds that score it is optimistic, the same search is then run again
   with the subset choice inside the validation loop; the difference between
   the two is the selection optimism attributable to feature choice alone.

2. HOW TO WEIGHT.  A cell mean computed from ten plants is not equally precise
   in every cell: the within-cell standard deviation of green weight varies
   several-fold across cells.  Weighting each cell by the reciprocal of the
   squared standard error of its mean is the generalised-least-squares answer
   and costs nothing to test.

3. WHICH UNIT.  Predicting the genotype mean over both seasons (31 units) is a
   third possible unit of analysis, coarser than the cell and closer still to
   what a breeder selects on.

4. WHAT COUNTS AS SUCCESS.  A breeder does not need the predicted weight to be
   correct in grams; the decision is which genotypes to keep.  Rank correlation
   and top-k recovery measure that directly, and are reported alongside R2 for
   every design.

Outputs (results/tables/): 20_subsets.csv, 20_subset_nested.json,
  20_weighting.csv, 20_genotype_level.csv, 20_rank_metrics.csv, 20_summary.json
"""
import itertools
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arsenal import (GroupKFoldArray, GroupOut, build_frames, build_pool,
                     evaluate, make_pipe, metrics)
from common_setup import GENOTYPE, MORPH_FEATURES, TAB_DIR, TARGET_EN, YEAR

t_start = time.time()
print("=" * 100)
print("20 — STRUCTURED STATISTICAL APPROACHES")
print("=" * 100, flush=True)

PLANT, CELL, GENO = build_frames()
POOL = build_pool("cell")
yc = CELL[TARGET_EN].values.astype(float)
g_cell = (CELL[YEAR].astype(str) + "_" + CELL[GENOTYPE].astype(str)).values
g_geno = CELL[GENOTYPE].astype(str).values
g_year = CELL[YEAR].astype(str).values
LOG_MORPH = [f"log {c}" for c in MORPH_FEATURES]

DESIGNS = {
    "grouped 5-fold": GroupKFoldArray(g_cell, n_splits=5),
    "LOCO": GroupOut(g_cell),
    "LOGO": GroupOut(g_geno),
    "LOYO": GroupOut(g_year),
}

# ===========================================================================
# 1. EXHAUSTIVE SUBSET SEARCH
# ===========================================================================
print("\n" + "=" * 100)
print("1. EXHAUSTIVE SUBSET SEARCH OVER THE EIGHT TRAITS")
print("=" * 100, flush=True)

SUBSETS = [c for k in range(1, 9) for c in itertools.combinations(range(8), k)]
SUB_MODELS = ["Huber", "Ridge", "ARD Regression", "Bayesian Ridge", "PLS (3 comp)"]
print(f"  subsets: {len(SUBSETS)}   estimators: {len(SUB_MODELS)}   "
      f"target: log   view: log-transformed traits", flush=True)

inner5 = GroupKFoldArray(g_cell, n_splits=5)


def score_subset(args):
    idx, mname = args
    cols = [LOG_MORPH[i] for i in idx]
    try:
        met, _ = evaluate(POOL[mname], CELL, cols, "log", yc, inner5)
        return {"Model": mname, "k": len(idx),
                "Traits": ", ".join(MORPH_FEATURES[i] for i in idx),
                "idx": ",".join(map(str, idx)), "Status": "ok", **met}
    except Exception as e:
        return {"Model": mname, "k": len(idx),
                "Traits": ", ".join(MORPH_FEATURES[i] for i in idx),
                "idx": ",".join(map(str, idx)), "Status": f"FAILED: {type(e).__name__}"}


t0 = time.time()
sub_res = Parallel(n_jobs=-1, batch_size=32)(
    delayed(score_subset)((s, m)) for m in SUB_MODELS for s in SUBSETS)
subs = pd.DataFrame(sub_res)
subs.to_csv(os.path.join(TAB_DIR, "20_subsets.csv"), index=False)
ok = subs[subs.Status == "ok"]
print(f"  {len(ok)} subset-model combinations in {time.time()-t0:.0f} s\n")

print("TOP 12 SUBSETS (grouped five-fold)")
print(ok.sort_values("pooled_R2", ascending=False)
      [["Model", "k", "Traits", "pooled_R2", "pooled_MAE"]].head(12)
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

print("\nBEST R2 BY NUMBER OF TRAITS")
byk = ok.sort_values("pooled_R2", ascending=False).groupby("k").head(1).sort_values("k")
print(byk[["k", "Model", "Traits", "pooled_R2", "pooled_MAE"]]
      .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# --- the same search with the subset choice inside the validation loop ------
print("\nNESTED SUBSET SELECTION (choice made on the training cells only)")
print("-" * 100, flush=True)


def nested_subset(design_name, mname="Huber"):
    spl = DESIGNS[design_name]
    oof = np.full(len(yc), np.nan)
    picks = []
    for tr, te in spl.split():
        inner = GroupKFoldArray(g_cell[tr], n_splits=min(5, len(tr)))
        best, best_s = None, -np.inf
        for idx in SUBSETS:
            cols = [LOG_MORPH[i] for i in idx]
            X = CELL[cols].astype(float)
            o = np.full(len(tr), np.nan)
            try:
                for a, b in inner.split(np.arange(len(tr))):
                    p = make_pipe(POOL[mname], "log")
                    p.fit(X.iloc[tr[a]], yc[tr[a]])
                    o[b] = np.asarray(p.predict(X.iloc[tr[b]])).ravel()
                s = r2_score(yc[tr], o)
            except Exception:
                s = -np.inf
            if s > best_s:
                best_s, best = s, idx
        cols = [LOG_MORPH[i] for i in best]
        X = CELL[cols].astype(float)
        p = make_pipe(POOL[mname], "log")
        p.fit(X.iloc[tr], yc[tr])
        oof[te] = np.asarray(p.predict(X.iloc[te])).ravel()
        picks.append(best)
    return metrics(yc, oof), picks


nested_out = {}
for d in ("LOCO", "LOGO"):
    t0 = time.time()
    met, picks = nested_subset(d)
    from collections import Counter
    cnt = Counter(picks)
    freq = [{"Traits": ", ".join(MORPH_FEATURES[i] for i in k), "k": len(k),
             "times": v, "share_pct": 100 * v / len(picks)}
            for k, v in cnt.most_common(5)]
    trait_freq = Counter(i for p in picks for i in p)
    nested_out[d] = {"metrics": met, "top_picks": freq,
                     "trait_selection_share_pct": {
                         MORPH_FEATURES[i]: 100 * trait_freq.get(i, 0) / len(picks)
                         for i in range(8)}}
    print(f"  {d}: R2 = {met['pooled_R2']:.4f}   MAE = {met['pooled_MAE']:.2f}   "
          f"({time.time()-t0:.0f} s)")
    print(f"     most frequent subset: {freq[0]['Traits']}  ({freq[0]['share_pct']:.1f} %)")
    print("     how often each trait was selected:")
    for k, v in sorted(nested_out[d]["trait_selection_share_pct"].items(),
                       key=lambda kv: -kv[1]):
        print(f"       {k:32} {v:5.1f} %")

best_exhaustive = float(ok["pooled_R2"].max())
with open(os.path.join(TAB_DIR, "20_subset_nested.json"), "w", encoding="utf-8") as f:
    json.dump({"exhaustive_best_grouped5fold": best_exhaustive,
               "n_subsets": len(SUBSETS), "nested": nested_out}, f, indent=2)

# ===========================================================================
# 2. WEIGHTING BY THE PRECISION OF EACH CELL MEAN
# ===========================================================================
print("\n" + "=" * 100)
print("2. WEIGHTING EACH CELL BY THE PRECISION OF ITS MEAN")
print("=" * 100, flush=True)

se = CELL["_target_sd"].values / np.sqrt(CELL["_n_plants"].values)
print(f"  standard error of the cell mean: min {se.min():.2f} g, "
      f"median {np.median(se):.2f} g, max {se.max():.2f} g "
      f"({se.max()/se.min():.1f}-fold range)")
W = {
    "unweighted": None,
    "inverse variance (1/SE^2)": 1.0 / np.maximum(se, 1e-6) ** 2,
    "inverse SE (1/SE)": 1.0 / np.maximum(se, 1e-6),
}
w_rows = []
for wname, w in W.items():
    if w is not None:
        w = w / w.mean()
    for mname in ["Huber", "Ridge", "Bayesian Ridge", "Linear"]:
        for d in ("grouped 5-fold", "LOCO"):
            try:
                met, _ = evaluate(POOL[mname], CELL, LOG_MORPH, "log", yc,
                                  DESIGNS[d], sample_weight=w)
                w_rows.append({"Weighting": wname, "Model": mname, "Design": d,
                               "Status": "ok", **met})
            except Exception as e:
                w_rows.append({"Weighting": wname, "Model": mname, "Design": d,
                               "Status": f"FAILED: {type(e).__name__}"})
wt = pd.DataFrame(w_rows)
wt.to_csv(os.path.join(TAB_DIR, "20_weighting.csv"), index=False)
piv = wt[wt.Status == "ok"].pivot_table(index=["Design", "Model"],
                                        columns="Weighting", values="pooled_R2")
print()
print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

# ===========================================================================
# 3. THE GENOTYPE AS THE UNIT (31 units, both seasons averaged)
# ===========================================================================
print("\n" + "=" * 100)
print("3. GENOTYPE MEANS OVER BOTH SEASONS AS THE UNIT (31 units)")
print("=" * 100, flush=True)

yg = GENO[TARGET_EN].values.astype(float)
gg = GENO[GENOTYPE].astype(str).values
geno_designs = {"LOGO (31 folds)": GroupOut(gg),
                "grouped 5-fold": GroupKFoldArray(gg, n_splits=5)}
g_rows = []
for mname in ["Huber", "Ridge", "ARD Regression", "Bayesian Ridge", "PLS (3 comp)",
              "GP (linear+RBF)", "Random Forest", "GAM (splines)"]:
    for view_name, cols in (("log8", LOG_MORPH), ("raw8", MORPH_FEATURES)):
        for tname in ("log", "raw"):
            for d, spl in geno_designs.items():
                try:
                    met, _ = evaluate(POOL[mname], GENO, cols, tname, yg, spl)
                    g_rows.append({"Model": mname, "View": view_name, "Target": tname,
                                   "Design": d, "Status": "ok", **met})
                except Exception as e:
                    g_rows.append({"Model": mname, "View": view_name, "Target": tname,
                                   "Design": d, "Status": f"FAILED: {type(e).__name__}"})
gl = pd.DataFrame(g_rows)
gl.to_csv(os.path.join(TAB_DIR, "20_genotype_level.csv"), index=False)
gok = gl[gl.Status == "ok"]
for d in geno_designs:
    print(f"\n--- {d} ---")
    print(gok[gok.Design == d].sort_values("pooled_R2", ascending=False)
          [["Model", "View", "Target", "pooled_R2", "pooled_MAE", "pooled_MAPE",
            "spearman"]].head(8).to_string(index=False,
                                           float_format=lambda v: f"{v:.4f}"))

# ===========================================================================
# 4. RANK METRICS FOR THE DECISION THAT IS ACTUALLY MADE
# ===========================================================================
print("\n" + "=" * 100)
print("4. RANK AGREEMENT — the quantity a selection decision depends on")
print("=" * 100, flush=True)

rank_rows = []
for mname in ["Huber", "ARD Regression", "Ridge"]:
    for d in DESIGNS:
        try:
            met, oof = evaluate(POOL[mname], CELL, LOG_MORPH, "log", yc, DESIGNS[d])
            row = {"Unit": "genotype x year cell", "Model": mname, "Design": d, **met}
            rank_rows.append(row)
        except Exception:
            pass
for mname in ["Huber", "ARD Regression", "Ridge"]:
    try:
        met, oof = evaluate(POOL[mname], GENO, LOG_MORPH, "log", yg,
                            GroupOut(gg))
        rank_rows.append({"Unit": "genotype (both seasons)", "Model": mname,
                          "Design": "LOGO (31 folds)", **met})
    except Exception:
        pass
rk = pd.DataFrame(rank_rows)
rk.to_csv(os.path.join(TAB_DIR, "20_rank_metrics.csv"), index=False)
cols_show = [c for c in ["Unit", "Model", "Design", "pooled_R2", "spearman", "kendall"]
             if c in rk.columns] + [c for c in rk.columns if c.startswith("top")]
print(rk[cols_show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))

summary = {
    "exhaustive_subset_best_grouped5fold": best_exhaustive,
    "nested_subset": {k: v["metrics"] for k, v in nested_out.items()},
    "weighting_best": (wt[wt.Status == "ok"].sort_values("pooled_R2", ascending=False)
                       .head(1).to_dict("records")[0] if len(wt[wt.Status == "ok"]) else {}),
    "genotype_level_best": (gok.sort_values("pooled_R2", ascending=False)
                            .head(1).to_dict("records")[0] if len(gok) else {}),
    "se_range_g": [float(se.min()), float(np.median(se)), float(se.max())],
    "elapsed_s": float(time.time() - t_start),
}
with open(os.path.join(TAB_DIR, "20_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)
print(f"\nTotal elapsed {time.time()-t_start:.0f} s")
