import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import trim_mean

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arsenal import ALLOMETRIC, GroupKFoldArray, GroupOut, build_pool, evaluate
from common_setup import (GENOTYPE, MORPH_FEATURES, TAB_DIR, TARGET_EN, YEAR,
                          load_data)

t_start = time.time()
print("=" * 100)
print("22 — SUMMARISING THE TEN RECORDS OF A CELL INTO PREDICTORS")
print("=" * 100, flush=True)

df, _, _, _, _ = load_data()
df = df.copy()
df["_pos"] = df.groupby([YEAR, GENOTYPE]).cumcount() + 1

resp = (df.groupby([YEAR, GENOTYPE], as_index=False)[TARGET_EN]
          .mean().sort_values([YEAR, GENOTYPE]).reset_index(drop=True))


def summarise(how):
    src = df[df["_pos"] <= 9] if how == "mean of 1-9" else df
    grp = src.groupby([YEAR, GENOTYPE])[MORPH_FEATURES]
    if how in ("mean", "mean of 1-9", "mean + spread"):
        X = grp.mean()
    elif how == "median":
        X = grp.median()
    elif how == "trimmed":
        X = grp.agg(lambda s: trim_mean(s.values, 0.1))
    elif how == "max":
        X = grp.max()
    elif how == "min":
        X = grp.min()
    else:
        raise ValueError(how)
    X = X.reset_index().sort_values([YEAR, GENOTYPE]).reset_index(drop=True)
    if how == "mean + spread":
        sd = (src.groupby([YEAR, GENOTYPE])[MORPH_FEATURES].std(ddof=1)
                 .reset_index().sort_values([YEAR, GENOTYPE]).reset_index(drop=True))
        for c in MORPH_FEATURES:
            X[f"sd {c}"] = sd[c].values
    for name, fn in ALLOMETRIC.items():
        X[name] = fn(X)
    for c in list(X.columns):
        if c in (YEAR, GENOTYPE):
            continue
        v = X[c].astype(float)
        if np.all(v.values > 0):
            X[f"log {c}"] = np.log(v)
    X = X.merge(resp, on=[YEAR, GENOTYPE])
    return X


HOWS = ["mean", "median", "trimmed", "mean of 1-9", "max", "min", "mean + spread"]
FRAMES = {h: summarise(h) for h in HOWS}
for h, F in FRAMES.items():
    print(f"  {h:15} {F.shape[0]} cells, {F.shape[1]} columns")

POOL = build_pool("cell")
MODELS = ["Huber", "Ridge", "ARD Regression", "Bayesian Ridge", "PLS (3 comp)",
          "GP (linear+RBF)", "Random Forest", "GAM (splines)"]

base = FRAMES["mean"]
yc = base[TARGET_EN].values.astype(float)
g_cell = (base[YEAR].astype(str) + "_" + base[GENOTYPE].astype(str)).values
g_geno = base[GENOTYPE].astype(str).values
g_year = base[YEAR].astype(str).values
DESIGNS = {
    "grouped 5-fold": GroupKFoldArray(g_cell, n_splits=5),
    "LOCO": GroupOut(g_cell),
    "LOGO": GroupOut(g_geno),
    "LOYO": GroupOut(g_year),
}


def cols_for(how, view):
    F = FRAMES[how]
    if view == "log8":
        cols = [f"log {c}" for c in MORPH_FEATURES]
    elif view == "raw8":
        cols = list(MORPH_FEATURES)
    elif view == "log8+spread":
        cols = [f"log {c}" for c in MORPH_FEATURES] + \
               [c for c in F.columns if c.startswith("sd ")]
    else:
        raise ValueError(view)
    return [c for c in cols if c in F.columns]


def job(args):
    how, view, model, target, dname = args
    F = FRAMES[how]
    cols = cols_for(how, view)
    if not cols:
        return None
    yv = F[TARGET_EN].values.astype(float)
    try:
        met, _ = evaluate(POOL[model], F, cols, target, yv, DESIGNS[dname])
        return {"Summary": how, "View": view, "Model": model, "Target": target,
                "Design": dname, "n_predictors": len(cols), "Status": "ok", **met}
    except Exception as e:
        return {"Summary": how, "View": view, "Model": model, "Target": target,
                "Design": dname, "n_predictors": len(cols),
                "Status": f"FAILED: {type(e).__name__}"}


JOBS = [(h, v, m, t, d)
        for h in HOWS
        for v in (["log8", "raw8"] + (["log8+spread"] if h == "mean + spread" else []))
        for m in MODELS for t in ("log", "raw")
        for d in DESIGNS]
print(f"\n  {len(JOBS)} evaluations", flush=True)

t0 = time.time()
rows = [r for r in Parallel(n_jobs=-1, batch_size=8)(delayed(job)(j) for j in JOBS)
        if r is not None]
out = pd.DataFrame(rows)
out.to_csv(os.path.join(TAB_DIR, "22_summaries.csv"), index=False)
ok = out[out.Status == "ok"]
print(f"  completed in {time.time()-t0:.0f} s\n")

for d in DESIGNS:
    sub = ok[ok.Design == d]
    if not len(sub):
        continue
    print(f"--- {d}: best configuration for each summarisation rule ---")
    b = sub.sort_values("pooled_R2", ascending=False).groupby("Summary").head(1)
    print(b.sort_values("pooled_R2", ascending=False)
          [["Summary", "View", "Model", "Target", "pooled_R2", "pooled_MAE",
            "pooled_MAPE", "spearman"]]
          .to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()

print("=" * 100)
print("LIKE FOR LIKE — Huber, log traits, log response, every design")
print("=" * 100)
lk = ok[(ok.Model == "Huber") & (ok.View == "log8") & (ok.Target == "log")]
print(lk.pivot_table(index="Summary", columns="Design", values="pooled_R2")
      .to_string(float_format=lambda v: f"{v:.4f}"))

best = ok.loc[ok[ok.Design == "LOCO"]["pooled_R2"].idxmax()] if len(ok[ok.Design == "LOCO"]) else None
summary = {
    "rules": HOWS,
    "best_LOCO": ({k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                   for k, v in best.items()} if best is not None else {}),
    "like_for_like_LOCO": {
        r["Summary"]: float(r["pooled_R2"])
        for _, r in lk[lk.Design == "LOCO"].iterrows()},
    "elapsed_s": float(time.time() - t_start),
}
with open(os.path.join(TAB_DIR, "22_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\nTotal elapsed {time.time()-t_start:.0f} s")
