# -*- coding: utf-8 -*-
"""
23 — The selection ladder, and the model that comes out of it.

Script 21 produced a result that decides how the revision should report its
headline.  Widening the search from 88 candidates to 540 *raised* the best
observed score and *lowered* the honestly validated one, because the inner
cross-validation of a 540-member space is itself noisy enough to pick a
configuration that does not generalise: in 71 per cent of the outer folds it
chose a three-component principal-component regression that then performed
poorly on the held-out cell.

The right response is not to hide the wide search but to report what each size
of search actually buys.  This script assembles the ladder:

    no selection at all      one model, fixed in advance
    SPACE C, 8 candidates    robust linear regression of the eight measured
                             traits, identity or logarithmic response
    SPACE B, 90 candidates   adds the strategy dimension of script 19
    SPACE A, 540 candidates  the wide structural grid of script 21

SPACE C is declared here on grounds that do not depend on any model score.
Section 18(a) established, from the structure of the records alone, that every
genotype x year cell contains one record whose predictor values sit several
within-cell standard deviations above the other nine.  A loss function that is
robust to a single extreme observation is therefore indicated before any model
is fitted, which is why the space contains least-absolute-deviation and Huber
regression, with ordinary ridge regression kept as the non-robust reference.
The eight measured traits are used as measured, with no derived variables, and
the response is either untransformed or logarithmic, the latter being the
standard allometric form in which biomass scales multiplicatively with plant
dimensions.

One property of the median regression deserves note because it removes a
correction that the earlier revision needed.  Least-absolute-deviation
regression on log y estimates the conditional *median* of log y, and because
the exponential is monotone, exponentiating gives the conditional median of y
exactly.  No smearing estimator is required for that quantity; a smearing
correction is needed only if the conditional mean is wanted instead, and both
are reported.

Outputs (results/tables/): 23_ladder.csv, 23_spaceC_frequency.csv,
  23_fixed_models.csv, 23_stability.csv, 23_final_model.json
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

from arsenal import (GroupKFoldArray, GroupOut, build_frames, build_pool,
                     evaluate, make_pipe, metrics)
from candidates import Context, build_candidate, spec_label
from common_setup import (FINAL_MODEL_NAME, FINAL_TARGET as _FT,
                          FINAL_VIEW as _FV, GENOTYPE, MORPH_FEATURES,
                          REFERENCE_MODEL_NAME, TAB_DIR, YEAR)

t_start = time.time()
print("=" * 100)
print("23 — THE SELECTION LADDER")
print("=" * 100, flush=True)

PLANT, CELL, _ = build_frames()
ctx = Context(PLANT, CELL)
POOL = build_pool("cell")
yc = ctx.y_cell

DESIGNS = {
    "Leave-one-cell-out (62 outer folds)": GroupOut(ctx.g_cell),
    "Leave-one-genotype-out (31 outer folds)": GroupOut(ctx.g_geno),
    "Leave-one-year-out (2 outer folds)": GroupOut(ctx.g_year),
}

# ===========================================================================
# SPACE C — declared from the data structure, not from any model score
# ===========================================================================
ROBUST = ["Quantile (median)", "Huber", "Huber (eps 1.1)", "Ridge"]
SPACE_C = [("direct", m, "raw8", t) for m in ROBUST for t in ("raw", "log")]
print(f"\nSPACE C : {len(SPACE_C)} candidates "
      f"({len(ROBUST)} loss functions x identity or logarithmic response, "
      f"eight measured traits)")
for s in SPACE_C:
    print("   ", spec_label(s))


def inner_score(spec, tr, n_splits=5):
    inner = GroupKFoldArray(ctx.g_cell[tr], n_splits=min(n_splits, len(tr)))
    fp = build_candidate(ctx, spec, POOL)
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
    space, tr, te = args
    scores = [inner_score(s, tr) for s in space]
    k = int(np.argmax(scores))
    best = space[k]
    try:
        pred = build_candidate(ctx, best, POOL)(tr, te)
    except Exception:
        pred = np.full(len(te), float(np.mean(yc[tr])))
    return te, np.asarray(pred).ravel(), best


print("\nNested selection over SPACE C")
print("-" * 100, flush=True)
ladder_rows, freq_rows, oofC = [], [], {}
for dname, spl in DESIGNS.items():
    folds = list(spl.split())
    t0 = time.time()
    out = Parallel(n_jobs=-1, batch_size=1)(
        delayed(one_fold)((SPACE_C, tr, te)) for tr, te in folds)
    oof = np.full(len(yc), np.nan)
    picks = []
    for te, pred, best in out:
        oof[te] = pred
        picks.append(best)
    met = metrics(yc, oof)
    oofC[dname] = oof
    ladder_rows.append({"Selection space": "SPACE C (theory-driven)",
                        "n_candidates": len(SPACE_C), "Design": dname, **met})
    c = Counter(spec_label(p) for p in picks)
    for cfg, n in c.most_common():
        freq_rows.append({"Design": dname, "Configuration": cfg, "Times selected": n,
                          "Share %": 100 * n / len(picks)})
    print(f"  {dname:42} R2={met['pooled_R2']:7.4f}  MAE={met['pooled_MAE']:7.2f}  "
          f"MAPE={met['pooled_MAPE']:6.2f}  rho={met['spearman']:.3f}  "
          f"({time.time()-t0:.0f} s)", flush=True)
    print(f"      chosen most often: {c.most_common(1)[0][0]} "
          f"({100*c.most_common(1)[0][1]/len(picks):.1f} %)")
    short = dname.split()[0].lower().replace("-", "")
    pd.DataFrame({"actual": yc, "predicted": oof, YEAR: CELL[YEAR],
                  GENOTYPE: CELL[GENOTYPE]}).to_csv(
        os.path.join(TAB_DIR, f"23_oof_C_{short}.csv"), index=False)

pd.DataFrame(freq_rows).to_csv(os.path.join(TAB_DIR, "23_spaceC_frequency.csv"),
                               index=False)

# ===========================================================================
# FIXED MODELS — no selection at all
# ===========================================================================
print("\n" + "=" * 100)
print("FIXED MODELS — specified in advance, nothing selected")
print("=" * 100, flush=True)

FIXED = [("Quantile (median)", "raw8", "log"),
         ("Huber (eps 1.1)", "raw8", "log"),
         ("Huber", "raw8", "log"),
         ("Ridge", "raw8", "log")]
fx_rows = []
for (m, v, t) in FIXED:
    for dname, spl in DESIGNS.items():
        met, oof = evaluate(POOL[m], CELL, MORPH_FEATURES if v == "raw8" else v,
                            t, yc, spl)
        fx_rows.append({"Model": m, "View": v, "Target": t, "Design": dname, **met})
    # grouped five-fold as well, for comparability with the sweep
    met5, _ = evaluate(POOL[m], CELL, MORPH_FEATURES, t, yc,
                       GroupKFoldArray(ctx.g_cell, n_splits=5))
    fx_rows.append({"Model": m, "View": v, "Target": t,
                    "Design": "Grouped five-fold over cells", **met5})
fx = pd.DataFrame(fx_rows)
fx.to_csv(os.path.join(TAB_DIR, "23_fixed_models.csv"), index=False)
print(fx.pivot_table(index=["Model", "Target"], columns="Design", values="pooled_R2")
      .to_string(float_format=lambda v: f"{v:.4f}"))

for (m, v, t) in FIXED:
    for dname in DESIGNS:
        r = fx[(fx.Model == m) & (fx.Design == dname)].iloc[0]
        ladder_rows.append({"Selection space": "None (model fixed in advance)",
                            "n_candidates": 1, "Design": dname,
                            **{k: r[k] for k in r.index if k not in
                               ("Model", "View", "Target", "Design")},
                            "Model": m})

# ===========================================================================
# THE LADDER
# ===========================================================================
for path, label, ncol in (("21_summary.json", None, None),
                          ("14_summary.json", "SPACE of script 14", 88)):
    p = os.path.join(TAB_DIR, path)
    if not os.path.exists(p):
        continue
    js = json.load(open(p, encoding="utf-8"))
    for r in js.get("results", []):
        ladder_rows.append({
            "Selection space": (label or f"{r.get('Space', '?')} "
                                f"({r.get('n_candidates', '?')} candidates)"),
            "n_candidates": r.get("n_candidates", ncol),
            "Design": r["Design"],
            "pooled_R2": r["pooled_R2"], "pooled_RMSE": r.get("pooled_RMSE"),
            "pooled_MAE": r["pooled_MAE"], "pooled_MAPE": r["pooled_MAPE"],
            "spearman": r.get("spearman")})

lad = pd.DataFrame(ladder_rows)
lad.to_csv(os.path.join(TAB_DIR, "23_ladder.csv"), index=False)

print("\n" + "=" * 100)
print("THE LADDER — honest performance as a function of how much was searched")
print("=" * 100)
best_fixed = lad[(lad["Selection space"] == "None (model fixed in advance)")
                 & (lad.get("Model") == "Quantile (median)")]
show = lad[lad["Selection space"] != "None (model fixed in advance)"]
show = pd.concat([best_fixed, show], ignore_index=True)
piv = show.pivot_table(index=["Selection space", "n_candidates"], columns="Design",
                       values="pooled_R2")
print(piv.to_string(float_format=lambda v: f"{v:.4f}"))

# ===========================================================================
# STABILITY OF THE FIXED MODEL
# ===========================================================================
print("\n" + "=" * 100)
print("STABILITY — 50 independent random partitions of the 62 cells")
print("=" * 100, flush=True)


def stab_one(args):
    m, seed = args
    try:
        met, _ = evaluate(POOL[m], CELL, MORPH_FEATURES, "log", yc,
                          GroupKFoldArray(ctx.g_cell, n_splits=5, seed=seed))
        return {"Model": m, "seed": seed, "R2": met["pooled_R2"],
                "MAE": met["pooled_MAE"], "MAPE": met["pooled_MAPE"],
                "spearman": met.get("spearman", np.nan)}
    except Exception:
        return None


st = [r for r in Parallel(n_jobs=-1, batch_size=8)(
    delayed(stab_one)((m, s)) for m, _, _ in FIXED for s in range(1000, 1050))
    if r is not None]
stab = pd.DataFrame(st)
stab.to_csv(os.path.join(TAB_DIR, "23_stability.csv"), index=False)
agg = stab.groupby("Model").agg(R2_mean=("R2", "mean"), R2_sd=("R2", "std"),
                                MAE_mean=("MAE", "mean"), MAE_sd=("MAE", "std"),
                                MAPE_mean=("MAPE", "mean"), MAPE_sd=("MAPE", "std"),
                                n=("R2", "size")).sort_values("R2_mean",
                                                              ascending=False)
print(agg.to_string(float_format=lambda v: f"{v:.4f}"))

# ===========================================================================
# THE FINAL MODEL
# ===========================================================================
FINAL_MODEL, FINAL_VIEW, FINAL_TARGET = FINAL_MODEL_NAME, _FV, _FT
print("\n" + "=" * 100)
print(f"FINAL MODEL: {FINAL_MODEL}, eight measured traits, "
      f"{'logarithmic' if FINAL_TARGET == 'log' else FINAL_TARGET} response")
print("=" * 100)

pipe = make_pipe(POOL[FINAL_MODEL], FINAL_TARGET)
pipe.fit(CELL[MORPH_FEATURES].astype(float), yc)
mdl = pipe.named_steps["model"].regressor_
scaler = pipe.named_steps["scale"]
coef_scaled = np.asarray(mdl.coef_).ravel()
# undo the standardisation so the coefficients are exponents on the raw traits
coef_raw = coef_scaled / scaler.scale_
intercept = float(mdl.intercept_) - float(np.sum(coef_scaled * scaler.mean_
                                                 / scaler.scale_))
coefs = pd.DataFrame({"Trait": MORPH_FEATURES,
                      "Coefficient (standardised)": coef_scaled,
                      "Coefficient (per unit of the trait)": coef_raw,
                      "Multiplicative effect of +1 unit": np.exp(coef_raw)})
coefs.to_csv(os.path.join(TAB_DIR, "23_final_coefficients.csv"), index=False)
print(f"\nlog(green weight, g) = {intercept:.4f}")
for _, r in coefs.iterrows():
    print(f"    {r['Coefficient (per unit of the trait)']:+.5f} x {r['Trait']}"
          f"     (x{r['Multiplicative effect of +1 unit']:.4f} per unit)")

final = {"model": FINAL_MODEL, "view": FINAL_VIEW, "target": FINAL_TARGET,
         "intercept": intercept,
         "coefficients": coefs.to_dict("records"),
         "designs": {}, "stability": {}}
for dname in list(DESIGNS) + ["Grouped five-fold over cells"]:
    r = fx[(fx.Model == FINAL_MODEL) & (fx.Design == dname)]
    if len(r):
        rr = r.iloc[0]
        final["designs"][dname] = {k: float(rr[k]) for k in
                                   ["pooled_R2", "pooled_RMSE", "pooled_MAE",
                                    "pooled_MAPE", "bias", "spearman", "kendall"]
                                   if k in rr.index and pd.notna(rr[k])}
        tk = [c for c in rr.index if c.startswith("top")]
        for c in tk:
            final["designs"][dname][c] = float(rr[c])
srow = agg.loc[FINAL_MODEL]
final["stability"] = {"R2_mean": float(srow.R2_mean), "R2_sd": float(srow.R2_sd),
                      "MAE_mean": float(srow.MAE_mean), "MAE_sd": float(srow.MAE_sd),
                      "n_partitions": int(srow.n)}
final["ladder"] = ladder_rows
final["nested_spaceC"] = {r["Design"]: {k: r[k] for k in r if k != "Design"}
                          for r in ladder_rows
                          if r["Selection space"] == "SPACE C (theory-driven)"}
with open(os.path.join(TAB_DIR, "23_final_model.json"), "w", encoding="utf-8") as f:
    json.dump(final, f, indent=2, default=float)

# ---------------------------------------------------------------------------
# THE REDUCED MODEL: the traits that nested subset selection kept in every fold
# ---------------------------------------------------------------------------
_sub_p = os.path.join(TAB_DIR, "20_subset_nested.json")
if os.path.exists(_sub_p):
    _sub = json.load(open(_sub_p, encoding="utf-8"))
    share = _sub["nested"]["LOCO"]["trait_selection_share_pct"]
    KEEP = [t for t in MORPH_FEATURES if share.get(t, 0) >= 50]
    print("\n" + "=" * 100)
    print("REDUCED MODEL — the traits kept in at least half of the outer folds")
    print("=" * 100)
    print("  traits:", ", ".join(KEEP))
    red = {"traits": KEEP, "designs": {}}
    for dname, spl in list(DESIGNS.items()) + [
            ("Grouped five-fold over cells", GroupKFoldArray(ctx.g_cell, n_splits=5))]:
        met, _ = evaluate(POOL[FINAL_MODEL], CELL, KEEP, FINAL_TARGET, yc, spl)
        red["designs"][dname] = met
        print(f"  {dname:42} R2={met['pooled_R2']:7.4f}  MAE={met['pooled_MAE']:7.2f}  "
              f"MAPE={met['pooled_MAPE']:6.2f}  rho={met.get('spearman', float('nan')):.3f}")
    _rs = [r for r in Parallel(n_jobs=-1, batch_size=8)(
        delayed(lambda s: evaluate(POOL[FINAL_MODEL], CELL, KEEP, FINAL_TARGET, yc,
                                   GroupKFoldArray(ctx.g_cell, n_splits=5, seed=s))[0])(s)
        for s in range(1000, 1050))]
    red["stability"] = {"R2_mean": float(np.mean([m["pooled_R2"] for m in _rs])),
                        "R2_sd": float(np.std([m["pooled_R2"] for m in _rs], ddof=1)),
                        "n_partitions": len(_rs)}
    print(f"  stability over {red['stability']['n_partitions']} partitions: "
          f"R2 = {red['stability']['R2_mean']:.4f} +/- {red['stability']['R2_sd']:.4f}")
    final["reduced_model"] = red
    with open(os.path.join(TAB_DIR, "23_final_model.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, default=float)

# --- adapter files in the schema the manuscript builders already consume ----
_key = {"Grouped five-fold over cells": "grouped5fold",
        "Leave-one-cell-out (62 outer folds)": "LOCO",
        "Leave-one-genotype-out (31 outer folds)": "LOGO",
        "Leave-one-year-out (2 outer folds)": "LOYO"}
adapter = {"model": FINAL_MODEL, "features": list(MORPH_FEATURES),
           "target": FINAL_TARGET, "designs": {}}
for dname, short in _key.items():
    r = fx[(fx.Model == FINAL_MODEL) & (fx.Design == dname)]
    if not len(r):
        continue
    rr = r.iloc[0]
    d = {"R2": float(rr["pooled_R2"]), "RMSE": float(rr["pooled_RMSE"]),
         "MAE": float(rr["pooled_MAE"]), "MAPE": float(rr["pooled_MAPE"]),
         "bias": float(rr["bias"])}
    for extra in ("spearman", "kendall"):
        if extra in rr.index and pd.notna(rr[extra]):
            d[extra] = float(rr[extra])
    tk = [c for c in rr.index if c.startswith("top")]
    for c in tk:
        d[c] = float(rr[c])
    adapter["designs"][short] = d
ref = {"model": REFERENCE_MODEL_NAME, "features": list(MORPH_FEATURES),
       "target": FINAL_TARGET, "designs": {}}
for dname, short in _key.items():
    r = fx[(fx.Model == REFERENCE_MODEL_NAME) & (fx.Design == dname)]
    if not len(r):
        continue
    rr = r.iloc[0]
    d = {"R2": float(rr["pooled_R2"]), "RMSE": float(rr["pooled_RMSE"]),
         "MAE": float(rr["pooled_MAE"]), "MAPE": float(rr["pooled_MAPE"]),
         "bias": float(rr["bias"])}
    for extra in ("spearman", "kendall"):
        if extra in rr.index and pd.notna(rr[extra]):
            d[extra] = float(rr[extra])
    for c in [c for c in rr.index if c.startswith("top")]:
        d[c] = float(rr[c])
    ref["designs"][short] = d
adapter["reference"] = ref

with open(os.path.join(TAB_DIR, "23_final_diagnostics.json"), "w", encoding="utf-8") as f:
    json.dump(adapter, f, indent=2)

stab_adapter = {m: {"R2_mean": float(row.R2_mean), "R2_sd": float(row.R2_sd),
                    "MAE_mean": float(row.MAE_mean), "MAE_sd": float(row.MAE_sd),
                    "MAPE_mean": float(row.MAPE_mean), "MAPE_sd": float(row.MAPE_sd),
                    "n_partitions": int(row.n)}
                for m, row in agg.iterrows()}
with open(os.path.join(TAB_DIR, "23_stability.json"), "w", encoding="utf-8") as f:
    json.dump(stab_adapter, f, indent=2)

print("\nPerformance of the final model")
for d, m in final["designs"].items():
    print(f"  {d:42} R2={m['pooled_R2']:7.4f}  MAE={m['pooled_MAE']:7.2f}  "
          f"MAPE={m['pooled_MAPE']:6.2f}  rho={m.get('spearman', float('nan')):.3f}")
print(f"  stability over {final['stability']['n_partitions']} partitions: "
      f"R2 = {final['stability']['R2_mean']:.4f} +/- {final['stability']['R2_sd']:.4f}")
print(f"\nTotal elapsed {time.time()-t_start:.0f} s")
