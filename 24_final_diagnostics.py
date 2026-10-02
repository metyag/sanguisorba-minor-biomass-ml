# -*- coding: utf-8 -*-
"""
24 — The plant-level comparison on a consistent protocol, and the diagnostics
     that the final model needs before it can be recommended for use.

Three parts.

(a) THE SAME ARSENAL AT THE PLANT LEVEL.  Script 18 compared 47 estimators at
    the cell level.  Repeating a restricted version of that comparison on the
    620 individual plants, blocked on the same cells and scored the same way,
    is what licenses any statement about how the ranking of method families
    depends on sample size.  It is restricted rather than complete only because
    a plant-level fit costs about ten times a cell-level one.

(b) RETRANSFORMATION.  The final model is fitted to log green weight by least
    absolute deviations, so exponentiating its prediction returns the
    conditional *median* exactly, since the exponential is monotone and the
    median commutes with monotone transformations.  If the conditional *mean*
    is wanted instead --- for a total-yield estimate rather than a per-cell
    typical value --- a smearing correction is required.  Both are computed,
    together with what each does to the error statistics, so that the choice is
    made explicitly rather than by accident.

(c) PREDICTION INTERVALS.  A point prediction is not enough to act on.  The
    half-widths are empirical quantiles of the absolute and relative
    leave-one-cell-out residuals of the 62 cells. The "empirical coverage"
    columns count the cells inside the interval on the same residuals, so they
    follow from the quantile level and are not an independent check of
    calibration; no finite-sample coverage guarantee is implied.

Outputs (results/tables/): 24_plant_sweep.csv, 24_retransformation.csv,
  24_conformal.csv, 24_summary.json
"""
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arsenal import (GroupKFoldArray, GroupOut, build_frames, build_pool, evaluate,
                     family_of, make_pipe, view_columns)
from common_setup import GENOTYPE, MORPH_FEATURES, TAB_DIR, TARGET_EN, YEAR, mape

t_start = time.time()
print("=" * 100)
print("24 — PLANT-LEVEL COMPARISON AND FINAL-MODEL DIAGNOSTICS")
print("=" * 100, flush=True)

PLANT, CELL, _ = build_frames()
yc = CELL[TARGET_EN].values.astype(float)
g_cell = (CELL[YEAR].astype(str) + "_" + CELL[GENOTYPE].astype(str)).values
g_geno = CELL[GENOTYPE].astype(str).values
g_year = CELL[YEAR].astype(str).values

# ===========================================================================
# (a) PLANT LEVEL, SAME PROTOCOL
# ===========================================================================
yp = PLANT[TARGET_EN].values.astype(float)
gp_cell = (PLANT[YEAR].astype(str) + "_" + PLANT[GENOTYPE].astype(str)).values
POOL_P = build_pool("plant")
P_VIEWS = ["raw8", "log8", "raw13"]
P_TARGETS = ["raw", "log"]
P_JOBS = [(m, v, t) for m in POOL_P for v in P_VIEWS for t in P_TARGETS]
print(f"\n(a) plant level: {len(P_JOBS)} configurations "
      f"({len(POOL_P)} estimators x {len(P_VIEWS)} views x {len(P_TARGETS)} transforms), "
      f"grouped five-fold blocked on the genotype x year cell", flush=True)

spl_p = GroupKFoldArray(gp_cell, n_splits=5)


def p_one(cfg):
    m, v, t = cfg
    try:
        met, _ = evaluate(POOL_P[m], PLANT, view_columns(v), t, yp, spl_p)
        return {"Model": m, "View": v, "Target": t, "Family": family_of(m),
                "Status": "ok", **met}
    except Exception as e:
        return {"Model": m, "View": v, "Target": t, "Family": family_of(m),
                "Status": f"FAILED: {type(e).__name__}"}


t0 = time.time()
pl = pd.DataFrame(Parallel(n_jobs=-1, batch_size=4)(delayed(p_one)(c) for c in P_JOBS))
pl.to_csv(os.path.join(TAB_DIR, "24_plant_sweep.csv"), index=False)
pok = pl[pl.Status == "ok"]
print(f"    completed {len(pok)}/{len(P_JOBS)} in {time.time()-t0:.0f} s\n")

cell_sweep = pd.read_csv(os.path.join(TAB_DIR, "18_sweep_cell.csv"))
cell_sweep = cell_sweep[cell_sweep.Status == "ok"]
cell_sweep = cell_sweep[cell_sweep.View.isin(P_VIEWS) & cell_sweep.Target.isin(P_TARGETS)]

fam_cmp = pd.DataFrame({
    "Plant level (620 records)": pok.groupby("Family")["pooled_R2"].max(),
    "Cell level (62 cells)": cell_sweep.groupby("Family")["pooled_R2"].max(),
}).sort_values("Cell level (62 cells)", ascending=False)
fam_cmp.to_csv(os.path.join(TAB_DIR, "24_family_comparison.csv"))
print("BEST R2 BY METHOD FAMILY, IDENTICAL PROTOCOL AT BOTH LEVELS")
print(fam_cmp.to_string(float_format=lambda v: f"{v:.4f}"))

# The manuscript quotes the linear-minus-trees gap at each level and the largest
# disagreement between the two levels. Those are differences of numbers in the
# table above rather than numbers in it, so they are written out here; otherwise
# the document verifier cannot trace them and has to report them as untraced.
_pl, _cl = "Plant level (620 records)", "Cell level (62 cells)"
fam_gaps = pd.DataFrame([{
    "Quantity": "linear minus tree ensembles, plant level",
    "Value": float(fam_cmp.loc["Regularised linear", _pl]
                   - fam_cmp.loc["Tree ensembles", _pl])},
    {"Quantity": "linear minus tree ensembles, cell level",
     "Value": float(fam_cmp.loc["Regularised linear", _cl]
                    - fam_cmp.loc["Tree ensembles", _cl])},
    {"Quantity": "largest |cell - plant| over the families",
     "Value": float((fam_cmp[_cl] - fam_cmp[_pl]).abs().max())},
])
fam_gaps.to_csv(os.path.join(TAB_DIR, "24_family_gaps.csv"), index=False)
print("\nDIFFERENCES QUOTED IN THE TEXT")
print(fam_gaps.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ===========================================================================
# (b) RETRANSFORMATION: median versus mean
# ===========================================================================
print("\n" + "=" * 100)
print("(b) RETRANSFORMATION — direct exponentiation versus a mean correction")
print("=" * 100, flush=True)

POOL = build_pool("cell")
from common_setup import FINAL_MODEL_NAME as _FM, FINAL_TARGET as _FT
MODEL, TARGET = _FM, _FT
DESIGNS = {"Leave-one-cell-out (62 outer folds)": GroupOut(g_cell),
           "Leave-one-genotype-out (31 outer folds)": GroupOut(g_geno),
           "Leave-one-year-out (2 outer folds)": GroupOut(g_year)}

rt_rows = []
oof_store = {}
for dname, spl in DESIGNS.items():
    X = CELL[MORPH_FEATURES].astype(float)
    naive = np.full(len(yc), np.nan)
    smear = np.full(len(yc), np.nan)
    param = np.full(len(yc), np.nan)
    for tr, te in spl.split():
        p = make_pipe(POOL[MODEL], TARGET)
        p.fit(X.iloc[tr], yc[tr])
        pred = np.asarray(p.predict(X.iloc[te])).ravel()
        naive[te] = pred
        # residuals on the log scale, from the training cells only
        fitted_tr = np.asarray(p.predict(X.iloc[tr])).ravel()
        r_log = np.log(yc[tr]) - np.log(np.maximum(fitted_tr, 1e-9))
        smear[te] = pred * float(np.mean(np.exp(r_log)))
        param[te] = pred * float(np.exp(0.5 * np.var(r_log, ddof=1)))
    oof_store[dname] = {"naive": naive.copy(), "smearing": smear.copy()}
    for label, v in (("exp of the log prediction", naive),
                     ("Duan smearing (conditional mean)", smear),
                     ("parametric exp(sigma^2/2) (conditional mean)", param)):
        rt_rows.append({"Design": dname, "Retransformation": label,
                        "R2": float(r2_score(yc, v)),
                        "MAE": float(mean_absolute_error(yc, v)),
                        "MAPE": float(mape(yc, v)),
                        "bias_g": float(np.mean(v - yc)),
                        "total_error_pct": float(100 * (v.sum() - yc.sum()) / yc.sum())})
rt = pd.DataFrame(rt_rows)
rt.to_csv(os.path.join(TAB_DIR, "24_retransformation.csv"), index=False)
print(rt.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ===========================================================================
# (c) PREDICTION INTERVALS FROM THE EMPIRICAL QUANTILES OF THE LOCO RESIDUALS
# ===========================================================================
print("\n" + "=" * 100)
print("(c) CONFORMAL PREDICTION INTERVALS from the leave-one-cell-out residuals")
print("=" * 100, flush=True)

loco_pred = oof_store["Leave-one-cell-out (62 outer folds)"]["naive"]
resid = yc - loco_pred
rel = resid / np.maximum(loco_pred, 1e-9)
cf_rows = []
for alpha in (0.20, 0.10, 0.05):
    q_abs = float(np.quantile(np.abs(resid), 1 - alpha))
    q_rel = float(np.quantile(np.abs(rel), 1 - alpha))
    cov_abs = float(np.mean(np.abs(resid) <= q_abs))
    lo, hi = loco_pred * (1 - q_rel), loco_pred * (1 + q_rel)
    cov_rel = float(np.mean((yc >= lo) & (yc <= hi)))
    cf_rows.append({"Nominal coverage %": 100 * (1 - alpha),
                    "Absolute half-width (g)": q_abs,
                    "Empirical coverage, absolute %": 100 * cov_abs,
                    "Relative half-width %": 100 * q_rel,
                    "Empirical coverage, relative %": 100 * cov_rel,
                    "Mean interval width (g), relative": float(np.mean(hi - lo))})
cf = pd.DataFrame(cf_rows)
cf.to_csv(os.path.join(TAB_DIR, "24_conformal.csv"), index=False)
print(cf.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

# ===========================================================================
# summary
# ===========================================================================
summary = {
    "plant_level_configs": int(len(P_JOBS)),
    "plant_level_best": (pok.loc[pok["pooled_R2"].idxmax()].to_dict()
                         if len(pok) else {}),
    "family_comparison": fam_cmp.reset_index().to_dict("records"),
    "retransformation": rt_rows,
    "conformal": cf_rows,
    "elapsed_s": float(time.time() - t_start),
}
with open(os.path.join(TAB_DIR, "24_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=float)
print(f"\nTotal elapsed {time.time()-t_start:.0f} s")
