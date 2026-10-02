# -*- coding: utf-8 -*-
"""
14 — Nested model SELECTION: an unbiased headline number.

Why this is necessary. Scripts 09, 12 and 13 searched a large space (34 estimators
x 4 feature sets x 2 target transforms) and reported the best cell of that search.
Reporting the maximum of a large search as if it were an out-of-sample estimate is
itself a form of optimism: the winner is chosen partly because it suited the
particular folds. A reviewer who has just objected to leakage will spot this
immediately.

The fix is to make the SELECTION part of the procedure being validated:

    for each outer fold:
        hold out the cells of that fold
        run an inner grouped CV over the training cells only
        choose the (model, feature set, target) that wins the inner CV
        refit that choice on the training cells and predict the held-out cells

The pooled out-of-fold R2 that results is an estimate of "what this whole model
selection procedure achieves on cells it has never seen", which is the quantity
that should be reported. It is normally a little lower than the best cell of the
search, and that difference is exactly the selection optimism.

Outputs (results/tables/):
  14_nested_selection.csv, 14_selection_frequency.csv, 14_summary.json
"""
import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (load_data, make_pipeline, TAB_DIR, mape, SEED,
                          MORPH_FEATURES, YEAR, GENOTYPE, TARGET_EN)

df, y, groups_cell, groups_geno, groups_year = load_data()
ENG = ["Stem volume proxy", "Leaf area proxy", "Height x stems",
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


CELL = add_engineered(
    df.groupby([YEAR, GENOTYPE], as_index=False)[MORPH_FEATURES + [TARGET_EN]]
    .mean().sort_values([YEAR, GENOTYPE]).reset_index(drop=True))
yc = CELL[TARGET_EN].values
g_cell = (CELL[YEAR].astype(str) + "_" + CELL[GENOTYPE].astype(str)).values
g_geno = CELL[GENOTYPE].astype(str).values
g_year = CELL[YEAR].astype(str).values

FEATURE_SETS = {
    "A. Morphology only (8)": MORPH_FEATURES,
    "B. Morphology + allometric (13)": MORPH_FEATURES + ENG,
    "C. Season + morphology (9)": [YEAR] + MORPH_FEATURES,
    "D. Season + morphology + allometric (14)": [YEAR] + MORPH_FEATURES + ENG,
}

# ---------------------------------------------------------------------------
# Candidate pool: the configurations that were competitive in script 13,
# restricted to estimators that are cheap enough to refit inside a nested loop.
# ---------------------------------------------------------------------------
from sklearn.linear_model import HuberRegressor, Ridge, BayesianRidge, ARDRegression, ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor

POOL = {
    "Huber": HuberRegressor(epsilon=1.35, max_iter=3000),
    "Ridge": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=20000),
    "Bayesian Ridge": BayesianRidge(),
    "ARD Regression": ARDRegression(),
    "PLS (3 comp)": PLSRegression(n_components=3),
    "Kernel Ridge (RBF)": KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1),
    "GP (Matern 1.5)": GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-2),
        normalize_y=True, random_state=SEED, alpha=1e-8),
    "GP (RBF)": GaussianProcessRegressor(
        kernel=ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-2),
        normalize_y=True, random_state=SEED, alpha=1e-8),
    "ExtraTrees": ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=1),
    "Random Forest": RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=1),
}
if os.environ.get("TABPFN", "0") == "1":
    try:
        from tabpfn import TabPFNRegressor
        POOL["TabPFN (transformer)"] = TabPFNRegressor(device="cpu", random_state=SEED)
    except Exception as e:
        print("TabPFN not in pool:", e)

CANDIDATES = [(m, fs, tt) for m in POOL for fs in FEATURE_SETS for tt in ("raw", "log")]
print("=" * 100)
print("14 — NESTED MODEL SELECTION (cell level)")
print("=" * 100)
print(f"Cells: {len(CELL)}   candidate configurations: {len(CANDIDATES)} "
      f"({len(POOL)} estimators x {len(FEATURE_SETS)} feature sets x 2 target transforms)")


def fit_pred(m_name, fs_name, tt, tr_idx, te_idx):
    feats = FEATURE_SETS[fs_name]
    reg = POOL[m_name]
    r = (TransformedTargetRegressor(regressor=clone(reg), func=np.log1p,
                                    inverse_func=np.expm1) if tt == "log" else clone(reg))
    p = make_pipeline(r, feats, genotype_categorical=False)
    p.fit(CELL.iloc[tr_idx][feats], yc[tr_idx])
    return np.asarray(p.predict(CELL.iloc[te_idx][feats])).ravel()


class GroupOut:
    def __init__(self, g):
        self.g = np.asarray(g)

    def split(self, idx):
        for u in np.unique(self.g[idx]):
            te = idx[self.g[idx] == u]
            tr = idx[self.g[idx] != u]
            yield tr, te

    def n(self, idx):
        return len(np.unique(self.g[idx]))


def inner_score(m_name, fs_name, tt, tr_idx, inner_groups, n_splits=5):
    """Grouped k-fold R2 on the training cells only."""
    gk = GroupKFold(n_splits=min(n_splits, len(np.unique(inner_groups))))
    oof = np.full(len(tr_idx), np.nan)
    Xtr = CELL.iloc[tr_idx]
    for a, b in gk.split(Xtr, yc[tr_idx], groups=inner_groups):
        try:
            oof[b] = fit_pred(m_name, fs_name, tt, tr_idx[a], tr_idx[b])
        except Exception:
            return -np.inf
    m = ~np.isnan(oof)
    if m.sum() < 3:
        return -np.inf
    try:
        return r2_score(yc[tr_idx][m], oof[m])
    except Exception:
        return -np.inf


def nested_selection(outer_splitter_groups, label):
    """outer_splitter_groups: array defining the outer held-out unit."""
    go = GroupOut(outer_splitter_groups)
    all_idx = np.arange(len(CELL))
    oof = np.full(len(CELL), np.nan)
    picks = []
    t0 = time.time()
    for k, (tr, te) in enumerate(go.split(all_idx), 1):
        inner_groups = g_cell[tr]          # inner folds always block on the cell
        best, best_s = None, -np.inf
        for cfg in CANDIDATES:
            s = inner_score(cfg[0], cfg[1], cfg[2], tr, inner_groups)
            if s > best_s:
                best_s, best = s, cfg
        try:
            oof[te] = fit_pred(best[0], best[1], best[2], tr, te)
        except Exception:
            oof[te] = np.mean(yc[tr])
        picks.append(best)
        if k % 10 == 0 or k == go.n(all_idx):
            print(f"    fold {k}/{go.n(all_idx)}  last pick: {best}  "
                  f"({time.time()-t0:.0f} s)", flush=True)
    m = ~np.isnan(oof)
    met = {"Design": label,
           "pooled_R2": float(r2_score(yc[m], oof[m])),
           "pooled_RMSE": float(np.sqrt(mean_squared_error(yc[m], oof[m]))),
           "pooled_MAE": float(mean_absolute_error(yc[m], oof[m])),
           "pooled_MAPE": float(mape(yc[m], oof[m]))}
    return met, picks, oof


results, all_picks = [], {}
for groups_arr, label in [(g_cell, "Leave-one-cell-out (62 outer folds)"),
                          (g_geno, "Leave-one-genotype-out (31 outer folds)"),
                          (g_year, "Leave-one-year-out (2 outer folds)")]:
    print(f"\n--- {label} ---", flush=True)
    met, picks, oof = nested_selection(groups_arr, label)
    results.append(met)
    all_picks[label] = picks
    print(f"  => R2 = {met['pooled_R2']:.4f}   RMSE = {met['pooled_RMSE']:.2f}   "
          f"MAE = {met['pooled_MAE']:.2f}   MAPE = {met['pooled_MAPE']:.2f}", flush=True)
    pd.DataFrame({"actual": yc, "predicted": oof, YEAR: CELL[YEAR], GENOTYPE: CELL[GENOTYPE]}) \
        .to_csv(os.path.join(TAB_DIR,
                             f"14_oof_{label.split()[0].lower().replace('-', '')}.csv"),
                index=False)

res = pd.DataFrame(results)
res.to_csv(os.path.join(TAB_DIR, "14_nested_selection.csv"), index=False)

freq_rows = []
for label, picks in all_picks.items():
    c = Counter(picks)
    for (m, fs, tt), n in c.most_common():
        freq_rows.append({"Design": label, "Model": m, "Feature set": fs,
                          "Target": tt, "Times selected": n,
                          "Share %": 100 * n / len(picks)})
freq = pd.DataFrame(freq_rows)
freq.to_csv(os.path.join(TAB_DIR, "14_selection_frequency.csv"), index=False)

print("\n" + "=" * 100)
print("NESTED-SELECTION RESULTS (unbiased: the selection is inside the loop)")
print("=" * 100)
print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print("\nWhich configuration the inner loop chose, by frequency:")
print(freq.to_string(index=False, float_format=lambda v: f"{v:.1f}"))

with open(os.path.join(TAB_DIR, "14_summary.json"), "w", encoding="utf-8") as f:
    json.dump({"results": results,
               "selection_frequency": freq_rows,
               "n_candidates": len(CANDIDATES)}, f, indent=2)
print("\nSaved to", TAB_DIR)
