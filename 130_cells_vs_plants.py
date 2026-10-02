# -*- coding: utf-8 -*-
"""130 - Which resource is the limiting one: the number of plants or the number of
cells?

The learning curve of Section 3.2 (Figure 9b) thins only the TRAINING ROWS (sklearn
learning_curve with shuffle=True), so it varies the number of plants; even in the
smallest subset almost every training cell is still represented. The curve on its own
therefore cannot test the claim that the number of cells is the limiting resource.

This script tests that claim directly. The outer validation is Design B (GroupKFold(5),
groups are the genotype x year cells); the test folds are never touched. For each
training fold two families are built:
  (a) ALL training cells, k plants per cell        (k = 2, 4, 6, 8, 10)
  (b) m WHOLE cells, ten plants each               (m = 10, 20, 30, 40, all)
and the two are also compared at an equal record budget (about 200, 300, 400 records).
Model her yerde ayni: ExtraTrees (800 agac, seed 42), sekiz morfolojik ozellik,
egitim kati icinde kurulan on isleme hatti.

Cikti: results/tables/130_cells_vs_plants.csv
  Family (cells fixed / plants fixed / matched budget), Draws, Cells, PlantsPerCell,
  Records, R2_mean (katlarin ortalamasi, cekilislere gore ortalanmis), R2_sd
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

KOK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KOK)
os.chdir(KOK)
sys.stdout.reconfigure(encoding="utf-8")
from common_setup import MORPH_FEATURES, SEED, TAB_DIR, load_data, make_pipeline  # noqa: E402

DRAWS = 15            # independent random draws per configuration
PLANTS = [2, 4, 6, 8, 10]
CELLS = [10, 20, 30, 40, None]      # None = egitim katindaki tum cells
BUDGETS = [200, 300, 400]


def model():
    return make_pipeline(ExtraTreesRegressor(n_estimators=800, random_state=SEED, n_jobs=-1),
                         MORPH_FEATURES)


def fold_r2(df, y, tr_idx, te_idx):
    yv = np.asarray(y)
    p = model()
    p.fit(df.iloc[tr_idx][MORPH_FEATURES], yv[tr_idx])
    return r2_score(yv[te_idx], p.predict(df.iloc[te_idx][MORPH_FEATURES]))


def pick(rng, cell_index, n_cells, n_plants):
    """n_cells hucre pick, her birinden n_plants records pick; records indekslerini dondur."""
    keys = list(cell_index)
    if n_cells is not None and n_cells < len(keys):
        keys = [keys[i] for i in rng.choice(len(keys), n_cells, replace=False)]
    chosen = []
    for a in keys:
        idx = cell_index[a]
        if n_plants < len(idx):
            idx = [idx[i] for i in rng.choice(len(idx), n_plants, replace=False)]
        chosen.extend(idx)
    return np.array(sorted(chosen))


def run_draws(df, y, gc, n_cells, n_plants, draws=DRAWS):
    """Her cekiliste bes katin R2 ortalamasi; cekilisler uzerinden ortalama ve SD."""
    g = np.asarray(gc, dtype=object)
    cv = list(GroupKFold(n_splits=5).split(df, y, groups=g))
    records = []
    for c in range(draws):
        rng = np.random.default_rng(1000 + c)
        folds, n_records = [], []
        for tr, te in cv:
            cells = {}
            for pos in tr:
                cells.setdefault(g[pos], []).append(pos)
            selection = pick(rng, cells, n_cells, n_plants)
            folds.append(fold_r2(df, y, selection, te))
            n_records.append(len(selection))
        records.append((float(np.mean(folds)), float(np.mean(n_records))))
    r2 = [k[0] for k in records]
    return float(np.mean(r2)), float(np.std(r2, ddof=1)), float(np.mean([k[1] for k in records]))


def main():
    df, y, gc, _, _ = load_data()
    rows = []
    for k in PLANTS:
        m, s, n = run_draws(df, y, gc, None, k)
        rows.append({"Family": "all training cells, k plants each", "Cells": "all (49-50)",
                         "PlantsPerCell": k, "Records": round(n, 1), "R2_mean": m, "R2_sd": s})
        print("plants per cell", k, round(m, 4))
    for c in CELLS:
        m, s, n = run_draws(df, y, gc, c, 10)
        rows.append({"Family": "whole cells, ten plants each", "Cells": c if c else "all (49-50)",
                         "PlantsPerCell": 10, "Records": round(n, 1), "R2_mean": m, "R2_sd": s})
        print("whole cells", c, round(m, 4))
    # Equal record budget: the same number of records, on one side many cells with
    # few plants, on the other few cells with all ten plants.
    n_train_cells = 49  # the smallest training fold; the budget is based on it
    for b in BUDGETS:
        k = max(1, int(round(b / n_train_cells)))
        m1, s1, n1 = run_draws(df, y, gc, None, k)
        c = max(1, int(round(b / 10)))
        m2, s2, n2 = run_draws(df, y, gc, c, 10)
        rows.append({"Family": "matched budget: many cells, few plants", "Cells": "all (49-50)",
                         "PlantsPerCell": k, "Records": round(n1, 1), "R2_mean": m1, "R2_sd": s1})
        rows.append({"Family": "matched budget: few cells, ten plants", "Cells": c,
                         "PlantsPerCell": 10, "Records": round(n2, 1), "R2_mean": m2, "R2_sd": s2})
        print("budget", b, "many cells", round(m1, 4), "(%.0f records)" % n1,
              "| few cells", round(m2, 4), "(%.0f records)" % n2)
    t = pd.DataFrame(rows)
    t["Draws"] = DRAWS
    t = t[["Family", "Draws", "Cells", "PlantsPerCell", "Records", "R2_mean", "R2_sd"]]
    yol = os.path.join(TAB_DIR, "130_cells_vs_plants.csv")
    t.to_csv(yol, index=False)
    print(t.to_string(index=False))
    print("written:", yol)


if __name__ == "__main__":
    main()
