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
from common_setup import MORPH_FEATURES, SEED, TAB_DIR, load_data, make_pipeline

DRAWS = 15
PLANTS = [2, 4, 6, 8, 10]
CELLS = [10, 20, 30, 40, None]
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
    n_train_cells = 49
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
