# -*- coding: utf-8 -*-
"""
29 - Rank agreement of the adopted cell-level model WITHIN each season.

Why: the rank correlation and the top-ten recovery reported for the adopted model
(23_final_model.json) are pooled over all 62 cells. The two seasons differ about
2.4-fold in mean fresh weight and all ten highest-yielding cells belong to the second
season, so part of the pooled rank agreement is the season contrast itself. A breeder
compares genotypes within a season, so the same statistics are computed here within
each season, from the same out-of-fold predictions that Figure 10 and Table 6 use.

The predictions are recomputed exactly as in 25_figures_arsenal.py (arsenal.evaluate on
the cell frame, eight trait means, logarithmic response, GroupOut over the design's
groups); the pooled values are checked against 23_final_model.json before anything is
written, so the file cannot drift from Table 6.

Output: results/tables/29_within_season_rank.csv
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr

from arsenal import GroupOut, build_frames, build_pool, evaluate
from common_setup import GENOTYPE, MORPH_FEATURES, TAB_DIR, TARGET_EN, YEAR

FINAL = json.load(open(os.path.join(TAB_DIR, "23_final_model.json"), encoding="utf-8"))
MODEL = FINAL["model"]
assert FINAL["view"] == "raw8" and FINAL["target"] == "log", "adopted configuration changed"

_, CELL, _ = build_frames()
y = CELL[TARGET_EN].values.astype(float)
season = CELL[YEAR].values
groups = {
    "Leave-one-cell-out (62 outer folds)":
        (CELL[YEAR].astype(str) + "_" + CELL[GENOTYPE].astype(str)).values,
    "Leave-one-genotype-out (31 outer folds)": CELL[GENOTYPE].astype(str).values,
    "Leave-one-year-out (2 outer folds)": CELL[YEAR].astype(str).values,
}
POOL = build_pool("cell")

rows = []
for design, g in groups.items():
    met, pred = evaluate(POOL[MODEL], CELL, MORPH_FEATURES, "log", y, GroupOut(g))
    ref = FINAL["designs"][design]
    for key in ("pooled_R2", "spearman", "kendall"):
        assert abs(float(met[key]) - float(ref[key])) < 1e-9, (design, key, met[key], ref[key])
    top10 = set(np.argsort(-y)[:10])
    rows.append({"Design": design, "Season": "pooled", "n_cells": len(y),
                 "spearman": float(spearmanr(y, pred).correlation),
                 "kendall": float(kendalltau(y, pred).correlation),
                 "top10_cells_in_season_2": int(sum(1 for i in top10 if season[i] == 2))})
    for s in np.unique(season):
        m = season == s
        rows.append({"Design": design, "Season": int(s), "n_cells": int(m.sum()),
                     "spearman": float(spearmanr(y[m], pred[m]).correlation),
                     "kendall": float(kendalltau(y[m], pred[m]).correlation),
                     "top10_cells_in_season_2": np.nan})

out = pd.DataFrame(rows)
out.to_csv(os.path.join(TAB_DIR, "29_within_season_rank.csv"), index=False)
print(out.to_string(index=False))
