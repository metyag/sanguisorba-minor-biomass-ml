import json
import os
import sys
import warnings

import numpy as np
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from arsenal import (GroupKFoldArray, GroupOut, build_frames, build_pool, evaluate,
                     make_pipe, metrics)
from candidates import Context
from common_setup import (FINAL_MODEL_NAME, FINAL_TARGET, GENOTYPE, MORPH_FEATURES,
                          TAB_DIR, TARGET_EN, YEAR, cv_schemes, load_data)

POOL = build_pool("cell")
EST = POOL[FINAL_MODEL_NAME]
FINAL = json.load(open(os.path.join(TAB_DIR, "23_final_model.json"), encoding="utf-8"))


def ust_on(y, p, k=10):
    return int(len(set(np.argsort(-y)[:k]) & set(np.argsort(-p)[:k])))


def sezon_ici(y, p, sezon):
    out = {}
    for s in sorted(np.unique(sezon)):
        m = sezon == s
        yt, yp = y[m], p[m]
        out[str(s)] = {"n": int(m.sum()),
                       "R2": float(r2_score(yt, yp)),
                       "MAE": float(mean_absolute_error(yt, yp)),
                       "spearman": float(stats.spearmanr(yt, yp).statistic),
                       "top10_hits": ust_on(yt, yp)}
    return out


def hucre_tasarimlari(ctx):
    return {"Leave-one-cell-out": GroupOut(ctx.g_cell),
            "Leave-one-genotype-out": GroupOut(ctx.g_geno),
            "Leave-one-year-out": GroupOut(ctx.g_year),
            "Grouped 5-fold over cells": GroupKFoldArray(ctx.g_cell, n_splits=5)}


sonuc = {}

PLANT, CELL, _ = build_frames()
ctx = Context(PLANT, CELL)
yc = ctx.y_cell
sezon_c = CELL[YEAR].astype(int).values
a = {}
for ad, spl in hucre_tasarimlari(ctx).items():
    met, oof = evaluate(EST, CELL, MORPH_FEATURES, FINAL_TARGET, yc, spl)
    a[ad] = {"pooled": {"R2": met["pooled_R2"], "MAE": met["pooled_MAE"],
                        "spearman": met["spearman"], "top10_hits": ust_on(yc, oof)},
             "within_season": sezon_ici(yc, oof, sezon_c)}
ref = FINAL["designs"]["Leave-one-cell-out (62 outer folds)"]["pooled_R2"]
if abs(a["Leave-one-cell-out"]["pooled"]["R2"] - ref) > 1e-9:
    raise SystemExit("24: leave-one-cell-out R2 does not reproduce 23_final_model.json")
ilk_on = np.argsort(-yc)[:10]
a["ten_highest_cells_in_season_2"] = int((sezon_c[ilk_on] == 2).sum())
sonuc["within_season"] = a

df, y, g_cell, g_geno, g_year = load_data()[:5]
X = df[MORPH_FEATURES].astype(float)
y = np.asarray(y, dtype=float)
sema = cv_schemes(g_cell, g_geno, g_year)
plan = {"B. GroupKFold (genotype × year cell)": sema["B. GroupKFold by genotype × year cell (62 groups, 5 folds)"],
        "C. Leave-one-genotype-out": sema["C. Leave-One-Genotype-Out (31 folds)"],
        "D. Leave-one-year-out": sema["D. Leave-One-Year-Out (2 folds)"],
        "Leave-one-cell-out (plants)": (LeaveOneGroupOut(), g_cell, "loco")}
hucre = np.asarray(g_cell)
b = {}
for ad, (spl, grup, kod) in plan.items():
    oof = np.full(len(y), np.nan)
    for tr, te in spl.split(X, y, grup):
        pipe = make_pipe(EST, FINAL_TARGET, True)
        pipe.fit(X.iloc[tr], y[tr])
        oof[te] = np.asarray(pipe.predict(X.iloc[te])).ravel()
    u = sorted(np.unique(hucre), key=str)
    yh = np.array([y[hucre == k].mean() for k in u])
    ph = np.array([oof[hucre == k].mean() for k in u])
    b[ad] = {"plant_R2": float(r2_score(y, oof)), "plant_MAE": float(mean_absolute_error(y, oof)),
             "cell_mean_R2": float(r2_score(yh, ph)), "cell_mean_MAE": float(mean_absolute_error(yh, ph)),
             "cell_mean_spearman": float(stats.spearmanr(yh, ph).statistic), "code": kod}
sonuc["plant_level_huber_log"] = b

P9, C9, _ = build_frames(exclude_last_plant=True)
ctx9 = Context(P9, C9)
c = {}
for ad, spl in hucre_tasarimlari(ctx9).items():
    met, oof = evaluate(EST, C9, MORPH_FEATURES, FINAL_TARGET, ctx9.y_cell, spl)
    c[ad] = {"R2": met["pooled_R2"], "MAE": met["pooled_MAE"], "spearman": met["spearman"],
             "top10_hits": ust_on(ctx9.y_cell, oof)}
c["records"] = "1-9"
sonuc["records_1_to_9"] = c

yol = os.path.join(TAB_DIR, "26_r4_ek_analizler.json")
with open(yol, "w", encoding="utf-8") as f:
    json.dump(sonuc, f, indent=2, ensure_ascii=False)
print(json.dumps(sonuc, indent=1, ensure_ascii=False))
print("written:", yol)
