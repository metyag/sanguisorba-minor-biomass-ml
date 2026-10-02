# -*- coding: utf-8 -*-
"""129 - Are 0.448 (forward selection, eight traits) and 0.469 (last point of the
learning curve) the same configuration?

Section 3.2 prints, in two places, the fold-wise mean R2 of "all eight morphological
traits, Design B, ExtraTrees (800 trees, seed 42), GroupKFold(5)": the last step of the
forward selection gives 0.448 (03_forward_selection.csv) and the last point of the
learning curve gives 0.469 (05_learning_curves.json). The two can be read as a
contradiction.

This script reproduces both from the same configuration and shows that the difference
comes only from the order of the columns and of the training rows, to which randomised
trees are sensitive:
  - columns in forward-selection order     -> 0.448 (the value in the file)
  - columns in MORPH_FEATURES order        -> different
  - learning_curve, shuffle=True, seed 42  -> 0.469 (the value in the file)
  - learning_curve, shuffle=False          -> different

Output: results/tables/129_same_configuration.csv. Both file values are expected to be
reproduced exactly.
"""
import io
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold, cross_validate, learning_curve

KOK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KOK)
os.chdir(KOK)
from common_setup import MORPH_FEATURES, SEED, TAB_DIR, load_data, make_pipeline  # noqa: E402

df, y, gc, _, _ = load_data()
BEST = ExtraTreesRegressor(n_estimators=800, random_state=SEED, n_jobs=-1)
fs = pd.read_csv(os.path.join(TAB_DIR, "03_forward_selection.csv"))
f = fs[(fs.Design == "B_groupcell") & (fs["Feature set"] == "MORPH")].sort_values("Step")
sira = list(f["Added feature"])
dosya_fs = float(f["Cumulative R2_mean"].iloc[-1])
lc = json.load(io.open(os.path.join(TAB_DIR, "05_learning_curves.json"), encoding="utf-8"))
dosya_lc = float(lc[[k for k in lc if k.startswith("(b)")][0]]["val_r2_mean"][-1])

rows = []
for ad, cols in (("forward-selection column order", sira), ("MORPH_FEATURES column order", MORPH_FEATURES)):
    out = cross_validate(make_pipeline(BEST, cols, genotype_categorical=True), df[cols], y,
                         cv=GroupKFold(n_splits=5), groups=gc, scoring={"r2": "r2"}, n_jobs=-1)
    rows.append({"Variant": ad, "foldwise_mean_R2": float(np.mean(out["test_r2"]))})
pipe = make_pipeline(BEST, MORPH_FEATURES, genotype_categorical=True)
for ad, kar in (("learning_curve, shuffled rows (seed 42)", True), ("learning_curve, unshuffled rows", False)):
    kw = {"shuffle": True, "random_state": SEED} if kar else {"shuffle": False}
    _, _, te = learning_curve(pipe, df[MORPH_FEATURES], y, groups=gc, cv=GroupKFold(n_splits=5),
                              train_sizes=[1.0], scoring="r2", n_jobs=-1, **kw)
    rows.append({"Variant": ad, "foldwise_mean_R2": float(te.mean())})
t = pd.DataFrame(rows)
t.to_csv(os.path.join(TAB_DIR, "129_same_configuration.csv"), index=False)
print(t.to_string(index=False))
ok1 = abs(t.iloc[0]["foldwise_mean_R2"] - dosya_fs) < 1e-9
ok2 = abs(t.iloc[2]["foldwise_mean_R2"] - dosya_lc) < 1e-9
print("0.448 reproduced:", ok1, "| 0.469 reproduced:", ok2)
sys.exit(0 if ok1 and ok2 else 1)
