import io
import json
import os
import sys
from collections import defaultdict

import pandas as pd

KOK = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, KOK)
os.chdir(KOK)
sys.stdout.reconfigure(encoding="utf-8")
from common_setup import GENOTYPE, MORPH_FEATURES, TAB_DIR, YEAR, load_data

df, _, _, _, _ = load_data()
h = df[GENOTYPE].astype(str) + "/" + df[YEAR].astype(str)
counts = defaultdict(lambda: {"Combinations": 0, "Sharing": 0})
total = sharing = 0
for f in MORPH_FEATURES:
    diz = defaultdict(list)
    for hh, g in df.assign(_h=h).groupby("_h", sort=False):
        if len(g) >= 9:
            diz[tuple(g.iloc[:9][f].round(6).tolist())].append(hh)
    for key, cells in diz.items():
        n_ayri = len(set(key))
        counts[n_ayri]["Combinations"] += len(cells)
        total += len(cells)
        if len(cells) > 1:
            counts[n_ayri]["Sharing"] += len(cells)
            sharing += len(cells)

J = json.load(io.open(os.path.join(TAB_DIR, "26_audit_summary.json"), encoding="utf-8"))
if total != int(J["identical_first9_morph_cells"]) or sharing != int(J["identical_first9_morph_sharing"]):
    raise SystemExit("131: could not reproduce the totals of 26_audit_summary.json (%d/%d)"
                     % (total, sharing))

t = pd.DataFrame([{"DistinctValues": k, "Combinations": v["Combinations"], "Sharing": v["Sharing"]}
                  for k, v in sorted(counts.items())])
yol = os.path.join(TAB_DIR, "131_repeated_sequences.csv")
t.to_csv(yol, index=False)
print(t.to_string(index=False))
print("trait x cell combinations:", total, "| sequence repeated:", sharing)
print("repeated sequences carrying eight or nine distinct values:",
      int(t[t.DistinctValues >= 8]["Sharing"].sum()))
print("written:", yol)
