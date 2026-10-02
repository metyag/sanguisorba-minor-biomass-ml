# -*- coding: utf-8 -*-
"""
26 — Data audit: every structural anomaly in the data file, cell by cell.

This script does not model anything. It looks at the 620 records themselves and
tests, for each genotype x year cell, a battery of signatures that independent
measurement cannot produce. For every signature it reports how many cells show
it and which ones, and it writes an Excel workbook in which the raw data are
laid out cell by cell with the flags attached, so that the finding can be
checked by eye against the field books.

Signatures tested, per cell and per trait:

  MONOTONE          records 1-9 are ordered from smallest to largest.
                    Probability for nine independent measurements: 1/9!
  ARITHMETIC        records 1-9 increase by a constant step.
  CONSECUTIVE PAIRS records 1-10 form five consecutive pairs, each pair summing
                    to the same total. This is what generating ten numbers
                    around a fixed mean produces, and it cannot arise from
                    measurement.
  SYMMETRIC PAIRS   the same, pairing first with last, second with ninth, etc.
  INTERLEAVED       odd-numbered records increase while even-numbered records
                    decrease, i.e. the series alternates inwards from the two
                    extremes towards the mean.
  RECORD 10         the tenth record sits far outside the first nine.

Biological plausibility is checked separately: leaflet width larger than
leaflet length, leaflet longer than the leaf that bears it, and values outside
the range the literature reports for the species.

Outputs (results/tables/):
  26_cell_flags.csv        one row per cell x trait, with every flag
  26_cell_summary.csv      one row per cell, how many traits are flagged
  26_signature_counts.csv  how many cells show each signature
  26_implausible.csv       records that are biologically impossible
  26_record10_identity.csv what the tenth record equals, if anything
  26_DATA_AUDIT.xlsx       the raw data, cell by cell, with flags
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (DATA_PATH, GENOTYPE, MORPH_FEATURES, TAB_DIR,
                          TARGET_EN, YEAR, load_data)

TOL = 1e-6
ALL_TRAITS = MORPH_FEATURES + [TARGET_EN]

df, _, _, _, _ = load_data()
df = df.copy()
df["Record"] = df.groupby([YEAR, GENOTYPE]).cumcount() + 1
df["Cell"] = df[YEAR].astype(str) + "_" + df[GENOTYPE].astype(str)

print("=" * 100)
print("26 — DATA AUDIT OF %s" % os.path.basename(DATA_PATH))
print("=" * 100)
print(f"records {len(df)}   cells {df.Cell.nunique()}   "
      f"records per cell {df.groupby('Cell').size().unique()}")


# ===========================================================================
# signature tests
# ===========================================================================
def const(a, tol=1e-6):
    a = np.asarray(a, dtype=float)
    return bool(len(a) > 1 and np.ptp(a) <= tol * max(1.0, np.abs(a).mean()))


def sig_monotone(v9):
    return bool(np.all(np.diff(v9) >= -TOL))


def sig_arithmetic(v9):
    # A constant NON-ZERO step. This is deliberately stricter than the
    # 'Records 1-9 arithmetic' column of 18_integrity.csv, which counts any
    # constant step and so also counts the cells whose nine values are
    # identical (step zero). Table 8 of the manuscript reports the column of
    # script 18; the counts here are therefore smaller for some traits.
    d = np.diff(v9)
    return bool(const(d) and abs(d[0]) > TOL)


def sig_consecutive_pairs(v10):
    """v[0]+v[1] == v[2]+v[3] == ... : five consecutive pairs with one total."""
    s = [v10[2 * i] + v10[2 * i + 1] for i in range(5)]
    return bool(const(s) and np.ptp(v10) > TOL)


def sig_symmetric_pairs(v10):
    """v[0]+v[9] == v[1]+v[8] == ... : the series is symmetric about its mean."""
    s = [v10[i] + v10[9 - i] for i in range(5)]
    return bool(const(s) and np.ptp(v10) > TOL)


def sig_interleaved(v10):
    """odd positions rise, even positions fall — alternating inwards."""
    odd, even = v10[0::2], v10[1::2]
    return bool(np.all(np.diff(odd) >= -TOL) and np.all(np.diff(even) <= TOL)
                and np.ptp(v10) > TOL and not np.all(np.diff(v10) >= -TOL))


def sig_lattice(v10):
    """
    The ten values, sorted, lie on a single arithmetic lattice: every gap
    between consecutive sorted values is the same step, or twice it. The
    'twice it' case is what happens when ten values are laid out symmetrically
    about a mean and the centre point of the lattice is skipped.

    Ten independent measurements of a biological quantity do not do this.
    """
    s = np.sort(v10)
    d = np.diff(s)
    d = d[d > TOL]
    if len(d) < 8:
        return False, np.nan, 0
    step = d.min()
    if step <= TOL:
        return False, np.nan, 0
    k = d / step
    ok = np.all(np.abs(k - np.round(k)) < 1e-6) and np.max(np.round(k)) <= 2
    return bool(ok), float(step), int(np.sum(np.round(k) == 2))


rows = []
for cell, sub in df.groupby("Cell"):
    sub = sub.sort_values("Record")
    for t in ALL_TRAITS:
        v10 = sub[t].values.astype(float)
        v9 = v10[:9]
        sd9 = v9.std(ddof=1)
        z10 = (v10[9] - v9.mean()) / sd9 if sd9 > 0 else np.nan
        lat, lat_step, lat_gaps = sig_lattice(v10)
        rows.append({
            "lattice": lat, "lattice_step": lat_step, "lattice_gaps": lat_gaps,
            "Cell": cell, "Year": sub[YEAR].iloc[0], "Genotype": sub[GENOTYPE].iloc[0],
            "Trait": t,
            "monotone_1_9": sig_monotone(v9),
            "arithmetic_1_9": sig_arithmetic(v9),
            "consecutive_pairs": sig_consecutive_pairs(v10),
            "symmetric_pairs": sig_symmetric_pairs(v10),
            "interleaved": sig_interleaved(v10),
            "constant": const(v10),
            "record10_z": z10,
            "record10_extreme": bool(np.isfinite(z10) and abs(z10) > 3),
            "pair_total": (v10[0] + v10[1]) if sig_consecutive_pairs(v10) else np.nan,
            "cell_mean": float(v10.mean()),
            "values": " ".join(f"{x:g}" for x in v10),
        })
flags = pd.DataFrame(rows)
flags.to_csv(os.path.join(TAB_DIR, "26_cell_flags.csv"), index=False)

SIGS = ["monotone_1_9", "arithmetic_1_9", "consecutive_pairs", "symmetric_pairs",
        "interleaved", "lattice", "constant", "record10_extreme"]
n_cells = flags.Cell.nunique()

print("\n" + "=" * 100)
print("A. HOW MANY OF THE 62 CELLS SHOW EACH SIGNATURE, BY TRAIT")
print("=" * 100)
piv = flags.pivot_table(index="Trait", columns=None, values=SIGS, aggfunc="sum")
piv = piv[SIGS].reindex(ALL_TRAITS).astype(int)
piv.columns = ["monotone", "arithmetic", "consec.pairs", "symm.pairs",
               "interleaved", "lattice", "constant", "rec10 |z|>3"]
print(piv.to_string())
print("\n  'lattice' = the ten sorted values sit on one arithmetic lattice, every")
print("  gap equal to a single step or exactly twice it. This is the signature of")
print("  values laid out around a target mean rather than measured.")

print("\n" + "=" * 100)
print("B. CELLS IN WHICH THE RESPONSE ITSELF IS CONSTRUCTED")
print("=" * 100)
w = flags[flags.Trait == TARGET_EN]
any_pair = w[w.consecutive_pairs | w.symmetric_pairs | w.interleaved | w.lattice]
print(f"  green weight shows at least one construction signature in "
      f"{len(any_pair)} of {n_cells} cells")
print(f"    on a single arithmetic lattice          : {int(w.lattice.sum())}")
print(f"    symmetric pairs with a constant total   : {int(w.symmetric_pairs.sum())}")
print(f"    consecutive pairs with a constant total : {int(w.consecutive_pairs.sum())}")
print(f"    interleaved (alternating inwards)       : {int(w.interleaved.sum())}")
print(f"    strictly ordered                        : {int(w.monotone_1_9.sum())}")
none_of = ~(w.consecutive_pairs | w.symmetric_pairs | w.interleaved | w.lattice
            | w.monotone_1_9)
print(f"    none of the above                       : {int(none_of.sum())}")

if len(any_pair):
    print("\n  Examples (green weight, the ten records in file order):")
    for _, r in any_pair.head(10).iterrows():
        v = [float(x) for x in r["values"].split()]
        kinds = [k for k, on in (("lattice", r.lattice),
                                 ("symmetric pairs", r.symmetric_pairs),
                                 ("consecutive pairs", r.consecutive_pairs),
                                 ("interleaved", r.interleaved)) if on]
        print(f"    year {r.Year} genotype {r.Genotype:>3} [{', '.join(kinds)}]")
        print("        file order : " + "  ".join(f"{x:g}" for x in v))
        s = sorted(v)
        print("        sorted     : " + "  ".join(f"{x:g}" for x in s))
        print("        gaps       : " + "  ".join(f"{b - a:g}" for a, b in zip(s, s[1:]))
              + f"      mean = {r.cell_mean:g}")
        if r.consecutive_pairs:
            tot = [v[2 * i] + v[2 * i + 1] for i in range(5)]
            print("        every consecutive pair sums to " + f"{tot[0]:g}"
                  + f", exactly twice the mean")
        elif r.symmetric_pairs:
            tot = [v[i] + v[9 - i] for i in range(5)]
            print("        every symmetric pair sums to " + f"{tot[0]:g}"
                  + f", exactly twice the mean")

# ===========================================================================
# C. what is record 10?
# ===========================================================================
print("\n" + "=" * 100)
print("C. WHAT IS THE TENTH RECORD?  (tested against the first nine)")
print("=" * 100)
id_rows = []
for t in ALL_TRAITS:
    hits = {"= max of 1-9": 0, "= 2 x mean of 1-9": 0, "= sum/  something": 0,
            "> max of 1-9": 0, "< min of 1-9": 0}
    ratios = []
    for cell, sub in df.groupby("Cell"):
        sub = sub.sort_values("Record")
        v = sub[t].values.astype(float)
        v9, v10 = v[:9], v[9]
        if abs(v10 - v9.max()) < TOL:
            hits["= max of 1-9"] += 1
        if abs(v10 - 2 * v9.mean()) < 1e-3 * max(1.0, abs(v10)):
            hits["= 2 x mean of 1-9"] += 1
        if v10 > v9.max() + TOL:
            hits["> max of 1-9"] += 1
        if v10 < v9.min() - TOL:
            hits["< min of 1-9"] += 1
        if v9.mean() > 0:
            ratios.append(v10 / v9.mean())
    r = np.array(ratios)
    id_rows.append({"Trait": t, **hits,
                    "median record10 / mean(1-9)": float(np.median(r)),
                    "IQR of that ratio": float(np.percentile(r, 75) - np.percentile(r, 25))})
ident = pd.DataFrame(id_rows)
ident.to_csv(os.path.join(TAB_DIR, "26_record10_identity.csv"), index=False)
print(ident.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
print("\n  No single arithmetic identity accounts for the tenth record; it is")
print("  consistently larger than the other nine in the predictors and ordinary")
print("  in the response, which is the pattern reported in Section 3.19.")

# ===========================================================================
# D. biological plausibility
# ===========================================================================
print("\n" + "=" * 100)
print("D. BIOLOGICALLY IMPOSSIBLE OR IMPLAUSIBLE RECORDS")
print("=" * 100)
imp = []
for _, r in df.iterrows():
    reasons = []
    if r["Leaflet width"] > r["Leaflet length"]:
        reasons.append(f"leaflet width ({r['Leaflet width']:g}) exceeds leaflet "
                       f"length ({r['Leaflet length']:g})")
    if r["Leaflet length"] > r["Leaf length"]:
        reasons.append(f"leaflet longer ({r['Leaflet length']:g}) than the leaf that "
                       f"bears it ({r['Leaf length']:g})")
    if r["Leaflet width"] < 0.15:
        reasons.append(f"leaflet width {r['Leaflet width']:g} cm = "
                       f"{10 * r['Leaflet width']:.0f} mm, below any reported value")
    if r["Number of leaflets per leaf"] < 3:
        reasons.append(f"{r['Number of leaflets per leaf']:g} leaflets per leaf "
                       f"(Sanguisorba minor leaves are imparipinnate, 7-25)")
    if reasons:
        imp.append({"Year": r[YEAR], "Genotype": r[GENOTYPE], "Record": r["Record"],
                    "Cell": r["Cell"],
                    **{t: r[t] for t in ALL_TRAITS},
                    "Problem": "; ".join(reasons)})
impdf = pd.DataFrame(imp)
impdf.to_csv(os.path.join(TAB_DIR, "26_implausible.csv"), index=False)
print(f"  records with at least one impossible or implausible value: "
      f"{len(impdf)} of {len(df)}")
if len(impdf):
    cnt = {}
    for _, r in impdf.iterrows():
        for part in r["Problem"].split("; "):
            if "exceeds" in part:
                key = "leaflet WIDER than it is LONG"
            elif "longer" in part:
                key = "leaflet LONGER than the leaf bearing it"
            elif "below any reported" in part:
                key = "leaflet width under 1.5 mm"
            else:
                key = "fewer than 3 leaflets per leaf"
            cnt[key] = cnt.get(key, 0) + 1
    for k, v in sorted(cnt.items(), key=lambda kv: -kv[1]):
        print(f"    {k:44} {v:4} records")
    print("\n  Worst cases (widest leaflet relative to its length):")
    tmp = impdf.copy()
    tmp["ratio"] = tmp["Leaflet width"] / tmp["Leaflet length"]
    show = tmp.sort_values("ratio", ascending=False).head(10)[
        ["Year", "Genotype", "Record", "Leaf length", "Leaflet length",
         "Leaflet width", "ratio"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    print("\n  Of the flagged records, how many are the tenth of their cell:")
    print(f"    {int((impdf['Record'] == 10).sum())} of {len(impdf)}")

# ===========================================================================
# E. duplicated records
# ===========================================================================
print("\n" + "=" * 100)
print("E. DUPLICATED RECORDS")
print("=" * 100)
dup_full = df.duplicated(subset=ALL_TRAITS, keep=False)
dup_traits = df.duplicated(subset=MORPH_FEATURES, keep=False)
print(f"  records identical in all nine columns          : {int(dup_full.sum())}")
print(f"  records identical in the eight trait columns   : {int(dup_traits.sum())}")
if dup_traits.sum():
    g = df[dup_traits].groupby(MORPH_FEATURES).agg(
        n=("Cell", "size"), cells=("Cell", lambda s: ", ".join(sorted(set(s)))))
    g = g[g.n > 1].sort_values("n", ascending=False)
    print(f"  distinct trait vectors that repeat             : {len(g)}")
    print("\n  The most repeated trait vectors and the cells they appear in:")
    print(g.head(8)[["n", "cells"]].to_string(max_colwidth=70))

# ===========================================================================
# F. per-cell summary
# ===========================================================================
summ = (flags.groupby(["Cell", "Year", "Genotype"])[SIGS].sum().reset_index())
summ["traits_flagged"] = flags.groupby(["Cell", "Year", "Genotype"]).apply(
    lambda g: int((g[["monotone_1_9", "arithmetic_1_9", "consecutive_pairs",
                      "symmetric_pairs", "interleaved"]].any(axis=1)).sum())).values
summ = summ.sort_values("traits_flagged", ascending=False)
summ.to_csv(os.path.join(TAB_DIR, "26_cell_summary.csv"), index=False)

print("\n" + "=" * 100)
print("F. HOW MANY OF THE NINE COLUMNS ARE FLAGGED IN EACH CELL")
print("=" * 100)
hist = summ.traits_flagged.value_counts().sort_index(ascending=False)
for k, v in hist.items():
    print(f"    {k} of 9 columns flagged : {v:3} cells")
clean = summ[summ.traits_flagged == 0]
print(f"\n  cells with no structural signature at all: {len(clean)} of {n_cells}")

SIG_LABEL = {
    "monotone_1_9": "records 1-9 ordered smallest to largest",
    "arithmetic_1_9": "records 1-9 with a constant step",
    "consecutive_pairs": "five consecutive pairs with one total",
    "symmetric_pairs": "symmetric pairs with one total",
    "interleaved": "interleaved around the mean",
    "lattice": "all ten values on one arithmetic lattice",
    "constant": "all ten values identical",
    "record10_extreme": "record 10 more than 3 SD from the rest",
}
counts = pd.DataFrame({
    "Signature": [SIG_LABEL[s] for s in SIGS],
    "Trait-cells flagged (of %d)" % (n_cells * len(ALL_TRAITS)):
        [int(flags[s].sum()) for s in SIGS],
})
counts.to_csv(os.path.join(TAB_DIR, "26_signature_counts.csv"), index=False)
print("\n" + counts.to_string(index=False))

# ===========================================================================
# G. Excel workbook for inspection by eye
# ===========================================================================
xl = os.path.join(TAB_DIR, "26_DATA_AUDIT.xlsx")
with pd.ExcelWriter(xl, engine="openpyxl") as xw:
    out = df[[YEAR, GENOTYPE, "Record"] + ALL_TRAITS].copy()
    out.to_excel(xw, sheet_name="raw data", index=False)
    flags.drop(columns=["values"]).to_excel(xw, sheet_name="flags by cell x trait",
                                            index=False)
    summ.to_excel(xw, sheet_name="flags by cell", index=False)
    counts.to_excel(xw, sheet_name="signature counts", index=False)
    ident.to_excel(xw, sheet_name="what is record 10", index=False)
    if len(impdf):
        impdf.to_excel(xw, sheet_name="implausible records", index=False)
    wt = flags[flags.Trait == TARGET_EN][
        ["Year", "Genotype", "consecutive_pairs", "symmetric_pairs", "interleaved",
         "monotone_1_9", "pair_total", "cell_mean", "values"]]
    wt.to_excel(xw, sheet_name="green weight, cell by cell", index=False)
print("\nWorkbook written:", xl)

# ---------------------------------------------------------------------------
# IDENTICAL FIRST-NINE SEQUENCES IN SEPARATE CELLS
# ---------------------------------------------------------------------------
# Ordering is an arrangement WITHIN a cell and does not change the cell mean.
# Two SEPARATE plots carrying the same nine individual values is a different
# matter: independent field measurement does not produce that. It is stronger
# evidence than the ordering for the claim that the effective sample is 62 and
# not 620, and Section 3.4 reads it from here rather than restating it by hand.
from collections import defaultdict as _dd

_dup = {}
_h = df[GENOTYPE].astype(str) + "/" + df[YEAR].astype(str)
for _f in ALL_TRAITS:
    _diz = _dd(list)
    for _hh, _g in df.assign(_h=_h).groupby("_h", sort=False):
        if len(_g) >= 9:
            _diz[tuple(_g.iloc[:9][_f].round(6).tolist())].append(_hh)
    _dup[_f] = {
        "cells": int(sum(len(v) for v in _diz.values())),
        "distinct_sequences": int(len(_diz)),
        "cells_sharing": int(sum(len(v) for v in _diz.values() if len(v) > 1)),
        "largest_group": int(max((len(v) for v in _diz.values()), default=0)),
    }
_morph_cells = sum(_dup[f]["cells"] for f in MORPH_FEATURES)
_morph_share = sum(_dup[f]["cells_sharing"] for f in MORPH_FEATURES)

summary = {
    "n_records": int(len(df)), "n_cells": int(n_cells),
    "identical_first9_by_trait": _dup,
    "identical_first9_morph_cells": int(_morph_cells),
    "identical_first9_morph_sharing": int(_morph_share),
    "identical_first9_target_sharing": int(_dup[TARGET_EN]["cells_sharing"]),
    "identical_first9_target_distinct": int(_dup[TARGET_EN]["distinct_sequences"]),
    "signature_counts": {s: int(flags[s].sum()) for s in SIGS},
    "weight_constructed_cells": int(len(any_pair)),
    "weight_consecutive_pairs": int(w.consecutive_pairs.sum()),
    "weight_symmetric_pairs": int(w.symmetric_pairs.sum()),
    "weight_interleaved": int(w.interleaved.sum()),
    "weight_monotone": int(w.monotone_1_9.sum()),
    "cells_with_no_signature": int(len(clean)),
    "implausible_records": int(len(impdf)),
    "duplicate_trait_rows": int(dup_traits.sum()),
}
with open(os.path.join(TAB_DIR, "26_audit_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\nDone.")
