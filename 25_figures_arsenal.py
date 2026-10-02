# -*- coding: utf-8 -*-
"""
25 — Figures for the expanded search (scripts 18-24).

Regenerates Figures 20-23 for the model that the expanded search selected, and
adds Figures 24-27 for the results that had no figure before: the selection
ladder, the strategy comparison, the exhaustive subset search and the conformal
prediction intervals.

House style: 400 dpi, white background, black text, English labels, the
Okabe-Ito colour-blind-safe palette, and no colour used to carry information
that is not also carried by position or label.
"""
import json
import os
import sys
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import FIG_DIR, MORPH_FEATURES, TAB_DIR

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "font.family": "DejaVu Sans", "font.size": 11,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlecolor": "black",
    "axes.labelsize": 11.5, "axes.labelcolor": "black",
    # The Editor was promised axis and tick labels of at least 11 pt; this script
    # was producing 10 pt, so the promise was not being kept for its figures.
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "xtick.color": "black", "ytick.color": "black", "text.color": "black",
    "legend.fontsize": 10, "axes.edgecolor": "black", "axes.linewidth": 1.0,
    "savefig.dpi": 400, "savefig.bbox": "tight",
})
OI = {"orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermilion": "#D55E00",
      "purple": "#CC79A7", "grey": "#999999"}


def save(fig, name):
    for ext in ("png", "tif"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=400,
                    bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"} if ext == "tif" else None)
    plt.close(fig)
    print("  wrote", name)


def T(name):
    return pd.read_csv(os.path.join(TAB_DIR, name))


def J(name):
    return json.load(open(os.path.join(TAB_DIR, name), encoding="utf-8"))


FINAL = J("23_final_model.json")
MODEL = FINAL["model"]
print("Figures for the final model:", MODEL)

# ===========================================================================
# Figure 20 — agreement diagnostics
# ===========================================================================
oof = T("23_oof_C_leaveonecellout.csv")
fx = T("23_fixed_models.csv")
row = fx[(fx.Model == MODEL) & (fx.Design == "Leave-one-cell-out (62 outer folds)")].iloc[0]

# the fixed-model out-of-fold predictions, recomputed here so the figure and the
# table cannot drift apart
from arsenal import GroupOut, build_frames, build_pool, evaluate
from common_setup import GENOTYPE, TARGET_EN, YEAR

_, CELL, _ = build_frames()
yc = CELL[TARGET_EN].values.astype(float)
g_cell = (CELL[YEAR].astype(str) + "_" + CELL[GENOTYPE].astype(str)).values
g_geno = CELL[GENOTYPE].astype(str).values
POOL = build_pool("cell")
met_loco, pred_loco = evaluate(POOL[MODEL], CELL, MORPH_FEATURES, "log", yc,
                               GroupOut(g_cell))
met_logo, pred_logo = evaluate(POOL[MODEL], CELL, MORPH_FEATURES, "log", yc,
                               GroupOut(g_geno))
season = CELL[YEAR].values
resid = yc - pred_loco

fig, ax = plt.subplots(2, 2, figsize=(12.4, 10.2))
a = ax[0, 0]
for s, col, mk in zip(np.unique(season), (OI["blue"], OI["orange"]), ("o", "s")):
    m = season == s
    a.scatter(yc[m], pred_loco[m], s=58, c=col, marker=mk, edgecolor="black",
              linewidth=0.6, alpha=0.9, label=f"Season {s}")
lim = [min(yc.min(), pred_loco.min()) * 0.92, max(yc.max(), pred_loco.max()) * 1.05]
a.plot(lim, lim, "k--", lw=1.2, label="1 : 1")
a.set_xlim(lim)
a.set_ylim(lim)
a.set_xlabel("Observed cell mean fresh plant weight (g)")
a.set_ylabel("Predicted (g)")
a.set_title(f"(a) Unseen cell: R² = {met_loco['pooled_R2']:.3f}, "
            f"MAE = {met_loco['pooled_MAE']:.2f} g")
a.legend(frameon=True, loc="upper left")
a.grid(alpha=0.25, ls=":")

a = ax[0, 1]
for s, col, mk in zip(np.unique(season), (OI["blue"], OI["orange"]), ("o", "s")):
    m = season == s
    a.scatter(yc[m], pred_logo[m], s=58, c=col, marker=mk, edgecolor="black",
              linewidth=0.6, alpha=0.9, label=f"Season {s}")
a.plot(lim, lim, "k--", lw=1.2)
a.set_xlim(lim)
a.set_ylim(lim)
a.set_xlabel("Observed cell mean fresh plant weight (g)")
a.set_ylabel("Predicted (g)")
a.set_title(f"(b) Unseen genotype: R² = {met_logo['pooled_R2']:.3f}, "
            f"MAE = {met_logo['pooled_MAE']:.2f} g")
a.legend(frameon=True, loc="upper left")
a.grid(alpha=0.25, ls=":")

a = ax[1, 0]
a.scatter(pred_loco, resid, s=58, c=OI["green"], edgecolor="black", linewidth=0.6,
          alpha=0.9)
a.axhline(0, color="black", lw=1.2)
a.set_xlabel("Predicted (g)")
a.set_ylabel("Residual, observed − predicted (g)")
a.set_title("(c) Residuals against the prediction")
a.grid(alpha=0.25, ls=":")

a = ax[1, 1]
a.hist(resid, bins=16, color=OI["sky"], edgecolor="black", linewidth=0.7)
a.axvline(0, color="black", lw=1.4)
a.axvline(float(np.mean(resid)), color=OI["vermilion"], lw=1.8, ls="--",
          label=f"mean = {np.mean(resid):+.2f} g")
a.set_xlabel("Residual (g)")
a.set_ylabel("Number of cells")
a.set_title("(d) Distribution of the residuals")
a.legend(frameon=True)
a.grid(axis="y", alpha=0.25, ls=":")

fig.suptitle(f"Final model — {MODEL} regression of log fresh plant weight on the eight "
             f"measured trait means", fontsize=13.5, fontweight="bold", y=0.995)
fig.tight_layout()
save(fig, "Figure_20_final_model_diagnostics")

# ===========================================================================
# Figure 21 — the four validation designs
# ===========================================================================
order = ["Grouped five-fold over cells", "Leave-one-cell-out (62 outer folds)",
         "Leave-one-genotype-out (31 outer folds)", "Leave-one-year-out (2 outer folds)"]
short = ["Grouped\nfive-fold", "Unseen\ncell", "Unseen\ngenotype", "Unseen\nseason"]
sub = fx[fx.Model == MODEL].set_index("Design").reindex(order)

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.9))
for a, col, lab, colr in ((axes[0], "pooled_R2", "R²", OI["blue"]),
                          (axes[1], "pooled_MAE", "Mean absolute error (g)", OI["orange"]),
                          (axes[2], "pooled_MAPE", "Mean absolute percentage error (%)",
                           OI["green"])):
    v = sub[col].values
    a.bar(range(4), v, color=colr, edgecolor="black", width=0.62)
    for i, val in enumerate(v):
        a.text(i, val + (0.012 * max(v)), f"{val:.3f}" if col == "pooled_R2" else f"{val:.1f}",
               ha="center", va="bottom", fontsize=11, fontweight="bold")
    a.set_xticks(range(4))
    a.set_xticklabels(short)
    a.set_ylabel(lab)
    a.grid(axis="y", alpha=0.25, ls=":")
    a.set_ylim(0, max(v) * 1.18)
axes[0].set_title("(a) Explained variance")
axes[1].set_title("(b) Absolute error")
axes[2].set_title("(c) Relative error")
fig.suptitle("Final model under four leakage-free validation designs",
             fontsize=13.5, fontweight="bold", y=1.02)
fig.tight_layout()
save(fig, "Figure_21_final_designs")

# ===========================================================================
# Figure 22 — method families at both analysis levels, one protocol
# ===========================================================================
fam = pd.read_csv(os.path.join(TAB_DIR, "24_family_comparison.csv"), index_col=0)
fam = fam.sort_values("Cell level (62 cells)", ascending=False)
labels = [i.replace(" and ", " and\n").replace("Regularised linear", "Regularised\nlinear")
            .replace("Latent variable", "Latent\nvariable")
            .replace("Tree ensembles", "Tree\nensembles")
            .replace("Nearest neighbours", "Nearest\nneighbours")
            .replace("Additive splines", "Additive\nsplines")
            .replace("Neural network", "Neural\nnetwork") for i in fam.index]
x = np.arange(len(fam))
fig, ax = plt.subplots(figsize=(12.6, 6.0))
ax.bar(x - 0.2, fam["Plant level (620 records)"], width=0.4, color=OI["sky"],
       edgecolor="black", label="Plant level (620 records)")
ax.bar(x + 0.2, fam["Cell level (62 cells)"], width=0.4, color=OI["green"],
       edgecolor="black", label="Cell level (62 cells)")
for i, (p, c) in enumerate(zip(fam["Plant level (620 records)"],
                               fam["Cell level (62 cells)"])):
    ax.text(i - 0.2, p + 0.008, f"{p:.3f}", ha="center", va="bottom", fontsize=9.5)
    ax.text(i + 0.2, c + 0.008, f"{c:.3f}", ha="center", va="bottom", fontsize=9.5,
            fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("Best R² achieved by the family")
ax.set_ylim(0, 0.80)
ax.legend(frameon=True, loc="upper right")
ax.grid(axis="y", alpha=0.25, ls=":")
ax.set_title("Best performance by method family at two analysis levels\n"
             "(47 estimators, identical evaluator, folds blocked on the "
             "genotype × year cell)")
fig.tight_layout()
save(fig, "Figure_22_method_families")

# ===========================================================================
# Figure 23 — the fitted coefficients
# ===========================================================================
co = T("23_final_coefficients.csv").copy()
co["abs"] = co["Coefficient (standardised)"].abs()
co = co.sort_values("Coefficient (standardised)")
cols = [OI["vermilion"] if v < 0 else OI["blue"]
        for v in co["Coefficient (standardised)"]]
fig, ax = plt.subplots(figsize=(10.4, 6.0))
ax.barh(range(len(co)), co["Coefficient (standardised)"], color=cols,
        edgecolor="black")
ax.set_yticks(range(len(co)))
ax.set_yticklabels(co["Trait"])
ax.axvline(0, color="black", lw=1.2)
for i, v in enumerate(co["Coefficient (standardised)"]):
    ax.text(v + (0.004 if v >= 0 else -0.004), i, f"{v:+.3f}",
            va="center", ha="left" if v >= 0 else "right", fontsize=10,
            fontweight="bold")
ax.set_xlabel("Coefficient on log fresh plant weight, per standard deviation of the trait")
ax.set_title(f"Fitted {MODEL} regression\n"
             "(partial effects; the traits are strongly intercorrelated, so a "
             "negative sign is not a negative marginal association)")
ax.grid(axis="x", alpha=0.25, ls=":")
fig.tight_layout()
save(fig, "Figure_23_final_coefficients")

# ===========================================================================
# Figure 24 — the selection ladder
# ===========================================================================
lad = T("23_ladder.csv")
lad = lad[lad["Design"].isin(["Leave-one-cell-out (62 outer folds)",
                              "Leave-one-genotype-out (31 outer folds)"])]
fixed = lad[(lad["Selection space"] == "None (model fixed in advance)")
            & (lad.get("Model") == MODEL)]
other = lad[lad["Selection space"] != "None (model fixed in advance)"]
lad2 = pd.concat([fixed, other], ignore_index=True)
piv = lad2.pivot_table(index=["Selection space", "n_candidates"], columns="Design",
                       values="pooled_R2").reset_index()
piv = piv.sort_values("n_candidates")
LC = "Leave-one-cell-out (62 outer folds)"
LG = "Leave-one-genotype-out (31 outer folds)"

# the two 88/90-candidate spaces sit on top of each other on a logarithmic
# axis, so the ladder is drawn against rank with the size shown in the label
piv = piv.reset_index(drop=True)
LAB = {1: "no selection\n(model fixed\nin advance)",
       8: "SPACE C\n8 candidates",
       88: "space of the\nfirst revision\n88 candidates",
       90: "SPACE B\n90 candidates",
       540: "SPACE A\n540 candidates"}
xpos = np.arange(len(piv))
fig, ax = plt.subplots(figsize=(11.6, 6.4))
ax.plot(xpos, piv[LC], "o-", color=OI["blue"], lw=2.2, ms=11,
        markeredgecolor="black", label="Unseen cell")
ax.plot(xpos, piv[LG], "s--", color=OI["orange"], lw=2.2, ms=11,
        markeredgecolor="black", label="Unseen genotype")
for i, r in piv.iterrows():
    ax.annotate(f"{r[LC]:.3f}", (i, r[LC]), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=10, fontweight="bold",
                color=OI["blue"])
    ax.annotate(f"{r[LG]:.3f}", (i, r[LG]), textcoords="offset points",
                xytext=(0, -20), ha="center", fontsize=10, fontweight="bold",
                color=OI["orange"])
ax.set_xticks(xpos)
ax.set_xticklabels([LAB.get(int(v), str(int(v))) for v in piv["n_candidates"]],
                   fontsize=9.5)
ax.set_xlim(-0.45, len(piv) - 0.55)
ax.set_xlabel("How many configurations the selection procedure was allowed to consider")
ax.set_ylabel("Honestly validated R²")
ax.set_title("Validated performance against the size of the search\n"
             "(the selection is inside the validation loop at every point except "
             "the first, where the model is fixed in advance)")
ax.grid(alpha=0.25, ls=":")
ax.legend(frameon=True, loc="lower left")
fig.tight_layout()
save(fig, "Figure_24_selection_ladder")

# ===========================================================================
# Figure 25 — strategies that were tried
# ===========================================================================
hy = T("19_hybrid.csv")
hy = hy[(hy.Status == "ok") & (hy.Design == "LOCO")]
# the reference is the best direct fit found anywhere in the script-18 sweep,
# not merely the best of the ten base configurations that script 19 started from
sl = T("18_sweep_shortlist.csv")
sl = sl[(sl.Status == "ok") & (sl.Design == "LOCO leave-one-cell-out (62)")]
direct = float(max(hy[hy.Strategy == "direct"]["pooled_R2"].max(),
                   sl["pooled_R2"].max()))
bu = hy[hy.Strategy == "bottomup"]["pooled_R2"].max()
bupm = hy[hy.Strategy == "bottomup_pm"]["pooled_R2"].max()
aug = (hy[hy.Strategy.str.startswith("aug:")].groupby("Scheme")["pooled_R2"].max()
       .sort_values(ascending=False))
bl = T("19_blends.csv")
bl = bl[(bl.Status == "ok") & (bl.Design == "LOCO")]
blend = bl["pooled_R2"].max()
wt = T("20_weighting.csv")
wt = wt[(wt.Status == "ok") & (wt.Design == "LOCO")]
wbest = wt[wt.Weighting != "unweighted"]["pooled_R2"].max()
su = T("22_summaries.csv")
su = su[(su.Status == "ok") & (su.Design == "LOCO")]
sbest = su[su.Summary != "mean"]["pooled_R2"].max()
gl = T("20_genotype_level.csv")
gl = gl[(gl.Status == "ok") & (gl.Design == "LOGO (31 folds)")]
gbest = gl["pooled_R2"].max()

names = (["Direct fit on cell means\n(the approach adopted)"]
         + [f"Augmentation: {s}" for s in aug.index]
         + ["Blend of several models",
            "Bottom-up from plants\n(mean of predictions)",
            "Bottom-up from plants\n(prediction of means)",
            "Weighting by cell precision",
            "Alternative predictor summary",
            "Genotype as the unit (31)"])
vals = ([direct] + list(aug.values) + [blend, bu, bupm, wbest, sbest, gbest])
o = np.argsort(vals)[::-1]
names = [names[i] for i in o]
vals = [vals[i] for i in o]
cols = [OI["green"] if n.startswith("Direct") else OI["grey"] for n in names]

fig, ax = plt.subplots(figsize=(11.4, 7.4))
ax.barh(range(len(vals))[::-1], vals, color=cols, edgecolor="black")
ax.set_yticks(range(len(vals))[::-1])
ax.set_yticklabels(names, fontsize=9.5)
for i, v in enumerate(vals):
    ax.text(v + 0.006, len(vals) - 1 - i, f"{v:.3f}", va="center", fontsize=10,
            fontweight="bold" if names[i].startswith("Direct") else "normal")
ax.axvline(direct, color=OI["green"], ls="--", lw=1.5)
ax.set_xlabel("Best R² for an unseen genotype × year cell")
ax.set_xlim(0, max(vals) * 1.14)
_stab_sd = J("23_final_model.json")["stability"]["R2_sd"]
_gap = max(vals) - direct
ax.set_title("Every strategy that was tried, scored under the same "
             "leave-one-cell-out protocol\n"
             f"(the best elaboration exceeded a direct fit by {_gap:+.3f}, against "
             f"a standard deviation of {_stab_sd:.3f} in the estimate itself)")
ax.grid(axis="x", alpha=0.25, ls=":")
fig.tight_layout()
save(fig, "Figure_25_strategies")

# ===========================================================================
# Figure 26 — the exhaustive subset search
# ===========================================================================
subs = T("20_subsets.csv")
subs = subs[subs.Status == "ok"]
nested = J("20_subset_nested.json")
byk = subs.groupby("k")["pooled_R2"].max()
allk = subs.groupby("k")["pooled_R2"]

fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6))
a = axes[0]
a.plot(byk.index, byk.values, "o-", color=OI["blue"], lw=2.2, ms=10,
       markeredgecolor="black", label="Best subset of this size")
a.plot(allk.median().index, allk.median().values, "s--", color=OI["grey"], lw=1.6,
       ms=7, markeredgecolor="black", label="Median subset of this size")
a.set_xlabel("Number of traits in the model")
a.set_ylabel("R² (grouped five-fold over cells)")
a.set_title("(a) How many traits are needed")
a.grid(alpha=0.25, ls=":")
a.legend(frameon=True, loc="lower right")
for k, v in byk.items():
    a.annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 9),
               ha="center", fontsize=9)

a = axes[1]
sel = nested["nested"]["LOCO"]["trait_selection_share_pct"]
sel = dict(sorted(sel.items(), key=lambda kv: kv[1]))
a.barh(range(len(sel)), list(sel.values()),
       color=[OI["green"] if v >= 50 else OI["grey"] for v in sel.values()],
       edgecolor="black")
a.set_yticks(range(len(sel)))
a.set_yticklabels(list(sel), fontsize=10)
a.set_xlabel("Percentage of the 62 outer folds in which the trait was selected")
a.set_xlim(0, 108)
for i, v in enumerate(sel.values()):
    a.text(v + 1.5, i, f"{v:.0f}", va="center", fontsize=10)
a.set_title("(b) Which traits survive selection inside the loop")
a.grid(axis="x", alpha=0.25, ls=":")
fig.tight_layout()
save(fig, "Figure_26_subset_search")

# ===========================================================================
# Figure 27 — prediction intervals
# ===========================================================================
# The half-width is the empirical quantile of the 62 leave-one-cell-out relative
# residuals. The share of cells inside the band is computed from the same residuals
# and is therefore fixed by the quantile level (49, 55 and 58 of 62), so it is not a
# check of calibration: it is not printed in the title and the former calibration
# panel is not drawn.
cf = T("24_conformal.csv")
order_i = np.argsort(pred_loco)
p_sorted = pred_loco[order_i]
y_sorted = yc[order_i]
q90 = float(cf[cf["Nominal coverage %"] == 90]["Relative half-width %"].iloc[0]) / 100
rank = np.arange(1, len(p_sorted) + 1)

fig, a = plt.subplots(figsize=(9.8, 5.8))
a.fill_between(rank, p_sorted * (1 - q90), p_sorted * (1 + q90), color=OI["sky"],
               alpha=0.45, label=f"Nominal 90 % interval (prediction ± {100 * q90:.2f} %)")
a.plot(rank, p_sorted, color=OI["blue"], lw=2.0, label="Prediction")
a.scatter(rank, y_sorted, s=34, c="black", zorder=3, label="Observed cell mean")
a.set_xlabel("Genotype × year cells, ordered by prediction")
# R2 (t25, V5-08): the index runs 1-62; the default ticks left the first and last cell
# outside the labelled range.
a.set_xticks([1, 10, 20, 30, 40, 50, len(p_sorted)])
a.set_ylabel("Fresh plant weight (g)")
a.set_title("Nominal 90 % prediction intervals from the leave-one-cell-out residuals")
a.legend(frameon=True, loc="upper left", fontsize=10)
a.grid(alpha=0.25, ls=":")
fig.tight_layout()
save(fig, "Figure_27_conformal_intervals")

print("\nAll figures written to", FIG_DIR)
