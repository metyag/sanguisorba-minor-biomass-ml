# -*- coding: utf-8 -*-
"""
05b — The figures that 05_figures.py had not yet produced.

Split out because the partial-dependence computation in scikit-learn falls back
to the single-threaded brute-force method for Pipeline estimators; it is run here
with n_jobs=-1 and a coarser grid (40 points), which is more than enough for a
smooth curve and is roughly an order of magnitude faster.

Produces:
  Figure_12_partial_dependence_morph
  Figure_12b_partial_dependence_full
  Figure_13_learning_curves          (+ 05_learning_curves.json)
  Figure_14_pca_scree_biplot
  Figure_15_pca_contribution
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.model_selection import KFold, GroupKFold, learning_curve
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (load_data, make_pipeline, SEED, FIG_DIR, TAB_DIR,
                          FULL_FEATURES, MORPH_FEATURES, TARGET_EN, YEAR, GENOTYPE)

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": "black",
    "axes.labelsize": 12, "axes.labelcolor": "black",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "xtick.color": "black", "ytick.color": "black",
    "legend.fontsize": 11, "legend.frameon": True, "legend.edgecolor": "black",
    "axes.edgecolor": "black", "axes.linewidth": 1.0,
    "text.color": "black", "figure.dpi": 110, "savefig.dpi": 400,
    "savefig.bbox": "tight", "axes.grid": False,
})
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7"}


def save(fig, name):
    for ext in ("png", "tif"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), dpi=400, bbox_inches="tight",
                    pil_kwargs={"compression": "tiff_lzw"} if ext == "tif" else None)
    plt.close(fig)
    print(f"  saved  {name}", flush=True)


df, y, groups_cell, groups_geno, groups_year = load_data()
BEST = ExtraTreesRegressor(n_estimators=800, random_state=SEED, n_jobs=-1)
print("05b — remaining figures", flush=True)

# ===============================================================
# Partial dependence — morphological traits only
# ===============================================================
print("\nPartial dependence (morphological traits)", flush=True)
top = pd.read_csv(os.path.join(TAB_DIR, "03_permutation_importance.csv"))
top = top[(top.Design == "B_groupcell") & (top["Feature set"] == "MORPH")] \
    .sort_values("Permutation dR2_mean", ascending=False)["Feature"].tolist()[:6]
print("  features:", top, flush=True)

# The three count traits are stored as integers, and partial dependence refuses
# integer columns outright, so the frame is cast to float here for the same
# reason it is cast in the full-feature block below.
X_morph = df[MORPH_FEATURES].astype(float)
pipe = make_pipeline(BEST, MORPH_FEATURES, genotype_categorical=True)
pipe.fit(X_morph, y)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
PartialDependenceDisplay.from_estimator(
    pipe, X_morph, features=top, ax=axes.ravel()[:len(top)],
    grid_resolution=40, n_jobs=-1,
    line_kw={"color": OI["blue"], "linewidth": 2.4})
for ax in axes.ravel():
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_ylabel("Partial dependence (g)", fontsize=11)
fig.suptitle("Partial dependence of predicted fresh plant weight on the six most important "
             "morphological traits\n(ExtraTrees, morphological traits only)",
             fontsize=13.5, fontweight="bold", y=1.01)
fig.tight_layout()
save(fig, "Figure_12_partial_dependence_morph")

# ===============================================================
# Partial dependence — year, genotype, plant height (full model)
# ===============================================================
print("\nPartial dependence (full feature set)", flush=True)
# Year and Genotype are stored as int64 in the source file; scikit-learn refuses
# to compute partial dependence on integer columns because the grid would be
# silently rounded, so the frame is cast to float first.
X_full = df[FULL_FEATURES].astype(float)
pipe_f = make_pipeline(BEST, FULL_FEATURES, genotype_categorical=False)
pipe_f.fit(X_full, y)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
PartialDependenceDisplay.from_estimator(
    pipe_f, X_full, features=[YEAR, GENOTYPE, "Plant height"],
    ax=axes, grid_resolution=40, n_jobs=-1,
    line_kw={"color": OI["vermillion"], "linewidth": 2.4})
for ax in axes:
    ax.grid(alpha=0.25, linestyle=":")
    ax.set_ylabel("Partial dependence (g)", fontsize=11)
fig.suptitle("Partial dependence on year, genotype and plant height (full feature set)",
             fontsize=13, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "Figure_12b_partial_dependence_full")

# ===============================================================
# Learning curves
# ===============================================================
print("\nLearning curves", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
configs = [(KFold(n_splits=5, shuffle=True, random_state=SEED), None, FULL_FEATURES,
            "(a) Full feature set, random 5-fold CV", OI["vermillion"]),
           (GroupKFold(n_splits=5), groups_cell, MORPH_FEATURES,
            "(b) Morphological traits only, GroupKFold by cell", OI["blue"])]
lc_out = {}
for ax, (spl, grp, feats, title, c) in zip(axes, configs):
    p = make_pipeline(BEST, feats, genotype_categorical=True)
    sizes, tr, te = learning_curve(p, df[feats], y, groups=grp, cv=spl,
                                   train_sizes=np.linspace(0.2, 1.0, 9),
                                   scoring="r2", n_jobs=-1,
                                   shuffle=True, random_state=SEED)
    ax.plot(sizes, tr.mean(1), "o-", color=OI["orange"], lw=2.2, ms=7,
            markeredgecolor="black", label="Training R$^2$")
    ax.fill_between(sizes, tr.mean(1) - tr.std(1), tr.mean(1) + tr.std(1),
                    alpha=0.18, color=OI["orange"])
    ax.plot(sizes, te.mean(1), "s-", color=c, lw=2.2, ms=7,
            markeredgecolor="black", label="Validation R$^2$")
    ax.fill_between(sizes, te.mean(1) - te.std(1), te.mean(1) + te.std(1),
                    alpha=0.18, color=c)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xlabel("Number of training samples")
    ax.set_ylabel("R$^2$")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25, linestyle=":")
    lc_out[title] = {"train_sizes": sizes.tolist(),
                     "train_r2_mean": tr.mean(1).tolist(),
                     "val_r2_mean": te.mean(1).tolist(),
                     "val_r2_sd": te.std(1).tolist()}
    print(f"  {title}: validation R2 {te.mean(1)[0]:.3f} -> {te.mean(1)[-1]:.3f}", flush=True)
fig.tight_layout()
save(fig, "Figure_13_learning_curves")
with open(os.path.join(TAB_DIR, "05_learning_curves.json"), "w", encoding="utf-8") as f:
    json.dump(lc_out, f, indent=2)

# ===============================================================
# PCA scree plot and biplot
# ===============================================================
print("\nPCA scree and biplot", flush=True)
PCA_VARS = MORPH_FEATURES + [TARGET_EN]
Z = StandardScaler().fit_transform(df[PCA_VARS].values)
p = PCA(random_state=SEED).fit(Z)
eig, evr = p.explained_variance_, p.explained_variance_ratio_

fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
ax = axes[0]
xs = np.arange(1, len(eig) + 1)
ax.bar(xs, eig, color=OI["sky"], edgecolor="black", label="Eigenvalue")
ax.plot(xs, eig, "o-", color=OI["vermillion"], lw=2, ms=7, markeredgecolor="black")
ax.axhline(1, color="black", ls="--", lw=1.6, label="Kaiser criterion (eigenvalue = 1)")
ax.set_xlabel("Principal component")
ax.set_ylabel("Eigenvalue")
ax.set_xticks(xs)
ax.set_xticklabels([f"PC{i}" for i in xs])
ax.set_title("(a) Scree plot")
ax2 = ax.twinx()
ax2.plot(xs, 100 * np.cumsum(evr), "s--", color=OI["green"], lw=2, ms=6,
         markeredgecolor="black", label="Cumulative % variance")
ax2.set_ylabel("Cumulative variance explained (%)", color="black")
ax2.set_ylim(0, 105)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right")
ax.grid(alpha=0.25, linestyle=":")

ax = axes[1]
sc = p.transform(Z)[:, :2]
ld = p.components_[:2].T * np.sqrt(eig[:2])
scale = 0.9 * np.abs(sc).max() / np.abs(ld).max()
cols = [OI["blue"] if v == 1 else OI["orange"] for v in df[YEAR]]
ax.scatter(sc[:, 0], sc[:, 1], c=cols, s=20, alpha=0.45, edgecolor="none")
for i, v in enumerate(PCA_VARS):
    ax.arrow(0, 0, ld[i, 0] * scale, ld[i, 1] * scale, color="black",
             width=0.012, head_width=0.14, length_includes_head=True, alpha=0.9)
    ax.text(ld[i, 0] * scale * 1.13, ld[i, 1] * scale * 1.13, v,
            fontsize=10.5, fontweight="bold", ha="center", va="center", color="black",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6, alpha=0.85))
ax.axhline(0, color="grey", lw=0.9)
ax.axvline(0, color="grey", lw=0.9)
ax.set_xlabel(f"PC1 ({100*evr[0]:.2f}% of variance)")
ax.set_ylabel(f"PC2 ({100*evr[1]:.2f}% of variance)")
ax.set_title("(b) Biplot of morphological traits and fresh plant weight")
ax.legend(handles=[Patch(facecolor=OI["blue"], label="Year 1 (2022)"),
                   Patch(facecolor=OI["orange"], label="Year 2 (2023)")], loc="upper left")
ax.grid(alpha=0.25, linestyle=":")
fig.tight_layout()
save(fig, "Figure_14_pca_scree_biplot")

# ===============================================================
# Does PCA help prediction?
# ===============================================================
print("\nPCA contribution to prediction", flush=True)
pb = pd.read_csv(os.path.join(TAB_DIR, "04_pca_ml_benchmark.csv"))
SHORT = {"Raw morphological traits (8)": "Raw traits\n(all 8)",
         "PCA scores, all 8 PCs (rotation only)": "PCA\nall 8 PCs",
         "PCA scores, first 2 PCs": "PCA\nfirst 2 PCs",
         "PCA scores, first 3 PCs": "PCA\nfirst 3 PCs",
         "PCA scores, first 4 PCs": "PCA\nfirst 4 PCs"}
ORDER = ["Raw morphological traits (8)", "PCA scores, first 2 PCs", "PCA scores, first 3 PCs",
         "PCA scores, first 4 PCs", "PCA scores, all 8 PCs (rotation only)"]
fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), sharey=True)
for ax, dname in zip(axes, sorted(pb["Design"].unique())):
    sub = pb[pb.Design == dname]
    reps = [r for r in ORDER if r in set(sub["Input representation"])]
    best = sub.loc[sub.groupby("Input representation")["R2_mean"].idxmax()]
    best = best.set_index("Input representation").reindex(reps)
    ax.bar(range(len(reps)), best["R2_mean"], yerr=best["R2_sd"],
           color=[OI["green"] if r.startswith("Raw") else OI["sky"] for r in reps],
           edgecolor="black", capsize=4)
    # place each label clear of the top of its error bar
    for i, (r2, sd, m) in enumerate(zip(best["R2_mean"], best["R2_sd"], best["Model"])):
        ax.text(i, r2 + abs(sd) + 0.04, f"{r2:.3f}\n({m})", ha="center", va="bottom",
                fontsize=9.5, color="black")
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels([SHORT.get(r, r) for r in reps], fontsize=10.5)
    ax.set_title(dname)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.axhline(0, color="black", lw=0.9)
axes[0].set_ylabel("Best cross-validated R$^2$ (fold-wise mean)")
axes[0].set_ylim(-0.35, 1.22)
fig.suptitle("Contribution of PCA to predictive performance\n"
             "(best of 17 models for each input representation; PCA fitted inside training folds)",
             fontsize=13.5, fontweight="bold", y=1.03)
fig.tight_layout()
save(fig, "Figure_15_pca_contribution")

print("\nAll remaining figures written to:", FIG_DIR)
