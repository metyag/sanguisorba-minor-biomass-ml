# -*- coding: utf-8 -*-
"""
05 — Regenerate every figure to publication standard.

The Editor reported that in the submitted version the figure text was
unreadable, headings were rendered in a blue tone that made them invisible, and
axis annotations overlapped. All figures are therefore redrawn with:
  * black text on a white background, no coloured headings
  * minimum font size 11 pt (titles 13 pt, tick labels 11 pt)
  * a colour-blind-safe palette (Okabe-Ito) with black edges
  * no annotation drawn over data, titles or axes (checked on the rendered image)
  * English labels throughout (the original figures were labelled in Turkish)
  * 400 dpi TIFF/PNG output

Requires 02 and 03 to have been run first.
"""
import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.model_selection import KFold, GroupKFold, learning_curve
from sklearn.inspection import PartialDependenceDisplay
from sklearn.ensemble import ExtraTreesRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (load_data, make_pipeline, make_preprocessor, SEED, FIG_DIR, TAB_DIR,
                          FULL_FEATURES, MORPH_FEATURES, TARGET_EN, YEAR, GENOTYPE)

# ---------------------------------------------------------------
# Global style — readability first
# ---------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "DejaVu Sans", "font.size": 12,
    "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlecolor": "black",
    "axes.labelsize": 12, "axes.labelcolor": "black", "axes.labelweight": "normal",
    "xtick.labelsize": 11, "ytick.labelsize": 11,
    "xtick.color": "black", "ytick.color": "black",
    "legend.fontsize": 11, "legend.frameon": True, "legend.edgecolor": "black",
    "axes.edgecolor": "black", "axes.linewidth": 1.0,
    "text.color": "black", "figure.dpi": 110, "savefig.dpi": 400,
    "savefig.bbox": "tight", "axes.grid": False,
})

# Okabe-Ito colour-blind-safe palette
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
      "sky": "#56B4E9", "yellow": "#F0E442", "purple": "#CC79A7", "grey": "#999999"}
DESIGN_COLORS = {"A. Random 5-fold": OI["vermillion"], "B. GroupKFold (cell)": OI["blue"],
                 "C. Leave-one-genotype-out": OI["green"], "D. Leave-one-year-out": OI["purple"]}


# Set SKIP_EXISTING=1 in the environment to regenerate only the missing figures.
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "0") == "1"


def exists(name):
    return SKIP_EXISTING and os.path.exists(os.path.join(FIG_DIR, f"{name}.png"))


def save(fig, name):
    # A legend taken out of the layout with set_in_layout(False) is invisible to
    # bbox_inches="tight", which then crops it out of the saved image: the legend
    # of the forward-selection figures disappeared this way, leaving the 95 %
    # threshold and minimum-feature-count lines unlabelled. Every legend is
    # passed explicitly so that the saved bounding box has to contain it.
    ek = [ax.get_legend() for ax in fig.axes if ax.get_legend() is not None]
    ek += list(fig.legends)
    # R2 (t24, F6-06): a Figure-level title is a fig.texts entry; with an explicit
    # bbox_extra_artists it was cropped out of every saved image.
    ek += list(fig.texts)
    for ext in ("png", "tif"):
        p = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(p, dpi=400, bbox_inches="tight", bbox_extra_artists=ek,
                    pil_kwargs={"compression": "tiff_lzw"} if ext == "tif" else None)
    plt.close(fig)
    print(f"  saved  {name}.png / .tif", flush=True)



# ---------------------------------------------------------------
# When SKIP_EXISTING=1, jump straight to the figures that are missing.
# ---------------------------------------------------------------
df, y, groups_cell, groups_geno, groups_year = load_data()
BEST = ExtraTreesRegressor(n_estimators=800, random_state=SEED, n_jobs=-1)
print("=" * 80)
print("05 — FIGURE REGENERATION")
print("=" * 80)

# ===============================================================
# FIGURE 2 (NEW) — Why random CV is optimistic: variance decomposition
# ===============================================================
print("\nFigure: hierarchical structure of the dataset")
fig, ax = plt.subplots(1, 1, figsize=(7.6, 5.4))

cell_means = df.groupby([YEAR, GENOTYPE])[TARGET_EN].mean()
order = cell_means.sort_values().index
# Ranks start at 1, so that the 62 cells read 1-62 on an axis that says so.
pos = {c: i for i, c in enumerate(order, start=1)}
xs = [pos[(r[YEAR], r[GENOTYPE])] for _, r in df.iterrows()]
cols = [OI["blue"] if r[YEAR] == 1 else OI["orange"] for _, r in df.iterrows()]
ax.scatter(xs, df[TARGET_EN], c=cols, s=18, alpha=0.75, edgecolor="none")
ax.plot(range(1, len(order) + 1), [cell_means[c] for c in order], color="black", lw=1.6,
        label="Genotype × year cell mean")
ax.set_xticks([1, 10, 20, 30, 40, 50, 62])
ax.set_xlabel("Genotype × year cell (rank 1-62, ordered by cell mean)")
ax.set_ylabel("Fresh plant weight (g)")
ax.set_title("All 10 plants of a cell occupy a narrow band")
ax.legend(handles=[Patch(facecolor=OI["blue"], label="Year 1 (2022)"),
                   Patch(facecolor=OI["orange"], label="Year 2 (2023)"),
                   plt.Line2D([], [], color="black", lw=1.6, label="Cell mean")],
          loc="upper left")
ax.grid(alpha=0.25, linestyle=":")

# Panel (b), which stacked the between- and within-group variance shares of the
# three grouping factors, has been removed: those two shares are columns three
# and four of Table 1, so the panel repeated the table exactly.
fig.tight_layout()
save(fig, "Figure_02_dataset_structure")

# ===============================================================
# FIGURE 3 (NEW) — Model performance under the four validation designs
# ===============================================================
print("\nFigure: model performance across validation designs")
vm = pd.read_csv(os.path.join(TAB_DIR, "02_validation_matrix_full.csv"))
vm = vm[vm["Status"] == "ok"]
code2lab = {"random": "A. Random 5-fold", "groupcell": "B. GroupKFold (cell)",
            "logo": "C. Leave-one-genotype-out", "loyo": "D. Leave-one-year-out"}
vm["Design"] = vm["design_code"].map(code2lab)

for fs_key, fs_tag, fs_title in [
        ("FULL", "full", "Full feature set (year + genotype + 8 morphological traits)"),
        ("MORPH-ONLY", "morph", "Morphological traits only (8 traits)")]:
    sub = vm[vm["Feature set"].str.startswith(fs_key)]
    order = (sub[sub.Design == "B. GroupKFold (cell)"]
             .sort_values("pooled_R2", ascending=False)["Model"].tolist())
    fig, ax = plt.subplots(figsize=(14, 6.6))
    w = 0.2
    xpos = np.arange(len(order))
    for i, d in enumerate(code2lab.values()):
        s = sub[sub.Design == d].set_index("Model").reindex(order)
        # Error bars are shown only for the two five-fold designs, where a fold-wise
        # SD is meaningful; under leave-one-genotype-out a per-fold R2 is unstable
        # (see the metric convention in the Methods), so no bar is drawn.
        err = s["perfold_R2_sd"] if d.startswith(("A.", "B.")) else None
        if err is not None:
            # The upper whisker is truncated at R2 = 1, as the caption states, so that
            # no whisker runs through the frame and suggests a value above one.
            up = np.minimum(err.values, np.maximum(0.0, 1.0 - s["pooled_R2"].values))
            err = np.vstack([err.values, up])
        ax.bar(xpos + (i - 1.5) * w, s["pooled_R2"], w, yerr=err,
               label=d, color=DESIGN_COLORS[d], edgecolor="black", linewidth=0.6,
               capsize=2.5, error_kw={"linewidth": 0.9})
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels(order, rotation=42, ha="right")
    ax.set_ylabel("Pooled out-of-fold R$^2$")
    lo = min(-0.35, float(np.nanmin(sub["pooled_R2"])) - 0.1)
    ax.set_ylim(lo, 1.05)
    ax.set_title(f"Predictive performance under four validation designs\n{fs_title}")
    # Outside the axes: at lower left it covered the end of a leave-one-year-out bar.
    ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), ncol=1)
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.tight_layout()
    save(fig, f"Figure_03_validation_designs_{fs_tag}")

# ===============================================================
# FIGURE 4 — Four metrics for the morphology-only model under grouped CV
# ===============================================================
print("\nFigure: four-metric comparison (morphology-only, grouped CV)")
sub = vm[(vm["Feature set"].str.startswith("MORPH-ONLY")) &
         (vm["design_code"] == "groupcell")].sort_values("pooled_R2", ascending=False)
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
specs = [("pooled_R2", "perfold_R2_sd", "R$^2$ (higher is better)", OI["blue"]),
         ("pooled_MAE", "perfold_MAE_sd", "MAE (g, lower is better)", OI["orange"]),
         ("pooled_MSE", "perfold_MSE_sd", "MSE (g$^2$, lower is better)", OI["green"]),
         ("pooled_MAPE", "perfold_MAPE_sd", "MAPE (%, lower is better)", OI["purple"])]
for ax, (m, s, lab, c) in zip(axes.ravel(), specs):
    ax.bar(sub["Model"], sub[m], yerr=sub[s], color=c, edgecolor="black",
           linewidth=0.6, capsize=3)
    ax.set_xticks(range(len(sub)))
    ax.set_xticklabels(sub["Model"], rotation=52, ha="right", fontsize=10)
    ax.set_ylabel(lab)
    ax.set_title(lab.split(" (")[0])
    ax.grid(axis="y", alpha=0.25, linestyle=":")
    ax.axhline(0, color="black", lw=0.9)
fig.suptitle("Model comparison — morphological traits only, GroupKFold by genotype x year cell",
             fontsize=14, fontweight="bold", y=1.00)
fig.tight_layout()
save(fig, "Figure_04_four_metrics_morph_grouped")

# ===============================================================
# FIGURE 5 — Univariate R2 per feature, random vs grouped CV
# ===============================================================
print("\nFigure: univariate feature contribution")
uni = pd.read_csv(os.path.join(TAB_DIR, "03_univariate_r2.csv"))
piv = uni.pivot(index="Feature", columns="Design", values="R2_mean")
sd = uni.pivot(index="Feature", columns="Design", values="R2_sd")
piv = piv.sort_values("A_random")
sd = sd.reindex(piv.index)
fig, ax = plt.subplots(figsize=(11, 7))
ypos = np.arange(len(piv))
h = 0.38
# R2 (t25, V5-01): the upper whisker is truncated at R2 = 1, as in Figure 3, so that no
# whisker suggests a value above the attainable maximum.
_ub = np.vstack([sd["B_groupcell"].values,
                 np.minimum(sd["B_groupcell"].values,
                            np.maximum(0.0, 1.0 - piv["B_groupcell"].values))])
_ua = np.vstack([sd["A_random"].values,
                 np.minimum(sd["A_random"].values,
                            np.maximum(0.0, 1.0 - piv["A_random"].values))])
ax.barh(ypos + h / 2, piv["A_random"], h, xerr=_ua,
        color=OI["vermillion"], edgecolor="black", capsize=3, label="A. Random 5-fold")
ax.barh(ypos - h / 2, piv["B_groupcell"], h, xerr=_ub,
        color=OI["blue"], edgecolor="black", capsize=3, label="B. GroupKFold (genotype × year cell)")
ax.axvline(0, color="black", lw=1.1)
ax.set_yticks(ypos); ax.set_yticklabels(piv.index)
ax.set_xlabel("R$^2$ when the feature is used alone (mean $\\pm$ SD)")
ax.set_title("Individual predictive contribution of each variable")
# The legend sat over the two bars of the bottom row, which the Editor would
# read as another illegible figure. The lower-left quadrant is empty on this
# chart because only genotype and the two leaf traits reach far negative values.
ax.legend(loc="lower left", framealpha=1.0)
ax.grid(axis="x", alpha=0.25, linestyle=":")
fig.tight_layout()
save(fig, "Figure_05_univariate_r2")

# ===============================================================
# FIGURE 6 — Forward selection curves (the key clarification figure)
# ===============================================================
def _gosterge_alta(fig, axes):
    """Place each panel legend just below that panel's x-axis label."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    for ax in axes:
        bb = ax.xaxis.label.get_window_extent(r).transformed(ax.transAxes.inverted())
        ax.get_legend().set_bbox_to_anchor((0.5, bb.y0 - 0.02), transform=ax.transAxes)


print("\nFigure: forward feature selection")
fwd = pd.read_csv(os.path.join(TAB_DIR, "03_forward_selection.csv"))
fig, axes = plt.subplots(1, 2, figsize=(16, 6.6), sharey=False)
panels = [("A_random", "FULL", "(a) Random 5-fold CV, full feature set", OI["vermillion"]),
          ("B_groupcell", "FULL", "(b) GroupKFold by genotype × year cell, full feature set", OI["blue"])]
for ax, (d, s, title, c) in zip(axes, panels):
    sub = fwd[(fwd.Design == d) & (fwd["Feature set"] == s)].sort_values("Step")
    ax.errorbar(sub["Step"], sub["Cumulative R2_mean"], yerr=sub["Cumulative R2_sd"],
                fmt="o-", color=c, ecolor="black", lw=2.2, ms=8, capsize=4,
                markeredgecolor="black", label="Cumulative CV R$^2$")
    mx = sub["Max R2"].iloc[0]; thr = sub["95% threshold"].iloc[0]
    mk = int(sub["Minimum feature count"].iloc[0])
    ax.axhline(thr, color=OI["green"], ls="--", lw=1.6,
               label=f"95% of maximum = {thr:.3f}")
    ax.axvline(mk, color=OI["purple"], ls=":", lw=1.8,
               label=f"Minimum feature count = {mk}")
    # The traits are named on the x axis rather than written over the points, where
    # they ran across the panel titles and the error bars.
    ax.set_xticks(sub["Step"])
    ax.set_xticklabels([f"{int(k)}. {n}" for k, n in zip(sub["Step"], sub["Added feature"])],
                       rotation=40, ha="right", fontsize=10)
    ax.set_xlabel("Variable added at each step")
    ax.set_ylabel("Cumulative R$^2$ (fold-wise mean $\\pm$ SD)")
    ax.set_title(title)
    _top = float((sub["Cumulative R2_mean"] + sub["Cumulative R2_sd"]).max())
    _bot = float((sub["Cumulative R2_mean"] - sub["Cumulative R2_sd"]).min())
    ax.set_ylim(min(0.0, _bot) - 0.05, _top + 0.08)
    ax.grid(alpha=0.25, linestyle=":")
    _leg = ax.legend(loc="upper center", ncol=1, fontsize=10, framealpha=1.0)
    # Kept out of the layout, so that tight_layout does not shrink the panel to make
    # room for it; it is placed below the axis label once the layout is fixed.
    _leg.set_in_layout(False)
fig.tight_layout()
_gosterge_alta(fig, axes)
save(fig, "Figure_06_forward_selection")

# Morphology-only forward selection
fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
panels = [("A_random", "MORPH", "(a) Random 5-fold CV, morphological traits only", OI["vermillion"]),
          ("B_groupcell", "MORPH", "(b) GroupKFold by cell, morphological traits only", OI["blue"])]
for ax, (d, s, title, c) in zip(axes, panels):
    sub = fwd[(fwd.Design == d) & (fwd["Feature set"] == s)].sort_values("Step")
    ax.errorbar(sub["Step"], sub["Cumulative R2_mean"], yerr=sub["Cumulative R2_sd"],
                fmt="o-", color=c, ecolor="black", lw=2.2, ms=8, capsize=4,
                markeredgecolor="black", label="Cumulative CV R$^2$")
    thr = sub["95% threshold"].iloc[0]; mk = int(sub["Minimum feature count"].iloc[0])
    ax.axhline(thr, color=OI["green"], ls="--", lw=1.6, label=f"95% of maximum = {thr:.3f}")
    ax.axvline(mk, color=OI["purple"], ls=":", lw=1.8, label=f"Minimum feature count = {mk}")
    # The traits are named on the x axis rather than written over the points, where
    # they ran across the panel titles and the error bars.
    ax.set_xticks(sub["Step"])
    ax.set_xticklabels([f"{int(k)}. {n}" for k, n in zip(sub["Step"], sub["Added feature"])],
                       rotation=40, ha="right", fontsize=10)
    ax.set_xlabel("Variable added at each step")
    ax.set_ylabel("Cumulative R$^2$ (fold-wise mean $\\pm$ SD)")
    ax.set_title(title)
    _top = float((sub["Cumulative R2_mean"] + sub["Cumulative R2_sd"]).max())
    _bot = float((sub["Cumulative R2_mean"] - sub["Cumulative R2_sd"]).min())
    ax.set_ylim(min(0.0, _bot) - 0.05, _top + 0.08)
    ax.grid(alpha=0.25, linestyle=":")
    _leg = ax.legend(loc="upper center", ncol=1, fontsize=10, framealpha=1.0)
    # Kept out of the layout, so that tight_layout does not shrink the panel to make
    # room for it; it is placed below the axis label once the layout is fixed.
    _leg.set_in_layout(False)
fig.tight_layout()
_gosterge_alta(fig, axes)
save(fig, "Figure_07_forward_selection_morph")

# ===============================================================
# FIGURE 8 — RF impurity vs permutation importance
# ===============================================================
print("\nFigure: feature importance")
rf = pd.read_csv(os.path.join(TAB_DIR, "03_rf_importance.csv"))
perm = pd.read_csv(os.path.join(TAB_DIR, "03_permutation_importance.csv"))
fig, axes = plt.subplots(1, 3, figsize=(19, 6.4))

s = rf[rf["Feature set"] == "FULL"].sort_values("RF importance")
axes[0].barh(s["Feature"], s["RF importance"], color=OI["orange"], edgecolor="black")
axes[0].set_xlabel("Impurity-based importance")
axes[0].set_title("(a) Random Forest impurity importance\n(full feature set, fitted on all data)")

s = perm[(perm.Design == "A_random") & (perm["Feature set"] == "FULL")] \
    .sort_values("Permutation dR2_mean")
axes[1].barh(s["Feature"], s["Permutation dR2_mean"], xerr=s["Permutation dR2_sd"],
             color=OI["vermillion"], edgecolor="black", capsize=3)
axes[1].set_xlabel("Drop in R$^2$ when permuted")
axes[1].set_title("(b) Permutation importance\nheld-out folds, random 5-fold CV")

s = perm[(perm.Design == "B_groupcell") & (perm["Feature set"] == "FULL")] \
    .sort_values("Permutation dR2_mean")
axes[2].barh(s["Feature"], s["Permutation dR2_mean"], xerr=s["Permutation dR2_sd"],
             color=OI["blue"], edgecolor="black", capsize=3)
axes[2].set_xlabel("Drop in R$^2$ when permuted")
axes[2].set_title("(c) Permutation importance\nheld-out folds, GroupKFold by cell")
for ax in axes:
    ax.axvline(0, color="black", lw=1.0)
    ax.grid(axis="x", alpha=0.25, linestyle=":")
fig.tight_layout()
save(fig, "Figure_08_feature_importance")

# ===============================================================
# FIGURE 9 — Correlation matrix
# ===============================================================
print("\nFigure: correlation matrix")
corr = pd.read_csv(os.path.join(TAB_DIR, "03_correlation_matrix.csv"), index_col=0)
fig, ax = plt.subplots(figsize=(11.5, 9.5))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
            vmin=-1, vmax=1, square=True, linewidths=0.6, linecolor="white",
            cbar_kws={"shrink": 0.8, "label": "Pearson r"},
            annot_kws={"size": 10, "color": "black"}, ax=ax)
ax.set_title("Pearson correlation matrix of predictors and fresh plant weight")
plt.setp(ax.get_xticklabels(), rotation=42, ha="right", fontsize=11)
plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)
fig.tight_layout()
save(fig, "Figure_09_correlation_matrix")

# ===============================================================
# FIGURE 10 — Agreement diagnostics for the morphology-only model, grouped CV
# ===============================================================
print("\nFigure: agreement diagnostics")
diag = json.load(open(os.path.join(TAB_DIR, "03_diagnostics.json"), encoding="utf-8"))
for tag, title in [("B_groupcell_MORPH",
                    "Morphological traits only — GroupKFold by genotype × year cell"),
                   ("A_random_FULL",
                    "Full feature set — random 5-fold CV (as originally submitted)")]:
    oof = pd.read_csv(os.path.join(TAB_DIR, f"03_oof_predictions_{tag}.csv"))
    d = diag[tag]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))
    mn = min(oof["actual"].min(), oof["predicted_oof"].min())
    mx = max(oof["actual"].max(), oof["predicted_oof"].max())
    axes[0].scatter(oof["actual"], oof["predicted_oof"], s=26, alpha=0.6,
                    color=OI["blue"], edgecolor="black", linewidth=0.3)
    axes[0].plot([mn, mx], [mn, mx], "--", color=OI["vermillion"], lw=2, label="y = x")
    axes[0].set_xlabel("Measured fresh plant weight (g)")
    axes[0].set_ylabel("Out-of-fold prediction (g)")
    axes[0].set_title(f"(a) Predicted vs measured\nR$^2$ = {d['R2']:.3f}, RMSE = {d['RMSE']:.2f} g")
    axes[0].legend(); axes[0].grid(alpha=0.25, linestyle=":")

    axes[1].scatter(oof["predicted_oof"], oof["residual"], s=26, alpha=0.6,
                    color=OI["green"], edgecolor="black", linewidth=0.3)
    axes[1].axhline(0, color=OI["vermillion"], ls="--", lw=2)
    axes[1].set_xlabel("Predicted fresh plant weight (g)")
    axes[1].set_ylabel("Residual (measured - predicted, g)")
    axes[1].set_title("(b) Residuals versus prediction")
    axes[1].grid(alpha=0.25, linestyle=":")

    mean_ap = (oof["actual"] + oof["predicted_oof"]) / 2
    axes[2].scatter(mean_ap, oof["residual"], s=26, alpha=0.6,
                    color=OI["purple"], edgecolor="black", linewidth=0.3)
    axes[2].axhline(d["BlandAltman_mean_difference"], color="black", lw=2,
                    label=f"Mean difference = {d['BlandAltman_mean_difference']:+.2f} g")
    axes[2].axhline(d["BlandAltman_LoA_upper"], color=OI["vermillion"], ls="--", lw=1.8,
                    label=f"Upper limit of agreement (mean difference + 1.96 SD) = {d['BlandAltman_LoA_upper']:.2f} g")
    axes[2].axhline(d["BlandAltman_LoA_lower"], color=OI["vermillion"], ls="--", lw=1.8,
                    label=f"Lower limit of agreement (mean difference − 1.96 SD) = {d['BlandAltman_LoA_lower']:.2f} g")
    axes[2].set_xlabel("Mean of measured and predicted (g)")
    axes[2].set_ylabel("Difference, measured − predicted (g)")
    axes[2].set_title("(c) Bland-Altman agreement")
    # Below the panel: inside it the three long labels covered the highest points.
    axes[2].legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    axes[2].grid(alpha=0.25, linestyle=":")
    fig.suptitle(title, fontsize=13.5, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, f"Figure_10_diagnostics_{tag}")

# ===============================================================
# FIGURE 11 — SHAP for the morphology-only model
# ===============================================================
print("\nFigure: SHAP")
shap_ok = False
try:
    import shap
    pre = make_preprocessor(MORPH_FEATURES, genotype_categorical=True)
    Xp = pre.fit_transform(df[MORPH_FEATURES])
    et = ExtraTreesRegressor(n_estimators=800, random_state=SEED, n_jobs=-1).fit(Xp, y)
    expl = shap.TreeExplainer(et)
    sv = expl.shap_values(Xp, check_additivity=False)

    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(sv, Xp, feature_names=MORPH_FEATURES, show=False, max_display=10,
                      plot_size=None)
    plt.title("SHAP summary — ExtraTrees on morphological traits only",
              fontsize=13, fontweight="bold", color="black")
    plt.xlabel("SHAP value (impact on predicted fresh plant weight, g)", fontsize=12)
    plt.tight_layout()
    save(plt.gcf(), "Figure_11_shap_beeswarm_morph")

    mabs = np.abs(sv).mean(axis=0)
    sh_df = pd.DataFrame({"Feature": MORPH_FEATURES, "Mean |SHAP| (g)": mabs}) \
        .sort_values("Mean |SHAP| (g)", ascending=False)
    sh_df.to_csv(os.path.join(TAB_DIR, "05_shap_mean_abs_morph.csv"), index=False)
    print(sh_df.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    fig, ax = plt.subplots(figsize=(9.5, 6))
    s = sh_df.sort_values("Mean |SHAP| (g)")
    ax.barh(s["Feature"], s["Mean |SHAP| (g)"], color=OI["blue"], edgecolor="black")
    ax.set_xlabel("Mean |SHAP value| (g)")
    ax.set_title("Global feature contribution (mean absolute SHAP)\nExtraTrees, morphological traits only")
    ax.grid(axis="x", alpha=0.25, linestyle=":")
    fig.tight_layout()
    save(fig, "Figure_11b_shap_bar_morph")

    # SHAP for the full feature set, for comparison with the submitted Figure 8
    pre_f = make_preprocessor(FULL_FEATURES, genotype_categorical=False)
    Xf = pre_f.fit_transform(df[FULL_FEATURES])
    etf = ExtraTreesRegressor(n_estimators=800, random_state=SEED, n_jobs=-1).fit(Xf, y)
    svf = shap.TreeExplainer(etf).shap_values(Xf, check_additivity=False)
    fig = plt.figure(figsize=(10, 7))
    shap.summary_plot(svf, Xf, feature_names=FULL_FEATURES, show=False, max_display=10,
                      plot_size=None)
    plt.title("SHAP summary — ExtraTrees on the full feature set",
              fontsize=13, fontweight="bold", color="black")
    plt.xlabel("SHAP value (impact on predicted fresh plant weight, g)", fontsize=12)
    plt.tight_layout()
    save(plt.gcf(), "Figure_11c_shap_beeswarm_full")
    mabs_f = np.abs(svf).mean(axis=0)
    pd.DataFrame({"Feature": FULL_FEATURES, "Mean |SHAP| (g)": mabs_f}) \
        .sort_values("Mean |SHAP| (g)", ascending=False) \
        .to_csv(os.path.join(TAB_DIR, "05_shap_mean_abs_full.csv"), index=False)
    shap_ok = True
except Exception as e:
    print("  SHAP skipped:", e)

# ===============================================================
# FIGURE 12 — Partial dependence
# ===============================================================
print("\nFigure: partial dependence")
try:
    pipe = make_pipeline(BEST, MORPH_FEATURES, genotype_categorical=True)
    pipe.fit(df[MORPH_FEATURES], y)
    top = pd.read_csv(os.path.join(TAB_DIR, "03_permutation_importance.csv"))
    top = top[(top.Design == "B_groupcell") & (top["Feature set"] == "MORPH")] \
        .sort_values("Permutation dR2_mean", ascending=False)["Feature"].tolist()[:6]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    PartialDependenceDisplay.from_estimator(
        pipe, df[MORPH_FEATURES], features=top, ax=axes.ravel()[:len(top)],
        grid_resolution=40, n_jobs=-1,
        line_kw={"color": OI["blue"], "linewidth": 2.4})
    for ax in axes.ravel():
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_ylabel("Partial dependence (g)", fontsize=11)
    fig.suptitle("Partial dependence of predicted fresh plant weight on the six most "
                 "important morphological traits\n(ExtraTrees, morphological traits only)",
                 fontsize=13.5, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "Figure_12_partial_dependence_morph")

    # PDP for year and genotype in the full model, to document their step-like shape
    pipe_f = make_pipeline(BEST, FULL_FEATURES, genotype_categorical=False)
    pipe_f.fit(df[FULL_FEATURES], y)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    PartialDependenceDisplay.from_estimator(
        pipe_f, df[FULL_FEATURES], features=[YEAR, GENOTYPE, "Plant height"],
        ax=axes, grid_resolution=40, n_jobs=-1,
        line_kw={"color": OI["vermillion"], "linewidth": 2.4})
    for ax in axes:
        ax.grid(alpha=0.25, linestyle=":")
        ax.set_ylabel("Partial dependence (g)", fontsize=11)
    fig.suptitle("Partial dependence on year, genotype and plant height (full feature set)",
                 fontsize=13, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "Figure_12b_partial_dependence_full")
except Exception as e:
    print("  PDP skipped:", e)

# ===============================================================
# FIGURE 13 — Learning curves under both designs
# ===============================================================
print("\nFigure: learning curves")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
configs = [(KFold(n_splits=5, shuffle=True, random_state=SEED), None, FULL_FEATURES,
            # R2 (t25, V5-07): vermillion sits next to the orange training curve; the
            # validation curve of panel (a) uses a clearly separated hue.
            "(a) Full feature set, random 5-fold CV", OI["purple"]),
           (GroupKFold(n_splits=5), groups_cell, MORPH_FEATURES,
            "(b) Morphological traits only, GroupKFold by cell", OI["blue"])]
lc_out = {}
for ax, (spl, grp, feats, title, c) in zip(axes, configs):
    pipe = make_pipeline(BEST, feats, genotype_categorical=True)
    # shuffle=True: the rows of the source file are ordered by genotype, so without
    # shuffling the smaller training subsets would consist of the first few genotypes
    # only and the curve would describe the sampling order rather than the sample size.
    sizes, tr, te = learning_curve(pipe, df[feats], y, groups=grp, cv=spl,
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
    ax.set_xlabel("Number of training samples"); ax.set_ylabel("R$^2$")
    ax.set_title(title); ax.legend(loc="lower right"); ax.grid(alpha=0.25, linestyle=":")
    lc_out[title] = {"train_sizes": sizes.tolist(),
                     "train_r2_mean": tr.mean(1).tolist(),
                     "val_r2_mean": te.mean(1).tolist(),
                     "val_r2_sd": te.std(1).tolist()}
fig.tight_layout()
save(fig, "Figure_13_learning_curves")
with open(os.path.join(TAB_DIR, "05_learning_curves.json"), "w", encoding="utf-8") as f:
    json.dump(lc_out, f, indent=2)

# ===============================================================
# FIGURE 14 — PCA scree plot and biplot
# ===============================================================
print("\nFigure: PCA scree and biplot")
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
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
    ax.set_xlabel("Principal component"); ax.set_ylabel("Eigenvalue")
    ax.set_xticks(xs); ax.set_xticklabels([f"PC{i}" for i in xs])
    ax.set_title("(a) Scree plot")
    ax2 = ax.twinx()
    ax2.plot(xs, 100 * np.cumsum(evr), "s--", color=OI["green"], lw=2, ms=6,
             markeredgecolor="black", label="Cumulative % variance")
    ax2.set_ylabel("Cumulative variance explained (%)", color="black")
    ax2.set_ylim(0, 105)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
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
    ax.axhline(0, color="grey", lw=0.9); ax.axvline(0, color="grey", lw=0.9)
    ax.set_xlabel(f"PC1 ({100*evr[0]:.2f}% of variance)")
    ax.set_ylabel(f"PC2 ({100*evr[1]:.2f}% of variance)")
    ax.set_title("(b) Biplot of morphological traits and fresh plant weight")
    ax.legend(handles=[Patch(facecolor=OI["blue"], label="Year 1 (2022)"),
                       Patch(facecolor=OI["orange"], label="Year 2 (2023)")], loc="upper left")
    ax.grid(alpha=0.25, linestyle=":")
    fig.tight_layout()
    save(fig, "Figure_14_pca_scree_biplot")
except Exception as e:
    print("  PCA figure skipped:", e)

# ===============================================================
# FIGURE 15 — Does PCA help prediction?
# ===============================================================
print("\nFigure: PCA contribution to prediction")
try:
    pb = pd.read_csv(os.path.join(TAB_DIR, "04_pca_ml_benchmark.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), sharey=True)
    for ax, dname in zip(axes, sorted(pb["Design"].unique())):
        sub = pb[pb.Design == dname]
        reps = sorted(sub["Input representation"].unique(),
                      key=lambda s: (not s.startswith("Raw"), s))
        best = sub.loc[sub.groupby("Input representation")["R2_mean"].idxmax()]
        best = best.set_index("Input representation").reindex(reps)
        ax.bar(range(len(reps)), best["R2_mean"], yerr=best["R2_sd"],
               color=[OI["green"] if r.startswith("Raw") else OI["sky"] for r in reps],
               edgecolor="black", capsize=4)
        for i, (r2, m) in enumerate(zip(best["R2_mean"], best["Model"])):
            ax.text(i, r2 + 0.02, f"{r2:.3f}\n({m})", ha="center", va="bottom", fontsize=9.5)
        ax.set_xticks(range(len(reps)))
        ax.set_xticklabels([r.replace(", ", ",\n") for r in reps], fontsize=10)
        ax.set_title(dname); ax.grid(axis="y", alpha=0.25, linestyle=":")
        ax.axhline(0, color="black", lw=0.9)
    axes[0].set_ylabel("Best cross-validated R$^2$")
    fig.suptitle("Contribution of PCA to predictive performance\n"
                 "(best of 17 models for each input representation; PCA fitted inside training folds)",
                 fontsize=13.5, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "Figure_15_pca_contribution")
except Exception as e:
    print("  PCA benchmark figure skipped:", e)

print("\nAll figures written to:", FIG_DIR)
