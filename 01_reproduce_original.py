# -*- coding: utf-8 -*-
"""
01 — Reproduce the originally submitted analysis EXACTLY, and quantify the
     hierarchical structure of the dataset that the reviewers questioned.

Outputs (results/tables/):
  01_variance_decomposition.csv
  01_repro_model_benchmark.csv        (original Table 1)
  01_repro_univariate_r2.csv          (original Table 2)
  01_repro_forward_selection.csv      (original Table 3)
"""
import sys, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import load_data, build_models, make_pipeline, SCORING, SEED, TAB_DIR, FULL_FEATURES, TARGET_EN, YEAR, GENOTYPE

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 50)

df, y, groups_cell, groups_geno, groups_year = load_data()
X = df[FULL_FEATURES]

print("=" * 78)
print("01 — REPRODUCTION OF THE ORIGINALLY SUBMITTED ANALYSIS")
print("=" * 78)
print(f"n observations = {len(df)}, n features = {X.shape[1]}")
print(f"Genotypes = {df[GENOTYPE].nunique()}, Years = {df[YEAR].nunique()}, "
      f"genotype x year cells = {len(np.unique(groups_cell))}")

# ---------------------------------------------------------------
# PART 1 — Variance decomposition: how much of the target variance lies
#          BETWEEN genotype x year cells vs WITHIN cells?
#          This is the quantitative answer to the leakage question.
# ---------------------------------------------------------------
print("\n" + "-" * 78)
print("PART 1 — Hierarchical variance decomposition of fresh plant weight")
print("-" * 78)

t = df[TARGET_EN].values
overall = t.mean()
ss_total = float(((t - overall) ** 2).sum())

rows = []
for label, g in [("Genotype x Year cell (62 cells)", groups_cell),
                 ("Genotype (31 groups, both years pooled)", groups_geno),
                 ("Year (2 groups)", groups_year)]:
    s = pd.Series(t)
    grp = s.groupby(pd.Series(g))
    means = grp.transform("mean")
    ss_between = float(((means - overall) ** 2).sum())
    ss_within = float(((s - means) ** 2).sum())
    rows.append({
        "Grouping factor": label,
        "n groups": len(np.unique(g)),
        "SS between": ss_between,
        "SS within": ss_within,
        "SS total": ss_total,
        "% variance between groups": 100 * ss_between / ss_total,
        "% variance within groups": 100 * ss_within / ss_total,
        "R2 of a pure group-mean lookup": ss_between / ss_total,
    })
var_df = pd.DataFrame(rows)
print(var_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
var_df.to_csv(os.path.join(TAB_DIR, "01_variance_decomposition.csv"), index=False)

print("\n>>> INTERPRETATION: the 'R2 of a pure group-mean lookup' column is the R2 that")
print(">>> a model could achieve by memorising nothing but the mean of each cell.")
print(">>> Under random plant-level CV such a lookup is fully available to the model,")
print(">>> because all 10 plants of every cell are split across training and test folds.")

# Descriptive statistics by year
print("\nTarget by year:")
by_year = df.groupby(YEAR)[TARGET_EN].agg(["count", "mean", "std", "min", "max"])
print(by_year.to_string())
by_year.to_csv(os.path.join(TAB_DIR, "01_target_by_year.csv"))

# Within-cell coefficient of variation
cell_stats = df.groupby([YEAR, GENOTYPE])[TARGET_EN].agg(["count", "mean", "std"])
cell_stats["CV%"] = 100 * cell_stats["std"] / cell_stats["mean"]
print(f"\nWithin-cell CV%: mean={cell_stats['CV%'].mean():.2f}, "
      f"min={cell_stats['CV%'].min():.2f}, max={cell_stats['CV%'].max():.2f}")
cell_stats.to_csv(os.path.join(TAB_DIR, "01_cell_statistics.csv"))

# ---------------------------------------------------------------
# PART 2 — Reproduce Table 1 (17-model benchmark, random 5-fold CV)
#          NOTE: the original code let pandas decide column types. Because
#          'yıl' and 'genotip' are stored as int64 in the Excel file, BOTH were
#          treated as NUMERIC and standardised; the OneHotEncoder branch of the
#          original ColumnTransformer never actually received any column.
#          We reproduce that exactly here (genotype_categorical=False).
# ---------------------------------------------------------------
print("\n" + "-" * 78)
print("PART 2 — Reproducing Table 1 (random 5-fold CV, genotype as integer code)")
print("-" * 78)

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
models = build_models()
print(f"Models evaluated: {len(models)}")

rows = []
for name, reg in models.items():
    pipe = make_pipeline(reg, FULL_FEATURES, genotype_categorical=False)
    out = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1, error_score="raise")
    rows.append({
        "Model": name,
        "R2_mean": np.mean(out["test_r2"]),    "R2_sd": np.std(out["test_r2"]),
        "MSE_mean": -np.mean(out["test_mse"]), "MSE_sd": np.std(-out["test_mse"]),
        "MAE_mean": -np.mean(out["test_mae"]), "MAE_sd": np.std(-out["test_mae"]),
        "MAPE_mean": -np.mean(out["test_mape"]), "MAPE_sd": np.std(-out["test_mape"]),
    })
    print(f"  {name:24s} R2 = {rows[-1]['R2_mean']:.4f} +/- {rows[-1]['R2_sd']:.4f}")

repro = pd.DataFrame(rows).sort_values("R2_mean", ascending=False).reset_index(drop=True)
repro.to_csv(os.path.join(TAB_DIR, "01_repro_model_benchmark.csv"), index=False)
print("\n" + repro.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_name = repro.loc[0, "Model"]
best_reg = models[best_name]
print(f"\nBest model reproduced: {best_name}  "
      f"(fold-wise R2 = {repro.loc[0, 'R2_mean']:.4f} +/- {repro.loc[0, 'R2_sd']:.4f})")

# ---------------------------------------------------------------
# PART 3 — Reproduce Table 2 (univariate R2 per feature)
# ---------------------------------------------------------------
print("\n" + "-" * 78)
print("PART 3 — Reproducing Table 2 (univariate R2, ExtraTrees, random 5-fold CV)")
print("-" * 78)

uni_rows = []
for feat in FULL_FEATURES:
    pipe = make_pipeline(best_reg, [feat], genotype_categorical=False)
    out = cross_validate(pipe, df[[feat]], y, cv=cv, scoring=SCORING, n_jobs=-1, error_score="raise")
    uni_rows.append({
        "Feature": feat,
        "R2_mean": np.mean(out["test_r2"]), "R2_sd": np.std(out["test_r2"]),
        "MAE": -np.mean(out["test_mae"]), "MAPE": -np.mean(out["test_mape"]),
    })
uni = pd.DataFrame(uni_rows).sort_values("R2_mean", ascending=False).reset_index(drop=True)
uni.insert(0, "Rank", range(1, len(uni) + 1))
uni.to_csv(os.path.join(TAB_DIR, "01_repro_univariate_r2.csv"), index=False)
print(uni.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

# ---------------------------------------------------------------
# PART 4 — Reproduce Table 3 (forward feature selection)
# ---------------------------------------------------------------
print("\n" + "-" * 78)
print("PART 4 — Reproducing Table 3 (forward feature selection, random 5-fold CV)")
print("-" * 78)

selected, remaining, fwd_rows = [], list(FULL_FEATURES), []
while remaining:
    best_r2, best_feat, best_sd = -np.inf, None, 0.0
    for feat in remaining:
        cand = selected + [feat]
        pipe = make_pipeline(best_reg, cand, genotype_categorical=False)
        out = cross_validate(pipe, df[cand], y, cv=cv, scoring={"r2": "r2"}, n_jobs=-1, error_score="raise")
        m, s = np.mean(out["test_r2"]), np.std(out["test_r2"])
        if m > best_r2:
            best_r2, best_feat, best_sd = m, feat, s
    selected.append(best_feat)
    remaining.remove(best_feat)
    fwd_rows.append({
        "Step": len(selected), "Added feature": best_feat,
        "Cumulative R2_mean": best_r2, "Cumulative R2_sd": best_sd,
        "Cumulative set": ", ".join(selected),
    })
    print(f"  Step {len(selected):2d}: +{best_feat:30s} R2 = {best_r2:.4f} +/- {best_sd:.4f}")

fwd = pd.DataFrame(fwd_rows)
fwd["Delta R2"] = fwd["Cumulative R2_mean"].diff()
fwd.to_csv(os.path.join(TAB_DIR, "01_repro_forward_selection.csv"), index=False)
max_r2 = fwd["Cumulative R2_mean"].max()
thr = 0.95 * max_r2
min_k = int(np.argmax(fwd["Cumulative R2_mean"].values >= thr)) + 1
print(f"\nMax R2 = {max_r2:.4f} at step {int(fwd['Cumulative R2_mean'].idxmax())+1}; "
      f"95% threshold = {thr:.4f}; minimum feature count = {min_k}")

summary = {
    "n_obs": int(len(df)),
    "n_genotypes": int(df[GENOTYPE].nunique()),
    "n_years": int(df[YEAR].nunique()),
    "n_cells": int(len(np.unique(groups_cell))),
    "pct_variance_between_cells": float(var_df.loc[0, "% variance between groups"]),
    "R2_of_cell_mean_lookup": float(var_df.loc[0, "R2 of a pure group-mean lookup"]),
    "best_model_random_cv": best_name,
    "best_R2_random_cv": float(repro.loc[0, "R2_mean"]),
    "forward_max_R2": float(max_r2),
    "forward_min_k": int(min_k),
}
with open(os.path.join(TAB_DIR, "01_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print("\nSaved: " + TAB_DIR)
print(json.dumps(summary, indent=2))
