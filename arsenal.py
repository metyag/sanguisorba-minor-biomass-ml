# -*- coding: utf-8 -*-
"""
arsenal.py — the complete inventory of modelling options that the revision tests.

Everything downstream (scripts 18-22) imports its data views, its estimator pool
and its evaluation harness from here, so that the wide exploratory sweep and the
nested-selection run that produces the headline number use *identical* code paths.
That identity is what makes the nested estimate an honest description of the
exploratory search rather than a separate experiment.

Three things are defined here.

1. DATA VIEWS.  The cell-level frame is built once and carries, for every
   morphological trait, the raw value, its natural logarithm, a set of
   allometric products, and the logarithms of those products.  A "view" is just
   a named list of columns, so switching between the raw and the log-log
   parameterisation costs nothing and cannot silently differ between scripts.

   The log-log view matters more than it looks.  Biomass scales with plant
   dimensions as a power law, so the additive model that is correct on the
   log scale is log y = b0 + sum_i b_i log x_i.  Taking logarithms of the
   *target* alone, which is what scripts 12-15 did, only half-implements that.

2. ESTIMATOR POOL.  Every estimator is registered with the sample sizes it is
   allowed to run at, so that the same dictionary can serve the 62-cell problem
   and the 620-plant problem without hand-editing.

3. HARNESS.  A single evaluator that produces pooled out-of-fold predictions
   under an arbitrary grouping, plus the rank statistics that matter for a
   selection decision (Spearman, Kendall, top-k recovery).

No numbers are hard-coded anywhere; everything is computed from the Excel file.
"""
import warnings

import numpy as np
from scipy import stats

from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.impute import SimpleImputer

from common_setup import (GENOTYPE, MORPH_FEATURES, SEED, TARGET_EN, YEAR,
                          load_data, mape)

warnings.filterwarnings("ignore")

# ===========================================================================
# 1. DATA VIEWS
# ===========================================================================
ALLOMETRIC = {
    "Stem volume proxy":  lambda X: np.pi * (X["Main stem diameter"] / 2.0) ** 2
                                    * X["Plant height"] * X["Number of main stems"],
    "Leaf area proxy":    lambda X: (X["Number of leaves"] * X["Number of leaflets per leaf"]
                                     * X["Leaflet length"] * X["Leaflet width"]),
    "Height x stems":     lambda X: X["Plant height"] * X["Number of main stems"],
    "Total leaf length":  lambda X: X["Number of leaves"] * X["Leaf length"],
    "Height squared":     lambda X: X["Plant height"] ** 2,
}
ALLO_NAMES = list(ALLOMETRIC)
LOG_MORPH = [f"log {c}" for c in MORPH_FEATURES]
LOG_ALLO = [f"log {c}" for c in ALLO_NAMES]


def _augment(frame):
    """Add allometric products and the logarithm of every positive column."""
    X = frame.copy()
    for name, fn in ALLOMETRIC.items():
        X[name] = fn(X)
    for c in MORPH_FEATURES + ALLO_NAMES:
        v = X[c].astype(float)
        # every trait is strictly positive in this dataset; the guard is only
        # so that a future dataset with a zero does not fail silently
        X[f"log {c}"] = np.log(np.where(v > 0, v, np.nan))
    return X


def build_frames(exclude_last_plant=False):
    """
    Return (PLANT, CELL, GENO) frames.

    exclude_last_plant : drop the tenth record of every genotype x year cell.
        The tenth record is a systematic outlier in the predictors (see
        script 18's data-integrity section); this switch exists so that every
        result can be reported both ways rather than resting on an assumption
        about what that record represents.
    """
    df, _, _, _, _ = load_data()
    df = df.copy()
    df["_pos"] = df.groupby([YEAR, GENOTYPE]).cumcount() + 1
    if exclude_last_plant:
        df = df[df["_pos"] <= 9].reset_index(drop=True)

    cols = MORPH_FEATURES + [TARGET_EN]
    cell = (df.groupby([YEAR, GENOTYPE], as_index=False)[cols]
              .mean().sort_values([YEAR, GENOTYPE]).reset_index(drop=True))
    # within-cell dispersion of the target, used by the weighted fits
    sd = (df.groupby([YEAR, GENOTYPE])[TARGET_EN].std(ddof=1)
            .reset_index(name="_target_sd"))
    n = (df.groupby([YEAR, GENOTYPE])[TARGET_EN].size()
           .reset_index(name="_n_plants"))
    cell = cell.merge(sd, on=[YEAR, GENOTYPE]).merge(n, on=[YEAR, GENOTYPE])

    geno = (df.groupby([GENOTYPE], as_index=False)[cols]
              .mean().sort_values([GENOTYPE]).reset_index(drop=True))

    return _augment(df), _augment(cell), _augment(geno)


# --- named column lists -----------------------------------------------------
VIEWS = {
    "raw8":            MORPH_FEATURES,
    "log8":            LOG_MORPH,
    "raw8+year":       [YEAR] + MORPH_FEATURES,
    "log8+year":       [YEAR] + LOG_MORPH,
    "raw13":           MORPH_FEATURES + ALLO_NAMES,
    "log13":           LOG_MORPH + LOG_ALLO,
    "raw13+year":      [YEAR] + MORPH_FEATURES + ALLO_NAMES,
    "log13+year":      [YEAR] + LOG_MORPH + LOG_ALLO,
    "mixed":           MORPH_FEATURES + LOG_MORPH,
}


def view_columns(name):
    return list(VIEWS[name])


# ===========================================================================
# 2. TARGET TRANSFORMS
# ===========================================================================
def wrap_target(reg, transform):
    """Wrap an estimator so that it is fitted on a transformed response."""
    if transform == "raw":
        return clone(reg)
    if transform == "log":
        return TransformedTargetRegressor(regressor=clone(reg),
                                          func=np.log, inverse_func=np.exp)
    if transform == "log1p":
        return TransformedTargetRegressor(regressor=clone(reg),
                                          func=np.log1p, inverse_func=np.expm1)
    if transform == "sqrt":
        return TransformedTargetRegressor(regressor=clone(reg),
                                          func=np.sqrt, inverse_func=np.square)
    if transform == "yeo":
        return TransformedTargetRegressor(
            regressor=clone(reg),
            transformer=PowerTransformer(method="yeo-johnson", standardize=False))
    if transform == "boxcox":
        return TransformedTargetRegressor(
            regressor=clone(reg),
            transformer=PowerTransformer(method="box-cox", standardize=False))
    raise ValueError(f"unknown target transform: {transform}")


TARGETS = ["raw", "log", "sqrt", "yeo", "boxcox"]


# ===========================================================================
# 3. ESTIMATOR POOL
# ===========================================================================
from sklearn.linear_model import (ARDRegression, BayesianRidge, ElasticNetCV,
                                  HuberRegressor, LassoCV, LinearRegression,
                                  OrthogonalMatchingPursuitCV, QuantileRegressor,
                                  RANSACRegressor, Ridge, RidgeCV, TheilSenRegressor)
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (ConstantKernel, DotProduct, Matern,
                                              RationalQuadratic, RBF, WhiteKernel)
from sklearn.kernel_ridge import KernelRidge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import (AdaBoostRegressor, BaggingRegressor,
                              ExtraTreesRegressor, GradientBoostingRegressor,
                              HistGradientBoostingRegressor, RandomForestRegressor)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except Exception:
    HAS_CAT = False
try:
    from pygam import LinearGAM, s
    HAS_GAM = True
except Exception:
    HAS_GAM = False


def pcr(k):
    """Principal-component regression as a two-step pipeline."""
    return Pipeline([("pca", PCA(n_components=k, random_state=SEED)),
                     ("lin", LinearRegression())])


class GAMWrapper(BaseEstimator, RegressorMixin):
    """Thin scikit-learn wrapper around pygam's LinearGAM with per-term splines."""

    def __init__(self, n_splines=8, lam=0.6):
        self.n_splines = n_splines
        self.lam = lam

    def fit(self, X, y):
        from pygam import LinearGAM, s
        X = np.asarray(X, dtype=float)
        terms = s(0, n_splines=self.n_splines)
        for j in range(1, X.shape[1]):
            terms = terms + s(j, n_splines=self.n_splines)
        self.gam_ = LinearGAM(terms, lam=self.lam).fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.gam_.predict(np.asarray(X, dtype=float))


class MonotoneLGBM(BaseEstimator, RegressorMixin):
    """
    LightGBM with every feature constrained to a non-decreasing effect.

    Biomass cannot fall when a plant becomes taller, thicker or leafier while
    everything else is held constant, so the monotone constraint encodes real
    agronomic knowledge and acts as a regulariser that costs no sample size.
    """

    def __init__(self, n_estimators=400, learning_rate=0.05, num_leaves=7,
                 min_child_samples=5):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples

    def fit(self, X, y):
        from lightgbm import LGBMRegressor
        X = np.asarray(X, dtype=float)
        self.model_ = LGBMRegressor(
            n_estimators=self.n_estimators, learning_rate=self.learning_rate,
            num_leaves=self.num_leaves, min_child_samples=self.min_child_samples,
            monotone_constraints=[1] * X.shape[1], verbose=-1,
            random_state=SEED, n_jobs=1).fit(X, np.asarray(y, dtype=float))
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(X, dtype=float))


def build_pool(level="cell"):
    """
    level : "cell" (n = 62) or "plant" (n = 620).

    The tree ensembles are given deliberately small leaves at cell level; the
    defaults assume hundreds of observations and simply cannot be evaluated
    fairly on 62 units otherwise.
    """
    small = (level == "cell")
    leaf = 2 if small else 5
    pool = {
        # -- regularised linear -------------------------------------------
        "Linear":               LinearRegression(),
        "Ridge":                Ridge(alpha=1.0),
        "RidgeCV":              RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "LassoCV":              LassoCV(alphas=60, max_iter=50000, random_state=SEED),
        "ElasticNetCV":         ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1],
                                             alphas=40, max_iter=50000,
                                             random_state=SEED),
        "Huber":                HuberRegressor(epsilon=1.35, max_iter=5000),
        "Huber (eps 1.1)":      HuberRegressor(epsilon=1.1, max_iter=5000),
        "Huber (eps 2.0)":      HuberRegressor(epsilon=2.0, max_iter=5000),
        "Bayesian Ridge":       BayesianRidge(),
        "ARD Regression":       ARDRegression(),
        "TheilSen":             TheilSenRegressor(random_state=SEED, max_subpopulation=2000),
        "RANSAC":               RANSACRegressor(random_state=SEED, min_samples=0.6),
        "Quantile (median)":    QuantileRegressor(quantile=0.5, alpha=0.001,
                                                  solver="highs"),
        "OMP":                  OrthogonalMatchingPursuitCV(max_iter=8),
        # -- latent variable ----------------------------------------------
        "PLS (2 comp)":         PLSRegression(n_components=2),
        "PLS (3 comp)":         PLSRegression(n_components=3),
        "PLS (4 comp)":         PLSRegression(n_components=4),
        "PLS (5 comp)":         PLSRegression(n_components=5),
        "PCR (3 comp)":         pcr(3),
        "PCR (4 comp)":         pcr(4),
        "PCR (5 comp)":         pcr(5),
        # -- kernel and Gaussian process ----------------------------------
        "Kernel Ridge (RBF)":   KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1),
        "Kernel Ridge (poly2)": KernelRidge(kernel="polynomial", degree=2, alpha=1.0),
        "Kernel Ridge (poly3)": KernelRidge(kernel="polynomial", degree=3, alpha=1.0),
        "Kernel Ridge (lapl)":  KernelRidge(kernel="laplacian", alpha=1.0, gamma=0.1),
        "SVR (RBF)":            SVR(C=50, gamma="scale", epsilon=0.1),
        "SVR (linear)":         SVR(kernel="linear", C=10, epsilon=0.1),
        "GP (RBF)":             GaussianProcessRegressor(
                                    kernel=ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-2),
                                    normalize_y=True, random_state=SEED, alpha=1e-8),
        "GP (Matern 1.5)":      GaussianProcessRegressor(
                                    kernel=ConstantKernel(1.0) * Matern(1.0, nu=1.5) + WhiteKernel(1e-2),
                                    normalize_y=True, random_state=SEED, alpha=1e-8),
        "GP (Matern 2.5)":      GaussianProcessRegressor(
                                    kernel=ConstantKernel(1.0) * Matern(1.0, nu=2.5) + WhiteKernel(1e-2),
                                    normalize_y=True, random_state=SEED, alpha=1e-8),
        "GP (RatQuad)":         GaussianProcessRegressor(
                                    kernel=ConstantKernel(1.0) * RationalQuadratic() + WhiteKernel(1e-2),
                                    normalize_y=True, random_state=SEED, alpha=1e-8),
        "GP (linear+RBF)":      GaussianProcessRegressor(
                                    kernel=ConstantKernel(1.0) * DotProduct()
                                           + ConstantKernel(1.0) * RBF(1.0) + WhiteKernel(1e-2),
                                    normalize_y=True, random_state=SEED, alpha=1e-8),
        # -- neighbours ----------------------------------------------------
        "KNN (5, dist)":        KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "KNN (9, dist)":        KNeighborsRegressor(n_neighbors=9, weights="distance"),
        # -- tree ensembles -------------------------------------------------
        "Random Forest":        RandomForestRegressor(n_estimators=600, min_samples_leaf=leaf,
                                                      random_state=SEED, n_jobs=1),
        "ExtraTrees":           ExtraTreesRegressor(n_estimators=800, min_samples_leaf=leaf,
                                                    random_state=SEED, n_jobs=1),
        "Gradient Boosting":    GradientBoostingRegressor(n_estimators=400, learning_rate=0.05,
                                                          max_depth=2 if small else 3,
                                                          random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05,
                                                              min_samples_leaf=leaf,
                                                              random_state=SEED),
        "AdaBoost":             AdaBoostRegressor(n_estimators=400, learning_rate=0.05,
                                                  random_state=SEED),
        "Bagging (Trees)":      BaggingRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
        # -- neural ----------------------------------------------------------
        "MLP (32-16)":          MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=6000,
                                             random_state=SEED, early_stopping=False,
                                             alpha=1e-2),
        "MLP (64-32)":          MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=6000,
                                             random_state=SEED, alpha=1e-2),
    }
    if HAS_XGB:
        pool["XGBoost"] = XGBRegressor(
            n_estimators=600, learning_rate=0.03, max_depth=2 if small else 4,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=2.0,
            min_child_weight=leaf, random_state=SEED, n_jobs=1, verbosity=0)
    if HAS_LGBM:
        pool["LightGBM"] = LGBMRegressor(
            n_estimators=600, learning_rate=0.03, num_leaves=7 if small else 31,
            min_child_samples=leaf, verbose=-1, random_state=SEED, n_jobs=1)
        pool["LightGBM (monotone)"] = MonotoneLGBM(
            n_estimators=400, learning_rate=0.05, num_leaves=7 if small else 31,
            min_child_samples=leaf)
    if HAS_CAT:
        pool["CatBoost"] = CatBoostRegressor(
            iterations=600, learning_rate=0.03, depth=3 if small else 6,
            l2_leaf_reg=6.0, random_seed=SEED, verbose=0, allow_writing_files=False,
            thread_count=1)
    if HAS_GAM:
        pool["GAM (splines)"] = GAMWrapper(n_splines=6 if small else 12, lam=0.6)
    return pool


FAMILY = {
    "Regularised linear": ["Linear", "Ridge", "RidgeCV", "LassoCV", "ElasticNetCV",
                           "Huber", "Huber (eps 1.1)", "Huber (eps 2.0)",
                           "Bayesian Ridge", "ARD Regression", "TheilSen", "RANSAC",
                           "Quantile (median)", "OMP"],
    "Latent variable": ["PLS (2 comp)", "PLS (3 comp)", "PLS (4 comp)", "PLS (5 comp)",
                        "PCR (3 comp)", "PCR (4 comp)", "PCR (5 comp)"],
    "Kernel and Gaussian process": ["Kernel Ridge (RBF)", "Kernel Ridge (poly2)",
                                    "Kernel Ridge (poly3)", "Kernel Ridge (lapl)",
                                    "SVR (RBF)", "SVR (linear)", "GP (RBF)",
                                    "GP (Matern 1.5)", "GP (Matern 2.5)",
                                    "GP (RatQuad)", "GP (linear+RBF)"],
    "Nearest neighbours": ["KNN (5, dist)", "KNN (9, dist)"],
    "Tree ensembles": ["Random Forest", "ExtraTrees", "Gradient Boosting",
                       "HistGradientBoosting", "AdaBoost", "Bagging (Trees)",
                       "XGBoost", "LightGBM", "LightGBM (monotone)", "CatBoost"],
    "Additive splines": ["GAM (splines)"],
    "Neural network": ["MLP (32-16)", "MLP (64-32)"],
}


def family_of(model_name):
    for fam, members in FAMILY.items():
        if model_name in members:
            return fam
    return "Other"


# ===========================================================================
# 4. HARNESS
# ===========================================================================
def make_pipe(reg, target, standardise=True):
    """Impute, optionally standardise, then fit the target-wrapped estimator."""
    steps = [("impute", SimpleImputer(strategy="median"))]
    if standardise:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", wrap_target(reg, target)))
    return Pipeline(steps)


class GroupOut:
    """Leave-one-group-out over an explicit label array."""

    def __init__(self, g):
        self.g = np.asarray(g)

    def split(self, idx=None):
        idx = np.arange(len(self.g)) if idx is None else np.asarray(idx)
        for u in np.unique(self.g[idx]):
            yield idx[self.g[idx] != u], idx[self.g[idx] == u]

    def n(self, idx=None):
        idx = np.arange(len(self.g)) if idx is None else np.asarray(idx)
        return len(np.unique(self.g[idx]))


def metrics(y_true, y_pred, top_k=10):
    """Accuracy metrics plus the rank statistics a selection decision needs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[m], y_pred[m]
    out = {
        "pooled_R2": float(r2_score(y_true, y_pred)),
        "pooled_RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "pooled_MAE": float(mean_absolute_error(y_true, y_pred)),
        "pooled_MAPE": float(mape(y_true, y_pred)),
        "bias": float(np.mean(y_pred - y_true)),
    }
    if len(y_true) > 4:
        out["spearman"] = float(stats.spearmanr(y_true, y_pred).statistic)
        out["kendall"] = float(stats.kendalltau(y_true, y_pred).statistic)
        k = min(top_k, len(y_true) // 2)
        if k >= 2:
            true_top = set(np.argsort(-y_true)[:k])
            pred_top = set(np.argsort(-y_pred)[:k])
            out[f"top{k}_recovery"] = float(len(true_top & pred_top) / k)
    return out


def evaluate(reg, frame, cols, target, y, splitter, standardise=True,
             sample_weight=None):
    """Pooled out-of-fold evaluation under an arbitrary leave-group-out design."""
    X = frame[cols].astype(float)
    oof = np.full(len(y), np.nan)
    for tr, te in splitter.split():
        pipe = make_pipe(reg, target, standardise)
        if sample_weight is not None:
            try:
                pipe.fit(X.iloc[tr], y[tr], model__sample_weight=sample_weight[tr])
            except Exception:
                pipe.fit(X.iloc[tr], y[tr])
        else:
            pipe.fit(X.iloc[tr], y[tr])
        oof[te] = np.asarray(pipe.predict(X.iloc[te])).ravel()
    return metrics(y, oof), oof


class GroupKFoldArray:
    """Grouped k-fold expressed through the same interface as GroupOut."""

    def __init__(self, g, n_splits=5, seed=SEED):
        self.g = np.asarray(g)
        self.n_splits = n_splits
        u = np.unique(self.g)
        rng = np.random.RandomState(seed)
        order = rng.permutation(len(u))
        self.assign = {u[order[i]]: i % n_splits for i in range(len(u))}

    def split(self, idx=None):
        idx = np.arange(len(self.g)) if idx is None else np.asarray(idx)
        f = np.array([self.assign[v] for v in self.g[idx]])
        for k in range(self.n_splits):
            if (f == k).sum() == 0:
                continue
            yield idx[f != k], idx[f == k]

    def n(self, idx=None):
        return self.n_splits
