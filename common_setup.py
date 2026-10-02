# -*- coding: utf-8 -*-
"""
Common setup module for the revision analyses.

Provides: data loading, English trait names, model zoo (17 models),
metrics, and cross-validation splitters (random / grouped / LOGO / LOYO).
"""
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold, GroupKFold, LeaveOneGroupOut

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, BaggingRegressor, HistGradientBoostingRegressor,
    VotingRegressor, StackingRegressor
)

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

SEED = 42

HERE = os.path.dirname(os.path.abspath(__file__))
# The plant-level data file is not distributed with the code; it is available
# from the corresponding author on reasonable request. Place it
# at data/plant_records.xlsx or set the environment variable BURNET_DATA.
DATA_PATH = os.environ.get("BURNET_DATA",
                           os.path.join(HERE, "data", "plant_records.xlsx"))
RESULTS_DIR = os.path.join(HERE, "results")
FIG_DIR = os.path.join(RESULTS_DIR, "figures")
TAB_DIR = os.path.join(RESULTS_DIR, "tables")
for d in (RESULTS_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

# ---------------------------------------------------------------
# Turkish column names -> English names used in the manuscript
# ---------------------------------------------------------------
TR2EN = {
    "yıl": "Year",
    "genotip": "Genotype",
    "Bitki boyu": "Plant height",
    "ana sap kalınlğı": "Main stem diameter",
    "ana sap sayısı": "Number of main stems",
    "yaprak sayısı": "Number of leaves",
    "yapraktaki yaprakçık sayısı": "Number of leaflets per leaf",
    "yaprak boyu": "Leaf length",
    "yaprakçık eni": "Leaflet width",
    "yaprakçık boyu": "Leaflet length",
    "yeşil bitki ağırlığı": "Fresh plant weight",
}

TARGET_EN = "Fresh plant weight"
YEAR = "Year"
GENOTYPE = "Genotype"

MORPH_FEATURES = [
    "Plant height", "Main stem diameter", "Number of main stems",
    "Number of leaves", "Number of leaflets per leaf", "Leaf length",
    "Leaflet width", "Leaflet length",
]
FULL_FEATURES = [YEAR, GENOTYPE] + MORPH_FEATURES
YEARGEN_FEATURES = [YEAR, GENOTYPE]

# ---------------------------------------------------------------
# The adopted cell-level model, named in one place so that no script
# can drift from another.  Huber regression is a robust loss, which is
# what the structure of the records indicated before any model was
# fitted; ordinary ridge regression is carried alongside as the
# non-robust reference so the choice stays visible to the reader.
# ---------------------------------------------------------------
FINAL_MODEL_NAME = "Huber"
REFERENCE_MODEL_NAME = "Ridge"
FINAL_TARGET = "log"
FINAL_VIEW = "raw8"

FEATURE_SETS = {
    "FULL (10 variables: year + genotype + 8 morphological)": FULL_FEATURES,
    "MORPH-ONLY (8 morphological traits)": MORPH_FEATURES,
    "YEAR+GENOTYPE only (2 identifiers)": YEARGEN_FEATURES,
}


def load_data():
    """Load the Excel file, rename the columns to English and return

        df           the full frame, English column names
        y            the response, float array
        groups_cell  genotype x year cell label, one per record
        groups_geno  genotype label, one per record
        groups_year  season label, one per record
    """
    if not os.path.exists(DATA_PATH):
        raise SystemExit(
            "Data file not found:\n  %s\n\n"
            "The plant-level data file is not distributed with this code; it is"
            " available from the corresponding author on reasonable request."
            " Place it at data/plant_records.xlsx, or set the environment variable"
            " BURNET_DATA to its path.\n\n"
            "Install the pinned package versions first:\n"
            "  pip install -r requirements.txt\n"
            "then run the analysis in order with:\n"
            "  python run_pipeline.py\n"
            "Tables are written to results/tables/ and figures to results/figures/."
            % DATA_PATH)
    df = pd.read_excel(DATA_PATH)
    df.columns = [c.strip() for c in df.columns]
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    missing = [c for c in TR2EN if c not in df.columns]
    if missing:
        raise ValueError(f"Expected columns missing from Excel: {missing}\nFound: {list(df.columns)}")
    df = df.rename(columns=TR2EN)
    df[TARGET_EN] = pd.to_numeric(df[TARGET_EN], errors="coerce")
    df = df.dropna(subset=[TARGET_EN]).reset_index(drop=True)

    y = df[TARGET_EN].astype(float).values
    # Grouping variables
    groups_cell = (df[YEAR].astype(str) + "_" + df[GENOTYPE].astype(str)).values  # genotype × year cell
    groups_geno = df[GENOTYPE].astype(str).values                                  # genotype (both years)
    groups_year = df[YEAR].astype(str).values                                      # year
    return df, y, groups_cell, groups_geno, groups_year


# ---------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------
def make_preprocessor(features, genotype_categorical=True):
    """
    Build a ColumnTransformer for the given feature list.

    genotype_categorical=True  -> Genotype is one-hot encoded (statistically correct:
                                  the 31 genotype codes are nominal labels, not a scale).
    genotype_categorical=False -> Genotype is kept as an integer code and scaled,
                                  which is what the ORIGINAL submitted analysis did.
    """
    cat_cols = [GENOTYPE] if (genotype_categorical and GENOTYPE in features) else []
    num_cols = [c for c in features if c not in cat_cols]

    transformers = []
    if num_cols:
        transformers.append((
            "num",
            Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]),
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      # sparse_output=False: HistGradientBoosting cannot consume
                      # a sparse matrix, and dense output keeps every estimator on
                      # exactly the same input representation.
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
            cat_cols,
        ))
    return ColumnTransformer(transformers=transformers, remainder="drop")


# ---------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------
def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100)


SCORING = {
    "r2": "r2",
    "mse": make_scorer(mean_squared_error, greater_is_better=False),
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "mape": make_scorer(mape, greater_is_better=False),
}


# ---------------------------------------------------------------
# Model zoo — identical hyper-parameters to the original submission
# ---------------------------------------------------------------
def build_models():
    models = {
        "Linear": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, max_iter=10000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000),
        "Huber": HuberRegressor(epsilon=1.35, max_iter=1000),
        "KNN": KNeighborsRegressor(n_neighbors=9, weights="distance"),
        "SVR (RBF)": SVR(C=50, gamma="scale", epsilon=0.1),
        "Decision Tree": DecisionTreeRegressor(random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=600, random_state=SEED, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=800, random_state=SEED, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=600, learning_rate=0.05, random_state=SEED),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, random_state=SEED),
        "AdaBoost": AdaBoostRegressor(n_estimators=600, learning_rate=0.05, random_state=SEED),
        "Bagging (Trees)": BaggingRegressor(
            estimator=DecisionTreeRegressor(random_state=SEED),
            n_estimators=300, random_state=SEED, n_jobs=-1),
    }
    if HAS_XGB:
        models["XGBoost"] = XGBRegressor(
            n_estimators=1200, learning_rate=0.03, max_depth=5,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=0)

    voting_est = [
        ("rf", RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)),
        ("et", ExtraTreesRegressor(n_estimators=600, random_state=SEED, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, random_state=SEED)),
    ]
    stack_est = [
        ("rf", RandomForestRegressor(n_estimators=250, random_state=SEED, n_jobs=-1)),
        ("et", ExtraTreesRegressor(n_estimators=400, random_state=SEED, n_jobs=-1)),
        ("gbr", GradientBoostingRegressor(n_estimators=350, learning_rate=0.05, random_state=SEED)),
    ]
    if HAS_XGB:
        xgb_clone = XGBRegressor(
            n_estimators=1200, learning_rate=0.03, max_depth=5,
            subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
            random_state=SEED, n_jobs=-1, verbosity=0)
        voting_est.append(("xgb", xgb_clone))
        stack_est.append(("xgb", xgb_clone))

    models["Hybrid Voting"] = VotingRegressor(estimators=voting_est)
    models["Hybrid Stacking"] = StackingRegressor(
        estimators=stack_est, final_estimator=Ridge(alpha=1.0),
        passthrough=False, n_jobs=1)
    return models


def make_pipeline(model, features, genotype_categorical=True):
    from sklearn.base import clone
    return Pipeline([
        ("preprocess", make_preprocessor(features, genotype_categorical)),
        ("model", clone(model)),
    ])


# ---------------------------------------------------------------
# Cross-validation schemes
# ---------------------------------------------------------------
class LeaveOneYearOut:
    """2-fold splitter: train on one year, test on the other (and vice versa)."""

    def __init__(self, groups_year):
        self.groups_year = np.asarray(groups_year)

    def split(self, X, y=None, groups=None):
        for yr in np.unique(self.groups_year):
            test = np.where(self.groups_year == yr)[0]
            train = np.where(self.groups_year != yr)[0]
            yield train, test

    def get_n_splits(self, X=None, y=None, groups=None):
        return len(np.unique(self.groups_year))


def cv_schemes(groups_cell, groups_geno, groups_year):
    """Return dict name -> (splitter, groups_array, short_code)."""
    return {
        "A. Random 5-fold (plant level) [as originally submitted]":
            (KFold(n_splits=5, shuffle=True, random_state=SEED), None, "random"),
        "B. GroupKFold by genotype × year cell (62 groups, 5 folds)":
            (GroupKFold(n_splits=5), groups_cell, "groupcell"),
        "C. Leave-One-Genotype-Out (31 folds)":
            (LeaveOneGroupOut(), groups_geno, "logo"),
        "D. Leave-One-Year-Out (2 folds)":
            (LeaveOneYearOut(groups_year), None, "loyo"),
    }
