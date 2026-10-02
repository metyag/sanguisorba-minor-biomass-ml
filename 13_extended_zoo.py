# -*- coding: utf-8 -*-
"""
13 — Extended model zoo: can any modern method beat the allometric Huber baseline?

Everything is evaluated under the SAME leakage-free protocol as before (whole
genotype x year cells held out). Only the estimator changes.

Methods added beyond the seventeen of the manuscript
  Small-sample specialists      Gaussian process regression (three kernels),
                                kernel ridge, Bayesian ridge, ARD regression,
                                partial least squares, principal component regression
  Neural                        multi-layer perceptron (three widths)
  Tabular transformers          TabPFN v2 (a transformer pre-trained on synthetic
                                tabular tasks; designed for n < 1000 and <= 100
                                features, which is exactly this problem)
                                FT-Transformer (feature tokenizer + transformer
                                encoder, trained from scratch here)
  Boosting                      the existing XGBoost, plus LightGBM/CatBoost if present

NOTE ON VISION TRANSFORMERS. A ViT cannot be applied to these data: it consumes
images, and this dataset contains ten scalar measurements per plant, no imagery.
The transformer architectures used here are the tabular ones, which are the correct
analogue for this data type.

Levels
  CELL  : 62 genotype x year means  (the breeding decision unit)
  PLANT : 620 individual plants     (the strict per-plant question)

Outputs (results/tables/):
  13_extended_cell.csv, 13_extended_plant.csv, 13_extended_summary.json
"""
import sys, os, json, time, warnings
import numpy as np
import pandas as pd
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, RationalQuadratic, WhiteKernel, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, ARDRegression, Ridge
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (load_data, build_models, make_pipeline, TAB_DIR, mape, SEED,
                          MORPH_FEATURES, YEAR, GENOTYPE, TARGET_EN)

df, y, groups_cell, groups_geno, groups_year = load_data()

ENG = ["Stem volume proxy", "Leaf area proxy", "Height x stems",
       "Total leaf length", "Height squared"]


def add_engineered(frame):
    X = frame.copy()
    h, d, ns = X["Plant height"], X["Main stem diameter"], X["Number of main stems"]
    nl, npl = X["Number of leaves"], X["Number of leaflets per leaf"]
    ll, lw, lfl = X["Leaflet length"], X["Leaflet width"], X["Leaf length"]
    X["Stem volume proxy"] = np.pi * (d / 2.0) ** 2 * h * ns
    X["Leaf area proxy"] = nl * npl * ll * lw
    X["Height x stems"] = h * ns
    X["Total leaf length"] = nl * lfl
    X["Height squared"] = h ** 2
    return X


CELL = add_engineered(
    df.groupby([YEAR, GENOTYPE], as_index=False)[MORPH_FEATURES + [TARGET_EN]]
    .mean().sort_values([YEAR, GENOTYPE]).reset_index(drop=True))
PLANT = add_engineered(df)

# ---------------------------------------------------------------------------
# Optional heavy dependencies
# ---------------------------------------------------------------------------
HAS_TABPFN = HAS_TORCH = False
try:
    import torch
    HAS_TORCH = True
except Exception:
    pass
# TabPFN needs a one-time interactive licence acceptance at priorlabs.ai before it
# will download its weights. That cannot be done non-interactively, so it is disabled
# here; set TABPFN=1 in the environment after accepting the licence to enable it.
if os.environ.get("TABPFN", "0") == "1":
    try:
        from tabpfn import TabPFNRegressor
        HAS_TABPFN = True
    except Exception:
        pass
print(f"torch: {HAS_TORCH}   tabpfn: {HAS_TABPFN}", flush=True)


class FTTransformerRegressor(BaseEstimator, RegressorMixin):
    """
    Minimal Feature-Tokenizer Transformer for tabular regression (Gorishniy et al.).
    Each numeric feature is projected to a token; a CLS token is prepended; a small
    transformer encoder mixes them; the CLS embedding is read out by an MLP head.
    Inputs are already standardised by the surrounding pipeline.
    """

    def __init__(self, d_token=32, n_blocks=2, n_heads=4, dropout=0.1,
                 lr=1e-3, epochs=300, weight_decay=1e-4, batch_size=64, seed=SEED):
        self.d_token, self.n_blocks, self.n_heads, self.dropout = d_token, n_blocks, n_heads, dropout
        self.lr, self.epochs, self.weight_decay = lr, epochs, weight_decay
        self.batch_size, self.seed = batch_size, seed

    def _build(self, n_features):
        import torch.nn as nn
        torch.manual_seed(self.seed)

        class Net(nn.Module):
            def __init__(s, nf, d, nb, nh, do):
                super().__init__()
                s.w = nn.Parameter(torch.randn(nf, d) * 0.02)
                s.b = nn.Parameter(torch.zeros(nf, d))
                s.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
                layer = nn.TransformerEncoderLayer(d_model=d, nhead=nh,
                                                   dim_feedforward=2 * d, dropout=do,
                                                   batch_first=True, activation="gelu")
                s.enc = nn.TransformerEncoder(layer, num_layers=nb)
                s.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

            def forward(s, x):
                t = x.unsqueeze(-1) * s.w + s.b              # (B, F, d)
                t = torch.cat([s.cls.expand(x.shape[0], -1, -1), t], dim=1)
                t = s.enc(t)
                return s.head(t[:, 0]).squeeze(-1)

        return Net(n_features, self.d_token, self.n_blocks, self.n_heads, self.dropout)

    def fit(self, X, yy):
        import torch.nn as nn
        X = np.asarray(X, dtype=np.float32)
        yy = np.asarray(yy, dtype=np.float32)
        self._ymu, self._ysd = float(yy.mean()), float(yy.std() + 1e-8)
        yn = (yy - self._ymu) / self._ysd
        self.model_ = self._build(X.shape[1])
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        lossf = nn.MSELoss()
        Xt, yt = torch.tensor(X), torch.tensor(yn)
        n = len(Xt)
        g = torch.Generator().manual_seed(self.seed)
        self.model_.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = lossf(self.model_(Xt[idx]), yt[idx])
                loss.backward()
                opt.step()
            sched.step()
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            p = self.model_(torch.tensor(X)).numpy()
        return p * self._ysd + self._ymu


def extra_models(n_samples):
    """Estimators added on top of the manuscript's seventeen."""
    k_rbf = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(1e-2)
    k_mat = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=1.5) + WhiteKernel(1e-2)
    k_rq = ConstantKernel(1.0) * RationalQuadratic(length_scale=1.0) + WhiteKernel(1e-2)
    m = {
        "GP (RBF)": GaussianProcessRegressor(kernel=k_rbf, normalize_y=True,
                                             random_state=SEED, alpha=1e-8),
        "GP (Matern 1.5)": GaussianProcessRegressor(kernel=k_mat, normalize_y=True,
                                                    random_state=SEED, alpha=1e-8),
        "GP (RationalQuadratic)": GaussianProcessRegressor(kernel=k_rq, normalize_y=True,
                                                           random_state=SEED, alpha=1e-8),
        "Kernel Ridge (RBF)": KernelRidge(kernel="rbf", alpha=1.0, gamma=0.1),
        "Kernel Ridge (poly2)": KernelRidge(kernel="poly", degree=2, alpha=1.0),
        "Bayesian Ridge": BayesianRidge(),
        "ARD Regression": ARDRegression(),
        "PLS (3 comp)": PLSRegression(n_components=3),
        "PLS (5 comp)": PLSRegression(n_components=5),
        "PCR (4 comp)": SkPipeline([("pca", PCA(n_components=4, random_state=SEED)),
                                    ("lin", Ridge(alpha=1.0))]),
        "MLP (64-32)": MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=4000,
                                    random_state=SEED, early_stopping=False,
                                    learning_rate_init=3e-3, alpha=1e-3),
        "MLP (128-64-32)": MLPRegressor(hidden_layer_sizes=(128, 64, 32), max_iter=4000,
                                        random_state=SEED, early_stopping=False,
                                        learning_rate_init=3e-3, alpha=1e-3),
        "MLP (256)": MLPRegressor(hidden_layer_sizes=(256,), max_iter=4000,
                                  random_state=SEED, early_stopping=False,
                                  learning_rate_init=3e-3, alpha=1e-2),
    }
    if HAS_TABPFN:
        try:
            m["TabPFN (transformer)"] = TabPFNRegressor(device="cpu", random_state=SEED)
        except Exception as e:
            print("TabPFN unavailable:", e)
    if HAS_TORCH:
        m["FT-Transformer (d=32)"] = FTTransformerRegressor(d_token=32, n_blocks=2, epochs=300)
        m["FT-Transformer (d=64)"] = FTTransformerRegressor(d_token=64, n_blocks=3, epochs=400)
    try:
        from lightgbm import LGBMRegressor
        m["LightGBM"] = LGBMRegressor(n_estimators=600, learning_rate=0.05,
                                      random_state=SEED, verbose=-1)
    except Exception:
        pass
    try:
        from catboost import CatBoostRegressor
        m["CatBoost"] = CatBoostRegressor(iterations=800, learning_rate=0.05,
                                          random_seed=SEED, verbose=0)
    except Exception:
        pass
    return m


class GroupOut:
    def __init__(self, g):
        self.g = np.asarray(g)

    def split(self, X, y=None, groups=None):
        for u in np.unique(self.g):
            yield np.where(self.g != u)[0], np.where(self.g == u)[0]

    def get_n_splits(self, *a, **k):
        return len(np.unique(self.g))


def evaluate(reg, frame, target, feats, log_t, splitter, groups):
    X, yy = frame[feats], frame[target].values
    oof = np.full(len(yy), np.nan)
    for tr, te in splitter.split(X, yy, groups=groups):
        r = (TransformedTargetRegressor(regressor=clone(reg), func=np.log1p,
                                        inverse_func=np.expm1) if log_t else clone(reg))
        p = make_pipeline(r, feats, genotype_categorical=False)
        p.fit(X.iloc[tr], yy[tr])
        pred = np.asarray(p.predict(X.iloc[te])).ravel()
        oof[te] = pred
    m = ~np.isnan(oof)
    return {"pooled_R2": float(r2_score(yy[m], oof[m])),
            "pooled_RMSE": float(np.sqrt(mean_squared_error(yy[m], oof[m]))),
            "pooled_MAE": float(mean_absolute_error(yy[m], oof[m])),
            "pooled_MAPE": float(mape(yy[m], oof[m]))}


FEATURE_SETS = {
    "A. Morphology only (8)": MORPH_FEATURES,
    "B. Morphology + allometric (13)": MORPH_FEATURES + ENG,
    "C. Season + morphology (9)": [YEAR] + MORPH_FEATURES,
    "D. Season + morphology + allometric (14)": [YEAR] + MORPH_FEATURES + ENG,
}

print("=" * 104)
print("13 — EXTENDED MODEL ZOO")
print("=" * 104, flush=True)

# ===========================================================================
# CELL LEVEL
# ===========================================================================
g_cell_c = (CELL[YEAR].astype(str) + "_" + CELL[GENOTYPE].astype(str)).values
g_geno_c = CELL[GENOTYPE].astype(str).values
g_year_c = CELL[YEAR].astype(str).values

ALL = {**build_models(), **extra_models(len(CELL))}
print(f"\nCELL LEVEL — {len(CELL)} cells, {len(ALL)} estimators\n", flush=True)

rows, t0 = [], time.time()
for fs_name, feats in FEATURE_SETS.items():
    for t_name, log_t in [("raw", False), ("log", True)]:
        for m_name, reg in ALL.items():
            try:
                met = evaluate(reg, CELL, TARGET_EN, feats, log_t,
                               GroupKFold(n_splits=5), g_cell_c)
                rows.append({"Feature set": fs_name, "Target": t_name, "Model": m_name,
                             "Status": "ok", **met})
            except Exception as e:
                rows.append({"Feature set": fs_name, "Target": t_name, "Model": m_name,
                             "Status": f"FAILED: {str(e)[:90]}"})
        sub = [r for r in rows if r["Feature set"] == fs_name and r["Target"] == t_name
               and r["Status"] == "ok"]
        if sub:
            b = max(sub, key=lambda r: r["pooled_R2"])
            print(f"  {fs_name:<42}{t_name:<5}best {b['Model']:<24}"
                  f"R2={b['pooled_R2']:7.4f}  MAE={b['pooled_MAE']:7.2f}  "
                  f"MAPE={b['pooled_MAPE']:6.2f}", flush=True)

cellres = pd.DataFrame(rows)
cellres.to_csv(os.path.join(TAB_DIR, "13_extended_cell.csv"), index=False)
okc = cellres[cellres.Status == "ok"].copy()
print(f"\nCell-level elapsed {time.time()-t0:.0f} s; "
      f"failures {int((cellres.Status!='ok').sum())}")

print("\n" + "=" * 104)
print("CELL LEVEL — top 15 by grouped 5-fold R2")
print("=" * 104)
top = okc.sort_values("pooled_R2", ascending=False).head(15)
print(top[["Feature set", "Target", "Model", "pooled_R2", "pooled_RMSE",
           "pooled_MAE", "pooled_MAPE"]].to_string(index=False,
                                                   float_format=lambda v: f"{v:.4f}"))

# confirm the shortlist under the harder designs
print("\n" + "-" * 104)
print("CELL LEVEL — shortlist under leave-one-cell-out / genotype-out / year-out")
print("-" * 104, flush=True)
conf = []
for _, cfg in okc.sort_values("pooled_R2", ascending=False).head(6).iterrows():
    feats = FEATURE_SETS[cfg["Feature set"]]
    log_t = cfg["Target"] == "log"
    reg = ALL[cfg["Model"]]
    row = {"Feature set": cfg["Feature set"], "Target": cfg["Target"],
           "Model": cfg["Model"], "5FOLD_R2": float(cfg["pooled_R2"])}
    for dname, spl in [("LOCO", GroupOut(g_cell_c)), ("LOGO", GroupOut(g_geno_c)),
                       ("LOYO", GroupOut(g_year_c))]:
        try:
            met = evaluate(reg, CELL, TARGET_EN, feats, log_t, spl, None)
            row[f"{dname}_R2"] = met["pooled_R2"]
            row[f"{dname}_MAE"] = met["pooled_MAE"]
            row[f"{dname}_MAPE"] = met["pooled_MAPE"]
        except Exception as e:
            row[f"{dname}_R2"] = np.nan
    conf.append(row)
    print(f"  {cfg['Model']:<24}{cfg['Feature set'][:30]:<32}{cfg['Target']:<4}"
          f"5F={row['5FOLD_R2']:6.4f}  LOCO={row.get('LOCO_R2', float('nan')):6.4f}  "
          f"LOGO={row.get('LOGO_R2', float('nan')):6.4f}  "
          f"LOYO={row.get('LOYO_R2', float('nan')):7.4f}", flush=True)
confdf = pd.DataFrame(conf)
confdf.to_csv(os.path.join(TAB_DIR, "13_extended_cell_confirm.csv"), index=False)

# ===========================================================================
# PLANT LEVEL
# ===========================================================================
print("\n" + "=" * 104)
print(f"PLANT LEVEL — {len(PLANT)} plants")
print("=" * 104, flush=True)
ALLP = {**build_models(), **extra_models(len(PLANT))}
# GPs are O(n^3); skip them at n = 620 to keep the run tractable
for k in list(ALLP):
    if k.startswith("GP ("):
        ALLP.pop(k)

rows, t0 = [], time.time()
for fs_name, feats in FEATURE_SETS.items():
    for t_name, log_t in [("raw", False), ("log", True)]:
        for m_name, reg in ALLP.items():
            try:
                met = evaluate(reg, PLANT, TARGET_EN, feats, log_t,
                               GroupKFold(n_splits=5), groups_cell)
                rows.append({"Feature set": fs_name, "Target": t_name, "Model": m_name,
                             "Status": "ok", **met})
            except Exception as e:
                rows.append({"Feature set": fs_name, "Target": t_name, "Model": m_name,
                             "Status": f"FAILED: {str(e)[:90]}"})
        sub = [r for r in rows if r["Feature set"] == fs_name and r["Target"] == t_name
               and r["Status"] == "ok"]
        if sub:
            b = max(sub, key=lambda r: r["pooled_R2"])
            print(f"  {fs_name:<42}{t_name:<5}best {b['Model']:<24}"
                  f"R2={b['pooled_R2']:7.4f}  MAE={b['pooled_MAE']:7.2f}  "
                  f"MAPE={b['pooled_MAPE']:6.2f}", flush=True)

plantres = pd.DataFrame(rows)
plantres.to_csv(os.path.join(TAB_DIR, "13_extended_plant.csv"), index=False)
okp = plantres[plantres.Status == "ok"].copy()
print(f"\nPlant-level elapsed {time.time()-t0:.0f} s; "
      f"failures {int((plantres.Status!='ok').sum())}")
print("\nPLANT LEVEL — top 10")
print(okp.sort_values("pooled_R2", ascending=False).head(10)[
    ["Feature set", "Target", "Model", "pooled_R2", "pooled_MAE", "pooled_MAPE"]]
    .to_string(index=False, float_format=lambda v: f"{v:.4f}"))

summary = {
    "has_torch": HAS_TORCH, "has_tabpfn": HAS_TABPFN,
    "n_estimators_cell": len(ALL), "n_estimators_plant": len(ALLP),
    "cell_best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                  for k, v in okc.loc[okc.pooled_R2.idxmax()].items() if k != "Status"},
    "plant_best": {k: (float(v) if isinstance(v, (int, float, np.floating)) else str(v))
                   for k, v in okp.loc[okp.pooled_R2.idxmax()].items() if k != "Status"},
    "cell_confirmation": conf,
    "note": ("A vision transformer is not applicable: these data are ten scalar "
             "measurements per plant, not images. The transformer architectures "
             "evaluated are the tabular ones (TabPFN, FT-Transformer)."),
}
with open(os.path.join(TAB_DIR, "13_extended_summary.json"), "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, default=str)
print("\n" + json.dumps(summary["cell_best"], indent=2, default=str))
print(json.dumps(summary["plant_best"], indent=2, default=str))
