import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from arsenal import GroupKFoldArray, view_columns, wrap_target
from common_setup import GENOTYPE, SEED, TARGET_EN, YEAR


class Context:

    def __init__(self, PLANT, CELL):
        self.PLANT = PLANT.reset_index(drop=True)
        self.CELL = CELL.reset_index(drop=True)
        self.y_cell = self.CELL[TARGET_EN].values.astype(float)
        self.y_plant = self.PLANT[TARGET_EN].values.astype(float)
        key_c = (self.CELL[YEAR].astype(str) + "_" + self.CELL[GENOTYPE].astype(str)).values
        key_p = (self.PLANT[YEAR].astype(str) + "_" + self.PLANT[GENOTYPE].astype(str)).values
        self.cell_key = key_c
        self.plant_key = key_p
        self.rows_of_cell = [np.where(key_p == k)[0] for k in key_c]
        self.g_cell = key_c
        self.g_geno = self.CELL[GENOTYPE].astype(str).values
        self.g_year = self.CELL[YEAR].astype(str).values


def _pipe(reg, target, standardise=True):
    steps = [("impute", SimpleImputer(strategy="median"))]
    if standardise:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", wrap_target(reg, target)))
    return Pipeline(steps)


def direct(ctx, model_name, view, target, pool):
    cols = view_columns(view)
    Xc = ctx.CELL[cols].astype(float)

    def fit_predict(tr, te):
        p = _pipe(pool[model_name], target)
        p.fit(Xc.iloc[tr], ctx.y_cell[tr])
        return np.asarray(p.predict(Xc.iloc[te])).ravel()

    return fit_predict


def bottom_up(ctx, model_name, view, target, pool, aggregate="mean_of_predictions"):
    cols = view_columns(view)
    Xp = ctx.PLANT[cols].astype(float)
    Xc = ctx.CELL[cols].astype(float)

    def fit_predict(tr, te):
        rows = np.concatenate([ctx.rows_of_cell[i] for i in tr])
        p = _pipe(pool[model_name], target)
        p.fit(Xp.iloc[rows], ctx.y_plant[rows])
        if aggregate == "prediction_of_means":
            return np.asarray(p.predict(Xc.iloc[te])).ravel()
        out = np.empty(len(te), dtype=float)
        for j, i in enumerate(te):
            r = ctx.rows_of_cell[i]
            out[j] = float(np.mean(np.asarray(p.predict(Xp.iloc[r])).ravel()))
        return out

    return fit_predict


def _bootstrap_cells(ctx, tr, cols, n_aug, rng, size=None):
    Xp = ctx.PLANT[cols].astype(float).values
    yp = ctx.y_plant
    Xs, ys = [], []
    for i in tr:
        r = ctx.rows_of_cell[i]
        m = len(r) if size is None else size
        for _ in range(n_aug):
            pick = rng.choice(r, size=m, replace=True)
            Xs.append(Xp[pick].mean(axis=0))
            ys.append(yp[pick].mean())
    return np.asarray(Xs), np.asarray(ys)


def _jackknife_cells(ctx, tr, cols):
    Xp = ctx.PLANT[cols].astype(float).values
    yp = ctx.y_plant
    Xs, ys = [], []
    for i in tr:
        r = ctx.rows_of_cell[i]
        for drop in range(len(r)):
            keep = np.delete(r, drop)
            Xs.append(Xp[keep].mean(axis=0))
            ys.append(yp[keep].mean())
    return np.asarray(Xs), np.asarray(ys)


def _mixup_cells(ctx, tr, cols, n_aug, rng, alpha=0.4, c_mixup=False, bw=0.5):
    Xc = ctx.CELL[cols].astype(float).values[tr]
    yc = ctx.y_cell[tr]
    n = len(tr)
    Xs, ys = [], []
    for _ in range(n_aug):
        i = rng.randint(n)
        if c_mixup:
            d = np.abs(yc - yc[i])
            w = np.exp(-(d / (bw * (yc.std() + 1e-9))) ** 2)
            w[i] = 0.0
            if w.sum() <= 0:
                continue
            j = rng.choice(n, p=w / w.sum())
        else:
            j = rng.randint(n)
        lam = rng.beta(alpha, alpha)
        Xs.append(lam * Xc[i] + (1 - lam) * Xc[j])
        ys.append(lam * yc[i] + (1 - lam) * yc[j])
    return np.asarray(Xs), np.asarray(ys)


def _noise_cells(ctx, tr, cols, n_aug, rng, scale=1.0):
    Xp = ctx.PLANT[cols].astype(float).values
    Xc = ctx.CELL[cols].astype(float).values
    yc = ctx.y_cell
    Xs, ys = [], []
    for i in tr:
        r = ctx.rows_of_cell[i]
        se = Xp[r].std(axis=0, ddof=1) / np.sqrt(len(r))
        se_y = ctx.y_plant[r].std(ddof=1) / np.sqrt(len(r))
        for _ in range(n_aug):
            Xs.append(Xc[i] + scale * se * rng.randn(len(cols)))
            ys.append(yc[i] + scale * se_y * rng.randn())
    return np.asarray(Xs), np.asarray(ys)


AUGMENTERS = {
    "bootstrap": lambda ctx, tr, cols, n, rng: _bootstrap_cells(ctx, tr, cols, n, rng),
    "subsample7": lambda ctx, tr, cols, n, rng: _bootstrap_cells(ctx, tr, cols, n, rng, size=7),
    "jackknife": lambda ctx, tr, cols, n, rng: _jackknife_cells(ctx, tr, cols),
    "mixup": lambda ctx, tr, cols, n, rng: _mixup_cells(ctx, tr, cols, n * len(tr), rng),
    "cmixup": lambda ctx, tr, cols, n, rng: _mixup_cells(ctx, tr, cols, n * len(tr), rng,
                                                         c_mixup=True),
    "noise": lambda ctx, tr, cols, n, rng: _noise_cells(ctx, tr, cols, n, rng),
}


def augmented(ctx, model_name, view, target, pool, scheme, n_aug=20,
              keep_original=True, seed=SEED):
    cols = view_columns(view)
    Xc = ctx.CELL[cols].astype(float)

    def fit_predict(tr, te):
        rng = np.random.RandomState(seed + len(tr))
        Xa, ya = AUGMENTERS[scheme](ctx, tr, cols, n_aug, rng)
        if keep_original and len(Xa):
            Xa = np.vstack([Xc.values[tr], Xa])
            ya = np.concatenate([ctx.y_cell[tr], ya])
        elif not len(Xa):
            Xa, ya = Xc.values[tr], ctx.y_cell[tr]
        p = _pipe(pool[model_name], target)
        p.fit(pd.DataFrame(Xa, columns=cols), ya)
        return np.asarray(p.predict(Xc.iloc[te])).ravel()

    return fit_predict


def residual_hybrid(ctx, base_name, resid_name, view, target, pool):
    cols = view_columns(view)
    Xc = ctx.CELL[cols].astype(float)

    def fit_predict(tr, te):
        base = _pipe(pool[base_name], target)
        base.fit(Xc.iloc[tr], ctx.y_cell[tr])
        r_tr = ctx.y_cell[tr] - np.asarray(base.predict(Xc.iloc[tr])).ravel()
        head = _pipe(pool[resid_name], "raw")
        head.fit(Xc.iloc[tr], r_tr)
        return (np.asarray(base.predict(Xc.iloc[te])).ravel()
                + np.asarray(head.predict(Xc.iloc[te])).ravel())

    return fit_predict


def blend_median(members):

    def fit_predict(tr, te):
        return np.median(np.vstack([m(tr, te) for m in members]), axis=0)

    return fit_predict


def blend_fixed(members, weights=None):
    w = np.ones(len(members)) / len(members) if weights is None else np.asarray(weights,
                                                                               dtype=float)
    w = w / w.sum()

    def fit_predict(tr, te):
        preds = np.vstack([m(tr, te) for m in members])
        return np.average(preds, axis=0, weights=w)

    return fit_predict


def blend_learned(ctx, member_specs, build, n_splits=5, seed=SEED, nonneg=True):
    from scipy.optimize import nnls

    def fit_predict(tr, te):
        inner = GroupKFoldArray(ctx.g_cell[tr], n_splits=min(n_splits, len(tr)),
                                seed=seed)
        P = np.full((len(member_specs), len(tr)), np.nan)
        for a, b in inner.split(np.arange(len(tr))):
            for k, spec in enumerate(member_specs):
                try:
                    P[k, b] = build(spec)(tr[a], tr[b])
                except Exception:
                    P[k, b] = np.nan
        good = np.all(np.isfinite(P), axis=1)
        if good.sum() == 0:
            return build(member_specs[0])(tr, te)
        A = P[good].T
        yv = ctx.y_cell[tr]
        if nonneg:
            w, _ = nnls(A, yv)
            if w.sum() <= 0:
                w = np.ones(A.shape[1])
            w = w / w.sum()
        else:
            w = np.linalg.lstsq(A, yv, rcond=None)[0]
        full = np.vstack([build(s)(tr, te) for s, g in zip(member_specs, good) if g])
        return np.average(full, axis=0, weights=w)

    return fit_predict


def build_candidate(ctx, spec, pool):
    kind = spec[0]
    if kind == "direct":
        return direct(ctx, spec[1], spec[2], spec[3], pool)
    if kind == "bottomup":
        return bottom_up(ctx, spec[1], spec[2], spec[3], pool)
    if kind == "bottomup_pm":
        return bottom_up(ctx, spec[1], spec[2], spec[3], pool,
                         aggregate="prediction_of_means")
    if kind == "aug":
        return augmented(ctx, spec[1], spec[2], spec[3], pool, spec[4],
                         n_aug=spec[5] if len(spec) > 5 else 20)
    if kind == "resid":
        return residual_hybrid(ctx, spec[1], spec[2], spec[3], spec[4], pool)
    if kind == "blend":
        return blend_fixed([build_candidate(ctx, s, pool) for s in spec[1]])
    if kind == "blendmed":
        return blend_median([build_candidate(ctx, s, pool) for s in spec[1]])
    if kind == "blendfit":
        return blend_learned(ctx, list(spec[1]),
                             build=lambda s: build_candidate(ctx, s, pool))
    raise ValueError(f"unknown candidate kind: {kind}")


def spec_label(spec):
    kind = spec[0]
    if kind in ("direct", "bottomup", "bottomup_pm"):
        return f"{kind}:{spec[1]}|{spec[2]}|{spec[3]}"
    if kind == "aug":
        return f"aug[{spec[4]}]:{spec[1]}|{spec[2]}|{spec[3]}"
    if kind == "resid":
        return f"resid:{spec[1]}+{spec[2]}|{spec[3]}|{spec[4]}"
    if kind in ("blend", "blendmed", "blendfit"):
        return f"{kind}({len(spec[1])})"
    return str(spec)


def run_candidate(ctx, spec, pool, splitter):
    fp = build_candidate(ctx, spec, pool)
    oof = np.full(len(ctx.y_cell), np.nan)
    for tr, te in splitter.split():
        oof[te] = fp(tr, te)
    return oof
