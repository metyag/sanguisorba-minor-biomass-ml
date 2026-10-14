import json
import os
import sys

import numpy as np
from sklearn.inspection import partial_dependence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common_setup import (TAB_DIR, MORPH_FEATURES, TARGET_EN, load_data,
                          build_models, make_pipeline)

GRID = 40
OZELLIKLER = ["Plant height", "Leaf length"]


def _egri_ozeti(g, v):
    d = np.diff(v)
    artis = np.flatnonzero(d > 0)
    pencere = 5
    if len(d) >= pencere:
        toplam = np.convolve(d, np.ones(pencere), mode="valid")
        i = int(np.argmax(toplam))
        dik = (float(g[i]), float(g[i + pencere]))
    else:
        dik = (float(g[0]), float(g[-1]))
    if len(d) >= pencere:
        j = int(np.argmin(np.abs(np.convolve(d, np.ones(pencere), mode="valid"))))
        duz = (float(g[j]), float(g[j + pencere]))
    else:
        duz = (float(g[0]), float(g[-1]))
    return {
        "grid_min": float(g.min()), "grid_max": float(g.max()),
        "pdp_min": float(v.min()), "pdp_max": float(v.max()),
        "span_g": float(v.max() - v.min()),
        "net_rise_g": float(v[-1] - v[0]),
        "monotone_increasing": bool((d >= -1e-9).all()),
        "n_decreasing_steps": int((d < -1e-9).sum()),
        "largest_decrease_g": float(d.min()),
        "steepest_from_cm": dik[0], "steepest_to_cm": dik[1],
        "flattest_from_cm": duz[0], "flattest_to_cm": duz[1],
        "n_increasing_steps": int(len(artis)),
        "n_steps": int(len(d)),
    }


def main():
    r = load_data()
    df = r[0] if isinstance(r, tuple) else r
    X = df[MORPH_FEATURES].astype(float)
    y = df[TARGET_EN].values.astype(float)

    pipe = make_pipeline(build_models()["ExtraTrees"], MORPH_FEATURES,
                         genotype_categorical=True)
    pipe.fit(X, y)

    cikti = {"estimator": "ExtraTrees", "feature_set": "MORPH",
             "grid_resolution": GRID, "features": {}}
    for f in OZELLIKLER:
        p = partial_dependence(pipe, X, [f], grid_resolution=GRID, kind="average")
        g = np.asarray(p["grid_values"][0], dtype=float)
        v = np.asarray(p["average"][0], dtype=float)
        cikti["features"][f] = _egri_ozeti(g, v)
        cikti["features"][f]["grid"] = [round(float(x), 4) for x in g]
        cikti["features"][f]["pdp"] = [round(float(x), 4) for x in v]

    yol = os.path.join(TAB_DIR, "05_pdp_summary.json")
    with open(yol, "w", encoding="utf-8") as fh:
        json.dump(cikti, fh, indent=1)
    print("written:", yol)
    for f, d in cikti["features"].items():
        print("  %-14s span %.1f g | monotone %s | %d decreasing steps of %d"
              % (f, d["span_g"], d["monotone_increasing"],
                 d["n_decreasing_steps"], d["n_steps"]))


if __name__ == "__main__":
    main()
