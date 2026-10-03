"""M5 - Forecasting engine: baselines and the ARIMA + neural-residual hybrid.

All models work on log1p(activity). Evaluation is strictly chronological (walk-forward):
parameters are fitted on the training period only, then the ARIMA state is *appended* with
newly observed epochs (no refit, no look-ahead) at every forecast origin.

Hybrid = ARIMA forecast + learned correction of ARIMA's residuals from lagged residuals and
community covariates (sentiment shift, topic novelty, interaction structure, growth).
The residual learner sits behind a small interface so the Sem-8 LSTM can replace it.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

COVARIATES = ["sent_delta", "topic_novelty", "internal_ratio", "growth", "log_users"]


def build_panel(features: pd.DataFrame, n_epochs: int, min_coverage: float = 0.9) -> dict[int, pd.DataFrame]:
    """Dense per-community time series (gaps interpolated) for communities that persist."""
    panel = {}
    for cid, d in features.groupby("cid"):
        if d["epoch"].nunique() < min_coverage * n_epochs:
            continue
        d = d.set_index("epoch").reindex(range(n_epochs))
        d["activity"] = d["activity"].interpolate(limit_direction="both")
        for c in ["active_users", "sentiment", "internal_ratio", "topic_novelty", "growth"]:
            d[c] = d[c].interpolate(limit_direction="both")
        d["sent_delta"] = d["sentiment"] - d["sentiment"].shift(1).rolling(3, min_periods=1).mean()
        d["sent_delta"] = d["sent_delta"].fillna(0.0)
        d["log_users"] = np.log1p(d["active_users"])
        panel[int(cid)] = d[["activity", "active_users", "sentiment"] + COVARIATES].copy()
    return panel


def _fit_arima(y: np.ndarray, order):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(y, order=tuple(order)).fit()


def _append(res, y_new: np.ndarray):
    if len(y_new) == 0:
        return res
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return res.append(y_new, refit=False)


def _make_residual_model(kind: str, seed: int):
    if kind == "gbr":
        return MultiOutputRegressor(GradientBoostingRegressor(n_estimators=150, max_depth=2,
                                                              learning_rate=0.05, subsample=0.8,
                                                              random_state=seed))
    return make_pipeline(StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=(32, 16), alpha=0.1, max_iter=1500,
                                      random_state=seed))


def _row(resid: np.ndarray, cov: np.ndarray, t: int, lags: int) -> np.ndarray:
    return np.concatenate([[resid[t - j] for j in range(lags)], cov[t]])


def run_forecasts(panel: dict[int, pd.DataFrame], n_epochs: int, cfg: dict) -> tuple[pd.DataFrame, int]:
    """Walk-forward forecasts for every model. Returns (forecasts DataFrame, split epoch)."""
    H = cfg["horizon"]
    split = int(n_epochs * (1 - cfg["test_fraction"]))
    order, lags = cfg["arima_order"], cfg["hybrid_lags"]
    burn = 8                                             # ignore ARIMA start-up residuals
    cids = sorted(panel)

    ys = {c: np.log1p(panel[c]["activity"].to_numpy()) for c in cids}
    covs = {c: panel[c][COVARIATES].to_numpy() for c in cids}
    fits = {c: _fit_arima(ys[c][:split], order) for c in cids}

    # ---- global ARIMA on the total activity of all tracked communities ----
    total = np.log1p(sum(panel[c]["activity"].to_numpy() for c in cids))
    gfit = _fit_arima(total[:split], order)

    # ---- train the residual learner on the training period only (pooled over communities) ----
    X, Y = [], []
    for c in cids:
        r = fits[c].resid
        for t in range(burn + lags, split - H):
            X.append(_row(r, covs[c], t, lags))
            Y.append(r[t + 1:t + 1 + H])
    res_model = _make_residual_model(cfg["residual_model"], cfg["seed"]).fit(np.array(X), np.array(Y))

    rows = []
    for origin in range(split - 1, n_epochs - 1):
        hh = min(H, n_epochs - 1 - origin)
        g_app = _append(gfit, total[split:origin + 1])
        g_fc = np.expm1(g_app.forecast(hh))
        shares = {c: panel[c]["activity"].iloc[origin] for c in cids}
        s_tot = sum(shares.values()) or 1.0
        for c in cids:
            app = _append(fits[c], ys[c][split:origin + 1])
            fc = app.forecast(hh)
            corr = res_model.predict(_row(app.resid, covs[c], origin, lags).reshape(1, -1))[0][:hh]
            preds = {
                "naive": np.full(hh, np.expm1(ys[c][origin])),
                "global_arima": g_fc * shares[c] / s_tot,
                "community_arima": np.expm1(fc),
                "hybrid": np.expm1(fc + np.clip(corr, -1.5, 1.5)),
            }
            for m, p in preds.items():
                for h in range(hh):
                    tgt = origin + 1 + h
                    rows.append((m, c, origin, h + 1, tgt, float(panel[c]["activity"].iloc[tgt]),
                                 float(max(p[h], 0.0))))
    f = pd.DataFrame(rows, columns=["model", "cid", "origin", "h", "target_epoch", "y_true", "y_pred"])
    return f, split
