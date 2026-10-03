"""M6 (part) - critical-mass thresholds and early-warning alerts."""
from __future__ import annotations

import pandas as pd


def thresholds(panel: dict[int, pd.DataFrame], split: int, k: float = 2.0) -> dict[int, float]:
    """Critical mass per community = train mean + k * train std of activity (train period only)."""
    out = {}
    for cid, d in panel.items():
        y = d["activity"].to_numpy()[:split]
        out[cid] = float(y.mean() + k * y.std())
    return out


def generate_alerts(forecasts: pd.DataFrame, panel: dict[int, pd.DataFrame], thr: dict[int, float],
                    model: str) -> pd.DataFrame:
    """Alert at origin t when activity is below critical mass now but the forecast crosses it
    within the horizon."""
    f = forecasts[forecasts["model"] == model]
    rows = []
    for (cid, origin), g in f.groupby(["cid", "origin"]):
        y_now = float(panel[cid].loc[origin, "activity"])
        peak_row = g.loc[g["y_pred"].idxmax()]
        if y_now < thr[cid] and peak_row["y_pred"] >= thr[cid]:
            rows.append(dict(cid=int(cid), origin=int(origin), expected_epoch=int(peak_row["target_epoch"]),
                             predicted_peak=round(float(peak_row["y_pred"]), 1), threshold=round(thr[cid], 1),
                             severity=round(float(peak_row["y_pred"]) / thr[cid], 2)))
    cols = ["cid", "origin", "expected_epoch", "predicted_peak", "threshold", "severity"]
    return pd.DataFrame(rows, columns=cols)
