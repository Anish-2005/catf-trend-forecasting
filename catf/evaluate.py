"""M6 - Accuracy metrics, critical-mass events and the LT90 early-warning lead-time metric."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .warning import generate_alerts


def accuracy(forecasts: pd.DataFrame) -> pd.DataFrame:
    def smape(a, p):
        d = (np.abs(a) + np.abs(p)) / 2
        return float(np.mean(np.where(d == 0, 0, np.abs(a - p) / np.where(d == 0, 1, d))) * 100)

    rows = []
    for m, g in forecasts.groupby("model"):
        e = g["y_true"] - g["y_pred"]
        rows.append(dict(model=m, MAE=float(e.abs().mean()), RMSE=float(np.sqrt((e ** 2).mean())),
                         sMAPE=smape(g["y_true"].to_numpy(), g["y_pred"].to_numpy()), n=len(g)))
    return pd.DataFrame(rows).set_index("model")


def find_events(panel: dict[int, pd.DataFrame], thr: dict[int, float], split: int, cooldown: int = 7) -> pd.DataFrame:
    """Critical-mass events: upward crossings of the threshold in the test period."""
    rows = []
    for cid, d in panel.items():
        y = d["activity"].to_numpy()
        last = -10 ** 9
        for t in range(max(split, 1), len(y)):
            if y[t] >= thr[cid] and y[t - 1] < thr[cid] and t - last > cooldown:
                rows.append(dict(cid=cid, epoch=t))
                last = t
    return pd.DataFrame(rows, columns=["cid", "epoch"])


def lead_times(alerts: pd.DataFrame, events: pd.DataFrame, horizon: int):
    """For each event, lead = event epoch - earliest alert origin inside [event-H, event-1]."""
    leads = []
    for _, ev in events.iterrows():
        a = alerts[(alerts["cid"] == ev["cid"]) & (alerts["origin"] >= ev["epoch"] - horizon)
                   & (alerts["origin"] < ev["epoch"])]
        leads.append(int(ev["epoch"] - a["origin"].min()) if len(a) else 0)
    return leads


def lt90(leads: list[int]) -> float:
    """LT90: lead time (epochs) that at least 90 % of the *detected* events achieve or exceed,
    i.e. the 10th percentile of the lead-time distribution of detected events."""
    det = [x for x in leads if x > 0]
    return float(np.quantile(det, 0.10, method="lower")) if det else float("nan")


def early_warning(forecasts, panel, thr, events, horizon, models) -> pd.DataFrame:
    rows = []
    for m in models:
        al = generate_alerts(forecasts, panel, thr, m)
        leads = lead_times(al, events, horizon)
        tp = 0
        for _, a in al.iterrows():
            ev = events[events["cid"] == a["cid"]]
            tp += int(((ev["epoch"] > a["origin"]) & (ev["epoch"] <= a["origin"] + horizon)).any())
        det = [x for x in leads if x > 0]
        rows.append(dict(model=m, events=len(events), detected=len(det),
                         recall=len(det) / len(events) if len(events) else float("nan"),
                         alerts=len(al), alert_precision=tp / len(al) if len(al) else float("nan"),
                         mean_lead=float(np.mean(det)) if det else 0.0, LT90=lt90(leads)))
    return pd.DataFrame(rows).set_index("model")
