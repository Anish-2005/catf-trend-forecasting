"""M2 - Epoch binning and temporal graph snapshots."""
from __future__ import annotations

import networkx as nx
import pandas as pd


def assign_epochs(df: pd.DataFrame, length_hours: int = 24):
    """Add an integer `epoch` column. Returns (df, t0)."""
    t0 = df["timestamp"].min().normalize()
    step = pd.Timedelta(hours=length_hours)
    out = df.copy()
    out["epoch"] = ((out["timestamp"] - t0) // step).astype(int)
    return out, t0


def build_snapshots(df: pd.DataFrame, window: int = 3) -> dict[int, nx.Graph]:
    """One weighted undirected interaction graph per epoch.

    Snapshot G_t = (V_t, E_t) is built from the interactions of epochs t-window+1 .. t, which
    stabilises community detection on sparse epochs while remaining strictly causal.
    """
    edges = df.dropna(subset=["target"])
    edges = edges[edges["user"] != edges["target"]]
    per_epoch = {e: g for e, g in edges.groupby("epoch")}
    n_epochs = int(df["epoch"].max()) + 1
    snaps: dict[int, nx.Graph] = {}
    for t in range(n_epochs):
        parts = [per_epoch[e] for e in range(max(0, t - window + 1), t + 1) if e in per_epoch]
        G = nx.Graph()
        if parts:
            win = pd.concat(parts)
            pair = win.assign(a=win[["user", "target"]].min(axis=1), b=win[["user", "target"]].max(axis=1))
            w = pair.groupby(["a", "b"]).size()
            G.add_weighted_edges_from((a, b, int(c)) for (a, b), c in w.items())
        snaps[t] = G
    return snaps
