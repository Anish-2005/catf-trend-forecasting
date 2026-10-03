"""End-to-end CATF pipeline (M1 -> M6). Writes everything the dashboard (M7) reads."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score

from . import community, data, evaluate, features, forecast, graph, warning
from .config import ROOT, load_config


def run(cfg: dict | None = None, input_csv: str | None = None, out_dir: str | None = None,
        verbose: bool = True) -> dict:
    cfg = cfg or load_config()
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    out = Path(out_dir or cfg["output_dir"])
    out = out if out.is_absolute() else ROOT / out
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    # M1 ---------------------------------------------------------------------------------
    truth_events = None
    csv_path = input_csv or (cfg["data"]["csv_path"] if cfg["data"]["source"] == "csv" else None)
    if csv_path:
        df = data.load_csv(csv_path)
        log(f"[M1] loaded {len(df):,} interactions from {csv_path}")
    else:
        s = cfg["data"]["synthetic"]
        df, truth_events = data.generate_synthetic(s["n_epochs"], s["n_communities"],
                                                   s["users_per_community"], s["seed"],
                                                   cfg["epoch"]["length_hours"])
        log(f"[M1] generated {len(df):,} synthetic interactions")
    df, _ = graph.assign_epochs(df, cfg["epoch"]["length_hours"])
    n_epochs = int(df["epoch"].max()) + 1

    # M2 ---------------------------------------------------------------------------------
    snaps = graph.build_snapshots(df, cfg["epoch"]["window_epochs"])
    log(f"[M2] built {len(snaps)} temporal graph snapshots")

    # M3 ---------------------------------------------------------------------------------
    cc = cfg["community"]
    warm = cfg["epoch"]["window_epochs"] - 1          # skip warm-up epochs with a partial window
    comms = {e: ([] if e < warm else community.detect(G, cc["min_size"], cc["resolution"], cc["seed"]))
             for e, G in snaps.items()}
    membership, events = community.track(comms, cc["match_jaccard"])
    log(f"[M3] {membership['cid'].nunique()} tracked communities; "
        f"lineage events: {events['event'].value_counts().to_dict()}")

    # M4 ---------------------------------------------------------------------------------
    fc = cfg["forecast"]
    split = int(n_epochs * (1 - fc["test_fraction"]))
    feats, topic_words = features.extract_features(df, membership, split, cfg["features"]["n_topics"],
                                                   cfg["features"]["novelty_window"], cfg["community"]["seed"])
    log(f"[M4] features for {feats['cid'].nunique()} communities x {n_epochs} epochs")

    # M5 ---------------------------------------------------------------------------------
    panel = forecast.build_panel(feats, n_epochs, fc["min_coverage"])
    if not panel:
        raise RuntimeError("No community persists long enough to forecast; lower forecast.min_coverage.")
    forecasts, split = forecast.run_forecasts(panel, n_epochs, fc)
    log(f"[M5] {len(forecasts):,} forecasts from {forecasts['model'].nunique()} models "
        f"on {len(panel)} communities (split at epoch {split})")

    # M6 ---------------------------------------------------------------------------------
    wc = cfg["warning"]
    thr = warning.thresholds(panel, split, wc["critical_k"])
    ev = evaluate.find_events(panel, thr, split, wc["cooldown_epochs"])
    acc = evaluate.accuracy(forecasts)
    ew = evaluate.early_warning(forecasts, panel, thr, ev, fc["horizon"], list(acc.index))
    alerts = warning.generate_alerts(forecasts, panel, thr, wc["alert_model"])
    log(f"[M6] {len(ev)} critical-mass events in the test period\n{acc.round(2)}\n{ew.round(2)}")

    summary = dict(
        n_interactions=int(len(df)), n_epochs=n_epochs, split_epoch=split, horizon=fc["horizon"],
        n_tracked_communities=int(membership["cid"].nunique()), n_forecast_communities=len(panel),
        n_events=int(len(ev)), runtime_seconds=round(time.time() - t_start, 1),
        synthetic=truth_events is not None,
    )
    if "true_community" in df.columns:
        post = df.join(membership.set_index(["epoch", "user"])["cid"], on=["epoch", "user"]).dropna(subset=["cid"])
        nmi = [normalized_mutual_info_score(g["true_community"], g["cid"]) for _, g in post.groupby("epoch")]
        summary["community_nmi_mean"] = round(float(np.mean(nmi)), 3)

    # persist ----------------------------------------------------------------------------
    membership.to_csv(out / "membership.csv", index=False)
    events.to_csv(out / "lineage_events.csv", index=False)
    feats.to_csv(out / "features.csv", index=False)
    forecasts.to_csv(out / "forecasts.csv", index=False)
    alerts.to_csv(out / "alerts.csv", index=False)
    ev.to_csv(out / "critical_events.csv", index=False)
    acc.to_csv(out / "metrics_accuracy.csv")
    ew.to_csv(out / "metrics_early_warning.csv")
    pd.DataFrame({"cid": list(thr), "threshold": list(thr.values())}).to_csv(out / "thresholds.csv", index=False)
    pd.concat([d.assign(cid=c, epoch=d.index) for c, d in panel.items()]).to_csv(out / "panel.csv", index=False)
    (out / "topics.json").write_text(json.dumps(topic_words, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    if truth_events is not None:
        truth_events.to_csv(out / "planted_events.csv", index=False)
    log(f"[done] {summary['runtime_seconds']}s -> {out}")
    return dict(summary=summary, accuracy=acc, early_warning=ew, alerts=alerts, out_dir=out)
