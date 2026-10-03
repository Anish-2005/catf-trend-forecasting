import numpy as np
import pandas as pd
import pytest

from catf import community, data, evaluate, features, graph
from catf.config import load_config
from catf.lexicon import sentiment_score
from catf.pipeline import run


@pytest.fixture(scope="module")
def tiny():
    df, ev = data.generate_synthetic(n_epochs=20, n_communities=3, users_per_community=25, seed=1)
    df, _ = graph.assign_epochs(df, 24)
    return df, ev


def test_generator_is_deterministic():
    a, _ = data.generate_synthetic(10, 3, 20, seed=5)
    b, _ = data.generate_synthetic(10, 3, 20, seed=5)
    pd.testing.assert_frame_equal(a, b)


def test_epochs_and_snapshots(tiny):
    df, _ = tiny
    assert df["epoch"].min() == 0 and df["epoch"].max() == 19
    snaps = graph.build_snapshots(df, window=3)
    assert len(snaps) == 20
    assert all(d["weight"] > 0 for _, _, d in snaps[10].edges(data=True))
    assert snaps[10].number_of_edges() >= snaps[0].number_of_edges() * 0.5


def test_community_recovery(tiny):
    df, _ = tiny
    G = graph.build_snapshots(df, 3)[15]
    comms = community.detect(G, min_size=5, seed=1)
    assert len(comms) == 3


def test_tracking_events():
    A, B, C = set(range(0, 20)), set(range(20, 40)), set(range(40, 60))
    seq = {0: [A, B], 1: [A, B], 2: [A | B], 3: [A | B, C], 4: [A | B]}
    memb, ev = community.track(seq, jaccard_min=0.3)
    kinds = ev["event"].tolist()
    assert "merge" in kinds and "birth" in kinds and "death" in kinds
    assert memb[memb["epoch"] == 0]["cid"].nunique() == 2


def test_tracking_split():
    A = set(range(0, 40))
    seq = {0: [A], 1: [set(range(0, 25)), set(range(25, 40)) | set(range(100, 110))]}
    _, ev = community.track(seq, jaccard_min=0.3)
    assert "split" in ev["event"].tolist()


def test_sentiment():
    assert sentiment_score("great love this") == 1.0
    assert sentiment_score("terrible scam") == -1.0
    assert sentiment_score("the a is") == 0.0


def test_lt90_and_leads():
    alerts = pd.DataFrame({"cid": [0, 0, 1], "origin": [8, 9, 20]})
    events = pd.DataFrame({"cid": [0, 1, 2], "epoch": [11, 22, 30]})
    leads = evaluate.lead_times(alerts, events, horizon=5)
    assert leads == [3, 2, 0]                      # third event never warned
    assert evaluate.lt90(leads) == 2.0
    assert np.isnan(evaluate.lt90([0, 0]))


def test_topic_features_have_no_lookahead(tiny):
    df, _ = tiny
    snaps = graph.build_snapshots(df, 3)
    comms = {e: community.detect(G, 5, seed=1) for e, G in snaps.items()}
    memb, _ = community.track(comms)
    feats, words = features.extract_features(df, memb, fit_until_epoch=12, n_topics=4)
    assert {"activity", "sentiment", "topic_novelty", "growth"} <= set(feats.columns)
    assert len(words) == 4


def test_csv_validation(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"timestamp": ["2026-01-01"], "user": ["a"]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        data.load_csv(str(p))


def test_end_to_end(tmp_path):
    cfg = load_config(overrides={
        "data": {"synthetic": {"n_epochs": 60, "n_communities": 3, "users_per_community": 30}},
        "features": {"n_topics": 4}, "forecast": {"horizon": 3}})
    res = run(cfg, out_dir=str(tmp_path), verbose=False)
    for name in ["summary.json", "forecasts.csv", "metrics_accuracy.csv", "alerts.csv", "panel.csv"]:
        assert (tmp_path / name).exists()
    acc = res["accuracy"]
    assert set(acc.index) == {"naive", "global_arima", "community_arima", "hybrid"}
    assert np.isfinite(acc[["MAE", "RMSE", "sMAPE"]].to_numpy()).all()
    assert res["summary"]["community_nmi_mean"] > 0.8
