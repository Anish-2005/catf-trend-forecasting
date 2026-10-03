"""M1 - Data ingestion: synthetic community-structured generator and CSV loader.

Interaction schema (one row per post / reply):
    timestamp  - ISO string or unix seconds
    user       - author id
    target     - user replied to / mentioned (edge target)
    text       - post text
    true_community (optional, synthetic only) - planted label used for evaluation
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .lexicon import NEGATIVE, POSITIVE

TOPICS = {
    "technology": "software cloud chip release update developer opensource code device battery network launch".split(),
    "finance": "market stock bank invest rate inflation budget fund trade earnings loan economy".split(),
    "health": "doctor clinic vaccine diet fitness sleep therapy hospital medicine wellness symptom checkup".split(),
    "sports": "match league coach goal season player stadium score tournament team training final".split(),
    "gaming": "console level quest multiplayer stream patch esports server controller mod dungeon speedrun".split(),
    "politics": "election policy senate vote budgetbill council debate campaign reform ministry law cabinet".split(),
    "music": "album concert guitar playlist vocal tour band lyrics studio festival melody single".split(),
    "travel": "flight hotel beach visa itinerary trek island passport hostel route sightseeing ticket".split(),
}
EMERGING = "breaking rumor leak viral shutdown recall surge exposed trending alert".split()
FILLER = "the a is this that really just today people think about new see more some one very".split()

EXPECTED_COLUMNS = ["timestamp", "user", "target", "text"]


def generate_synthetic(n_epochs: int = 120, n_communities: int = 6, users_per_community: int = 60,
                       seed: int = 42, epoch_hours: int = 24, start: str = "2026-01-01"):
    """Generate an interaction log with planted, drifting communities and trend bursts.

    Community drift (migration, merge, split) happens before 65 % of the timeline so that the
    chronological test period is structurally stable. Trend bursts carry *precursor* signals
    (emerging vocabulary and sentiment shift) a few epochs before activity peaks - this is the
    modelling assumption the hybrid model exploits. Results on this data validate the pipeline,
    not the science.

    Returns (interactions DataFrame, planted_events DataFrame).
    """
    rng = np.random.default_rng(seed)
    K, U, T = n_communities, users_per_community, n_epochs
    n_users = K * U
    users = np.array([f"user_{i:04d}" for i in range(n_users)])
    label = np.repeat(np.arange(K), U)
    topic_names = list(TOPICS)
    topic_of = {c: topic_names[c % len(topic_names)] for c in range(K + 1)}

    rate_per_user = {c: rng.uniform(0.9, 1.4) for c in range(K + 1)}
    phase = {c: rng.uniform(0, 2 * np.pi) for c in range(K + 1)}

    # --- planned drift (only if enough communities) ---
    t_mig, t_merge, t_split = int(0.30 * T), int(0.45 * T), int(0.55 * T)
    drift = K >= 6

    # --- planned trend bursts: (community, onset fraction, polarity) ---
    plan = [(0, 0.17, -1), (1, 0.38, 1), (2, 0.50, -1),
            (0, 0.70, -1), (2, 0.73, 1), (3, 0.77, -1), (1, 0.80, 1),
            (5, 0.83, -1), (6, 0.86, 1), (0, 0.88, -1), (2, 0.92, 1), (3, 0.95, -1)]
    events = []
    for cid, frac, pol in plan:
        if cid >= K + (1 if drift else 0) or (cid == K and not drift):
            continue
        onset = min(int(frac * T), T - 9)
        events.append(dict(community=cid, onset=onset, ramp=6, decay=5.0,
                           peak=float(rng.uniform(3.0, 4.2)), polarity=pol))
    ev_df = pd.DataFrame(events)

    def burst(c: int, t: int):
        mult, p_em, pol_shift = 1.0, 0.0, 0.0
        for ev in events:
            if ev["community"] != c:
                continue
            o, R = ev["onset"], ev["ramp"]
            if o - 3 <= t < o:                        # precursor window
                p_em = max(p_em, 0.08 + 0.07 * (t - (o - 3)))
                pol_shift = ev["polarity"] * (0.15 + 0.1 * (t - (o - 3)))
            elif o <= t < o + R:                      # ramp up
                mult *= 1 + (ev["peak"] - 1) * (t - o + 1) / R
                p_em, pol_shift = max(p_em, 0.40), ev["polarity"] * 0.5
            elif t >= o + R:                          # decay
                d = np.exp(-(t - o - R + 1) / ev["decay"])
                mult *= 1 + (ev["peak"] - 1) * d
                p_em, pol_shift = max(p_em, 0.40 * d), ev["polarity"] * 0.5 * d
        return mult, p_em, pol_shift

    t0 = pd.Timestamp(start)
    rows = []
    for t in range(T):
        if drift:
            if t == t_mig:
                label[np.where(label == 1)[0][:8]] = 2
            if t == t_merge:
                label[label == 4] = 3
            if t == t_split:
                idx = np.where(label == 5)[0]
                label[idx[len(idx) // 2:]] = K
        epoch_start = t0 + pd.Timedelta(hours=epoch_hours * t)
        for c in np.unique(label):
            members = np.where(label == c)[0]
            season = 1 + 0.2 * np.sin(2 * np.pi * t / 7 + phase[c])
            mult, p_em, pol_shift = burst(int(c), t)
            lam = rate_per_user[int(c)] * len(members) * season * mult * rng.lognormal(0, 0.12)
            n = rng.poisson(lam)
            if n == 0:
                continue
            w = 1 / (np.arange(len(members)) + 1) ** 0.6
            authors = rng.choice(members, n, p=w / w.sum())
            internal = rng.random(n) < 0.85
            tgt_in = rng.choice(members, n)
            tgt_out = rng.integers(0, n_users, n)
            targets = np.where(internal, tgt_in, tgt_out)
            targets = np.where(targets == authors, (targets + 1) % n_users, targets)
            secs = rng.uniform(0, epoch_hours * 3600, n)
            p_sent = 0.15
            p_topic = max(0.2, 0.55 - p_em)
            probs = np.array([p_topic, p_em, p_sent, max(0.0, 1 - p_topic - p_em - p_sent)])
            probs /= probs.sum()
            lens = rng.integers(7, 13, n)
            cats = rng.choice(4, size=(n, 12), p=probs)
            pos_prob = float(np.clip(0.6 + pol_shift, 0.02, 0.98))
            pos_flag = rng.random((n, 12)) < pos_prob
            tw = topic_of[int(c)]
            for i in range(n):
                words = []
                for j in range(lens[i]):
                    k = cats[i, j]
                    if k == 0:
                        words.append(TOPICS[tw][rng.integers(len(TOPICS[tw]))])
                    elif k == 1:
                        words.append(EMERGING[rng.integers(len(EMERGING))])
                    elif k == 2:
                        pool = sorted(POSITIVE) if pos_flag[i, j] else sorted(NEGATIVE)
                        words.append(pool[rng.integers(len(pool))])
                    else:
                        words.append(FILLER[rng.integers(len(FILLER))])
                rows.append((epoch_start + pd.Timedelta(seconds=float(secs[i])), users[authors[i]],
                             users[targets[i]], " ".join(words), int(c)))
    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS + ["true_community"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df, ev_df


def load_csv(path: str) -> pd.DataFrame:
    """Load a real interaction log (e.g. Reddit comments exported to CSV)."""
    df = pd.read_csv(path)
    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {missing}. "
                         f"Expected columns: {EXPECTED_COLUMNS} (+ optional true_community).")
    ts = df["timestamp"]
    if pd.api.types.is_numeric_dtype(ts):
        df["timestamp"] = pd.to_datetime(ts, unit="s")
    else:
        df["timestamp"] = pd.to_datetime(ts)
    df = df.dropna(subset=["timestamp", "user"]).copy()
    df["text"] = df["text"].fillna("").astype(str)
    return df.sort_values("timestamp").reset_index(drop=True)


def anonymise(df: pd.DataFrame) -> pd.DataFrame:
    """Replace user identifiers by stable pseudonyms (NFR-04 privacy)."""
    ids = pd.unique(pd.concat([df["user"], df["target"].dropna()]))
    mapping = {u: f"u{i:06d}" for i, u in enumerate(ids)}
    out = df.copy()
    out["user"] = out["user"].map(mapping)
    out["target"] = out["target"].map(mapping)
    return out
