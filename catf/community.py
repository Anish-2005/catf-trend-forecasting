"""M3 - Community detection per epoch and tracking of community lineage across epochs."""
from __future__ import annotations

import networkx as nx
import pandas as pd
from networkx.algorithms.community import louvain_communities


def detect(G: nx.Graph, min_size: int = 8, resolution: float = 1.0, seed: int = 42) -> list[set]:
    """Louvain communities, dropping those smaller than `min_size`."""
    if G.number_of_edges() == 0:
        return []
    comms = louvain_communities(G, weight="weight", resolution=resolution, seed=seed)
    return [set(c) for c in comms if len(c) >= min_size]


def track(comms_by_epoch: dict[int, list[set]], jaccard_min: float = 0.30):
    """Match communities across consecutive epochs by member overlap (Jaccard).

    Returns (membership, events):
      membership - DataFrame[epoch, user, cid]   (cid is a persistent community id)
      events     - DataFrame[epoch, event, cid, other_cid, size] with
                   event in {birth, continue, merge, split, death}
    """
    next_id = 0
    prev: dict[int, set] = {}
    rows, events = [], []

    def new_id() -> int:
        nonlocal next_id
        next_id += 1
        return next_id - 1

    for e in sorted(comms_by_epoch):
        cur = sorted(comms_by_epoch[e], key=len, reverse=True)
        assigned: dict[int, int] = {}
        used: set[int] = set()
        pairs = []
        for i, c in enumerate(cur):
            for pid, p in prev.items():
                inter = len(c & p)
                if inter:
                    pairs.append((inter / len(c | p), i, pid))
        pairs.sort(reverse=True)
        for j, i, pid in pairs:
            if j < jaccard_min:
                break
            if i in assigned or pid in used:
                continue
            assigned[i] = pid
            used.add(pid)
            events.append((e, "continue", pid, None, len(cur[i])))

        merged: set[int] = set()
        for i, pid in list(assigned.items()):          # merge: other predecessors absorbed into pid
            for q, pset in prev.items():
                if q in used or q in merged:
                    continue
                if len(cur[i] & pset) / len(pset) >= 0.5:
                    merged.add(q)
                    events.append((e, "merge", pid, q, len(cur[i])))

        for i, c in enumerate(cur):
            if i in assigned:
                continue
            parent = next((pid for pid in used if len(c & prev[pid]) / len(c) >= 0.5), None)
            cid = new_id()
            assigned[i] = cid
            if parent is not None:                     # split: child carved out of a continuing parent
                events.append((e, "split", cid, parent, len(c)))
            else:
                events.append((e, "birth", cid, None, len(c)))

        for q in prev:                                 # death: predecessor neither continued nor merged
            if q not in used and q not in merged:
                events.append((e, "death", q, None, len(prev[q])))

        prev = {assigned[i]: cur[i] for i in range(len(cur))}
        for i, c in enumerate(cur):
            rows.extend((e, u, assigned[i]) for u in c)

    membership = pd.DataFrame(rows, columns=["epoch", "user", "cid"])
    ev = pd.DataFrame(events, columns=["epoch", "event", "cid", "other_cid", "size"])
    return membership, ev
