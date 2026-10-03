"""M4 - Feature extraction per (community, epoch): activity, structure, sentiment, topics."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

from .lexicon import sentiment_score


def fit_topics(texts: pd.Series, n_topics: int = 8, seed: int = 42):
    """TF-IDF + NMF topic model. Fit on training-period text only (no look-ahead)."""
    vec = TfidfVectorizer(max_features=3000, min_df=5, max_df=0.6)
    X = vec.fit_transform(texts)
    nmf = NMF(n_components=n_topics, init="nndsvda", random_state=seed, max_iter=1000, tol=1e-3)
    nmf.fit(X)
    vocab = np.array(vec.get_feature_names_out())
    top_words = {k: vocab[np.argsort(comp)[::-1][:8]].tolist() for k, comp in enumerate(nmf.components_)}
    return vec, nmf, top_words


def doc_topics(vec, nmf, texts: pd.Series) -> np.ndarray:
    W = nmf.transform(vec.transform(texts))
    s = W.sum(axis=1, keepdims=True)
    W = np.divide(W, s, out=np.full_like(W, 1.0 / W.shape[1]), where=s > 0)
    return W


def extract_features(df: pd.DataFrame, membership: pd.DataFrame, fit_until_epoch: int,
                     n_topics: int = 8, novelty_window: int = 7, seed: int = 42):
    """Return (features DataFrame[cid, epoch, ...], topic top-words dict)."""
    memb = membership.set_index(["epoch", "user"])["cid"]
    posts = df.join(memb.rename("cid"), on=["epoch", "user"])
    posts["tgt_cid"] = posts.join(memb.rename("tc"), on=["epoch", "target"])["tc"]
    posts = posts.dropna(subset=["cid"]).copy()
    posts["cid"] = posts["cid"].astype(int)
    posts["sent"] = posts["text"].map(sentiment_score)

    train_text = df.loc[df["epoch"] < fit_until_epoch, "text"]
    vec, nmf, top_words = fit_topics(train_text, n_topics, seed)
    W = doc_topics(vec, nmf, posts["text"])
    tcols = [f"topic_{k}" for k in range(n_topics)]
    posts[tcols] = W

    posts["has_tgt"] = posts["target"].notna()
    posts["internal"] = posts["has_tgt"] & (posts["tgt_cid"] == posts["cid"])
    g = posts.groupby(["cid", "epoch"])
    feat = pd.DataFrame({
        "activity": g.size(),
        "active_users": g["user"].nunique(),
        "sentiment": g["sent"].mean(),
        "internal_ratio": g["internal"].sum() / g["has_tgt"].sum().clip(lower=1),
    })
    feat = feat.join(g[tcols].mean()).reset_index().sort_values(["cid", "epoch"])

    out = []
    for cid, d in feat.groupby("cid"):
        d = d.sort_values("epoch").copy()
        T = d[tcols].to_numpy()
        base = pd.DataFrame(T).shift(1).rolling(novelty_window, min_periods=3).mean().to_numpy()
        nov = np.abs(T - base).sum(axis=1)
        d["topic_novelty"] = np.nan_to_num(nov, nan=0.0)
        d["growth"] = np.log1p(d["activity"]).diff().fillna(0.0)
        out.append(d)
    return pd.concat(out, ignore_index=True), top_words
