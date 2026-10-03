"""Small sentiment lexicon (prototype). Swap for VADER / a transformer model in Sem 8."""
POSITIVE = set("""great love amazing excellent happy win good awesome impressive success
brilliant fantastic proud wonderful improved gain strong reliable""".split())
NEGATIVE = set("""terrible worry panic awful scam crash fear bad broken fail failure
angry shortage outage danger risk loss weak fraud""".split())


def sentiment_score(text: str) -> float:
    """Return a score in [-1, 1]: (pos - neg) / (pos + neg); 0 when no lexicon word occurs."""
    pos = neg = 0
    for tok in str(text).lower().split():
        if tok in POSITIVE:
            pos += 1
        elif tok in NEGATIVE:
            neg += 1
    total = pos + neg
    return 0.0 if total == 0 else (pos - neg) / total
