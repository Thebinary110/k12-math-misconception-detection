"""Baseline 0: hierarchical Dirichlet prior question -> topic -> global, pseudo-counts tuned on out-of-fold MAP@3."""
import itertools
import json

import numpy as np
import pandas as pd

from common import ROOT, classes, folds, load, map3, top3


def fit(df, cls, a_q, a_t):
    idx = {c: i for i, c in enumerate(cls)}
    y = df["root_code"].map(idx).values
    g = np.bincount(y, minlength=len(cls)) + 0.1
    g = g / g.sum()
    topic = {}
    for t, d in df.groupby("topic_name"):
        n = np.bincount(d["root_code"].map(idx), minlength=len(cls))
        topic[t] = (n + a_t * g) / (n.sum() + a_t)
    ques = {}
    for q, d in df.groupby("question_id"):
        n = np.bincount(d["root_code"].map(idx), minlength=len(cls))
        pt = topic.get(d["topic_name"].iloc[0], g)
        ques[q] = (n + a_q * pt) / (n.sum() + a_q)
    return g, topic, ques


def predict(df, model):
    g, topic, ques = model
    return np.stack([ques.get(q, topic.get(t, g)) for q, t in zip(df["question_id"], df["topic_name"])])


def oof(train, cls, a_q, a_t, splits):
    P = np.zeros((len(train), len(cls)))
    for tr_idx, va_idx in splits:
        P[va_idx] = predict(train.iloc[va_idx], fit(train.iloc[tr_idx], cls, a_q, a_t))
    return P


def main():
    train, test, taxonomy = load()
    cls = classes(train, taxonomy)
    y = train["root_code"].values
    splits = folds(train)
    grid = {}
    for a_q, a_t in itertools.product([1, 2, 4, 8, 16, 32, 64], [3, 10, 30, 100, 300, 1000]):
        grid[(a_q, a_t)] = map3(y, top3(oof(train, cls, a_q, a_t, splits), cls))
    (a_q, a_t), best = max(grid.items(), key=lambda kv: kv[1])
    # robustness: same setting on other fold seeds
    seeds = [map3(y, top3(oof(train, cls, a_q, a_t, folds(train, seed=s)), cls)) for s in range(5)]
    print(f"best a_q={a_q} a_t={a_t} OOF MAP@3={best:.4f} | 5 fold seeds: {np.mean(seeds):.4f} +- {np.std(seeds):.4f}")
    print(f"global-only prior: {map3(y, top3(oof(train, cls, 1e9, 1e9, splits), cls)):.4f}")

    P_te = predict(test, fit(train, cls, a_q, a_t))
    out = ROOT / "subs"
    out.mkdir(exist_ok=True)
    pd.DataFrame({"trace_id": test["trace_id"], "root_code": [" ".join(p) for p in top3(P_te, cls)]}).to_csv(
        out / "sub_prior.csv", index=False)
    np.save(out / "probs_prior_test.npy", P_te)
    json.dump({"a_q": a_q, "a_t": a_t, "oof_map3": best, "seed_mean": float(np.mean(seeds))},
              open(out / "prior.json", "w"))


if __name__ == "__main__":
    main()
