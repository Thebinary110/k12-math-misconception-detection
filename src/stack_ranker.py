"""Candidate-level stacking: every source proposes codes; a LightGBM ranker scores (row, candidate) pairs from all
sources' probabilities/ranks plus answer-check and taxonomy features. Out-of-fold on the shared folds.

    python stack_ranker.py --llm q38 --ft ft9b          # sources that are complete on train AND test
"""
import process_naming  # noqa: F401  (process name; must run before vLLM starts)
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from hierarchical_prior import fit, predict
from tfidf_baseline import features as text_features, lr_probs, read_ocr
from blend_predictions import llm_probs, rule_match
from common import ROOT, classes, folds, load, map3

TOPK = 8


def sources(train, test, cls, args):
    """name -> (P_train_oof, P_test). Every train matrix is out-of-fold."""
    yi = train["root_code"].map({c: i for i, c in enumerate(cls)}).values
    splits = folds(train)
    src = {}
    P = np.zeros((len(train), len(cls)))
    for tr_idx, va_idx in splits:
        P[va_idx] = predict(train.iloc[va_idx], fit(train.iloc[tr_idx], cls, 8, 30))
    src["prior"] = (P, predict(test, fit(train, cls, 8, 30)))
    t_tr, m_tr = text_features(train, read_ocr("train", "q38"))
    t_te, m_te = text_features(test, read_ocr("test", "q38"))
    cache = ROOT / "out" / "tfidf_q38.npz"
    if cache.exists():
        z = np.load(cache)
        src["tfidf"] = (z["oof"], z["test"])
    else:
        P = np.zeros((len(train), len(cls)))
        for tr_idx, va_idx in splits:
            P[va_idx] = lr_probs([t_tr[i] for i in tr_idx], m_tr[tr_idx], yi[tr_idx], [t_tr[i] for i in va_idx],
                                 m_tr[va_idx], len(cls))
        src["tfidf"] = (P, lr_probs(t_tr, m_tr, yi, t_te, m_te, len(cls)))
        np.savez(cache, oof=src["tfidf"][0], test=src["tfidf"][1])
    signals_tr, signals_te = {"rule": m_tr}, {"rule": m_te}
    for tag in args.llm:
        P_tr, pc_tr, _ = llm_probs(train, tag, "train", cls)
        P_te, pc_te, _ = llm_probs(test, tag, "test", cls)
        src[f"llm_{tag}"] = (P_tr, P_te)
        signals_tr[f"ac_{tag}"], signals_te[f"ac_{tag}"] = pc_tr, pc_te
    for tag in args.ft:
        src[f"ft_{tag}"] = (np.load(ROOT / "out" / f"ft_{tag}_oof.npy"), np.load(ROOT / "out" / f"ft_{tag}_test.npy"))
    for rt in getattr(args, 'retr', []) or []:
        sfx = f'_{rt}' if rt else ''
        name = f'retr{sfx}' if rt else 'retr'
        src[name] = (np.load(ROOT / 'out' / f'retr{sfx}_oof.npy'), np.load(ROOT / 'out' / f'retr{sfx}_test.npy'))
    return src, signals_tr, signals_te


def candidate_frame(src, signals, which, cls, nature):
    names = list(src)
    mats = [src[n][which] for n in names]
    n_rows = mats[0].shape[0]
    ranks = [np.argsort(np.argsort(-m, axis=1), axis=1) for m in mats]
    none = cls.index("NONE")
    rows = []
    for i in range(n_rows):
        cand = {none}
        for m in mats:
            cand.update(np.argsort(-m[i])[:TOPK].tolist())
        for c in cand:
            f = {"row": i, "cand": c, "is_none": int(c == none), "nature": nature.get(cls[c], -1)}
            for n, m, r in zip(names, mats, ranks):
                f[f"p_{n}"] = m[i, c]
                f[f"r_{n}"] = r[i, c]
            for k, v in signals.items():
                f[k] = v[i]
            f["p_mean"] = np.mean([m[i, c] for m in mats])
            f["r_min"] = min(r[i, c] for r in ranks)
            rows.append(f)
    return pd.DataFrame(rows)


def ranker():
    return lgb.LGBMRanker(objective="lambdarank", n_estimators=300, learning_rate=0.03, num_leaves=15,
                          min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                          lambdarank_truncation_level=3, verbose=-1)


def top3_from(frame, scores, n_rows, cls):
    frame = frame.assign(s=scores).sort_values(["row", "s"], ascending=[True, False])
    out = [[] for _ in range(n_rows)]
    for r, c in zip(frame["row"], frame["cand"]):
        if len(out[r]) < 3:
            out[r].append(cls[c])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", nargs="*", default=["q38"])
    ap.add_argument("--retr", nargs="*", default=[], help="retrieval source tags; e.g. --retr '' base (loads out/retr[_<tag>]_{oof,test}.npy)")
    ap.add_argument("--ft", nargs="*", default=[])
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    train, test, taxonomy = load()
    cls = classes(train, taxonomy)
    nature = {r.id: i for i, r in enumerate(taxonomy.itertuples())}
    nature = {k: ["conceptual", "procedural", "careless", "meta"].index(v) if v in
              ["conceptual", "procedural", "careless", "meta"] else -1
              for k, v in zip(taxonomy["id"], taxonomy["nature"])}
    y = train["root_code"].values
    src, sig_tr, sig_te = sources(train, test, cls, args)
    for n, (P, _) in src.items():
        print(f"source {n:14s} OOF MAP@3 {map3(y, [[cls[j] for j in np.argsort(-p)[:3]] for p in P]):.4f}")

    F = candidate_frame(src, sig_tr, 0, cls, nature)
    F["label"] = [int(cls[c] == y[r]) for r, c in zip(F["row"], F["cand"])]
    print(f"candidates per row {len(F) / len(train):.1f} | gold covered {F.groupby('row')['label'].max().mean():.3f}")
    feats = [c for c in F.columns if c not in ("row", "cand", "label")]
    fold_of = np.zeros(len(train), int)
    for k, (_, va) in enumerate(folds(train)):
        fold_of[va] = k
    oof_scores = np.zeros(len(F))
    for k in range(fold_of.max() + 1):
        tr_m, va_m = fold_of[F["row"]] != k, fold_of[F["row"]] == k
        Ft = F[tr_m].sort_values("row")
        m = ranker().fit(Ft[feats], Ft["label"], group=Ft.groupby("row").size().values)
        oof_scores[va_m] = m.predict(F.loc[va_m, feats])
    stack_oof = map3(y, top3_from(F, oof_scores, len(train), cls))
    print(f"STACK OOF MAP@3 {stack_oof:.4f}")

    Fs = F.sort_values("row")
    m = ranker().fit(Fs[feats], Fs["label"], group=Fs.groupby("row").size().values)
    T = candidate_frame(src, sig_te, 1, cls, nature)
    preds = top3_from(T, m.predict(T[feats]), len(test), cls)
    name = args.name or "sub_stack_" + "+".join(list(src)[2:]) + ".csv"
    pd.DataFrame({"trace_id": test["trace_id"], "root_code": [" ".join(p) for p in preds]}).to_csv(
        ROOT / "subs" / name, index=False)
    json.dump({"sources": list(src), "oof": stack_oof}, open(ROOT / "subs" / (name + ".json"), "w"))
    imp = pd.Series(m.feature_importances_, feats).sort_values(ascending=False).head(10)
    print("top features:", imp.to_dict())
    print("wrote", name)


if __name__ == "__main__":
    main()
