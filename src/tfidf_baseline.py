"""Baseline 1 (no LLM): prior + TF-IDF/LR on the VLM transcription + rule-based answer check, blended on OOF MAP@3."""
import itertools
import json
import sys

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from answer_parsing import answer_matches, lines_and_final
from hierarchical_prior import fit, predict
from common import ROOT, classes, folds, load, map3, top3


def read_ocr(split, tag):
    return {r["trace_id"]: r["text"] for r in map(json.loads, open(ROOT / "out" / f"ocr_{split}_{tag}.jsonl"))}


def features(df, ocr):
    text, match = [], []
    for r in df.itertuples():
        lines, final = lines_and_final(ocr.get(r.trace_id, ""))
        text.append(r.question_text + "\n" + "\n".join(lines) + "\nFINAL " + final)
        match.append(answer_matches(final, r.correct_answer))
    return text, np.array(match, dtype=float)


def lr_probs(t_fit, m_fit, y_fit, t_pred, m_pred, n_cls):
    w = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X = hstack([w.fit_transform(t_fit), csr_matrix(m_fit[:, None] * 3.0)]).tocsr()
    Xp = hstack([w.transform(t_pred), csr_matrix(m_pred[:, None] * 3.0)]).tocsr()
    clf = LogisticRegression(C=4.0, max_iter=3000).fit(X, y_fit)
    P = np.zeros((Xp.shape[0], n_cls))
    P[:, clf.classes_] = clf.predict_proba(Xp)
    return P


def flag_adjust(P, match, none_idx, up, down):
    P = P.copy()
    P[match == 1, none_idx] += up
    P[match == 0, none_idx] *= down
    return P / P.sum(1, keepdims=True)


def main(tag="q38"):
    train, test, taxonomy = load()
    cls = classes(train, taxonomy)
    idx = {c: i for i, c in enumerate(cls)}
    none = idx["NONE"]
    y, yi = train["root_code"].values, train["root_code"].map(idx).values
    t_tr, m_tr = features(train, read_ocr("train", tag))
    splits = folds(train)

    P_prior, P_lr = np.zeros((len(train), len(cls))), np.zeros((len(train), len(cls)))
    for tr_idx, va_idx in splits:
        P_prior[va_idx] = predict(train.iloc[va_idx], fit(train.iloc[tr_idx], cls, 8, 30))
        P_lr[va_idx] = lr_probs([t_tr[i] for i in tr_idx], m_tr[tr_idx], yi[tr_idx],
                                [t_tr[i] for i in va_idx], m_tr[va_idx], len(cls))
    print(f"prior {map3(y, top3(P_prior, cls)):.4f} | LR on VLM text {map3(y, top3(P_lr, cls)):.4f}")
    best = (0, None)
    for a, up, down in itertools.product([0, .2, .4, .5, .6, .8], [0, .5, 1, 2], [1, .5, .2, .05]):
        P = a * P_lr + (1 - a) * flag_adjust(P_prior, m_tr, none, up, down)
        s = map3(y, top3(P, cls))
        if s > best[0]:
            best = (s, (a, up, down))
    print(f"best blend OOF MAP@3 {best[0]:.4f} with (a, up, down) = {best[1]}")

    try:
        ocr_te = read_ocr("test", tag)
    except FileNotFoundError:
        return
    if len(ocr_te) < len(test):
        print(f"test OCR incomplete ({len(ocr_te)}/{len(test)}); no submission written")
        return
    a, up, down = best[1]
    t_te, m_te = features(test, ocr_te)
    P_te = a * lr_probs(t_tr, m_tr, yi, t_te, m_te, len(cls)) + (1 - a) * flag_adjust(
        predict(test, fit(train, cls, 8, 30)), m_te, none, up, down)
    pd.DataFrame({"trace_id": test["trace_id"], "root_code": [" ".join(p) for p in top3(P_te, cls)]}).to_csv(
        ROOT / "subs" / "sub_text.csv", index=False)
    print("wrote subs/sub_text.csv")


if __name__ == "__main__":
    main(*sys.argv[1:])
