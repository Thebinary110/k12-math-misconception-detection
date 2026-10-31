"""Blend LLM diagnoses (one or more runs) with the hierarchical prior; weights tuned on train OOF MAP@3.

    python blend_predictions.py q38 [q38think ...]      # diag_{train,test}_<tag>.jsonl
"""
import process_naming  # noqa: F401  (process name; must run before vLLM starts)
import itertools
import json
import sys

import numpy as np
import pandas as pd

from answer_parsing import answer_matches, lines_and_final
from hierarchical_prior import fit, predict
from common import ROOT, classes, folds, load, map3, top3

RANK_W = (1.0, 0.5, 1 / 3)


def llm_probs(df, tag, split, cls):
    idx = {c: i for i, c in enumerate(cls)}
    rows = {r["trace_id"]: r for r in map(json.loads, open(ROOT / "out" / f"diag_{split}_{tag}.jsonl"))}
    P = np.zeros((len(df), len(cls)))
    pc = np.full(len(df), 0.5)
    for i, t in enumerate(df["trace_id"]):
        r = rows.get(t)
        if r is None:
            continue
        tops = r["top3"] if r["top3"] and isinstance(r["top3"][0], list) else [r["top3"]]
        acs = r["answer_correct"] if isinstance(r["answer_correct"], list) else [r["answer_correct"]]
        for top in tops:
            for k, c in enumerate(top):
                P[i, idx[c]] += RANK_W[k]
        votes = [a for a in acs if a is not None]
        if votes:
            pc[i] = np.mean(votes)
        if P[i].sum():
            P[i] /= P[i].sum()
    return P, pc, len(rows)


def rule_match(df, split, ocr_tag="q38"):
    ocr = {r["trace_id"]: r["text"] for r in map(json.loads, open(ROOT / "out" / f"ocr_{split}_{ocr_tag}.jsonl"))}
    return np.array([answer_matches(lines_and_final(ocr.get(t, ""))[1], k)
                     for t, k in zip(df["trace_id"], df["correct_answer"])], dtype=float)


def combine(P_llms, pc, rule, P_prior, none, w):
    """w = (llm weights..., prior weight, answer-signal strength)."""
    *wl, wp, s = w
    P = sum(a * p for a, p in zip(wl, P_llms)) + wp * P_prior
    p_ok = 0.5 * pc + 0.5 * rule
    P = P * np.where(np.arange(P.shape[1]) == none, 1.0, 1.0)[None]
    P[:, none] *= np.exp(s * (p_ok - 0.5))
    return P


def main(tags):
    train, test, taxonomy = load()
    cls = classes(train, taxonomy)
    none = cls.index("NONE")
    y = train["root_code"].values
    P_prior = np.zeros((len(train), len(cls)))
    for tr_idx, va_idx in folds(train):
        P_prior[va_idx] = predict(train.iloc[va_idx], fit(train.iloc[tr_idx], cls, 8, 30))
    llm_tr = [llm_probs(train, t, "train", cls) for t in tags]
    for t, (P, pc, n) in zip(tags, llm_tr):
        print(f"{t}: {n} train rows | LLM alone MAP@3 {map3(y, top3(P + 1e-6 * P_prior, cls)):.4f} | "
              f"answer-correct vs gold flag {np.mean((pc > 0.5) == train.final_answer_correct.values):.3f}")
    pc_tr = np.mean([x[1] for x in llm_tr], axis=0)
    rule_tr = rule_match(train, "train")
    grid = [g for g in itertools.product(*([[0.2, 0.4, 0.6, 0.8, 1.0]] * len(tags)), [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.2],
                                         [0, 1, 2, 4, 8])]
    best = max(((map3(y, top3(combine([x[0] for x in llm_tr], pc_tr, rule_tr, P_prior, none, w), cls)), w)
                for w in grid), key=lambda t: t[0])
    print(f"blend OOF MAP@3 {best[0]:.4f} weights {best[1]}")

    llm_te = [llm_probs(test, t, "test", cls) for t in tags]
    if any(n < len(test) for _, _, n in llm_te):
        print("test diagnoses incomplete; no submission written")
        return
    P_te = combine([x[0] for x in llm_te], np.mean([x[1] for x in llm_te], axis=0), rule_match(test, "test"),
                   predict(test, fit(train, cls, 8, 30)), none, best[1])
    name = "sub_blend_" + "+".join(tags) + ".csv"
    pd.DataFrame({"trace_id": test["trace_id"], "root_code": [" ".join(p) for p in top3(P_te, cls)]}).to_csv(
        ROOT / "subs" / name, index=False)
    json.dump({"tags": tags, "oof": best[0], "weights": best[1]}, open(ROOT / "subs" / (name + ".json"), "w"))
    print("wrote", name)


if __name__ == "__main__":
    main(sys.argv[1:] or ["q38"])
