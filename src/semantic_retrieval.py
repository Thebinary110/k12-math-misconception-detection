"""Semantic kNN retrieval over labelled workings: embed (topic + question + OCR working), find the nearest labelled
train neighbours, and borrow their root-code distribution. A cheap, no-LLM source for the blend/stack — the retrieval
signal that the top misconception solutions lean on. Train predictions are out-of-fold on the shared CV folds.

    python retrieval_knn.py                 # report OOF MAP@3 and cache out/retr_{oof,test}.npy
    python retrieval_knn.py --k 20 --temp 0.1

Caches embeddings in out/emb_{train,test}.npy so re-tuning k/temp costs no GPU.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from answer_parsing import lines_and_final
from common import ROOT, classes, folds, load, map3, top3


def working_text(ocr):
    lines, final = lines_and_final(ocr)
    return " ; ".join(lines) + (f" ; final: {final}" if final else "")


def build_texts(df, split, ocr_tag):
    ocr = {}
    f = ROOT / "out" / f"ocr_{split}_{ocr_tag}.jsonl"
    for l in open(f):
        r = json.loads(l)
        ocr[r["trace_id"]] = r["text"]
    texts = []
    for r in df.itertuples():
        w = working_text(ocr.get(r.trace_id, ""))
        texts.append(f"query: topic: {r.topic_name} | question: {r.question_text} | working: {w}")
    return texts


def embed(model, texts, cache):
    if cache.exists():
        return np.load(cache)
    E = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
    E = np.asarray(E, dtype=np.float32)
    np.save(cache, E)
    return E


def knn_probs(E_q, E_ref, y_ref, n_cls, k, temp, mask_same=None):
    """Weighted vote of the k nearest reference rows' labels. mask_same[i] = index in ref to exclude (self), or -1."""
    sims = E_q @ E_ref.T                              # cosine (rows are normalized)
    if mask_same is not None:
        for i, j in enumerate(mask_same):
            if j >= 0:
                sims[i, j] = -1e9
    P = np.zeros((E_q.shape[0], n_cls), dtype=np.float32)
    for i in range(E_q.shape[0]):
        idx = np.argpartition(-sims[i], k)[:k]
        w = np.exp(sims[i, idx] / temp)
        w /= w.sum() + 1e-9
        for j, wj in zip(idx, w):
            P[i, y_ref[j]] += wj
    return P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "models" / "mE5-small"))
    ap.add_argument("--ocr-tag", default="q38")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--tag", default="", help="suffix for cache/output files, so several embedders coexist as sources")
    args = ap.parse_args()
    sfx = f"_{args.tag}" if args.tag else ""

    train, test, _ = load()
    cls = classes(train, pd_read_taxonomy())
    yi = train["root_code"].map({c: i for i, c in enumerate(cls)}).values

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    Etr = embed(model, build_texts(train, "train", args.ocr_tag), ROOT / "out" / f"emb_train{sfx}.npy")
    Ete = embed(model, build_texts(test, "test", args.ocr_tag), ROOT / "out" / f"emb_test{sfx}.npy")

    # OOF for train: neighbours only from the other folds
    P_oof = np.zeros((len(train), len(cls)), dtype=np.float32)
    for tr_idx, va_idx in folds(train):
        P_oof[va_idx] = knn_probs(Etr[va_idx], Etr[tr_idx], yi[tr_idx], len(cls), args.k, args.temp)
    # test: neighbours from all train
    P_test = knn_probs(Ete, Etr, yi, len(cls), args.k, args.temp)

    oof_map3 = map3(train["root_code"].values, top3(P_oof, cls))
    print(f"retrieval kNN [{args.tag or 'mE5-small'}] k={args.k} temp={args.temp}  OOF MAP@3 = {oof_map3:.4f}")
    np.save(ROOT / "out" / f"retr{sfx}_oof.npy", P_oof)
    np.save(ROOT / "out" / f"retr{sfx}_test.npy", P_test)
    print(f"wrote out/retr{sfx}_oof.npy, out/retr{sfx}_test.npy")


def pd_read_taxonomy():
    import pandas as pd
    return pd.read_csv(ROOT / "data" / "taxonomy_v1.csv")


if __name__ == "__main__":
    main()
