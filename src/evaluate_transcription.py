"""Objectively score any OCR output against the gold handwriting transcripts (train only has gold).

    python evaluate_transcription.py --tag q38                 # score out/ocr_train_q38.jsonl
    python evaluate_transcription.py --tag olmocr --file out/ocr_train_olmocr.jsonl

Compares on the PARSED working (the L-lines + FINAL that actually reach the reasoner, via answers.lines_and_final),
not raw text, so code-fences / prose preamble that the reasoner already ignores are not counted against a model.
Gold annotator notes ([...]) are stripped first (strip_notes) since they are absent at test time.

Metrics:
  content CER  - char edit distance on the whitespace/notation-normalized concatenation of the parsed lines
  content WER  - word edit distance on the same
  empty%       - rows whose parse yields ZERO L-lines (reasoner sees an empty working -> truly harmed)
  thin%        - rows whose parse yields exactly ONE L-line
  final-acc    - fraction of rows where the parsed FINAL answer matches gold's final line (answers.answer_matches)
These let two OCR models be compared apples-to-apples for the downstream task.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from answer_parsing import lines_and_final, strip_notes, answer_matches

ROOT = Path(__file__).resolve().parents[1]


def content(text):
    """Parsed working -> a single normalized string for edit-distance (notation/whitespace-insensitive)."""
    lines, _ = lines_and_final(text)
    s = " ".join(lines).lower()
    for a, b in [("−", "-"), ("–", "-"), ("×", "x"), ("·", "."), ("⇒", "=>"), ("÷", "/")]:
        s = s.replace(a, b)
    return "".join(s.split())            # drop all whitespace: line-splitting differences must not count


def gold_content(text):
    return content(text)                 # gold is already in L#| form; strip_notes applied by caller


def lev(a, b):
    """Levenshtein distance over sequences (str or list)."""
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return max(la, lb)
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * lb
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[lb]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="q38")
    ap.add_argument("--split", default="train", help="only 'train' has gold transcripts")
    ap.add_argument("--file", default="", help="override path to the ocr jsonl (else out/ocr_<split>_<tag>.jsonl)")
    ap.add_argument("--dump-worst", type=int, default=0, help="print this many worst-CER rows for inspection")
    args = ap.parse_args()

    df = pd.read_csv(ROOT / "data" / f"{args.split}.csv").set_index("trace_id")
    if "transcription_steps" not in df.columns:
        raise SystemExit(f"{args.split}.csv has no gold transcription_steps; CER needs train")
    path = Path(args.file) if args.file else ROOT / "out" / f"ocr_{args.split}_{args.tag}.jsonl"
    ocr = {}
    for l in open(path):
        r = json.loads(l)
        ocr[r["trace_id"]] = r["text"]

    cers, wers, rows = [], [], []
    empty = thin = final_ok = final_n = 0
    for tid, gold_raw in df["transcription_steps"].items():
        if tid not in ocr or pd.isna(gold_raw):
            continue
        g = gold_content(strip_notes(gold_raw))
        o = content(ocr[tid])
        if not g:
            continue
        lines, final = lines_and_final(ocr[tid])
        if len(lines) == 0:
            empty += 1
        elif len(lines) == 1:
            thin += 1
        # final-answer accuracy: gold final = last gold line's rhs (same rule lines_and_final uses)
        gl, gf = lines_and_final(strip_notes(gold_raw))
        if gf:
            final_n += 1
            final_ok += answer_matches(final, gf)
        cer = lev(o, g) / len(g)
        cers.append(cer)
        wers.append(lev(" ".join(lines_and_final(ocr[tid])[0]).split(),
                        " ".join(gl).split()) / max(1, len(" ".join(gl).split())))
        rows.append((cer, tid))

    n = len(cers)
    c = np.array(cers)
    w = np.array(wers)
    print(f"OCR eval: {path.name}  ({n} rows vs gold)")
    print(f"  content CER  mean {c.mean()*100:5.1f}%  median {np.median(c)*100:5.1f}%   (<10%: {(c<.1).mean()*100:.0f}%  <20%: {(c<.2).mean()*100:.0f}%  <30%: {(c<.3).mean()*100:.0f}%)")
    print(f"  content WER  mean {w.mean()*100:5.1f}%  median {np.median(w)*100:5.1f}%")
    print(f"  empty L-lines (harmed): {empty} ({100*empty/n:.1f}%)    thin (1 line): {thin} ({100*thin/n:.1f}%)")
    print(f"  final-answer match vs gold: {final_ok}/{final_n} ({100*final_ok/max(1,final_n):.1f}%)")
    if args.dump_worst:
        rows.sort(reverse=True)
        print(f"\n  --- {args.dump_worst} worst rows ---")
        for cer, tid in rows[:args.dump_worst]:
            print(f"  CER {cer*100:6.0f}%  {tid}")
            print(f"     gold: {gold_content(strip_notes(df.loc[tid,'transcription_steps']))[:110]}")
            print(f"     ocr : {content(ocr[tid])[:110]}")


if __name__ == "__main__":
    main()
