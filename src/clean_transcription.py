"""Conservatively clean an OCR jsonl so more rows parse into usable L-lines, WITHOUT disturbing good rows.

    python clean_transcription.py --tag q38            # writes out/ocr_<split>_<tag>_clean.jsonl
    python evaluate_transcription.py --tag q38_clean       # then re-score to see the delta

Policy (safety first): a row that already yields >=2 L-lines under answers.lines_and_final is passed through
BYTE-IDENTICAL. Only rows that currently parse to 0 or 1 line are salvaged, by recognising the markdown tag
variants the strict parser misses (**L1:**, L1:, **Line 1:**, "1. Line 1:", backtick-quoted math) and dropping
pure-commentary lines (Note:, Wait, ...). Consecutive duplicate lines and a FINAL line that merely repeats the
last L-line are de-duplicated for every row (harmless). Salvage never invents content: if nothing structured is
found the original text is kept so the reasoner's own fallbacks still apply.
"""
import argparse
import json
import re
from pathlib import Path

from answer_parsing import lines_and_final

ROOT = Path(__file__).resolve().parents[1]

# a line that is really a tagged transcription line, in any of the observed markdown variants ->  (index, content)
TAG = re.compile(r"^\s*(?:[-*>#.\d\s]*)?(?:\*\*)?\s*(?:L|Line|Baris|Langkah)\s*(\d+)\s*(?:\*\*)?\s*[:|]\s*\*{0,2}\s*(.*?)\s*\*{0,2}\s*$",
                 re.I)
FINALTAG = re.compile(r"^\s*(?:\*\*)?\s*(?:FINAL|Final answer|Jawaban akhir)\s*(?:\*\*)?\s*[:|]\s*\*{0,2}\s*(.*?)\s*\*{0,2}\s*$",
                      re.I)
# lines that are commentary, not transcription
NOISE = re.compile(r"^\s*(?:\*|-|>|#|\(|\*\()?\s*(note|wait|this (line|is)|it looks|the student|looking at|let'?s|however|implying|contextually|likely|---+|based on)",
                   re.I)
BACKTICK = re.compile(r"`([^`]+)`")
MATHY = re.compile(r"[0-9=+\-×÷√/]")


def strip_md(s):
    s = s.strip().strip("*").strip()
    s = s.replace("`", "")
    s = re.sub(r"\s*\((?:Note|note)[^)]*\)\s*", " ", s)   # inline (Note: ...)
    return s.strip()


def salvage(text):
    """Return a list of transcription line-contents recovered from a chatty/markdown output, or []."""
    out = []
    lines = str(text).splitlines()
    for i, raw in enumerate(lines):
        m = TAG.match(raw)
        if not m:
            continue
        content = strip_md(m.group(2))
        if not content:                       # tag with the math on the following line(s) -> grab a backtick expr
            for nxt in lines[i + 1:i + 3]:
                bt = BACKTICK.search(nxt)
                if bt and MATHY.search(bt.group(1)):
                    content = strip_md(bt.group(1))
                    break
        if content and not NOISE.match(content):
            out.append(content)
    return out


def dedup(lines, final):
    """Drop consecutive duplicate lines and a FINAL that just repeats the last line."""
    ded = []
    for ln in lines:
        if not ded or ded[-1] != ln:
            ded.append(ln)
    if final and ded and final.strip() == ded[-1].strip():
        pass  # keep; FINAL repeating the last line is fine for the parser, harmless
    return ded, final


def clean_text(text):
    """Return (clean_text, changed, recovered). Minimal-diff: a row is rewritten ONLY when salvage recovers
    extra lines or de-dup actually removes a duplicate; otherwise the original text is returned untouched so
    the diff stays auditable and good rows are provably unchanged."""
    lines, final = lines_and_final(text)
    recovered = False
    if len(lines) < 2:
        sal = salvage(text)
        if len(sal) > len(lines):
            lines = sal
            recovered = True
            if not final:
                final = lines[-1].split("=")[-1].strip() if lines else ""   # reasoner's own rule
    deduped, final = dedup(lines, final)
    # only touch the row if we recovered content or removed a real duplicate line
    if not recovered and deduped == lines:
        return text, False, False
    if not deduped:
        return text, False, False
    rebuilt = "\n".join(f"L{i} | {ln}" for i, ln in enumerate(deduped, 1)) + f"\nFINAL | {final or 'none'}"
    return rebuilt, True, recovered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="q38")
    ap.add_argument("--split", default="train")
    ap.add_argument("--file", default="")
    ap.add_argument("--out-tag", default="", help="output tag (default <tag>_clean)")
    args = ap.parse_args()

    src = Path(args.file) if args.file else ROOT / "out" / f"ocr_{args.split}_{args.tag}.jsonl"
    out_tag = args.out_tag or f"{args.tag}_clean"
    dst = ROOT / "out" / f"ocr_{args.split}_{out_tag}.jsonl"

    n = changed = recovered = 0
    with open(dst, "w") as w:
        for l in open(src):
            r = json.loads(l)
            n += 1
            ct, ch, rec = clean_text(r["text"])
            changed += ch
            recovered += rec
            w.write(json.dumps({"trace_id": r["trace_id"], "text": ct}, ensure_ascii=False) + "\n")
    print(f"cleaned {n} rows -> {dst.name}")
    print(f"  rows modified: {changed}   rows RECOVERED from empty/thin markdown: {recovered}")


if __name__ == "__main__":
    main()
