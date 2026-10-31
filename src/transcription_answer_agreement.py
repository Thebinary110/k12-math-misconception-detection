"""Quality of a VLM reading on train: text similarity to the gold transcription and agreement of the
answer-correct signal with the gold final_answer_correct flag (the strongest predictor of NONE)."""
import difflib
import json
import sys

import pandas as pd

from answer_parsing import answer_matches, lines_and_final, strip_notes
from common import DATA, ROOT


def main(tag="q38"):
    tr = pd.read_csv(DATA / "train.csv").set_index("trace_id")
    rows = [json.loads(l) for l in open(ROOT / "out" / f"ocr_train_{tag}.jsonl")]
    sims, agree, n_final = [], [], 0
    for r in rows:
        g = tr.loc[r["trace_id"]]
        lines, final = lines_and_final(r["text"])
        gold = "\n".join(l.split("|", 1)[-1].strip() for l in strip_notes(g.transcription_steps).splitlines())
        sims.append(difflib.SequenceMatcher(None, "\n".join(lines), gold).ratio())
        n_final += bool(final)
        agree.append(answer_matches(final, g.correct_answer) == bool(g.final_answer_correct))
    s = pd.Series(sims)
    print(f"{tag}: n={len(rows)} text similarity mean {s.mean():.3f} median {s.median():.3f} | "
          f"final answer found {n_final / len(rows):.3f} | answer-correct agrees with gold flag {sum(agree) / len(agree):.3f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
