"""Diagnose each working with an open LLM (vLLM, offline, text only): question + key + VLM transcription + the full
taxonomy + other students' labelled workings on the same question (leave-one-out on train) -> answer check + top-3 codes.

    python diagnose_misconceptions.py --split train     # out-of-fold style predictions for blending/calibration
    python diagnose_misconceptions.py --split test
"""
import process_naming  # noqa: F401  (process name; must run before vLLM starts)
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from answer_parsing import lines_and_final, strip_notes

ROOT = Path(__file__).resolve().parents[1]
MAX_EXAMPLES = 6

SYSTEM = """You are an experienced Indonesian mathematics teacher. You diagnose a student's handwritten working
(already transcribed; the transcription may contain reading errors) using the misconception taxonomy below.
The ROOT misconception is the major error that the rest of the working follows from, not necessarily the first error.
Answer NONE if the working is correct or has only minor slips that do not change the result.

TAXONOMY (code: name - description)
{taxonomy}
NONE: no major misconception"""

USER = """QUESTION: {question}
ANSWER KEY: {key}

{examples}STUDENT WORKING TO DIAGNOSE:
{working}

First decide whether the student's final answer is mathematically equivalent to the answer key (ignore units and
formatting). Then give the three most likely ROOT codes, most likely first. Reply in exactly this format:
ANSWER_CORRECT: yes|no
TOP3: CODE, CODE, CODE"""


def taxonomy_text(tax):
    return "\n".join(f"{r.id}: {r.name} - {str(r.description)[:220]}" for r in tax.itertuples())


def example_block(rows):
    if not rows:
        return ""
    parts = ["OTHER STUDENTS' WORKINGS ON THIS QUESTION (diagnosed by teachers):"]
    for i, r in enumerate(rows, 1):
        expl = ""
        try:
            root = [m for m in json.loads(r.misconceptions or "[]") if m.get("root")]
            if root:
                expl = f" ({root[0].get('text', '')[:200]})"
        except (ValueError, TypeError):
            pass
        parts.append(f"Example {i}:\n{strip_notes(r.transcription_steps)}\nROOT: {r.root_code}{expl}")
    return "\n\n".join(parts) + "\n\n"


def parse(text, valid):
    text = text.split("</think>")[-1]
    correct = re.search(r"ANSWER_CORRECT:\s*(yes|no)", text, re.I)
    top = re.search(r"TOP3:\s*(.*)", text)
    codes = [c for c in re.findall(r"[A-Z]{2,4}-\d{2}|NONE", top.group(1) if top else text) if c in valid]
    return (correct.group(1).lower() == "yes") if correct else None, list(dict.fromkeys(codes))[:3]


def budget_forced(llm, prompts, args):
    """Reason for at most args.think_budget tokens, then close the reasoning and force the answer (budget forcing).
    Returns run(start, end) -> objects with .outputs[k].text = reasoning + '</think>' + answer, like llm.chat."""
    from types import SimpleNamespace
    from vllm import SamplingParams
    tok = llm.get_tokenizer()

    def as_raw(msgs):
        images, conv = [], []
        for m in msgs:
            if isinstance(m["content"], list):
                content = []
                for part in m["content"]:
                    if part["type"] == "image_pil":
                        images.append(part["image_pil"])
                        content.append({"type": "image"})
                    else:
                        content.append(part)
                conv.append({"role": m["role"], "content": content})
            else:
                conv.append(m)
        text = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        return text, images

    think = SamplingParams(temperature=0.7 if args.n > 1 else 0.0, top_p=0.95, max_tokens=args.think_budget,
                           stop=["</think>"])
    answer = SamplingParams(temperature=0.0, max_tokens=64)

    def item(text, images):
        return {"prompt": text, "multi_modal_data": {"image": images[0]}} if images else {"prompt": text}

    def run(start, end):
        raw = [as_raw(p) for p in prompts[start:end]]
        outs = [[] for _ in raw]
        for _ in range(args.n):
            r1 = llm.generate([item(t, im) for t, im in raw], think, use_tqdm=False)
            # Prime the forced answer with the literal "ANSWER_CORRECT:" prefix so cap-out rows emit the
            # structured block instead of continuing to ramble; keep the prefix in the stored text so parse() sees it.
            cont = [(t + o.outputs[0].text.rstrip() + "\n</think>\n\nANSWER_CORRECT:", im) for (t, im), o in zip(raw, r1)]
            r2 = llm.generate([item(t, im) for t, im in cont], answer, use_tqdm=False)
            for k, (a, b) in enumerate(zip(r1, r2)):
                outs[k].append(SimpleNamespace(text=a.outputs[0].text + "</think>ANSWER_CORRECT:" + b.outputs[0].text))
        return [SimpleNamespace(outputs=o) for o in outs]

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--ocr-tag", default="q38")
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.8-27B-FP8"))
    ap.add_argument("--tag", default="q38")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--offset", type=int, default=0, help="skip this many rows first (to split a split across machines)")
    ap.add_argument("--gpu-mem", type=float, default=0.5)  # GB10 is decode-bandwidth-bound; more KV/concurrency gave ~no speedup, so proven 0.5 kept
    ap.add_argument("--thinking", action="store_true", help="let the model reason before answering")
    ap.add_argument("--image", action="store_true", help="also show the photo of the working")
    ap.add_argument("--n", type=int, default=1, help="samples per row (votes); >1 uses temperature 0.7")
    ap.add_argument("--max-seqs", type=int, default=64)
    ap.add_argument("--think-budget", type=int, default=0, help="reason at most this many tokens, then force the answer")
    args = ap.parse_args()

    train = pd.read_csv(ROOT / "data" / "train.csv")
    tax = pd.read_csv(ROOT / "data" / "taxonomy_v1.csv")
    df = pd.read_csv(ROOT / "data" / f"{args.split}.csv")
    ocr = {}
    for l in open(ROOT / "out" / f"ocr_{args.split}_{args.ocr_tag}.jsonl"):
        r = json.loads(l)
        ocr[r["trace_id"]] = r["text"]
    df = df[df["trace_id"].isin(ocr)]
    df = df.iloc[args.offset:]
    if args.limit:
        df = df.head(args.limit)
    out = ROOT / "out" / f"diag_{args.split}_{args.tag}.jsonl"
    done = {json.loads(l)["trace_id"] for l in open(out)} if out.exists() else set()
    df = df[~df["trace_id"].isin(done)]
    print(f"{len(done)} cached, {len(df)} to diagnose")
    if df.empty:
        return

    by_q = {q: g for q, g in train.groupby("question_id")}
    system = SYSTEM.format(taxonomy=taxonomy_text(tax))
    valid = set(tax["id"]) | {"NONE"}
    prompts = []
    for r in df.itertuples():
        g = by_q.get(r.question_id)
        ex = [] if g is None else [e for e in g.itertuples() if e.trace_id != r.trace_id][:MAX_EXAMPLES]
        lines, final = lines_and_final(ocr[r.trace_id])
        working = "\n".join(f"L{i} | {l}" for i, l in enumerate(lines, 1)) + f"\nFINAL ANSWER WRITTEN: {final or '(none)'}"
        text = USER.format(question=r.question_text, key=r.correct_answer, examples=example_block(ex), working=working)
        if args.image:
            from transcribe_workings import IMG, load_image
            user = [{"type": "image_pil", "image_pil": load_image(IMG / args.split / r.image_file, 1024 * 1024)},
                    {"type": "text", "text": "The photo shows the student's original handwriting.\n\n" + text}]
        else:
            user = text
        prompts.append([{"role": "system", "content": system}, {"role": "user", "content": user}])

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model,  max_model_len=24576, gpu_memory_utilization=args.gpu_mem,
              enable_prefix_caching=True, max_num_seqs=args.max_seqs,
              **({"limit_mm_per_prompt": {"image": 1}} if args.image else {}))
    params = SamplingParams(temperature=0.7 if args.n > 1 else 0.0, top_p=0.95, n=args.n,
                            max_tokens=6144 if args.thinking else 64)
    if args.think_budget:
        run = budget_forced(llm, prompts, args)
    for start in range(0, len(prompts), 128):
        if args.think_budget:
            res = run(start, start + 128)
        else:
            res = llm.chat(prompts[start:start + 128], params,
                           chat_template_kwargs={"enable_thinking": args.thinking}, use_tqdm=False)
        with open(out, "a") as f:
            for r, o in zip(df.iloc[start:start + 128].itertuples(), res):
                samples = [parse(c.text, valid) for c in o.outputs]
                f.write(json.dumps({"trace_id": r.trace_id,
                                    "answer_correct": [s[0] for s in samples], "top3": [s[1] for s in samples],
                                    "raw": o.outputs[0].text.split("</think>")[-1][-300:]},
                                   ensure_ascii=False) + "\n")
        print(f"{min(start + 128, len(prompts))}/{len(prompts)}", flush=True)


if __name__ == "__main__":
    main()
