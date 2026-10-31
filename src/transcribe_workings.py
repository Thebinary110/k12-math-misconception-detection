"""Read handwritten workings with an open VLM (vLLM, offline): photo -> `L1 | ...` transcription + final answer.

    python transcribe_workings.py --split train --limit 100      # quick quality check against the gold transcription
    python transcribe_workings.py --split test

Results are cached per trace in out/ocr_<split>_<tag>.jsonl and resumed on rerun.
"""
import process_naming  # noqa: F401  (process name; must run before vLLM starts)
import argparse
import json
import math
import time
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "data" / "img" / "images"

PROMPT = """This photo shows an Indonesian school student's handwritten working for the question below.

Question: {question}

Transcribe the student's handwriting exactly as written, line by line, including mistakes. Do not correct, solve or
complete anything. Skip crossed-out text. Use Unicode maths symbols (×, ÷, −, √, ², ³, ≠, ≤) and a/b for fractions.
Write one line per handwritten line in this format:
L1 | ...
L2 | ...
Then write the student's final answer on the last line as:
FINAL | <final answer as written by the student, or "none" if there is none>"""


def load_image(path, max_pixels):
    im = Image.open(path)
    im = ImageOps.exif_transpose(im).convert("RGB")
    s = math.sqrt(max_pixels / (im.width * im.height))
    if s < 1:
        im = im.resize((max(28, int(im.width * s)), max(28, int(im.height * s))), Image.LANCZOS)
    return im


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.8-27B-FP8"))
    ap.add_argument("--tag", default="q38")
    ap.add_argument("--max-pixels", type=int, default=1024 * 1024)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--max-seqs", type=int, default=128)
    ap.add_argument("--chunk", type=int, default=512)
    args = ap.parse_args()

    jobs = []
    for split in args.split.split(","):
        df = pd.read_csv(ROOT / "data" / f"{split}.csv")
        if args.limit:
            df = df.sample(args.limit, random_state=0)
        out = ROOT / "out" / f"ocr_{split}_{args.tag}.jsonl"
        out.parent.mkdir(exist_ok=True)
        done = {json.loads(l)["trace_id"] for l in open(out)} if out.exists() else set()
        todo = df[~df["trace_id"].isin(done)]
        print(f"{split}: {len(done)} cached, {len(todo)} to read")
        if not todo.empty:
            jobs.append((split, todo, out))
    if not jobs:
        return

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, max_model_len=8192, gpu_memory_utilization=args.gpu_mem,
              limit_mm_per_prompt={"image": 1}, max_num_seqs=args.max_seqs)
    params = SamplingParams(temperature=0.0, max_tokens=1024)
    t0 = time.time()
    for split, todo, out in jobs:
        # large chunks keep the scheduler full; each chunk is cached so an interrupted run resumes
        for start in range(0, len(todo), args.chunk):
            chunk = todo.iloc[start:start + args.chunk]
            msgs = [[{"role": "user", "content": [
                {"type": "image_pil", "image_pil": load_image(IMG / split / r.image_file, args.max_pixels)},
                {"type": "text", "text": PROMPT.format(question=r.question_text)}]}] for r in chunk.itertuples()]
            res = llm.chat(msgs, params, chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            with open(out, "a") as f:
                for r, o in zip(chunk.itertuples(), res):
                    f.write(json.dumps({"trace_id": r.trace_id, "text": o.outputs[0].text}, ensure_ascii=False) + "\n")
            n = min(start + args.chunk, len(todo))
            print(f"{split} {n}/{len(todo)} | {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
