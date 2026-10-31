"""LoRA + classification head over the 76 labels (75 codes + NONE) on the VLM transcription.
K-fold: out-of-fold probabilities on train (for blending) and fold-averaged probabilities on test.

    python finetune.py --model ../models/Qwen3.5-9B --tag ft9b --folds 5
Outputs out/ft_<tag>_oof.npy, out/ft_<tag>_test.npy (class order = common.classes).
"""
import process_naming  # noqa: F401  (process name; must run before vLLM starts)
import argparse
import json
import math
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoTokenizer

from answer_parsing import lines_and_final
from common import ROOT, classes, folds, load, map3, top3


def build_text(r, ocr, diag):
    lines, final = lines_and_final(ocr.get(r.trace_id, ""))
    d = diag.get(r.trace_id) or {}
    ac = d.get("answer_correct")
    ac = ac[0] if isinstance(ac, list) else ac
    hint = "" if ac is None else f"\nAnswer check: {'matches the key' if ac else 'does not match the key'}"
    return (f"Topic: {r.topic_name} | Grade {r.grade}\nQuestion: {r.question_text}\nAnswer key: {r.correct_answer}\n"
            "Student working:\n" + "\n".join(f"L{i} | {l}" for i, l in enumerate(lines, 1))
            + f"\nFinal answer: {final or '(none)'}{hint}")


def load_model(path, n_labels):
    from transformers import Qwen3_5TextForSequenceClassification
    m = Qwen3_5TextForSequenceClassification.from_pretrained(path, num_labels=n_labels, dtype=torch.bfloat16,
                                                              device_map={"": 0})
    m.config.pad_token_id = m.config.pad_token_id or m.config.eos_token_id
    return m


def batches(idx, bs, shuffle, rng):
    idx = np.array(idx)
    if shuffle:
        idx = idx[rng.permutation(len(idx))]
    for i in range(0, len(idx), bs):
        yield idx[i:i + bs]


def encode(tok, texts, max_len):
    return tok(texts, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to("cuda")


@torch.no_grad()
def predict(model, tok, texts, max_len, bs=8):
    model.eval()
    out = []
    order = np.argsort([len(t) for t in texts])
    for b in batches(order, bs, False, None):
        logits = model(**encode(tok, [texts[i] for i in b], max_len)).logits.float()
        out.append((b, torch.softmax(logits, -1).cpu().numpy()))
    P = np.zeros((len(texts), out[0][1].shape[1]))
    for b, p in out:
        P[b] = p
    return P


def train_fold(args, tok, texts, y, tr_idx, n_labels):
    model = load_model(args.model, n_labels)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(r=args.r, lora_alpha=2 * args.r, lora_dropout=0.05, task_type="SEQ_CLS",
                                             target_modules="all-linear", modules_to_save=["score"]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(tr_idx) / (args.bs * args.accum))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, steps // 10)) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
    rng = np.random.default_rng(args.seed)
    yt = torch.tensor(y)
    step, t0 = 0, time.time()
    for ep in range(args.epochs):
        model.train()
        for i, b in enumerate(batches(tr_idx, args.bs, True, rng)):
            logits = model(**encode(tok, [texts[j] for j in b], args.max_len)).logits.float()
            loss = torch.nn.functional.cross_entropy(logits, yt[b].cuda(), label_smoothing=0.05) / args.accum
            loss.backward()
            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(); step += 1
                if step % 10 == 0:
                    print(f"  ep {ep} step {step}/{steps} loss {loss.item() * args.accum:.3f} {time.time() - t0:.0f}s",
                          flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--only-folds", default="", help="comma list, e.g. 0 for a quick single-fold check")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    train, test, taxonomy = load()
    cls = classes(train, taxonomy)
    y = train["root_code"].map({c: i for i, c in enumerate(cls)}).values
    read = lambda f: {r["trace_id"]: r for r in map(json.loads, open(ROOT / "out" / f))}
    ocr_tr = {k: v["text"] for k, v in read("ocr_train_q38.jsonl").items()}
    ocr_te = {k: v["text"] for k, v in read("ocr_test_q38.jsonl").items()}
    diag_tr, diag_te = read("diag_train_q38.jsonl"), read("diag_test_q38.jsonl")
    t_tr = [build_text(r, ocr_tr, diag_tr) for r in train.itertuples()]
    t_te = [build_text(r, ocr_te, diag_te) for r in test.itertuples()]
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    oof_path, te_path = ROOT / "out" / f"ft_{args.tag}_oof.npy", ROOT / "out" / f"ft_{args.tag}_test.npy"
    oof = np.load(oof_path) if oof_path.exists() else np.zeros((len(train), len(cls)))
    P_te = np.load(te_path) if te_path.exists() else np.zeros((len(test), len(cls)))
    only = {int(x) for x in args.only_folds.split(",") if x}
    for k, (tr_idx, va_idx) in enumerate(folds(train, args.folds)):
        if (only and k not in only) or oof[va_idx].sum() > 0:
            continue
        print(f"fold {k}: train {len(tr_idx)} valid {len(va_idx)}", flush=True)
        model = train_fold(args, tok, t_tr, y, tr_idx, len(cls))
        oof[va_idx] = predict(model, tok, [t_tr[i] for i in va_idx], args.max_len)
        print(f"fold {k} MAP@3 {map3(train['root_code'].values[va_idx], top3(oof[va_idx], cls)):.4f}", flush=True)
        P_te += predict(model, tok, t_te, args.max_len) / args.folds
        np.save(oof_path, oof); np.save(te_path, P_te)
        del model; torch.cuda.empty_cache()
    done = oof.sum(1) > 0
    print(f"OOF MAP@3 on {done.sum()} rows: {map3(train['root_code'].values[done], top3(oof[done], cls)):.4f}")


if __name__ == "__main__":
    main()
