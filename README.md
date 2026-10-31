# Misconception Detection in K‑12 Student Math Workings

Open‑weights, fully offline pipeline for the CoLearn *Misconception Detection in K‑12 Student Math Workings*
competition. Given a photo of a student's handwritten solution, the question, and the answer key, the system predicts
the **root misconception** behind the student's error, as the top‑3 taxonomy codes scored by **MAP@3**.

The competition requires the final solution to run offline with open weights only (no hosted APIs on the test images),
so every stage here is a local vLLM / scikit‑learn / LightGBM component.

## Result

Best public leaderboard score so far: **0.726** (reasoning stack + semantic retrieval), up from a 0.535 statistical baseline.

| # | Submission | What it is | Public LB |
|---|------------|------------|-----------|
| 1 | `01_hierarchical_prior` | Question/topic/global Dirichlet prior only | 0.535 |
| 2 | `02_zeroshot_diagnoser_blend` | OCR + zero‑shot LLM diagnoser + prior, blended | 0.649 |
| 3 | `03_partial_reasoning_hybrid` | Blend base with 2048‑token reasoning on part of the test set | 0.661 |
| 4 | `04_full_reasoning_hybrid` | Raw 2048‑token reasoning top‑3 on the full test set | 0.714 |
| 5 | `05_reasoning_plus_zeroshot_blend` | Weighted blend of reasoning + zero‑shot + prior | 0.717 |
| 6 | `06_reasoning_stack` | LGBMRanker stack over prior + TF‑IDF + zero‑shot + reasoning | 0.722 |
| 7 | `07_reasoning_stack_plus_retrieval` **(best)** | Stack above + a semantic kNN retrieval source | **0.726** |

(Out‑of‑fold cross‑validation tracks the leaderboard closely throughout, which is what made the climb steerable.)

## How it works

The pipeline turns an image into a diagnosis in two model stages (read, then reason) and then calibrates the raw
model signal against cheaper statistical signals.

### 1. Transcription — `src/transcribe_workings.py`
A vision‑language model (Qwen3‑27B‑FP8) transcribes each handwritten working **literally**, line by line, into a
compact `L1 | … L2 | … FINAL | …` format. The prompt forbids correcting or completing the work — preserving the
student's *actual* (often wrong) steps is essential, because the error is the label we are trying to predict.

### 2. Diagnosis — `src/diagnose_misconceptions.py`
For each working the diagnoser model is shown: the question, the answer key, the transcription, the **full
misconception taxonomy**, and a few **other students' labelled workings on the same question** (leave‑one‑out on
train). It returns whether the student's final answer is correct and its top‑3 root‑misconception codes.

Two modes:
- **Zero‑shot** — a single forward pass. Fast; the backbone of submission #2.
- **Budget‑forced reasoning** — the model is allowed to reason for up to 2048 tokens, then the reasoning is closed and
  the structured answer is *forced* (two‑pass decoding). A "cap‑out" guard primes the forced answer with the
  `ANSWER_CORRECT:` prefix so rows that exhaust the thinking budget still emit a parseable block instead of rambling.
  This reasoning signal is the single biggest driver of the climb from 0.649 to 0.72.

### 3. Statistical baselines — `src/hierarchical_prior.py`, `src/tfidf_baseline.py`
A hierarchical Dirichlet prior (per question → per topic → global) and a TF‑IDF + logistic‑regression text model over
the transcriptions. Individually modest, but valuable as calibrated, full‑coverage features in the stack.

### 4. Blend & stack — `src/blend_predictions.py`, `src/stack_ranker.py`
- **Blend**: grid‑searched weighted average of the per‑class probabilities from each source.
- **Stack**: an `LGBMRanker` learns, from 5‑fold out‑of‑fold predictions, how to combine the prior, TF‑IDF,
  zero‑shot, and reasoning signals into a single ranking. This is the best submission.

### 4b. Semantic retrieval — `src/semantic_retrieval.py`
A cheap, no-LLM source inspired by how the strongest misconception solutions work: embed each
`(topic + question + transcription)` with a small multilingual sentence model (`multilingual-e5-small`, chosen for the
Indonesian text), find the nearest labelled training workings (out-of-fold on the shared folds), and borrow their
root-code distribution as a probability vector. Standalone it is modest (OOF ≈ 0.52), but as a *diverse* stack source
it lifts the ensemble — stack OOF 0.7168 → 0.7238, public LB 0.722 → **0.726** — for the cost of one embedding pass.
Embeddings are cached so re-tuning k/temperature needs no GPU. `stack_ranker.py --retr` adds it to the stack.

### 5. Transcription quality tools — `src/evaluate_transcription.py`, `src/clean_transcription.py`
Only the training split ships with gold human transcripts, which lets us score any OCR model **objectively** rather
than only through downstream MAP@3:
- `evaluate_transcription.py` reports content‑level CER/WER on the *parsed* working (what the reasoner actually
  consumes), plus the fraction of rows that parse to an empty/thin working and the final‑answer match rate.
- `clean_transcription.py` is a conservative, minimal‑diff post‑processor that recovers rows where the model
  transcribed correctly but used markdown tag variants the strict parser misses — provably without disturbing rows
  that already parse cleanly.

Measured on the 1500 gold transcripts, the current transcription has a **median content CER ≈ 15 %** (57 % of rows
under 20 %); the remaining errors are concentrated in genuinely hard images (number lines, question‑text bleed),
which motivates ongoing work on dedicated OCR backbones and light fine‑tuning.

### Experimental components
- `src/finetune_classifier.py` — a LoRA‑based direct classifier (explored, paused: weaker than the reasoning route).
- `src/ceiling_analysis.py` — oracle/ceiling probes on the available signals.
- `src/transcription_answer_agreement.py` — agreement of the model's answer‑correct flag with the gold flag (the
  strongest single predictor of the "no misconception" class).

## Orchestration — `orchestration/`
Long GPU jobs are run detached and supervised so a dropped session never kills them:
- `pipeline_supervisor.sh` — a cron‑driven supervisor that runs the first unfinished stage of
  `pipeline_stages.txt`, one at a time, resuming from per‑row caches. Submitting to Kaggle is always manual.
- `pipeline_stages.txt` — the declarative stage list (each line: name · done‑check · command).
- `finalize_and_blend.sh` — waits for the reasoning shards, merges them, then runs blend + stack.
- `remote_exec.py` / `remote_push.py` / `remote_pull.py` — optional helpers to offload a shard to a second machine
  over SSH (host/user/password all taken from environment variables; nothing is stored on disk).

## Layout
```
src/            model stages, baselines, blend/stack, and the transcription‑quality tools
orchestration/  supervisor, stage list, finalize, and optional remote‑offload helpers
submissions/    every Kaggle submission CSV, named by stage and its score
```

## Reproducing
Requires the competition data under `data/` (`train.csv`, `test.csv`, `taxonomy_v1.csv`, and the images), the open
model weights under `models/`, and two virtualenvs (a general `.venv`, and a `.venv-vllm` with vLLM on `PATH`). Then:
```bash
# 1. transcribe, 2. diagnose (zero-shot or --think-budget 2048), 3. blend/stack
python src/transcribe_workings.py --split train,test
python src/diagnose_misconceptions.py --split test  --image --think-budget 2048 --tag reason
python src/diagnose_misconceptions.py --split train --image --think-budget 2048 --tag reason --limit 1500
python src/stack_ranker.py --llm reason zeroshot --name submission.csv
# score any OCR against the gold transcripts
python src/evaluate_transcription.py --tag <tag>
```
Or let the supervisor drive the whole chain:
```bash
REPO_ROOT=$(pwd) ./orchestration/pipeline_supervisor.sh
```

## Notes
- Metric: MAP@3. Daily submission budget: 5. CV = 5‑fold stratified on the root code.
- Credentials (Kaggle token, SSH password, HF token) are **only ever passed via environment variables**, never
  written to disk or committed.
