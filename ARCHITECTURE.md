# Best-Scoring Architecture

Current best **public leaderboard: 0.73883** (MAP@3), from a candidate-level stacking ensemble over
three diverse reasoners plus statistical and retrieval sources. This document describes that exact
configuration and why each piece is there.

## Pipeline overview

```
            handwriting photo + question + answer key
                              │
                   ┌──────────┴───────────┐
                   │  1. Transcription     │   Qwen3-27B VLM (offline)
                   │  image → L1|…  FINAL|… │   literal, mistakes preserved
                   └──────────┬───────────┘
                              │ transcription text
        ┌─────────────────────┼─────────────────────────────┐
        │                     │                              │
 ┌──────┴──────┐   ┌──────────┴──────────┐        ┌──────────┴──────────┐
 │ 2. Reasoners │   │ 3. Statistical base │        │ 4. Semantic retrieval│
 │  (3 diverse) │   │  prior + TF-IDF     │        │  kNN over labelled   │
 └──────┬──────┘   └──────────┬──────────┘        └──────────┬──────────┘
        └─────────────────────┼──────────────────────────────┘
                              │ per-class probabilities from every source
                   ┌──────────┴───────────┐
                   │ 5. Candidate stack    │   LGBMRanker (lambdarank)
                   │  (row, code) pairs     │   → top-3 codes
                   └──────────┬───────────┘
                              │
                        MAP@3 submission
```

## Stage detail

### 1. Transcription — `src/transcribe_workings.py`
A vision-language model (Qwen3-27B) transcribes each handwritten working **literally**, line by line, into a
compact `L1 | … FINAL | …` format. The prompt forbids correcting, solving, or completing the work — preserving
the student's *actual* (often wrong) steps is essential, because the error is the label we predict. Measured
against the 1,500 gold transcripts: median content CER ≈ 15%.

### 2. Reasoners — `src/diagnose_misconceptions.py`
Each reasoner sees the question, answer key, transcription, the full taxonomy, and a few other labelled workings
on the same question (leave-one-out on train). It returns whether the final answer is correct plus its top-3
root-misconception codes. The ensemble uses three **diverse** reasoners, which is what lifts the stack:

| Source | Model / mode | Standalone OOF MAP@3 |
|--------|--------------|----------------------|
| `llm_q38ti2k` | Qwen3-27B, budget-forced reasoning (2048 tokens) | 0.656 |
| `llm_q38`     | Qwen3-27B, zero-shot single pass                | 0.596 |
| `llm_r1`      | R1-Distill-Qwen-32B, budget-forced reasoning    | 0.579 |

**Budget-forced reasoning:** the model reasons for up to 2048 tokens, then the reasoning is closed and the
structured answer is *forced* (two-pass decoding). A cap-out guard primes the forced answer so rows that exhaust
the budget still emit a parseable block. The R1-distill reasoner agrees with the Qwen reasoner on only ~55% of
top-1 predictions (union-oracle hit@3 ≈ 0.81 on train) — that **diversity**, not raw standalone strength, is why
adding it moved the stack.

### 3. Statistical baselines — `src/hierarchical_prior.py`, `src/tfidf_baseline.py`
A hierarchical Dirichlet prior (per question → per topic → global) and a TF-IDF + logistic-regression model over
the transcriptions. Individually modest, but valuable as calibrated, full-coverage features in the stack.

| Source | OOF MAP@3 |
|--------|-----------|
| `prior` | 0.522 |
| `tfidf` | 0.552 |

### 4. Semantic retrieval — `src/semantic_retrieval.py`
Embed each `(topic + question + transcription)` with a small multilingual sentence model (`multilingual-e5-small`,
chosen for the Indonesian text), find the nearest labelled training workings (out-of-fold on the shared CV folds),
and borrow their root-code distribution. A cheap, no-LLM, diverse source.

| Source | OOF MAP@3 |
|--------|-----------|
| `retr` | 0.519 |

### 5. Candidate-level stack — `src/stack_ranker.py`
Every source proposes codes; an `LGBMRanker` (lambdarank) scores each `(row, candidate)` pair using the sources'
probabilities and ranks, cross-source agreement, the answer-correct signal, and taxonomy features
(nature / severity). The candidate pool (union of each source's top-K plus NONE) already contains the gold code
**~96.8%** of the time, so the ranker's job is precision, not recall.

**Best configuration:**

```
sources = [prior, tfidf, llm_q38ti2k, llm_q38, llm_r1, retr]
stack OOF MAP@3 = 0.7309   →   public LB = 0.73883
```

## Score progression

| Step | What changed | Public LB |
|------|--------------|-----------|
| 1 | Hierarchical prior only | 0.535 |
| 2 | + zero-shot diagnoser blend | 0.649 |
| 3 | + partial reasoning | 0.661 |
| 4 | Full reasoning top-3 | 0.714 |
| 5 | Weighted reasoning + zero-shot + prior blend | 0.717 |
| 6 | LGBMRanker stack (prior + TF-IDF + zero-shot + reasoning) | 0.722 |
| 7 | + semantic kNN retrieval source | 0.726 |
| 8 | **+ R1-distill reasoner (3-reasoner ensemble)** | **0.73883** |

Out-of-fold CV tracks the leaderboard closely throughout, which is what makes the climb steerable.

## Key design decisions
- **Diversity over strength.** A second reasoner that only *agrees* adds nothing; the R1-distill reasoner helped
  because it is wrong in *different* places (top-1 agreement ~55%). Adding a less-diverse model (SEA-LION) *hurt*
  the stack (OOF 0.726) and was dropped.
- **Candidate-level ranking, not probability averaging.** Framing it as ranking `(row, code)` pairs lets the model
  use cross-source agreement directly; recall is already solved by the candidate pool (~97%).
- **Honest out-of-fold everywhere.** Every train matrix fed to the stack is out-of-fold on shared 5-fold splits,
  so the stack cannot peek — which is why OOF and LB stay aligned.

## Where the remaining points are
Error analysis: the candidate pool holds the gold code ~97% of the time but the final top-3 keeps it ~72% — the
gap is rows where all current sources are confidently wrong *together*. Closing it needs genuinely better/different
sources (more diverse reasoners, in-domain fine-tuning on expanded data), not more reranker features.
