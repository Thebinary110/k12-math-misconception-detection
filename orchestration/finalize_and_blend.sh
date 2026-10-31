#!/bin/bash
# Wait for the three reasoning shards, merge the two test halves, then blend + stack. Leaves submission files in subs/.
# Submitting to Kaggle stays manual. Run detached:  SPARK_PW=... setsid nohup ./finalize_and_blend.sh > log_finalize.log 2>&1 &
set -u
R="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"; export REPO_ROOT="$R"
cd "$R"
rc() { [ -f "$R/out/$1" ] && wc -l < "$R/out/$1" || echo 0; }        # local row count
ra() { SPARK_PW="$SPARK_PW" .venv/bin/python orchestration/remote_exec.py "wc -l < $R/out/diag_test_q38ti2k_a.jsonl 2>/dev/null" 2>/dev/null | tr -d '[:space:]'; }

echo "$(date +%T) waiting for shards..."
while true; do
  A=$(rc diag_test_q38ti2k_a.jsonl); B=$(rc diag_test_q38ti2k_b.jsonl); T=$(rc diag_train_q38ti2k.jsonl)  # shard A now LOCAL (the remote worker offline)
  echo "$(date +%T) A=${A:-0}/2400 B=$B/1098 train=$T/600"
  [ "${A:-0}" -ge 2400 ] && [ "$B" -ge 1098 ] && [ "$T" -ge 600 ] && break
  sleep 120
done

echo "$(date +%T) shard A is local (the remote worker offline) — no pull needed"
# merge the two test halves into one tag, de-duplicated by trace_id
.venv/bin/python - <<'PY'
import json, os, pathlib
R = pathlib.Path(os.environ["REPO_ROOT"])
seen, out = set(), []
for part in ["a", "b"]:
    f = R / f"out/diag_test_q38ti2k_{part}.jsonl"
    for line in open(f):
        t = json.loads(line)["trace_id"]
        if t not in seen:
            seen.add(t); out.append(line)
open(R / "out/diag_test_q38ti2k.jsonl", "w").writelines(out)
print("merged test rows:", len(out))
PY

cd "$R/src"
echo "$(date +%T) blend (reasoning only, and reasoning+zero-shot)"
../.venv/bin/python blend_predictions.py q38ti2k > "$R/log_blend_q38ti2k.log" 2>&1 || echo "blend q38ti2k failed"
../.venv/bin/python blend_predictions.py q38ti2k q38 > "$R/log_blend_combo.log" 2>&1 || echo "blend combo failed"
echo "$(date +%T) stack (all LLM sources)"
../.venv/bin/python stack_ranker.py --llm q38ti2k q38 --name sub_stack_reason.csv > "$R/log_stack_reason.log" 2>&1 || echo "stack failed"
echo "$(date +%T) FINALIZE_DONE"
grep -hE 'OOF MAP@3|blend OOF|STACK OOF|wrote' "$R"/log_blend_q38ti2k.log "$R"/log_blend_combo.log "$R"/log_stack_reason.log 2>/dev/null
