#!/bin/bash
# Cron supervisor (every 5 min + @reboot). Runs the first unfinished stage of stages.txt, detached, unless a GPU
# stage is already running. Each stage line:   name | done-check (shell) | command (run from src/)
# Stages resume from their caches. Submitting to Kaggle is never automated.
R="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
exec 9>"$R/.pipeline.lock"
flock -n 9 || exit 0
export PATH="$R/.venv-vllm/bin:$PATH"
export VLLM_PROCESS_NAME_PREFIX="${PIPELINE_PROC_NAME:-k12-misconception-pipeline}"
# a stage runs in its own session; its process group id is kept in .stage.pgid (they share one process name)
[ -f "$R/.stage.pgid" ] && pgrep -g "$(cat "$R/.stage.pgid")" >/dev/null && exit 0
pgrep -f '(transcribe_workings|diagnose_misconceptions|blend_predictions|finetune_classifier)\.py|run_pipeline\.sh' >/dev/null && exit 0  

lines() { [ -f "$R/$1" ] && wc -l < "$R/$1" || echo 0; }
export -f lines; export R
while IFS='|' read -r name check cmd; do
  name=$(echo "$name" | xargs); [ -z "$name" ] || [ "${name:0:1}" = "#" ] && continue
  bash -c "$check" >/dev/null 2>&1 && continue
  echo "$(date '+%F %T') start $name" >> "$R/log_cron.log"
  cd "$R/src" && setsid nohup bash -c "exec -a "$VLLM_PROCESS_NAME_PREFIX" bash -c \"\$0\"" "$cmd" >> "$R/log_$name.log" 2>&1 < /dev/null &
  echo $! > "$R/.stage.pgid"
  exit 0
done < "$R/orchestration/pipeline_stages.txt"
