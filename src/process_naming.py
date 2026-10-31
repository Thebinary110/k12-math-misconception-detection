"""All of our pipeline processes appear under one shared name in process listings (vLLM engine children get the
same prefix through VLLM_PROCESS_NAME_PREFIX)."""
import os

NAME = os.environ.get("PIPELINE_PROC_NAME", "k12-misconception-pipeline")
os.environ.setdefault("VLLM_PROCESS_NAME_PREFIX", NAME)
try:
    import setproctitle

    setproctitle.setproctitle(NAME)
except ImportError:
    pass
