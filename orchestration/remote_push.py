"""Copy project paths to the same location on the remote worker through the SSH session (tar | ssh tar), no listener.
    SPARK_PW=... python remote_push.py path [path ...]"""
import os
import subprocess
import sys
import time

import paramiko

try:
    import setproctitle
    setproctitle.setproctitle(os.environ.get("PIPELINE_PROC_NAME", "k12-misconception-pipeline"))
except ImportError:
    pass

ROOT = os.path.dirname(os.path.abspath(__file__))
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
c.connect(os.environ.get("REMOTE_HOST", "remote-worker.local"), username=os.environ.get("REMOTE_USER", "worker"), password=os.environ["SPARK_PW"], look_for_keys=False,
          allow_agent=False, timeout=15, compress=False)
c.get_transport().window_size = 2 ** 30
for rel in sys.argv[1:]:
    t0, sent = time.time(), 0
    ch = c.get_transport().open_session(window_size=2 ** 30, max_packet_size=2 ** 15)
    ch.exec_command(f"mkdir -p '{ROOT}' && tar xf - -C '{ROOT}'")
    p = subprocess.Popen(["tar", "cf", "-", "-C", ROOT, rel], stdout=subprocess.PIPE)
    for chunk in iter(lambda: p.stdout.read(1 << 22), b""):
        ch.sendall(chunk)
        sent += len(chunk)
    p.wait()
    ch.shutdown_write()
    code = ch.recv_exit_status()
    print(f"{rel}: {sent / 1e9:.1f} GB in {time.time() - t0:.0f}s ({sent / 1e6 / max(1, time.time() - t0):.0f} MB/s) "
          f"exit {code}", flush=True)
c.close()
