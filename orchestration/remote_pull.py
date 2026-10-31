"""Pull paths FROM the remote worker to the same location here (ssh tar | tar).  SPARK_PW=... python remote_pull.py path ..."""
import os
import subprocess
import sys

import paramiko

ROOT = os.path.dirname(os.path.abspath(__file__))
c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
c.connect(os.environ.get("REMOTE_HOST", "remote-worker.local"), username=os.environ.get("REMOTE_USER", "worker"), password=os.environ["SPARK_PW"], look_for_keys=False,
          allow_agent=False, timeout=15)
for rel in sys.argv[1:]:
    ch = c.get_transport().open_session(window_size=2 ** 30, max_packet_size=2 ** 15)
    ch.exec_command(f"tar cf - -C '{ROOT}' '{rel}'")
    p = subprocess.Popen(["tar", "xf", "-", "-C", ROOT], stdin=subprocess.PIPE)
    while True:
        data = ch.recv(1 << 22)
        if not data:
            break
        p.stdin.write(data)
    p.stdin.close()
    p.wait()
    print(f"pulled {rel} (exit {ch.recv_exit_status()})", flush=True)
c.close()
