"""Run one shell command on the remote worker over the direct link (password from the SPARK_PW env var, never stored).
    SPARK_PW=... python remote_exec.py 'command'"""
import os
import sys

import paramiko

c = paramiko.SSHClient()
c.set_missing_host_key_policy(paramiko.RejectPolicy())
c.load_host_keys(os.path.expanduser("~/.ssh/known_hosts"))
c.connect(os.environ.get("REMOTE_HOST", "remote-worker.local"), username=os.environ.get("REMOTE_USER", "worker"), password=os.environ["SPARK_PW"], look_for_keys=False,
          allow_agent=False, timeout=15)
_, out, err = c.exec_command(sys.argv[1], timeout=None)
for line in out:
    print(line, end="", flush=True)
e = err.read().decode()
if e:
    print(e, end="", file=sys.stderr)
code = out.channel.recv_exit_status()
c.close()
sys.exit(code)
