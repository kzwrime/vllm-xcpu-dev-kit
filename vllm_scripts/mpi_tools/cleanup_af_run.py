"""Signal only processes carrying this launcher's unique AFD run ID."""
import os
from pathlib import Path
import re
import signal
import sys
import time

run_id, signal_name = sys.argv[1:]
if not re.fullmatch(r"afd_[0-9_]+", run_id):
    raise SystemExit("Invalid AFD run ID")
sig = {"TERM": signal.SIGTERM, "KILL": signal.SIGKILL, "CHECK": None}[signal_name]
marker = f"VLLM_AF_RUN_ID={run_id}".encode()

def matching_pids():
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            if marker in (entry / "environ").read_bytes().split(b"\0"):
                yield int(entry.name)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass

if signal_name == "CHECK":
    for attempt in range(51):
        remaining = list(matching_pids())
        if not remaining:
            print(f"{os.uname().nodename}: CLEAN run={run_id}")
            break
        time.sleep(0.1)
    else:
        raise SystemExit(f"{os.uname().nodename}: remaining PIDs {remaining}")
else:
    for pid in matching_pids():
        try:
            os.kill(pid, sig)
            print(f"{os.uname().nodename}: {signal_name} pid={pid}", flush=True)
        except ProcessLookupError:
            pass
