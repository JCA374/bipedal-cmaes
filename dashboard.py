"""Local live training dashboard (stdlib only, localhost only).

  .venv/bin/python dashboard.py --run runs/v3-f20          # then open http://localhost:8770

The page polls /api/status every 15 s: stats.csv, hall of fame, milestones, restarts and
overrides, machine health (load, memory, swap), background jobs and log tails. /api/champion
returns a replay of the current champion, re-exported whenever best/params.npy changes.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARGS = None
_champ = {"mtime": None, "data": None, "busy": False}


def tail(path, n=12):
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 20000))
            lines = f.read().decode(errors="replace").splitlines()
        return lines[-n:]
    except OSError:
        return []


def meminfo():
    d = {}
    for line in open("/proc/meminfo"):
        k, v = line.split(":")
        d[k] = int(v.split()[0]) // 1024
    return {"avail_mb": d["MemAvailable"], "total_mb": d["MemTotal"], "swap_used_mb": d["SwapTotal"] - d["SwapFree"]}


def jobs():
    out = subprocess.run(["ps", "-eo", "pid,pcpu,etime,args"], capture_output=True, text=True).stdout.splitlines()[1:]
    want = {"training": r"train\.py --name", "train queue": r"train_queue\.sh", "finalize": r"finalize\.sh",
            "voice (Kokoro/Whisper)": r"voice-text/scripts|kokoro-narrate/|listen-check/|narration-prep/", "video render": r"make_lessons\.py",
            "site refresh": r"site_refresh\.sh"}
    res = {}
    for name, pat in want.items():
        rows = [r.split(None, 3) for r in out if re.search(pat, r) and "ps -eo" not in r]
        if rows:
            res[name] = {"n": len(rows), "cpu": round(sum(float(r[1]) for r in rows)), "since": rows[0][2]}
    return res


def deadline():
    """The running train_queue.sh's DEADLINE (from its environment), else --deadline."""
    hhmm = ARGS.deadline
    pids = subprocess.run(["pgrep", "-f", "[t]rain_queue.sh"], capture_output=True, text=True).stdout.split()
    for pid in pids:
        try:
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
            hhmm = next((e.split(b"=", 1)[1].decode() for e in env if e.startswith(b"DEADLINE=")), hhmm)
        except OSError:
            pass
    now = datetime.now()
    t = datetime.combine(now.date(), datetime.strptime(hhmm, "%H:%M").time())
    return (t if t > now else t + timedelta(days=1)).isoformat(timespec="minutes")


def status():
    run = Path(ARGS.run)
    rows = list(csv.DictReader(open(run / "stats.csv"))) if (run / "stats.csv").exists() else []
    num = lambda v: None if v in ("", None) else float(v)
    stats = [{k: num(v) for k, v in r.items()} for r in rows]
    cfg = json.loads((run / "run_config.json").read_text()) if (run / "run_config.json").exists() else {}
    best = json.loads((run / "best" / "meta.json").read_text()) if (run / "best" / "meta.json").exists() else {}
    hof = json.loads((run / "hof.json").read_text()) if (run / "hof.json").exists() else []
    ms = []
    if (run / "milestones").exists():
        for p in sorted((run / "milestones").iterdir()):
            m = json.loads((p / "meta.json").read_text())
            ms.append({"name": p.name, "generation": m.get("generation"), "mean": m.get("mean")})
    ov = json.loads((run / "overrides.json").read_text()) if (run / "overrides.json").exists() else {}
    log = run.parent.parent / f"runs_{run.name}.log"
    return {
        "now": datetime.now().isoformat(timespec="seconds"), "run": run.name, "deadline": deadline(),
        "stats": stats, "config": cfg.get("effective", {}), "history": cfg.get("history", []),
        "champion": best, "hof": hof[:12], "milestones": ms, "overrides": ov,
        "load": os.getloadavg(), "cpus": os.cpu_count(), "mem": meminfo(), "jobs": jobs(),
        "logs": {"training": [l for l in tail(log, 400) if not l.startswith("gen ")][-8:],
                 "queue": tail(ROOT / "train_queue.log", 6), "finalize": tail(ROOT / "finalize.log", 8)},
        "champion_mtime": (run / "best" / "params.npy").stat().st_mtime if (run / "best" / "params.npy").exists() else None,
    }


def champion():
    p = Path(ARGS.run) / "best" / "params.npy"
    if not p.exists():
        return None
    mt = p.stat().st_mtime
    if _champ["mtime"] != mt and not _champ["busy"]:
        _champ["busy"] = True
        out = ROOT / "results" / "dashboard_champion.json"
        out.parent.mkdir(exist_ok=True)
        r = subprocess.run(["nice", "-n", "10", sys.executable, str(ROOT / "experiments" / "export_episode.py"),
                            "--run", ARGS.run, "--seed", "30000008", "--label", "current champion", "--out", str(out)],
                           capture_output=True, text=True)
        if r.returncode == 0:
            _champ.update(mtime=mt, data=out.read_text())
        _champ["busy"] = False
    return _champ["data"]


class H(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(ROOT / "docs"), **kw)

    def log_message(self, *a):
        pass

    def send_json(self, body):
        b = body.encode() if isinstance(body, str) else json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        if self.path.startswith("/api/status"):
            return self.send_json(status())
        if self.path.startswith("/api/champion"):
            return self.send_json(champion() or "null")
        if self.path in ("/", "/index.html"):
            b = (ROOT / "dashboard.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(b)
            return
        return super().do_GET()   # /assets/* from docs/


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/v3-f20")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--deadline", default="09:00")
    ARGS = ap.parse_args()
    ARGS.run = str((ROOT / ARGS.run).resolve()) if not os.path.isabs(ARGS.run) else ARGS.run
    print(f"dashboard for {ARGS.run}: http://localhost:{ARGS.port}")
    ThreadingHTTPServer(("127.0.0.1", ARGS.port), H).serve_forever()


if __name__ == "__main__":
    main()
